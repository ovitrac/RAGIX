"""
KOAS-Translate · stage 2 — harmonize.

Stitch a chapter's segments (preferring ``final_translation`` over
``raw_translation``) and revise the result for fluency, terminology, and
typography, storing it in ``chapter_revisions`` for rebuild.

Long chapters are split into windows of ~``target_words`` at paragraph
boundaries and revised window-by-window (bounded per-call latency); the windows
are concatenated. Empty model output falls back to the unrevised window so
content is never lost. Idempotent: chapters with an existing revision are skipped
unless ``force``.

Ported behaviour-unchanged from the pipeline's ``harmonize.py``.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

from . import glossary as glossary_mod
from . import tm_store
from .backends import Backend, resolve_backend

DEFAULT_MODEL = "granite4.1-translate"
PROMPT_VERSION = "v2"
DEFAULT_TARGET_WORDS = 2000
DEFAULT_MAX_WORDS = 3000
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "harmonize.txt"


def split_into_windows(fr_text: str, *, target_words: int, max_words: int) -> List[str]:
    """Split *fr_text* into ~``target_words`` windows at paragraph boundaries.

    A single paragraph over ``max_words`` is emitted alone — never split inside a
    paragraph (the model needs paragraph context for fluency).
    """
    paras = [p for p in re.split(r"\n\s*\n", fr_text.strip()) if p.strip()]
    windows: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def _flush() -> None:
        nonlocal buf, buf_words
        if buf:
            windows.append("\n\n".join(buf))
            buf, buf_words = [], 0

    for p in paras:
        pw = len(p.split())
        if pw > max_words:
            _flush()
            windows.append(p)
            continue
        if buf_words + pw > max_words and buf:
            _flush()
        buf.append(p)
        buf_words += pw
        if buf_words >= target_words:
            _flush()
    _flush()
    return windows


class TranslateHarmonizeKernel(Kernel):
    """Stage 2 — chapter-level monolingual FR revision → chapter_revisions."""

    name = "translate_harmonize"
    version = "1.0.0"
    category = "translate"
    stage = 2
    description = "Chapter-level monolingual French revision (windowed) into chapter_revisions."
    requires = ["translate_qa"]

    #: Optional injected backend (tests / programmatic); see backends.resolve_backend.
    backend: Optional[Backend] = None

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        cfg = input.config or {}
        tm_path = Path(cfg.get("tm_path", input.workspace / "out" / "tm.sqlite"))
        model = cfg.get("model", DEFAULT_MODEL)
        force = bool(cfg.get("force", False))
        target = int(cfg.get("target_words", DEFAULT_TARGET_WORDS))
        max_w = int(cfg.get("max_words", DEFAULT_MAX_WORDS))

        template = cfg.get("prompt_template")
        if template is None:
            template = Path(cfg.get("prompt_path", DEFAULT_PROMPT_PATH)).read_text(encoding="utf-8")
        glossary_path = cfg.get("glossary_path")
        glossary_text = (
            glossary_mod.format_for_prompt(glossary_mod.load(glossary_path))
            if glossary_path else ""
        )
        backend = resolve_backend(self.backend, cfg, DEFAULT_MODEL)

        revised_chapters = 0
        skipped = 0
        incomplete: List[str] = []
        empty_windows = 0

        with tm_store.connect(tm_path) as conn:
            by_chapter: "OrderedDict[str, list]" = OrderedDict()
            for row in tm_store.iter_segments(conn):
                by_chapter.setdefault(row["chapter"] or "__preamble__", []).append(row)

            for chapter, rows in by_chapter.items():
                if not force and tm_store.get_chapter_revision(conn, chapter) is not None:
                    skipped += 1
                    continue

                parts: List[str] = []
                complete = True
                for r in rows:
                    fr = r["final_translation"] or r["raw_translation"]
                    if not fr:
                        incomplete.append(chapter)
                        complete = False
                        break
                    parts.append(fr)
                if not complete:
                    continue

                stitched = "\n\n".join(parts)
                windows = split_into_windows(stitched, target_words=target, max_words=max_w)
                revised_parts: List[str] = []
                for window in windows:
                    prompt = (template
                              .replace("{{GLOSSARY}}", glossary_text)
                              .replace("{{FR_TEXT}}", window))
                    revised = backend(prompt).strip()
                    if not revised:          # silent-empty → keep unrevised content
                        empty_windows += 1
                        revised = window
                    revised_parts.append(revised)

                tm_store.save_chapter_revision(
                    conn, chapter, "\n\n".join(revised_parts),
                    model=model, prompt_version=PROMPT_VERSION)
                conn.commit()
                revised_chapters += 1

        return {
            "n_revised": revised_chapters,
            "n_skipped": skipped,
            "incomplete_chapters": incomplete,
            "empty_windows": empty_windows,
            "model": model,
            "tm_path": str(tm_path),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            return f"translate_harmonize failed: {data['error']}"
        return (f"translate_harmonize: revised {data['n_revised']} chapter(s), "
                f"skipped {data['n_skipped']}, {len(data['incomplete_chapters'])} "
                f"incomplete [{data['model']}]")
