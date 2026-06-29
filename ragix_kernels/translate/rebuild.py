"""
KOAS-Translate · stage 3 — rebuild.

Assemble the final translated Markdown from the translation memory:

  1. For each chapter, prefer ``chapter_revisions.revised_text`` (post-harmonize);
     otherwise stitch its segments' ``final_translation`` or ``raw_translation``
     (falling back to the protected English ``source_text`` when none exists,
     recorded as a problem).
  2. Reinstate every ``⟦P####⟧`` token via the shared protected-span codec,
     applying the union of the chapter's per-segment maps (harmonization may
     reorder paragraphs but must not drop placeholders). Unresolved tokens are
     reported and left in place.

Ported behaviour-unchanged from the translation pipeline's ``rebuild.py``.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.shared.protected_spans import restore

from . import tm_store


def assemble(conn, *, only_translated: bool = False) -> tuple[str, List[str], Dict[str, Any]]:
    """Stitch + restore the whole document from an open TM connection.

    Returns ``(final_markdown, problems, stats)``. Pure read over the TM, so it
    is unit-testable without the kernel wrapper.
    """
    # Optional cutoff: stop at the first segment lacking any translation.
    cutoff = None
    if only_translated:
        for row in tm_store.iter_segments(conn):
            if (row["final_translation"] or row["raw_translation"]) is None:
                cutoff = row["order_idx"]
                break

    by_chapter: "OrderedDict[str, list]" = OrderedDict()
    n_segments = 0
    for row in tm_store.iter_segments(conn):
        if cutoff is not None and row["order_idx"] >= cutoff:
            continue
        n_segments += 1
        key = row["chapter"] or "__preamble__"
        by_chapter.setdefault(key, []).append(row)

    parts: List[str] = []
    problems: List[str] = []
    n_unresolved = 0

    for chapter, rows in by_chapter.items():
        mapping: Dict[str, str] = {}
        for r in rows:
            mapping.update(json.loads(r["protected_map"]))

        revised = tm_store.get_chapter_revision(conn, chapter)
        if revised is not None:
            text = revised
        else:
            pieces: List[str] = []
            for r in rows:
                fr = r["final_translation"] or r["raw_translation"]
                if fr is None:
                    problems.append(f"{chapter}/{r['segment_id']}: no translation available")
                    fr = r["source_text"]  # fall back to (protected) EN
                pieces.append(fr)
            text = "\n\n".join(pieces)

        restored, report = restore(text, mapping)
        if report.hallucinated:
            n_unresolved += len(report.hallucinated)
            uniq = sorted(set(report.hallucinated))
            problems.append(
                f"{chapter}: {len(report.hallucinated)} unresolved placeholder(s): "
                f"{', '.join(uniq[:5])}{'…' if len(uniq) > 5 else ''}")
        # The chapter heading already lives (translated) inside the first
        # segment — do not prepend the English `# {chapter}`.
        parts.append(restored.strip())

    final = "\n\n".join(parts).rstrip() + "\n"
    stats = {
        "n_chapters": len(by_chapter),
        "n_segments": n_segments,
        "n_chars": len(final),
        "n_words": len(final.split()),
        "n_unresolved": n_unresolved,
        "cutoff": cutoff,
    }
    return final, problems, stats


class TranslateRebuildKernel(Kernel):
    """Stage 3 — stitch chapters + restore protected spans → final.md."""

    name = "translate_rebuild"
    version = "1.0.0"
    category = "translate"
    stage = 3
    description = "Assemble the final translated Markdown from the TM, restoring protected spans."
    requires = ["translate_harmonize"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        cfg = input.config or {}
        out_dir = Path(cfg.get("out_dir", input.workspace / "out"))
        tm_path = Path(cfg.get("tm_path", out_dir / "tm.sqlite"))
        only_translated = bool(cfg.get("only_translated", False))

        with tm_store.connect(tm_path) as conn:
            final, problems, stats = assemble(conn, only_translated=only_translated)

        out_dir.mkdir(parents=True, exist_ok=True)
        final_md = Path(cfg.get("final_md", out_dir / "final.md"))
        final_md.write_text(final, encoding="utf-8")

        return {
            "final_md": str(final_md),
            "only_translated": only_translated,
            "problems": problems,
            **stats,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            return f"translate_rebuild failed: {data['error']}"
        return (f"translate_rebuild: {Path(data['final_md']).name} "
                f"({data['n_chars']:,} chars, ~{data['n_words']:,} words), "
                f"{data['n_chapters']} chapter(s), {len(data['problems'])} problem(s)")
