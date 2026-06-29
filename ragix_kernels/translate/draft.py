"""
KOAS-Translate · stage 2 — draft (Pass A).

Walks the translation memory in ``order_idx`` order and, for each segment lacking
a raw translation, builds the strict-translation prompt (glossary + continuity
context + protected source chunk), calls the LLM backend, and stores the result.
Idempotent: already-translated segments are skipped and supply continuity context
for the next chunk; continuity resets across chapter boundaries.

The LLM call is abstracted behind a ``Backend`` callable (``prompt -> completion``)
so the gating/flow logic is testable with a deterministic stub. The default
backend wraps ``ragix_core.llm_backends.OllamaLLM`` (replacing the pipeline's
bespoke ollama_client). Reproducibility comes from the pinned Ollama modelfile
(temperature 0, greedy) — see ``modelfiles/granite4.1-translate.Modelfile``.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

from . import glossary as glossary_mod
from . import tm_store

#: A translation backend: full prompt → completion text.
Backend = Callable[[str], str]

DEFAULT_MODEL = "granite4.1-translate"
PROMPT_VERSION = "v2"
PREVIOUS_FR_TAIL_PARAGRAPHS = 2
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "translate.txt"


def build_ollama_backend(model: str, base_url: str = "http://localhost:11434") -> Backend:
    """Default backend: a single-prompt wrapper over ``llm_backends.OllamaLLM``."""
    from ragix_core.llm_backends import OllamaLLM
    llm = OllamaLLM(model=model, base_url=base_url)

    def _generate(prompt: str) -> str:
        return llm.generate(system_prompt=prompt, history=[])

    return _generate


def _tail_paragraphs(text: str, n: int) -> str:
    paras = [p for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paras[-n:]) if paras else ""


def build_prompt(
    template: str,
    *,
    glossary_text: str,
    chapter_summary: str,
    previous_fr_context: str,
    source_chunk: str,
) -> str:
    return (template
            .replace("{{GLOSSARY}}", glossary_text)
            .replace("{{CHAPTER_SUMMARY}}", chapter_summary or "(none)")
            .replace("{{PREVIOUS_FR_CONTEXT}}", previous_fr_context or "(none)")
            .replace("{{SOURCE_CHUNK}}", source_chunk))


class TranslateDraftKernel(Kernel):
    """Stage 2 — Pass A EN→FR translation over the TM (idempotent, resumable)."""

    name = "translate_draft"
    version = "1.0.0"
    category = "translate"
    stage = 2
    description = "Pass A: chunk-by-chunk EN→FR translation of TM segments via the LLM backend."
    requires = ["translate_segment"]

    #: Optional injected backend (tests / programmatic use). Kept off ``config``
    #: so the manifest stays JSON-serializable (the base hashes config). When
    #: None, the backend is built from config (model / ollama_url).
    backend: Optional[Backend] = None

    def _resolve_backend(self, cfg: Dict[str, Any]) -> Backend:
        if self.backend is not None:
            return self.backend
        return build_ollama_backend(
            cfg.get("model", DEFAULT_MODEL),
            cfg.get("ollama_url", "http://localhost:11434"),
        )

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        cfg = input.config or {}
        tm_path = Path(cfg.get("tm_path", input.workspace / "out" / "tm.sqlite"))
        model = cfg.get("model", DEFAULT_MODEL)
        limit: Optional[int] = cfg.get("limit")
        strict = bool(cfg.get("strict", False))

        template = cfg.get("prompt_template")
        if template is None:
            template = Path(cfg.get("prompt_path", DEFAULT_PROMPT_PATH)).read_text(encoding="utf-8")

        glossary_path = cfg.get("glossary_path")
        glossary_text = (
            glossary_mod.format_for_prompt(glossary_mod.load(glossary_path))
            if glossary_path else ""
        )

        backend = self._resolve_backend(cfg)

        translated = 0
        cached = 0
        failures: List[Dict[str, str]] = []

        with tm_store.connect(tm_path) as conn:
            rows = list(tm_store.iter_segments(conn))
            previous_fr = ""
            last_chapter: Any = object()  # sentinel ≠ any real chapter (incl. None)

            for row in rows:
                if not tm_store.needs_translation(row):
                    previous_fr = _tail_paragraphs(row["raw_translation"],
                                                   PREVIOUS_FR_TAIL_PARAGRAPHS)
                    last_chapter = row["chapter"]
                    cached += 1
                    continue

                if limit is not None and translated >= limit:
                    break

                if row["chapter"] != last_chapter:
                    previous_fr = ""
                    last_chapter = row["chapter"]

                prompt = build_prompt(
                    template,
                    glossary_text=glossary_text,
                    chapter_summary="",
                    previous_fr_context=previous_fr,
                    source_chunk=row["source_text"],
                )

                try:
                    fr = backend(prompt).strip()
                except Exception as e:  # noqa: BLE001
                    if strict:
                        raise
                    failures.append({"segment_id": row["segment_id"], "reason": str(e)})
                    continue

                if not fr:  # silent-empty failure mode of safety-tuned models
                    if strict:
                        raise RuntimeError(f"empty output on {row['segment_id']}")
                    failures.append({"segment_id": row["segment_id"], "reason": "empty"})
                    continue

                tm_store.save_translation(conn, row["segment_id"], fr,
                                          model=model, prompt_version=PROMPT_VERSION)
                conn.commit()
                previous_fr = _tail_paragraphs(fr, PREVIOUS_FR_TAIL_PARAGRAPHS)
                translated += 1

        return {
            "n_translated": translated,
            "n_cached": cached,
            "n_failures": len(failures),
            "failures": failures,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "limit": limit,
            "tm_path": str(tm_path),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            return f"translate_draft failed: {data['error']}"
        return (f"translate_draft: translated {data['n_translated']} new chunk(s), "
                f"skipped {data['n_cached']} cached, {data['n_failures']} failure(s) "
                f"[{data['model']}]")
