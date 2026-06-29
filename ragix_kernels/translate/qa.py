"""
KOAS-Translate · stage 2 — qa (Pass B).

For every segment with a ``raw_translation`` but no ``qa_report``, ask the model
to compare SOURCE and TRANSLATION and emit strict JSON
(``{"status": "ok"|"revise", "issues": [...]}``), stored verbatim in the TM.

Policy (ported behaviour-unchanged from the pipeline's ``qa.py``):
  - ``status == "ok"``     → promote ``raw_translation`` to ``final_translation``.
  - ``status == "revise"`` → leave ``final_translation`` NULL (issues surface for
    human / harmonize review).
Unparseable JSON, non-object JSON, or backend errors are recorded as a synthetic
"revise" so the segment never silently vanishes.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

from . import glossary as glossary_mod
from . import tm_store
from .backends import Backend, DEFAULT_LANG_PAIR, load_prompt, parse_json_lenient, resolve_backend

DEFAULT_MODEL = "granite4.0-translate"          # JSON-reliable Granite derivative
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "qa.txt"


def _revise(problem: str) -> Dict[str, Any]:
    return {"status": "revise", "issues": [
        {"type": "grammar", "source": "", "problem": problem, "suggested_fix": ""}]}


def _build_prompt(template: str, *, glossary_text: str,
                  source_chunk: str, translation_chunk: str) -> str:
    return (template
            .replace("{{GLOSSARY}}", glossary_text)
            .replace("{{SOURCE_CHUNK}}", source_chunk)
            .replace("{{TRANSLATION_CHUNK}}", translation_chunk))


def _apply_verdict(conn, segment_id: str, report: Dict[str, Any], raw_translation: str) -> None:
    """Promote raw → final iff QA says OK; otherwise leave final NULL."""
    if report.get("status") == "ok":
        tm_store.save_final(conn, segment_id, raw_translation)


class TranslateQAKernel(Kernel):
    """Stage 2 — Pass B bilingual JSON QA; gates final_translation on status=ok."""

    name = "translate_qa"
    version = "1.0.0"
    category = "translate"
    stage = 2
    description = "Pass B: bilingual JSON QA of each translated segment; promote raw→final when OK."
    requires = ["translate_draft"]

    #: Optional injected backend (tests / programmatic); see backends.resolve_backend.
    backend: Optional[Backend] = None

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        cfg = input.config or {}
        tm_path = Path(cfg.get("tm_path", input.workspace / "out" / "tm.sqlite"))
        model = cfg.get("model", DEFAULT_MODEL)

        lang_pair = cfg.get("lang_pair", DEFAULT_LANG_PAIR)
        template = load_prompt(cfg, DEFAULT_PROMPT_PATH)
        glossary_path = cfg.get("glossary_path")
        glossary_text = (
            glossary_mod.format_for_prompt(glossary_mod.load(glossary_path))
            if glossary_path else ""
        )
        backend = resolve_backend(self.backend, cfg, DEFAULT_MODEL)

        ok = revise = skipped = 0
        by_type: "Counter[str]" = Counter()

        with tm_store.connect(tm_path) as conn:
            for row in list(tm_store.iter_segments(conn)):
                if row["qa_report"] is not None:
                    skipped += 1
                    if tm_store.needs_final(row):
                        import json
                        _apply_verdict(conn, row["segment_id"],
                                       json.loads(row["qa_report"]), row["raw_translation"])
                    continue
                if row["raw_translation"] is None:
                    continue  # not yet translated

                prompt = _build_prompt(
                    template, glossary_text=glossary_text,
                    source_chunk=row["source_text"],
                    translation_chunk=row["raw_translation"],
                )
                try:
                    raw = backend(prompt)
                    report = parse_json_lenient(raw)
                    if not isinstance(report, dict):
                        report = _revise(f"QA returned non-object: {report!r:.200}")
                except ValueError as e:
                    report = _revise(f"QA pass returned unparseable JSON: {e}")
                except Exception as e:  # noqa: BLE001 — timeouts etc. must not abort
                    report = _revise(f"QA pass raised {type(e).__name__}: {e}")

                tm_store.save_qa(conn, row["segment_id"], report)
                _apply_verdict(conn, row["segment_id"], report, row["raw_translation"])
                conn.commit()

                if report.get("status") == "ok":
                    ok += 1
                else:
                    revise += 1
                    for issue in report.get("issues", []) or []:
                        by_type[issue.get("type", "unknown") if isinstance(issue, dict)
                                else "malformed"] += 1

        return {
            "n_ok": ok,
            "n_revise": revise,
            "n_skipped": skipped,
            "issues_by_type": dict(by_type),
            "model": model,
            "lang_pair": lang_pair,
            "tm_path": str(tm_path),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            return f"translate_qa failed: {data['error']}"
        return (f"translate_qa: ok={data['n_ok']} revise={data['n_revise']} "
                f"skipped={data['n_skipped']} [{data['model']}]")
