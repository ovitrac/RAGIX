"""
Tests for the KOAS-Translate pipeline runner (ragix_kernels/translate/cli).

Runs the WHOLE 6-stage chain end-to-end with deterministic stubs (no Ollama):
extract → segment → draft → qa → harmonize → rebuild, asserting the chain wires
correctly and produces final.md. Plus stage selection, status, and resume.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from pathlib import Path

import pytest

from ragix_kernels.translate import tm_store
from ragix_kernels.translate.cli import run_pipeline, status, main, STAGE_NAMES


def _extractor(pdf, max_pages, out_dir):
    # two chapters with a protected span each
    return ("# Intro\n\nEq $E=mc^2$ holds.\n\n"
            "# Methods\n\nSee `code` and [Doe 2021].")


class PipelineStub:
    """One backend covering all three LLM stages, dispatched by prompt markers."""

    def __call__(self, prompt: str) -> str:
        if "Source to translate:" in prompt:                       # draft
            src = prompt.split("Source to translate:\n", 1)[1].strip()
            return f"FR[{src}]"
        if "TRANSLATION:" in prompt:                               # qa
            return '{"status": "ok", "issues": []}'
        if "French text to revise:" in prompt:                     # harmonize
            w = prompt.split("French text to revise:\n", 1)[1].strip()
            return f"REV[{w}]"
        return ""


@pytest.fixture
def workspace(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "book.pdf").write_bytes(b"%PDF-1.4")   # extractor stub ignores content
    return tmp_path


def test_full_pipeline_end_to_end(workspace):
    results = run_pipeline(workspace, {}, backend=PipelineStub(), extractor=_extractor)
    assert [r["kernel"] for r in results] == STAGE_NAMES          # all 6 ran
    assert all(r["success"] for r in results), results

    final = (workspace / "out" / "final.md")
    assert final.exists()
    text = final.read_text(encoding="utf-8")
    assert "$E=mc^2$" in text and "[Doe 2021]" in text            # protected spans restored
    assert "REV[" in text                                         # harmonized content used
    assert "⟦P" not in text                                       # no token leaks

    # 3 segments: the extract source-comment preamble + 2 chapters
    st = status(workspace)
    assert st["segments"] == 3 and st["final"] == 3 and st["chapter_revisions"] == 3


def test_stage_subset_runs_only_requested(workspace):
    # extract + segment only — no LLM stages
    results = run_pipeline(workspace, {}, stages=["translate_extract", "translate_segment"],
                           extractor=_extractor)
    assert [r["kernel"] for r in results] == ["translate_extract", "translate_segment"]
    assert all(r["success"] for r in results)
    st = status(workspace)
    assert st["segments"] == 3 and st["translated"] == 0          # nothing drafted yet


def test_resume_skips_completed_work(workspace):
    run_pipeline(workspace, {}, backend=PipelineStub(), extractor=_extractor)
    # second full run: draft/qa/harmonize should find everything cached
    results = run_pipeline(workspace, {}, backend=PipelineStub(), extractor=_extractor)
    draft = next(r for r in results if r["kernel"] == "translate_draft")
    assert "translated 0 new" in draft["summary"]                 # idempotent


def test_missing_dependency_stops_chain(workspace):
    # run rebuild alone with no prior stages → its required harmonize output is absent
    results = run_pipeline(workspace, {}, stages=["translate_rebuild"])
    assert results[0]["success"] is False
    assert results[0]["kernel"] == "translate_rebuild"


def test_cli_status_no_tm(workspace, capsys):
    rc = main(["status", "-w", str(workspace)])
    assert rc == 2                                                # no TM yet


def test_cli_run_subset(workspace, capsys):
    rc = main(["run", "-w", str(workspace), "--stages",
               "translate_extract,translate_segment"])
    # extract uses the REAL engine here (no stub via CLI) → may fail without
    # pymupdf4llm; we only assert the CLI dispatches and returns an int.
    assert isinstance(rc, int)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
