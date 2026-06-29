"""
Tests for the KOAS-Translate segment kernel (ragix_kernels/translate/segment).

Covers chunk_markdown boundaries + protection, and the TranslateSegmentKernel
end-to-end (KernelInput → run → TM rows + chunks.jsonl), including round-trip via
the shared codec and idempotent re-runs.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import json

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.shared.protected_spans import restore, TOKEN_RE
from ragix_kernels.translate import tm_store
from ragix_kernels.translate.segment import (
    chunk_markdown,
    TranslateSegmentKernel,
)


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------

class TestChunkMarkdown:
    def test_chapter_boundary_forces_flush(self):
        md = "# Chapter One\n\nAlpha beta gamma.\n\n# Chapter Two\n\nDelta epsilon.\n"
        chunks = chunk_markdown(md)
        assert len(chunks) == 2
        assert chunks[0].chapter == "Chapter One"
        assert chunks[1].chapter == "Chapter Two"
        assert [c.order_idx for c in chunks] == [0, 1]
        assert [c.segment_id for c in chunks] == ["seg-00000", "seg-00001"]

    def test_protected_spans_masked_and_globally_unique(self):
        md = ("# A\n\nFormula $x_1$ and code `f()`.\n\n"
              "# B\n\nMore math $y_2$ here.\n")
        chunks = chunk_markdown(md)
        # no raw protected span leaks into source_text
        for c in chunks:
            assert "$x_1$" not in c.source_text
            for tok in c.protected_map:
                assert TOKEN_RE.fullmatch(tok)
        # token namespaces never overlap across chunks
        all_tokens = [t for c in chunks for t in c.protected_map]
        assert len(all_tokens) == len(set(all_tokens))

    def test_soft_target_splits_within_chapter(self):
        # one chapter, many paragraphs; small target forces multiple chunks
        paras = "\n\n".join(f"word{i} " * 30 for i in range(10))
        md = f"# Big\n\n{paras}\n"
        chunks = chunk_markdown(md, target_words=60, max_words=120)
        assert len(chunks) > 1
        assert all(c.chapter == "Big" for c in chunks)

    def test_no_headings_single_block(self):
        chunks = chunk_markdown("Just a plain paragraph with no headings.\n")
        assert len(chunks) == 1
        assert chunks[0].chapter is None


# ---------------------------------------------------------------------------
# TranslateSegmentKernel (full run)
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    src = out / "source.md"
    src.write_text(
        "# Intro\n\nThe equation $E=mc^2$ holds; see `code` and [Doe 2021].\n\n"
        "# Methods\n\nWe used https://example.org and ref [1-3].\n",
        encoding="utf-8",
    )
    return tmp_path, src


def _run(workspace):
    root, src = workspace
    kernel = TranslateSegmentKernel()
    inp = KernelInput(
        workspace=root,
        config={"tm_path": str(root / "out" / "tm.sqlite")},
        dependencies={"translate_extract": src},
    )
    return kernel.run(inp)


class TestSegmentKernel:
    def test_run_populates_tm_and_jsonl(self, workspace):
        out = _run(workspace)
        assert out.success, out.errors
        assert out.data["n_chunks"] == 2
        assert out.data["n_chapters"] == 2
        assert out.data["n_protected_spans"] >= 5  # math, code, cite, url, numeric

        root, _ = workspace
        assert (root / "out" / "chunks.jsonl").exists()
        with tm_store.connect(root / "out" / "tm.sqlite") as conn:
            rows = list(tm_store.iter_segments(conn))
        assert len(rows) == 2
        assert all(tm_store.needs_translation(r) for r in rows)

    def test_round_trip_via_shared_codec(self, workspace):
        _run(workspace)
        root, _ = workspace
        with tm_store.connect(root / "out" / "tm.sqlite") as conn:
            for row in tm_store.iter_segments(conn):
                mapping = json.loads(row["protected_map"])
                restored, report = restore(row["source_text"], mapping)
                assert report.ok                       # every token resolvable
                assert "$E=mc^2$" in restored or "[1-3]" in restored

    def test_idempotent_rerun(self, workspace):
        _run(workspace)
        out2 = _run(workspace)
        assert out2.success
        root, _ = workspace
        with tm_store.connect(root / "out" / "tm.sqlite") as conn:
            assert len(list(tm_store.iter_segments(conn))) == 2  # no duplicates

    def test_summary_is_concise(self, workspace):
        out = _run(workspace)
        assert "translate_segment" in out.summary
        assert len(out.summary) <= 500


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
