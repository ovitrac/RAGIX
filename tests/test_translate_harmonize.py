"""
Tests for the KOAS-Translate harmonize kernel (ragix_kernels/translate/harmonize).

Covers paragraph windowing and the chapter-revision flow (revise→store, idempotent
skip, force re-run, empty-output fallback, incomplete-chapter skip).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.translate import tm_store
from ragix_kernels.translate.harmonize import TranslateHarmonizeKernel, split_into_windows


class RevStub:
    """Wraps each window as REV[…]; can return empty for given call indices."""

    def __init__(self, empty_on=None):
        self.empty_on = set(empty_on or ())
        self.calls = 0

    def __call__(self, prompt):
        i = self.calls
        self.calls += 1
        if i in self.empty_on:
            return ""
        window = prompt.split("French text to revise:\n", 1)[1].strip()
        return f"REV[{window}]"


# ---------------------------------------------------------------------------
# split_into_windows
# ---------------------------------------------------------------------------

class TestWindows:
    def test_groups_paragraphs_to_target(self):
        text = "\n\n".join("w " * 30 for _ in range(6))   # 6 paras × 30 words
        wins = split_into_windows(text, target_words=60, max_words=120)
        assert len(wins) == 3                              # ~60 words each

    def test_oversized_paragraph_emitted_alone(self):
        big = "w " * 200
        text = f"small para here\n\n{big}\n\nanother small"
        wins = split_into_windows(text, target_words=50, max_words=100)
        assert any(len(w.split()) > 100 for w in wins)

    def test_single_window_when_small(self):
        assert split_into_windows("one\n\ntwo", target_words=100, max_words=200) == ["one\n\ntwo"]


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------

@pytest.fixture
def tm(tmp_path):
    """Two chapters, each with one finalized segment."""
    db = tmp_path / "out" / "tm.sqlite"
    with tm_store.connect(db) as conn:
        for sid, ch, order in [("seg-00000", "A", 0), ("seg-00001", "B", 1)]:
            tm_store.upsert_source_segment(
                conn, segment_id=sid, chapter=ch, section=None, order_idx=order,
                source_text=f"src {order}", protected_map={})
            tm_store.save_translation(conn, sid, f"FR {ch}.", model="m", prompt_version="v2")
            tm_store.save_final(conn, sid, f"FR {ch}.")
    return tmp_path, db


def _run(root, db, stub, **cfg):
    kernel = TranslateHarmonizeKernel()
    kernel.backend = stub
    inp = KernelInput(workspace=root, config={"tm_path": str(db), **cfg},
                      dependencies={"translate_qa": db})
    return kernel.run(inp)


class TestHarmonizeKernel:
    def test_revises_each_chapter(self, tm):
        root, db = tm
        out = _run(root, db, RevStub())
        assert out.data["n_revised"] == 2
        with tm_store.connect(db) as conn:
            assert tm_store.get_chapter_revision(conn, "A") == "REV[FR A.]"
            assert tm_store.get_chapter_revision(conn, "B") == "REV[FR B.]"

    def test_idempotent_skips_existing(self, tm):
        root, db = tm
        _run(root, db, RevStub())
        out2 = _run(root, db, RevStub())
        assert out2.data["n_revised"] == 0
        assert out2.data["n_skipped"] == 2

    def test_force_reruns(self, tm):
        root, db = tm
        _run(root, db, RevStub())
        out2 = _run(root, db, RevStub(), force=True)
        assert out2.data["n_revised"] == 2

    def test_empty_output_falls_back_to_unrevised(self, tm):
        root, db = tm
        out = _run(root, db, RevStub(empty_on={0}))     # first window returns ""
        assert out.data["empty_windows"] == 1
        with tm_store.connect(db) as conn:
            assert tm_store.get_chapter_revision(conn, "A") == "FR A."   # unrevised kept

    def test_incomplete_chapter_skipped(self, tmp_path):
        root = tmp_path
        db = tmp_path / "out" / "tm.sqlite"
        with tm_store.connect(db) as conn:
            tm_store.upsert_source_segment(
                conn, segment_id="seg-0", chapter="A", section=None, order_idx=0,
                source_text="s", protected_map={})   # no translation saved
        out = _run(root, db, RevStub())
        assert out.data["n_revised"] == 0
        assert "A" in out.data["incomplete_chapters"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
