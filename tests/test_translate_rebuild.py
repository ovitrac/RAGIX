"""
Tests for the KOAS-Translate rebuild kernel (ragix_kernels/translate/rebuild).

Covers the stitch/restore assembly (chapter-revision precedence, fallback,
unresolved-placeholder reporting, --only-translated cutoff) and the
TranslateRebuildKernel end-to-end against a hand-built TM.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import json

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.translate import tm_store
from ragix_kernels.translate.rebuild import assemble, TranslateRebuildKernel


@pytest.fixture
def tm(tmp_path):
    """A TM with two chapters; seg sources carry one protected span each."""
    db = tmp_path / "out" / "tm.sqlite"
    with tm_store.connect(db) as conn:
        tm_store.upsert_source_segment(
            conn, segment_id="seg-00000", chapter="Intro", section=None,
            order_idx=0, source_text="# Intro\n\nEq ⟦P0001⟧ holds.",
            protected_map={"⟦P0001⟧": "$E=mc^2$"})
        tm_store.upsert_source_segment(
            conn, segment_id="seg-00001", chapter="Methods", section=None,
            order_idx=1, source_text="# Methods\n\nSee ⟦P0002⟧.",
            protected_map={"⟦P0002⟧": "[1-3]"})
    return tmp_path, db


def _run(root, db, **cfg):
    kernel = TranslateRebuildKernel()
    inp = KernelInput(
        workspace=root,
        config={"tm_path": str(db), **cfg},
        dependencies={"translate_harmonize": db},   # ordering dep (file exists)
    )
    return kernel.run(inp)


class TestAssemble:
    def test_stitch_from_final_translation_restores_spans(self, tm):
        root, db = tm
        with tm_store.connect(db) as conn:
            tm_store.save_translation(conn, "seg-00000", "# Intro\n\nL'éq ⟦P0001⟧ tient.",
                                      model="m", prompt_version="v1")
            tm_store.save_final(conn, "seg-00000", "# Intro\n\nL'éq ⟦P0001⟧ tient.")
            tm_store.save_translation(conn, "seg-00001", "# Méthodes\n\nVoir ⟦P0002⟧.",
                                      model="m", prompt_version="v1")
            tm_store.save_final(conn, "seg-00001", "# Méthodes\n\nVoir ⟦P0002⟧.")
        with tm_store.connect(db) as conn:
            final, problems, stats = assemble(conn)
        assert "$E=mc^2$" in final and "[1-3]" in final   # spans restored
        assert "⟦P" not in final                           # no token leaks
        assert problems == []
        assert stats["n_chapters"] == 2 and stats["n_unresolved"] == 0

    def test_chapter_revision_takes_precedence(self, tm):
        root, db = tm
        with tm_store.connect(db) as conn:
            tm_store.save_final(conn, "seg-00000", "ignored seg text ⟦P0001⟧")
            tm_store.save_chapter_revision(conn, "Intro",
                                           "# Intro\n\nRévisé: ⟦P0001⟧ final.",
                                           model="m", prompt_version="v1")
        with tm_store.connect(db) as conn:
            final, problems, _ = assemble(conn)
        assert "Révisé: $E=mc^2$ final." in final
        assert "ignored seg text" not in final

    def test_unresolved_placeholder_reported(self, tm):
        root, db = tm
        with tm_store.connect(db) as conn:
            # translation invents a token with no mapping entry
            tm_store.save_final(conn, "seg-00000", "# Intro\n\n⟦P0001⟧ and ⟦P9999⟧")
            tm_store.save_final(conn, "seg-00001", "# Methods\n\n⟦P0002⟧")
        with tm_store.connect(db) as conn:
            final, problems, stats = assemble(conn)
        assert stats["n_unresolved"] == 1
        assert any("⟦P9999⟧" in p for p in problems)
        assert "⟦P9999⟧" in final          # left in place, not crashed

    def test_missing_translation_falls_back_to_source(self, tm):
        root, db = tm
        # no translations saved at all → fallback to protected EN source
        with tm_store.connect(db) as conn:
            final, problems, _ = assemble(conn)
        assert "$E=mc^2$" in final                       # source restored
        assert any("no translation available" in p for p in problems)

    def test_only_translated_cuts_at_first_gap(self, tm):
        root, db = tm
        with tm_store.connect(db) as conn:
            tm_store.save_final(conn, "seg-00000", "# Intro\n\n⟦P0001⟧ done.")
            # seg-00001 left untranslated
        with tm_store.connect(db) as conn:
            final, problems, stats = assemble(conn, only_translated=True)
        assert "Methods" not in final
        assert stats["n_segments"] == 1


class TestRebuildKernel:
    def test_run_writes_final_md(self, tm):
        root, db = tm
        with tm_store.connect(db) as conn:
            tm_store.save_final(conn, "seg-00000", "# Intro\n\n⟦P0001⟧.")
            tm_store.save_final(conn, "seg-00001", "# Methods\n\n⟦P0002⟧.")
        out = _run(root, db)
        assert out.success, out.errors
        from pathlib import Path
        final_md = Path(out.data["final_md"])
        assert final_md.exists()
        text = final_md.read_text(encoding="utf-8")
        assert "$E=mc^2$" in text and "[1-3]" in text
        assert "translate_rebuild" in out.summary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
