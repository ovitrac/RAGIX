"""
Tests for the KOAS-Translate translation memory (ragix_kernels/translate/tm_store).

Covers schema creation, idempotent upsert, source-change invalidation, the
stage writers/predicates, ordering, and chapter revisions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.translate import tm_store


@pytest.fixture
def db(tmp_path):
    return tmp_path / "tm.sqlite"


def _seg(conn, sid="c1.s1", *, order=0, chapter="1", text="Hello.", pmap=None):
    tm_store.upsert_source_segment(
        conn, segment_id=sid, chapter=chapter, section=None, order_idx=order,
        source_text=text, protected_map=pmap or {},
    )


class TestSchemaAndHash:
    def test_connect_creates_schema(self, db):
        with tm_store.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"segments", "chapter_revisions"} <= tables

    def test_source_hash_is_deterministic(self):
        assert tm_store.source_hash("abc") == tm_store.source_hash("abc")
        assert tm_store.source_hash("abc") != tm_store.source_hash("abd")


class TestUpsert:
    def test_insert_then_read(self, db):
        with tm_store.connect(db) as conn:
            _seg(conn, text="Source.", pmap={"⟦P0001⟧": "$x$"})
            row = tm_store.get_segment(conn, "c1.s1")
            assert row["source_text"] == "Source."
            assert row["protected_map"] == '{"⟦P0001⟧": "$x$"}'
            assert tm_store.needs_translation(row)

    def test_same_source_preserves_translation(self, db):
        with tm_store.connect(db) as conn:
            _seg(conn, text="Stable.")
            tm_store.save_translation(conn, "c1.s1", "Stable-FR",
                                      model="m", prompt_version="v1")
            # re-run segment with identical source + new metadata
            _seg(conn, order=5, text="Stable.")
            row = tm_store.get_segment(conn, "c1.s1")
            assert row["raw_translation"] == "Stable-FR"   # preserved
            assert row["order_idx"] == 5                    # metadata refreshed

    def test_source_change_invalidates_downstream(self, db):
        with tm_store.connect(db) as conn:
            _seg(conn, text="Old.")
            tm_store.save_translation(conn, "c1.s1", "Old-FR",
                                      model="m", prompt_version="v1")
            tm_store.save_qa(conn, "c1.s1", {"status": "ok"})
            tm_store.save_final(conn, "c1.s1", "Old-FR-final")
            # source changes → everything downstream cleared
            _seg(conn, text="New.")
            row = tm_store.get_segment(conn, "c1.s1")
            assert row["source_text"] == "New."
            assert row["raw_translation"] is None
            assert row["qa_report"] is None
            assert row["final_translation"] is None
            assert tm_store.needs_translation(row)


class TestStageWritersAndPredicates:
    def test_translation_qa_final_flow(self, db):
        with tm_store.connect(db) as conn:
            _seg(conn)
            row = tm_store.get_segment(conn, "c1.s1")
            assert tm_store.needs_translation(row)
            assert not tm_store.needs_qa(row)        # nothing translated yet

            tm_store.save_translation(conn, "c1.s1", "Bonjour.",
                                      model="granite", prompt_version="v2")
            row = tm_store.get_segment(conn, "c1.s1")
            assert not tm_store.needs_translation(row)
            assert tm_store.needs_qa(row)
            assert tm_store.needs_final(row)
            assert row["model"] == "granite"

            tm_store.save_qa(conn, "c1.s1", {"status": "ok"})
            row = tm_store.get_segment(conn, "c1.s1")
            assert not tm_store.needs_qa(row)

            tm_store.save_final(conn, "c1.s1", "Bonjour.")
            row = tm_store.get_segment(conn, "c1.s1")
            assert not tm_store.needs_final(row)
            assert row["final_translation"] == "Bonjour."

    def test_iter_segments_is_order_sorted(self, db):
        with tm_store.connect(db) as conn:
            _seg(conn, sid="b", order=2, text="2")
            _seg(conn, sid="a", order=1, text="1")
            _seg(conn, sid="c", order=3, text="3")
            ids = [r["segment_id"] for r in tm_store.iter_segments(conn)]
        assert ids == ["a", "b", "c"]


class TestChapterRevisions:
    def test_save_and_get(self, db):
        with tm_store.connect(db) as conn:
            assert tm_store.get_chapter_revision(conn, "1") is None
            tm_store.save_chapter_revision(conn, "1", "Texte révisé.",
                                           model="m", prompt_version="v1")
            assert tm_store.get_chapter_revision(conn, "1") == "Texte révisé."

    def test_upsert_overwrites(self, db):
        with tm_store.connect(db) as conn:
            tm_store.save_chapter_revision(conn, "1", "v1 text",
                                           model="m", prompt_version="v1")
            tm_store.save_chapter_revision(conn, "1", "v2 text",
                                           model="m", prompt_version="v2")
            assert tm_store.get_chapter_revision(conn, "1") == "v2 text"


def test_persists_across_connections(db):
    with tm_store.connect(db) as conn:
        _seg(conn, text="Persist.")
    with tm_store.connect(db) as conn:
        row = tm_store.get_segment(conn, "c1.s1")
        assert row is not None and row["source_text"] == "Persist."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
