"""
Tests for memory persistent store (store.py).

Validates:
- Write/read/update operations
- Revision history preservation
- Embedding storage and retrieval
- Link management
- Palace location CRUD
- Export/import roundtrip

Uses in-memory SQLite for speed and isolation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

import json
import pytest
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryLink, MemoryProvenance


@pytest.fixture
def store():
    """In-memory store for testing."""
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def sample_item():
    return MemoryItem(
        id="MEM-test001",
        tier="stm",
        type="fact",
        title="Test fact",
        content="SQLite is a good embedded database.",
        tags=["database", "sqlite"],
        entities=["SQLite"],
        provenance=MemoryProvenance(
            source_kind="chat", source_id="turn_1",
        ),
        confidence=0.8,
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestBasicCRUD:
    def test_write_and_read(self, store, sample_item):
        store.write_item(sample_item)
        read_back = store.read_item("MEM-test001")
        assert read_back is not None
        assert read_back.id == "MEM-test001"
        assert read_back.title == "Test fact"
        assert read_back.content == "SQLite is a good embedded database."
        assert read_back.tags == ["database", "sqlite"]
        assert read_back.tier == "stm"

    def test_read_nonexistent(self, store):
        assert store.read_item("MEM-doesnotexist") is None

    def test_update_item(self, store, sample_item):
        store.write_item(sample_item)
        updated = store.update_item("MEM-test001", {
            "title": "Updated title",
            "confidence": 0.95,
        })
        assert updated is not None
        assert updated.title == "Updated title"
        assert updated.confidence == 0.95

    def test_soft_delete(self, store, sample_item):
        store.write_item(sample_item)
        assert store.delete_item("MEM-test001")
        # Should not appear in default queries
        items = store.list_items()
        assert len(items) == 0
        # But still readable directly
        item = store.read_item("MEM-test001")
        assert item is not None
        assert item.archived

    def test_supersede(self, store, sample_item):
        store.write_item(sample_item)
        new_item = MemoryItem(
            id="MEM-test002", tier="mtm", type="fact",
            title="Superseding fact", content="Better version.",
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_2"),
        )
        store.write_item(new_item)
        assert store.supersede_item("MEM-test001", "MEM-test002")
        old = store.read_item("MEM-test001")
        assert old.superseded_by == "MEM-test002"
        assert old.archived


# ---------------------------------------------------------------------------
# Revision history
# ---------------------------------------------------------------------------

class TestRevisionHistory:
    def test_revisions_on_create_and_update(self, store, sample_item):
        store.write_item(sample_item)
        store.update_item("MEM-test001", {"title": "Rev 2"})
        store.update_item("MEM-test001", {"title": "Rev 3"})

        revisions = store.read_revisions("MEM-test001")
        assert len(revisions) >= 3
        assert revisions[0]["revision_num"] == 1
        # First revision snapshot should have original title
        snap0 = revisions[0]["snapshot"]
        assert snap0["title"] == "Test fact"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_write_and_read_embedding(self, store, sample_item):
        store.write_item(sample_item)
        vec = [0.1, 0.2, 0.3, 0.4]
        store.write_embedding("MEM-test001", vec, "mock-4d", 4)

        result = store.read_embedding("MEM-test001")
        assert result is not None
        read_vec, model = result
        assert model == "mock-4d"
        assert len(read_vec) == 4
        assert abs(read_vec[0] - 0.1) < 1e-5

    def test_all_embeddings(self, store, sample_item):
        store.write_item(sample_item)
        item2 = MemoryItem(
            id="MEM-test002", tier="stm", type="note",
            title="Second item", content="Another note.",
        )
        store.write_item(item2)

        store.write_embedding("MEM-test001", [1.0, 2.0], "mock-2d", 2)
        store.write_embedding("MEM-test002", [3.0, 4.0], "mock-2d", 2)

        all_emb = store.all_embeddings()
        assert len(all_emb) == 2


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

class TestLinks:
    def test_write_and_read_links(self, store, sample_item):
        store.write_item(sample_item)
        item2 = MemoryItem(
            id="MEM-test002", tier="stm", type="decision",
            title="Decision", content="Use SQLite.",
        )
        store.write_item(item2)

        link = MemoryLink(src_id="MEM-test001", dst_id="MEM-test002", rel="supports")
        store.write_link(link)

        links = store.read_links("MEM-test001")
        assert len(links) == 1
        assert links[0].rel == "supports"
        assert links[0].dst_id == "MEM-test002"


# ---------------------------------------------------------------------------
# Palace locations
# ---------------------------------------------------------------------------

class TestPalaceLocations:
    def test_write_and_read_location(self, store, sample_item):
        store.write_item(sample_item)
        store.write_palace_location("MEM-test001", "default", "database", "fact", "MEM-test001")

        loc = store.read_palace_location("MEM-test001")
        assert loc is not None
        assert loc["domain"] == "default"
        assert loc["room"] == "database"
        assert loc["shelf"] == "fact"

    def test_list_locations(self, store, sample_item):
        store.write_item(sample_item)
        store.write_palace_location("MEM-test001", "default", "database", "fact", "MEM-test001")

        locs = store.list_palace_locations(domain="default")
        assert len(locs) == 1


# ---------------------------------------------------------------------------
# Tag search
# ---------------------------------------------------------------------------

class TestTagSearch:
    def test_search_by_tags(self, store, sample_item):
        store.write_item(sample_item)
        item2 = MemoryItem(
            id="MEM-test002", tier="stm", type="note",
            title="Unrelated", content="Not about databases.",
            tags=["cooking"],
        )
        store.write_item(item2)

        results = store.search_by_tags(["database"])
        assert len(results) == 1
        assert results[0].id == "MEM-test001"

    def test_search_by_tags_partial(self, store, sample_item):
        store.write_item(sample_item)
        # Search with one matching + one non-matching tag
        results = store.search_by_tags(["sqlite", "redis"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_roundtrip(self, store, sample_item):
        store.write_item(sample_item)
        exported = store.export_jsonl()

        # Import into fresh store
        store2 = MemoryStore(db_path=":memory:")
        count = store2.import_jsonl(exported)
        assert count == 1

        item = store2.read_item("MEM-test001")
        assert item is not None
        assert item.title == "Test fact"
        store2.close()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats(self, store, sample_item):
        store.write_item(sample_item)
        stats = store.stats()
        assert stats["total_items"] == 1
        assert stats["by_tier"].get("stm") == 1
        assert stats["by_type"].get("fact") == 1
