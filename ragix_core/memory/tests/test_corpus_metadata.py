"""
Tests for V3.0 corpus metadata — CorpusMetadata, store CRUD, corpus_id on items.
"""

import pytest

from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import (
    CorpusMetadata,
    MemoryItem,
    MemoryProvenance,
    _generate_id,
    _now_iso,
)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "test.db"))
    yield s
    s.close()


class TestCorpusMetadata:
    """Test CorpusMetadata dataclass."""

    def test_create(self):
        cm = CorpusMetadata(
            corpus_id="grdf-rie-2026Q1",
            corpus_label="GRDF RIE — Q1 2026",
            doc_count=27,
            item_count=1199,
        )
        assert cm.corpus_id == "grdf-rie-2026Q1"
        assert cm.parent_corpus_id is None

    def test_to_dict(self):
        cm = CorpusMetadata(corpus_id="test", corpus_label="Test")
        d = cm.to_dict()
        assert d["corpus_id"] == "test"
        assert "ingested_at" in d

    def test_from_dict(self):
        d = {"corpus_id": "abc", "corpus_label": "ABC", "doc_count": 5}
        cm = CorpusMetadata.from_dict(d)
        assert cm.corpus_id == "abc"
        assert cm.doc_count == 5


class TestStoreCorpusMetadata:
    """Test store CRUD for corpus_metadata table."""

    def test_write_and_read(self, store):
        cm = CorpusMetadata(
            corpus_id="test-corpus",
            corpus_label="Test Corpus",
            doc_count=10,
            item_count=100,
        )
        store.write_corpus_metadata(cm)
        result = store.read_corpus_metadata("test-corpus")
        assert result is not None
        assert result.corpus_id == "test-corpus"
        assert result.doc_count == 10
        assert result.item_count == 100

    def test_read_nonexistent(self, store):
        assert store.read_corpus_metadata("nope") is None

    def test_list_corpora(self, store):
        for i in range(3):
            store.write_corpus_metadata(CorpusMetadata(
                corpus_id=f"corpus-{i}",
                corpus_label=f"Corpus {i}",
            ))
        corpora = store.list_corpora()
        assert len(corpora) == 3

    def test_list_corpora_with_scope(self, store):
        store.write_corpus_metadata(CorpusMetadata(
            corpus_id="a", corpus_label="A", scope="grdf",
        ))
        store.write_corpus_metadata(CorpusMetadata(
            corpus_id="b", corpus_label="B", scope="other",
        ))
        grdf = store.list_corpora(scope="grdf")
        assert len(grdf) == 1
        assert grdf[0].corpus_id == "a"

    def test_parent_corpus(self, store):
        store.write_corpus_metadata(CorpusMetadata(
            corpus_id="v2",
            corpus_label="V2",
            parent_corpus_id="v1",
        ))
        result = store.read_corpus_metadata("v2")
        assert result.parent_corpus_id == "v1"


class TestItemCorpusId:
    """Test corpus_id field on MemoryItem."""

    def test_item_has_corpus_id(self):
        item = MemoryItem(
            title="Test",
            content="Content",
            corpus_id="grdf-rie-2026Q1",
        )
        assert item.corpus_id == "grdf-rie-2026Q1"

    def test_item_corpus_id_default_none(self):
        item = MemoryItem(title="Test", content="Content")
        assert item.corpus_id is None

    def test_store_roundtrip(self, store):
        item = MemoryItem(
            title="Corpus test",
            content="Test content",
            corpus_id="test-corpus",
            provenance=MemoryProvenance(source_kind="document"),
        )
        store.write_item(item, reason="test")
        loaded = store.read_item(item.id)
        assert loaded.corpus_id == "test-corpus"

    def test_list_items_by_corpus(self, store):
        for cid in ("corpus-a", "corpus-b"):
            item = MemoryItem(
                title=f"Item in {cid}",
                content="content",
                corpus_id=cid,
                provenance=MemoryProvenance(source_kind="document"),
            )
            store.write_item(item, reason="test")

        a_items = store.list_items(corpus_id="corpus-a")
        b_items = store.list_items(corpus_id="corpus-b")
        assert len(a_items) == 1
        assert len(b_items) == 1
        assert a_items[0].corpus_id == "corpus-a"

    def test_item_to_dict_includes_corpus_id(self):
        item = MemoryItem(
            title="Test",
            content="Content",
            corpus_id="my-corpus",
        )
        d = item.to_dict()
        assert d["corpus_id"] == "my-corpus"

    def test_item_from_dict_preserves_corpus_id(self):
        d = {
            "id": "MEM-test",
            "title": "Test",
            "content": "Content",
            "corpus_id": "loaded-corpus",
            "tier": "stm",
            "type": "note",
        }
        item = MemoryItem.from_dict(d)
        assert item.corpus_id == "loaded-corpus"
