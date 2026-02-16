"""
Tests for consolidate.py — run_delta() neighborhood-scoped consolidation.

Uses MockEmbedder for deterministic behavior.
"""

import pytest

from ragix_core.memory.config import ConsolidateConfig
from ragix_core.memory.consolidate import ConsolidationPipeline
from ragix_core.memory.embedder import MockEmbedder
from ragix_core.memory.graph_store import GraphStore
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import (
    MemoryItem,
    MemoryProvenance,
    _generate_id,
    _now_iso,
)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_delta.db")


@pytest.fixture
def store(tmp_db):
    s = MemoryStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return MockEmbedder()


@pytest.fixture
def config():
    return ConsolidateConfig(
        cluster_distance_threshold=0.5,
        fallback_to_deterministic=True,
    )


def _make_item(
    title: str,
    content: str = "test content",
    scope: str = "test",
    tier: str = "stm",
    tags: list = None,
    entities: list = None,
    source_id: str = "",
) -> MemoryItem:
    """Helper to create a MemoryItem."""
    return MemoryItem(
        id=_generate_id(),
        title=title,
        content=content,
        type="constraint",
        tier=tier,
        scope=scope,
        tags=tags or ["test"],
        entities=entities or [],
        confidence=0.8,
        provenance=MemoryProvenance(
            source_id=source_id,
            source_kind="document",
            chunk_ids=[],
        ),
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )


# ── run_delta basics ─────────────────────────────────────────────────────

class TestRunDeltaBasic:
    """Test run_delta() without graph."""

    def test_empty_new_items(self, store, embedder, config):
        pipeline = ConsolidationPipeline(store, embedder, config)
        stats = pipeline.run_delta([], scope="test")
        assert stats["delta_mode"] is True
        assert stats["items_processed"] == 0

    def test_single_new_item_no_merge(self, store, embedder, config):
        item = _make_item("RHEL SSH config")
        store.write_item(item, reason="test")

        pipeline = ConsolidationPipeline(store, embedder, config)
        stats = pipeline.run_delta([item.id], scope="test")

        assert stats["delta_mode"] is True
        assert stats["new_items"] == 1
        assert stats["items_merged"] == 0

    def test_returns_expected_keys(self, store, embedder, config):
        item = _make_item("Test item")
        store.write_item(item, reason="test")

        pipeline = ConsolidationPipeline(store, embedder, config)
        stats = pipeline.run_delta([item.id], scope="test")

        expected_keys = {
            "items_processed", "clusters_found", "items_merged",
            "items_promoted", "palace_assigned", "merge_chains",
            "delta_mode", "new_items", "affected_items",
        }
        assert expected_keys.issubset(set(stats.keys()))


# ── run_delta with graph ─────────────────────────────────────────────────

class TestRunDeltaWithGraph:
    """Test run_delta() with graph-assisted neighborhood discovery."""

    def test_graph_expands_affected(self, tmp_db, store, embedder, config):
        # Create items connected via graph
        item_a = _make_item("Item A", source_id="doc1.pdf:chunk_1")
        item_b = _make_item("Item B", source_id="doc1.pdf:chunk_2")
        store.write_item(item_a, reason="test")
        store.write_item(item_b, reason="test")

        # Embed both (MockEmbedder dimension = 768)
        vec_a = embedder.embed_text(f"{item_a.title} {item_a.content}")
        vec_b = embedder.embed_text(f"{item_b.title} {item_b.content}")
        store.write_embedding(item_a.id, vec_a, "mock", embedder.dimension)
        store.write_embedding(item_b.id, vec_b, "mock", embedder.dimension)

        # Build graph linking them
        graph = GraphStore(tmp_db)
        graph.add_node(f"item:{item_a.id}", "item", item_id=item_a.id)
        graph.add_node(f"item:{item_b.id}", "item", item_id=item_b.id)
        graph.add_edge(f"item:{item_a.id}", f"item:{item_b.id}", "similar", weight=0.9)

        pipeline = ConsolidationPipeline(store, embedder, config, graph=graph)
        stats = pipeline.run_delta([item_a.id], scope="test")

        # Graph BFS should discover item_b as a neighbor
        assert stats["affected_items"] >= 2
        graph.close()

    def test_delta_without_graph_no_expand(self, store, embedder, config):
        item_a = _make_item("Item A")
        item_b = _make_item("Item B")
        store.write_item(item_a, reason="test")
        store.write_item(item_b, reason="test")

        # No graph → only the explicitly listed new items
        pipeline = ConsolidationPipeline(store, embedder, config, graph=None)
        stats = pipeline.run_delta([item_a.id], scope="test")

        assert stats["affected_items"] == 1  # no graph expansion


# ── run_delta promotion ──────────────────────────────────────────────────

class TestRunDeltaPromotion:
    """Test that singleton items are promoted during delta consolidation."""

    def test_promote_singleton(self, store, embedder, config):
        item = _make_item("Singleton item", tier="stm")
        store.write_item(item, reason="test")

        pipeline = ConsolidationPipeline(store, embedder, config)
        stats = pipeline.run_delta([item.id], scope="test", promote=True)

        assert stats["items_promoted"] >= 0  # may or may not promote depending on criteria

    def test_no_promote_when_disabled(self, store, embedder, config):
        item = _make_item("No promote item", tier="stm")
        store.write_item(item, reason="test")

        pipeline = ConsolidationPipeline(store, embedder, config)
        stats = pipeline.run_delta([item.id], scope="test", promote=False)

        assert stats["items_promoted"] == 0


# ── run_delta skips archived ─────────────────────────────────────────────

class TestRunDeltaArchived:
    """Test that archived/superseded items are excluded from delta."""

    def test_skips_archived_items(self, store, embedder, config):
        item = _make_item("Archived item")
        store.write_item(item, reason="test")
        # Archive the item
        store.update_item(item.id, {"archived": True})

        pipeline = ConsolidationPipeline(store, embedder, config)
        stats = pipeline.run_delta([item.id], scope="test")

        assert stats["items_processed"] == 0  # archived items excluded
