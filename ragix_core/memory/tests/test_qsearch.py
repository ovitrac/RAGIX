"""
Tests for Q*-style agenda search (qsearch.py).

Validates:
- Agenda expands highest-score node first
- Respects max_expansions budget
- Respects max_time_seconds budget
- Returns valid result structure

Uses MockEmbedder for deterministic behavior.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

import pytest
from ragix_core.memory.config import QSearchConfig
from ragix_core.memory.embedder import MockEmbedder
from ragix_core.memory.qsearch import QNode, QSearchEngine
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance


@pytest.fixture
def store():
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def embedder():
    return MockEmbedder(dimension=32, seed=42)


@pytest.fixture
def populated_store(store, embedder):
    """Populate store with test items."""
    items = [
        MemoryItem(
            id="MEM-q01", tier="stm", type="fact",
            title="Memory uses SQLite",
            content="The memory subsystem uses SQLite for persistence.",
            tags=["memory", "sqlite", "persistence"],
            entities=["SQLite", "memory"],
            provenance=MemoryProvenance(source_kind="doc", source_id="spec_1"),
            confidence=0.9,
        ),
        MemoryItem(
            id="MEM-q02", tier="stm", type="decision",
            title="Q*-search for recall",
            content="We use Q*-style agenda search instead of simple top-k.",
            tags=["memory", "qsearch", "recall"],
            entities=["Q*-search", "recall"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_20"),
            confidence=0.8,
        ),
        MemoryItem(
            id="MEM-q03", tier="stm", type="pattern",
            title="Hybrid scoring formula",
            content="S = w_r*R + w_p*P + w_c*C - w_d*D - w_x*X",
            tags=["scoring", "recall", "hybrid"],
            entities=["scoring", "hybrid-retrieval"],
            provenance=MemoryProvenance(
                source_kind="doc", source_id="spec_2",
                content_hashes=["sha256:formula123"],
            ),
            confidence=0.95,
            validation="verified",
        ),
        MemoryItem(
            id="MEM-q04", tier="stm", type="note",
            title="Unrelated cooking tip",
            content="Add salt to pasta water before boiling.",
            tags=["cooking", "food"],
            entities=["pasta"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_30"),
            confidence=0.3,
        ),
    ]

    for item in items:
        store.write_item(item)
        vec = embedder.embed_text(f"{item.title} {item.content}")
        store.write_embedding(item.id, vec, embedder.model_name, embedder.dimension)

    return store


class TestQSearchBasic:
    def test_search_returns_results(self, populated_store, embedder):
        engine = QSearchEngine(
            store=populated_store,
            embedder=embedder,
            config=QSearchConfig(max_expansions=5, max_time_seconds=5),
        )
        result = engine.search("How does memory recall work?")
        assert result["items"]
        assert result["best_score"] >= 0
        assert result["expansions"] >= 0

    def test_empty_store(self, embedder):
        empty_store = MemoryStore(db_path=":memory:")
        engine = QSearchEngine(
            store=empty_store, embedder=embedder,
            config=QSearchConfig(max_expansions=3),
        )
        result = engine.search("anything")
        assert result["items"] == []
        assert result["best_score"] == 0.0
        empty_store.close()


class TestQSearchBudget:
    def test_respects_max_expansions(self, populated_store, embedder):
        engine = QSearchEngine(
            store=populated_store,
            embedder=embedder,
            config=QSearchConfig(max_expansions=2, max_time_seconds=10),
        )
        result = engine.search("memory recall")
        assert result["expansions"] <= 2

    def test_respects_time_budget(self, populated_store, embedder):
        engine = QSearchEngine(
            store=populated_store,
            embedder=embedder,
            config=QSearchConfig(max_expansions=100, max_time_seconds=0.001),
        )
        result = engine.search("memory")
        # Should finish very quickly even with large expansion limit
        assert result["elapsed_seconds"] < 1.0


class TestQNodeOrdering:
    def test_higher_score_first(self):
        """QNode comparison: higher score = higher priority."""
        n1 = QNode(score=0.8)
        n2 = QNode(score=0.5)
        assert n1 < n2  # n1 has higher priority (for min-heap trick)

    def test_deterministic_results(self, populated_store, embedder):
        """Same query produces same results."""
        config = QSearchConfig(max_expansions=5)
        e1 = QSearchEngine(store=populated_store, embedder=embedder, config=config)
        e2 = QSearchEngine(store=populated_store, embedder=embedder, config=config)
        r1 = e1.search("memory architecture")
        r2 = e2.search("memory architecture")
        ids1 = [i["id"] for i in r1["items"]]
        ids2 = [i["id"] for i in r2["items"]]
        assert ids1 == ids2
