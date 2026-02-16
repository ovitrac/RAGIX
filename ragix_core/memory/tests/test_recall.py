"""
Tests for hybrid retrieval engine (recall.py).

Validates:
- Tag filter works
- Hybrid ranking is stable (deterministic)
- Inject and catalog formatting

Uses MockEmbedder for deterministic embeddings.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

import pytest
from ragix_core.memory.config import RecallConfig
from ragix_core.memory.embedder import MockEmbedder
from ragix_core.memory.recall import RecallEngine
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
    """Store with several items and embeddings."""
    items = [
        MemoryItem(
            id="MEM-arch01", tier="mtm", type="decision",
            title="Architecture: use SQLite",
            content="We decided to use SQLite for persistent storage.",
            tags=["architecture", "database", "sqlite"],
            entities=["SQLite", "storage"],
            provenance=MemoryProvenance(
                source_kind="doc", source_id="doc_1",
                content_hashes=["sha256:abc"],
            ),
            confidence=0.9,
            validation="verified",
        ),
        MemoryItem(
            id="MEM-arch02", tier="mtm", type="decision",
            title="Architecture: use Ollama for embeddings",
            content="Embeddings will use Ollama nomic-embed-text model.",
            tags=["architecture", "embeddings", "ollama"],
            entities=["Ollama", "nomic-embed-text"],
            provenance=MemoryProvenance(
                source_kind="chat", source_id="turn_5",
            ),
            confidence=0.7,
        ),
        MemoryItem(
            id="MEM-fact01", tier="stm", type="fact",
            title="Python version requirement",
            content="RAGIX requires Python 3.10 or later.",
            tags=["python", "requirements"],
            entities=["Python", "RAGIX"],
            provenance=MemoryProvenance(source_kind="doc", source_id="readme"),
            confidence=0.95,
            validation="verified",
        ),
        MemoryItem(
            id="MEM-todo01", tier="stm", type="todo",
            title="Add FAISS fallback",
            content="Implement FAISS vector index as alternative to SQLite.",
            tags=["todo", "vector-index", "faiss"],
            entities=["FAISS"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_10"),
            confidence=0.5,
        ),
    ]

    for item in items:
        store.write_item(item)
        vec = embedder.embed_text(f"{item.title} {item.content}")
        store.write_embedding(item.id, vec, embedder.model_name, embedder.dimension)

    return store


@pytest.fixture
def engine(populated_store, embedder):
    return RecallEngine(
        store=populated_store,
        embedder=embedder,
        config=RecallConfig(
            tag_weight=0.3,
            embedding_weight=0.5,
            provenance_weight=0.2,
        ),
    )


class TestRecallSearch:
    def test_search_returns_results(self, engine):
        results = engine.search("SQLite database architecture", k=5)
        assert len(results) > 0

    def test_search_deterministic(self, engine):
        """Same query produces same results."""
        r1 = engine.search("SQLite database", k=5)
        r2 = engine.search("SQLite database", k=5)
        assert [i.id for i in r1] == [i.id for i in r2]

    def test_tag_filter(self, engine, populated_store):
        """Search with specific tags finds matching items."""
        results = engine.search("architecture", tags=["architecture"], k=5)
        ids = {r.id for r in results}
        assert "MEM-arch01" in ids or "MEM-arch02" in ids

    def test_tier_filter(self, engine, populated_store):
        """Tier filter restricts results."""
        # Create engine with tier filter
        filtered = RecallEngine(
            store=populated_store,
            embedder=MockEmbedder(dimension=32, seed=42),
            config=RecallConfig(default_tier_filter="mtm"),
        )
        results = filtered.search("architecture", k=10)
        for item in results:
            assert item.tier == "mtm"

    def test_search_with_scores(self, engine):
        results = engine.search_with_scores("SQLite", k=3)
        assert len(results) > 0
        # Scores should be descending
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestRecallFormatting:
    def test_format_inject(self, engine, populated_store):
        items = engine.search("SQLite", k=2)
        inject_text = engine.format_inject(items, budget_tokens=500)
        assert "[MEMORY:" in inject_text
        assert "[/MEMORY]" in inject_text

    def test_format_catalog(self, engine, populated_store):
        items = engine.search("architecture", k=3)
        catalog = engine.format_catalog(items)
        assert "memory_catalog" in catalog
        assert len(catalog["memory_catalog"]) > 0
        entry = catalog["memory_catalog"][0]
        assert "id" in entry
        assert "title" in entry
        assert "tier" in entry
