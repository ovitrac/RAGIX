"""
Tests for FAISS-accelerated recall (recall.py).

Validates:
- FaissIndex.add() + search() basic flow
- Empty index returns empty results
- L2 normalization produces unit vectors
- RecallEngine with FAISS vs without (fallback to brute-force)
- _faiss_embedding_scores() returns scores in [0,1]

Uses MockEmbedder for deterministic embeddings.
Requires faiss-cpu to be installed (tests skip otherwise).
"""

import math
import pytest

from ragix_core.memory.config import RecallConfig
from ragix_core.memory.embedder import MockEmbedder, cosine_similarity
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance

# Check if FAISS is available
try:
    import faiss
    import numpy as np
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# Conditional import of FaissIndex
if _HAS_FAISS:
    from ragix_core.memory.recall import FaissIndex

pytestmark = pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")


@pytest.fixture
def embedder():
    return MockEmbedder(dimension=32, seed=42)


@pytest.fixture
def store():
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def populated_store(store, embedder):
    """Store with items and embeddings for FAISS testing."""
    items = [
        MemoryItem(
            id="FAISS-001", tier="mtm", type="fact",
            title="SQLite architecture",
            content="SQLite uses B-tree indexes for fast lookups.",
            tags=["database", "sqlite", "architecture"],
            provenance=MemoryProvenance(
                source_kind="doc", source_id="doc_1",
                content_hashes=["sha256:abc"],
            ),
            confidence=0.9, validation="verified",
        ),
        MemoryItem(
            id="FAISS-002", tier="mtm", type="decision",
            title="Vector search with FAISS",
            content="FAISS provides efficient approximate nearest neighbor search.",
            tags=["vector", "faiss", "search"],
            provenance=MemoryProvenance(source_kind="doc", source_id="doc_2"),
            confidence=0.85,
        ),
        MemoryItem(
            id="FAISS-003", tier="stm", type="note",
            title="Python embedding tools",
            content="Sentence transformers and Ollama both produce embeddings.",
            tags=["embeddings", "python", "tools"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_3"),
            confidence=0.7,
        ),
        MemoryItem(
            id="FAISS-004", tier="stm", type="todo",
            title="Benchmark retrieval speed",
            content="Compare FAISS vs brute-force cosine scan at 10k items.",
            tags=["benchmark", "performance", "faiss"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_4"),
            confidence=0.5,
        ),
    ]
    for item in items:
        store.write_item(item)
        vec = embedder.embed_text(f"{item.title} {item.content}")
        store.write_embedding(item.id, vec, embedder.model_name, embedder.dimension)
    return store


# ---------------------------------------------------------------------------
# FaissIndex basic operations
# ---------------------------------------------------------------------------

class TestFaissIndexBasic:
    def test_create_index(self):
        """FaissIndex can be created with a given dimension."""
        idx = FaissIndex(dimension=32)
        assert idx.ntotal == 0

    def test_add_single_vector(self, embedder):
        """Adding a single vector increments the count."""
        idx = FaissIndex(dimension=32)
        vec = embedder.embed_text("test text")
        idx.add("item-1", vec)
        assert idx.ntotal == 1

    def test_add_batch(self, embedder):
        """Batch add inserts all vectors."""
        idx = FaissIndex(dimension=32)
        texts = ["alpha", "beta", "gamma"]
        ids = [f"item-{i}" for i in range(3)]
        vectors = [embedder.embed_text(t) for t in texts]
        idx.add_batch(ids, vectors)
        assert idx.ntotal == 3

    def test_search_after_add(self, embedder):
        """Search returns results after adding vectors."""
        idx = FaissIndex(dimension=32)
        texts = ["SQLite database engine", "FAISS vector search", "Python requirements"]
        ids = [f"item-{i}" for i in range(3)]
        vectors = [embedder.embed_text(t) for t in texts]
        idx.add_batch(ids, vectors)

        query_vec = embedder.embed_text("database SQLite")
        results = idx.search(query_vec, k=2)
        assert len(results) == 2
        # Each result is (item_id, score)
        for item_id, score in results:
            assert isinstance(item_id, str)
            assert isinstance(score, float)

    def test_search_returns_self_as_best_match(self, embedder):
        """Searching with the exact same text should return its own vector first."""
        idx = FaissIndex(dimension=32)
        target_text = "SQLite database storage engine"
        idx.add("target", embedder.embed_text(target_text))
        idx.add("other1", embedder.embed_text("FAISS approximate nearest neighbor"))
        idx.add("other2", embedder.embed_text("Python programming language"))

        results = idx.search(embedder.embed_text(target_text), k=3)
        # Exact same text should produce the highest score (cosine=1.0)
        assert results[0][0] == "target"
        assert results[0][1] > 0.99  # near-perfect match

    def test_search_scores_descending(self, embedder):
        """Scores should be in non-increasing order."""
        idx = FaissIndex(dimension=32)
        for i in range(5):
            idx.add(f"item-{i}", embedder.embed_text(f"topic number {i}"))
        results = idx.search(embedder.embed_text("topic number 0"), k=5)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-6


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

class TestFaissIndexEmpty:
    def test_empty_search(self):
        """Searching an empty index returns empty list."""
        idx = FaissIndex(dimension=32)
        results = idx.search([0.0] * 32, k=5)
        assert results == []

    def test_reset_clears_index(self, embedder):
        """Reset returns index to empty state."""
        idx = FaissIndex(dimension=32)
        idx.add("item-1", embedder.embed_text("test"))
        assert idx.ntotal == 1
        idx.reset()
        assert idx.ntotal == 0
        results = idx.search(embedder.embed_text("test"), k=5)
        assert results == []


# ---------------------------------------------------------------------------
# L2 normalization
# ---------------------------------------------------------------------------

class TestFaissNormalization:
    def test_l2_normalization_produces_unit_vectors(self, embedder):
        """Vectors added to FaissIndex are L2-normalized internally."""
        vec = embedder.embed_text("test normalization")
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)
        norm = np.linalg.norm(arr)
        assert abs(norm - 1.0) < 1e-5

    def test_inner_product_equals_cosine_after_normalization(self, embedder):
        """After L2 normalization, inner product equals cosine similarity."""
        vec_a = embedder.embed_text("SQLite database")
        vec_b = embedder.embed_text("relational database")

        # Manual cosine similarity
        cos_sim = cosine_similarity(vec_a, vec_b)

        # FAISS inner product after normalization
        a = np.array(vec_a, dtype=np.float32).reshape(1, -1)
        b = np.array(vec_b, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(a)
        faiss.normalize_L2(b)
        ip = float(np.dot(a[0], b[0]))

        assert abs(cos_sim - ip) < 1e-4


# ---------------------------------------------------------------------------
# RecallEngine with FAISS
# ---------------------------------------------------------------------------

class TestRecallEngineWithFaiss:
    def test_faiss_index_built_lazily(self, populated_store, embedder):
        """FAISS index should be built on first search call."""
        from ragix_core.memory.recall import RecallEngine
        engine = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        assert engine._faiss_index is None
        engine.search("database", k=3)
        # After search, index should be built
        assert engine._faiss_index is not None
        assert engine._faiss_index.ntotal == 4

    def test_faiss_results_match_sequential(self, populated_store, embedder):
        """FAISS and sequential paths should agree on top result."""
        from ragix_core.memory.recall import RecallEngine

        # FAISS path (default when faiss available)
        engine_faiss = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        results_faiss = engine_faiss.search("SQLite database", k=4)

        # Sequential path (force no FAISS)
        engine_seq = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        engine_seq._faiss_built = True  # skip FAISS build
        engine_seq._faiss_index = None  # force sequential
        results_seq = engine_seq.search("SQLite database", k=4)

        # Both should return the same top item
        assert results_faiss[0].id == results_seq[0].id

    def test_invalidate_faiss_index(self, populated_store, embedder):
        """invalidate_faiss_index resets the cached index."""
        from ragix_core.memory.recall import RecallEngine
        engine = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        engine.search("test", k=1)  # triggers build
        assert engine._faiss_index is not None
        engine.invalidate_faiss_index()
        assert engine._faiss_index is None
        assert engine._faiss_built is False


# ---------------------------------------------------------------------------
# _faiss_embedding_scores
# ---------------------------------------------------------------------------

class TestFaissEmbeddingScores:
    def test_scores_in_valid_range(self, populated_store, embedder):
        """FAISS embedding scores should be in [0, 1]."""
        from ragix_core.memory.recall import RecallEngine
        engine = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        query_vec = embedder.embed_text("database architecture")
        scores = engine._faiss_embedding_scores(query_vec, fetch_k=10)
        assert scores is not None
        for item_id, score in scores.items():
            assert 0.0 <= score <= 1.0 + 1e-6, f"Score {score} out of range for {item_id}"

    def test_scores_none_without_query_vec(self, populated_store, embedder):
        """Should return None when query_vec is None."""
        from ragix_core.memory.recall import RecallEngine
        engine = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        scores = engine._faiss_embedding_scores(None, fetch_k=10)
        assert scores is None

    def test_scores_contain_known_items(self, populated_store, embedder):
        """Scores dict should contain item IDs from the store."""
        from ragix_core.memory.recall import RecallEngine
        engine = RecallEngine(
            store=populated_store, embedder=embedder,
            config=RecallConfig(),
        )
        query_vec = embedder.embed_text("FAISS vector search benchmark")
        scores = engine._faiss_embedding_scores(query_vec, fetch_k=10)
        assert scores is not None
        assert len(scores) > 0
        for item_id in scores:
            assert item_id.startswith("FAISS-")
