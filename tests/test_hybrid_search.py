"""
Tests for Hybrid Search Engine

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

from ragix_core.hybrid_search import (
    FusionStrategy,
    HybridSearchResult,
    HybridSearchEngine,
    create_hybrid_engine,
)
from ragix_core.bm25_index import BM25Index, BM25Document, BM25SearchResult
from ragix_core.vector_index import SearchResult


class TestFusionStrategy:
    """Tests for FusionStrategy enum."""

    def test_strategy_values(self):
        """Test all fusion strategies are defined."""
        assert FusionStrategy.RRF is not None
        assert FusionStrategy.WEIGHTED is not None
        assert FusionStrategy.INTERLEAVE is not None
        assert FusionStrategy.BM25_RERANK is not None
        assert FusionStrategy.VECTOR_RERANK is not None


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a hybrid search result."""
        result = HybridSearchResult(
            doc_id="chunk_001",
            file_path="src/main.py",
            name="main",
            content="def main(): pass",
            start_line=1,
            end_line=2,
            combined_score=0.85,
            bm25_score=0.7,
            vector_score=0.9,
            bm25_rank=2,
            vector_rank=1,
            source="both",
            bm25_matched_terms=["main", "function"],
        )

        assert result.doc_id == "chunk_001"
        assert result.combined_score == 0.85
        assert result.source == "both"
        assert "main" in result.bm25_matched_terms


class TestHybridSearchEngine:
    """Tests for HybridSearchEngine."""

    @pytest.fixture
    def mock_bm25_index(self):
        """Create a mock BM25 index."""
        index = Mock(spec=BM25Index)
        index.search.return_value = [
            BM25SearchResult(
                doc_id="chunk_001",
                score=0.8,
                matched_terms=["main", "function"],
                metadata={"file_path": "src/main.py", "name": "main"},
            ),
            BM25SearchResult(
                doc_id="chunk_002",
                score=0.6,
                matched_terms=["main"],
                metadata={"file_path": "src/utils.py", "name": "helper"},
            ),
        ]
        return index

    @pytest.fixture
    def mock_vector_index(self):
        """Create a mock vector index."""
        index = Mock()
        index.search.return_value = [
            SearchResult(
                chunk_id="chunk_001",
                file_path="src/main.py",
                name="main",
                chunk_type="function",
                content="def main(): pass",
                start_line=1,
                end_line=2,
                score=0.9,
            ),
            SearchResult(
                chunk_id="chunk_003",
                file_path="src/other.py",
                name="other",
                chunk_type="function",
                content="def other(): pass",
                start_line=1,
                end_line=2,
                score=0.7,
            ),
        ]
        return index

    @pytest.fixture
    def mock_embedding_backend(self):
        """Create a mock embedding backend."""
        backend = Mock()
        backend.embed_text.return_value = np.random.rand(384)
        return backend

    def test_search_rrf_fusion(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test RRF fusion strategy."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search(
            "main function",
            k=5,
            strategy=FusionStrategy.RRF,
        )

        assert len(results) > 0
        # chunk_001 appears in both, should rank high
        assert any(r.doc_id == "chunk_001" for r in results)

    def test_search_weighted_fusion(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test weighted fusion strategy."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search(
            "main function",
            k=5,
            strategy=FusionStrategy.WEIGHTED,
            bm25_weight=0.3,
            vector_weight=0.7,
        )

        assert len(results) > 0

    def test_search_interleave_fusion(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test interleave fusion strategy."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search(
            "main function",
            k=5,
            strategy=FusionStrategy.INTERLEAVE,
        )

        assert len(results) > 0

    def test_bm25_only_search(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test BM25-only search."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search(
            "main function",
            k=5,
            bm25_weight=1.0,
            vector_weight=0.0,
        )

        assert len(results) > 0
        # All results should come from BM25 only
        for r in results:
            assert r.source in ["bm25", "both"]

    def test_vector_only_search(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test vector-only search."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search(
            "main function",
            k=5,
            bm25_weight=0.0,
            vector_weight=1.0,
        )

        assert len(results) > 0
        # All results should come from vector only
        for r in results:
            assert r.source in ["vector", "both"]

    def test_source_tracking(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test that result sources are tracked correctly."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search("main function", k=10)

        # chunk_001 appears in both BM25 and vector results
        chunk_001_result = next(
            (r for r in results if r.doc_id == "chunk_001"), None
        )
        assert chunk_001_result is not None
        assert chunk_001_result.source == "both"

    def test_matched_terms_included(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test that BM25 matched terms are included."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search("main function", k=5)

        # chunk_001 should have matched terms from BM25
        chunk_001_result = next(
            (r for r in results if r.doc_id == "chunk_001"), None
        )
        assert chunk_001_result is not None
        assert len(chunk_001_result.bm25_matched_terms) > 0

    def test_k_parameter_respected(
        self, mock_bm25_index, mock_vector_index, mock_embedding_backend
    ):
        """Test that k parameter limits results."""
        engine = HybridSearchEngine(
            bm25_index=mock_bm25_index,
            vector_index=mock_vector_index,
            embedding_backend=mock_embedding_backend,
        )

        results = engine.search("main function", k=2)
        assert len(results) <= 2


class TestCreateHybridEngine:
    """Tests for create_hybrid_engine factory."""

    def test_create_engine_without_index(self, temp_dir: Path):
        """Test creating engine fails gracefully without index."""
        # Should raise or return None when index doesn't exist
        with pytest.raises(Exception):
            create_hybrid_engine(
                index_path=temp_dir / "nonexistent",
                embedding_model="all-MiniLM-L6-v2",
            )
