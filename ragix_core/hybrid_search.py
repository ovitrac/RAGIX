"""
Hybrid Search - Combines BM25 keyword search with vector similarity

Provides fusion strategies for combining sparse (BM25) and dense (vector)
retrieval results.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .bm25_index import BM25Index, BM25SearchResult, build_bm25_index_from_chunks
from .vector_index import VectorIndex, SearchResult, NumpyVectorIndex
from .embeddings import EmbeddingBackend, create_embedding_backend

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Strategy for combining BM25 and vector search results."""

    RRF = "rrf"                    # Reciprocal Rank Fusion
    WEIGHTED = "weighted"          # Weighted score combination
    INTERLEAVE = "interleave"      # Round-robin interleaving
    BM25_RERANK = "bm25_rerank"    # Vector search, BM25 rerank
    VECTOR_RERANK = "vector_rerank"  # BM25 search, vector rerank


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    doc_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    combined_score: float
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    matched_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<HybridResult {self.file_path}:{self.name} score={self.combined_score:.3f}>"


class HybridSearchEngine:
    """
    Hybrid search engine combining BM25 and vector search.

    Supports multiple fusion strategies for optimal retrieval.
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        vector_index: Optional[VectorIndex] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.RRF,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid search engine.

        Args:
            bm25_index: BM25 keyword index
            vector_index: Vector similarity index
            embedding_backend: Backend for query embedding
            fusion_strategy: Strategy for combining results
            bm25_weight: Weight for BM25 scores (for weighted fusion)
            vector_weight: Weight for vector scores (for weighted fusion)
            rrf_k: Parameter for RRF (controls rank sensitivity)
        """
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.embedding_backend = embedding_backend
        self.fusion_strategy = fusion_strategy
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = 10,
        strategy: Optional[FusionStrategy] = None,
        bm25_k: Optional[int] = None,
        vector_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of final results
            strategy: Override default fusion strategy
            bm25_k: Number of BM25 results (default: 2*k)
            vector_k: Number of vector results (default: 2*k)

        Returns:
            List of HybridSearchResult
        """
        strategy = strategy or self.fusion_strategy
        bm25_k = bm25_k or k * 2
        vector_k = vector_k or k * 2

        # Get results from both indexes
        bm25_results = self._search_bm25(query, bm25_k)
        vector_results = self._search_vector(query, vector_k)

        # Fuse results based on strategy
        if strategy == FusionStrategy.RRF:
            return self._fuse_rrf(bm25_results, vector_results, k)
        elif strategy == FusionStrategy.WEIGHTED:
            return self._fuse_weighted(bm25_results, vector_results, k)
        elif strategy == FusionStrategy.INTERLEAVE:
            return self._fuse_interleave(bm25_results, vector_results, k)
        elif strategy == FusionStrategy.BM25_RERANK:
            return self._fuse_bm25_rerank(bm25_results, vector_results, k)
        elif strategy == FusionStrategy.VECTOR_RERANK:
            return self._fuse_vector_rerank(bm25_results, vector_results, k)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def _search_bm25(self, query: str, k: int) -> List[BM25SearchResult]:
        """Search BM25 index."""
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, k)

    def _search_vector(self, query: str, k: int) -> List[SearchResult]:
        """Search vector index."""
        if self.vector_index is None or self.embedding_backend is None:
            return []

        query_embedding = self.embedding_backend.embed_text(query)
        return self.vector_index.search(query_embedding, k)

    def _fuse_rrf(
        self,
        bm25_results: List[BM25SearchResult],
        vector_results: List[SearchResult],
        k: int,
    ) -> List[HybridSearchResult]:
        """
        Reciprocal Rank Fusion (RRF).

        Score = sum(1 / (k + rank)) for each ranking list.
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.doc_id
            if doc_id not in scores:
                scores[doc_id] = {
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "chunk_type": result.chunk_type,
                    "name": result.name,
                    "rrf_score": 0.0,
                    "bm25_score": result.score,
                    "bm25_rank": rank,
                    "vector_score": None,
                    "vector_rank": None,
                    "matched_terms": result.matched_terms,
                }
            scores[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.chunk_id
            if doc_id not in scores:
                scores[doc_id] = {
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "chunk_type": result.chunk_type,
                    "name": result.name,
                    "rrf_score": 0.0,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "vector_score": result.score,
                    "vector_rank": rank,
                    "matched_terms": [],
                }
            else:
                scores[doc_id]["vector_score"] = result.score
                scores[doc_id]["vector_rank"] = rank

            scores[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )[:k]

        # Build results
        results = []
        for doc_id, data in sorted_docs:
            results.append(HybridSearchResult(
                doc_id=doc_id,
                file_path=data["file_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                chunk_type=data["chunk_type"],
                name=data["name"],
                combined_score=data["rrf_score"],
                bm25_score=data["bm25_score"],
                bm25_rank=data["bm25_rank"],
                vector_score=data["vector_score"],
                vector_rank=data["vector_rank"],
                matched_terms=data["matched_terms"],
            ))

        return results

    def _fuse_weighted(
        self,
        bm25_results: List[BM25SearchResult],
        vector_results: List[SearchResult],
        k: int,
    ) -> List[HybridSearchResult]:
        """
        Weighted score fusion.

        Normalizes scores and combines with weights.
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # Normalize BM25 scores
        bm25_max = max((r.score for r in bm25_results), default=1.0)
        if bm25_max == 0:
            bm25_max = 1.0

        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.doc_id
            norm_score = result.score / bm25_max

            if doc_id not in scores:
                scores[doc_id] = {
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "chunk_type": result.chunk_type,
                    "name": result.name,
                    "weighted_score": 0.0,
                    "bm25_score": result.score,
                    "bm25_rank": rank,
                    "vector_score": None,
                    "vector_rank": None,
                    "matched_terms": result.matched_terms,
                }

            scores[doc_id]["weighted_score"] += self.bm25_weight * norm_score

        # Normalize vector scores (already 0-1 for cosine similarity)
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.chunk_id

            if doc_id not in scores:
                scores[doc_id] = {
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "chunk_type": result.chunk_type,
                    "name": result.name,
                    "weighted_score": 0.0,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "vector_score": result.score,
                    "vector_rank": rank,
                    "matched_terms": [],
                }
            else:
                scores[doc_id]["vector_score"] = result.score
                scores[doc_id]["vector_rank"] = rank

            scores[doc_id]["weighted_score"] += self.vector_weight * result.score

        # Sort by weighted score
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True
        )[:k]

        # Build results
        results = []
        for doc_id, data in sorted_docs:
            results.append(HybridSearchResult(
                doc_id=doc_id,
                file_path=data["file_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                chunk_type=data["chunk_type"],
                name=data["name"],
                combined_score=data["weighted_score"],
                bm25_score=data["bm25_score"],
                bm25_rank=data["bm25_rank"],
                vector_score=data["vector_score"],
                vector_rank=data["vector_rank"],
                matched_terms=data["matched_terms"],
            ))

        return results

    def _fuse_interleave(
        self,
        bm25_results: List[BM25SearchResult],
        vector_results: List[SearchResult],
        k: int,
    ) -> List[HybridSearchResult]:
        """
        Round-robin interleaving of results.

        Alternates between BM25 and vector results, deduplicating.
        """
        seen: Set[str] = set()
        results = []
        bm25_idx = 0
        vector_idx = 0

        while len(results) < k:
            # Try BM25
            while bm25_idx < len(bm25_results):
                result = bm25_results[bm25_idx]
                bm25_idx += 1
                if result.doc_id not in seen:
                    seen.add(result.doc_id)
                    results.append(HybridSearchResult(
                        doc_id=result.doc_id,
                        file_path=result.file_path,
                        start_line=result.start_line,
                        end_line=result.end_line,
                        chunk_type=result.chunk_type,
                        name=result.name,
                        combined_score=1.0 - len(results) / k,  # Decay score
                        bm25_score=result.score,
                        bm25_rank=bm25_idx,
                        matched_terms=result.matched_terms,
                    ))
                    break

            if len(results) >= k:
                break

            # Try vector
            while vector_idx < len(vector_results):
                result = vector_results[vector_idx]
                vector_idx += 1
                if result.chunk_id not in seen:
                    seen.add(result.chunk_id)
                    results.append(HybridSearchResult(
                        doc_id=result.chunk_id,
                        file_path=result.file_path,
                        start_line=result.start_line,
                        end_line=result.end_line,
                        chunk_type=result.chunk_type,
                        name=result.name,
                        combined_score=1.0 - len(results) / k,
                        vector_score=result.score,
                        vector_rank=vector_idx,
                    ))
                    break

            # Exit if no more results
            if bm25_idx >= len(bm25_results) and vector_idx >= len(vector_results):
                break

        return results

    def _fuse_bm25_rerank(
        self,
        bm25_results: List[BM25SearchResult],
        vector_results: List[SearchResult],
        k: int,
    ) -> List[HybridSearchResult]:
        """
        Use vector search, then rerank with BM25.

        Good for semantic search with keyword boosting.
        """
        # Create BM25 score lookup
        bm25_scores = {r.doc_id: (r.score, r.matched_terms) for r in bm25_results}

        results = []
        for rank, result in enumerate(vector_results[:k * 2], 1):
            bm25_score, matched = bm25_scores.get(result.chunk_id, (0.0, []))

            # Boost score if found in BM25 results
            boost = 1.0 + (bm25_score / max(s for s, _ in bm25_scores.values()) if bm25_scores else 0)
            combined = result.score * boost

            results.append(HybridSearchResult(
                doc_id=result.chunk_id,
                file_path=result.file_path,
                start_line=result.start_line,
                end_line=result.end_line,
                chunk_type=result.chunk_type,
                name=result.name,
                combined_score=combined,
                bm25_score=bm25_score if bm25_score > 0 else None,
                vector_score=result.score,
                vector_rank=rank,
                matched_terms=matched,
            ))

        # Re-sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:k]

    def _fuse_vector_rerank(
        self,
        bm25_results: List[BM25SearchResult],
        vector_results: List[SearchResult],
        k: int,
    ) -> List[HybridSearchResult]:
        """
        Use BM25 search, then rerank with vector similarity.

        Good for keyword search with semantic refinement.
        """
        # Create vector score lookup
        vector_scores = {r.chunk_id: r.score for r in vector_results}

        results = []
        for rank, result in enumerate(bm25_results[:k * 2], 1):
            vector_score = vector_scores.get(result.doc_id, 0.0)

            # Boost score if found in vector results
            boost = 1.0 + vector_score
            combined = result.score * boost

            results.append(HybridSearchResult(
                doc_id=result.doc_id,
                file_path=result.file_path,
                start_line=result.start_line,
                end_line=result.end_line,
                chunk_type=result.chunk_type,
                name=result.name,
                combined_score=combined,
                bm25_score=result.score,
                bm25_rank=rank,
                vector_score=vector_score if vector_score > 0 else None,
                matched_terms=result.matched_terms,
            ))

        # Re-sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:k]

    def save(self, path: Path):
        """Save indexes to disk."""
        path.mkdir(parents=True, exist_ok=True)

        if self.bm25_index:
            self.bm25_index.save(path / "bm25")

        if self.vector_index:
            self.vector_index.save(path / "vector")

        logger.info(f"Saved hybrid search indexes to {path}")

    def load(self, path: Path):
        """Load indexes from disk."""
        bm25_path = path / "bm25"
        vector_path = path / "vector"

        if bm25_path.exists():
            if self.bm25_index is None:
                self.bm25_index = BM25Index()
            self.bm25_index.load(bm25_path)

        if vector_path.exists():
            if self.vector_index is None:
                # Try to load and determine dimension
                import json
                meta_path = vector_path / "index.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self.vector_index = NumpyVectorIndex(meta["dimension"])
            if self.vector_index:
                self.vector_index.load(vector_path)

        logger.info(f"Loaded hybrid search indexes from {path}")


def create_hybrid_search_engine(
    index_path: Path,
    fusion_strategy: FusionStrategy = FusionStrategy.RRF,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> HybridSearchEngine:
    """
    Factory function to create a hybrid search engine from saved indexes.

    Args:
        index_path: Path to saved indexes
        fusion_strategy: Fusion strategy to use
        embedding_model: Model for query embedding

    Returns:
        Configured HybridSearchEngine
    """
    # Create embedding backend
    try:
        embedding_backend = create_embedding_backend("sentence-transformers")
    except ImportError:
        logger.warning("sentence-transformers not available, vector search disabled")
        embedding_backend = None

    # Create engine
    engine = HybridSearchEngine(
        fusion_strategy=fusion_strategy,
        embedding_backend=embedding_backend,
    )

    # Load indexes
    engine.load(index_path)

    return engine
