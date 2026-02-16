"""
Hybrid Retrieval Engine — Memory Recall

Combines three signals for ranking:
  1. Tag overlap (weighted Jaccard)
  2. Embedding similarity (cosine)
  3. Provenance quality (verified > unverified; doc-hashed > chat-only)

Supports three modes:
  - inject:  top items formatted for context injection
  - catalog: frontier items as structured catalog
  - hybrid:  both injection + catalog

Optional FAISS acceleration (V31-1, V32 GPU):
  When faiss-cpu is installed, embedding search uses IndexFlatIP on
  L2-normalized vectors (inner product = cosine similarity).  The index
  is built lazily on first search call and cached.  If faiss-cpu is
  absent, the engine falls back to sequential cosine scanning with no
  behavior change.

  V32: When an NVIDIA GPU is detected and faiss GPU bindings are available,
  the FAISS index is transparently transferred to GPU after batch loading
  via index_cpu_to_gpu().  GPU transfer is attempted automatically and
  falls back to CPU silently on failure.  Control via use_gpu parameter.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ragix_core.memory.config import RecallConfig
from ragix_core.memory.embedder import MemoryEmbedder, cosine_similarity
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FAISS detection
# ---------------------------------------------------------------------------

_faiss_available = False
_np = None  # numpy, loaded alongside faiss
try:
    import faiss  # type: ignore[import-untyped]
    import numpy as _np  # type: ignore[no-redef]

    _faiss_available = True
    logger.debug("FAISS detected — accelerated embedding search enabled")
except ImportError:
    logger.debug("faiss-cpu not installed — using sequential cosine scan")


# ---------------------------------------------------------------------------
# FAISS Index Wrapper
# ---------------------------------------------------------------------------

class FaissIndex:
    """
    Thin wrapper around faiss.IndexFlatIP for cosine similarity search.

    Vectors are L2-normalized before insertion so that inner product
    equals cosine similarity.  The index maps integer ordinals back to
    item IDs via a parallel list.

    V32 GPU support: when *use_gpu=True* and an NVIDIA GPU with faiss GPU
    bindings is detected, the index is transparently transferred to GPU
    after ``add_batch()`` via ``faiss.index_cpu_to_gpu()``.  Falls back
    to CPU silently on failure.
    """

    def __init__(self, dimension: int, use_gpu: bool = True):
        """Initialize FAISS IndexFlatIP with given vector dimension.

        Args:
            dimension: Vector dimension (must match embedding model output).
            use_gpu: Attempt GPU transfer after batch loading.  Ignored if
                no NVIDIA GPU or faiss GPU bindings are available.
        """
        if not _faiss_available:
            raise RuntimeError("faiss-cpu is not installed")
        self._dimension = dimension
        self._cpu_index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._search_index = self._cpu_index  # active index for search
        self._ids: List[str] = []  # ordinal -> item_id
        self._use_gpu = use_gpu
        self._on_gpu = False
        self._gpu_res = None  # faiss.StandardGpuResources, kept alive

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return self._cpu_index.ntotal

    @property
    def on_gpu(self) -> bool:
        """True if index has been transferred to GPU."""
        return self._on_gpu

    # -- Build ---------------------------------------------------------------

    def add(self, item_id: str, vector: List[float]) -> None:
        """Add a single L2-normalized vector to the CPU index."""
        vec = _np.array(vector, dtype=_np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self._cpu_index.add(vec)
        self._ids.append(item_id)
        # Single add stays on CPU; use add_batch for GPU transfer path

    def add_batch(
        self, ids: List[str], vectors: List[List[float]]
    ) -> None:
        """Add a batch of vectors (L2-normalized internally).

        After batch loading, attempts GPU transfer if use_gpu=True.
        """
        if not ids:
            return
        mat = _np.array(vectors, dtype=_np.float32)
        faiss.normalize_L2(mat)
        self._cpu_index.add(mat)
        self._ids.extend(ids)
        # Attempt GPU transfer after bulk load
        self._try_gpu_transfer()

    def _try_gpu_transfer(self) -> None:
        """Transfer CPU index to GPU if available and requested."""
        if not self._use_gpu or self._on_gpu:
            return
        if self._cpu_index.ntotal == 0:
            return
        try:
            from ragix_core.shared.gpu_detect import has_faiss_gpu

            if not has_faiss_gpu():
                return
            res = faiss.StandardGpuResources()
            self._search_index = faiss.index_cpu_to_gpu(res, 0, self._cpu_index)
            self._gpu_res = res  # prevent GC
            self._on_gpu = True
            logger.info(
                f"FAISS index on GPU ({self._cpu_index.ntotal} vectors, "
                f"dim={self._dimension})"
            )
        except Exception as e:
            logger.warning(f"GPU transfer failed, using CPU: {e}")
            self._search_index = self._cpu_index

    # -- Search --------------------------------------------------------------

    def search(
        self, query_vec: List[float], k: int
    ) -> List[Tuple[str, float]]:
        """
        Return top-k (item_id, cosine_similarity) pairs.

        The query vector is L2-normalized before search so that inner
        product scores correspond to cosine similarity.  Searches on GPU
        index if available, otherwise CPU.
        """
        if self._cpu_index.ntotal == 0:
            return []
        k = min(k, self._cpu_index.ntotal)
        qvec = _np.array(query_vec, dtype=_np.float32).reshape(1, -1)
        faiss.normalize_L2(qvec)
        scores, indices = self._search_index.search(qvec, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS sentinel for missing results
            results.append((self._ids[idx], float(score)))
        return results

    # -- Utilities -----------------------------------------------------------

    def reset(self) -> None:
        """Clear the index and release GPU resources."""
        self._cpu_index.reset()
        self._search_index = self._cpu_index
        self._ids.clear()
        self._on_gpu = False
        self._gpu_res = None

# ---------------------------------------------------------------------------
# Provenance quality scoring
# ---------------------------------------------------------------------------

_PROVENANCE_SCORES = {
    # (source_kind, has_content_hashes) -> score
    ("doc", True): 1.0,
    ("doc", False): 0.7,
    ("tool", True): 0.8,
    ("tool", False): 0.5,
    ("mixed", True): 0.6,
    ("mixed", False): 0.4,
    ("chat", True): 0.5,
    ("chat", False): 0.2,
}

_VALIDATION_SCORES = {
    "verified": 1.0,
    "unverified": 0.3,
    "contested": 0.1,
    "retracted": 0.0,
}


def _provenance_score(item: MemoryItem) -> float:
    """Score provenance quality (0-1)."""
    has_hashes = bool(item.provenance.content_hashes)
    key = (item.provenance.source_kind, has_hashes)
    prov = _PROVENANCE_SCORES.get(key, 0.2)
    val = _VALIDATION_SCORES.get(item.validation, 0.3)
    return 0.6 * prov + 0.4 * val


def _tag_overlap(query_tags: List[str], item_tags: List[str]) -> float:
    """Weighted Jaccard similarity between tag sets."""
    if not query_tags or not item_tags:
        return 0.0
    q = set(t.lower() for t in query_tags)
    i = set(t.lower() for t in item_tags)
    intersection = len(q & i)
    union = len(q | i)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Recall Engine
# ---------------------------------------------------------------------------

class RecallEngine:
    """
    Hybrid retrieval engine combining tags, embeddings, and provenance.

    When faiss-cpu is installed, the engine builds a FAISS IndexFlatIP
    lazily on first search.  Subsequent searches use the index for O(1)
    approximate nearest-neighbor lookup instead of scanning all stored
    embeddings sequentially.  Tag overlap and provenance quality are
    still applied as post-filter weights on top of the embedding score.

    If faiss-cpu is absent, behavior is unchanged (sequential cosine scan).
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: MemoryEmbedder,
        config: Optional[RecallConfig] = None,
    ):
        """Initialize recall engine with store, embedder, and scoring config."""
        self._store = store
        self._embedder = embedder
        self._config = config or RecallConfig()
        # FAISS acceleration state
        self._faiss_index: Optional[FaissIndex] = None
        self._faiss_built = False

    # -- FAISS index management -----------------------------------------------

    def _build_faiss_index(self) -> None:
        """
        Build FAISS index from all stored embeddings (lazy, once).

        Uses store.all_embeddings() to bulk-load vectors.  Vectors are
        L2-normalized inside FaissIndex.add_batch() so that inner product
        scores equal cosine similarity.

        If faiss-cpu is not installed or the store has no embeddings,
        the index remains None and the engine falls back to sequential
        cosine scanning.
        """
        if self._faiss_built:
            return
        self._faiss_built = True  # mark attempted even on failure

        if not _faiss_available:
            return

        try:
            all_emb = self._store.all_embeddings(exclude_archived=True)
        except Exception as e:
            logger.warning(f"Failed to load embeddings for FAISS: {e}")
            return

        if not all_emb:
            logger.debug("No embeddings in store — FAISS index not built")
            return

        # Infer dimension from first vector
        dimension = len(all_emb[0][1])
        ids = [item_id for item_id, _ in all_emb]
        vectors = [vec for _, vec in all_emb]

        try:
            idx = FaissIndex(dimension, use_gpu=True)
            idx.add_batch(ids, vectors)
            self._faiss_index = idx
            gpu_tag = " [GPU]" if idx.on_gpu else " [CPU]"
            logger.info(
                f"FAISS index built{gpu_tag}: {idx.ntotal} vectors, dim={dimension}"
            )
        except Exception as e:
            logger.warning(f"FAISS index build failed: {e}")
            self._faiss_index = None

    def invalidate_faiss_index(self) -> None:
        """
        Force rebuild of FAISS index on next search.

        Call this after store mutations (write, delete, consolidation) to
        ensure the index reflects the current store state.
        """
        if self._faiss_index is not None:
            self._faiss_index.reset()
        self._faiss_index = None
        self._faiss_built = False

    def search(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        tier: Optional[str] = None,
        type_filter: Optional[str] = None,
        scope: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[MemoryItem]:
        """
        Hybrid search: combines embedding similarity, tag overlap,
        and provenance quality.

        Returns top-k items sorted by combined score.

        When FAISS is available, embedding similarity is computed via
        IndexFlatIP (built lazily on first call).  Tag overlap and
        provenance quality are still applied as post-filter weights.
        """
        k = k or self._config.catalog_k
        tier = tier or self._config.default_tier_filter

        # Step 1: Get candidate items — FTS5 when query present, else scan
        candidates = []
        if query and query.strip():
            # FTS5 candidate generation (BM25-ranked shortlist)
            candidates = self._store.search_fulltext(
                query=query,
                tier=tier,
                type_filter=type_filter,
                scope=scope,
                exclude_archived=self._config.exclude_archived,
                limit=max(k * 5, 50),  # generous shortlist for re-ranking
            )

        # Fallback to scan if FTS5 returned nothing (or no query)
        if not candidates:
            candidates = self._store.list_items(
                tier=tier,
                type_filter=type_filter,
                scope=scope,
                exclude_archived=self._config.exclude_archived,
                limit=200,
            )

        if not candidates:
            return []

        # Step 2: Compute query embedding
        try:
            query_vec = self._embedder.embed_text(query)
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}; falling back to tag-only")
            query_vec = None

        # Step 3: Extract tags from query (simple tokenization)
        query_tags = tags or self._extract_query_tags(query)

        # Step 4: Score each candidate — FAISS path or sequential scan
        scored: List[Tuple[float, MemoryItem]] = []

        # Pre-compute FAISS embedding scores if possible
        faiss_scores = self._faiss_embedding_scores(query_vec, k * 3)

        if faiss_scores is not None:
            # FAISS path: use pre-computed embedding scores
            for item in candidates:
                emb_score = faiss_scores.get(item.id, 0.0)
                score = self._score_item(
                    item, query_vec=None, query_tags=query_tags,
                    precomputed_emb_score=emb_score,
                )
                scored.append((score, item))
        else:
            # Sequential fallback: per-item cosine scan
            for item in candidates:
                score = self._score_item(item, query_vec, query_tags)
                scored.append((score, item))

        # Step 5: Sort and return top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def search_with_scores(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        tier: Optional[str] = None,
        type_filter: Optional[str] = None,
        scope: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        """Search returning items with their scores."""
        k = k or self._config.catalog_k
        tier = tier or self._config.default_tier_filter

        candidates = []
        if query and query.strip():
            candidates = self._store.search_fulltext(
                query=query, tier=tier, type_filter=type_filter, scope=scope,
                exclude_archived=self._config.exclude_archived,
                limit=max(k * 5, 50),
            )
        if not candidates:
            candidates = self._store.list_items(
                tier=tier, type_filter=type_filter, scope=scope,
                exclude_archived=self._config.exclude_archived, limit=200,
            )
        if not candidates:
            return []

        try:
            query_vec = self._embedder.embed_text(query)
        except Exception:
            query_vec = None

        query_tags = tags or self._extract_query_tags(query)

        # Pre-compute FAISS embedding scores if possible
        faiss_scores = self._faiss_embedding_scores(query_vec, k * 3)

        scored = []
        if faiss_scores is not None:
            for item in candidates:
                emb_score = faiss_scores.get(item.id, 0.0)
                score = self._score_item(
                    item, query_vec=None, query_tags=query_tags,
                    precomputed_emb_score=emb_score,
                )
                scored.append((item, score))
        else:
            for item in candidates:
                score = self._score_item(item, query_vec, query_tags)
                scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def format_inject(
        self,
        items: List[MemoryItem],
        budget_tokens: Optional[int] = None,
    ) -> str:
        """Format items for context injection, respecting token budget."""
        budget = budget_tokens or self._config.inject_budget_tokens
        parts = []
        used = 0
        for item in items:
            text = item.format_inject()
            tokens_est = len(text) // 4
            if used + tokens_est > budget:
                break
            parts.append(text)
            used += tokens_est
        return "\n\n".join(parts)

    def format_catalog(self, items: List[MemoryItem]) -> Dict[str, Any]:
        """Format items as memory catalog."""
        return {
            "memory_catalog": [item.format_catalog_entry() for item in items],
        }

    # -- FAISS helpers -----------------------------------------------------

    def _faiss_embedding_scores(
        self,
        query_vec: Optional[List[float]],
        fetch_k: int,
    ) -> Optional[Dict[str, float]]:
        """
        Compute embedding similarity scores via FAISS if available.

        Returns a dict {item_id: cosine_score} for the top fetch_k
        neighbors, or None if FAISS is not usable (missing library,
        no embeddings, or query embedding failed).
        """
        if query_vec is None:
            return None

        # Lazy-build index on first call
        self._build_faiss_index()

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return None

        try:
            hits = self._faiss_index.search(query_vec, fetch_k)
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}; falling back to scan")
            return None

        return {item_id: max(0.0, score) for item_id, score in hits}

    # -- Scoring -----------------------------------------------------------

    def _score_item(
        self,
        item: MemoryItem,
        query_vec: Optional[List[float]],
        query_tags: List[str],
        precomputed_emb_score: Optional[float] = None,
    ) -> float:
        """
        Compute hybrid score for a candidate item.

        Score = w_tag * tag_overlap + w_emb * embedding_sim + w_prov * provenance

        If *precomputed_emb_score* is provided (from FAISS), it is used
        directly and no per-item store lookup is performed.
        """
        w_tag = self._config.tag_weight
        w_emb = self._config.embedding_weight
        w_prov = self._config.provenance_weight

        # Tag overlap
        tag_score = _tag_overlap(query_tags, item.tags)

        # Embedding similarity — FAISS pre-computed or sequential scan
        if precomputed_emb_score is not None:
            emb_score = precomputed_emb_score
        else:
            emb_score = 0.0
            if query_vec is not None:
                emb_data = self._store.read_embedding(item.id)
                if emb_data is not None:
                    item_vec, _ = emb_data
                    emb_score = max(0.0, cosine_similarity(query_vec, item_vec))

        # Provenance quality
        prov_score = _provenance_score(item)

        combined = w_tag * tag_score + w_emb * emb_score + w_prov * prov_score
        return combined

    def _extract_query_tags(self, query: str) -> List[str]:
        """Extract candidate tags from query text (simple word tokenization)."""
        # Remove common stopwords and short words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "between", "through", "during", "before",
            "after", "above", "below", "and", "but", "or", "nor", "not",
            "so", "yet", "both", "either", "neither", "each", "every",
            "all", "any", "few", "more", "most", "other", "some", "such",
            "no", "only", "own", "same", "than", "too", "very",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "how", "when", "where", "why",
            # French stopwords
            "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
            "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi",
            "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
            "son", "sa", "ses", "notre", "votre", "leur", "nous", "vous",
            "il", "elle", "ils", "elles", "on", "se", "en", "y",
            "est", "sont", "a", "ont", "fait", "par", "pour", "sur",
            "dans", "avec", "sans", "sous", "entre", "vers", "chez",
        }
        import re
        words = re.findall(r"\b\w{3,}\b", query.lower())
        return [w for w in words if w not in stopwords]
