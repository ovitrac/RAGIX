"""
Q*-Style Agenda Search Controller

Implements a priority-queue-based search that expands partial explanations
by combining memory items and (optionally) document chunks. This is NOT
simple top-k retrieval — it iteratively builds support sets.

Node = partial explanation with:
  - support_memory_ids: memory items backing the explanation
  - support_doc_refs: document chunks (from RAG layer, if available)
  - open_subgoals: questions that still need answering
  - score: composite quality metric

Operators:
  1. expand_memory  — pull more memory items around missing entities/tags
  2. expand_docs    — pull doc chunks linked to entities (if RAG available)
  3. bridge         — generate a subgoal via small LLM (optional)
  4. verify         — request deterministic validation (optional)

Scoring: S = w_r*R + w_p*P + w_c*C - w_d*D - w_x*X
  R = relevance (embedding similarity aggregate)
  P = provenance quality
  C = coverage gain (new entities satisfied)
  D = duplication penalty
  X = contradiction risk

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_core.memory.config import QSearchConfig
from ragix_core.memory.embedder import MemoryEmbedder, cosine_similarity
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, _generate_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QNode — partial explanation state
# ---------------------------------------------------------------------------

@dataclass
class QNode:
    """A node in the Q*-search tree representing a partial explanation."""

    id: str = field(default_factory=lambda: _generate_id("QN"))
    question: str = ""
    support_memory_ids: List[str] = field(default_factory=list)
    support_doc_refs: List[Dict[str, str]] = field(default_factory=list)
    open_subgoals: List[str] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    # Entities covered by this node's support set
    covered_entities: Set[str] = field(default_factory=set)
    # Parent node id for tree tracing
    parent_id: Optional[str] = None

    def __lt__(self, other: QNode) -> bool:
        """For heapq: higher score = higher priority (negate for min-heap)."""
        return self.score > other.score  # max-heap via negation in heapq


# ---------------------------------------------------------------------------
# Q*-Search Engine
# ---------------------------------------------------------------------------

class QSearchEngine:
    """
    Q*-style agenda search for memory recall.

    Iteratively expands the best partial explanation until:
    - All subgoals resolved AND score > threshold, OR
    - Budget exhausted (max expansions, max tokens, max time)
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: MemoryEmbedder,
        config: Optional[QSearchConfig] = None,
    ):
        """Initialize Q*-search engine with store, embedder, and budget config."""
        self._store = store
        self._embedder = embedder
        self._config = config or QSearchConfig()

    def search(
        self,
        question: str,
        initial_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run Q*-search for the given question.

        Returns dict with:
          - items: list of catalog entries for the best support set
          - best_score: score of the best node
          - expansions: number of expansions performed
          - elapsed_seconds: wall-clock time
        """
        start_time = time.monotonic()

        # Step 1: Initial retrieval
        query_vec = self._safe_embed(question)
        initial_items = self._retrieve_initial(question, query_vec, k=initial_k)

        if not initial_items:
            return {"items": [], "best_score": 0.0, "expansions": 0, "elapsed_seconds": 0.0}

        # Step 2: Build root node
        root = QNode(
            question=question,
            support_memory_ids=[item.id for item in initial_items],
            covered_entities=self._extract_entities(initial_items),
        )
        root.score = self._score_node(root, query_vec)

        # Step 3: Agenda loop
        agenda: List[QNode] = [root]
        best_node = root
        expansions = 0
        seen_ids: Set[str] = set(root.support_memory_ids)

        while agenda and expansions < self._config.max_expansions:
            elapsed = time.monotonic() - start_time
            if elapsed > self._config.max_time_seconds:
                logger.info(f"Q*-search: time budget reached ({elapsed:.1f}s)")
                break

            # Pop best node
            current = heapq.heappop(agenda)

            if current.score > best_node.score:
                best_node = current

            # Check stop condition
            if (
                not current.open_subgoals
                and current.score >= self._config.score_threshold
            ):
                best_node = current
                break

            # Expand: try to find more supporting memory
            children = self._expand_memory(current, query_vec, seen_ids)
            for child in children:
                child.score = self._score_node(child, query_vec)
                heapq.heappush(agenda, child)
                seen_ids.update(child.support_memory_ids)

            expansions += 1

        elapsed = time.monotonic() - start_time

        # Build result from best node
        items = self._store.read_items(best_node.support_memory_ids)
        catalog_entries = [item.format_catalog_entry() for item in items]
        # Add score to each entry
        for entry in catalog_entries:
            entry["score"] = best_node.score

        return {
            "items": catalog_entries,
            "best_score": best_node.score,
            "expansions": expansions,
            "elapsed_seconds": round(elapsed, 3),
        }

    # -- Operators ---------------------------------------------------------

    def _expand_memory(
        self,
        node: QNode,
        query_vec: Optional[List[float]],
        seen_ids: Set[str],
    ) -> List[QNode]:
        """
        Expand a node by finding additional memory items.

        Searches for items related to uncovered entities/tags.
        """
        # Get current support items for their tags/entities
        support_items = self._store.read_items(node.support_memory_ids)
        all_tags = set()
        all_entities = set()
        for item in support_items:
            all_tags.update(item.tags)
            all_entities.update(item.entities)

        # Find items with overlapping tags that aren't already in support set
        candidates = []
        if all_tags:
            tag_results = self._store.search_by_tags(
                tags=list(all_tags), limit=20,
            )
            for item in tag_results:
                if item.id not in seen_ids:
                    candidates.append(item)

        if not candidates:
            return []

        # Score candidates by embedding similarity to query
        scored: List[Tuple[float, MemoryItem]] = []
        for item in candidates:
            if query_vec is not None:
                emb_data = self._store.read_embedding(item.id)
                if emb_data is not None:
                    sim = cosine_similarity(query_vec, emb_data[0])
                    scored.append((sim, item))
                else:
                    scored.append((0.0, item))
            else:
                scored.append((0.0, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Create child nodes (top 3 candidates, one per child)
        children = []
        for sim, item in scored[:3]:
            child = QNode(
                question=node.question,
                support_memory_ids=node.support_memory_ids + [item.id],
                support_doc_refs=list(node.support_doc_refs),
                open_subgoals=list(node.open_subgoals),
                depth=node.depth + 1,
                covered_entities=node.covered_entities | set(item.entities),
                parent_id=node.id,
            )
            children.append(child)

        return children

    # -- Scoring -----------------------------------------------------------

    def _score_node(
        self,
        node: QNode,
        query_vec: Optional[List[float]],
    ) -> float:
        """
        Compute composite score for a node.

        S = w_r*R + w_p*P + w_c*C - w_d*D - w_x*X
        """
        items = self._store.read_items(node.support_memory_ids)
        if not items:
            return 0.0

        cfg = self._config

        # R: Relevance (average embedding similarity to query)
        relevance = 0.0
        if query_vec is not None:
            sims = []
            for item in items:
                emb_data = self._store.read_embedding(item.id)
                if emb_data is not None:
                    sims.append(max(0.0, cosine_similarity(query_vec, emb_data[0])))
            relevance = sum(sims) / len(sims) if sims else 0.0

        # P: Provenance quality (average)
        prov_scores = []
        for item in items:
            val_score = {"verified": 1.0, "unverified": 0.3, "contested": 0.1, "retracted": 0.0}
            has_hashes = bool(item.provenance.content_hashes)
            src_score = 0.8 if has_hashes else 0.3
            prov_scores.append(0.6 * src_score + 0.4 * val_score.get(item.validation, 0.3))
        provenance = sum(prov_scores) / len(prov_scores) if prov_scores else 0.0

        # C: Coverage gain (fraction of covered entities vs total known)
        all_entities = set()
        for item in items:
            all_entities.update(item.entities)
        coverage = len(node.covered_entities) / max(len(all_entities), 1)

        # D: Duplication penalty (pairwise overlap in content)
        duplication = self._compute_duplication(items)

        # X: Contradiction risk (any contested/retracted items)
        contradiction = sum(
            1 for item in items
            if item.validation in ("contested", "retracted")
        ) / max(len(items), 1)

        score = (
            cfg.w_relevance * relevance
            + cfg.w_provenance * provenance
            + cfg.w_coverage * coverage
            - cfg.w_duplication * duplication
            - cfg.w_contradiction * contradiction
        )
        return max(0.0, score)

    def _compute_duplication(self, items: List[MemoryItem]) -> float:
        """Estimate duplication among items (0 = no duplication, 1 = all same)."""
        if len(items) <= 1:
            return 0.0
        # Simple: check tag overlap between pairs
        pairs = 0
        overlap_sum = 0.0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                tags_i = set(t.lower() for t in items[i].tags)
                tags_j = set(t.lower() for t in items[j].tags)
                if tags_i and tags_j:
                    jaccard = len(tags_i & tags_j) / len(tags_i | tags_j)
                    overlap_sum += jaccard
                pairs += 1
        return overlap_sum / pairs if pairs > 0 else 0.0

    # -- Helpers -----------------------------------------------------------

    def _retrieve_initial(
        self,
        question: str,
        query_vec: Optional[List[float]],
        k: int = 5,
    ) -> List[MemoryItem]:
        """Retrieve initial candidate items for the root node."""
        # Get all items with embeddings
        all_embs = self._store.all_embeddings(exclude_archived=True)
        if not all_embs or query_vec is None:
            # Fallback to tag-based search
            tags = self._extract_tags(question)
            if tags:
                return self._store.search_by_tags(tags, limit=k)
            return self._store.list_items(limit=k)

        # Score by cosine similarity
        scored = []
        for item_id, vec in all_embs:
            sim = cosine_similarity(query_vec, vec)
            scored.append((sim, item_id))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_ids = [iid for _, iid in scored[:k]]
        return self._store.read_items(top_ids)

    def _extract_entities(self, items: List[MemoryItem]) -> Set[str]:
        """Collect all entities from a list of items."""
        entities = set()
        for item in items:
            entities.update(item.entities)
        return entities

    def _extract_tags(self, text: str) -> List[str]:
        """Simple tag extraction from text."""
        import re
        words = re.findall(r"\b\w{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "has",
            "what", "when", "where", "how", "who", "which", "this", "that",
        }
        return [w for w in words if w not in stopwords][:10]

    def _safe_embed(self, text: str) -> Optional[List[float]]:
        """Embed text, returning None on failure."""
        try:
            return self._embedder.embed_text(text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
