"""
saqqara.store.retrieve — two lanes, one fusion, every hit citable.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.8, K7.9, K7.10, K7.11, K7.13.

The dense lane reads its vectors from the DATABASE and loads them into a
`ragix_core` index on open. The database is the source of truth; the index is a
rebuildable cache. Deleting the cache and reopening must reproduce identical hits
(K7.8) — that is what distinguishes a cache from a second store, and a second
store is a thing that can disagree with the first.

Lane ranks are kept, never replaced by the fused rank. A hit found by one lane
carries `None` for the other, because "this lane did not return it" and "this lane
ranked it worst" are different facts and a number cannot say the first.

**On the fusion, stated rather than glossed.** `ragix_core.hybrid_search` fuses
over RANKS — `1 / (k + rank)` — which is the semantics this package requires and
the reason the store may use it at all. But `HybridSearchEngine._fuse_rrf` returns
a code-chunk shape (`file_path`, `start_line`, `chunk_type`), so a document family
cannot call it without inventing fields it has no values for. What is reproduced
here is the FORMULA and its `rrf_k`, which is arithmetic; what is not reproduced
is a component. Generalising that method in `ragix_core` would let this call it
directly and is worth doing — it is a gap upstream, recorded here rather than
worked around silently.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..model import Node, Tree
from .records import ChunkRecord, Hit, node_at

__all__ = ["RRF_K", "Retriever", "provenance_of", "related_of"]

#: The rank-sensitivity constant of Reciprocal Rank Fusion. 60 is the value
#: `ragix_core` defaults to; it is repeated rather than imported so the store's
#: behaviour does not change silently when that default does.
RRF_K = 60


class Retriever:
    """The two lanes over one store, with the dense index built from the DB."""

    def __init__(self, store: Any, model: str = "", backend: str = "numpy",
                 rrf_k: int = RRF_K) -> None:
        self.store = store
        self.model = model
        self.backend = backend
        self.rrf_k = rrf_k
        self._index: Optional[Any] = None
        self._chunk_ids: list[str] = []

    # ------------------------------------------------------------- the index

    def _ensure_index(self) -> Optional[Any]:
        """Load vectors from the database into a ragix_core index.

        Rebuilt from the DB every time it is needed and never persisted as an
        authoritative artefact: a cache that outlives its source is a second
        opinion about what the corpus contains.
        """
        if self._index is not None:
            return self._index
        if not self.model:
            return None

        records = self.store.get_embeddings(self.model)
        if not records:
            return None

        from ragix_core.vector_index import create_vector_index

        index = create_vector_index(dimension=records[0].dimensions, backend=self.backend)
        index.add([list(r.vector) for r in records],
                  [{"chunk_id": r.chunk_id} for r in records])
        self._index = index
        self._chunk_ids = [r.chunk_id for r in records]
        return index

    def invalidate(self) -> None:
        """Drop the cache. The next query rebuilds it from the database."""
        self._index = None
        self._chunk_ids = []

    # -------------------------------------------------------------- the lanes

    def dense(self, vector: Iterable[float], top_k: int) -> list[Hit]:
        """Vectors from the DB, ranked. Empty when nothing has been embedded."""
        index = self._ensure_index()
        if index is None:
            return []
        results = index.search(list(vector), k=top_k)

        by_id = {c.chunk_id: c for c in self.store.get_chunks()}
        hits: list[Hit] = []
        for rank, result in enumerate(results, 1):
            chunk_id = _chunk_id_of(result)
            chunk = by_id.get(chunk_id)
            if chunk is None:
                # A vector whose chunk is gone is not a hit. It is counted by the
                # store's own accounting, never returned as a result with no text.
                continue
            hits.append(Hit(chunk=chunk, dense_rank=rank))
        return hits

    def lexical(self, query: str, top_k: int) -> list[Hit]:
        return self.store.lexical_search(query, top_k)

    # --------------------------------------------------------------- fusion

    def search(self, query: str, vector: Optional[Iterable[float]] = None,
               top_k: int = 10, dense_k: int = 40, lexical_k: int = 40) -> list[Hit]:
        """Both lanes, fused over ranks, with the lane ranks preserved."""
        dense = self.dense(vector, dense_k) if vector is not None else []
        lexical = self.lexical(query, lexical_k)

        merged: dict[str, Hit] = {}
        score: dict[str, float] = {}

        for hit in dense:
            merged[hit.chunk.chunk_id] = Hit(chunk=hit.chunk, dense_rank=hit.dense_rank)
            score[hit.chunk.chunk_id] = 1.0 / (self.rrf_k + hit.dense_rank)

        for hit in lexical:
            existing = merged.get(hit.chunk.chunk_id)
            if existing is None:
                merged[hit.chunk.chunk_id] = Hit(chunk=hit.chunk,
                                                 lexical_rank=hit.lexical_rank)
                score[hit.chunk.chunk_id] = 0.0
            else:
                existing.lexical_rank = hit.lexical_rank
            score[hit.chunk.chunk_id] += 1.0 / (self.rrf_k + hit.lexical_rank)

        ordered = sorted(merged.values(),
                         key=lambda h: (-score[h.chunk.chunk_id], h.chunk.chunk_id))
        for rank, hit in enumerate(ordered[:top_k], 1):
            hit.final_rank = rank
        return ordered[:top_k]


def _chunk_id_of(result: Any) -> Optional[str]:
    """The chunk id a ragix_core search result carries, wherever it keeps it."""
    for holder in (result, getattr(result, "metadata", None)):
        if holder is None:
            continue
        if isinstance(holder, dict) and "chunk_id" in holder:
            return holder["chunk_id"]
        found = getattr(holder, "chunk_id", None)
        if isinstance(found, str):
            return found
    return None


# ------------------------------------------------------- what a hit leads back to

def provenance_of(tree: Tree, chunk: ChunkRecord) -> list[dict[str, Any]]:
    """The citation chain of every node a chunk names.

    This is what a hit is FOR. A chunk that could not answer this would be text
    with a score, which is the thing this package exists not to produce.
    """
    chains = []
    for node_id in chunk.node_ids:
        node = node_at(tree, node_id)
        if node is None:
            continue
        chains.append({
            "node_id": node_id,
            "kind": node.kind,
            "origin": node.origin,
            "confidence": node.confidence,
            "source_path": node.provenance.source_path,
            "source_format": node.provenance.source_format,
            "chain": [loc.to_dict() for loc in node.provenance.chain],
        })
    return chains


def related_of(store: Any, chunk: ChunkRecord) -> dict[str, list[dict[str, Any]]]:
    """Objects a chunk covers, and the nodes bound to them.

    Read from the edges an analyzer decided and the store kept — nothing here
    decides a relation, it only follows one.
    """
    objects = {o.node_id: o for o in store.get_objects(chunk.doc_id)}
    edges = store.get_edges(chunk.doc_id)
    covered = set(chunk.node_ids)

    found_objects = [objects[n].to_dict() for n in chunk.node_ids if n in objects]
    related = [
        {"src": e.src, "dst": e.dst, "type": e.type}
        for e in edges if e.src in covered or e.dst in covered
    ]
    return {"objects": found_objects, "related": related}
