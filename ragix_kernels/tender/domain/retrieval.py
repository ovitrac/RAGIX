"""tender.retrieval — the policy the document kernel deliberately does not own.

The public kernel stops at fusion over ranks: `saqqara.store.retrieve.Retriever`
runs the two lanes, fuses `1/(k + rank)`, keeps each lane's rank beside the
fused one, and its own specification says why it stops there — authority,
freshness and quality boosts *"need policies this package does not own"*. This
module is those policies: weights, boosts, filters, structural expansion, and a
context budget.

Written as a re-specification rather than moved. The lane it replaces reached
three modules of the in-tree substrate, and a file carried across a boundary
brings its imports with it; the boundary would have been breached by the act of
drawing it.

Three mappings that are **meaning, not shape** — each one a place where a
name-for-name rewrite compiles and is wrong:

1. **`doc_types` filters on `routing_class`, never on the kernel's
   `doc_class`.** The kernel's field is the FILE FORMAT; the filter wants the
   structural class — `P2` is "a pdf that declares no outline", not "a pdf".
   Mapped field to field, the filter still filters, on something else.
2. **A `Hit` carries its `ChunkRecord`.** The lane it replaces held `chunk_id`,
   `doc_id` and `text` on the hit and fetched the chunk again to select it. The
   chunk is already in hand, so the round trip is gone rather than translated.
3. **Fusion here is WEIGHTED, and that is why it is not delegated.** The
   kernel's `search()` fuses the lanes unweighted, which is the property its
   gates pin (K7.9, and K7.17/K7.18 for the baselines). Lane weights are policy,
   so this module calls `dense()` and `lexical()` and fuses them itself — it
   does not reimplement a kernel decision, it adds one the kernel declined.

Ranks are never collapsed: dense, lexical and final travel separately into the
trace, and every boost is echoed with the value that produced it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ragix_kernels.saqqara.store.records import ChunkRecord, Hit, ObjectRecord
from ragix_kernels.saqqara.store.retrieve import RRF_K, Retriever

from .records import DocumentFacts

__all__ = ["ExpansionCfg", "Filters", "RetrievalConfig", "RetrievalResult",
           "RetrievedChunk", "retrieve"]


@dataclass
class ExpansionCfg:
    parents: bool = True
    edges: tuple[str, ...] = ("binds", "refers_to", "continues")
    objects: bool = True
    max_objects: int = 5


@dataclass
class Filters:
    """Constraints on DOCUMENT attributes, applied after both lanes ran.

    ISO date strings compare lexicographically. `provenance` matches source-path
    basenames or doc_ids. `doc_types` names ROUTING classes.
    """

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    doc_types: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"date_from": self.date_from, "date_to": self.date_to,
                "doc_types": self.doc_types, "provenance": self.provenance}

    def empty(self) -> bool:
        return not (self.date_from or self.date_to
                    or self.doc_types or self.provenance)


@dataclass
class RetrievalConfig:
    """The policy knobs. The three the kernel also declares are read from its
    configuration rather than defaulted twice — one rule, one place."""

    dense_k: int = 40
    lexical_k: int = 40
    rrf_k0: int = RRF_K
    weights: dict[str, float] = field(
        default_factory=lambda: {"dense": 1.0, "lexical": 1.0})
    boosts: dict[str, float] = field(
        default_factory=lambda: {"authority": 0.0, "freshness": 0.1,
                                 "quality": 0.1, "duplicate_penalty": 0.2})
    expansion: ExpansionCfg = field(default_factory=ExpansionCfg)
    context_budget_chars: int = 24000

    @classmethod
    def from_store_config(cls, store_config: Any, **overrides: Any) -> "RetrievalConfig":
        """Take `dense_k`, `lexical_k` and `rrf_k` from the kernel's own
        `retrieval` section, so the shared knobs cannot drift apart."""
        section = store_config.section("retrieval") if store_config else {}
        base = cls(dense_k=int(section.get("dense_k", cls.dense_k)),
                   lexical_k=int(section.get("lexical_k", cls.lexical_k)),
                   rrf_k0=int(section.get("rrf_k", cls.rrf_k0)))
        for key, value in overrides.items():
            setattr(base, key, value)
        return base


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    dense_rank: Optional[int]          # 1-based within its lane; None = absent
    lexical_rank: Optional[int]
    final_rank: int
    score: float
    boosts: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    selected: list[RetrievedChunk] = field(default_factory=list)
    parents: list[ChunkRecord] = field(default_factory=list)
    related: list[ChunkRecord] = field(default_factory=list)
    objects: list[ObjectRecord] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def _text_fingerprint(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def _year(date: Optional[str]) -> Optional[int]:
    if date and len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


class _DocCache:
    """One store round trip per document, per kind.

    `get_chunk(chunk_id)` is not in the store protocol, so a chunk is found
    through its document's chunk list — which this cache already holds.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self._docs: dict[str, Any] = {}
        self._facts: dict[str, Optional[DocumentFacts]] = {}
        self._chunks: dict[str, list[ChunkRecord]] = {}
        self._objects: dict[str, list[ObjectRecord]] = {}
        self._edges: dict[str, list[Any]] = {}

    def doc(self, doc_id: str) -> Any:
        if doc_id not in self._docs:
            self._docs[doc_id] = self.store.get_document(doc_id)
        return self._docs[doc_id]

    def facts(self, doc_id: str) -> Optional[DocumentFacts]:
        if doc_id not in self._facts:
            document = self.doc(doc_id)
            self._facts[doc_id] = DocumentFacts.of(document) if document else None
        return self._facts[doc_id]

    def chunks(self, doc_id: str) -> list[ChunkRecord]:
        if doc_id not in self._chunks:
            self._chunks[doc_id] = self.store.get_chunks(doc_id)
        return self._chunks[doc_id]

    def chunk(self, doc_id: str, chunk_id: str) -> Optional[ChunkRecord]:
        return next((c for c in self.chunks(doc_id) if c.chunk_id == chunk_id), None)

    def objects(self, doc_id: str) -> list[ObjectRecord]:
        if doc_id not in self._objects:
            self._objects[doc_id] = self.store.get_objects(doc_id)
        return self._objects[doc_id]

    def edges(self, doc_id: str) -> list[Any]:
        if doc_id not in self._edges:
            self._edges[doc_id] = self.store.get_edges(doc_id)
        return self._edges[doc_id]


def _passes(filters: Filters, cache: _DocCache, doc_id: str) -> bool:
    document = cache.doc(doc_id)
    if document is None:
        return False
    facts = cache.facts(doc_id)

    if filters.date_from or filters.date_to:
        date = facts.document_date if facts else None
        if date is None:
            return False               # a dated filter excludes undated documents
        if filters.date_from and date < filters.date_from:
            return False
        if filters.date_to and date > filters.date_to:
            return False

    if filters.doc_types:
        # The ROUTING class, not the kernel's `doc_class`. A document carrying
        # no facts is excluded rather than assumed into a class.
        if facts is None or facts.routing_class not in filters.doc_types:
            return False

    if filters.provenance:
        basename = document.source_path.rsplit("/", 1)[-1]
        if doc_id not in filters.provenance and basename not in filters.provenance:
            return False
    return True


def retrieve(store: Any, embedder: Any, query: str,
             cfg: Optional[RetrievalConfig] = None,
             filters: Optional[Filters] = None,
             model: str = "") -> RetrievalResult:
    """Both lanes through the kernel's retriever, then this side's policy."""
    cfg = cfg or RetrievalConfig()
    filters = filters or Filters()
    cache = _DocCache(store)
    retriever = Retriever(store, model=model, rrf_k=cfg.rrf_k0)

    # -- lanes: the kernel runs them, and each hit arrives with its chunk -----
    vector = embedder.embed_batch([query])[0] if embedder is not None else None
    dense: list[Hit] = (retriever.dense(vector, top_k=cfg.dense_k)
                        if vector is not None else [])
    lexical: list[Hit] = retriever.lexical(query, top_k=cfg.lexical_k)

    dense_rank = {h.chunk.chunk_id: h.dense_rank for h in dense}
    lexical_rank = {h.chunk.chunk_id: h.lexical_rank for h in lexical}
    by_id: dict[str, Hit] = {h.chunk.chunk_id: h for h in [*dense, *lexical]}

    if not filters.empty():
        by_id = {cid: h for cid, h in by_id.items()
                 if _passes(filters, cache, h.chunk.doc_id)}

    # -- weighted RRF, then additive boosts ----------------------------------
    w_dense = cfg.weights.get("dense", 1.0)
    w_lex = cfg.weights.get("lexical", 1.0)
    b = cfg.boosts
    # Freshness is relative to the NEWEST candidate, never to a wall clock: the
    # same corpus and the same query must give the same ranking tomorrow.
    newest = max((y for y in (_year(f.document_date if f else None)
                              for f in (cache.facts(h.chunk.doc_id)
                                        for h in by_id.values()))
                  if y is not None), default=None)

    fused: list[tuple[float, dict[str, float], Hit]] = []
    for cid, hit in by_id.items():
        score = 0.0
        if cid in dense_rank:
            score += w_dense / (cfg.rrf_k0 + dense_rank[cid])
        if cid in lexical_rank:
            score += w_lex / (cfg.rrf_k0 + lexical_rank[cid])
        boosts: dict[str, float] = {}
        document = cache.doc(hit.chunk.doc_id)
        facts = cache.facts(hit.chunk.doc_id)
        if document is not None:
            authority = float((document.meta or {}).get("authority", 0.0))
            if b.get("authority") and authority:
                boosts["authority"] = b["authority"] * authority
        if facts is not None:
            year = _year(facts.document_date)
            if b.get("freshness") and newest is not None and year is not None:
                boosts["freshness"] = b["freshness"] / (1 + (newest - year))
            debris = facts.quality.get("debris_score")
            if b.get("quality") and debris is not None:
                boosts["quality"] = b["quality"] * (1.0 - float(debris))
        # Boosts scale like an RRF term: a top-rank lane contributes about 1/k0.
        score += sum(boosts.values()) / cfg.rrf_k0
        fused.append((score, boosts, hit))

    fused.sort(key=lambda t: (-t[0], t[2].chunk.chunk_id))     # deterministic ties

    if b.get("duplicate_penalty"):
        seen: set[str] = set()
        penalised: list[tuple[float, dict[str, float], Hit]] = []
        for score, boosts, hit in fused:
            fingerprint = _text_fingerprint(hit.chunk.text)
            if fingerprint in seen:
                boosts = {**boosts, "duplicate_penalty": -b["duplicate_penalty"]}
                score -= b["duplicate_penalty"] / cfg.rrf_k0
            else:
                seen.add(fingerprint)
            penalised.append((score, boosts, hit))
        penalised.sort(key=lambda t: (-t[0], t[2].chunk.chunk_id))
        fused = penalised

    # -- selection under the context budget ----------------------------------
    selected: list[RetrievedChunk] = []
    budget, used = cfg.context_budget_chars, 0
    for score, boosts, hit in fused:
        chunk = hit.chunk
        if selected and used + len(chunk.text) > budget:
            continue                    # skip the oversized, keep trying smaller
        used += len(chunk.text)
        selected.append(RetrievedChunk(
            chunk=chunk, dense_rank=dense_rank.get(chunk.chunk_id),
            lexical_rank=lexical_rank.get(chunk.chunk_id),
            final_rank=len(selected) + 1, score=score, boosts=boosts))

    # -- structural expansion -------------------------------------------------
    parents: list[ChunkRecord] = []
    related: list[ChunkRecord] = []
    object_ids: list[tuple[str, str]] = []
    seen_chunks = {rc.chunk.chunk_id for rc in selected}
    edges_followed: list[str] = []

    for rc in selected:
        chunk = rc.chunk
        if cfg.expansion.objects:
            for node_id in chunk.object_refs:
                if (chunk.doc_id, node_id) not in object_ids:
                    object_ids.append((chunk.doc_id, node_id))
        if cfg.expansion.parents and chunk.parent_id \
                and chunk.parent_id not in seen_chunks:
            parent = cache.chunk(chunk.doc_id, chunk.parent_id)
            if parent is not None and used + len(parent.text) <= budget:
                parents.append(parent)
                seen_chunks.add(parent.chunk_id)
                used += len(parent.text)
        if cfg.expansion.edges:
            node_set = set(chunk.node_ids)
            object_nodes = {o.node_id for o in cache.objects(chunk.doc_id)}
            for edge in cache.edges(chunk.doc_id):
                if edge.type not in cfg.expansion.edges or edge.src not in node_set:
                    continue
                edges_followed.append(edge.type)
                if edge.dst in object_nodes:
                    if cfg.expansion.objects \
                            and (chunk.doc_id, edge.dst) not in object_ids:
                        object_ids.append((chunk.doc_id, edge.dst))
                    continue
                for other in cache.chunks(chunk.doc_id):
                    if edge.dst in other.node_ids \
                            and other.chunk_id not in seen_chunks \
                            and used + len(other.text) <= budget:
                        related.append(other)
                        seen_chunks.add(other.chunk_id)
                        used += len(other.text)
                        break

    objects: list[ObjectRecord] = []
    if cfg.expansion.objects:
        for doc_id, node_id in object_ids[:cfg.expansion.max_objects]:
            for obj in cache.objects(doc_id):
                if obj.node_id == node_id:
                    objects.append(obj)
                    break

    def _hit(rc: RetrievedChunk) -> dict[str, Any]:
        text = rc.chunk.text
        snippet = text[:240]
        if len(text) > 240 and " " in snippet:
            snippet = snippet.rsplit(" ", 1)[0] + " …"
        return {"chunk_id": rc.chunk.chunk_id, "dense_rank": rc.dense_rank,
                "lexical_rank": rc.lexical_rank, "final_rank": rc.final_rank,
                "score": round(rc.score, 5), "boosts": rc.boosts,
                # the exact RRF terms: the fusion is inspectable, not asserted
                "dense_contrib": round(w_dense / (cfg.rrf_k0 + rc.dense_rank), 5)
                                 if rc.dense_rank else 0.0,
                "lexical_contrib": round(w_lex / (cfg.rrf_k0 + rc.lexical_rank), 5)
                                   if rc.lexical_rank else 0.0,
                "snippet": snippet}

    trace = {
        "hits": [_hit(rc) for rc in selected],
        "filters": filters.to_dict() if not filters.empty() else {},
        "expansion": {"parents": [p.chunk_id for p in parents],
                      "edges": edges_followed,
                      "objects": [o.node_id for o in objects],
                      "related": [r.chunk_id for r in related]},
        "dense_k": cfg.dense_k, "lexical_k": cfg.lexical_k,
        "rrf_k0": cfg.rrf_k0, "context_chars_used": used,
        "dense_lane": "disabled (no embedder)" if vector is None else "on",
        "dropped_objects": max(0, len(object_ids) - cfg.expansion.max_objects),
    }
    return RetrievalResult(selected=selected, parents=parents, related=related,
                           objects=objects, trace=trace)
