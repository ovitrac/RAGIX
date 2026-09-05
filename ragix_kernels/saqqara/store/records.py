"""
saqqara.store.records — what the store keeps, and what identifies it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.1, K7.2, K7.5, K7.9.

Two identity rules carry the whole store, and both say the same thing: a thing is
what it contains, never where it was found or when.

- `doc_id` is the source sha256. The same bytes at two paths are one document, and
  moving a file changes nothing — the same rule the asset store already applies.
- `chunk_id` is a digest over (doc_id, level, node_ids, text). Re-chunking an
  unchanged tree therefore writes nothing, which is what makes embedding
  incremental rather than a re-run (K7.5, K7.6).

Neither id contains a float, a timestamp or a path. A float in an id is a
platform-dependent id; a timestamp in an id makes every re-read a new row.

**Node addressing.** The frozen model carries no node identifier, and adding one
would change a frozen interface. The store derives an address instead: a node's
child-index path from the root — `""` for the root, `"0"`, `"0.3.2"`. It is stable
because the tree's own order is stable across runs (K4.1), it resolves by walking,
and it costs the model nothing. An address that does not resolve returns `None`
rather than raising from deep inside a traversal: absence is an answer here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ..model import CANONICAL_JSON, Node, Tree

__all__ = [
    "EDGE_TYPES",
    "ChunkRecord",
    "DocumentRecord",
    "EdgeRecord",
    "EmbeddingRecord",
    "EmbeddingRefusalRecord",
    "Hit",
    "ObjectRecord",
    "chunk_id_for",
    "doc_id_for",
    "node_at",
    "node_ids_of",
]

#: The closed vocabulary of edges between nodes. An edge outside it is refused:
#: an unnamed relation is one nobody can query for, and storing it pretends
#: otherwise.
EDGE_TYPES = ("binds", "refers_to", "continues", "supersedes")

#: The kinds of object the store keeps beside the tree.
OBJECT_KINDS = ("table", "figure", "vector_region")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADDRESS = re.compile(r"^(\d+(\.\d+)*)?$")


# ------------------------------------------------------------------ identity

def doc_id_for(source_sha256: str) -> str:
    """The document id: its bytes, checked rather than trusted.

    A caller that passes a path here has made the mistake this function exists to
    prevent, so the shape is verified instead of accepted.
    """
    if not isinstance(source_sha256, str) or not _SHA256.match(source_sha256):
        raise ValueError(
            f"doc_id must be a lowercase hex sha256 of the source bytes, not {source_sha256!r}"
        )
    return source_sha256


def chunk_id_for(doc_id: str, level: int, node_ids: list[str], text: str) -> str:
    """The chunk id: everything the chunk is, and nothing about when it was made.

    Serialised through the canonical JSON the rest of the package uses, so the id
    does not depend on Python's repr or on dictionary order.
    """
    payload = json.dumps(
        {"doc_id": doc_id, "level": int(level), "node_ids": list(node_ids), "text": text},
        **CANONICAL_JSON,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------ node addressing

def node_ids_of(tree: Tree) -> list[str]:
    """Every node's address, in walk order — root first, then depth first."""

    def walk(node: Node, address: str) -> Iterator[str]:
        yield address
        for i, child in enumerate(node.children):
            yield from walk(child, f"{address}.{i}" if address else str(i))

    return list(walk(tree.root, ""))


def node_at(tree: Tree, node_id: str) -> Optional[Node]:
    """The node at an address, or None. Absence answers; it does not raise."""
    if not isinstance(node_id, str) or not _ADDRESS.match(node_id):
        return None
    node = tree.root
    if not node_id:
        return node
    for step in node_id.split("."):
        index = int(step)
        if index >= len(node.children):
            return None
        node = node.children[index]
    return node


# --------------------------------------------------------------------- records

@dataclass
class DocumentRecord:
    """A document as stored: its bytes' identity, its tree, and how it was read."""

    doc_id: str
    corpus: str
    doc_class: str
    source_path: str
    source_sha256: str
    kernel: str
    kernel_version: str
    tree: Optional[Tree] = None
    meta: dict[str, Any] = field(default_factory=dict)
    digested_at: Optional[str] = None
    trashed: bool = False

    def __post_init__(self) -> None:
        doc_id_for(self.doc_id)
        if self.doc_id != self.source_sha256:
            raise ValueError("doc_id must be the source sha256; a second identity is a second document")

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "digested_at": self.digested_at,
            "doc_class": self.doc_class,
            "doc_id": self.doc_id,
            "kernel": self.kernel,
            "kernel_version": self.kernel_version,
            "meta": self.meta,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "trashed": self.trashed,
            "tree": self.tree.to_dict() if self.tree is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        tree = data.get("tree")
        return cls(
            doc_id=data["doc_id"], corpus=data["corpus"], doc_class=data["doc_class"],
            source_path=data["source_path"], source_sha256=data["source_sha256"],
            kernel=data["kernel"], kernel_version=data["kernel_version"],
            tree=Tree.from_dict(tree) if tree else None,
            meta=data.get("meta", {}), digested_at=data.get("digested_at"),
            trashed=bool(data.get("trashed", False)),
        )


@dataclass
class ObjectRecord:
    """A table, figure or drawn region, kept beside the tree that cites it."""

    doc_id: str
    node_id: str
    kind: str
    page: Optional[int] = None
    bbox: Optional[list[float]] = None
    caption: Optional[str] = None
    asset_ref: Optional[str] = None
    cells: Optional[list[list[Any]]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in OBJECT_KINDS:
            raise ValueError(f"object kind must be one of {OBJECT_KINDS}, not {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_ref": self.asset_ref, "bbox": self.bbox, "caption": self.caption,
            "cells": self.cells, "doc_id": self.doc_id, "kind": self.kind,
            "meta": self.meta, "node_id": self.node_id, "page": self.page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectRecord":
        return cls(**{k: data.get(k) for k in
                      ("doc_id", "node_id", "kind", "page", "bbox", "caption", "asset_ref", "cells")},
                   meta=data.get("meta", {}))


@dataclass
class EdgeRecord:
    """A named relation between two nodes of one document."""

    doc_id: str
    src: str
    dst: str
    type: str

    def __post_init__(self) -> None:
        if self.type not in EDGE_TYPES:
            raise ValueError(f"edge type must be one of {EDGE_TYPES}, not {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "dst": self.dst, "src": self.src, "type": self.type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeRecord":
        return cls(doc_id=data["doc_id"], src=data["src"], dst=data["dst"], type=data["type"])


@dataclass
class ChunkRecord:
    """A retrievable span, and the nodes it was cut from.

    `node_ids` is what makes a hit citable: without it a chunk is text with no
    way back to the document, which is the state this package exists to avoid.
    """

    chunk_id: str
    doc_id: str
    seq: int
    text: str
    level: int
    node_ids: list[str]
    parent_id: Optional[str] = None
    section_path: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    lang: Optional[str] = None
    object_refs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise ValueError(
                "a chunk names at least one node: a chunk citing nothing cannot be "
                "traced back to the document it came from"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "lang": self.lang,
            "level": self.level, "meta": self.meta, "node_ids": self.node_ids,
            "object_refs": self.object_refs, "pages": self.pages,
            "parent_id": self.parent_id, "section_path": self.section_path,
            "seq": self.seq, "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=data["chunk_id"], doc_id=data["doc_id"], seq=data["seq"],
            text=data["text"], level=data["level"], node_ids=list(data["node_ids"]),
            parent_id=data.get("parent_id"), section_path=list(data.get("section_path", [])),
            pages=list(data.get("pages", [])), lang=data.get("lang"),
            object_refs=list(data.get("object_refs", [])), meta=data.get("meta", {}),
        )


@dataclass
class EmbeddingRecord:
    """One vector for one chunk under one model.

    Keyed `(chunk_id, model)` so two models coexist rather than overwrite: an
    embedding is an opinion, and whose opinion it is belongs to the key.
    """

    chunk_id: str
    model: str
    dimensions: int
    vector: tuple[float, ...]
    indexed_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.vector = tuple(float(v) for v in self.vector)
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"vector has {len(self.vector)} components but declares {self.dimensions}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "dimensions": self.dimensions,
            "indexed_at": self.indexed_at, "model": self.model,
            "vector": list(self.vector),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingRecord":
        return cls(chunk_id=data["chunk_id"], model=data["model"],
                   dimensions=data["dimensions"], vector=tuple(data["vector"]),
                   indexed_at=data.get("indexed_at"))


@dataclass
class EmbeddingRefusalRecord:
    """One chunk one model would not embed, kept where the vectors are kept.

    A refusal lives in the store rather than only in the run's report because the
    question it answers — "what does this lane hold, and what is missing from it" —
    is asked of the store long after the run has exited, by a reader, a replay on
    another machine, or the CLI. A count that exists only in a log is a count
    nobody can check.

    It is **not** a mark on the chunk. `existing_embeddings` never consults these
    rows, so the next run asks again: a model change may accept what this one
    refused, and a permanent mark would quietly turn one server's answer into a
    property of the text.

    `signals` carries what the rules read — the model, the batch size, the length,
    the server's own words — on the same principle as the abstention register:
    a record naming a reason and nothing it saw says which test failed and nothing
    about why.
    """

    chunk_id: str
    doc_id: str
    model: str
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)
    refused_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(
                "a refusal without a reason is a defect, not a refusal: "
                f"chunk {self.chunk_id[:12]} under model {self.model!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "model": self.model,
            "reason": self.reason, "refused_at": self.refused_at,
            "signals": dict(self.signals),
        }


@dataclass
class Hit:
    """A retrieved chunk with the ranks that produced it.

    The lane ranks are kept beside the fused rank, never replaced by it. A hit
    found by one lane carries `None` for the other, not a rank of zero or last:
    "this lane did not return it" and "this lane ranked it worst" are different
    facts, and a number cannot say the first.
    """

    chunk: ChunkRecord
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    final_rank: Optional[int] = None
    boosts: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boosts": self.boosts, "chunk": self.chunk.to_dict(),
            "dense_rank": self.dense_rank, "final_rank": self.final_rank,
            "lexical_rank": self.lexical_rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hit":
        return cls(
            chunk=ChunkRecord.from_dict(data["chunk"]),
            dense_rank=data.get("dense_rank"), lexical_rank=data.get("lexical_rank"),
            final_rank=data.get("final_rank"), boosts=data.get("boosts", {}),
        )
