"""
saqqara.store.feed — a read document becomes stored records.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.1, K7.3, K7.11.

Two entry points, one path underneath. `feed_tree` takes what a kernel already has
in memory; `feed_result` takes what a previous run wrote to `stage1/saqqara.json`.
Both end in the same records, because a store fed two ways that disagreed would be
a store whose content depended on how you got there.

What is extracted, and from what:

Facts are serialised through the model's own helper rather than a copy of it.
An `Abstention` reaching here is stored as its dict, and anything neither
JSON-native nor `to_dict`-aware raises — the first version of this module had its
own version that fell back to `repr()`, which is the silent stringification K1.10
now forbids, written into the module whose job is to lose nothing.

- **Objects** are nodes whose kind the store keeps beside the tree — `table`,
  `figure`, `vector_region`. Their `asset_ref` is the asset store's sha256 when
  the reader recorded one; a figure with no stored bytes still becomes an object,
  because a figure that was seen is a fact whether or not its pixels were kept.
- **Edges** are read from the facts an analyzer already wrote. `caption_of` and
  `captioned_by` become `binds`. Nothing is inferred here: the feed copies a
  decision another layer made and recorded, and a feed that inferred relations
  would be an analyzer nobody gated.

Neither reads the source file again. Everything comes from the tree, which is the
only thing that carries provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from ..analyzers.grid import grid_cells
from ..model import Node, Tree, _serialisable
from .chunker import ChunkPlan, chunk_tree
from .records import (
    ChunkRecord,
    DocumentRecord,
    EdgeRecord,
    ObjectRecord,
    doc_id_for,
    node_ids_of,
)

__all__ = ["CELL_COLUMNS", "FeedResult", "feed_result", "feed_tree", "objects_of", "edges_of"]

#: Node kinds the store keeps as objects beside the tree.
OBJECT_FROM_KIND = {"table": "table", "figure": "figure", "vector_region": "vector_region"}

#: One persisted table cell, `objects.cells_json` being a list of these rows, when
#: `store_options.table_cells` is on (K7.11). Positions are the grid vocabulary the
#: header-band rules read — one-based, A1 ranges, the same for a spreadsheet, a
#: word-processing table and a slide table — so a cell is joined to the
#: `header_rows` and `label_cols` recorded in the object's meta without knowing
#: its format. `node_id` is the cell's address in the stored tree: the way back to
#: its citation. `merged` is the A1 extent of a merge anchored here, or null. A
#: vertical-merge continuation carries no value and is covered by its anchor's
#: extent, as the grid mapping has it.
CELL_COLUMNS = ("node_id", "row", "col", "merged", "text")


def _cells_of(table: Node, address_of: dict[int, str]) -> list[list[Any]]:
    """A table's cells in the grid vocabulary, in row then column order."""
    cells, _mapping = grid_cells(table)
    return [[address_of[id(c.node)], c.row, c.col, c.merged, c.text]
            for c in sorted(cells, key=lambda c: (c.row, c.col))]

#: The fact that names a binding, and the edge it becomes. `caption_of` carries
#: the LOCATOR of the figure a caption belongs to.
#:
#: `captioned_by` is deliberately absent. It carries the locator of the text line
#: the caption was made from — the caption's own provenance, not a relation to a
#: figure. Mapping it to `binds` as well would double every edge and assert a
#: binding between a caption and the line it came from, which no analyzer decided.
BINDING_FROM_FACT = {"caption_of": "binds"}


class FeedResult:
    """What one document contributed, and what was refused on the way in."""

    def __init__(self, document: DocumentRecord, objects: list[ObjectRecord],
                 edges: list[EdgeRecord], plan: ChunkPlan) -> None:
        self.document = document
        self.objects = objects
        self.edges = edges
        self.plan = plan

    @property
    def chunks(self) -> list[ChunkRecord]:
        return self.plan.chunks

    def counts(self) -> dict[str, int]:
        return {"objects": len(self.objects), "edges": len(self.edges), **self.plan.counts()}


def _address_index(tree: Tree) -> dict[str, Node]:
    index: dict[str, Node] = {}
    for address in node_ids_of(tree):
        node = tree.root
        if address:
            for step in address.split("."):
                node = node.children[int(step)]
        index[address] = node
    return index


def objects_of(tree: Tree, doc_id: str, *, table_cells: bool = False) -> list[ObjectRecord]:
    """Every node the store keeps as an object, with what the reader recorded.

    `table_cells` (default off) also keeps each table's cells, as `CELL_COLUMNS`
    rows. Off, a table's `cells` stays None, as it always was.
    """
    found: list[ObjectRecord] = []
    index = _address_index(tree)
    address_of = {id(node): address for address, node in index.items()} if table_cells else {}
    for address, node in index.items():
        kind = OBJECT_FROM_KIND.get(node.kind)
        if kind is None:
            continue
        facts = node.facts or {}
        bbox = None
        for key in ("bbox", "box"):
            if isinstance(facts.get(key), (list, tuple)) and len(facts[key]) == 4:
                bbox = [float(v) for v in facts[key]]
                break
        if bbox is None and all(k in facts for k in ("x", "y", "w", "h")):
            bbox = [float(facts["x"]), float(facts["y"]),
                    float(facts["x"]) + float(facts["w"]),
                    float(facts["y"]) + float(facts["h"])]
        page = next((getattr(loc, "page", None) for loc in node.provenance.chain
                     if isinstance(getattr(loc, "page", None), int)), None)
        found.append(ObjectRecord(
            doc_id=doc_id, node_id=address, kind=kind, page=page, bbox=bbox,
            caption=facts.get("caption"),
            asset_ref=facts.get("asset_ref") or facts.get("sha256"),
            cells=(_cells_of(node, address_of) if table_cells and kind == "table"
                   else facts.get("cells")),
            meta={k: _serialisable(v) for k, v in facts.items()
                  if k not in ("bbox", "box", "caption", "asset_ref", "sha256", "cells")},
        ))
    return found


def edges_of(tree: Tree, doc_id: str,
             refusals: Optional[list[dict[str, Any]]] = None) -> list[EdgeRecord]:
    """Bindings an analyzer decided and recorded, copied — never re-decided here.

    An analyzer records its partner as a serialised LOCATOR, not as an address:
    the model has no addresses, and a locator is what a node already carries. So
    the locator is resolved back to the node that answers to it.

    Resolution can be ambiguous — two nodes may share a leaf locator, a figure and
    the page region it sits in, for instance. Where the ambiguity cannot be
    settled by the kind the fact names, the edge is REFUSED and counted rather
    than resolved by picking the first: an edge chosen arbitrarily is an assertion
    nobody made.
    """
    index = _address_index(tree)
    trace = refusals if refusals is not None else []

    by_locator: dict[str, list[str]] = {}
    for address, node in index.items():
        key = json.dumps(node.provenance.leaf.to_dict(), sort_keys=True)
        by_locator.setdefault(key, []).append(address)

    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str, str]] = set()

    for address, node in index.items():
        for fact, edge_type in BINDING_FROM_FACT.items():
            target = (node.facts or {}).get(fact)
            if not isinstance(target, dict):
                continue
            candidates = by_locator.get(json.dumps(target, sort_keys=True), [])
            if not candidates:
                trace.append({"reason": "binding-target-not-found",
                              "node_id": address, "fact": fact})
                continue
            figures = [a for a in candidates if index[a].kind == "figure"]
            resolved = figures or candidates
            if len(resolved) != 1:
                trace.append({"reason": "binding-target-ambiguous", "node_id": address,
                              "fact": fact, "candidates": len(resolved)})
                continue
            key = (address, resolved[0], edge_type)
            if key not in seen:
                seen.add(key)
                edges.append(EdgeRecord(doc_id=doc_id, src=address,
                                        dst=resolved[0], type=edge_type))
    return edges


def feed_tree(
    tree: Tree,
    source_path: str,
    source_sha256: str,
    *,
    doc_class: str = "document",
    corpus: str = "default",
    kernel: str = "saqqara",
    kernel_version: str = "1.0",
    digested_at: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
    table_cells: bool = False,
    **chunker_options: Any,
) -> FeedResult:
    """Turn one tree into the records a store accepts.

    `table_cells` is `store_options.table_cells` (default off): see `objects_of`.
    """
    doc_id = doc_id_for(source_sha256)
    document = DocumentRecord(
        doc_id=doc_id, corpus=corpus, doc_class=doc_class, source_path=source_path,
        source_sha256=source_sha256, kernel=kernel, kernel_version=kernel_version,
        tree=tree, meta=meta or {}, digested_at=digested_at,
    )
    objects = objects_of(tree, doc_id, table_cells=table_cells)
    plan = chunk_tree(tree, doc_id=doc_id, **chunker_options)
    # A binding the feed could not resolve joins the chunker's refusals rather
    # than vanishing: the two are the same kind of fact about one document.
    edges = edges_of(tree, doc_id, refusals=plan.refusals)

    # A chunk's object_refs are the objects whose node it covers. Computed here
    # rather than in the chunker, which knows about text and not about assets.
    by_node = {obj.node_id: obj for obj in objects}
    for chunk in plan.chunks:
        refs = [by_node[nid].node_id for nid in chunk.node_ids if nid in by_node]
        if refs:
            chunk.object_refs = refs

    return FeedResult(document, objects, edges, plan)


def feed_result(payload: dict[str, Any] | str | Path, **options: Any) -> list[FeedResult]:
    """Feed from what a previous run stored.

    Accepts the parsed result, a path to `stage1/saqqara.json`, or its text. The
    envelope wraps the kernel's data under "data"; both shapes are read, because
    a caller holding one should not have to know which.
    """
    if isinstance(payload, (str, Path)):
        path = Path(payload)
        payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    documents = data.get("documents") or []
    if not documents:
        raise ValueError(
            "no documents in this result: feeding an empty run would create an "
            "empty store that looks like a successful one"
        )

    fed: list[FeedResult] = []
    for entry in documents:
        tree = Tree.from_dict(entry["tree"])
        fed.append(feed_tree(
            tree,
            source_path=entry["path"],
            source_sha256=entry["sha256"],
            doc_class=entry.get("format", "document"),
            **options,
        ))
    return fed
