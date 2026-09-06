"""
saqqara.store.chunker — a tree becomes retrievable spans, along its own structure.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.3, K7.4, K7.5.

The structure is already known, so it is used. Chunks are cut along the tree, not
across a byte stream: one chunk per semantic unit at level 0, and one roll-up per
section at level 1, each level-0 unit naming its roll-up as `parent_id`. That is
small-to-big retrieval — match the precise thing, return with the context it sat
in — and it costs nothing here because the tree was built before any of this.

Three rules, each carried by a proposition:

- **A chunk names the nodes it came from.** A chunk with no node is refused and
  counted, never stored: text with no way back to a document is the thing this
  package exists not to produce (K7.3).
- **A roll-up covers exactly its children.** Not approximately, not a window over
  them — the union of its level-0 children's node ids, in order (K7.4).
- **A unit is never split by a window unless it must be.** An oversized node
  becomes its own chunk and is flagged rather than cut; only a single node longer
  than the fallback window is windowed, and every piece of it says so.

The window is the declared fallback, not the default. A chunker that windows by
default has thrown away the structure it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ..model import Node, Tree
from .records import ChunkRecord, chunk_id_for, node_ids_of

__all__ = ["CHUNKABLE_KINDS", "NON_CHUNKABLE_KINDS", "ChunkPlan", "chunk_tree"]

#: Kinds that carry text a reader would retrieve.
#:
#: Written first from the shapes the fixture GENERATORS produce, which was wrong:
#: the adapters emit `slide`, `shape` and `note` for presentations and `cell` for
#: grids, none of which were here. A whole format contributed zero chunks and the
#: gates did not notice, because they assert on trees the generators built rather
#: than on trees a reader produced. The demo found it in one run.
#: `marker` joined the list on the same evidence, from the demo corpus rather than
#: from a fixture: three Word forms (Annexes 5, 6 and 7) carried **6 330 characters**
#: in 22, 70 and 7 marker nodes that reached no chunk: the field labels a form is
#: answered on — the place-and-date line above a signature, and a lot name followed
#: by its yes and no boxes. A form whose labels are not retrievable cannot be
#: matched to the question it answers, which is the whole task here.
CHUNKABLE_KINDS = (
    "paragraph", "heading", "list", "list_item", "table", "caption", "block",
    "slide", "shape", "note", "cell", "marker",
)

#: Kinds a builder can produce that are deliberately NOT chunked, each with the
#: reason. Declared as a table rather than left as an absence: twice now a
#: text-carrying kind was missing from the list above and nothing noticed, because
#: "not chunkable" and "nobody thought about it" look identical in a tuple. The
#: gate pairs this with what the format plans actually emit, so a new kind must be
#: put in one list or the other before it can reach a tree.
NON_CHUNKABLE_KINDS = {
    "section": "a section names a roll-up and is the context of its chunks, never a chunk",
    "page": "a container; the paragraphs and figures on the page carry its text",
    "figure": "an object placement; what it shows is text only through its caption",
}

#: Kinds that AGGREGATE their descendants' text into their own chunk. A cell
#: inside a table is already in the table's chunk; chunking it again would return
#: the same words twice under two ids and call that two pieces of evidence.
AGGREGATING_KINDS = ("table",)

#: Kinds that open a section and therefore name a roll-up.
SECTION_KINDS = ("section", "heading")

#: Default limits. Declared here rather than passed everywhere, so that a caller
#: reading the defaults sees the same numbers the tests assert.
UNIT_MAX_CHARS = 1200
WINDOW_FALLBACK_CHARS = 4000
WINDOW_OVERLAP_CHARS = 200


@dataclass
class ChunkPlan:
    """What the chunker produced, and what it refused.

    `refusals` is not an error list. A node with no text is not a failure — it is
    a container, or an empty cell — and the count is what distinguishes "nothing
    to chunk" from "the chunker did nothing".
    """

    chunks: list[ChunkRecord] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        by_level: dict[str, int] = {}
        for chunk in self.chunks:
            by_level[f"level_{chunk.level}"] = by_level.get(f"level_{chunk.level}", 0) + 1
        return {"chunks": len(self.chunks), "refused": len(self.refusals), **by_level}


def _text_of(node: Node) -> str:
    """The text a node contributes, or "" — a table contributes its cells' text."""
    if node.text:
        return node.text.strip()
    if node.kind == "table":
        parts = [c.text.strip() for c in node.walk() if c is not node and c.text]
        return " | ".join(p for p in parts if p)
    return ""


def _windows(text: str, size: int, overlap: int) -> Iterator[str]:
    """The declared fallback: only reached by a single unit too large to keep whole."""
    step = max(size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece:
            yield piece
        if start + size >= len(text):
            return


def chunk_tree(
    tree: Tree,
    doc_id: str,
    unit_max_chars: int = UNIT_MAX_CHARS,
    rollup_levels: int = 1,
    window_fallback_chars: int = WINDOW_FALLBACK_CHARS,
) -> ChunkPlan:
    """Cut `tree` into chunks along its own structure.

    Returns the plan rather than writing: what to store and what was refused are
    two answers, and a function that stored as it went could only report the first.
    """
    plan = ChunkPlan()
    addresses = node_ids_of(tree)
    by_address = {}
    for address in addresses:
        node = tree.root
        if address:
            for step in address.split("."):
                node = node.children[int(step)]
        by_address[address] = node

    # ---- level 0: one chunk per unit, in document order
    section_path: list[str] = []
    section_of: dict[str, list[str]] = {}   # section key -> node addresses
    section_titles: dict[str, list[str]] = {}
    current_section = ""
    seq = 0

    absorbed: set[str] = set()

    for address in addresses:
        node = by_address[address]

        if address in absorbed:
            plan.refusals.append({"reason": "absorbed-by-an-aggregating-node",
                                  "node_id": address, "kind": node.kind})
            continue

        if node.kind in SECTION_KINDS:
            title = _text_of(node) or node.facts.get("title") or ""
            depth = node.level if node.level is not None else len(section_path)
            section_path = section_path[:max(depth - 1, 0)] + ([title] if title else [])
            current_section = address
            section_titles[current_section] = list(section_path)
            section_of.setdefault(current_section, [])
            if node.kind != "heading":
                # A section is the context of a chunk, never a chunk. It is still
                # counted: leaving the loop without a reason is the silent drop
                # this package refuses everywhere else, and it was one here until
                # the accounting test asked where node "0" had gone.
                plan.refusals.append({"reason": "section-is-context",
                                      "node_id": address, "kind": node.kind})
                continue

        if node.kind not in CHUNKABLE_KINDS:
            plan.refusals.append({"reason": "not-a-chunkable-kind",
                                  "node_id": address, "kind": node.kind})
            continue

        text = _text_of(node)
        if not text:
            plan.refusals.append({"reason": "no-text", "node_id": address, "kind": node.kind})
            continue

        if node.kind in AGGREGATING_KINDS:
            # Its descendants' text is in this chunk, so they are not chunks.
            prefix = f"{address}." if address else ""
            absorbed |= {a for a in addresses if a.startswith(prefix) and a != address}

        pieces = [text]
        oversize = len(text) > unit_max_chars
        windowed = len(text) > window_fallback_chars
        if windowed:
            pieces = list(_windows(text, window_fallback_chars, WINDOW_OVERLAP_CHARS))

        for piece in pieces:
            meta: dict[str, Any] = {}
            if oversize:
                meta["oversize"] = True
            if windowed:
                meta["fallback"] = "window"
            chunk = ChunkRecord(
                chunk_id=chunk_id_for(doc_id=doc_id, level=0, node_ids=[address], text=piece),
                doc_id=doc_id, seq=seq, text=piece, level=0, node_ids=[address],
                section_path=list(section_titles.get(current_section, [])),
                pages=_pages_of(node), meta=meta,
            )
            plan.chunks.append(chunk)
            section_of.setdefault(current_section, []).append(address)
            seq += 1

    if rollup_levels < 1:
        return plan

    # ---- level 1: one roll-up per section, covering exactly its children
    for section_key, member_addresses in section_of.items():
        if not member_addresses:
            continue
        members = [c for c in plan.chunks if c.level == 0 and c.node_ids[0] in member_addresses]
        if not members:
            continue
        node_ids = []
        for chunk in members:
            for nid in chunk.node_ids:
                if nid not in node_ids:
                    node_ids.append(nid)
        text = "\n".join(c.text for c in members)
        rollup = ChunkRecord(
            chunk_id=chunk_id_for(doc_id=doc_id, level=1, node_ids=node_ids, text=text),
            doc_id=doc_id, seq=seq, text=text, level=1, node_ids=node_ids,
            parent_id=None, section_path=list(section_titles.get(section_key, [])),
            pages=sorted({p for c in members for p in c.pages}),
        )
        plan.chunks.append(rollup)
        seq += 1
        for chunk in members:
            chunk.parent_id = rollup.chunk_id

    return plan


def _pages_of(node: Node) -> list[int]:
    """Page numbers a node's citation mentions, if its format has pages at all."""
    pages = []
    for locator in node.provenance.chain:
        page = getattr(locator, "page", None)
        if isinstance(page, int) and page not in pages:
            pages.append(page)
    return pages
