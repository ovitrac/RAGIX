"""
saqqara.store.refine — the vector a refusal cost, restored by splitting the text.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-06

Gate K7.22. Ruling D-0016.

A level-1 roll-up of 18 190 to 116 280 characters is refused by the embedder, or
is accepted and dilutes what it says into one vector. Either way the section has
no usable vector while its text is intact: the roll-up equals the concatenation of
its children, every child is embedded, and the roll-up is in the lexical index. **A
refusal costs a coarse vector, never text.**

The answer is a **pass**, not a one-off split. A pass takes the previous result and
a budget, splits every text that is over that budget or refused into parts, embeds
**only the parts**, and leaves its result to the next pass under a smaller budget.
A text that has been split has no vector of its own: its vectors are its leaves'.
A leaf still refused at the smallest budget stays a refusal record — an honest end,
never a silent one.

The unit of retrieval is never a window (see `retrieve.py`): a text is scored by
the maximum cosine over its leaves, and the trace says which leaf matched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .records import ChunkRecord, chunk_id_for

__all__ = ["RefinePass", "RefinePlan", "split_into_windows", "refine_store"]

#: How children are joined into a parent's text. The chunker joins a roll-up's
#: members with a newline; a window is a run of those members and must join them
#: the same way, or the union of the windows is not the parent's text.
JOIN = "\n"


@dataclass
class RefinePass:
    """What one pass did, in the terms the report needs."""

    number: int
    budget: int
    read: str = ""            # digest of the result this pass read
    split: int = 0            # texts split
    parts: int = 0            # parts made
    embedded: int = 0
    refused: int = 0
    dropped_vectors: int = 0  # vectors removed from texts that were split
    documents_attempted: int = 0
    documents_completed: int = 0
    stopped_on: Optional[str] = None   # the document a failure stopped on, if any

    def to_dict(self) -> dict[str, Any]:
        return {"budget": self.budget, "documents_attempted": self.documents_attempted,
                "documents_completed": self.documents_completed,
                "dropped_vectors": self.dropped_vectors, "embedded": self.embedded,
                "number": self.number, "parts": self.parts, "read": self.read,
                "refused": self.refused, "split": self.split,
                "stopped_on": self.stopped_on}


@dataclass
class RefinePlan:
    """Every pass that ran, in order, and what remains."""

    passes: list[RefinePass] = field(default_factory=list)
    remaining_refusals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passes": [p.to_dict() for p in self.passes],
                "parts": sum(p.parts for p in self.passes),
                "remaining_refusals": self.remaining_refusals}


def split_into_windows(
    parent: ChunkRecord,
    children: list[ChunkRecord],
    budget: int,
    *,
    pass_number: int,
    overlap_children: int = 0,
) -> list[ChunkRecord]:
    """Contiguous runs of `children` whose joined text fits `budget`.

    The runs partition the children — no gap, and no overlap unless declared. A
    child that alone exceeds the budget is its own window and is **not cut**: a
    unit is the smallest thing this package will call a piece of evidence, and
    cutting one to fit a number would produce a citation to half a sentence. Such a
    window may be refused by the embedder, and that refusal is recorded like any
    other.

    Every part carries, in `meta["part"]`, what it takes to check it after the
    fact: the pass and budget that made it, its index, its character span within
    the parent, and the parent's length at the time of the split.
    """
    if budget < 1:
        raise ValueError(f"a window budget is a character count, at least 1: got {budget}")
    if overlap_children < 0:
        raise ValueError(f"overlap is a number of children, never negative: got {overlap_children}")
    if not children:
        return []

    # Character offset of each child within the parent's text, under the same join.
    starts: list[int] = []
    cursor = 0
    for child in children:
        starts.append(cursor)
        cursor += len(child.text) + len(JOIN)
    parent_chars = len(parent.text)

    runs: list[list[int]] = []
    index = 0
    while index < len(children):
        run = [index]
        length = len(children[index].text)
        nxt = index + 1
        while nxt < len(children):
            grown = length + len(JOIN) + len(children[nxt].text)
            if grown > budget:
                break
            run.append(nxt)
            length = grown
            nxt += 1
        runs.append(run)
        if nxt >= len(children):
            break
        # Overlap is declared in children, and never so large that a run repeats
        # itself: a step of zero would loop for ever on the same window.
        index = max(nxt - overlap_children, run[0] + 1)

    windows: list[ChunkRecord] = []
    for position, run in enumerate(runs):
        members = [children[i] for i in run]
        node_ids: list[str] = []
        for member in members:
            for nid in member.node_ids:
                if nid not in node_ids:
                    node_ids.append(nid)
        text = JOIN.join(m.text for m in members)
        span = [starts[run[0]], starts[run[-1]] + len(members[-1].text)]
        windows.append(ChunkRecord(
            chunk_id=chunk_id_for(doc_id=parent.doc_id, level=parent.level,
                                  node_ids=node_ids, text=text),
            doc_id=parent.doc_id,
            seq=parent.seq,
            text=text,
            level=parent.level,
            node_ids=node_ids,
            parent_id=parent.chunk_id,
            section_path=list(parent.section_path),
            pages=sorted({p for m in members for p in m.pages}),
            lang=parent.lang,
            meta={**dict(parent.meta), "part": {
                "kind": "window", "pass": pass_number, "budget": budget,
                "index": position, "span": span, "parent_chars": parent_chars,
            }},
        ))
    return windows


def _children_of(chunks: list[ChunkRecord], parent: ChunkRecord) -> list[ChunkRecord]:
    """The chunks a text was built from, in order."""
    return sorted((c for c in chunks if c.parent_id == parent.chunk_id),
                  key=lambda c: (c.seq, c.node_ids))


def _leaves(chunks: list[ChunkRecord]) -> set[str]:
    """Ids that are nobody's parent: the texts that own vectors."""
    parents = {c.parent_id for c in chunks if c.parent_id}
    return {c.chunk_id for c in chunks if c.chunk_id not in parents}


@dataclass
class _Outcome:
    """What refining one document did, so the pass can add it up."""

    split: int = 0
    embedded: int = 0
    refused: int = 0
    vectors: int = 0


def _documents_with_candidates(store: Any, budget: int, doc_id: Optional[str]) -> list[str]:
    """The documents holding a text over budget or refused, in a stable order."""
    refused = {r.chunk_id for r in store.get_embedding_refusals(doc_id)}
    chunks = [c for c in store.get_chunks(doc_id) if c.level >= 1]
    leaves = _leaves(chunks)
    return sorted({c.doc_id for c in chunks
                   if c.chunk_id in leaves
                   and (len(c.text) > budget or c.chunk_id in refused)})


def _refine_one(store: Any, embedder: Any, model: str, document: str, budget: int,
                pass_number: int, overlap_children: int) -> tuple[list[ChunkRecord], _Outcome]:
    """Split, store and embed one document's oversized texts, in that order.

    **The order is the atomicity.** The first version of this split every document,
    dropped every split text's vector, and only then embedded — one call carrying
    the parts of 145 documents, which `embed_missing` refuses because its refusal
    register is per document. It raised **after** the drops: 215 texts and 1 027
    parts, none with a vector, and 165 vectors the lane had were gone. The kernel
    failed correctly and the store was left worse than it started.

    So: one document at a time, and the vectors of the texts that were split are
    dropped **only after** their parts are embedded. A failure anywhere leaves that
    document exactly as it was — its texts with the vectors they had, its parts
    without any — and the pass says where it stopped.
    """
    from .embed import embed_missing

    outcome = _Outcome()
    chunks = [c for c in store.get_chunks(document) if c.level >= 1]
    units = [c for c in store.get_chunks(document) if c.level == 0]
    refused = {r.chunk_id for r in store.get_embedding_refusals(document)}
    leaves = _leaves(chunks)

    made: list[ChunkRecord] = []
    split_parents: list[str] = []
    for parent in sorted((c for c in chunks if c.chunk_id in leaves
                          and (len(c.text) > budget or c.chunk_id in refused)),
                         key=lambda c: (c.seq, c.chunk_id)):
        children = _children_of(chunks, parent) or _children_of(units, parent)
        if len(children) < 2:
            # A text of one unit is already the smallest thing this package cites;
            # it keeps its refusal rather than being cut.
            continue
        windows = split_into_windows(parent, children, budget,
                                     pass_number=pass_number,
                                     overlap_children=overlap_children)
        if len(windows) < 2:
            continue
        made.extend(windows)
        split_parents.append(parent.chunk_id)
        outcome.split += 1

    if not made:
        return [], outcome

    store.add_chunks(made)
    embedded = embed_missing(store, made, embedder, model=model, replace_refusals=False)
    outcome.embedded = embedded.embedded
    outcome.refused = embedded.refused
    # Only now, with the parts embedded and stored, does the coarse vector go.
    outcome.vectors = store.delete_embeddings(split_parents, model)
    return made, outcome


def refine_store(
    store: Any,
    embedder: Any,
    model: str,
    budgets: Iterable[int],
    *,
    overlap_children: int = 0,
    doc_id: Optional[str] = None,
) -> RefinePlan:
    """One pass per declared budget: split what is over budget or refused, embed the parts.

    Each pass reads **the previous result** — the store as the last pass left it,
    identified by a digest so the record says what was read — and stops when a pass
    finds nothing to split. Splitting a text removes its own vector: after a split
    the text is represented by its parts, and keeping the coarse vector beside them
    would let one text answer twice with two different meanings.
    """
    from .embed import embed_missing

    plan = RefinePlan()
    if embedder is None:
        return plan

    for number, budget in enumerate(budgets, 1):
        this = RefinePass(number=number, budget=budget, read=store.result_digest(doc_id))
        documents = _documents_with_candidates(store, budget, doc_id)
        this.documents_attempted = len(documents)

        for document in documents:
            try:
                made, dropped = _refine_one(store, embedder, model, document, budget,
                                            number, overlap_children)
            except Exception as exc:
                # The store is consistent at this point by construction: this
                # document's texts still hold the vectors they had, and its parts,
                # if any were written, simply have none. What is lost is the work,
                # never the lane. The message says where the pass stopped, because
                # a run that refined 87 documents of 145 and says only "failed"
                # leaves the next person to find that out by reading the store.
                this.stopped_on = document
                plan.passes.append(this)
                raise RuntimeError(
                    f"refinement stopped on document {document[:12]} in pass "
                    f"{number} (budget {budget}): {this.documents_completed} of "
                    f"{this.documents_attempted} document(s) were completed, and "
                    f"every text still holds the vectors it had. Cause: {exc}"
                ) from exc
            this.split += dropped.split
            this.parts += len(made)
            this.embedded += dropped.embedded
            this.refused += dropped.refused
            this.dropped_vectors += dropped.vectors
            this.documents_completed += 1

        if not this.parts:
            plan.passes.append(this)
            break

        plan.passes.append(this)

    plan.remaining_refusals = [r.to_dict() for r in store.get_embedding_refusals(doc_id)]
    return plan
