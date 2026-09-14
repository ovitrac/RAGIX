"""tender.pipeline — the corpus lifecycle over the public kernel.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The digestion driver, re-specified. What it drives changed underneath it: the
kernel now owns the parts that used to be written here, and this module is what
is left once they are subtracted.

**Owned upstream, therefore absent here** — chunking and the content-derived
chunk id, object and edge extraction, embed-once (`embed_missing` computes only
the pairs a model has not seen), trash / restore / purge, and the store's own
refusals. `feed_tree` turns one analyzed tree into the records a store accepts;
this module never assembles a `DocumentRecord` by hand.

**Owned here, because the kernel offers none of it** — the corpus lifecycle
Olivier's contract of 2026-07-10 states:

    idempotent  ->  explicit update  ->  restorable delete  ->  autoupdate

  - an unchanged file is a no-op, and says so;
  - a **changed** file already in the store is reported `update_required` and is
    **never silently re-digested**: updates are an act, not a side effect;
  - `sync` re-digests same-name files whose content moved, indexes new ones, and
    **reports orphans without deleting them** (CLAUDE.md §12);
  - `rechunk` rebuilds the views from the trees ALREADY stored — no file is
    opened, no reader runs.

**`find_doc_by_source` is a name-trap, and it is on the parameter.** Both stores
carry a method of that name; the old one takes a **source name**, the kernel's
takes a **sha256**. Measured: `find_doc_by_source("note.md")` and
`find_doc_by_source(str(path))` both return None, only the sha resolves — and a
*changed* file's new sha resolves to nothing either. A migration that kept the
call shape would therefore find nothing, treat every file as new, and add a
second document on every edit: no error, no counted drop, a store that silently
grows. The lookup this lifecycle needs is by name, so it is written out here as
`_by_source_name` and reads `list_documents()` — which returns records, not ids.

**`document_date` is not written, and that is a declared gap.** No adapter reads
document core properties; the old pipeline took the date from the intake's
`modified`/`created`. Absence of a date is `undated` — a state, not a missing
value — and the retrieval lane already excludes undated documents from a dated
filter, fail-closed. The consequence to state plainly: **the freshness axis stays
inert** until the readers declare the date upstream. Filesystem mtime is not a
substitute for the same reason `digested_at` is not: it is a fact about the copy.

**The query path is not here.** `tender.retrieval` is the lane; a second
`search()` in this module would be a second place for one rule.

Ports are passed in rather than built from a config object: the lab has no
config re-specification, and inventing one to serve four call sites would be an
abstraction ahead of its use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .routing import classify

__all__ = [
    "DigestReport",
    "delete_document",
    "index_paths",
    "rechunk",
    "restore_document",
    "sync",
    "update_document",
]


@dataclass
class DigestReport:
    """What one document contributed, and what was decided about it.

    `routing_class`, never `doc_class`: the kernel's `doc_class` on the same
    record is the file format, and one name over two referents is the trap this
    lane exists on the far side of.
    """

    doc_id: str
    routing_class: str
    n_chunks: int = 0
    n_embedded: int = 0
    n_reused: int = 0
    notes: dict[str, Any] = field(default_factory=dict)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_source_name(store: Any, name: str) -> Optional[Any]:
    """The stored document whose source file has this NAME, or None.

    Not `store.find_doc_by_source`, which keys on the content sha and so cannot
    answer "is this file already here, under different content?" — the one
    question the lifecycle turns on. See the module docstring.
    """
    for document in store.list_documents():
        if Path(document.source_path).name == name:
            return document
    return None


def _resolve(store: Any, identifier: str) -> Optional[Any]:
    """A stored document by doc_id or by source name — in that order."""
    return store.get_document(identifier) or _by_source_name(store, identifier)


def _digest_one(path: Path, store: Any, embedder: Any, *, model: str,
                corpus: str, notes: Optional[dict] = None,
                **chunker_options: Any) -> DigestReport:
    """Read, classify, feed and embed one file. The single write path."""
    from ragix_kernels.saqqara.store.embed import embed_missing
    from ragix_kernels.saqqara.store.feed import feed_tree

    from .substrate import read_tree

    read = read_tree(path)
    routing = classify(read.tree)
    meta = routing.facts().into({})

    result = feed_tree(
        read.tree, str(path), _file_sha(path),
        doc_class=read.source_format, corpus=corpus,
        kernel_version=read.reader_version, meta=meta,
        digested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **chunker_options,
    )
    doc_id = result.document.doc_id
    store.upsert_document(result.document, result.objects, result.edges)
    store.replace_chunks(doc_id, result.chunks)
    plan = embed_missing(store, result.chunks, embedder, model)

    account = dict(notes or {})
    # What the builder refused to place, and what the chunker refused to chunk.
    # A drop nobody reads is a silent drop.
    if read.dropped:
        account["builder_drops"] = read.dropped
    refusals = result.plan.counts().get("refusals")
    if refusals:
        account["chunk_refusals"] = refusals
    if plan.disabled:
        account["embedding_disabled"] = True

    return DigestReport(
        doc_id=doc_id, routing_class=routing.routing_class,
        n_chunks=len(result.chunks), n_embedded=plan.embedded,
        n_reused=plan.skipped, notes=account,
    )


def index_paths(paths: Sequence[str | Path], store: Any, embedder: Any, *,
                model: str = "", corpus: str = "default",
                force: bool = False, **chunker_options: Any) -> list[DigestReport]:
    """Idempotent indexing.

    An unchanged file is a no-op that says `unchanged`. A file already stored
    whose content has moved is reported `update_required` and is **not**
    re-digested here — `update_document` is the act that replaces it.
    """
    reports: list[DigestReport] = []
    for raw in paths:
        path = Path(raw)
        existing = None if force else _by_source_name(store, path.name)
        if existing is not None:
            facts = _stored_class(existing)
            if existing.source_sha256 == _file_sha(path):
                reports.append(DigestReport(
                    doc_id=existing.doc_id, routing_class=facts,
                    n_chunks=len(store.get_chunks(existing.doc_id)),
                    notes={"unchanged": True}))
                continue
            reports.append(DigestReport(
                doc_id=existing.doc_id, routing_class=facts,
                n_chunks=len(store.get_chunks(existing.doc_id)),
                notes={"update_required": True,
                       "hint": "source changed — call update_document or sync"}))
            continue
        reports.append(_digest_one(path, store, embedder, model=model,
                                   corpus=corpus, **chunker_options))
    return reports


def update_document(path: str | Path, store: Any, embedder: Any, *,
                    model: str = "", corpus: str = "default",
                    **chunker_options: Any) -> DigestReport:
    """Re-digest a file and replace the document stored under its name.

    The doc_id is derived from the content, so a changed file lands under a new
    one; the previous record is then purged and the move **declared** rather than
    left as a second document that nothing points at.
    """
    path = Path(path)
    previous = _by_source_name(store, path.name)
    if previous is None:
        raise KeyError(f"no stored document with source {path.name!r} — index it first")

    report = _digest_one(path, store, embedder, model=model, corpus=corpus,
                         notes={"updated": True,
                                "previous_sha": previous.source_sha256},
                         **chunker_options)
    if report.doc_id != previous.doc_id:
        store.delete_document(previous.doc_id, purge=True)
        report.notes["replaces"] = previous.doc_id
    return report


def sync(paths: Sequence[str | Path], store: Any, embedder: Any, *,
         model: str = "", corpus: str = "default",
         **chunker_options: Any) -> dict[str, Any]:
    """Autoupdate a corpus against a list of files.

    New files are indexed; a stored document whose same-name file has changed is
    updated; a stored document whose source file is absent is **reported as an
    orphan and never deleted** — removal is a decision, and it is the operator's.
    """
    added: list[DigestReport] = []
    updated: list[DigestReport] = []
    unchanged: list[str] = []
    seen: set[str] = set()

    for raw in paths:
        path = Path(raw)
        seen.add(path.name)
        existing = _by_source_name(store, path.name)
        if existing is None:
            added.append(_digest_one(path, store, embedder, model=model,
                                     corpus=corpus, **chunker_options))
        elif existing.source_sha256 != _file_sha(path):
            updated.append(update_document(path, store, embedder, model=model,
                                           corpus=corpus, **chunker_options))
        else:
            unchanged.append(existing.doc_id)

    orphans = [d.doc_id for d in store.list_documents()
               if Path(d.source_path).name not in seen]
    return {"added": added, "updated": updated,
            "unchanged": unchanged, "orphans": orphans}


def rechunk(store: Any, embedder: Any, *, model: str = "",
            **chunker_options: Any) -> list[DigestReport]:
    """Rebuild the chunk and embedding views from the STORED trees.

    No file is opened and no reader runs: the tree in the store is the substrate,
    and a re-chunk that re-read the sources would be measuring the files rather
    than the stored corpus. The kernel returns the tree as a `Tree`, so there is
    no deserialisation step to get wrong.
    """
    from ragix_kernels.saqqara.store.chunker import chunk_tree
    from ragix_kernels.saqqara.store.embed import embed_missing

    reports = []
    for document in store.list_documents():
        plan = chunk_tree(document.tree, doc_id=document.doc_id, **chunker_options)
        store.replace_chunks(document.doc_id, plan.chunks)
        embedded = embed_missing(store, plan.chunks, embedder, model)
        reports.append(DigestReport(
            doc_id=document.doc_id, routing_class=_stored_class(document),
            n_chunks=len(plan.chunks), n_embedded=embedded.embedded,
            n_reused=embedded.skipped))
    return reports


def delete_document(identifier: str, store: Any, *, purge: bool = False) -> str:
    """Delete by doc_id or source name. Trash is the default; purge is opt-in.

    A trashed document keeps its chunks — which is what makes a restore free of
    re-embedding — and drops out of the store's searches while parked.
    """
    document = _resolve(store, identifier)
    if document is None:
        raise KeyError(f"no stored document matches {identifier!r}")
    store.delete_document(document.doc_id, purge=purge)
    return document.doc_id


def restore_document(identifier: str, store: Any) -> str:
    """Bring a parked document back. By doc_id or source name."""
    document = _resolve(store, identifier)
    if document is None:
        for parked in store.list_documents(include_trashed=True):
            if parked.doc_id == identifier or Path(parked.source_path).name == identifier:
                document = parked
                break
    if document is None:
        raise KeyError(f"no stored document matches {identifier!r}")
    store.restore_document(document.doc_id)
    return document.doc_id


def _stored_class(document: Any) -> str:
    """The routing class recorded on a stored document, or `?` if it carries none.

    `?` rather than a guess: a document indexed before this lane existed has no
    routing class, and inventing one would put a value into the field the
    retrieval policy filters on.
    """
    from .records import DocumentFacts

    facts = DocumentFacts.of(document)
    return facts.routing_class if facts is not None else "?"
