"""The re-specified retrieval lane, over a real saqqara store.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The lane wraps the kernel's two lanes and adds what the kernel declines to own.
These tests exercise it end to end rather than checking that names line up: the
three mappings it rests on are meaning-not-shape, and a name-for-name rewrite
would import, run, and be wrong.

The first test is the one that matters most. `doc_types` must filter on the
ROUTING class, and the fixture is built so that filtering on the kernel's
`doc_class` instead would give the wrong answer rather than an error.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from ragix_kernels.saqqara.store.records import (          # noqa: E402
    ChunkRecord, DocumentRecord, EmbeddingRecord, chunk_id_for, doc_id_for)
from ragix_kernels.saqqara.store.sqlite import SqliteDocumentStore  # noqa: E402

from ragix_kernels.tender.domain.records import DocumentFacts                    # noqa: E402
from ragix_kernels.tender.domain.retrieval import (                              # noqa: E402
    Filters, RetrievalConfig, retrieve)


class _Embedder:
    """Deterministic vectors, so a ranking is a fact and not a coincidence."""

    def __init__(self, table: dict[str, tuple[float, ...]]) -> None:
        self.table = table

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self.table.get(t, (0.0, 0.0)) for t in texts]


def _document(store, name, routing_class, *, date=None, quality=None,
              texts=(), vectors=None, authority=None):
    # a real digest: the store refuses an id that is not a sha256 of the bytes,
    # and a fixture that fakes one is testing a different store
    sha = hashlib.sha256(name.encode()).hexdigest()
    facts = DocumentFacts(routing_class=routing_class, document_date=date,
                          quality=quality or {})
    meta = facts.into({"authority": authority} if authority is not None else {})
    doc = DocumentRecord(doc_id=doc_id_for(sha), corpus="default",
                         # the FILE FORMAT — deliberately the same on every
                         # document here, so a filter reading this field cannot
                         # tell them apart
                         doc_class="pdf",
                         source_path=f"{name}.pdf", source_sha256=sha,
                         kernel="saqqara", kernel_version="1.0", meta=meta)
    store.upsert_document(doc)
    chunks = []
    for i, text in enumerate(texts):
        node = f"{name}-n{i}"
        chunks.append(ChunkRecord(
            chunk_id=chunk_id_for(doc.doc_id, 0, [node], text), doc_id=doc.doc_id,
            seq=i, text=text, level=0, node_ids=[node]))
    store.replace_chunks(doc.doc_id, chunks)
    if vectors:
        store.upsert_embeddings([
            EmbeddingRecord(chunk_id=c.chunk_id, model="probe", dimensions=2,
                            vector=v)
            for c, v in zip(chunks, vectors)])
    return doc, chunks


@pytest.fixture
def store(tmp_path):
    with SqliteDocumentStore(path=str(tmp_path / "t.db")) as s:
        yield s


def test_doc_types_filters_on_the_routing_class_not_the_file_format(store):
    """The trap, made into a test.

    Both documents are `doc_class="pdf"` to the kernel and differ only in their
    routing class. A filter reading the kernel's field would return both — no
    error, no empty result, just the wrong set.
    """
    _document(store, "a", "P1", texts=["les penalites de retard du marche"])
    _document(store, "b", "P2", texts=["les penalites applicables au titulaire"])

    result = retrieve(store, None, "penalites",
                      filters=Filters(doc_types=["P2"]))

    paths = {store.get_document(rc.chunk.doc_id).source_path
             for rc in result.selected}
    assert paths == {"b.pdf"}, "the P1 document must not survive a P2 filter"


def test_a_document_carrying_no_facts_is_excluded_rather_than_assumed(store):
    """Fail closed: a policy that guessed a class would filter on its guess."""
    doc, _ = _document(store, "c", "P2", texts=["penalites contractuelles"])
    store.upsert_document(DocumentRecord(
        doc_id=doc.doc_id, corpus="default", doc_class="pdf",
        source_path="c.pdf", source_sha256=doc.source_sha256,
        kernel="saqqara", kernel_version="1.0", meta={}))   # facts stripped

    result = retrieve(store, None, "penalites", filters=Filters(doc_types=["P2"]))
    assert result.selected == []


def test_a_dated_filter_excludes_an_undated_document(store):
    _document(store, "d", "P2", date=None, texts=["penalites sans date"])
    _document(store, "e", "P2", date="2026-03-01", texts=["penalites datees"])

    result = retrieve(store, None, "penalites",
                      filters=Filters(date_from="2026-01-01"))
    paths = {store.get_document(rc.chunk.doc_id).source_path
             for rc in result.selected}
    assert paths == {"e.pdf"}


def test_without_an_embedder_the_dense_lane_is_declared_not_silent(store):
    """`provider: none` is a choice; a lane that vanished without saying so
    would leave the caller to infer it from a shorter result."""
    _document(store, "f", "P2", texts=["penalites de retard"])

    result = retrieve(store, None, "penalites")
    assert result.selected
    assert result.trace["dense_lane"] == "disabled (no embedder)"
    assert all(rc.dense_rank is None for rc in result.selected)


def test_both_lanes_contribute_and_their_ranks_stay_separate(store):
    _document(store, "g", "P2",
              texts=["penalites de retard applicables au titulaire",
                     "un paragraphe sans rapport avec la question"],
              vectors=[(1.0, 0.0), (0.0, 1.0)])

    embedder = _Embedder({"penalites": (1.0, 0.0)})
    result = retrieve(store, embedder, "penalites", model="probe")

    top = result.selected[0]
    assert top.dense_rank == 1 and top.lexical_rank == 1
    assert top.final_rank == 1
    # the trace exposes each lane's contribution rather than one blended number
    entry = result.trace["hits"][0]
    assert entry["dense_contrib"] > 0 and entry["lexical_contrib"] > 0
    assert entry["score"] >= entry["dense_contrib"] + entry["lexical_contrib"]


def test_freshness_prefers_the_newer_document_and_uses_no_wall_clock(store):
    _document(store, "h", "P2", date="2019-01-01", texts=["penalites anciennes"])
    _document(store, "i", "P2", date="2026-01-01", texts=["penalites recentes"])

    first = retrieve(store, None, "penalites")
    again = retrieve(store, None, "penalites")

    newest_path = store.get_document(first.selected[0].chunk.doc_id).source_path
    assert newest_path == "i.pdf"
    assert first.selected[0].boosts.get("freshness")
    # relative to the newest CANDIDATE: the same corpus ranks the same tomorrow
    assert [rc.chunk.chunk_id for rc in first.selected] == \
           [rc.chunk.chunk_id for rc in again.selected]


def test_a_duplicate_is_demoted_and_says_why(store):
    same = "penalites de retard applicables au titulaire"
    _document(store, "j", "P2", texts=[same])
    _document(store, "k", "P2", texts=[same.upper()])   # same text, normalised

    result = retrieve(store, None, "penalites")
    assert len(result.selected) == 2
    assert result.selected[-1].boosts.get("duplicate_penalty") is not None
    assert result.selected[0].boosts.get("duplicate_penalty") is None


def test_the_context_budget_bounds_what_is_selected(store):
    _document(store, "l", "P2",
              texts=["penalites " + "x" * 400, "penalites " + "y" * 400])

    cfg = RetrievalConfig(context_budget_chars=450)
    result = retrieve(store, None, "penalites", cfg=cfg)
    assert len(result.selected) == 1
    assert result.trace["context_chars_used"] <= 450
