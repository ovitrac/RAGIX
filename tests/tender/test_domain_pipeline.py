"""The corpus lifecycle over the public kernel — idempotent, explicit, restorable.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The four properties of Olivier's digester contract (2026-07-10), asserted against
generated fixtures rather than a corpus: a lifecycle property is about the
lifecycle, and a test that skips when the corpus is absent proves nothing on the
day it matters.

Two of these tests exist because of what was *measured* about the kernel's store
rather than what was assumed:

  - `find_doc_by_source` keys on the content sha, so it cannot answer the
    question this lifecycle turns on ("is this file here, under other
    content?"). `test_the_kernels_lookup_cannot_answer_the_lifecycles_question`
    pins that, so a later simplification back to the kernel call fails here
    rather than in a store that silently grows;
  - a trashed document keeps its chunks and drops out of the store's searches,
    which is what makes a restore free of re-embedding. Asserted, not assumed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from ragix_kernels.saqqara.store.sqlite import SqliteDocumentStore   # noqa: E402

from ragix_kernels.tender.domain import pipeline                                          # noqa: E402
from ragix_kernels.tender.domain.records import DocumentFacts                             # noqa: E402


class CountingEmbedder:
    """A deterministic embedder that counts what it was actually asked to embed.

    The counter is the assertion in two tests: "only the delta" and "a restore
    re-embeds nothing" are both claims about how many texts reached this method.
    """

    model = "counting"

    def __init__(self) -> None:
        self.embedded = 0

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.embedded += len(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode()).digest()
        return tuple(byte / 255.0 for byte in digest[:8])


@pytest.fixture
def env(tmp_path):
    store = SqliteDocumentStore(path=str(tmp_path / "life.db"))
    work = tmp_path / "docs"
    work.mkdir()
    handbook = work / "handbook.md"
    handbook.write_text(
        "# Manuel\n\nUn paragraphe sur la recherche rapide dans les tableaux.\n",
        encoding="utf-8")
    return store, CountingEmbedder(), work


def _index(env, *paths, **kw):
    store, embedder, _ = env
    return pipeline.index_paths(list(paths), store, embedder,
                                model=embedder.model, **kw)


# ── idempotence ──────────────────────────────────────────────────────────────

def test_reindexing_an_unchanged_file_is_a_no_op_that_says_so(env):
    store, embedder, work = env
    first = _index(env, work / "handbook.md")[0]
    assert first.n_embedded > 0
    assert first.routing_class == "M1"

    after = embedder.embedded
    second = _index(env, work / "handbook.md")[0]

    assert second.notes.get("unchanged") is True
    assert second.doc_id == first.doc_id
    assert embedder.embedded == after, "an unchanged file must not reach the embedder"


# ── a changed file is an act, not a side effect ──────────────────────────────

def test_a_changed_file_is_reported_and_not_silently_re_digested(env):
    """The second assertion is the one that matters: the store must still hold
    the OLD content. Reporting `update_required` while quietly re-digesting
    would satisfy a test that only read the note."""
    store, embedder, work = env
    path = work / "handbook.md"
    first = _index(env, path)[0]
    path.write_text(path.read_text(encoding="utf-8") + "\n\n# Annexe\n\nSujet nouveau.\n",
                    encoding="utf-8")

    report = _index(env, path)[0]

    assert report.notes.get("update_required") is True
    stored = " ".join(chunk.text for chunk in store.get_chunks(first.doc_id))
    assert "Sujet nouveau" not in stored, "the store was re-digested behind the report"


def test_update_document_replaces_the_record_and_declares_the_move(env):
    store, embedder, work = env
    path = work / "handbook.md"
    first = _index(env, path)[0]
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text(encoding="utf-8") + "\n\n# Annexe\n\nSujet nouveau.\n",
                    encoding="utf-8")

    before = embedder.embedded
    report = pipeline.update_document(path, store, embedder, model=embedder.model)

    assert report.notes["updated"] is True
    assert report.notes["previous_sha"] == original_sha, (
        "the note must name the content that was replaced, not any sha")
    assert report.doc_id != first.doc_id, "the doc id follows the content"
    assert report.notes["replaces"] == first.doc_id
    assert store.get_document(first.doc_id) is None, "the superseded record stays"

    stored = " ".join(chunk.text for chunk in store.get_chunks(report.doc_id))
    assert "Sujet nouveau" in stored
    assert embedder.embedded > before, "the new content was never embedded"


def test_an_update_before_an_index_raises_rather_than_indexing(env):
    """Fail closed: `update` names an existing document, and a missing one is
    not an invitation to create it."""
    store, embedder, work = env
    with pytest.raises(KeyError, match="handbook.md"):
        pipeline.update_document(work / "handbook.md", store, embedder,
                                 model=embedder.model)


# ── trash, restore, purge ────────────────────────────────────────────────────

def test_delete_parks_the_document_and_a_restore_re_embeds_nothing(env):
    store, embedder, work = env
    report = _index(env, work / "handbook.md")[0]
    assert store.lexical_search("tableaux", top_k=5)

    parked = pipeline.delete_document("handbook.md", store)
    assert parked == report.doc_id
    listed = [d.doc_id for d in store.list_documents()]
    trashed = [d.doc_id for d in store.list_documents(include_trashed=True)]
    assert parked not in listed and parked in trashed
    assert not store.lexical_search("tableaux", top_k=5), "a parked document stays out of search"

    before = embedder.embedded
    pipeline.restore_document(parked, store)

    assert parked in [d.doc_id for d in store.list_documents()]
    assert store.lexical_search("tableaux", top_k=5)
    assert embedder.embedded == before, "a restore must not re-embed"


def test_purge_is_the_opt_in_and_it_is_irreversible(env):
    store, embedder, work = env
    report = _index(env, work / "handbook.md")[0]

    pipeline.delete_document(report.doc_id, store, purge=True)

    assert store.get_document(report.doc_id) is None
    assert report.doc_id not in [d.doc_id for d in store.list_documents(include_trashed=True)]


def test_deleting_something_absent_raises(env):
    store, _embedder, _work = env
    with pytest.raises(KeyError, match="ghost.md"):
        pipeline.delete_document("ghost.md", store)


# ── sync ─────────────────────────────────────────────────────────────────────

def test_sync_adds_updates_and_reports_orphans_without_deleting_them(env):
    store, embedder, work = env
    first, second = work / "handbook.md", work / "second.md"
    second.write_text("# Second\n\nContenu independant sur les organigrammes.\n",
                      encoding="utf-8")
    _index(env, first)
    first.write_text(first.read_text(encoding="utf-8") + "\n\n# Journal\n\nMis a jour.\n",
                     encoding="utf-8")

    report = pipeline.sync([first, second], store, embedder, model=embedder.model)

    assert len(report["added"]) == 1
    assert len(report["updated"]) == 1
    assert report["updated"][0].notes["updated"] is True
    assert report["orphans"] == []

    again = pipeline.sync([first], store, embedder, model=embedder.model)

    assert len(again["unchanged"]) == 1
    assert len(again["orphans"]) == 1
    orphan = again["orphans"][0]
    assert store.get_document(orphan) is not None, "an orphan is reported, never deleted"


# ── rechunk ──────────────────────────────────────────────────────────────────

def test_rechunk_rebuilds_from_the_stored_tree_with_the_source_gone(env):
    """The invariant is 'no file access'. Deleting the source before the call is
    what makes the test able to observe it: a rechunk that read the file would
    raise rather than quietly agree."""
    store, embedder, work = env
    path = work / "handbook.md"
    indexed = _index(env, path)[0]
    path.unlink()

    reports = pipeline.rechunk(store, embedder, model=embedder.model)

    assert [r.doc_id for r in reports] == [indexed.doc_id]
    assert reports[0].n_chunks == indexed.n_chunks
    assert reports[0].n_embedded == 0, "the same chunks were already embedded"
    assert reports[0].routing_class == "M1"


# ── the join with the retrieval lane ─────────────────────────────────────────

def test_the_document_it_writes_carries_the_class_the_retrieval_lane_filters_on(env):
    """End to end across the two lanes of (f): nothing else writes this field,
    and the filter reading it is in another module."""
    from ragix_kernels.tender.domain.retrieval import Filters, retrieve

    store, embedder, work = env
    docx = pytest.importorskip("docx", reason="python-docx")
    styled = work / "styled.docx"
    document = docx.Document()
    for index in range(3):
        document.add_paragraph(f"Titre {index + 1}", style="Heading 1")
    document.add_paragraph("Un paragraphe sur les tableaux et la recherche rapide.")
    document.save(styled)

    reports = pipeline.index_paths([work / "handbook.md", styled], store, embedder,
                                   model=embedder.model)
    classes = {r.routing_class for r in reports}
    assert classes == {"M1", "D1"}

    stored = {DocumentFacts.of(d).routing_class for d in store.list_documents()}
    assert stored == {"M1", "D1"}

    result = retrieve(store, embedder, "tableaux", model=embedder.model,
                      filters=Filters(doc_types=["D1"]))
    kept = {hit.chunk.doc_id for hit in result.selected}
    assert kept, "the filtered lane returned nothing at all — nothing is proved"
    assert kept == {r.doc_id for r in reports if r.routing_class == "D1"}


# ── the trap that shaped `_by_source_name` ───────────────────────────────────

def test_the_kernels_lookup_cannot_answer_the_lifecycles_question(env):
    """`find_doc_by_source` keys on the content sha. Both stores carry the name;
    only one takes a name. This is why `_by_source_name` exists — and why
    replacing it with the kernel call would make every file look new."""
    store, embedder, work = env
    path = work / "handbook.md"
    indexed = _index(env, path)[0]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    assert store.find_doc_by_source(path.name) is None
    assert store.find_doc_by_source(str(path)) is None
    assert store.find_doc_by_source(sha).doc_id == indexed.doc_id

    path.write_text(path.read_text(encoding="utf-8") + "\n\nAutre chose.\n",
                    encoding="utf-8")
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.find_doc_by_source(changed) is None
    assert pipeline._by_source_name(store, path.name).doc_id == indexed.doc_id
