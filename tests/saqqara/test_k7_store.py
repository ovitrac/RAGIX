"""
Gate K7 — the document store: one SQLite file, embeddings beside the chunks.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

The store keeps what the kernel read, so that a citation survives the process that
produced it. Identity is the rule that makes that possible: a document is its
bytes, a chunk is its content, and neither is a path or a position in a run.

This file grows with the store; it opens on the records, which every later
proposition depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.model import CANONICAL_JSON, Tree  # noqa: E402
from ragix_kernels.saqqara.store.records import (  # noqa: E402
    ChunkRecord,
    DocumentRecord,
    EdgeRecord,
    EmbeddingRecord,
    Hit,
    ObjectRecord,
    chunk_id_for,
    doc_id_for,
    node_at,
    node_ids_of,
)


@pytest.fixture(scope="module")
def tree() -> Tree:
    """A spreadsheet tree — the deepest of the generated shapes, so addressing
    is exercised on nesting rather than on a flat list of paragraphs."""
    return G.build_trees()["xlsx"]


@pytest.fixture(scope="module")
def trees() -> dict[str, Tree]:
    return G.build_trees()


# --------------------------------------------------- K7.1 a document is its bytes

def test_k7_1_doc_id_is_the_source_sha256():
    """Identity is the bytes, so the same document at two paths is one document.

    Falsified by: a doc_id that changes when only the path changes.
    """
    sha = "a" * 64
    here = DocumentRecord(doc_id=doc_id_for(sha), corpus="default", doc_class="x",
                          source_path="/one/m.xlsx", source_sha256=sha,
                          kernel="saqqara", kernel_version="1.0")
    there = DocumentRecord(doc_id=doc_id_for(sha), corpus="default", doc_class="x",
                           source_path="/another/place/m.xlsx", source_sha256=sha,
                           kernel="saqqara", kernel_version="1.0")
    assert here.doc_id == there.doc_id == sha


def test_k7_1_doc_id_refuses_anything_that_is_not_a_digest():
    """A digest is checked, not trusted: an id built from a path is the defect."""
    for bad in ("", "/some/path.xlsx", "a" * 63, "z" * 64, "A" * 64):
        with pytest.raises(ValueError):
            doc_id_for(bad)


# ------------------------------------------------ K7.2 the tree survives the store

@pytest.mark.parametrize("fmt", ["xlsx", "docx", "pdf", "pptx", "md"])
def test_k7_2_tree_json_round_trips(trees, fmt):
    """A stored tree returns as the same tree, or the store lost the document.

    Every format, because a round trip that holds for one shape and not another
    is a claim about that shape. Falsified by: a round trip that changes the
    canonical JSON.
    """
    tree = trees[fmt]
    record = DocumentRecord(
        doc_id=doc_id_for("b" * 64), corpus="default", doc_class=fmt,
        source_path=f"/m.{fmt}", source_sha256="b" * 64,
        kernel="saqqara", kernel_version="1.0", tree=tree,
    )
    back = DocumentRecord.from_dict(record.to_dict())
    assert back.tree is not None
    assert json.dumps(back.tree.to_dict(), **CANONICAL_JSON) == json.dumps(
        tree.to_dict(), **CANONICAL_JSON
    )


def test_k7_2_every_record_serialises_with_sorted_keys(tree):
    """Records are compared and hashed as text; key order is part of that."""
    records = [
        DocumentRecord(doc_id="c" * 64, corpus="default", doc_class="x",
                       source_path="/p", source_sha256="c" * 64,
                       kernel="saqqara", kernel_version="1.0"),
        ObjectRecord(doc_id="c" * 64, node_id="0.1", kind="figure"),
        EdgeRecord(doc_id="c" * 64, src="0.1", dst="0.2", type="binds"),
        ChunkRecord(chunk_id="d" * 64, doc_id="c" * 64, seq=0, text="t",
                    level=0, node_ids=["0.1"]),
        EmbeddingRecord(chunk_id="d" * 64, model="m", dimensions=2,
                        vector=(0.5, 0.25)),
    ]
    for record in records:
        keys = list(record.to_dict())
        assert keys == sorted(keys), f"{type(record).__name__} keys unsorted"
        assert type(record).from_dict(record.to_dict()).to_dict() == record.to_dict()


def test_k7_2_an_edge_type_outside_the_vocabulary_is_refused():
    """The edge kinds are a closed vocabulary; an unknown one is not stored."""
    for kind in ("binds", "refers_to", "continues", "supersedes"):
        EdgeRecord(doc_id="c" * 64, src="0.1", dst="0.2", type=kind)
    with pytest.raises(ValueError):
        EdgeRecord(doc_id="c" * 64, src="0.1", dst="0.2", type="relates_to")


# ------------------------------------------- K7.5 a chunk is its content

def test_k7_5_chunk_id_is_content_derived(tree):
    """The same content yields the same id, so re-chunking writes nothing.

    Falsified by: an id that depends on when or in what order it was computed.
    """
    first = chunk_id_for(doc_id="e" * 64, level=0, node_ids=["0.1", "0.2"], text="hello")
    again = chunk_id_for(doc_id="e" * 64, level=0, node_ids=["0.1", "0.2"], text="hello")
    assert first == again and len(first) == 64


def test_k7_5_chunk_id_changes_with_every_part_of_its_content():
    """Each component is load-bearing; a component nobody varies is decoration."""
    base = dict(doc_id="e" * 64, level=0, node_ids=["0.1"], text="hello")
    ids = {
        chunk_id_for(**base),
        chunk_id_for(**{**base, "doc_id": "f" * 64}),
        chunk_id_for(**{**base, "level": 1}),
        chunk_id_for(**{**base, "node_ids": ["0.2"]}),
        chunk_id_for(**{**base, "text": "goodbye"}),
    }
    assert len(ids) == 5, "two different chunks share an id"


def test_k7_5_node_ids_order_is_part_of_the_content():
    """A chunk covering the same nodes in another order is another chunk."""
    a = chunk_id_for(doc_id="e" * 64, level=0, node_ids=["0.1", "0.2"], text="t")
    b = chunk_id_for(doc_id="e" * 64, level=0, node_ids=["0.2", "0.1"], text="t")
    assert a != b


def test_k7_5_a_chunk_without_a_node_is_refused():
    """A chunk that cites nothing cannot be traced back, so it is not a chunk."""
    with pytest.raises(ValueError):
        ChunkRecord(chunk_id="d" * 64, doc_id="c" * 64, seq=0, text="t",
                    level=0, node_ids=[])


# ------------------------------------------- node addressing, which K7.3/K7.10 need

def test_k7_2_node_ids_address_every_node_and_resolve_back(tree):
    """The model carries no node id, so the store derives one from the structure.

    A node's address is its child-index path from the root ("", "0", "0.3.2").
    It needs no change to the frozen model, it is stable because the tree's own
    order is stable (K4.1), and it resolves by walking. Falsified by: an address
    that does not resolve, or two nodes sharing one.
    """
    ids = node_ids_of(tree)
    assert len(ids) == len(set(ids)), "two nodes share an address"
    assert len(ids) == sum(1 for _ in tree.walk())
    for node_id in ids:
        assert node_at(tree, node_id) is not None
    assert node_at(tree, "") is tree.root


def test_k7_2_an_address_that_does_not_resolve_returns_none(tree):
    """Absence answers, rather than raising an index error from deep inside."""
    for bad in ("9999", "0.9999", "not-a-path", "0..1"):
        assert node_at(tree, bad) is None


def test_k7_10_a_resolved_node_carries_a_non_empty_provenance_chain(tree):
    """The point of an address: it leads back to a citation, for every node."""
    for node_id in node_ids_of(tree):
        node = node_at(tree, node_id)
        assert node.provenance.chain, f"{node_id} resolves to a node citing nothing"


# ---------------------------------------------------------------- the Hit contract

def test_k7_9_a_hit_keeps_its_lane_ranks_separate():
    """A fused score never replaces the ranks it was computed from.

    Falsified by: a hit found by one lane whose other lane rank is 0 rather than
    None — absent and last are not the same answer.
    """
    chunk = ChunkRecord(chunk_id="d" * 64, doc_id="c" * 64, seq=0, text="t",
                        level=0, node_ids=["0.1"])
    hit = Hit(chunk=chunk, dense_rank=None, lexical_rank=3, final_rank=1)
    assert hit.dense_rank is None and hit.lexical_rank == 3
    assert hit.boosts == {}
    assert Hit.from_dict(hit.to_dict()).to_dict() == hit.to_dict()


# =========================================================== the SQLite store

from ragix_kernels.saqqara.store.ports import DocumentStore, build_store  # noqa: E402
from ragix_kernels.saqqara.store.sqlite import SqliteDocumentStore  # noqa: E402


def _doc(sha: str, path: str, tree: Tree | None = None) -> DocumentRecord:
    return DocumentRecord(doc_id=sha, corpus="default", doc_class="x",
                          source_path=path, source_sha256=sha,
                          kernel="saqqara", kernel_version="1.0", tree=tree)


def _chunks(doc_id: str, texts: list[str], level: int = 0) -> list[ChunkRecord]:
    out = []
    for i, text in enumerate(texts):
        node_ids = [f"0.{i}"]
        out.append(ChunkRecord(
            chunk_id=chunk_id_for(doc_id=doc_id, level=level, node_ids=node_ids, text=text),
            doc_id=doc_id, seq=i, text=text, level=level, node_ids=node_ids,
            section_path=["Body"],
        ))
    return out


@pytest.fixture
def store(tmp_path) -> SqliteDocumentStore:
    with SqliteDocumentStore(path=str(tmp_path / "s.db")) as s:
        yield s


def test_k7_1_the_store_satisfies_the_protocol(store):
    """The seam is real: a store that answers half the protocol is half a store."""
    assert isinstance(store, DocumentStore)


def test_k7_1_a_provider_is_built_by_name(tmp_path):
    built = build_store({"provider": "sqlite", "path": str(tmp_path / "b.db")})
    assert isinstance(built, SqliteDocumentStore)
    built.close()


def test_k7_1_an_unknown_provider_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="unknown store provider"):
        build_store({"provider": "postgres", "path": str(tmp_path / "n.db")})


def test_k7_1_the_same_bytes_at_two_paths_are_one_document(store, tree):
    """Identity is the bytes; the paths are a history, not a second document."""
    sha = "1" * 64
    store.upsert_document(_doc(sha, "/one/m.xlsx", tree))
    store.upsert_document(_doc(sha, "/another/m.xlsx", tree))
    assert len(store.list_documents()) == 1
    assert sorted(store.source_paths(sha)) == ["/another/m.xlsx", "/one/m.xlsx"]


def test_k7_2_a_stored_tree_returns_identical(store, trees):
    """The store keeps the document, not an approximation of it."""
    for i, (fmt, tree) in enumerate(sorted(trees.items())):
        sha = f"{i}" * 64
        store.upsert_document(_doc(sha, f"/m.{fmt}", tree))
        back = store.get_document(sha)
        assert json.dumps(back.tree.to_dict(), **CANONICAL_JSON) == json.dumps(
            tree.to_dict(), **CANONICAL_JSON), fmt


def test_k7_5_rechunking_an_unchanged_tree_writes_nothing(store, tree):
    """The claim that makes embedding incremental rather than a re-run."""
    sha = "2" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    chunks = _chunks(sha, ["alpha", "beta", "gamma"])

    first = store.replace_chunks(sha, chunks)
    assert first == {"inserted": 3, "deleted": 0, "unchanged": 0}

    again = store.replace_chunks(sha, chunks)
    assert again == {"inserted": 0, "deleted": 0, "unchanged": 3}
    assert len(store.get_chunks(sha)) == 3


def test_k7_5_a_changed_chunk_replaces_only_itself(store, tree):
    sha = "3" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    store.replace_chunks(sha, _chunks(sha, ["alpha", "beta"]))
    moved = store.replace_chunks(sha, _chunks(sha, ["alpha", "changed"]))
    assert moved == {"inserted": 1, "deleted": 1, "unchanged": 1}


def test_k7_6_embeddings_are_keyed_by_chunk_and_model(store, tree):
    """Two models coexist: whose opinion a vector is belongs to the key."""
    sha = "4" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    chunks = _chunks(sha, ["alpha", "beta"])
    store.replace_chunks(sha, chunks)
    ids = [c.chunk_id for c in chunks]

    store.upsert_embeddings([EmbeddingRecord(chunk_id=ids[0], model="m1",
                                             dimensions=2, vector=(1.0, 0.0))])
    store.upsert_embeddings([EmbeddingRecord(chunk_id=ids[0], model="m2",
                                             dimensions=2, vector=(0.0, 1.0))])
    assert store.status()["embeddings"] == 2
    assert store.status()["models"] == ["m1", "m2"]

    # embedding again embeds only what is missing
    assert store.existing_embeddings(ids, "m1") == {ids[0]}
    assert store.existing_embeddings(ids, "m2") == {ids[0]}


def test_k7_6_a_stored_vector_returns_component_wise(store, tree):
    """float32 round trip: a vector that changes on the way back is not a cache."""
    sha = "5" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    chunks = _chunks(sha, ["alpha"])
    store.replace_chunks(sha, chunks)
    store.upsert_embeddings([EmbeddingRecord(chunk_id=chunks[0].chunk_id, model="m",
                                             dimensions=3, vector=(0.5, -0.25, 0.125))])
    got = store.get_embeddings("m")
    assert len(got) == 1
    assert got[0].vector == (0.5, -0.25, 0.125)


def test_k7_12_delete_trashes_and_parks_its_embeddings(store, tree):
    """Delete is trash. The vectors are parked, because recomputing is not restoring."""
    sha = "6" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    chunks = _chunks(sha, ["alpha", "beta"])
    store.replace_chunks(sha, chunks)
    store.upsert_embeddings([
        EmbeddingRecord(chunk_id=c.chunk_id, model="m", dimensions=2, vector=(0.5, 0.5))
        for c in chunks])

    counted = store.delete_document(sha)
    assert counted["documents"] == 1 and counted["parked"] == 2

    status = store.status()
    assert status["documents"] == 0 and status["trashed"] == 1
    assert status["embeddings"] == 0 and status["embeddings_parked"] == 2
    assert store.list_documents() == []
    assert len(store.list_documents(include_trashed=True)) == 1


def test_k7_12_restore_returns_the_same_vectors(store, tree):
    """Byte-identical, not merely present."""
    sha = "7" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    chunks = _chunks(sha, ["alpha"])
    store.replace_chunks(sha, chunks)
    store.upsert_embeddings([EmbeddingRecord(chunk_id=chunks[0].chunk_id, model="m",
                                             dimensions=2, vector=(0.25, -0.5))])
    before = store.get_embeddings("m")[0].vector

    store.delete_document(sha)
    assert store.get_embeddings("m") == []
    assert store.restore_document(sha) is True

    after = store.get_embeddings("m")
    assert len(after) == 1 and after[0].vector == before
    assert store.status()["embeddings_parked"] == 0


def test_k7_12_restoring_a_live_document_reports_false(store, tree):
    """Absence of an effect is reported, not disguised as success."""
    sha = "8" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    assert store.restore_document(sha) is False
    assert store.restore_document("9" * 64) is False


def test_k7_12_purge_is_opt_in_and_counted(store, tree):
    """A purge that reports nothing cannot be told from a purge that did nothing."""
    sha = "a" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    chunks = _chunks(sha, ["alpha", "beta"])
    store.replace_chunks(sha, chunks)
    store.upsert_embeddings([
        EmbeddingRecord(chunk_id=c.chunk_id, model="m", dimensions=2, vector=(1.0, 0.0))
        for c in chunks])

    counted = store.delete_document(sha, purge=True)
    assert counted == {"documents": 1, "chunks": 2, "embeddings": 2, "parked": 0}

    status = store.status()
    assert status["documents"] == 0 and status["trashed"] == 0
    assert status["chunks"] == 0 and status["embeddings"] == 0
    assert any(d["reason"] == "document-purged" for d in status["drops"])
    assert store.get_document(sha) is None


def test_k7_12_deleting_an_absent_document_counts_zero(store):
    assert store.delete_document("b" * 64) == {
        "documents": 0, "chunks": 0, "embeddings": 0, "parked": 0}


def test_k7_13_lexical_search_finds_text_after_an_upsert(store, tree):
    """FTS is written with the chunk, so a search after an upsert finds it."""
    sha = "c" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    store.replace_chunks(sha, _chunks(sha, ["the quick brown fox", "a slow green turtle"]))

    hits = store.lexical_search("brown", top_k=5)
    assert len(hits) == 1
    assert "brown" in hits[0].chunk.text
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_rank is None, "a lane that did not answer says None, not a number"


def test_k7_13_a_trashed_document_leaves_the_lexical_lane(store, tree):
    """A store that returns rows it considers deleted has two answers."""
    sha = "d" * 64
    store.upsert_document(_doc(sha, "/m.xlsx", tree))
    store.replace_chunks(sha, _chunks(sha, ["findable text"]))
    assert len(store.lexical_search("findable", top_k=5)) == 1
    store.delete_document(sha)
    assert store.lexical_search("findable", top_k=5) == []


def test_k7_13_an_empty_query_returns_nothing_rather_than_everything(store):
    assert store.lexical_search("   ", top_k=5) == []


def test_k7_8_the_store_answers_its_own_dense_lane(store, trees):
    """The protocol is satisfied by the store, not by a helper the caller must find."""
    tree = trees["docx"]
    sha = "e" * 64
    fed = feed_tree(tree, source_path="/a.docx", source_sha256=sha)
    store.upsert_document(fed.document)
    store.replace_chunks(sha, fed.chunks)
    embed_missing(store, fed.chunks, build_embedder("dummy"), model="d")

    hits = store.search([0.1] * 384, top_k=5, model="d")
    assert hits and all(h.dense_rank is not None for h in hits)
    assert all(h.lexical_rank is None for h in hits)


def test_k7_8_changing_the_chunks_drops_the_cached_index(store, trees):
    """An index patched in parallel with its source is a second answer waiting.

    Falsified by: a search after a chunk change still answering from stale vectors.
    """
    tree = trees["docx"]
    sha = "f" * 64
    fed = feed_tree(tree, source_path="/a.docx", source_sha256=sha)
    store.upsert_document(fed.document)
    store.replace_chunks(sha, fed.chunks)
    embed_missing(store, fed.chunks, build_embedder("dummy"), model="d")

    before = store.search([0.1] * 384, top_k=10, model="d")
    assert before

    store.replace_chunks(sha, fed.chunks[:1])
    after = store.search([0.1] * 384, top_k=10, model="d")
    assert len(after) < len(before), "the cache outlived the rows it described"


# ================================================================ the chunker

from ragix_kernels.saqqara.store.chunker import (  # noqa: E402
    CHUNKABLE_KINDS,
    chunk_tree,
)


@pytest.mark.parametrize("fmt", ["xlsx", "docx", "pdf", "pptx", "md"])
def test_k7_3_every_chunk_names_a_node_that_resolves(trees, fmt):
    """A chunk that cites nothing cannot be traced back, so it is never stored.

    Falsified by: a chunk whose node_ids do not resolve in the tree it came from.
    """
    tree = trees[fmt]
    plan = chunk_tree(tree, doc_id="1" * 64)
    assert plan.chunks, f"{fmt} produced no chunk at all"
    for chunk in plan.chunks:
        assert chunk.node_ids, "a chunk with no node reached the plan"
        for node_id in chunk.node_ids:
            assert node_at(tree, node_id) is not None, f"{fmt}: {node_id} does not resolve"


@pytest.mark.parametrize("fmt", ["xlsx", "docx", "pdf", "pptx", "md"])
def test_k7_3_refusals_are_counted_not_silent(trees, fmt):
    """Containers and empty nodes are refused with a reason, never dropped quietly.

    Falsified by: a tree whose nodes are neither chunked nor accounted for.
    """
    tree = trees[fmt]
    plan = chunk_tree(tree, doc_id="1" * 64)
    level0 = [c for c in plan.chunks if c.level == 0]
    accounted = {c.node_ids[0] for c in level0} | {r["node_id"] for r in plan.refusals}
    assert accounted == set(node_ids_of(tree)), f"{fmt}: nodes neither chunked nor refused"
    for refusal in plan.refusals:
        assert refusal["reason"] in (
            "not-a-chunkable-kind", "no-text", "section-is-context")


def test_k7_3_a_chunk_record_with_no_node_cannot_be_built():
    """The refusal is in the record, so no path can construct an untraceable chunk."""
    with pytest.raises(ValueError):
        ChunkRecord(chunk_id="d" * 64, doc_id="c" * 64, seq=0, text="t",
                    level=0, node_ids=[])


def test_k7_4_a_rollup_covers_exactly_its_children(trees):
    """Exactly: the union of its level-0 children's nodes, not a window over them.

    Falsified by: a roll-up naming a node no child names, or missing one a child does.
    """
    tree = trees["docx"]
    plan = chunk_tree(tree, doc_id="2" * 64)
    rollups = [c for c in plan.chunks if c.level == 1]
    assert rollups, "no roll-up was produced"

    for rollup in rollups:
        children = [c for c in plan.chunks if c.level == 0 and c.parent_id == rollup.chunk_id]
        assert children, "a roll-up with no children is a roll-up of nothing"
        covered = []
        for child in children:
            for nid in child.node_ids:
                if nid not in covered:
                    covered.append(nid)
        assert rollup.node_ids == covered


def test_k7_4_a_rollup_has_no_parent_of_its_own(trees):
    """One level of roll-up. A roll-up with a parent would be a third level."""
    plan = chunk_tree(trees["docx"], doc_id="2" * 64)
    for rollup in [c for c in plan.chunks if c.level == 1]:
        assert rollup.parent_id is None


def test_k7_4_rollups_can_be_switched_off(trees):
    """rollup_levels=0 yields units only — the roll-up is a choice, not a fixture."""
    plan = chunk_tree(trees["docx"], doc_id="2" * 64, rollup_levels=0)
    assert plan.chunks and all(c.level == 0 for c in plan.chunks)
    assert all(c.parent_id is None for c in plan.chunks)


def test_k7_5_chunking_the_same_tree_twice_yields_the_same_ids(trees):
    """The claim the store's incremental write depends on."""
    for fmt, tree in trees.items():
        first = chunk_tree(tree, doc_id="3" * 64)
        again = chunk_tree(tree, doc_id="3" * 64)
        assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in again.chunks], fmt


def test_k7_5_rechunking_writes_nothing_through_the_store(store, trees):
    """End to end: tree -> chunks -> store, twice, with no row touched the second time."""
    sha = "4" * 64
    tree = trees["docx"]
    store.upsert_document(_doc(sha, "/m.docx", tree))
    plan = chunk_tree(tree, doc_id=sha)

    first = store.replace_chunks(sha, plan.chunks)
    assert first["inserted"] == len(plan.chunks) and first["deleted"] == 0

    again = store.replace_chunks(sha, chunk_tree(tree, doc_id=sha).chunks)
    assert again == {"inserted": 0, "deleted": 0, "unchanged": len(plan.chunks)}


def test_k7_3_an_oversized_unit_is_flagged_rather_than_cut(trees):
    """A semantic unit is kept whole; being large is recorded, not resolved by cutting."""
    tree = trees["md"]
    plan = chunk_tree(tree, doc_id="5" * 64, unit_max_chars=5)
    flagged = [c for c in plan.chunks if c.meta.get("oversize")]
    assert flagged, "nothing was flagged although the limit was tiny"
    assert all("fallback" not in c.meta for c in flagged), "an oversized unit was windowed"


def test_k7_3_only_a_unit_beyond_the_window_is_windowed_and_says_so(trees):
    """The window is the declared fallback: every piece of it carries the flag."""
    tree = trees["md"]
    plan = chunk_tree(tree, doc_id="6" * 64, unit_max_chars=5, window_fallback_chars=10)
    windowed = [c for c in plan.chunks if c.meta.get("fallback") == "window"]
    if windowed:
        assert all(len(c.text) <= 10 for c in windowed)
        for chunk in windowed:
            assert chunk.node_ids and node_at(tree, chunk.node_ids[0]) is not None


def test_k7_3_containers_are_never_chunks(trees):
    """A document or a section is the context of a chunk, never a chunk."""
    for fmt, tree in trees.items():
        plan = chunk_tree(tree, doc_id="7" * 64)
        for chunk in [c for c in plan.chunks if c.level == 0]:
            node = node_at(tree, chunk.node_ids[0])
            assert node.kind in CHUNKABLE_KINDS, f"{fmt}: {node.kind} became a chunk"


# =================================================================== the feed

import tempfile  # noqa: E402

from ragix_kernels.base import KernelInput  # noqa: E402
from ragix_kernels.saqqara.analyzers.caption_binding import CaptionBindingAnalyzer  # noqa: E402
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel  # noqa: E402
from ragix_kernels.saqqara.store.feed import edges_of, feed_result, feed_tree, objects_of  # noqa: E402


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    """One real kernel run over generated fixtures, reused by the feed tests."""
    workspace = tmp_path_factory.mktemp("feed")
    source = workspace / "corpus"
    source.mkdir()
    for name, fixture in (("a.docx", "docx_two_tier"), ("b.xlsx", "mixed_workbook"),
                          ("c.pdf", "pdf_caption_below"), ("d.md", "markdown_document")):
        G.FIXTURES[fixture](source / name)
    kernel = SaqqaraKernel()
    return kernel.run(KernelInput(workspace=workspace, config={"source": {"path": str(source)}}))


def test_k7_11_a_tree_feeds_documents_chunks_and_objects(trees):
    fed = feed_tree(trees["docx"], source_path="/a.docx", source_sha256="1" * 64)
    assert fed.document.doc_id == "1" * 64
    assert fed.chunks and all(c.node_ids for c in fed.chunks)
    counts = fed.counts()
    assert counts["chunks"] == len(fed.chunks)
    assert "refused" in counts


def test_k7_11_objects_are_the_kinds_the_store_keeps(trees):
    """A table is an object; a paragraph is not. The set is closed, not guessed."""
    for fmt, tree in trees.items():
        objects = objects_of(tree, doc_id="2" * 64)
        for obj in objects:
            assert obj.kind in ("table", "figure", "vector_region")
            assert node_at(tree, obj.node_id) is not None, f"{fmt}: object cites no node"


def test_k7_11_a_figure_with_no_stored_bytes_is_still_an_object():
    """A figure that was SEEN is a fact whether or not its pixels were kept."""
    from ragix_kernels.saqqara.model import DocumentLocator, Node, Provenance, Tree as T
    prov = Provenance(source_path="/x.docx", source_format="docx",
                      chain=(DocumentLocator(),), kernel="k", kernel_version="1")
    root = Node(kind="document", provenance=prov, children=[
        Node(kind="figure", provenance=prov, facts={"width": 10}),
    ])
    objects = objects_of(T(root=root), doc_id="3" * 64)
    assert len(objects) == 1
    assert objects[0].kind == "figure" and objects[0].asset_ref is None


def test_k7_11_bindings_become_edges_and_nothing_is_inferred(trees):
    """The feed copies a decision an analyzer recorded; it does not make one."""
    tree = trees["docx"]
    before = edges_of(tree, doc_id="4" * 64)
    assert before == [], "an edge appeared with no analyzer having decided one"


@pytest.fixture(scope="module")
def bound_tree(tmp_path_factory):
    """A tree that genuinely carries figures and bindings.

    Built the way the K6 gate builds it — adapter, asset store, builder, then the
    caption analyzer — because the DEFAULT kernel pipeline does not run the object
    analyzers, so a tree from a plain run has no figure to bind and no edge to
    store. The first version of these tests used a plain run and passed on empty
    lists: it asserted that nothing equalled nothing.
    """
    from ragix_kernels.saqqara.adapters import adapter_for, read_path
    from ragix_kernels.saqqara.assets import AssetStore
    from ragix_kernels.saqqara.builder import build_tree

    root = tmp_path_factory.mktemp("k7bound")
    path = G.FIXTURES["pdf_caption_below"](G.fixture_path("pdf_caption_below", root))
    adapter = adapter_for(path)
    records = read_path(path, store=AssetStore(root / "assets"))
    tree = build_tree(records, str(path), adapter.format, adapter.format,
                      adapter.version).tree
    CaptionBindingAnalyzer().run(tree)
    return tree


def test_k7_11_the_bound_fixture_actually_has_objects_and_bindings(bound_tree):
    """The control on the tests below: they must not be able to pass on nothing."""
    assert len(objects_of(bound_tree, doc_id="6" * 64)) >= 1
    assert len(edges_of(bound_tree, doc_id="6" * 64)) >= 1


def test_k7_11_a_bound_caption_reaches_the_store_as_an_edge(bound_tree, tmp_path):
    """End to end on the K6 fixture: bind, feed, store, read the edge back."""
    fed = feed_tree(bound_tree, source_path="/a.pdf", source_sha256="7" * 64)
    assert fed.objects and fed.edges, "the fixture stopped carrying what it is for"

    with SqliteDocumentStore(path=str(tmp_path / "e.db")) as s:
        s.upsert_document(fed.document, objects=fed.objects, edges=fed.edges)
        s.replace_chunks(fed.document.doc_id, fed.chunks)

        stored_objects = s.get_objects(fed.document.doc_id)
        stored_edges = s.get_edges(fed.document.doc_id)
        assert [o.to_dict() for o in stored_objects] == [o.to_dict() for o in fed.objects]
        assert [(e.src, e.dst, e.type) for e in stored_edges] == \
               [(e.src, e.dst, e.type) for e in fed.edges]

        for edge in stored_edges:
            assert edge.type == "binds"
            assert node_at(bound_tree, edge.src).kind == "caption"
            assert node_at(bound_tree, edge.dst).kind == "figure"


def test_k7_11_the_caption_source_line_does_not_become_an_edge(bound_tree):
    """captioned_by is the caption's own provenance, not a relation to a figure.

    Falsified by: an edge per caption to the text line it was made from, which
    would double the count and assert a binding no analyzer decided.
    """
    edges = edges_of(bound_tree, doc_id="8" * 64)
    captions = [n for n in bound_tree.walk() if n.kind == "caption"]
    assert len(edges) == len(captions), "one edge per bound caption, not two"


def test_k7_11_an_unresolvable_binding_is_refused_and_counted(bound_tree):
    """A binding whose target cannot be resolved is counted, never dropped."""
    from ragix_kernels.saqqara.model import Tree as T

    tree = T.from_dict(bound_tree.to_dict())
    for node in tree.walk():
        if node.facts and "caption_of" in node.facts:
            node.facts["caption_of"] = {"format": "pdf", "page": 9999}
    refusals: list = []
    edges = edges_of(tree, doc_id="9" * 64, refusals=refusals)
    assert edges == []
    assert refusals and all(r["reason"] == "binding-target-not-found" for r in refusals)


def test_k7_11_a_chunk_cites_the_objects_its_nodes_carry(trees):
    """object_refs is computed where assets are known, not inside the chunker."""
    tree = trees["docx"]
    fed = feed_tree(tree, source_path="/a.docx", source_sha256="5" * 64)
    object_nodes = {o.node_id for o in fed.objects}
    for chunk in fed.chunks:
        expected = [n for n in chunk.node_ids if n in object_nodes]
        assert chunk.object_refs == expected


def test_k7_1_both_feeds_agree(run_result, tmp_path):
    """A store fed two ways must not depend on which way it was fed."""
    stored = Path(run_result.output_file)
    from_disk = feed_result(stored)
    assert from_disk, "the stored result fed nothing"

    for fed in from_disk:
        tree = fed.document.tree
        again = feed_tree(tree, source_path=fed.document.source_path,
                          source_sha256=fed.document.source_sha256,
                          doc_class=fed.document.doc_class)
        assert [c.chunk_id for c in fed.chunks] == [c.chunk_id for c in again.chunks]
        assert [o.to_dict() for o in fed.objects] == [o.to_dict() for o in again.objects]
        assert [e.to_dict() for e in fed.edges] == [e.to_dict() for e in again.edges]


def test_k7_1_feeding_an_empty_result_is_refused(tmp_path):
    """An empty store that looks like a successful one is the failure to avoid."""
    with pytest.raises(ValueError, match="no documents"):
        feed_result({"data": {"documents": []}})


def test_k7_1_a_whole_run_stores_and_reads_back(run_result, tmp_path):
    """Four formats, one store, every chunk still resolving to its tree."""
    with SqliteDocumentStore(path=str(tmp_path / "w.db")) as s:
        for fed in feed_result(Path(run_result.output_file)):
            s.upsert_document(fed.document, objects=fed.objects, edges=fed.edges)
            s.replace_chunks(fed.document.doc_id, fed.chunks)

        status = s.status()
        assert status["documents"] == 4
        assert status["chunks"] > 0

        for doc in s.list_documents():
            tree = doc.tree
            for chunk in s.get_chunks(doc.doc_id):
                for node_id in chunk.node_ids:
                    node = node_at(tree, node_id)
                    assert node is not None
                    assert node.provenance.chain, "a stored chunk cites a node citing nothing"


# ============================================================ embedding, K7.6/K7.7

import os  # noqa: E402

from ragix_kernels.saqqara.store.embed import (  # noqa: E402
    PROVIDERS,
    build_embedder,
    embed_missing,
)


class _CountingEmbedder:
    """A dummy that reports how many texts it was actually asked to embed."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[int] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [[float(len(t) % 7), 0.5, -0.25, 1.0] for t in texts]


def _stored(store, tree, sha: str, texts: list[str]):
    store.upsert_document(_doc(sha, "/m.docx", tree))
    chunks = _chunks(sha, texts)
    store.replace_chunks(sha, chunks)
    return chunks


def test_k7_7_provider_none_builds_no_embedder(tmp_path):
    """Choosing to run without embeddings is a decision, and it is a provider."""
    assert build_embedder("none") is None
    assert "none" in PROVIDERS


def test_k7_7_an_unknown_provider_is_refused_by_name():
    with pytest.raises(ValueError, match="embedder provider"):
        build_embedder("word2vec")


def test_k7_7_no_embedder_writes_nothing_at_all(store, tree):
    """Not a zero vector, not a placeholder — nothing.

    Falsified by: any row in embeddings after indexing with provider none.
    """
    sha = "e" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta"])
    plan = embed_missing(store, chunks, embedder=None, model="")
    assert plan.disabled is True and plan.embedded == 0 and plan.skipped == 2
    assert store.status()["embeddings"] == 0
    assert store.get_embeddings("") == []


def test_k7_7_a_lexical_only_store_still_answers(store, tree):
    """The point of provider none: the store works, it just has one lane."""
    sha = "f" * 64
    chunks = _stored(store, tree, sha, ["the quick brown fox", "a slow green turtle"])
    embed_missing(store, chunks, embedder=None, model="")
    hits = store.lexical_search("turtle", top_k=5)
    assert len(hits) == 1 and hits[0].dense_rank is None


def test_k7_6_only_missing_chunks_are_embedded(store, tree):
    """The claim that makes re-indexing cheap: a set difference, not a re-run."""
    sha = "1" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta", "gamma"])
    embedder = _CountingEmbedder()

    first = embed_missing(store, chunks, embedder, model="m")
    assert first.embedded == 3 and embedder.calls == [3]

    second = embed_missing(store, chunks, embedder, model="m")
    assert second.embedded == 0 and second.skipped == 3
    assert embedder.calls == [3], "the embedder was asked again for chunks it had done"


def test_k7_6_a_new_chunk_embeds_only_itself(store, tree):
    sha = "2" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta"])
    embedder = _CountingEmbedder()
    embed_missing(store, chunks, embedder, model="m")

    grown = _stored(store, tree, sha, ["alpha", "beta", "gamma"])
    embed_missing(store, grown, embedder, model="m")
    assert embedder.calls == [2, 1], "a new chunk cost more than itself"


def test_k7_6_two_models_coexist_rather_than_overwrite(store, tree):
    """Whose opinion a vector is belongs to the key, so both are kept."""
    sha = "3" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta"])
    embed_missing(store, chunks, _CountingEmbedder(), model="m1")
    embed_missing(store, chunks, _CountingEmbedder(dimension=4), model="m2")

    status = store.status()
    assert status["models"] == ["m1", "m2"] and status["embeddings"] == 4
    assert len(store.get_embeddings("m1")) == 2
    assert len(store.get_embeddings("m2")) == 2


def test_k7_6_a_partial_answer_from_an_embedder_is_refused(store, tree):
    """Vectors that cannot be matched to their inputs are not stored."""
    class _Short:
        def embed_batch(self, texts):
            return [[1.0, 0.0]] * (len(texts) - 1)

    sha = "4" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta"])
    with pytest.raises(RuntimeError, match="cannot be matched"):
        embed_missing(store, chunks, _Short(), model="m")
    assert store.status()["embeddings"] == 0


def test_k7_6_an_empty_vector_is_refused(store, tree):
    """An absent embedding is not a zero vector; storing one would hide it."""
    class _Empty:
        def embed_batch(self, texts):
            return [[] for _ in texts]

    sha = "5" * 64
    chunks = _stored(store, tree, sha, ["alpha"])
    with pytest.raises(RuntimeError, match="empty vector"):
        embed_missing(store, chunks, _Empty(), model="m")
    assert store.status()["embeddings"] == 0


def test_k7_6_the_dummy_backend_from_ragix_core_is_usable(store, tree):
    """The family uses ragix_core's backends; it does not grow its own."""
    sha = "6" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta"])
    plan = embed_missing(store, chunks, build_embedder("dummy"), model="dummy-384")
    assert plan.embedded == 2
    assert all(e.dimensions == 384 for e in store.get_embeddings("dummy-384"))


@pytest.mark.skipif(os.environ.get("OLLAMA_LIVE") != "1",
                    reason="live embedding test; set OLLAMA_LIVE=1 to run")
def test_k7_6_the_ollama_backend_embeds_live(store, tree):
    """Opt-in: nothing in this suite talks to a model server by default."""
    sha = "7" * 64
    chunks = _stored(store, tree, sha, ["alpha", "beta"])
    embedder = build_embedder("ollama", model=os.environ.get("OLLAMA_EMBED_MODEL",
                                                             "nomic-embed-text"))
    plan = embed_missing(store, chunks, embedder, model="ollama-live")
    assert plan.embedded == 2
    vectors = store.get_embeddings("ollama-live")
    assert len({v.dimensions for v in vectors}) == 1
    assert all(any(c != 0.0 for c in v.vector) for v in vectors)


# ====================================================== retrieval, K7.8/K7.9/K7.10

from ragix_kernels.saqqara.store.retrieve import (  # noqa: E402
    RRF_K,
    Retriever,
    provenance_of,
    related_of,
)


@pytest.fixture
def indexed(store, trees):
    """A store holding one real document, chunked, embedded with a dummy backend."""
    tree = trees["docx"]
    sha = "b" * 64
    fed = feed_tree(tree, source_path="/a.docx", source_sha256=sha)
    store.upsert_document(fed.document, objects=fed.objects, edges=fed.edges)
    store.replace_chunks(sha, fed.chunks)
    embed_missing(store, fed.chunks, build_embedder("dummy"), model="d")
    return store, tree, sha, fed


def test_k7_8_the_dense_lane_reads_its_vectors_from_the_database(indexed):
    """Not from a file beside it: the DB is the source, the index is the cache."""
    store, _tree, _sha, fed = indexed
    retriever = Retriever(store, model="d")
    vector = [0.0] * store.get_embeddings("d")[0].dimensions
    vector[0] = 1.0

    hits = retriever.dense(vector, top_k=5)
    assert hits and all(h.dense_rank is not None for h in hits)
    assert all(h.lexical_rank is None for h in hits)


def test_k7_8_dropping_the_cache_reproduces_identical_hits(indexed):
    """A cache that changes the answer is a second store, not a cache.

    Falsified by: a different hit list after invalidate() and a rebuild.
    """
    store, _tree, _sha, _fed = indexed
    retriever = Retriever(store, model="d")
    vector = [0.1] * store.get_embeddings("d")[0].dimensions

    before = [(h.chunk.chunk_id, h.dense_rank) for h in retriever.dense(vector, top_k=10)]
    retriever.invalidate()
    after = [(h.chunk.chunk_id, h.dense_rank) for h in retriever.dense(vector, top_k=10)]
    assert before == after and before


def test_k7_8_a_store_with_no_vectors_has_an_empty_dense_lane(store, trees):
    """Empty because nothing was embedded — and the lexical lane still answers."""
    tree = trees["docx"]
    sha = "c" * 64
    fed = feed_tree(tree, source_path="/a.docx", source_sha256=sha)
    store.upsert_document(fed.document)
    store.replace_chunks(sha, fed.chunks)

    retriever = Retriever(store, model="d")
    assert retriever.dense([0.1, 0.2], top_k=5) == []
    assert retriever.lexical(fed.chunks[0].text.split()[0], top_k=5)


def test_k7_9_a_fused_hit_keeps_both_lane_ranks(indexed):
    """The fused rank is added beside the lane ranks, never in place of them."""
    store, _tree, _sha, fed = indexed
    retriever = Retriever(store, model="d")
    vector = [0.1] * store.get_embeddings("d")[0].dimensions
    word = fed.chunks[0].text.split()[0]

    hits = retriever.search(query=word, vector=vector, top_k=10)
    assert hits
    for hit in hits:
        assert hit.final_rank is not None
        assert hit.dense_rank is not None or hit.lexical_rank is not None
    assert any(h.dense_rank is not None for h in hits)


def test_k7_9_a_hit_from_one_lane_says_none_for_the_other(indexed):
    """Absent and last are different answers; only None can say the first."""
    store, _tree, _sha, fed = indexed
    retriever = Retriever(store, model="")   # no model -> no dense lane at all
    word = fed.chunks[0].text.split()[0]

    hits = retriever.search(query=word, vector=None, top_k=10)
    assert hits
    assert all(h.dense_rank is None for h in hits)
    assert all(h.lexical_rank is not None for h in hits)
    assert all(h.final_rank is not None for h in hits)


def test_k7_9_the_fusion_is_over_ranks_not_scores(indexed):
    """RRF: 1/(k+rank). A lane's raw score never enters the objective."""
    store, _tree, _sha, fed = indexed
    retriever = Retriever(store, model="d", rrf_k=RRF_K)
    vector = [0.1] * store.get_embeddings("d")[0].dimensions
    word = fed.chunks[0].text.split()[0]

    hits = retriever.search(query=word, vector=vector, top_k=10)
    ranks = [h.final_rank for h in hits]
    assert ranks == sorted(ranks) == list(range(1, len(hits) + 1))
    assert all(h.boosts == {} for h in hits), "boosts are declared but empty in this PR"


def test_k7_10_every_hit_resolves_to_a_non_empty_provenance_chain(indexed):
    """What a hit is FOR. Text with a score and no citation is the failure."""
    store, tree, _sha, fed = indexed
    retriever = Retriever(store, model="d")
    word = fed.chunks[0].text.split()[0]

    hits = retriever.search(query=word, vector=None, top_k=10)
    assert hits
    for hit in hits:
        chains = provenance_of(tree, hit.chunk)
        assert chains, "a hit that leads nowhere"
        for entry in chains:
            assert entry["chain"], "a node citing nothing"
            assert entry["source_path"] and entry["source_format"]


def test_k7_11_a_hit_carries_the_objects_and_edges_of_its_nodes(bound_tree, tmp_path):
    """related/objects follow edges an analyzer decided; nothing is decided here."""
    fed = feed_tree(bound_tree, source_path="/a.pdf", source_sha256="d" * 64)
    with SqliteDocumentStore(path=str(tmp_path / "r.db")) as s:
        s.upsert_document(fed.document, objects=fed.objects, edges=fed.edges)
        s.replace_chunks(fed.document.doc_id, fed.chunks)

        bound_nodes = {e.src for e in fed.edges} | {e.dst for e in fed.edges}
        touching = [c for c in fed.chunks if set(c.node_ids) & bound_nodes]
        assert touching, "no chunk covers a bound node; the fixture changed"

        found = related_of(s, touching[0])
        assert found["related"], "a chunk over a bound node reported no relation"
        for edge in found["related"]:
            assert edge["type"] == "binds"


# ============================================================ configuration, K7.14

from ragix_kernels.saqqara.store.config import (  # noqa: E402
    DEFAULTS_PATH,
    load_config,
    resolve_secret,
)


def test_k7_14_the_packaged_defaults_load_and_declare_every_section():
    """The defaults are the shape. If they do not load, nothing below means anything."""
    assert DEFAULTS_PATH.is_file()
    config = load_config()
    for section in ("source", "store", "embedder", "index", "chunker", "retrieval"):
        assert config.section(section), f"{section} is missing from the defaults"
    assert config.get("embedder.provider") == "none"
    assert config.get("retrieval.rrf_k") == 60


def test_k7_14_a_user_file_overlays_only_what_it_names(tmp_path):
    path = tmp_path / "saqqara.yaml"
    path.write_text("store:\n  corpus: mine\nretrieval:\n  top_k: 3\n", encoding="utf-8")
    config = load_config(path)
    assert config.get("store.corpus") == "mine"
    assert config.get("retrieval.top_k") == 3
    assert config.get("retrieval.rrf_k") == 60, "an untouched key lost its default"
    assert config.get("store.provider") == "sqlite"


def test_k7_14_an_unknown_key_is_refused_with_its_path(tmp_path):
    """A typo discarded in silence is the user's instruction thrown away.

    Falsified by: a run that accepts `embedder.privider` and uses the default.
    """
    path = tmp_path / "bad.yaml"
    path.write_text("embedder:\n  privider: ollama\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"embedder\.privider"):
        load_config(path)


def test_k7_14_an_unknown_top_level_key_is_refused_with_its_path(tmp_path):
    path = tmp_path / "bad2.yaml"
    path.write_text("embeder:\n  provider: ollama\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embeder"):
        load_config(path)


def test_k7_14_a_value_where_a_mapping_belongs_is_refused(tmp_path):
    """Shape is part of the contract, not only the key name."""
    path = tmp_path / "bad3.yaml"
    path.write_text("embedder: ollama\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embedder"):
        load_config(path)


def test_k7_14_a_missing_file_is_refused_rather_than_ignored(tmp_path):
    """Silently falling back to defaults would run the wrong configuration."""
    with pytest.raises(ValueError, match="no configuration file"):
        load_config(tmp_path / "absent.yaml")


def test_k7_14_an_impossible_provider_is_refused_by_name(tmp_path):
    path = tmp_path / "bad4.yaml"
    path.write_text("embedder:\n  provider: word2vec\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embedder.provider"):
        load_config(path)


def test_k7_14_an_override_is_validated_like_a_file():
    """A key typed on a command line is refused by the same rule as one in a file."""
    assert load_config(None, **{"retrieval.top_k": 5}).get("retrieval.top_k") == 5
    with pytest.raises(ValueError, match=r"retrieval\.topk"):
        load_config(None, **{"retrieval.topk": 5})


def test_k7_14_no_secret_is_resolved_during_load(monkeypatch, tmp_path):
    """A config object is always safe to write down, which is why nothing resolves.

    Falsified by: a loaded configuration whose serialisation contains the value.
    """
    monkeypatch.setenv("SAQQARA_TEST_KEY", "the-actual-secret")
    path = tmp_path / "s.yaml"
    path.write_text('embedder:\n  api_key_ref: "env:SAQQARA_TEST_KEY"\n', encoding="utf-8")

    config = load_config(path)
    serialised = json.dumps(config.to_dict())
    assert "the-actual-secret" not in serialised
    assert "env:SAQQARA_TEST_KEY" in serialised
    assert config.api_key() == "the-actual-secret", "it must still resolve at use"


def test_k7_14_a_secret_reference_resolves_from_the_environment(monkeypatch):
    monkeypatch.setenv("SAQQARA_TEST_KEY", "v")
    assert resolve_secret("env:SAQQARA_TEST_KEY") == "v"


def test_k7_14_an_unresolvable_secret_fails_closed(monkeypatch):
    """Returning the reference would send the string "env:TOKEN" as a credential."""
    monkeypatch.delenv("SAQQARA_ABSENT", raising=False)
    with pytest.raises(ValueError, match="unset or empty"):
        resolve_secret("env:SAQQARA_ABSENT")
    with pytest.raises(ValueError, match="named by reference"):
        resolve_secret("just-a-raw-value")
    with pytest.raises(ValueError, match="unknown secret scheme"):
        resolve_secret("vault:something")


def test_k7_14_a_labelled_secret_file_selects_its_line(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("Other abc\nEmbedKey s3cret\n", encoding="utf-8")
    assert resolve_secret(f"file:{path}#EmbedKey") == "s3cret"
    with pytest.raises(ValueError, match="not found"):
        resolve_secret(f"file:{path}#Missing")


def test_k7_14_an_unreadable_secret_file_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="cannot read secret file"):
        resolve_secret(f"file:{tmp_path / 'nope.txt'}")


# ================================================ the kernel and its surfaces, K7.15/K7.16

from ragix_kernels.saqqara.kernels.saqqara_index import SaqqaraIndexKernel  # noqa: E402


def test_k7_15_the_index_kernel_declares_what_it_needs_and_gives():
    """The registry orders it, rather than anyone remembering to."""
    assert SaqqaraIndexKernel.requires == ["document_tree"]
    assert SaqqaraIndexKernel.provides == ["document_store"]
    assert SaqqaraIndexKernel.stage == 2
    assert SaqqaraKernel.provides == ["document_tree", "traces", "merkle_root"]


def test_k7_15_the_registry_orders_the_reader_before_the_indexer():
    """Falsified by: an order in which the store is built before anything is read."""
    from ragix_kernels.registry import KernelRegistry

    KernelRegistry.discover()
    found = {k.name for k in (KernelRegistry.get("saqqara"), KernelRegistry.get("saqqara_index"))
             if k is not None}
    assert found == {"saqqara", "saqqara_index"}
    assert KernelRegistry.get("saqqara_index").stage > KernelRegistry.get("saqqara").stage


@pytest.fixture(scope="module")
def indexed_workspace(tmp_path_factory):
    """A real run then a real index, over generated fixtures, with no embedder."""
    workspace = tmp_path_factory.mktemp("k7kernel")
    source = workspace / "corpus"
    source.mkdir()
    for name, fixture in (("a.docx", "docx_two_tier"), ("b.md", "markdown_document")):
        G.FIXTURES[fixture](source / name)
    SaqqaraKernel().run(KernelInput(workspace=workspace,
                                    config={"source": {"path": str(source)}}))
    read = workspace / "stage1" / "saqqara.json"
    output = SaqqaraIndexKernel().run(KernelInput(
        workspace=workspace, config={}, dependencies={"document_tree": read}))
    return workspace, output


def test_k7_15_the_index_kernel_stores_what_the_reader_read(indexed_workspace):
    _workspace, output = indexed_workspace
    assert output.success is True
    assert len(output.data["documents"]) == 2
    assert output.data["status"]["chunks"] > 0
    assert output.data["status"]["dense"] == "disabled (no embedder)"
    assert output.data["embedded"] == 0, "provider none wrote a vector"


def test_k7_15_the_envelope_refuses_the_kernel_without_its_dependency(tmp_path):
    """The declaration is enforced by the envelope, not merely documented.

    Falsified by: a run that proceeds with no document_tree and produces an empty
    store that looks like a successful one.
    """
    output = SaqqaraIndexKernel().run(KernelInput(workspace=tmp_path, config={}))
    assert output.success is False
    assert any("document_tree" in str(e) for e in output.errors), output.errors


def test_k7_15_a_declared_dependency_that_is_absent_is_refused(tmp_path):
    """Declared but missing is refused too — the file is checked, not just the key."""
    output = SaqqaraIndexKernel().run(KernelInput(
        workspace=tmp_path, config={},
        dependencies={"document_tree": tmp_path / "never-written.json"}))
    assert output.success is False
    assert any("does not exist" in str(e) for e in output.errors), output.errors


def test_k7_16_the_cli_and_the_mcp_tool_return_the_same_hits(indexed_workspace):
    """One shape, or two surfaces that will disagree about a citation.

    Falsified by: any difference between the CLI's --json output and the MCP
    tool's hits for the same query on the same store.
    """
    import argparse
    import contextlib
    import io

    from ragix_kernels.saqqara.cli.saqqaractl import cmd_search
    from ragix_kernels.saqqara.mcp.tools import register_saqqara_tools

    workspace, _output = indexed_workspace
    query = "paragraphe"   # the generated fixtures are French

    args = argparse.Namespace(workspace=str(workspace), query=query, config=None,
                              top_k=5, json=True, verbose=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert cmd_search(args) == 0
    from_cli = json.loads(buffer.getvalue() or "[]")

    server = _StubServer()
    register_saqqara_tools(server)
    from_mcp = server.tools["koas_saqqara_search"](
        workspace=str(workspace), query=query, k=5)

    assert "error" not in from_mcp, from_mcp
    assert from_mcp["hits"] == from_cli


class _StubServer:
    def __init__(self):
        self.tools: dict = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


def test_k7_16_the_mcp_surface_declares_all_four_tools():
    from ragix_kernels.saqqara.mcp.tools import register_saqqara_tools

    server = _StubServer()
    register_saqqara_tools(server)
    assert list(server.tools) == ["koas_saqqara_run", "koas_saqqara_status",
                                  "koas_saqqara_index", "koas_saqqara_search"]


def test_k7_16_every_hit_from_either_surface_carries_its_citation(indexed_workspace):
    """The shape is not merely equal on both sides; it is the useful one."""
    import argparse
    import contextlib
    import io

    from ragix_kernels.saqqara.cli.saqqaractl import cmd_search

    workspace, _output = indexed_workspace
    args = argparse.Namespace(workspace=str(workspace), query="paragraphe", config=None,
                              top_k=5, json=True, verbose=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cmd_search(args)
    hits = json.loads(buffer.getvalue() or "[]")
    assert hits, "the fixture stopped producing a hit"
    for hit in hits:
        assert hit["provenance"], "a hit with no citation"
        for entry in hit["provenance"]:
            assert entry["chain"] and entry["source_path"]
        assert "dense_rank" in hit and "lexical_rank" in hit and "final_rank" in hit


def test_k7_15_a_provider_resolves_without_anyone_importing_its_module():
    """Registration must not depend on import order, and this is how it failed.

    register_store runs as an import side effect of the sqlite module. The kernel
    imports only ports, so in any process that had not separately imported sqlite,
    build_store raised "unknown store provider 'sqlite'; registered: none". The
    test suite could not see it: a test importing SqliteDocumentStore registers it
    for everything that follows.

    So this runs in a SUBPROCESS that imports ports and nothing else. Falsified by:
    a provider that resolves only when something else imported it first.
    """
    import subprocess
    import sys as _sys

    code = (
        "from ragix_kernels.saqqara.store.ports import build_store;"
        "import tempfile;"
        "s = build_store({'provider':'sqlite','path':tempfile.mkdtemp()+'/x.db'});"
        "print(type(s).__name__)"
    )
    done = subprocess.run([_sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "SqliteDocumentStore"
