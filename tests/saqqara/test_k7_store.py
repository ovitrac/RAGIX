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


def test_k7_8_the_dense_lane_refuses_rather_than_returning_empty(store):
    """An empty result claims nothing matched; this store cannot claim that."""
    with pytest.raises(NotImplementedError, match="dense lane"):
        store.search([0.1, 0.2], top_k=5, model="m")


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
