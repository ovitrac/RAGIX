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
