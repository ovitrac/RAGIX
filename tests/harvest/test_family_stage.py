"""Row 30's family stage: the order the data forces, and the provenance that makes it auditable (S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The stage is driven whole against a **synthetic three-family tree** with a fake server — no model, no
GPU, no network — because the shape to test is not the corpus's but the recurrence's:

  * a family that reads its documents directly, with no sub-family at all;
  * a family that splits into two sub-family leaves;
  * a family that splits into a sub-family which itself splits, so there are two bands to order.

What is asserted is what a real run cannot be allowed to get wrong: **a child is written before its
parent** (a parent's source is its children's abstracts, so the reverse order would summarise nothing),
every assembled record carries its provenance, `rebuild_parent` reproduces each source from the run
plus the cards it was given, and the declared counts gate refuses a tree that has changed shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PASS1 = HERE.parent / "demoE2E/verify/pass1"
#: the reference store is local and never committed; its path comes from the environment, and
#: every test that needs it is skipped where it is absent
STORE = Path(os.environ.get("HARVEST_REFERENCE_STORE", "/nonexistent/saqqara.db"))
#: long enough to clear `THIN_CONTENT`: a card of eleven words is refused as thin by R1, and a test
#: built on one would assert the quality gate rather than the stage under test.
ANSWER = ("La famille couvre les équipements décrits dans les documents rattachés, leurs marques et "
          "leurs emplacements. La maintenance préventive reste annuelle pour chaque ensemble, les "
          "interventions correctives étant déclenchées par appel auprès du titulaire.")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def synthetic(root: Path) -> tuple[Path, Path, dict[str, str]]:
    """A three-family tree and the document cards under it. Returns (families.json, cards, abstracts)."""
    docs = {f"doc{i}": f"Le document {i} décrit ses équipements et leur maintenance." for i in range(1, 9)}
    cards = root / "cards.jsonl"
    cards.write_text("\n".join(json.dumps(
        {"node_id": k, "level": "document", "piece": f"piece {k}", "ok": True, "abstract": v},
        ensure_ascii=False, sort_keys=True) for k, v in docs.items()) + "\n", encoding="utf-8")
    tree = [
        {"family_id": "direct", "name": "Famille directe", "node_id": "fam_direct",
         "documents": 2, "tokens": 1000, "reads_documents_directly": True,
         "node_ids": ["doc1", "doc2"], "sub_families": [], "sub_family_calls": 0},
        {"family_id": "splits", "name": "Famille scindée", "node_id": "fam_splits",
         "documents": 4, "tokens": 90000, "reads_documents_directly": False, "node_ids": [],
         "sub_families": [
             {"node_id": "sf_a", "key_chain": "A", "key": "A", "documents": 2, "tokens": 900,
              "fits_8192": True, "node_ids": ["doc3", "doc4"]},
             {"node_id": "sf_b", "key_chain": "B", "key": "B", "documents": 2, "tokens": 900,
              "fits_8192": True, "node_ids": ["doc5", "doc6"]}],
         "sub_family_calls": 2},
        {"family_id": "deep", "name": "Famille profonde", "node_id": "fam_deep",
         "documents": 2, "tokens": 90000, "reads_documents_directly": False, "node_ids": [],
         "sub_families": [
             {"node_id": "sf_outer", "key_chain": "C", "key": "C", "documents": 2, "tokens": 90000,
              "fits_8192": False, "children": [
                  {"node_id": "sf_inner1", "key_chain": "C/1", "key": "1", "documents": 1,
                   "tokens": 500, "fits_8192": True, "node_ids": ["doc7"]},
                  {"node_id": "sf_inner2", "key_chain": "C/2", "key": "2", "documents": 1,
                   "tokens": 500, "fits_8192": True, "node_ids": ["doc8"]}]}],
         "sub_family_calls": 3},
    ]
    fam = root / "families.json"
    fam.write_text(json.dumps({"tree": tree}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return fam, cards, docs


def drive(root: Path, expect: str = "5,3,1"):
    from ragix_kernels.harvest import core

    fam, cards, docs = synthetic(root)
    out = root / f"out_{expect.replace(',', '_')}"
    out.mkdir(exist_ok=True)
    core.get = lambda host, path, timeout=20.0: (
        {"models": [{"name": core.MODEL, "digest": core.DIGEST + "0" * 8}]} if path == "/api/tags"
        else {"models": []})
    core.post = lambda host, path, body, timeout: {
        "response": ANSWER, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 40}
    code = core.main(["--store", str(STORE), "--lab", str(HERE.parent), "--out", str(out),
                      "--select", "families", "--families", str(fam), "--cards", str(cards),
                      "--expect-families", expect])
    records = [json.loads(line) for line
               in (out / "core_abstracts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return code, records, docs


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    if not STORE.is_file():
        pytest.skip("the store is not on this host")
    return drive(tmp_path_factory.mktemp("families"))


def test_every_node_of_the_tree_was_written(run):
    code, records, _ = run
    assert code == 0
    ids = [r["node_id"] for r in records]
    assert sorted(ids) == sorted(["sf_inner1", "sf_inner2", "sf_outer", "sf_a", "sf_b",
                                  "fam_direct", "fam_splits", "fam_deep", "n_dce_families"])
    assert len(ids) == len(set(ids)), "a node was written twice"


def test_a_child_is_written_before_its_parent(run):
    """The order the data forces: a parent's source is its children's abstracts."""
    order = {r["node_id"]: i for i, r in enumerate(run[1])}
    for child, parent in (("sf_inner1", "sf_outer"), ("sf_inner2", "sf_outer"),
                          ("sf_outer", "fam_deep"), ("sf_a", "fam_splits"), ("sf_b", "fam_splits"),
                          ("fam_direct", "n_dce_families"), ("fam_deep", "n_dce_families")):
        assert order[child] < order[parent], f"{child} was written after {parent}"


def test_the_levels_and_their_contexts(run):
    by_id = {r["node_id"]: r for r in run[1]}
    assert by_id["sf_a"]["level"] == "sub_family" and by_id["sf_a"]["num_ctx"] == 8192
    assert by_id["fam_splits"]["level"] == "family" and by_id["fam_splits"]["num_ctx"] == 16384
    assert by_id["n_dce_families"]["level"] == "dce" and by_id["n_dce_families"]["num_ctx"] == 16384
    assert by_id["sf_a"]["ceiling"] == by_id["fam_splits"]["ceiling"] == 350
    assert by_id["n_dce_families"]["ceiling"] == 800


def test_every_assembled_record_carries_its_provenance(run):
    for record in run[1]:
        assert record.get("assembly") in ("rollup/1", "dce/1"), record["node_id"]
        used, hashes = record.get("children_used"), record.get("children_sha256")
        assert used and hashes and len(used) == len(hashes), record["node_id"]
        assert set(used) <= set(record.get("children") or [])


def test_the_audit_rebuilds_every_node(run):
    """`rebuild_parent` must reproduce each source from the run's records plus the cards it read —
    the cards belong in the map, a leaf sub-family's children being documents and not run nodes."""
    from ragix_kernels.harvest import pass1 as P

    _, records, docs = run
    by_id = {r["node_id"]: r for r in records}
    by_id.update({k: {"node_id": k, "piece": f"piece {k}", "abstract": v} for k, v in docs.items()})
    for record in records:
        assert P.rebuild_parent(record, by_id), record["node_id"]


def test_a_changed_child_makes_the_parent_refuse(run):
    from ragix_kernels.harvest import pass1 as P

    _, records, docs = run
    by_id = {r["node_id"]: r for r in records}
    by_id.update({k: {"node_id": k, "piece": f"piece {k}", "abstract": v} for k, v in docs.items()})
    parent = next(r for r in records if r["node_id"] == "sf_a")
    victim = parent["children_used"][0]
    by_id[victim] = {**by_id[victim], "abstract": by_id[victim]["abstract"] + " Ajout."}
    with pytest.raises(P.ParentDrift):
        P.rebuild_parent(parent, by_id)


def test_the_declared_count_gate_refuses_a_tree_of_another_shape(tmp_path):
    """A tree that has changed shape since it was sized must stop the row, not be summarised anyway."""
    if not STORE.is_file():
        pytest.skip("the store is not on this host")
    with pytest.raises(SystemExit) as exc:
        drive(tmp_path, expect="4,3,1")
    assert exc.value.code == 2
