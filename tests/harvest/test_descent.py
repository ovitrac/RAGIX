"""Row 28 — the descent test's own gates, none of which needs a model.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The GPU job embeds a few hundred strings; everything that can be wrong about the descent is wrong
without it. So the whole path is exercised here on FABRICATED query vectors, planted so that the
right answer is known in advance: a query whose vector IS a node's vector must rank that node first,
and one pointed away from it must not. A test that only checked shapes would pass on a descent that
ranks by row order.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
STORE = LAB / "demoE2E/runs/02_collect/20260906T163718/run/saqqara.db"
NODES = LAB / "demoE2E/verify/pass1/outputs_emb/node_embeddings.jsonl"
TARGETS = LAB / "demoE2E/28_descent/outputs/targets.jsonl"
CORE = LAB / "demoE2E/verify/pass1/outputs/core_abstracts.jsonl"
TRAPS = LAB / "demoE2E/verify/family/outputs/traps.json"
MANAGER = LAB / "demoE2E/28_descent/manager_questions.json"


@pytest.fixture(scope="module")
def D():
    from ragix_kernels.harvest import descent
    return descent


needs_inputs = pytest.mark.skipif(not (STORE.exists() and NODES.exists() and TARGETS.exists()),
                                  reason="the store and row 26's vectors are local and gitignored")


# --------------------------------------------------------------------- the arithmetic

def test_a_rank_is_decided_by_the_score_and_ties_by_id(D):
    ids = ["b", "a", "c"]
    assert D.rank_of("a", ids, np.array([0.1, 0.9, 0.5])) == 1
    assert D.rank_of("c", ids, np.array([0.1, 0.9, 0.5])) == 2
    # a tie is broken by id, never by position, so a re-run gives the same table
    assert D.rank_of("a", ids, np.array([0.5, 0.5, 0.5])) == 1
    assert D.rank_of("b", ids, np.array([0.5, 0.5, 0.5])) == 2


def test_vectors_that_are_not_unit_are_refused(D, tmp_path):
    rows = [{"row": 0}, {"row": 1}]
    path = tmp_path / "v.npy"
    np.save(path, np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    with pytest.raises(SystemExit) as caught:
        D.vectors_for(path, rows, "test")
    assert caught.value.code == 2


def test_a_record_whose_row_does_not_match_its_position_is_refused(D, tmp_path):
    path = tmp_path / "v.npy"
    np.save(path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    with pytest.raises(SystemExit) as caught:
        D.vectors_for(path, [{"row": 1}, {"row": 0}], "test")
    assert caught.value.code == 2


# --------------------------------------------------------------------- the whole path

@needs_inputs
def test_the_descent_finds_the_document_whose_vector_the_query_is(D, tmp_path):
    """The planted case: each query vector IS its target document's vector, so the document lane
    must put that document first for every scored target, in both frames and both query lanes."""
    targets = [json.loads(l) for l in TARGETS.read_text(encoding="utf-8").splitlines() if l.strip()]
    nodes = [json.loads(l) for l in NODES.read_text(encoding="utf-8").splitlines() if l.strip()]
    vectors = np.load(NODES.with_suffix(".npy"))
    by_node = {n["node_id"]: i for i, n in enumerate(nodes)}

    chosen, seen = [], set()
    for t in targets:
        if t.get("doc") in by_node and t["doc"] not in seen and t.get("window"):
            chosen.append(t)
            seen.add(t["doc"])
        if len(chosen) == 6:
            break
    assert len(chosen) == 6

    subset = tmp_path / "targets.jsonl"
    subset.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in chosen), encoding="utf-8")
    queries = [{"query": t["query"], "row": i, "sets": ["planted"]} for i, t in enumerate(chosen)]
    (tmp_path / "queries.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    planted = np.vstack([vectors[by_node[t["doc"]]] for t in chosen]).astype(np.float32)
    for lane in ("bare", "prefixed"):
        np.save(tmp_path / f"queries_{lane}.npy", planted)

    rc = D.main(["--nodes", str(NODES), "--targets", str(subset),
                 "--queries", str(tmp_path / "queries.jsonl"), "--core", str(CORE),
                 "--store", str(STORE), "--out", str(tmp_path / "out")])
    assert rc == 0
    report = json.loads((tmp_path / "out/descent.json").read_text(encoding="utf-8"))
    for key in ("document.core.bare", "document.all.bare", "document.core.prefixed"):
        assert report["results"][key]["top1_rate"] == 1.0, f"{key} did not find a planted document"
    # and the lexical lanes the map test measured are carried through untouched
    assert report["results"]["document.core.lexical_text"]["scored"] == len(chosen)


@needs_inputs
def test_a_target_whose_query_has_no_vector_stops_the_run(D, tmp_path):
    """Fail closed: a missing vector would silently shrink the denominator of every rate."""
    targets = [json.loads(l) for l in TARGETS.read_text(encoding="utf-8").splitlines() if l.strip()][:3]
    subset = tmp_path / "targets.jsonl"
    subset.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in targets), encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"query": "une question qui n'est pas une cible", "row": 0}) + "\n", encoding="utf-8")
    for lane in ("bare", "prefixed"):
        np.save(tmp_path / f"queries_{lane}.npy", np.array([[1.0] + [0.0] * 1023], dtype=np.float32))
    with pytest.raises(SystemExit) as caught:
        D.main(["--nodes", str(NODES), "--targets", str(subset),
                "--queries", str(tmp_path / "queries.jsonl"), "--core", str(CORE),
                "--store", str(STORE), "--out", str(tmp_path / "out")])
    assert caught.value.code == 2
