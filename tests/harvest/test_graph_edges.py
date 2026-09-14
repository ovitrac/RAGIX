"""Row 27: the graph, materialised in the sidecar's derived_edges from what already exists.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Six kinds, each by a declared rule over a named source, each edge carrying that rule and the source's
sha256: contains (the store's own tree), summarises (pass 1's children), governed_by (row 14's two
authorities over the deadline claims), shares_template (M2b's clusters), contradicts (M2d's traps — every
one of the sixteen, by construction), references (the reference grammar, resolved to documents where the
reference names one, and counted by reason where it does not). Append-only: a second run adds nothing, and
an edge already there keeps the run that wrote it.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from ragix_kernels.harvest.derived import DerivedStore

LAB = Path(__file__).resolve().parents[1]
STORE = LAB / "demoE2E/runs/02_collect/20260906T163718/run/saqqara.db"
needs_store = pytest.mark.skipif(not STORE.exists(), reason="the frozen store is local and gitignored")


def _load():
    from ragix_kernels.harvest import build_edges
    return build_edges


def _build(tmp_path, sidecar=None, run_id="r1"):
    side = sidecar or tmp_path / "derived.sqlite"
    rc = _load().main([
        "--store", str(STORE), "--sidecar", str(side), "--run-id", run_id, "--out", str(tmp_path / "out"),
        "--claims", str(LAB / "demoE2E/13_wp3d_32a/outputs/claims.jsonl"),
        "--authority", str(LAB / "demoE2E/14_wp3d_34a/outputs"),
        "--pass1", str(LAB / "demoE2E/verify/pass1/outputs/core_abstracts.jsonl"),
        "--template", str(LAB / "demoE2E/verify/family/outputs/template.json"),
        "--traps", str(LAB / "demoE2E/verify/family/outputs/traps.json")])
    return rc, side, json.loads((tmp_path / "out/edges_summary.json").read_text(encoding="utf-8"))


@needs_store
def test_every_kind_by_construction(tmp_path):
    rc, _, s = _build(tmp_path)
    assert rc == 0
    c = s["counts"]
    assert c["contains"] == 52724          # every chunk, from its document or from its parent
    assert c["summarises"] == 243          # pass 1's 25 parents over their children
    assert c["governed_by"] == 48          # 24 deadline claims under row 14's two authorities
    assert c["shares_template"] == 736     # M2b: 736 members across 38 clusters
    assert c["contradicts"] == 59          # 57 spans of 14 traps, and 2 recorded readings
    assert len(s["trap_hubs"]) == 16       # all sixteen traps, by construction
    r = s["references"]
    assert r["resolved"] > 0 and c["references"] == r["resolved"]
    assert r["resolved"] + r["self"] + sum(r["unresolved"].values()) == r["total"]   # nothing dropped silently


@needs_store
def test_every_edge_carries_its_rule_and_its_source_hash(tmp_path):
    _, side, _ = _build(tmp_path)
    con = sqlite3.connect(side)
    bad = [p for (p,) in con.execute("select payload_json from derived_edges")
           if not {"rule", "source_sha256"} <= set(json.loads(p))]
    assert not bad, bad[:2]


@needs_store
def test_a_second_run_appends_nothing_and_an_earlier_edge_keeps_its_run(tmp_path):
    side = tmp_path / "derived.sqlite"
    earlier = DerivedStore(str(side), "/store", "x" * 64, run_id="block-A")
    earlier.write_edge("contains", "n_dce", "n_CCTP_x", {"rule": "block A's pyramid"})
    earlier.commit(); earlier.close()
    _build(tmp_path, side, "r1")
    first = sqlite3.connect(side).execute("select count(*) from derived_edges").fetchone()[0]
    _build(tmp_path, side, "r2")
    con = sqlite3.connect(side)
    assert con.execute("select count(*) from derived_edges").fetchone()[0] == first
    assert con.execute("select run_id from derived_edges where source_id='n_dce'").fetchone()[0] == "block-A"
    assert con.execute("select count(*) from derived_edges where run_id='r2'").fetchone()[0] == 0
