"""`--workers N` must change the wall clock and nothing else (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The claim under test is the one the coordinating seat set: **the same record set as one worker**. It is
asserted against a fake server, so no model, no GPU and no network are involved and the comparison is
exact rather than statistical — with a real model the answers would differ between runs and nothing
could be concluded.

Three things are checked, because "the same" has three distinct failure modes here:
  * the same nodes, each written **once** — a missed lock writes a node twice or drops one;
  * the same content per node, timing aside — `seconds` is wall clock and is expected to differ;
  * the same summary, although the file's line order differs — at more than one worker the records are
    written in completion order, which is exactly why `core.py` sorts before reporting.
The fake server sleeps a little, so the workers really do overlap; without that the pool would finish
each call before starting the next and the test would prove nothing about concurrency.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PASS1 = HERE.parent / "demoE2E/verify/pass1"
#: the reference store is local and never committed; its path comes from the environment, and
#: every test that needs it is skipped where it is absent
STORE = Path(os.environ.get("HARVEST_REFERENCE_STORE", "/nonexistent/saqqara.db"))
PIECE = ("CCTP REF-1 - Annexe 4 - Fichiers de quantification adhérent/Catalogue des besoins/"
         "Catalogue des besoins (12) Equipements de manutention non medicaux et engins (v2).xlsx")
TIMED = ("seconds", "made_on")


def drive(tmp_path: Path, workers: int, paths: Path) -> tuple[list[dict], dict, int]:
    from ragix_kernels.harvest import core

    out = tmp_path / f"w{workers}"
    out.mkdir()
    # the SAME selection file for both runs: the DCE node id is built from the --select string, so a
    # per-run path would make the two runs differ in that id alone and the comparison would be void.
    core.get = lambda host, path, timeout=20.0: (
        {"models": [{"name": core.MODEL, "digest": core.DIGEST + "0" * 8}]} if path == "/api/tags"
        else {"models": []})

    # concurrency is COUNTED, not timed: the server records how many calls are in flight at once, so
    # the claim is asserted directly instead of being inferred from a wall clock that a loaded machine
    # can make say anything.
    flight = {"now": 0, "peak": 0}
    gate = threading.Lock()

    def fake_post(host, path, body, timeout):
        with gate:
            flight["now"] += 1
            flight["peak"] = max(flight["peak"], flight["now"])
        try:
            time.sleep(0.05)                  # long enough that a second worker can arrive
            return {"response": "Le lot couvre les engins de manutention. La visite est annuelle.",
                    "done_reason": "stop", "prompt_eval_count": 120, "eval_count": 40}
        finally:
            with gate:
                flight["now"] -= 1

    core.post = fake_post
    code = core.main(["--store", str(STORE), "--lab", str(HERE.parent), "--out", str(out),
                      "--select", f"paths:{paths}", "--sheets-as-nodes", "--expect", "5,1",
                      "--workers", str(workers)])
    assert code == 0
    records = [json.loads(line) for line
               in (out / "core_abstracts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads((out / "core_summary.json").read_text(encoding="utf-8"))
    return records, summary, flight["peak"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not STORE.is_file():
        pytest.skip("the store is not on this host")
    root = tmp_path_factory.mktemp("workers")
    paths = root / "paths.txt"
    paths.write_text(PIECE + "\n", encoding="utf-8")
    return {n: drive(root, n, paths) for n in (1, 4)}


def test_the_same_nodes_each_written_once(runs):
    one, four = [r[0] for r in (runs[1], runs[4])]
    ids_one = [r["node_id"] for r in one]
    ids_four = [r["node_id"] for r in four]
    assert len(ids_four) == len(set(ids_four)), "a node was written twice by the pool"
    assert sorted(ids_one) == sorted(ids_four)


def test_the_same_record_per_node_timing_aside(runs):
    by_one = {r["node_id"]: r for r in runs[1][0]}
    by_four = {r["node_id"]: r for r in runs[4][0]}
    for node_id, record in by_one.items():
        mine = {k: v for k, v in record.items() if k not in TIMED}
        theirs = {k: v for k, v in by_four[node_id].items() if k not in TIMED}
        assert mine == theirs, f"{node_id[:16]} differs between one worker and four"


def test_the_summary_does_not_depend_on_completion_order(runs):
    def strip(summary: dict) -> dict:
        out = {k: v for k, v in summary.items() if k not in ("run", "started", "finished")}
        out["verdict"] = {k: v for k, v in out["verdict"].items() if k != "wall_s"}
        out["by_level"] = {lv: {k: v for k, v in d.items() if not k.endswith("_s")}
                           for lv, d in out["by_level"].items()}
        return out

    assert strip(runs[1][1]) == strip(runs[4][1])


def test_the_pool_really_overlapped(runs):
    """Otherwise the three assertions above hold trivially and prove nothing about concurrency."""
    assert runs[1][2] == 1, f"one worker had {runs[1][2]} calls in flight at once"
    assert runs[4][2] > 1, "four workers never had two calls in flight: the pool did not fan out"
