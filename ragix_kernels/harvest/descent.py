#!/usr/bin/env python3
"""demoE2E 28 — the descent: question → document → window → leaf, and what each step costs.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE RUN (RUNBOOK row 28). CPU on the master, no model call: the node vectors are
row 26's, the query vectors row 28's own GPU job, the targets and their placement the map test's
(`map_test.py --emit-targets`) — this file does not re-derive M1 or M2, it joins them, so a
divergence between the two rows is impossible by construction rather than by care.

THE CLAIM UNDER TEST, WP §8.5(b) as the two-pass architecture states it: a coarse search over node
cards finds the PLACE where the evidence is, well enough that pass 2 can be pointed at it. The map
test answered that lexically and the answer was poor at document grain (abstract lane 0.108 of 798
targets in the top 3, text lane 0.241). Row 28 asks whether the embedding of the same abstract does
better, and it reports the same numbers beside it so the comparison is on one set of targets.

THE LEVELS
  document  cosine between the query and each document node's abstract vector. Reported in two
            frames, because they answer different questions: `core` ranks among the 26 contractual
            documents (the map test's frame, comparable with its numbers), `all` ranks among every
            document node embedded — core plus the 431 of row 24 — which is the frame a real
            question arrives in, with no one having pre-selected the core.
  window    two lanes. `text` is the map test's lexical lane, read from its own emitted ranks.
            `abstract_vector` ranks the document's windows by cosine on their pass-1 abstracts.
  leaf      inside the target's window, its leaves ranked by the map test's own lexical scorer
            (`Lane`, imported, never copied). The leaf that holds the target's offset is the hit.

THE LANES OF THE QUERY ITSELF
  bare / prefixed — `snowflake-arctic-embed2` is asymmetric: documents plain, queries prefixed
  `query: `. Both are measured rather than assumed, and both are reported.

THE TWO CONTROLS the row declares
  children_mean   the document's vector replaced by the mean of its windows' vectors, renormalised.
                  If this beats the abstract vector, what pass 1 buys at document grain is not the
                  abstract but the aggregation, and the abstract's cost is not paid back there.
  lexical_equal   the map test's text lane over documents, at the same candidate budget — the
                  baseline that makes a number mean something (ORIENTATION: every advanced method
                  ships with its baseline).

Exit 0 always: this is a measurement. It refuses (exit 2) only when an input is missing, when a
target's query has no vector, or when a vector set does not match its record — never because a
number is low. The exit thresholds are proposed in the RUNBOOK row for the lead to set.

    python3 demoE2E/28_descent/descent.py --nodes ... --targets ... --queries ... --store ... --out ...
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .map_test import Lane  # the map test's own lexical scorer, imported not copied

TOP = 3


def refuse(message: str) -> None:
    sys.stderr.write(f"descent: REFUSED — {message}\n")
    raise SystemExit(2)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        refuse(f"{path} is missing")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def vectors_for(path: Path, rows: list[dict], what: str) -> np.ndarray:
    if not path.is_file():
        refuse(f"{path} is missing")
    vectors = np.load(path)
    if vectors.shape[0] != len(rows):
        refuse(f"{what}: {vectors.shape[0]} vectors for {len(rows)} records")
    norms = np.linalg.norm(vectors, axis=1)
    if float(np.abs(norms - 1.0).max()) > 1e-3:
        refuse(f"{what}: the vectors are not L2-normalised, so a dot product is not a cosine")
    for position, row in enumerate(rows):
        if int(row.get("row", position)) != position:
            refuse(f"{what}: record {position} says row {row.get('row')} — the join would be wrong")
    return vectors.astype(np.float32)


def rank_of(target_id: str, candidates: list[str], scores: np.ndarray) -> int:
    """The rank of one candidate under a score, ties broken by id so a run repeats exactly."""
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
    for position, index in enumerate(order, start=1):
        if candidates[index] == target_id:
            return position
    return 0


def bucket() -> dict:
    return {"scored": 0, "top3": 0, "top1": 0, "ranks": []}


def record(b: dict, rank: int) -> None:
    b["scored"] += 1
    b["ranks"].append(rank)
    b["top3"] += rank <= TOP
    b["top1"] += rank == 1


def rates(b: dict) -> dict:
    if not b["scored"]:
        return {"scored": 0, "top3_rate": None, "top1_rate": None, "rank_median": None}
    return {"scored": b["scored"], "top3": b["top3"], "top1": b["top1"],
            "top3_rate": round(b["top3"] / b["scored"], 4),
            "top1_rate": round(b["top1"] / b["scored"], 4),
            "rank_median": statistics.median(b["ranks"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="row 28 — the descent test (CPU)")
    ap.add_argument("--nodes", required=True, type=Path, help="row 26's node_embeddings.jsonl")
    ap.add_argument("--node-vectors", type=Path, default=None)
    ap.add_argument("--targets", required=True, type=Path, help="map_test.py --emit-targets")
    ap.add_argument("--queries", required=True, type=Path, help="row 28's queries.jsonl")
    ap.add_argument("--core", required=True, type=Path, help="row 22's core_abstracts.jsonl")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    nodes = read_jsonl(a.nodes)
    node_vectors = vectors_for(a.node_vectors or a.nodes.with_suffix(".npy"), nodes, "node vectors")
    queries = read_jsonl(a.queries)
    lanes = {}
    for lane in ("bare", "prefixed"):
        lanes[lane] = vectors_for(a.queries.parent / f"queries_{lane}.npy", queries, f"{lane} queries")
    by_query = {q["query"]: q["row"] for q in queries}
    targets = read_jsonl(a.targets)

    # the document frames: core is row 22's own 26, all is every document node embedded
    core_ids = {r["node_id"] for r in read_jsonl(a.core) if r.get("level") == "document"}
    doc_rows = [(i, n["node_id"]) for i, n in enumerate(nodes) if n.get("level") == "document"]
    win_rows = [(i, n["node_id"]) for i, n in enumerate(nodes) if n.get("level") == "window"]
    frames = {"core": [(i, nid) for i, nid in doc_rows if nid in core_ids], "all": doc_rows}
    if not frames["core"]:
        refuse("none of row 22's document nodes is in the embedded set: the frames cannot be built")

    import sqlite3
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

    # the control: a document's vector as the mean of its windows' vectors, renormalised
    win_index = {nid: i for i, nid in win_rows}
    children_mean: dict[str, np.ndarray] = {}
    for i, doc_id in doc_rows:
        kids = [win_index[cid] for (cid,) in
                db.execute("select chunk_id from chunks where parent_id=? and level=1", (doc_id,))
                if cid in win_index]
        if kids:
            mean = node_vectors[kids].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm:
                children_mean[doc_id] = (mean / norm).astype(np.float32)

    results: dict[str, dict] = defaultdict(bucket)
    per_shape: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(bucket))
    leaf_cache: dict[str, Lane] = {}
    leaf_spans: dict[str, list[tuple[str, int, int]]] = {}
    missing_vectors = 0

    for t in targets:
        if t.get("out_of_frame") or t.get("unplaceable") or not t.get("doc"):
            continue
        row = by_query.get(" ".join(t["query"].split()))
        if row is None:
            missing_vectors += 1
            continue
        shape = t["set"]

        # ---- document, both query lanes and both frames, plus the two controls
        for lane, matrix in lanes.items():
            q = matrix[row]
            for frame, members in frames.items():
                ids = [nid for _, nid in members]
                scores = node_vectors[[i for i, _ in members]] @ q
                key = f"document.{frame}.{lane}"
                record(results[key], rank_of(t["doc"], ids, scores))
                record(per_shape[key][shape], rank_of(t["doc"], ids, scores))
            ids = [nid for _, nid in frames["core"] if nid in children_mean]
            if ids:
                scores = np.vstack([children_mean[nid] for nid in ids]) @ q
                key = f"document.core.{lane}.children_mean"
                record(results[key], rank_of(t["doc"], ids, scores))
                record(per_shape[key][shape], rank_of(t["doc"], ids, scores))

        # ---- the lexical lanes the map test already measured, on these same targets
        for lane_name, ranks in (t.get("lanes") or {}).items():
            if ranks.get("doc_rank"):
                record(results[f"document.core.lexical_{lane_name}"], ranks["doc_rank"])
                record(per_shape[f"document.core.lexical_{lane_name}"][shape], ranks["doc_rank"])
            if ranks.get("win_rank"):
                record(results[f"window.lexical_{lane_name}"], ranks["win_rank"])
                record(per_shape[f"window.lexical_{lane_name}"][shape], ranks["win_rank"])

        # ---- window by the abstract's vector, inside the target's own document
        if t.get("window"):
            kids = [(win_index[cid], cid) for (cid,) in
                    db.execute("select chunk_id from chunks where parent_id=? and level=1 order by seq",
                               (t["doc"],)) if cid in win_index]
            if kids and t["window"] in [cid for _, cid in kids]:
                ids = [cid for _, cid in kids]
                for lane, matrix in lanes.items():
                    scores = node_vectors[[i for i, _ in kids]] @ matrix[row]
                    key = f"window.abstract_vector.{lane}"
                    record(results[key], rank_of(t["window"], ids, scores))
                    record(per_shape[key][shape], rank_of(t["window"], ids, scores))

        # ---- leaf, inside the target's window, by the map test's own lexical scorer
        window = t.get("window")
        if window and (t.get("char_start") is not None or t.get("byte_start") is not None):
            if window not in leaf_cache:
                rows = db.execute("select chunk_id, text from chunks where parent_id=? and level=0 "
                                  "order by seq", (window,)).fetchall()
                if not rows:
                    rows = db.execute(
                        "select c.chunk_id, c.text from chunks c where c.parent_id=(select parent_id from "
                        "chunks where chunk_id=?) and c.level=0 order by c.seq", (window,)).fetchall()
                leaf_cache[window] = Lane({cid: text for cid, text in rows}) if rows else None
                position, spans = 0, []
                for cid, text in rows:
                    spans.append((cid, position, position + len(text)))
                    position += len(text) + 1
                leaf_spans[window] = spans
            lane_obj = leaf_cache.get(window)
            if lane_obj is not None and lane_obj.ids:
                roll = db.execute("select text from chunks where chunk_id=?", (t["doc"],)).fetchone()
                offset = t.get("char_start")
                if offset is None and roll is not None:
                    raw = roll[0].encode("utf-8")
                    offset = len(raw[:int(t["byte_start"])].decode("utf-8", "ignore"))
                held = [cid for cid, c0, c1 in leaf_spans[window] if c0 <= (offset or 0) < c1]
                if held:
                    order = {cid: i for i, (cid, _, _) in enumerate(leaf_spans[window])}
                    ranked = lane_obj.rank(t["query"], order)
                    rank = ranked.index(held[0]) + 1
                    record(results["leaf.lexical_text"], rank)
                    record(per_shape["leaf.lexical_text"][shape], rank)

    db.close()
    if missing_vectors:
        refuse(f"{missing_vectors} targets have no query vector: run the embedding job first")

    report = {
        "declared": {"top": TOP, "targets": len(targets), "queries": len(queries),
                     "nodes": {"documents": len(doc_rows), "windows": len(win_rows),
                               "core_documents": len(frames["core"])},
                     "levels": ["document", "window", "leaf"],
                     "query_lanes": sorted(lanes),
                     "controls": ["children_mean", "lexical_text (equal budget)"],
                     "note": "pass-1 abstracts are orientation, never evidence (WP §8.1); the manager "
                             "set has no gold and is not scored here"},
        "results": {k: rates(v) for k, v in sorted(results.items())},
        "per_shape": {k: {s: rates(b) for s, b in sorted(v.items())} for k, v in sorted(per_shape.items())},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "descent.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
                                        + "\n", encoding="utf-8")

    lines = ["# The descent — question → document → window → leaf", "",
             f"{len(targets)} targets · {len(queries)} distinct queries · "
             f"{len(frames['core'])} core documents, {len(doc_rows)} documents embedded, "
             f"{len(win_rows)} windows", "",
             "| step | lane | scored | top 3 | top 1 | median rank |", "|---|---|---|---|---|---|"]
    for key, r in report["results"].items():
        step, _, lane = key.partition(".")
        lines.append(f"| {step} | {lane} | {r['scored']} | {r['top3_rate']} | {r['top1_rate']} | "
                     f"{r['rank_median']} |")
    lines += ["", "## Per query shape, top 3", "", "| lane | " +
              " | ".join(sorted({s for v in report["per_shape"].values() for s in v})) + " |",
              "|---" * (1 + len({s for v in report["per_shape"].values() for s in v})) + "|"]
    shapes = sorted({s for v in report["per_shape"].values() for s in v})
    for key, v in report["per_shape"].items():
        lines.append(f"| {key} | " + " | ".join(str((v.get(s) or {}).get("top3_rate")) for s in shapes) + " |")
    (a.out / "descent.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": report["results"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
