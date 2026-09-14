#!/usr/bin/env python3
"""M11 — the map test: can a search over pass-1 cards find the place where the evidence is? (seat S4)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4), as WP §8.5(b) states the falsifier, on the coordinating seat's
word of 2026-09-11. The claim under test is the one the whole two-pass architecture rests on: if a coarse
search over node cards does not find the right *place*, pass 2 cannot be pointed at the right node and the
architecture collapses into harvesting everything. Nothing here calls a model; the cards are read from
pass 1's own output and the gold from files sealed before it ran.

M0 the cards. `core_abstracts.jsonl` (pass 1, row 22): one abstract per window, per document and for the
   DCE node. **The card in this run is the abstract alone** — the registers' values and the TF-IDF
   highlights of WP §8.2 are not in it — so this is the weakest form of the test, and a failure here does
   not condemn the node card, only the abstract as a locator.
M1 the targets, each read from a file sealed before pass 1 and each carrying the chunk id of the object
   that holds it:
   (a) `16_harvest_bakeoff/labels_aprime.yaml` — 30 objects, its values' `span_raw`;
   (b) `16_harvest_bakeoff/labels_bprime.yaml` — 25 objects, idem;
   (c) `13_wp3d_32a/outputs/claims.jsonl` — the claims' `value.raw`, one target per source span;
   (d) `verify/family/outputs/traps.json` — one target per located span, the query being the trap's own
       French question, which is what a buyer would actually ask.
   A target whose object is **not a node of this run** (the gold pools reach beyond the contractual core)
   is counted `out_of_frame` and never as a miss: it has no card to be found in.
M2 placement. A target names a chunk. Its **window** is that chunk when it is one of the run's windows;
   otherwise the window of the same document whose declared span (`meta.part.span`) contains the target's
   offset in the document roll-up — a leaf is placed by the sum of the preceding leaves' lengths plus one
   newline each, a roll-up span by its own byte range converted to characters. A target that cannot be
   placed in any window is counted `unplaceable` and reported, never dropped.
M3 the search, deterministic and lexical. Query and card are tokenised on non-letters, case-folded,
   accents kept, tokens of one character dropped; a token's weight is its IDF over the card set searched
   (natural log of (N + 1) / (n + 1) + 1); a card's score is the sum of the weights of the query tokens it
   contains, divided by the square root of the card's token count, so a long abstract does not win by
   length alone. Ties are broken by the node's order in the document, then by node id: no randomness, and
   the same inputs give the same ranks.
M4 the two questions, reported separately, exactly as WP §8.5(b) asks:
   (a) **document**: is the target's document in the top 3 of the run's documents, searching the document
       cards?
   (b) **window**: is the target's window in the top 3 of its own document's windows, searching that
       document's window cards?
   Reported per gold set and in total, with the top-1 rate and the median rank beside them, because a
   top-3 rate alone hides whether the right card was first or third.
M5 the baseline that makes the number mean something. The same two questions answered by searching the
   **objects' own text** instead of their abstracts (the store's text, no model involved). The abstract
   lane must be read against it: if the text lane already finds the place, pass 1 buys navigation and
   compression, not retrieval; if the abstract lane is close to the text lane, the abstracts carry the
   signal. Both numbers are printed; neither is adjusted.
Exit: 0 always — this is a measurement, not a gate. A refusal (exit 2) happens only when an input is
missing or a card set is empty, never because a number is low.
Output: `map_test.json` (the declaration, per-set and total rates, the misses with their ranks) and
`map_test.md` for reading.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import yaml

STORE_SHA_PREFIX = "53ff3f20f655ad66"
TOP = 3
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def refuse(msg: str) -> NoReturn:
    sys.stderr.write(f"map_test: REFUSED — {msg}\n")
    sys.exit(2)


def tokens(text: str) -> list[str]:
    return [t for t in (w.casefold() for w in TOKEN.findall(text or "")) if len(t) > 1]


class Lane:
    """One searchable set of cards: id -> text, with IDF over the set (M3)."""

    def __init__(self, cards: dict[str, str]) -> None:
        self.ids = sorted(cards)
        self.toks = {cid: Counter(tokens(cards[cid])) for cid in self.ids}
        n = len(self.ids)
        df: Counter = Counter()
        for cid in self.ids:
            df.update(set(self.toks[cid]))
        self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in df}
        self.norm = {cid: math.sqrt(sum(self.toks[cid].values())) or 1.0 for cid in self.ids}

    def rank(self, query: str, order: dict[str, int]) -> list[str]:
        q = set(tokens(query))
        scored = []
        for cid in self.ids:
            score = sum(self.idf.get(t, 0.0) for t in q if t in self.toks[cid]) / self.norm[cid]
            scored.append((-score, order.get(cid, 0), cid))
        scored.sort()
        return [cid for _, _, cid in scored]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M11 — the map test (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--lab", required=True, type=Path)
    ap.add_argument("--cards", required=True, type=Path, help="core_abstracts.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--emit-targets", type=Path, default=None,
                    help="write every target with M2's placement and each lane's rank, as JSONL, so a "
                         "later row measures another lane over these targets rather than re-deriving them")
    a = ap.parse_args(argv)

    for p in (a.store, a.cards):
        if not p.exists():
            refuse(f"{p} missing")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    rows = [json.loads(line) for line in a.cards.read_text(encoding="utf-8").splitlines() if line.strip()]
    cards = {r["node_id"]: r for r in rows if r.get("ok") and r.get("abstract")}
    windows = {k: v for k, v in cards.items() if v["level"] == "window"}
    documents = {k: v for k, v in cards.items() if v["level"] == "document"}
    if not windows or not documents:
        refuse(f"the card set is empty: {len(windows)} windows, {len(documents)} documents")

    # the store's own frame: every window with its document, its declared span and its order
    win_of_doc: dict[str, list[tuple[str, int, int, int]]] = {}   # roll_id -> (cid, c0, c1, seq)
    doc_of_win: dict[str, str] = {}
    roll_text: dict[str, str] = {}
    leaf_offset: dict[str, tuple[str, int]] = {}
    for roll_id in documents:
        row = db.execute("select text, doc_id from chunks where chunk_id=?", (roll_id,)).fetchone()
        if row is None:
            refuse(f"document node {roll_id[:12]} is not a chunk of the store")
        roll_text[roll_id] = row[0]
        doc_id = row[1]
        seq = 0
        for cid, meta in db.execute("select chunk_id, meta_json from chunks where parent_id=? and level=1 "
                                    "order by seq", (roll_id,)):
            span = (json.loads(meta or "{}").get("part") or {}).get("span")
            if span:
                win_of_doc.setdefault(roll_id, []).append((cid, int(span[0]), int(span[1]), seq))
                doc_of_win[cid] = roll_id
                seq += 1
        pos = 0
        for cid, length in db.execute("select chunk_id, length(text) from chunks where doc_id=? and level=0 "
                                      "order by seq", (doc_id,)):
            leaf_offset[cid] = (roll_id, pos)
            pos += length + 1

    def place(chunk_id: str, char_start: int = 0, byte_start: int | None = None) -> dict[str, Any]:
        """M2: the (document, window) a target belongs to, or the reason it has none."""
        if chunk_id in doc_of_win:
            return {"doc": doc_of_win[chunk_id], "window": chunk_id}
        if chunk_id in win_of_doc or chunk_id in roll_text:      # the target names a roll-up
            roll = chunk_id
            offset = char_start
            if byte_start is not None:
                offset = len(roll_text[roll].encode("utf-8")[:byte_start].decode("utf-8", "ignore"))
        elif chunk_id in leaf_offset:                            # the target names a leaf
            roll, base = leaf_offset[chunk_id]
            offset = base + char_start
        else:
            return {"out_of_frame": True}
        for cid, c0, c1, _ in win_of_doc.get(roll, []):
            if c0 <= offset < c1:
                return {"doc": roll, "window": cid}
        return {"doc": roll, "window": None, "unplaceable": True}

    # ---------------------------------------------------------------- the targets (M1)
    targets: list[dict[str, Any]] = []
    for name, key in (("A-prime", "labels_aprime.yaml"), ("B-prime", "labels_bprime.yaml")):
        path = a.lab / "demoE2E/16_harvest_bakeoff" / key
        if not path.exists():
            refuse(f"{path} missing")
        for obj in yaml.safe_load(path.read_text(encoding="utf-8")).get("objects") or []:
            for value in obj.get("values") or []:
                query = re.sub(r"\s+", " ", str(value.get("span_raw") or value.get("value") or "")).strip()
                if query:
                    targets.append({"set": name, "chunk_id": obj["chunk_id"], "query": query})
    claims_path = a.lab / "demoE2E/13_wp3d_32a/outputs/claims.jsonl"
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        claim = json.loads(line)
        raw = ((claim.get("value") or {}).get("raw") or "").strip()
        for source in (claim.get("provenance") or {}).get("sources") or []:
            if isinstance(source, dict) and source.get("chunk_id") and raw:
                targets.append({"set": "row 13 claims", "chunk_id": source["chunk_id"], "query": raw,
                                "char_start": int(source.get("char_start") or 0)})
    traps = json.loads((a.lab / "demoE2E/verify/family/outputs/traps.json").read_text(encoding="utf-8"))
    for trap in traps["traps"]:
        for span in trap.get("spans") or []:
            targets.append({"set": "traps", "chunk_id": span["chunk_id"], "query": trap["question_fr"],
                            "byte_start": int(span["bytes"][0]), "trap": trap["id"]})

    # ---------------------------------------------------------------- the two lanes (M3, M5)
    doc_order = {cid: i for i, cid in enumerate(sorted(documents))}
    lanes = {
        "abstract": {"documents": Lane({k: v["abstract"] for k, v in documents.items()}),
                     "windows": {roll: Lane({cid: windows[cid]["abstract"]
                                             for cid, _, _, _ in wins if cid in windows})
                                 for roll, wins in win_of_doc.items()}},
        "text": {"documents": Lane({k: roll_text[k] for k in documents}),
                 "windows": {roll: Lane({cid: db.execute("select text from chunks where chunk_id=?",
                                                         (cid,)).fetchone()[0]
                                         for cid, _, _, _ in wins if cid in windows})
                             for roll, wins in win_of_doc.items()}},
    }
    win_order = {roll: {cid: seq for cid, _, _, seq in wins} for roll, wins in win_of_doc.items()}

    results: dict[str, Any] = {}
    misses: list[dict[str, Any]] = []
    #: --emit-targets: every target with the placement THIS file computed and the rank each lane gave
    #: it, so a later row (28, the descent) measures a third lane over exactly these targets and this
    #: placement rather than re-deriving M1 and M2 and drifting from them. Added by the coding seat
    #: 2026-09-11, flagged for this seat's review; nothing above it changes.
    emitted: dict[int, dict[str, Any]] = {}
    for lane_name, lane in lanes.items():
        per_set: dict[str, dict[str, Any]] = {}
        for index, target in enumerate(targets):
            placed = place(target["chunk_id"], target.get("char_start", 0), target.get("byte_start"))
            bucket = per_set.setdefault(target["set"], {"targets": 0, "out_of_frame": 0, "unplaceable": 0,
                                                        "doc_top3": 0, "doc_top1": 0, "win_top3": 0,
                                                        "win_top1": 0, "scored": 0, "doc_ranks": [],
                                                        "win_ranks": []})
            bucket["targets"] += 1
            row = emitted.setdefault(index, {"set": target["set"], "query": target["query"],
                                             "chunk_id": target["chunk_id"], "trap": target.get("trap"),
                                             "char_start": target.get("char_start"),
                                             "byte_start": target.get("byte_start"),
                                             "doc": placed.get("doc"), "window": placed.get("window"),
                                             "out_of_frame": bool(placed.get("out_of_frame")),
                                             "unplaceable": bool(placed.get("unplaceable")), "lanes": {}})
            if placed.get("out_of_frame"):
                bucket["out_of_frame"] += 1
                continue
            if placed.get("unplaceable"):
                bucket["unplaceable"] += 1
                continue
            bucket["scored"] += 1
            ranked_docs = lane["documents"].rank(target["query"], doc_order)
            d_rank = ranked_docs.index(placed["doc"]) + 1
            bucket["doc_ranks"].append(d_rank)
            bucket["doc_top3"] += d_rank <= TOP
            bucket["doc_top1"] += d_rank == 1
            row["lanes"].setdefault(lane_name, {})["doc_rank"] = d_rank
            wl = lane["windows"].get(placed["doc"])
            if wl is None or placed["window"] not in wl.ids:
                continue
            ranked_wins = wl.rank(target["query"], win_order.get(placed["doc"], {}))
            w_rank = ranked_wins.index(placed["window"]) + 1
            bucket["win_ranks"].append(w_rank)
            bucket["win_top3"] += w_rank <= TOP
            bucket["win_top1"] += w_rank == 1
            row["lanes"].setdefault(lane_name, {})["win_rank"] = w_rank
            row["windows_in_document"] = len(wl.ids)
            if lane_name == "abstract" and (d_rank > TOP or w_rank > TOP) and len(misses) < 40:
                misses.append({"set": target["set"], "query": target["query"][:70],
                               "document": placed["doc"][:12], "doc_rank": d_rank,
                               "window": (placed["window"] or "")[:12], "window_rank": w_rank,
                               "windows_in_document": len(wl.ids)})
        totals: dict[str, Any] = {"targets": 0, "scored": 0, "out_of_frame": 0, "unplaceable": 0,
                                  "doc_top3": 0, "doc_top1": 0, "win_top3": 0, "win_top1": 0}
        dr: list[int] = []
        wr: list[int] = []
        for bucket in per_set.values():
            for k in totals:
                totals[k] += bucket[k]
            dr += bucket["doc_ranks"]
            wr += bucket["win_ranks"]
            bucket["doc_top3_rate"] = round(bucket["doc_top3"] / bucket["scored"], 4) if bucket["scored"] else None
            bucket["win_top3_rate"] = round(bucket["win_top3"] / len(bucket["win_ranks"]), 4) if bucket["win_ranks"] else None
            bucket["doc_rank_median"] = statistics.median(bucket["doc_ranks"]) if bucket["doc_ranks"] else None
            bucket["win_rank_median"] = statistics.median(bucket["win_ranks"]) if bucket["win_ranks"] else None
            del bucket["doc_ranks"], bucket["win_ranks"]
        totals["doc_top3_rate"] = round(totals["doc_top3"] / totals["scored"], 4) if totals["scored"] else None
        totals["doc_top1_rate"] = round(totals["doc_top1"] / totals["scored"], 4) if totals["scored"] else None
        totals["win_top3_rate"] = round(totals["win_top3"] / len(wr), 4) if wr else None
        totals["win_top1_rate"] = round(totals["win_top1"] / len(wr), 4) if wr else None
        totals["windows_scored"] = len(wr)
        totals["doc_rank_median"] = statistics.median(dr) if dr else None
        totals["win_rank_median"] = statistics.median(wr) if wr else None
        results[lane_name] = {"per_set": per_set, "totals": totals}

    report = {"declared": {"top": TOP, "lanes": list(lanes), "card": "the pass-1 abstract alone",
                           "scoring": "IDF-weighted token overlap over the lane, normalised by the square "
                                      "root of the card's token count; ties by document order then id",
                           "cards_file": str(a.cards), "cards": {"windows": len(windows),
                                                                 "documents": len(documents)},
                           "targets": len(targets)},
              "results": results, "misses_abstract_lane": misses}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "map_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
                                         + "\n", encoding="utf-8")
    if a.emit_targets:
        a.emit_targets.parent.mkdir(parents=True, exist_ok=True)
        with a.emit_targets.open("w", encoding="utf-8") as handle:
            for index in sorted(emitted):
                handle.write(json.dumps(emitted[index], ensure_ascii=False, sort_keys=True) + "\n")
        print(f"targets emitted: {len(emitted)} → {a.emit_targets}")

    lines = ["# The map test — can a search over pass-1 cards find the place?", "",
             f"{len(targets)} targets from four sealed sets · cards: {len(windows)} windows and "
             f"{len(documents)} documents, **the abstract alone** · lexical, IDF-weighted, deterministic",
             "", "| lane | targets scored | document in top 3 | document top 1 | window in top 3 | "
             "window top 1 | median doc rank | median window rank |", "|---|---|---|---|---|---|---|---|"]
    for lane_name, r in results.items():
        t = r["totals"]
        lines.append(f"| **{lane_name}** | {t['scored']} | {t['doc_top3_rate']} | {t['doc_top1_rate']} | "
                     f"{t['win_top3_rate']} | {t['win_top1_rate']} | {t['doc_rank_median']} | "
                     f"{t['win_rank_median']} |")
    lines += ["", "## Per gold set, the abstract lane", "",
              "| set | targets | scored | out of frame | document top 3 | window top 3 |",
              "|---|---|---|---|---|---|"]
    for name, bucket in sorted(results["abstract"]["per_set"].items()):
        lines.append(f"| {name} | {bucket['targets']} | {bucket['scored']} | {bucket['out_of_frame']} | "
                     f"{bucket['doc_top3_rate']} | {bucket['win_top3_rate']} |")
    if misses:
        lines += ["", "## Where the abstract lane misses (first 40)", "",
                  "| set | query | document rank | window rank | windows |", "|---|---|---|---|---|"]
        for m in misses:
            lines.append(f"| {m['set']} | {m['query'].replace('|', '/')} | {m['doc_rank']} | "
                         f"{m['window_rank']} | {m['windows_in_document']} |")
    (a.out / "map_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({lane: results[lane]["totals"] for lane in results}, ensure_ascii=False,
                     sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
