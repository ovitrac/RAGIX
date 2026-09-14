#!/usr/bin/env python3
"""demoE2E step 05 — the coordinating seat's check of the gold, independent of ``read.py``.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The builder verified its own output; this file re-derives the gate from the committed
gold and the store alone, with its own SQL and its own hashing, and adds what the builder
did not measure:

1. a seeded sample of ten items (six grid, two fiche A, two fiche B; the seed is the first
   eight hex digits of the sealed grid's sha256, so the draw is fixed by the gold itself),
   every location checked for byte-exact span at ``char_offset``, ``byte_offset`` consistent,
   ``span_sha256`` recomputed, chunk belonging to the stated document, pages within the
   chunk's pages, document path naming the role's piece when the role starts with one (RC, CCAP,
   CCTP, MAJ, DCE — a role such as "group 1 price" is not checked), and a reading-log entry;
2. the gate over every location of the three files (byte-exact and same document);
3. the fiche A/B agreement recomputed from the two YAML files, not from the work files;
4. **digit coverage**: for every answered or ambiguous item, whether each digit token of the
   value appears in a span (whitespace removed, so ``1·8·202·6`` counts as ``18`` and ``2026``),
   and when it does not, whether it appears in one of the item's own located chunks (the span
   was cut short) or in none of them (the figure comes from elsewhere). Article numbers in a
   value ("CCAP 13") count as digits too, so the measure is strict, not a verdict.

Writes ``coord_check.json`` beside the gold. Read-only on the store (``mode=ro``).

    python3 demoE2E/gold/coord_check.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
import sqlite3
from pathlib import Path

import yaml

LAB = Path(os.environ.get("HARVEST_LAB", ".")).resolve()     # the lab root the paths below hang from
HERE = LAB / "demoE2E/gold"
STORE = LAB / "demoE2E/runs/02_collect/20260906T163718/run/saqqara.db"
FILES = {"grid": ("gold_grid.yaml", "1", 6), "ficheA": ("gold_fiche.yaml", "A", 2),
         "ficheB": ("gold_fiche_passB.yaml", "B", 2)}


def digits(s) -> set[str]:
    return set(re.findall(r"\d+", str(s or "")))


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def main() -> int:
    con = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    gold = {t: yaml.safe_load((HERE / f).read_text())["items"] for t, (f, _, _) in FILES.items()}
    shas = {t: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for t, (f, _, _) in FILES.items()}
    by_id = {t: {i["id"]: i for i in items} for t, items in gold.items()}
    log = [json.loads(l) for l in (HERE / "reading_log.jsonl").read_text().splitlines() if l.strip()]
    logn = collections.Counter((r["pass"], r["item"]) for r in log)
    out: dict = {"store": str(STORE.relative_to(LAB)), "gold_sha256": shas}

    def chunk(cid):
        return con.execute("select * from chunks where chunk_id=?", (cid,)).fetchone()

    def check_loc(l) -> list[str]:
        r = chunk(l["chunk_id"])
        if r is None:
            return ["no-chunk"]
        t, sp, co = r["text"], l["span"], l["char_offset"]
        src = con.execute("select source_path from documents where doc_id=?", (l["doc_id"],)).fetchone()
        pages = json.loads(r["pages_json"] or "[]")
        piece = l["role"].split()[0].upper().rstrip(":")
        piece_named = piece in {"RC", "CCAP", "CCTP", "MAJ", "DCE"}
        checks = [("exact", t[co:co + len(sp)] == sp),
                  ("byte", len(t[:co].encode("utf-8")) == l["byte_offset"]),
                  ("sha", hashlib.sha256(sp.encode("utf-8")).hexdigest()[:16] == l["span_sha256"]),
                  ("doc", r["doc_id"] == l["doc_id"]),
                  ("pages", set(l["pages"]) <= set(pages)),
                  ("piece", not piece_named or (src is not None and piece in src["source_path"].upper()))]
        return [n for n, ok in checks if not ok]

    # 1. the seeded sample
    seed = int(shas["grid"][:8], 16)
    rng = random.Random(seed)
    sample = []
    for t, (_, p, n) in FILES.items():
        pool = [i for i in gold[t] if i["status"] != "absent"]
        sample += [(t, i["id"]) for i in rng.sample(pool, n)]
    sample_rows = []
    for t, qid in sample:
        it = by_id[t][qid]
        fails = {l["chunk_id"][:12]: f for l in it["locations"] if (f := check_loc(l))}
        sample_rows.append({"file": t, "id": qid, "status": it["status"], "locations": len(it["locations"]),
                            "failing": fails, "log_entries": logn[(FILES[t][1], qid)],
                            "path_steps": len(it.get("reading_path") or [])})
    out["sample"] = {"seed": seed, "items": sample_rows,
                     "locations": sum(r["locations"] for r in sample_rows),
                     "failing": sum(len(r["failing"]) for r in sample_rows),
                     "items_without_log": [r["id"] for r in sample_rows if r["log_entries"] == 0]}

    # 2. the gate over everything, and 4. digit coverage
    gate, cover = {}, {}
    for t, (_, p, _) in FILES.items():
        n = ok = 0
        nolog = [i["id"] for i in gold[t] if logn[(p, i["id"])] == 0]
        full = tot = inchunk = nowhere = 0
        nowhere_items = []
        for it in gold[t]:
            locs = it.get("locations") or []
            for l in locs:
                n += 1
                ok += not check_loc(l)
            if it["status"] == "absent":
                continue
            tot += 1
            joined = "".join(squash(l["span"]) for l in locs)
            texts = "".join(squash(chunk(l["chunk_id"])["text"]) for l in locs)
            miss = [d for d in digits(it["value"]) if d not in joined]
            full += not miss
            for d in miss:
                if d in texts:
                    inchunk += 1
                else:
                    nowhere += 1
                    nowhere_items.append([it["id"], d])
        gate[t] = {"locations": n, "exact": ok, "items_without_log": nolog}
        cover[t] = {"items": tot, "all_digits_in_a_span": full, "missing_in_own_chunk": inchunk,
                    "missing_nowhere_located": nowhere, "nowhere": nowhere_items}
    out["gate"], out["digit_coverage"] = gate, cover

    # 3. agreement from the YAML files
    A, B = by_id["ficheA"], by_id["ficheB"]
    ids = sorted(A)
    n = len(ids)
    agree = sum(A[i]["status"] == B[i]["status"] for i in ids)
    ca = collections.Counter(A[i]["status"] for i in ids)
    cb = collections.Counter(B[i]["status"] for i in ids)
    pe = sum(ca[s] * cb[s] for s in set(ca) | set(cb)) / n / n
    po = agree / n
    dj, ident = [], 0
    for i in ids:
        da = {l["doc_id"] for l in A[i].get("locations") or []}
        db = {l["doc_id"] for l in B[i].get("locations") or []}
        dj.append(len(da & db) / len(da | db) if da | db else 1.0)
        ident += da == db
    out["agreement"] = {"items": n, "status_agreement": round(po, 4), "status_kappa": round((po - pe) / (1 - pe), 4),
                        "status_chance": round(pe, 4), "disagreements": [i for i in ids if A[i]["status"] != B[i]["status"]],
                        "documents_identical": ident, "documents_jaccard_mean": round(sum(dj) / n, 3)}
    (HERE / "coord_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("gate", "agreement")}, ensure_ascii=False))
    print("sample", out["sample"]["locations"], "locations, failing", out["sample"]["failing"],
          "items_without_log", out["sample"]["items_without_log"])
    for t, c in cover.items():
        print(t, c["all_digits_in_a_span"], "/", c["items"], "in-chunk", c["missing_in_own_chunk"], "nowhere", c["nowhere"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
