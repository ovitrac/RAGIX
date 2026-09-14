#!/usr/bin/env python3
"""Select the questions of grid pass 2 — the ones both arms of step 03 agreed on — and hand
them to the reader shuffled and unlabelled.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Written after runbook row 05 step 4 (the three gold sha256 committed at 368e81e), as the row
requires. This script is the one exception to the gold's blindness rule: it opens exactly one
file under demoE2E/measurements/, the verdicts table of the 8 September run, and prints
**bucket sizes only** — never a question id next to a verdict. The reader therefore knows the
arms agreed on the twenty-eight questions, and not on what.

Selection: a question is kept when every arm rendered it PRÊT AVEC RÉSERVES, or every arm
rendered it PREUVE MANQUANTE. Questions with an irregular number of arms are counted and
dropped. The order is a shuffle seeded by the first eight hex digits of the sealed grid sha
(row 05: 0544e3a7…), the seed the coordinating seat used for its sample of ten, so the file is
reproducible and carries no information from the verdicts.

Run:  conda run -n tender-rag python demoE2E/gold/select_pass2.py
Out:  demoE2E/gold/work/pass2_ids.txt — one id per line, nothing else.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

LAB = Path(os.environ.get("HARVEST_LAB", ".")).resolve()     # the lab root the paths below hang from
HERE = LAB / "demoE2E/gold"
VERDICTS = LAB / "demoE2E/measurements/03_analyze_20260908T125120_verdicts.csv"
OUT = HERE / "work/pass2_ids.txt"

HEADER = ["model", "digest", "question_id", "verdict", "rendering",
          "citations", "candidates", "needs"]
KEEP = {"PRÊT AVEC RÉSERVES": "supported_with_caveats",
        "PREUVE MANQUANTE": "abstain_no_evidence"}
EXPECTED = {"PRÊT AVEC RÉSERVES": 9, "PREUVE MANQUANTE": 19}   # row 05, step 5
SEALED_GRID_SHA_PREFIX = "0544e3a7"                             # row 05, gold_grid.yaml


def main() -> int:
    if not VERDICTS.is_file():
        print(f"no verdicts file at {VERDICTS.relative_to(LAB)}")
        return 1
    with VERDICTS.open(encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != HEADER:
            print("unexpected header:", header)
            return 1
        rows = [dict(zip(header, r)) for r in rd]

    arms = sorted({r["model"] for r in rows})
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)

    inconsistent = sum(1 for r in rows
                       if r["rendering"] in KEEP and r["verdict"] != KEEP[r["rendering"]])
    if inconsistent:
        print(f"{inconsistent} row(s) whose rendering and verdict disagree — refused")
        return 1

    buckets = {k: 0 for k in KEEP}
    irregular = disagree = other = 0
    chosen: list[str] = []
    for q, rs in by_q.items():
        if len(rs) != len(arms) or {r["model"] for r in rs} != set(arms):
            irregular += 1
            continue
        renders = {r["rendering"] for r in rs}
        if len(renders) != 1:
            disagree += 1
            continue
        rendering = renders.pop()
        if rendering in KEEP:
            buckets[rendering] += 1
            chosen.append(q)
        else:
            other += 1

    seed = int(SEALED_GRID_SHA_PREFIX, 16)
    random.Random(seed).shuffle(chosen)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(f"{q}\n" for q in chosen), encoding="utf-8")

    print(json.dumps({
        "verdicts": VERDICTS.name, "arms": len(arms), "questions": len(by_q),
        "buckets": buckets, "expected": EXPECTED,
        "agreed_on_other_renderings": other, "arms_disagree": disagree, "irregular": irregular,
        "selected": len(chosen), "seed": seed,
        "out": str(OUT.relative_to(LAB)),
        "out_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))
    return 0 if buckets == EXPECTED else 2


if __name__ == "__main__":
    sys.exit(main())
