#!/usr/bin/env python3
"""Intra-reader agreement between two passes of the same gold (runbook row 05).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Compares the READER WITH HIMSELF, never with the run: it reads two record files
of demoE2E/gold/work/ and the store (mode=ro) to resolve chunk prefixes, and
nothing else. Four measures, each mechanical so a reviewer can recompute it:

  status    — agreement on answered / absent / ambiguous, raw and Cohen's kappa
  documents — Jaccard of the documents each pass cites for the item
  chunks    — Jaccard of the chunks each pass cites
  numbers   — equality of the sets of digit tokens in the two values

Cohen's kappa is reported beside the raw rate, and read with its known limit:
when one status dominates, kappa can be low while agreement is high.

    python3 demoE2E/gold/agreement.py work/fiche_A.jsonl work/fiche_B.jsonl out.json
"""
import json, os, re, sqlite3, sys
from collections import Counter
from pathlib import Path

LAB = Path(os.environ.get("HARVEST_LAB", ".")).resolve()     # the lab root the paths below hang from
STORE = LAB / "demoE2E/runs/02_collect/20260906T163718/run/saqqara.db"

def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip())}

def resolve(con, prefix):
    rows = con.execute("select chunk_id, doc_id from chunks where chunk_id like ?", (prefix + "%",)).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"chunk {prefix!r} resolves to {len(rows)}")
    return rows[0]

def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)

def digits(v):
    return set(re.findall(r"\d+(?:[.,]\d+)?", v or ""))

def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0, pe

def main(a_path, b_path, out_path):
    A, B = load(a_path), load(b_path)
    if set(A) != set(B):
        raise SystemExit(f"the two passes cover different items: {sorted(set(A) ^ set(B))}")
    con = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    items, pairs = [], []
    for i in sorted(A, key=lambda x: (len(x), x)):
        a, b = A[i], B[i]
        ca = {resolve(con, L["chunk"]) for L in a.get("locations", [])}
        cb = {resolve(con, L["chunk"]) for L in b.get("locations", [])}
        row = {"id": i, "status_A": a["status"], "status_B": b["status"],
               "status_agree": a["status"] == b["status"],
               "documents_jaccard": round(jaccard({d for _, d in ca}, {d for _, d in cb}), 3),
               "chunks_jaccard": round(jaccard({c for c, _ in ca}, {c for c, _ in cb}), 3),
               "numbers_agree": digits(a.get("value")) == digits(b.get("value")),
               "numbers_A": sorted(digits(a.get("value"))), "numbers_B": sorted(digits(b.get("value")))}
        items.append(row); pairs.append((a["status"], b["status"]))
    po, k, pe = kappa(pairs)
    n = len(items)
    summary = {
        "items": n,
        "status_agreement": round(po, 4), "status_kappa": round(k, 4), "status_chance": round(pe, 4),
        "status_disagreements": [r["id"] for r in items if not r["status_agree"]],
        "documents_jaccard_mean": round(sum(r["documents_jaccard"] for r in items) / n, 3),
        "documents_identical": sum(r["documents_jaccard"] == 1.0 for r in items),
        "chunks_jaccard_mean": round(sum(r["chunks_jaccard"] for r in items) / n, 3),
        "numbers_agree": sum(r["numbers_agree"] for r in items),
        "numbers_disagreements": [r["id"] for r in items if not r["numbers_agree"]],
        "limits": ["one agent session cannot forget: pass B measures re-derivation from the store, not an independent reading",
                   "kappa is depressed when one status dominates; read it beside the raw rate",
                   "numbers_agree compares digit tokens only: a reworded value with the same figures agrees, a value adding a derived figure does not"],
    }
    Path(out_path).write_text(json.dumps({"passes": [a_path, b_path], "summary": summary, "items": items},
                                         ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in items:
        flags = [] if r["status_agree"] else [f"STATUS {r['status_A']}→{r['status_B']}"]
        if not r["numbers_agree"]: flags.append(f"numbers A{r['numbers_A']} B{r['numbers_B']}")
        print(f"  {r['id']:<4} docs {r['documents_jaccard']:.2f} chunks {r['chunks_jaccard']:.2f} {' · '.join(flags)}")

if __name__ == "__main__":
    main(*sys.argv[1:4])
