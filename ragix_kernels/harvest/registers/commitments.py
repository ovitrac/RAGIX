#!/usr/bin/env python3
"""M2c — each CCTP's own numeric statements: the sentences a piece shares with at most two others that
carry a number with a unit, or a frequency word (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4). M2c's own rules; family.py's and template.py's are untouched,
and no kernel grammar is used (tender's grammars stay closed to this seat).

C0 sentences. family.py's frame, text, splitter and normalisation exactly (F1, F2, S1, N1: no header
   separator here, so df is family.py's P1 df over the 22 CCTPs). Gate: the distinct sentence count
   equals partition.json's in the same directory. Each piece's distinct sentences (first occurrence).
C1 pattern, over a sentence's raw text, case-insensitive:
   NUM  = digits with optional thousands groups ('1 000') and decimals ('300,00'), or '½', or a spelled
          French number (une, un, deux … vingt, trente … soixante, cent, mille), optionally followed by a
          parenthesised digit ('quatre (4)');
   UNIT = amount (€, euro, euros, EUR), percentage (%), duration (minutes, min, heures, h, journées,
          jours, semaines, mois, années, ans), count (fois, visites, passages, interventions), each ending
          on a word boundary, after optional whitespace;
   a match is NUM UNIT, not preceded by a word character; a frequency word (annuel, semestriel,
   trimestriel, mensuel, hebdomadaire, quotidien and their inflections) is a match of class 'frequency'.
C2 selection. A sentence of df <= 3 with at least one match. For df 2 or 3 the sharing pieces are given.
C3 template flag. A sentence overlapping by at least one byte the span of a member of a template.json
   cluster (M2b, >= 18 pieces) in the same piece is flagged template_member: it is listed, never ranked.
C4 the three a bidder reads first, per piece, among the sentences not flagged: ordered by df ascending,
   then a sentence with an amount or duration match before one without, then more matches first, then
   the earlier offset; the first three.
Gates (refuse, exit 2): family.py's store and frame gates; the distinct count against partition.json;
template.json present; every sentence and match span re-read from the store byte for byte.
Outputs: commitments.json; manifest.json gains a 'commitments' section (this script's sha256, counts).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn

from . import family as F
from ..fr.grammars import VERSION as GRAMMAR_VERSION
from ..fr.grammars import read_values

FEW = 3
SPELLED = ("une un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize "
           "vingt trente quarante cinquante soixante cent mille")
NUM = (r"(?:\d+(?:[  ]\d{3})*(?:[.,]\d+)?|½|(?:" + "|".join(SPELLED.split()) + r")(?![^\W\d_]))"
       r"(?:\s*\(\s*\d+\s*\))?")
UNITS = {
    "amount": r"€|euros?(?![^\W\d_])|eur(?![^\W\d_])",
    "percentage": r"%",
    "duration": r"(?:minutes?|min|heures?|h|journées?|jours?|semaines?|mois|années?|ans?)(?![^\W\d_])",
    "count": r"(?:fois|visites?|passages?|interventions?)(?![^\W\d_])",
}
MATCH_RE = re.compile(r"(?<!\w)" + NUM + r"\s*(?:" + "|".join(f"(?P<{k}>{v})" for k, v in UNITS.items()) + ")",
                      re.IGNORECASE)
FREQ_RE = re.compile(r"(?<!\w)(?:annuel|semestriel|trimestriel|mensuel)(?:le)?s?(?!\w)|(?<!\w)hebdomadaires?(?!\w)"
                     r"|(?<!\w)quotidien(?:ne)?s?(?!\w)", re.IGNORECASE)

#: AMENDMENT 2026-09-12 (the coordinating seat's item 4): the numbers are read by the kernel's grammar,
#: not by this register's own `MATCH_RE`. The regex above is KEPT and still declared in the output, as
#: the pattern the earlier registers were built with — it is no longer what reads them. The reason is
#: measured, not stylistic: `MATCH_RE` matched « 000 heures » inside « 1 000 heures » because it had no
#: left boundary for a thousands group, and it could not see a figure split across an OCR line break
#: (« 1\n500,00 € »), which the grammar reads whole. There is now ONE definition of a value in this
#: project, and a register that disagreed with the harvest about what a number is could never be
#: reconciled with it.
#:
#: What the register TAKES, by the grammar's kind:
KIND_CLASS = {"amount": "amount", "percentage": "percentage", "duration": "duration",
              "quantity": "count"}
#: and what it does not take. These are COUNTED AND REPORTED, never silently dropped: a date is a
#: commitment but not one of the four numeric classes this register declares, and a reference is an
#: identifier, not a quantity. Whether a date becomes a class of its own is the scientific lead's
#: call, and the diff read gives him the number it would add.
KINDS_NOT_TAKEN = ("date", "datetime", "period", "reference")


def read_c1(raw: str) -> tuple[list[tuple[int, int, str, str | None]], list[dict[str, Any]]]:
    """C1 through `tender.grammars_fr`: (taken, passed_over).

    A value the grammar declines — 1.5 declines the tail of a decimal split by a line break and says
    so — is passed over WITH its reason rather than taken at face value: a partial read that yields a
    wrong number is worse than no read, and this register's whole purpose is the number.
    """
    taken: list[tuple[int, int, str, str | None]] = []
    passed: list[dict[str, Any]] = []
    for v in read_values(raw):
        cls = KIND_CLASS.get(v.kind)
        if cls is None:
            passed.append({"kind": v.kind, "text": v.raw, "why": "kind not taken by this register"})
        elif v.normalized is None:
            passed.append({"kind": v.kind, "text": v.raw, "why": v.reason or "no normal form"})
        else:
            taken.append((v.start, v.end, cls, v.normalized))
    for mm in FREQ_RE.finditer(raw):
        taken.append((mm.start(), mm.end(), "frequency", None))
    taken.sort()
    return taken, passed


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"commitments: REFUSED — {msg}\n")
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2c — each CCTP's own numeric statements (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--lab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    store_sha = F.sha256_file(a.store)
    if not store_sha.startswith(F.STORE_SHA_PREFIX):
        fail(f"store sha256 {store_sha[:16]} is not {F.STORE_SHA_PREFIX}")
    wal = Path(str(a.store) + "-wal")
    if wal.exists() and wal.stat().st_size:
        fail("the store's WAL is not empty")
    for name in ("manifest.json", "partition.json", "template.json"):
        if not (a.out / name).exists():
            fail(f"{name} missing: family.py and template.py run first in the same directory")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    from ..pieces import load_pieces  # the run's own parser of the piece map

    cctp_paths = sorted(load_pieces(a.lab / "demoE2E/03_analyze/pieces.yaml")["CCTP"])
    if len(cctp_paths) != F.N_CCTP:
        fail(f"pieces.yaml lists {len(cctp_paths)} CCTP paths")
    docs = db.execute("select doc_id, source_path, doc_class from documents").fetchall()
    pieces: dict[str, F.Piece] = {}
    for p in cctp_paths:
        m = F.PIECE_RE.search(p)
        hits = [d for d in docs if d[2] == "pdf" and d[1].endswith("/" + p)]
        if not m or len(hits) != 1:
            fail(f"{p}: {len(hits)} store documents")
        no, doc_id = m.group(1), hits[0][0]
        leaves = db.execute("select chunk_id, text from chunks where doc_id=? and level=0 order by seq",
                            (doc_id,)).fetchall()
        rolls = db.execute("select chunk_id, text from chunks where doc_id=? and level=1 and parent_id is null",
                           (doc_id,)).fetchall()
        if len(rolls) != 1 or rolls[0][1] != "\n".join(t for _, t in leaves):
            fail(f"{p}: the roll-up is not its leaves joined by newlines")
        pc = F.Piece(no, doc_id, "CCTP/" + p.split("CCTP/", 1)[1], rolls[0][0], rolls[0][1], leaves)
        pc.split()
        pieces[no] = pc
    order = sorted(pieces)
    firsts = {no: pieces[no].first() for no in order}
    df: Counter = Counter()
    holders: dict[str, list[str]] = defaultdict(list)
    for no in order:
        for k in firsts[no]:
            df[k] += 1
            holders[k].append(no)
    partition = json.loads((a.out / "partition.json").read_text(encoding="utf-8"))
    if len(df) != partition["distinct_sentences"]:
        fail(f"{len(df)} distinct sentences, partition.json has {partition['distinct_sentences']}")

    # C3: the template members' byte ranges, per piece
    template = json.loads((a.out / "template.json").read_text(encoding="utf-8"))
    tspans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for c in template["clusters"]:
        for p, mem in c["members"].items():
            tspans[p].append((mem["bytes"][0], mem["bytes"][1]))

    cache: dict = {}
    spans_checked = 0

    def locate(no: str, c0: int, c1: int) -> dict:
        nonlocal spans_checked
        pc = pieces[no]
        loc = pc.locate(c0, c1)
        if not F.span_ok(db, cache, loc, pc.text[c0:c1]):
            fail(f"CCTP {no}: span at bytes {loc['bytes']} does not re-read from the store")
        spans_checked += 1
        return loc

    out_pieces: dict[str, Any] = {}
    by_class: Counter = Counter()
    n_sent = n_match = n_flag = 0
    not_taken: collections.Counter[str] = collections.Counter()   # what the grammar read and C1 declines
    for no in order:
        pc = pieces[no]
        rows: list[dict[str, Any]] = []
        for k, s in sorted(firsts[no].items(), key=lambda kv: kv[1]["c0"]):
            if df[k] > FEW:
                continue
            found, passed_over = read_c1(s["raw"])
            for item in passed_over:
                not_taken[item["kind"]] += 1
            if not found:
                continue
            loc = locate(no, s["c0"], s["c1"])
            b0, b1 = loc["bytes"]
            flagged = any(t0 < b1 and b0 < t1 for t0, t1 in tspans.get(no, []))
            matches = []
            for m0, m1, cls, normalized in found:
                mloc = locate(no, s["c0"] + m0, s["c0"] + m1)
                matches.append({"text": s["raw"][m0:m1], "class": cls, "bytes": mloc["bytes"],
                                **({"normalized": normalized} if normalized else {})})
                by_class[cls] += 1
            rows.append({"hash": s["hash"], "df": df[k], "sharing": holders[k] if df[k] > 1 else [],
                         "template_member": flagged, "text": F.norm1(s["raw"]), **loc, "matches": matches})
            n_sent += 1
            n_match += len(matches)
            n_flag += flagged
        ranked = sorted((i for i, r in enumerate(rows) if not r["template_member"]),
                        key=lambda i: (rows[i]["df"],
                                       0 if any(m["class"] in ("amount", "duration") for m in rows[i]["matches"]) else 1,
                                       -len(rows[i]["matches"]), rows[i]["bytes"][0]))
        out_pieces[no] = {"source_path": pc.rel, "rollup_chunk_id": pc.roll_id, "sentences": rows,
                          "top3": ranked[:3],
                          "counts": {"sentences": len(rows), "unique": sum(r["df"] == 1 for r in rows),
                                     "template_members": sum(r["template_member"] for r in rows),
                                     "matches": sum(len(r["matches"]) for r in rows)}}
    counts = {"pieces": len(order), "sentences": n_sent, "matches": n_match, "template_members": n_flag,
              "by_class": dict(sorted(by_class.items())), "spans_reread": spans_checked,
              "per_piece": {no: out_pieces[no]["counts"]["sentences"] for no in order}}
    F.dump(a.out / "commitments.json", {
        "declared": {"few": FEW, "reader": f"tender.grammars_fr {GRAMMAR_VERSION}",
                     "kinds_taken": KIND_CLASS, "kinds_not_taken": dict(not_taken),
                     "superseded_pattern": {"num": NUM, "units": UNITS},
                     "pattern": {"num": NUM, "units": UNITS, "frequency": FREQ_RE.pattern},
                     "top3": "not template_member; df asc; amount or duration first; more matches; earlier"},
        "counts": counts, "pieces": out_pieces})
    man_path = a.out / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["commitments"] = {"script_sha256": F.sha256_file(Path(__file__)), "counts": counts}
    F.dump(man_path, man)
    print(json.dumps({k: v for k, v in counts.items() if k != "per_piece"}, sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
