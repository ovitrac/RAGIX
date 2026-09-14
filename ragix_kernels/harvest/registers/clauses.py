#!/usr/bin/env python3
"""M7 — what the CCAP and the RC fix: their numeric and contractual statements (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4). M7's own rules; family.py's, template.py's and
commitments.py's are untouched — C1's pattern is *imported* from commitments.py, not copied, so the two
registers cannot drift. No kernel grammar is used (tender's grammars stay closed to this seat).

Why this exists: commitments.json covers the 22 CCTP only. The register-pointed rule of row 19 therefore
had nothing beneath the brief's « risques » (pénalités, paiement, résiliation) and « go/no-go »
(délais, procédure) columns, which the CCAP and the RC fix, not the CCTP.

K0 frame. The documents pieces.yaml files under the roles CCAP and RC — four paths, two per role — each
   labelled here, never inferred:
     CCAP     = 'CCAP <REF> signé.pdf'                                     (the contract)
     CCAP-A5  = 'CCAP <REF> - Annexe 5 - Mode opératoire ...docx'          (its annex 5)
     RC       = 'RC <REF> (v2)_signé.pdf'                                  (the consultation rules)
     MAJ-DCE  = 'MAJ du DCE (version 5).docx'                             (the DCE update notice)
   Gate: the two roles' path sets equal exactly these four; each is in the store exactly once; its single
   level-1 roll-up (parent_id NULL) equals its level-0 leaves joined by newlines. All four are read, not
   only the two signed contracts: the gap found in commitments.json was a document silently absent from a
   role, and a per-role frame with a gate cannot repeat it.
K1 text and sentences. family.py's text, splitter and normalisation exactly (F2, S1, N1, MIN_WORDS), over
   the roll-up; distinct sentences by first occurrence inside the document. df is not defined for a single
   document and is not computed, so C2's df <= 3 selection and C3's template flag do not apply here.
K2 pattern. commitments.py's C1 unchanged (`C.MATCH_RE`: NUM UNIT over amount, percentage, duration,
   count; `C.FREQ_RE`: the frequency words) **plus one class declared for these two documents**, matched
   case-insensitively on the sentence's raw text, word-bounded, accents exact:
     penalty = pénalité(s) · retenue(s) · [par] jour[s] [calendaire(s)|ouvré(s)|ouvrable(s)] de retard ·
               délai(s) de paiement · intérêt(s) moratoire(s) · résiliation(s) · garantie(s) ·
               avance(s) · révision(s)
   The class is lexical, not semantic: it marks the vocabulary of a risk, it does not read the clause.
   It needs no number — 'résiliation' without a figure is exactly what the « risques » column must show.
K3 selection. A distinct sentence with at least one match of any class (K2's or C1's). No df filter.
K4 ranking, per document. By the rank of the sentence's best class, then by the earlier byte offset;
   the first ten are `top10` — fewer when the document holds fewer. A sentence flagged by K6 is listed,
   never ranked (C3's discipline for template members). Ranks:
     0 penalty *and* a number — a penalty or payment term with at least one C1 match in the sentence;
     1 amount · 2 duration · 3 penalty alone · 4 percentage · 5 count · 6 frequency.
   AMENDED 2026-09-11 after the first exploratory run, which is why the rank of a bare penalty term is
   not 0: the K2 class is lexical, and in the RC all three of its hits are the participle of 'retenir'
   ('l'offre la mieux classée sera retenue'), not a retenue held on a payment. A term carrying a figure
   is what the « risques » column needs; a term alone stays in the register, below the numbers.
K6 what is listed and never ranked, with its reason in `excluded` (C3's discipline for template
   members). DECLARED 2026-09-11 with K4's amendment, each rule for a measured reason:
   'toc' — a contents entry. Both conditions: a leader run ('...' or '…', a separator in S1, so every
      line of a table of contents survives as one sentence) or its truncated tail ('..') touches the
      sentence across whitespace only, before it (inside the preceding 64 characters) or after it; **and**
      the sentence begins with a page number then a numbered heading ('22 18 .1.1 - pénalités de retard').
      The conjunction is what makes it precise: leader adjacency alone flagged two real penalty rows of
      the CCAP's p. 25 table, the head alone would trust a shape the body also uses. Without the rule the
      CCAP's ten first rows were its ten contents entries, which locate clauses instead of fixing anything.
   'cover' — the document's first distinct sentence when its span crosses a page boundary: the title
      block the splitter cannot cut (M2's defect D4), 615 characters over 30 leaves in the CCAP.
K5 locator. Each row carries its span (roll-up chunk_id, byte range, first and last leaf with the byte
   offset inside each) and `pages` = [min(pages of the first leaf), max(pages of the last leaf)], or []
   when a leaf carries none (the two .docx carry none).
K7 a cut number. A match of a numeric class is `cut` when the last non-space character before it is a
   digit: the number's head is then in another leaf and the match holds its tail only. DECLARED
   2026-09-11, measured: the CCAP's roll-up holds 'Forfaitaire\\n25\\n0,00 €' for a penalty of 250,00 €,
   so C1 reads '0,00 €' — a critical value read wrong by a deterministic extractor (§11 of CLAUDE.md).
   The register does not repair it: rejoining digit runs across a newline would glue two table cells
   ('150\\n50' is not 15050). It marks the row, the reader is warned, and the fix belongs upstream in the
   extraction. commitments.py is NOT amended today — the harvest of row 19 is reading commitments.json
   while this runs, and its 6 cases in 555 matches are reported instead.
Gates (refuse, exit 2): the store's sha256 prefix and an empty WAL; K0's frame; every sentence span and
every match span re-read from the store byte for byte (F.span_ok); `pages_json` read from the very leaves
the span names.
Outputs: clauses.json; manifest.json gains a 'clauses' section (this script's sha256, counts).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

from . import commitments as C
from . import family as F

#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
TOP = 10
LABELS = {
    "CCAP": ("CCAP", f"CCAP {CONSULTATION} signé.pdf"),
    "CCAP-A5": ("CCAP", f"CCAP {CONSULTATION} - Annexe 5 - Mode opératoire CMAD-MMP Maintenance Technique.docx"),
    "RC": ("RC", f"RC {CONSULTATION} (v2)_signé.pdf"),
    "MAJ-DCE": ("RC", "MAJ du DCE (version 5).docx"),
}
ORDER = ["CCAP", "CCAP-A5", "RC", "MAJ-DCE"]
NUMERIC = ("amount", "percentage", "duration", "count")
CLASS_RANK = {"penalty+number": 0, "amount": 1, "duration": 2, "penalty": 3, "percentage": 4,
              "count": 5, "frequency": 6}
TOC_BEFORE = 64
LEADER_AFTER_RE = re.compile(r"\s*(?:\.{2,}|…+)")
LEADER_BEFORE_RE = re.compile(r"(?:\.{2,}|…+)\s*$")
TOC_HEAD_RE = re.compile(r"^\d{1,3}\s+\d{1,2}(?:\s*\.\s*\d{1,2})*\s*[-–]\s")
PENALTY_TERMS = (
    r"p[ée]nalit[ée]s?",
    r"retenues?",
    r"(?:par\s+)?jours?(?:\s+(?:calendaires?|ouvr[ée]s?|ouvrables?))?\s+de\s+retard",
    r"d[ée]lais?\s+de\s+paiement",
    r"int[ée]r[êe]ts?\s+moratoires?",
    r"r[ée]siliations?",
    r"garanties?",
    r"avances?",
    r"r[ée]visions?",
)
PENALTY_RE = re.compile(r"(?<!\w)(?:" + "|".join(PENALTY_TERMS) + r")(?!\w)", re.IGNORECASE)


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"clauses: REFUSED — {msg}\n")
    sys.exit(2)


def cut_before(text: str, i: int) -> bool:
    """K7: the last non-space character before position i is a digit."""
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return j >= 0 and text[j].isdigit()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M7 — what the CCAP and the RC fix (S4).")
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
    man_path = a.out / "manifest.json"
    if not man_path.exists():
        fail("manifest.json missing: family.py runs first in the same directory")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    from ..pieces import load_pieces  # the run's own parser of the piece map

    mapped = load_pieces(a.lab / "demoE2E/03_analyze/pieces.yaml")
    for role in ("CCAP", "RC"):
        declared = sorted(p for _lbl, (r, p) in LABELS.items() if r == role)
        if sorted(mapped[role]) != declared:
            fail(f"role {role}: pieces.yaml lists {sorted(mapped[role])}, this script declares {declared}")
    docs = db.execute("select doc_id, source_path, doc_class from documents").fetchall()

    pieces: dict[str, F.Piece] = {}
    classes: dict[str, str] = {}
    leafpages: dict[str, list[int]] = {}
    for label in ORDER:
        role, path = LABELS[label]
        hits = [d for d in docs if d[1].endswith("/" + path)]
        if len(hits) != 1:
            fail(f"{label}: {len(hits)} store documents for {path!r}")
        doc_id, _, doc_class = hits[0]
        leaves = db.execute("select chunk_id, text, pages_json from chunks where doc_id=? and level=0 order by seq",
                            (doc_id,)).fetchall()
        rolls = db.execute("select chunk_id, text from chunks where doc_id=? and level=1 and parent_id is null",
                           (doc_id,)).fetchall()
        if len(rolls) != 1 or rolls[0][1] != "\n".join(t for _, t, _ in leaves):
            fail(f"{label}: {len(rolls)} roll-ups, or the roll-up is not its leaves joined by newlines")
        for cid, _, pj in leaves:
            leafpages[cid] = json.loads(pj) if pj else []
        pc = F.Piece(label, doc_id, path, rolls[0][0], rolls[0][1], [(c, t) for c, t, _ in leaves])
        pc.split()
        pieces[label], classes[label] = pc, doc_class

    cache: dict = {}
    spans_checked = 0

    def locate(label: str, c0: int, c1: int) -> dict:
        nonlocal spans_checked
        pc = pieces[label]
        loc = pc.locate(c0, c1)
        if not F.span_ok(db, cache, loc, pc.text[c0:c1]):
            fail(f"{label}: span at bytes {loc['bytes']} does not re-read from the store")
        spans_checked += 1
        return loc

    out_docs: dict[str, Any] = {}
    by_class: Counter = Counter()
    not_taken: Counter = Counter()      # what the grammar read and C1 declines (item 4)
    n_sent = n_match = n_cut = 0
    n_excl: Counter = Counter()
    for label in ORDER:
        pc = pieces[label]
        rows: list[dict[str, Any]] = []
        distinct = sorted(pc.first().values(), key=lambda s: s["c0"])
        first_c0 = distinct[0]["c0"] if distinct else -1
        for s in distinct:
            # AMENDMENT 2026-09-12: the numbers come from `commitments.read_c1`, which reads them with
            # the kernel's grammar; the penalty class declared for these two documents is this file's
            # own and is unchanged. One definition of a value, shared with the harvest (item 4).
            taken, passed_over = C.read_c1(s["raw"])
            for item in passed_over:
                not_taken[item["kind"]] += 1
            found: list[tuple[int, int, str, str | None]] = list(taken)
            for mm in PENALTY_RE.finditer(s["raw"]):
                found.append((mm.start(), mm.end(), "penalty", None))
            if not found:
                continue
            found.sort()
            loc = locate(label, s["c0"], s["c1"])
            pf = leafpages[loc["leaves"]["first"][0]]
            pl = leafpages[loc["leaves"]["last"][0]]
            matches = []
            for m0, m1, cls, normalized in found:
                mloc = locate(label, s["c0"] + m0, s["c0"] + m1)
                m: dict[str, Any] = {"text": s["raw"][m0:m1], "class": cls, "bytes": mloc["bytes"]}
                if normalized:
                    m["normalized"] = normalized
                if cls in NUMERIC and cut_before(pc.text, s["c0"] + m0):
                    m["cut"] = True
                    n_cut += 1
                matches.append(m)
                by_class[cls] += 1
            leader = bool(LEADER_AFTER_RE.match(pc.text, s["c1"])
                          or LEADER_BEFORE_RE.search(pc.text[max(0, s["c0"] - TOC_BEFORE):s["c0"]]))
            pages = [min(pf), max(pl)] if pf and pl else []
            excluded = None
            if leader and TOC_HEAD_RE.search(s["raw"]):
                excluded = "toc"
            elif s["c0"] == first_c0 and len(pages) == 2 and pages[0] < pages[1]:
                excluded = "cover"
            classes_here = {m["class"] for m in matches}
            if "penalty" in classes_here and classes_here & set(NUMERIC):
                classes_here.add("penalty+number")
            rows.append({"hash": s["hash"], "text": F.norm1(s["raw"]), **loc,
                         "pages": pages, "excluded": excluded,
                         "cut": any(m.get("cut") for m in matches),
                         "rank_class": min(classes_here, key=lambda c: CLASS_RANK[c]), "matches": matches})
            n_sent += 1
            n_match += len(matches)
            n_excl[excluded or "ranked"] += 1
        ranked = sorted((i for i, r in enumerate(rows) if not r["excluded"]),
                        key=lambda i: (CLASS_RANK[rows[i]["rank_class"]], rows[i]["bytes"][0]))
        out_docs[label] = {"role": LABELS[label][0], "source_path": LABELS[label][1],
                           "doc_class": classes[label], "rollup_chunk_id": pc.roll_id, "chars": len(pc.text),
                           "sentences": rows, "top10": ranked[:TOP],
                           "counts": {"split_sentences": len(pc.sentences), "distinct": len(pc.first()),
                                      "selected": len(rows), "rankable": len(ranked),
                                      "excluded": dict(sorted(Counter(
                                          r["excluded"] for r in rows if r["excluded"]).items())),
                                      "matches": sum(len(r["matches"]) for r in rows),
                                      "cut_rows": sum(r["cut"] for r in rows),
                                      "by_class": dict(sorted(Counter(
                                          m["class"] for r in rows for m in r["matches"]).items()))}}
    counts = {"documents": len(ORDER), "sentences": n_sent, "matches": n_match, "cut_matches": n_cut,
              "excluded": dict(sorted(n_excl.items())),
              "by_class": dict(sorted(by_class.items())), "spans_reread": spans_checked,
              "per_document": {lbl: out_docs[lbl]["counts"]["selected"] for lbl in ORDER}}
    F.dump(a.out / "clauses.json", {
        "declared": {"labels": {lbl: {"role": r, "path": p} for lbl, (r, p) in LABELS.items()},
                     "order": ORDER, "top": TOP, "class_rank": CLASS_RANK,
                     "reader": f"tender.grammars_fr {C.GRAMMAR_VERSION}",
                     "kinds_not_taken": dict(not_taken),
                     "pattern": {"c1_from": "commitments.py (imported, not copied)",
                                 "penalty": PENALTY_RE.pattern},
                     "cut": "a numeric match whose last non-space predecessor is a digit: the "
                            "number's head is in another leaf (K7); marked, never repaired",
                     "excluded": {"toc": "a leader run touching the sentence across whitespace "
                                         f"({TOC_BEFORE} characters before, any distance after) and a "
                                         f"contents head {TOC_HEAD_RE.pattern!r}",
                                  "cover": "the first distinct sentence when its span crosses a page "
                                           "boundary"},
                     "top10": "best class rank first, then the earlier offset; no df, no template flag"},
        "counts": counts, "documents": out_docs})
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["clauses"] = {"script_sha256": F.sha256_file(Path(__file__)), "counts": counts}
    F.dump(man_path, man)
    print(json.dumps({k: v for k, v in counts.items() if k != "by_class"}, sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
