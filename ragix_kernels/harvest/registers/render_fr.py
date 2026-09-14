#!/usr/bin/env python3
"""M2e — the French collection for the human: a deterministic rendering of the registers (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4). No new analysis: every line renders one record of
commitments.json, clauses.json, traps.json or template.json (same directory) and ends with that
record's id. AMENDED 2026-09-11: clauses.json is a fourth source and section (d) renders it; the page's
title names the three roles it now covers instead of the CCTP alone.
E1 text. A statement is its span re-read from the store (the roll-up's bytes), whitespace collapsed,
   case kept; '\\', '*', '_' and '`' are escaped for Markdown; the matched tokens are set in bold.
E2 shortening. A statement longer than MAX_CHARS (240) characters keeps, around each match, WINDOW (70)
   characters on each side, widened outward to the nearest space, overlapping windows merged, each cut
   marked '…'; a statement without a match keeps its first MAX_CHARS characters, widened to the next
   space, then '…'.
E3 locator. '<label>, p. P' from the pages_json of the span's first leaf, 'p. P–Q' when its last leaf
   lies on a later page, the byte range '[b0,b1)' when a leaf carries no page. The label is 'CCTP NN' for
   a piece of the family and clauses.json's own label ('CCAP', 'RC', 'CCAP-A5', 'MAJ-DCE') for a document
   of M7 (AMENDED 2026-09-11 with section (d); it read 'CCTP NN' only).
E4 sections. (a) « Ce qui change par lot »: per CCTP in number order, the ranked statements of
   commitments.json (top3), with the other pieces sharing a statement when its df > 1;
   (b) « Où les documents se contredisent »: the traps of traps.json in rank order, each as its French
   question, its pieces and the locator of its first span (« hors des CCTP » when it has none);
   (c) « Ce qui est commun »: the first TOP_CLUSTERS (10) clusters of template.json (its order: most
   pieces first), each as its reference sentence shortened by E2, its number of pieces and its number
   of varying slots;
   (d) « Ce que fixent le CCAP et le RC » (ADDED 2026-09-11): per document of clauses.json in its
   declared order, the ranked statements of its top10 with their class, their locator and their record
   id; a statement carrying a K7 `cut` match also carries a warning that names the cut value, so a
   number the extraction truncated is never printed as if it were read; a document whose register holds
   no statement says so with its sentence and character counts, never silently (the gap M7 answers was
   a document absent from a register).
E5 ids. commitments NN/k with the sentence hash's first 12 hex digits; traps Tnn; template Cn;
   clauses LABEL/k with the sentence hash's first 12 hex digits.
The header gives the four sources' and this script's sha256 (first 16 hex digits), nothing else.
Gates (refuse, exit 2): the store's sha256 prefix and an empty WAL; every rendered span re-read from the
store and equal to its record's text (N1 for commitments and template records, exact for traps).
Output: collection_fr.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import NoReturn

from . import family as F

#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
MAX_CHARS = 240
WINDOW = 70
TOP_CLUSTERS = 10
BOLD_ON, BOLD_OFF = "\x01", "\x02"


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"render_fr: REFUSED — {msg}\n")
    sys.exit(2)


def title_of(rel: str) -> str:
    name = rel.split("/", 1)[1]
    name = re.sub(r"^\d{2}\.CCTP\s*_?", "", name).replace("Lot _", "").replace(".pdf", "")
    return name.strip(" _")


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_").replace("`", "\\`")


def shorten(s: str) -> str:
    if len(s.replace(BOLD_ON, "").replace(BOLD_OFF, "")) <= MAX_CHARS:
        return s
    marks = [(m.start(), m.end()) for m in re.finditer("\x01[^\x02]*\x02", s)]
    if not marks:
        end = s.find(" ", MAX_CHARS)
        return (s if end == -1 else s[:end]) + " …"
    wins: list[tuple[int, int]] = []
    for a, b in marks:
        w0, w1 = max(0, a - WINDOW), min(len(s), b + WINDOW)
        if w0 > 0:
            sp = s.rfind(" ", 0, w0)
            w0 = sp + 1 if sp != -1 else 0
        if w1 < len(s):
            sp = s.find(" ", w1)
            w1 = sp if sp != -1 else len(s)
        if wins and w0 <= wins[-1][1]:
            wins[-1] = (wins[-1][0], max(wins[-1][1], w1))
        else:
            wins.append((w0, w1))
    out = " … ".join(s[a:b].strip() for a, b in wins)
    return ("… " if wins[0][0] > 0 else "") + out + (" …" if wins[-1][1] < len(s) else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2e — the French collection (S4).")
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
    srcs = {}
    for name in ("commitments.json", "clauses.json", "traps.json", "template.json"):
        path = a.out / name
        if not path.exists():
            fail(f"{name} missing: the earlier stages run first in the same directory")
        srcs[name] = (json.loads(path.read_text(encoding="utf-8")), F.sha256_file(path))
    C, T, TPL = srcs["commitments.json"][0], srcs["traps.json"][0], srcs["template.json"][0]
    K = srcs["clauses.json"][0]
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    text_cache: dict[str, bytes] = {}
    page_cache: dict[str, list[int]] = {}

    def chunk_bytes(cid: str) -> bytes:
        if cid not in text_cache:
            row = db.execute("select text from chunks where chunk_id=?", (cid,)).fetchone()
            if row is None:
                fail(f"chunk {cid[:12]} not in the store")
            text_cache[cid] = row[0].encode("utf-8")
        return text_cache[cid]

    def pages(cid: str) -> list[int]:
        if cid not in page_cache:
            row = db.execute("select pages_json from chunks where chunk_id=?", (cid,)).fetchone()
            page_cache[cid] = json.loads(row[0]) if row and row[0] else []
        return page_cache[cid]

    def locator(label: str, span: dict) -> str:
        pf, pl = pages(span["leaves"]["first"][0]), pages(span["leaves"]["last"][0])
        if pf and pl:
            p0, p1 = min(pf), max(pl)
            return f"{label}, p. {p0}" if p0 >= p1 else f"{label}, p. {p0}–{p1}"
        return f"{label}, octets [{span['bytes'][0]},{span['bytes'][1]})"

    #: A unit word the store broke across lines leaves the grammar's span ending mid-word: « 3 j » of
    #: « 3 j\nours », the value right (P3D) and the span short. The whitespace collapse below then puts a
    #: visible gap in the page — « **3 j** ours ». The coordinating seat's arbitration of 2026-09-12:
    #: **the page is for a reader, the record for the audit**, so the rendering rejoins the word and the
    #: record keeps the raw span untouched. Narrow on purpose: only when the gap contains a NEWLINE, the
    #: span's last token is a one- or two-letter abbreviation, and letters follow immediately — so
    #: « 4 heures » followed by « ouvrées » is never glued into one word.
    def rejoin(rb: bytes, m0: int, m1: int, limit: int) -> tuple[str, int]:
        body = rb[m0:m1].decode("utf-8")
        last = body.split()[-1] if body.split() else ""
        if not (1 <= len(last) <= 2 and last.isalpha()):
            return body, m1
        after = rb[m1:limit].decode("utf-8", "replace")
        joint = re.match(r"([^\S\n]*\n[^\S\n]*)([^\W\d_]+)", after)
        if not joint:
            return body, m1
        return body + joint.group(2), m1 + len(joint.group(0).encode("utf-8"))

    def render(span: dict, matches: list[dict], expect: str, exact: bool) -> str:
        rb = chunk_bytes(span["chunk_id"])
        b0, b1 = span["bytes"]
        raw = rb[b0:b1].decode("utf-8")
        if (raw if exact else F.norm1(raw)) != expect:
            fail(f"span {span['chunk_id'][:12]} [{b0},{b1}) does not match its record")
        parts, pos = [], b0
        for m in sorted(matches, key=lambda m: m["bytes"][0]):
            m0, m1 = m["bytes"]
            if m0 < pos:
                continue
            body, m1 = rejoin(rb, m0, m1, b1)
            parts += [rb[pos:m0].decode("utf-8"), BOLD_ON + body + BOLD_OFF]
            pos = m1
        parts.append(rb[pos:b1].decode("utf-8"))
        s = shorten(re.sub(r"\s+", " ", "".join(parts)).strip())
        return esc(s).replace(BOLD_ON, "**").replace(BOLD_OFF, "**")

    L = [f"# Collection — DCE {CONSULTATION} : les CCTP, le CCAP, le RC", "",
         "Sources : " + " · ".join(f"`{n}` {s[1][:16]}" for n, s in srcs.items()) +
         f" · rendu : `render_fr.py` {F.sha256_file(Path(__file__))[:16]}", "",
         "## Ce qui change par lot", ""]
    n_a = n_b = n_c = 0
    for no in sorted(C["pieces"]):
        p = C["pieces"][no]
        L += [f"### CCTP {no} — {esc(title_of(p['source_path']))}", ""]
        for k, i in enumerate(p["top3"], 1):
            r = p["sentences"][i]
            text = render(r, r["matches"], r["text"], exact=False)
            others = [x for x in r["sharing"] if x != no]
            also = f" · aussi dans CCTP {', '.join(others)}" if others else ""
            L.append(f"{k}. « {text} » — {locator('CCTP ' + no, r)}{also} · réf. commitments {no}/{k}"
                     f" · `{r['hash'][:12]}`")
            n_a += 1
        L.append("")
    L += ["## Où les documents se contredisent", ""]
    for t in T["traps"]:
        pieces = ", ".join(f"CCTP {x}" for x in t["pieces"]) or "hors des CCTP"
        if t["spans"]:
            s0 = t["spans"][0]
            render(s0, [], s0["text"], exact=True)
            loc = locator("CCTP " + s0["piece"], s0)
        else:
            loc = "—"
        L.append(f"{t['rank']}. **{t['id']}** — {esc(t['question_fr'])} — pièces : {pieces} · repère : {loc}"
                 f" · réf. traps {t['id']}")
        n_b += 1
    L += ["", "## Ce qui est commun", ""]
    for n, c in enumerate(TPL["clusters"][:TOP_CLUSTERS], 1):
        ref = c["reference"]
        text = render(ref, [], ref["text"], exact=False)
        vary = sum(s["varies"] for s in c["slots"])
        L.append(f"{n}. « {text} » — commun à {c['df']} des {F.N_CCTP} CCTP, {vary} emplacement(s) variable(s)"
                 f" · repère : {locator('CCTP ' + ref['piece'], ref)} · réf. template C{c['id']}")
        n_c += 1
    L += ["", "## Ce que fixent le CCAP et le RC", ""]
    n_d = 0
    for label in K["declared"]["order"]:
        d = K["documents"][label]
        name = re.sub(r"\.(?:pdf|docx)$", "", d["source_path"])
        L += [f"### {label} — {esc(name)}", ""]
        if not d["top10"]:
            L += [f"Aucun énoncé chiffré ni terme contractuel dans ce document : {d['counts']['distinct']}"
                  f" phrase(s) distincte(s), {d['chars']} caractères dans le magasin.", ""]
            continue
        for k, i in enumerate(d["top10"], 1):
            r = d["sentences"][i]
            text = render(r, r["matches"], r["text"], exact=False)
            cut = [m["text"] for m in r["matches"] if m.get("cut")]
            warn = (" · ⚠ valeur(s) possiblement tronquée(s) dans le texte extrait : "
                    + ", ".join(f"« {esc(re.sub(r'\\s+', ' ', c))} »" for c in cut)) if cut else ""
            L.append(f"{k}. « {text} » — {locator(label, r)} · classe : {r['rank_class']}"
                     f"{warn} · réf. clauses {label}/{k} · `{r['hash'][:12]}`")
            n_d += 1
        L.append("")
    (a.out / "collection_fr.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({"statements": n_a, "traps": n_b, "clusters": n_c, "clauses": n_d, "lines": len(L)},
                     sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
