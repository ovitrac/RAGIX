#!/usr/bin/env python3
"""M2b — template alignment of the CCTP family: near-identical sentences across the pieces, their
invariant tokens, and each piece's slot fills (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4). These are M2b's own rules; family.py's rules and the v2
proposals of the FINDINGS draft's §6 are untouched.

A0 input. The 22 CCTPs of family.py's frame F1, read and split exactly as family.py does (F2, S1, N1),
   with one preprocessing of M2b's own:
   A0.1 the running page header 'Cahier des Clauses Techniques Particulières – <REF>' (HEADER_RE,
        whitespace-tolerant) is a separator like the page marker: for this process only, family.PAGE_RE
        is replaced by PAGE_RE | HEADER_RE. Nothing else of the splitter changes; covers stay whole.
   Each piece contributes its distinct sentences (the first occurrence of each N1 key).
A1 tokens. A sentence's tokens are the maximal \\w runs of its raw text (letters, digits, underscore),
   casefolded, each with its character offsets; punctuation and whitespace are not tokens.
A2 similarity. The Jaccard index of two sentences' token sets. An edge joins two sentences of different
   pieces when J >= THETA (0.6). All pairs are compared, pruned only by the size bound min/max >= THETA,
   which no pair with J >= THETA can violate.
A3 clusters. Connected components of the edge graph; inside each, repeatedly: the medoid is the node
   with the most remaining neighbours (ties: the larger sum of J, then the earlier piece and offset);
   the cluster is the medoid plus, for every other piece, its remaining neighbour of highest J with the
   medoid (ties: the earlier offset); the members leave the pool; repeat while an edge remains. Every
   member is thus within THETA of its cluster's medoid, one member per piece, and no chain enters.
A4 template. For a cluster of at least MIN_PIECES (18) pieces the medoid is the reference; each member's
   tokens are aligned to it by difflib.SequenceMatcher (autojunk off). An invariant token is a reference
   token matched ('equal') in every member. A slot is a gap between two consecutive invariant tokens (or
   before the first, or after the last) where at least one member has a token; a member's fill is its
   tokens between its matches of the two invariants, given as its raw span (chunk_id = the piece's
   roll-up, UTF-8 byte offsets, first and last leaves) or null when empty. The template is the invariant
   tokens with <k> at slot k. A slot's variants group the fills by their text with every whitespace
   removed and casefolded, so that PDF fragmentation alone never makes a variant; an empty fill is the
   variant ''. A slot varies when it has two variants or more.
Gates (refuse, exit 2): family.py's own (the store's sha256 prefix and empty WAL, the frame, the
roll-ups); every member and fill span re-read from the store byte for byte.
Outputs: template.json; manifest.json, written by family.py in the same directory, gains a 'template'
section (this script's sha256 and the counts), the rest of the file unchanged.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn

from . import family as F

THETA = 0.6
MIN_PIECES = 18
#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
#: whitespace is tolerated between any two characters of the reference, as a PDF text layer splits it
HEADER_RE = re.compile(r"Cahier\s+des\s+Clauses\s+Techniques\s+Particulières\s*[–-]\s*"
                       + r"\s*".join(re.escape(c) for c in CONSULTATION.replace(" ", "")) + r"(?!\d)")
TOKEN_RE = re.compile(r"\w+")


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"template: REFUSED — {msg}\n")
    sys.exit(2)


def align(ref: list[str], mem: list[str]) -> dict[int, int]:
    sm = difflib.SequenceMatcher(None, ref, mem, autojunk=False)
    out: dict[int, int] = {}
    for a0, b0, size in sm.get_matching_blocks():
        for k in range(size):
            out[a0 + k] = b0 + k
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2b — template alignment of the CCTP family (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--lab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    # ---- inputs and family.py's gates -----------------------------------------------------------
    store_sha = F.sha256_file(a.store)
    if not store_sha.startswith(F.STORE_SHA_PREFIX):
        fail(f"store sha256 {store_sha[:16]} is not {F.STORE_SHA_PREFIX}")
    wal = Path(str(a.store) + "-wal")
    if wal.exists() and wal.stat().st_size:
        fail("the store's WAL is not empty")
    man_path = a.out / "manifest.json"
    if not man_path.exists():
        fail(f"{man_path} missing: family.py runs first in the same directory")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    from ..pieces import load_pieces  # the run's own parser of the piece map

    cctp_paths = sorted(load_pieces(a.lab / "demoE2E/03_analyze/pieces.yaml")["CCTP"])
    if len(cctp_paths) != F.N_CCTP:
        fail(f"pieces.yaml lists {len(cctp_paths)} CCTP paths")
    docs = db.execute("select doc_id, source_path, doc_class from documents").fetchall()
    F.PAGE_RE = re.compile(F.PAGE_RE.pattern + "|" + HEADER_RE.pattern, re.IGNORECASE)  # A0.1
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

    # ---- nodes and edges (A1, A2) ---------------------------------------------------------------
    nodes: list[tuple[str, dict, list[tuple[str, int, int]], frozenset[str]]] = []
    for no in order:
        for _, s in sorted(pieces[no].first().items(), key=lambda kv: kv[1]["c0"]):
            toks = [(mm.group().casefold(), mm.start(), mm.end()) for mm in TOKEN_RE.finditer(s["raw"])]
            if toks:
                nodes.append((no, s, toks, frozenset(t for t, _, _ in toks)))
    by_size = sorted(range(len(nodes)), key=lambda i: (len(nodes[i][3]), i))
    adj: dict[int, dict[int, float]] = defaultdict(dict)
    n_edges = 0
    for x, i in enumerate(by_size):
        si = nodes[i][3]
        li = len(si)
        for j in by_size[x + 1:]:
            sj = nodes[j][3]
            if len(sj) * THETA > li:
                break
            if nodes[i][0] == nodes[j][0]:
                continue
            inter = len(si & sj)
            jac = inter / (li + len(sj) - inter)
            if jac >= THETA:
                adj[i][j] = jac
                adj[j][i] = jac
                n_edges += 1

    # ---- clusters (A3) ----------------------------------------------------------------------------
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in sorted(adj):
        for j in sorted(adj[i]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in sorted(adj):
        comps[find(i)].append(i)
    clusters: list[tuple[int, dict[str, int]]] = []
    for root in sorted(comps):
        remaining = set(comps[root])
        while True:
            best: tuple[tuple[int, float], int] | None = None
            for n in sorted(remaining):
                nb = sorted(m for m in adj[n] if m in remaining)
                if not nb:
                    continue
                score = (len(nb), round(sum(adj[n][m] for m in nb), 9))
                if best is None or score > best[0]:
                    best = (score, n)
            if best is None:
                break
            med = best[1]
            members = {nodes[med][0]: med}
            per_piece: dict[str, list[int]] = defaultdict(list)
            for m in sorted(adj[med]):
                if m in remaining:
                    per_piece[nodes[m][0]].append(m)
            for p, ms in per_piece.items():
                members[p] = max(ms, key=lambda m: (adj[med][m], -m))
            clusters.append((med, members))
            remaining -= set(members.values())

    # ---- templates and fills (A4) -----------------------------------------------------------------
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

    big = [(med, mem) for med, mem in clusters if len(mem) >= MIN_PIECES]
    big.sort(key=lambda cm: (-len(cm[1]), nodes[cm[0]][0], nodes[cm[0]][1]["c0"]))
    out_clusters: list[dict[str, Any]] = []
    n_slots = n_vary = n_fills = 0
    for cid, (med, members) in enumerate(big):
        ref_piece, ref_s, ref_toks, _ = nodes[med]
        ref_words = [t for t, _, _ in ref_toks]
        maps = {p: align(ref_words, [t for t, _, _ in nodes[n][2]]) for p, n in members.items()}
        inv = [r for r in range(len(ref_words)) if all(r in mp for mp in maps.values())]
        anchors = [-1] + inv + [len(ref_words)]
        parts: list[str] = []
        slots: list[dict[str, Any]] = []
        for g in range(len(anchors) - 1):
            lo, hi = anchors[g], anchors[g + 1]
            fills: dict[str, Any] = {}
            for p in sorted(members):
                n = members[p]
                _, s, toks, _ = nodes[n]
                pa = -1 if lo == -1 else maps[p][lo]
                pb = len(toks) if hi == len(ref_words) else maps[p][hi]
                if pb - pa > 1:
                    c0, c1 = s["c0"] + toks[pa + 1][1], s["c0"] + toks[pb - 1][2]
                    fills[p] = {"text": pieces[p].text[c0:c1], **locate(p, c0, c1)}
                else:
                    fills[p] = None
            if any(f is not None for f in fills.values()):
                groups: dict[str, list[str]] = defaultdict(list)
                shown: dict[str, str] = {}
                for p, f in fills.items():
                    key = re.sub(r"\s+", "", f["text"]).casefold() if f else ""
                    groups[key].append(p)
                    shown.setdefault(key, F.norm1(f["text"]) if f else "")
                variants = sorted(({"text": shown[k], "pieces": v} for k, v in groups.items()),
                                  key=lambda d: (-len(d["pieces"]), d["text"]))
                parts.append(f"<{len(slots)}>")
                slots.append({"slot": len(slots), "varies": len(variants) >= 2, "variants": variants,
                              "fills": fills})
                n_slots += 1
                n_vary += len(variants) >= 2
                n_fills += sum(f is not None for f in fills.values())
            if hi < len(ref_words):
                parts.append(ref_words[hi])
        out_clusters.append({
            "id": cid, "df": len(members), "pieces": sorted(members),
            "absent": [p for p in order if p not in members],
            "reference": {"piece": ref_piece, "text": F.norm1(ref_s["raw"]),
                          **locate(ref_piece, ref_s["c0"], ref_s["c1"])},
            "members": {p: {"jaccard": round(adj[med][n], 6) if n != med else 1.0,
                            **locate(p, nodes[n][1]["c0"], nodes[n][1]["c1"])}
                        for p, n in sorted(members.items())},
            "template": " ".join(parts), "invariant_tokens": len(inv), "slots": slots,
        })

    hist = Counter(len(mem) for _, mem in clusters)
    counts = {"nodes": len(nodes), "edges": n_edges, "components": len(comps), "clusters": len(clusters),
              "clusters_by_pieces": {str(k): hist[k] for k in sorted(hist)},
              "clusters_ge_min_pieces": len(big), "slots": n_slots, "slots_varying": n_vary,
              "fills": n_fills, "spans_reread": spans_checked,
              "header_and_page_separators": sum(pc.seps.get("page", 0) for pc in pieces.values())}
    F.dump(a.out / "template.json", {
        "declared": {"theta": THETA, "min_pieces": MIN_PIECES, "tokens": "\\w runs, casefolded",
                     "similarity": "Jaccard over token sets", "alignment": "difflib.SequenceMatcher, autojunk off",
                     "preprocessing": "running page header as a separator (A0.1)"},
        "counts": counts, "clusters": out_clusters})
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["template"] = {"script_sha256": F.sha256_file(Path(__file__)), "counts": counts}
    F.dump(man_path, man)
    print(json.dumps(counts, sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
