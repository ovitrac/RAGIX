#!/usr/bin/env python3
"""M2 — the CCTP family: skeleton, shared and unique sentences, and the orphans.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14
Seat: S4 (evaluation and review tooling). Deterministic, CPU only, no model, the store read-only.

DECLARED BEFORE THE FIRST RUN (rule T4). Nothing below is tuned to the gold.

F1 frame. The 22 CCTP paths of demoE2E/03_analyze/pieces.yaml, read by the run's own parser
   (analyze.load_pieces), each matched to exactly one PDF document of the store by path suffix, and
   checked against the 22 n_CCTP_<doc_id prefix> document nodes of row 15
   (15_wp4_slice/outputs/nodes.json). A piece is named by the two digits that open its file name
   (00 … 21); CCTP 00 is the common piece ("Communs à toutes les maintenances").

F2 text. A piece's text is its document roll-up (level 1, parent_id NULL), which must equal
   '\\n'.join(leaf texts in seq order). Every offset is a UTF-8 byte offset [start, end) into that
   roll-up, so a span is re-read from one chunk_id; the first and last leaves it covers are given with
   their own byte offsets.

S1 splitter 'fr-simple/1'. Boundaries in the raw roll-up:
   (a) after '.', '!' or '?' followed by whitespace and then an uppercase letter, a digit, an opening
       quote, parenthesis or bracket, or a bullet glyph; not after a '.' that closes a word of ABBREV
       or a single letter (initials: E.R.P.);
   (b) after ';' followed by whitespace;
   (c) separators, removed from both neighbours: a leaf whose stripped text is one bullet glyph
       (BULLETS, or one Private Use Area character), a run of three or more '.' or of '…'
       (table-of-contents leaders), a page marker 'Page N sur M' or 'Page N/M'.
   Not boundaries: ':', '-', '–', a newline alone (leaves are PDF text runs, not lines of sense).
   Each segment is trimmed of edge whitespace. It is a sentence when its N1 form holds at least
   MIN_WORDS words of two letters or more; otherwise it is a fragment, counted, never partitioned.

N1 normalization (row 09's R9 convention, plus case): whitespace runs collapsed to one space, edges
   stripped, str.casefold(). key = that string; hash = sha256(key, UTF-8). A key repeated inside one
   piece counts once for that piece (repeats are counted).
N2 sensitivity, counted only, never used for a class or an orphan: every whitespace removed, then
   casefold — the reading R9 declined, reported because PDF text runs split words ('Vendre di').

P1 partition. df(key) = the number of the 22 CCTPs holding it. skeleton: df = 22; shared:
   2 <= df <= 21; unique: df = 1.
P2 declared variant: the same over the 21 lot CCTPs (00 excluded); skeleton21: df21 = 21.

T1 tokens. Words are maximal letter runs; a token is the word with accents stripped (NFD, combining
   marks dropped), casefolded, a final 's' or 'x' dropped when longer than four letters; kept when not
   in STOPWORDS (French function words, and the generic nouns every title or cover carries) and at
   least MAP_MIN_LEN letters (lot map) or VOCAB_MIN_LEN letters (orphan vocabularies).
   A piece's title is its cover text between 'Dispositions spécifiques' and 'Date et heure'.

L1 lot map. Lots are read from 'CCTP <REF> - Annexe 3 - Montants maximums par lot.xlsx', from the
   typed tree's cells (sheet, row; column A lot, B designation, C amount); lot numbers must run 1..N,
   contiguous, unique. AMENDED after the first run (2026-09-11 12:43), which read the store roll-up and
   was REFUSED by this gate (lots 32 and 33 twice): the workbook's fallback windows overlap by 200
   characters at each seam and the roll-up repeats them; the source (sha256 f95e3265…, lots 1..85) and
   the tree are clean. The gate is unchanged; only the reading moved from the roll-up to the cells. A designation, its ' - Département …' / ' - Régional' suffix removed, is
   mapped to the lot CCTP whose title tokens have the highest Jaccard index with its tokens; the
   maximum must be unique and at least LOT_MIN_JACCARD, otherwise the lot is unmapped (reported,
   never fires).

O  orphan rules, over the sentences of a lot CCTP P (P != 00) with df <= FEW, one record per distinct
   sentence of P (its first occurrence) and rule, all its evidence listed, nothing filtered after:
   O1 lot: the sentence names a lot or a range ('lot(s) [n°] a [à|au|-|–] b') and a named lot is
      mapped to a piece other than P.
   O2 title vocabulary: the sentence carries a token t with P not in O(t), where O(t) is the set of
      lot CCTPs whose title holds t, and t is vocabulary when 1 <= |O(t)| <= 2.
   O3 unique vocabulary: t is owned by Q (Q != 00) when it occurs in at least UVOC_MIN_SENT distinct
      unique sentences of Q, in no unique sentence of another piece, and in at most FEW pieces in all;
      O3 fires on a shared sentence (2 <= df <= FEW) of P != Q carrying t.

G1 gold, coord's reading (base/HANDOFF_REENTRY_20260911_COORD.md §7), mapped before the run:
   D1 the UPS piece (02) carries the doors piece's (09) object paragraph and regulation section and
      names lots 21-25 -> candidate: an orphan in 02 with owner 09;
   D2 the hoods piece (19) carries the refrigeration piece's (15) temporary-equipment clause ->
      an orphan in 19 with owner 15;
   D3 a trolleys piece (13 or 14) lends 'a UPS until full repair' -> an orphan in 13 or 14, owner 02;
   D4 the trolleys piece's cover renewal line has no year -> no rule targets it; the orphans of 13
      and 14 are listed for reading;
   D5 the extinguishers piece (07) asks for maintenance tables 'of generators' -> an orphan in 07
      with owner 01;
   D6 the handling-equipment piece (12) defines the 1 000-hour visit as 'the 50-hour operations
      completed' -> no rule targets it; the orphans of 12 are listed for reading;
   D7 RC and CCAP number the annexes differently; D8 group 4's weights sum to 52 -> outside the
      22-CCTP frame, not recoverable by construction.
   Candidates are listed mechanically; a recovery is confirmed by reading the candidate's span, and
   that reading goes into the FINDINGS draft, not into this script.

Gates (refuse, exit 2): the store's sha256 prefix and an empty or absent WAL; the frame (22 paths,
one document each, the same 22 as row 15's nodes); every roll-up equal to its joined leaves; the
covers' titles found for the 21 lot CCTPs; the lot numbers contiguous; every emitted span re-read
from the store byte for byte, from its roll-up and from its first and last leaves.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import NoReturn

#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
STORE_SHA_PREFIX = "53ff3f20f655ad66"
N_CCTP = 22
COMMON = "00"
SPLITTER = "fr-simple/1"
MIN_WORDS = 3
FEW = 3
MAP_MIN_LEN = 3
VOCAB_MIN_LEN = 5
LOT_MIN_JACCARD = 0.25
UVOC_MIN_SENT = 2
LOTS_FILE = f"CCTP {CONSULTATION} - Annexe 3 - Montants maximums par lot.xlsx"

ABBREV = frozenset("cf art m mm mme p pp ex env min max réf ref vol chap al n no tél tel fig éd ed st ste".split())
BULLETS = frozenset({"o", "•", "▪", "►", "■", "◦", "●", "➢", "✓"})
FUNCTION_WORDS = ("le la les un une des de du au aux et ou en pour par sur sous dans avec sans ni que qui "
                  "dont ce cet cette ces se sa son ses leur leurs il elle ils elles on nous vous est sont "
                  "être été plus moins tout tous toute toutes chaque autre autres ainsi afin lors entre dès "
                  "vers chez selon ne pas cas même")
GENERIC_NOUNS = ("maintenance préventive corrective technique techniques équipements installations "
                 "multimarques marque multi département régional systèmes production remise température "
                 "professionnelle semi source groupes opérations réseaux appel offres ouvert accord cadre "
                 "fournitures courantes services établissements adhérents achats centre dispositions "
                 "spécifiques date heure limites réception gestion")

END_RE = re.compile(r"[.!?](?=\s+[A-ZÀ-ÖØ-ÞŒ0-9«\"“(\[•-])")
SEMI_RE = re.compile(r";(?=\s)")
LEADER_RE = re.compile(r"\.{3,}|…+")
PAGE_RE = re.compile(r"Page\s*\d+\s*(?:sur|/)\s*\d+", re.IGNORECASE)
ABBR_RE = re.compile(r"([^\W\d_]+)\s*$")
WORD_RE = re.compile(r"[^\W\d_]+")
WORD2_RE = re.compile(r"[^\W\d_]{2,}")
WS_RE = re.compile(r"\s+")
PIECE_RE = re.compile(r"CCTP/(\d{2})\.")
TITLE_RE = re.compile(r"Dispositions\s+sp[ée]cifiques\s+(.*?)\s+Date\s+et\s+heure", re.IGNORECASE)
SUFFIX_RE = re.compile(r"\s*-\s*(?:D[ée]partement\b|R[ée]gional\b).*$", re.IGNORECASE)
LOT_RE = re.compile(r"\blots?\s*(?:n°\s*)?(\d{1,3})(?:\s*(?:à|au|-|–)\s*(\d{1,3}))?", re.IGNORECASE)


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"family: REFUSED — {msg}\n")
    sys.exit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def tok(word: str) -> str:
    w = "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c)).casefold()
    return w[:-1] if len(w) > 4 and w[-1] in "sx" else w


STOPWORDS = frozenset(tok(w) for w in (FUNCTION_WORDS + " " + GENERIC_NOUNS).split())


def tokens(text: str, min_len: int) -> set[str]:
    return {t for t in (tok(w) for w in WORD_RE.findall(text)) if len(t) >= min_len and t not in STOPWORDS}


def norm1(s: str) -> str:
    return WS_RE.sub(" ", s).strip().casefold()


def norm2(s: str) -> str:
    return WS_RE.sub("", s).casefold()


def is_bullet(s: str) -> bool:
    return s in BULLETS or (len(s) == 1 and 0xE000 <= ord(s) <= 0xF8FF)


class Piece:
    def __init__(self, no: str, doc_id: str, rel: str, roll_id: str, text: str, leaves: list[tuple[str, str]]):
        self.no, self.doc_id, self.rel, self.roll_id, self.text, self.leaves = no, doc_id, rel, roll_id, text, leaves
        self.starts, pos = [], 0
        for _, t in leaves:
            self.starts.append(pos)
            pos += len(t) + 1
        self.cum = [0]
        for ch in text:
            self.cum.append(self.cum[-1] + len(ch.encode("utf-8")))
        self.sentences: list[dict] = []
        self.fragments = 0
        self.seps: Counter = Counter()
        self.title = ""

    def split(self) -> None:
        text, n = self.text, len(self.text)
        excl = bytearray(n)
        cuts = {0, n}

        def sep(a: int, b: int, kind: str) -> None:
            excl[a:b] = b"\x01" * (b - a)
            cuts.update((a, b))
            self.seps[kind] += 1

        for (_, t), a in zip(self.leaves, self.starts):
            if is_bullet(t.strip()):
                sep(a, a + len(t), "bullet")
        for m in LEADER_RE.finditer(text):
            sep(m.start(), m.end(), "leader")
        for m in PAGE_RE.finditer(text):
            sep(m.start(), m.end(), "page")
        for m in END_RE.finditer(text):
            p = m.start()
            if text[p] == ".":
                w = ABBR_RE.search(text[max(0, p - 16):p])
                if w and (len(w.group(1)) == 1 or w.group(1).casefold() in ABBREV):
                    continue
            cuts.add(m.end())
        for m in SEMI_RE.finditer(text):
            cuts.add(m.end())
        pos = sorted(cuts)
        for a, b in zip(pos, pos[1:]):
            if a >= b or excl[a]:
                continue
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if a >= b:
                continue
            raw = text[a:b]
            key = norm1(raw)
            if len(WORD2_RE.findall(key)) < MIN_WORDS:
                self.fragments += 1
                continue
            self.sentences.append({"c0": a, "c1": b, "raw": raw, "key": key, "key2": norm2(raw),
                                   "hash": hashlib.sha256(key.encode("utf-8")).hexdigest()})

    def first(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for s in self.sentences:
            out.setdefault(s["key"], s)
        return out

    def locate(self, c0: int, c1: int) -> dict:
        i = bisect.bisect_right(self.starts, c0) - 1
        k = bisect.bisect_right(self.starts, c1 - 1) - 1
        (fi, ft), (li, lt) = self.leaves[i], self.leaves[k]
        return {"chunk_id": self.roll_id, "bytes": [self.cum[c0], self.cum[c1]],
                "leaves": {"first": [fi, len(ft[:c0 - self.starts[i]].encode("utf-8"))],
                           "last": [li, len(lt[:c1 - self.starts[k]].encode("utf-8"))],
                           "count": k - i + 1}}


def reread(db: sqlite3.Connection, cache: dict, chunk_id: str) -> bytes:
    if chunk_id not in cache:
        row = db.execute("select text from chunks where chunk_id=?", (chunk_id,)).fetchone()
        cache[chunk_id] = row[0].encode("utf-8") if row else None
    return cache[chunk_id]


def span_ok(db: sqlite3.Connection, cache: dict, loc: dict, span: str) -> bool:
    sb = span.encode("utf-8")
    roll = reread(db, cache, loc["chunk_id"])
    b0, b1 = loc["bytes"]
    if roll is None or roll[b0:b1] != sb:
        return False
    (fid, foff), (lid, lend) = loc["leaves"]["first"], loc["leaves"]["last"]
    ft, lt = reread(db, cache, fid), reread(db, cache, lid)
    if ft is None or lt is None:
        return False
    if loc["leaves"]["count"] == 1:
        return fid == lid and ft[foff:lend] == sb
    return sb.startswith(ft[foff:]) and sb.endswith(lt[:lend])


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 — the CCTP family characterization (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--lab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    # ---- inputs and their gates -------------------------------------------------------------
    store_sha = sha256_file(a.store)
    if not store_sha.startswith(STORE_SHA_PREFIX):
        fail(f"store sha256 {store_sha[:16]} is not {STORE_SHA_PREFIX}")
    wal = Path(str(a.store) + "-wal")
    wal_bytes = wal.stat().st_size if wal.exists() else 0
    if wal_bytes:
        fail(f"the store's WAL holds {wal_bytes} bytes")
    # immutable as rows 13-15 open it: the WAL is gated empty above, so reads touch no side file
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

    pieces_yaml = a.lab / "demoE2E/03_analyze/pieces.yaml"
    nodes_json = a.lab / "demoE2E/15_wp4_slice/outputs/nodes.json"
    from ..pieces import load_pieces  # the run's own parser of the piece map

    cctp_paths = sorted(load_pieces(pieces_yaml)["CCTP"])
    if len(cctp_paths) != N_CCTP:
        fail(f"pieces.yaml lists {len(cctp_paths)} CCTP paths, not {N_CCTP}")
    docs = db.execute("select doc_id, source_path, doc_class from documents").fetchall()
    pieces: dict[str, Piece] = {}
    for p in cctp_paths:
        m = PIECE_RE.search(p)
        hits = [d for d in docs if d[2] == "pdf" and d[1].endswith("/" + p)]
        if not m or len(hits) != 1:
            fail(f"{p}: {len(hits)} store documents, piece number {'found' if m else 'missing'}")
        no, doc_id = m.group(1), hits[0][0]
        if no in pieces:
            fail(f"piece number {no} twice")
        leaves = db.execute("select chunk_id, text from chunks where doc_id=? and level=0 order by seq",
                            (doc_id,)).fetchall()
        rolls = db.execute("select chunk_id, text from chunks where doc_id=? and level=1 and parent_id is null",
                           (doc_id,)).fetchall()
        if len(rolls) != 1:
            fail(f"{p}: {len(rolls)} document roll-ups")
        if rolls[0][1] != "\n".join(t for _, t in leaves):
            fail(f"{p}: the roll-up is not its leaves joined by newlines")
        pieces[no] = Piece(no, doc_id, "CCTP/" + p.split("CCTP/", 1)[1], rolls[0][0], rolls[0][1], leaves)
    nodes = json.loads(nodes_json.read_text(encoding="utf-8"))["documents"]
    node_prefixes = sorted(k.rsplit("_", 1)[1] for k in nodes if k.startswith("n_CCTP_"))
    frame_prefixes = sorted(pc.doc_id[:len(node_prefixes[0])] for pc in pieces.values()) if node_prefixes else []
    if node_prefixes != frame_prefixes:
        fail("the 22 CCTPs of pieces.yaml are not row 15's 22 CCTP document nodes")
    order = sorted(pieces)
    lot_pieces = [no for no in order if no != COMMON]

    # ---- sentences and the partition --------------------------------------------------------
    for no in order:
        pc = pieces[no]
        pc.split()
        if no != COMMON:
            t = TITLE_RE.search(WS_RE.sub(" ", pc.text))
            if not t:
                fail(f"CCTP {no}: no title between 'Dispositions spécifiques' and 'Date et heure'")
            pc.title = t.group(1)
    firsts = {no: pieces[no].first() for no in order}
    df, df21, dfn2 = Counter(), Counter(), Counter()
    holders: dict[str, list[str]] = defaultdict(list)
    for no in order:
        for k in firsts[no]:
            df[k] += 1
            holders[k].append(no)
            if no != COMMON:
                df21[k] += 1
        for k2 in {s["key2"] for s in pieces[no].sentences}:
            dfn2[k2] += 1

    def cls(d: int, n: int) -> str:
        return "skeleton" if d == n else ("unique" if d == 1 else "shared")

    cache: dict = {}
    spans_checked = 0

    def emit(no: str, s: dict) -> dict:
        nonlocal spans_checked
        loc = pieces[no].locate(s["c0"], s["c1"])
        if not span_ok(db, cache, loc, s["raw"]):
            fail(f"CCTP {no}: span at bytes {loc['bytes']} does not re-read from the store")
        spans_checked += 1
        return loc

    frame, pair = [], {}
    for no in order:
        pc, keys = pieces[no], firsts[no]
        counts = Counter(cls(df[k], N_CCTP) for k in keys)
        counts21 = Counter(cls(df21[k], N_CCTP - 1) for k in keys) if no != COMMON else None
        k2s = {s["key2"] for s in pc.sentences}
        counts_n2 = Counter(cls(dfn2[k2], N_CCTP) for k2 in k2s)
        frame.append({"piece": no, "doc_id": pc.doc_id, "source_path": pc.rel, "rollup_chunk_id": pc.roll_id,
                      "leaves": len(pc.leaves), "chars": len(pc.text), "bytes": pc.cum[-1],
                      "sentences": len(keys), "sentence_occurrences": len(pc.sentences),
                      "repeats_within": len(pc.sentences) - len(keys), "fragments": pc.fragments,
                      "separators": dict(sorted(pc.seps.items())),
                      "counts": {c: counts.get(c, 0) for c in ("skeleton", "shared", "unique")},
                      "counts21": ({c: counts21.get(c, 0) for c in ("skeleton", "shared", "unique")}
                                   if counts21 is not None else None),
                      "counts_n2": {c: counts_n2.get(c, 0) for c in ("skeleton", "shared", "unique")},
                      "title": pc.title or None})
        pair[no] = {o: len(set(keys) & set(firsts[o])) for o in order if o != no}

    def skeleton_of(dmap: Counter, n: int, among: list[str]) -> list[dict]:
        out = []
        for k in sorted(k for k, d in dmap.items() if d == n):
            rep = next(no for no in among if k in firsts[no])
            s = firsts[rep][k]
            out.append({"hash": s["hash"], "piece": rep, **emit(rep, s)})
        return sorted(out, key=lambda r: (r["piece"], r["bytes"][0]))

    skeleton = skeleton_of(df, N_CCTP, order)
    skeleton21 = skeleton_of(df21, N_CCTP - 1, lot_pieces)
    hist = Counter(df.values())
    hist_n2 = Counter(dfn2.values())
    partition = {
        "declared": {"splitter": SPLITTER, "normalization": "N1: whitespace runs collapsed, stripped, casefold",
                     "sensitivity": "N2: all whitespace removed, casefold (counted only)",
                     "min_words": MIN_WORDS, "classes": "skeleton df=22, shared 2..21, unique df=1",
                     "variant": "P2 over the 21 lot CCTPs (00 excluded): skeleton21 df21=21"},
        "frame": frame,
        "df_histogram": {str(d): hist[d] for d in sorted(hist)},
        "distinct_sentences": len(df),
        "skeleton": skeleton,
        "skeleton21": skeleton21,
        "pairwise_shared": pair,
        "sensitivity_n2": {"distinct": len(dfn2), "df_histogram": {str(d): hist_n2[d] for d in sorted(hist_n2)}},
    }

    # ---- the lot map --------------------------------------------------------------------------
    lot_docs = [d for d in docs if d[1].endswith("/" + LOTS_FILE)]
    if len(lot_docs) != 1:
        fail(f"{LOTS_FILE}: {len(lot_docs)} store documents")
    # the typed tree's cells, not the roll-up: an oversize workbook's fallback windows overlap by
    # 200 characters at each seam and its roll-up repeats them (first run, 12:43, refused on lots 32-33)
    tree = json.loads(db.execute("select tree_json from documents where doc_id=?",
                                 (lot_docs[0][0],)).fetchone()[0])
    grid: dict[tuple[str, int], dict[int, tuple[str, str]]] = defaultdict(dict)
    stack = [tree["root"]]
    while stack:
        node = stack.pop()
        if node.get("kind") == "cell":
            pv = node["provenance"]["chain"][0]
            grid[(pv["sheet"], pv["row"])][pv["col"]] = (pv["cell"], (node.get("text") or "").strip())
        stack.extend(node.get("children", []))
    rows = []
    for (sheet, _), cols in sorted(grid.items()):
        a_, b_, c_ = cols.get(1), cols.get(2), cols.get(3)
        if a_ and b_ and c_ and a_[1].isdigit() and not b_[1][:1].isdigit():
            rows.append((a_[1], b_[1], c_[1], f"{sheet}!{a_[0]}"))
    nums = [int(r[0]) for r in rows]
    if not nums or nums != list(range(1, len(nums) + 1)):
        fail(f"{LOTS_FILE}: lot numbers not contiguous from 1 ({len(nums)} rows)")
    title_map = {no: tokens(pieces[no].title, MAP_MIN_LEN) for no in lot_pieces}
    lots, lot_piece = [], {}
    for n, des, amount, cell in rows:
        base = SUFFIX_RE.sub("", des).strip()
        lt = tokens(base, MAP_MIN_LEN)
        scores = {no: (len(lt & tt) / len(lt | tt) if lt | tt else 0.0) for no, tt in title_map.items()}
        best = max(scores.values())
        top = sorted(no for no, s in scores.items() if s == best)
        piece = top[0] if len(top) == 1 and best >= LOT_MIN_JACCARD else None
        if piece:
            lot_piece[int(n)] = piece
        lots.append({"lot": int(n), "cell": cell, "designation": des, "amount": amount, "piece": piece,
                     "jaccard": round(best, 6), "tied": top if len(top) > 1 else []})

    # ---- vocabularies and the orphans -------------------------------------------------------
    title_vocab: dict[str, list[str]] = defaultdict(list)
    for no in lot_pieces:
        for t in sorted(tokens(pieces[no].title, VOCAB_MIN_LEN)):
            title_vocab[t].append(no)
    title_vocab = {t: v for t, v in title_vocab.items() if 1 <= len(v) <= 2}
    tok5 = {no: {k: tokens(s["raw"], VOCAB_MIN_LEN) for k, s in firsts[no].items()} for no in order}
    uniq_cnt: dict[str, Counter] = defaultdict(Counter)
    tok_df: Counter = Counter()
    for no in order:
        seen: set[str] = set()
        for k, ts in tok5[no].items():
            seen |= ts
            if df[k] == 1:
                for t in ts:
                    uniq_cnt[t][no] += 1
        for t in seen:
            tok_df[t] += 1
    owned = {}
    for t, c in uniq_cnt.items():
        (q, nq), others = max(c.items(), key=lambda kv: (kv[1], kv[0])), len(c) - 1
        if q != COMMON and nq >= UVOC_MIN_SENT and others == 0 and tok_df[t] <= FEW:
            owned[t] = q

    orphans = []
    for no in lot_pieces:
        for k, s in sorted(firsts[no].items(), key=lambda kv: kv[1]["c0"]):
            d = df[k]
            if d > FEW:
                continue
            fired = []
            named = []
            for m in LOT_RE.finditer(WS_RE.sub(" ", s["raw"])):
                lo = int(m.group(1))
                hi = int(m.group(2)) if m.group(2) else lo
                named.extend(range(lo, hi + 1) if lo <= hi <= lo + 100 else [lo])
            foreign = sorted({lot_piece[x] for x in named if x in lot_piece and lot_piece[x] != no})
            if foreign:
                fired.append(("O1", {"lots": sorted(set(named))}, foreign))
            ev2 = sorted((t, title_vocab[t]) for t in tok5[no][k] if t in title_vocab and no not in title_vocab[t])
            if ev2:
                fired.append(("O2", {"tokens": [[t, o] for t, o in ev2]}, sorted({x for _, o in ev2 for x in o})))
            if d >= 2:
                ev3 = sorted((t, owned[t]) for t in tok5[no][k] if t in owned and owned[t] != no)
                if ev3:
                    fired.append(("O3", {"tokens": [[t, q] for t, q in ev3]}, sorted({q for _, q in ev3})))
            for rule, evidence, owners in fired:
                orphans.append({"piece": no, "rule": rule, "evidence": evidence, "owners": owners,
                                "class": cls(d, N_CCTP), "df": d, "sharing": holders[k], "hash": s["hash"],
                                "span": s["raw"], **emit(no, s)})
    orphans.sort(key=lambda r: (r["piece"], r["bytes"][0], r["rule"]))
    for i, r in enumerate(orphans):
        r["index"] = i

    # ---- the gold, listed mechanically --------------------------------------------------------
    def cand(where: set[str], owner: str | None) -> list[int]:
        return [r["index"] for r in orphans if r["piece"] in where and (owner is None or owner in r["owners"])]

    gold_spec = {
        "D1": ("UPS piece carries the doors piece's object paragraph and regulation section, names lots 21-25",
               {"02"}, "09"),
        "D2": ("hoods piece carries the refrigeration piece's temporary-equipment clause", {"19"}, "15"),
        "D3": ("a trolleys piece lends 'a UPS until full repair'", {"13", "14"}, "02"),
        "D4": ("the trolleys piece's cover renewal line has no year (no rule targets it; read these)",
               {"13", "14"}, None),
        "D5": ("the extinguishers piece asks for maintenance tables 'of generators'", {"07"}, "01"),
        "D6": ("handling equipment defines the 1 000-hour visit as 'the 50-hour operations completed' "
               "(no rule targets it; read these)", {"12"}, None),
    }
    gold = {}
    for d_id, (what, where, owner) in gold_spec.items():
        c = cand(where, owner)
        gold[d_id] = {"reading": what, "pieces": sorted(where), "owner": owner, "candidates": c,
                      "status": ("to read" if owner is None else ("candidate" if c else "no candidate"))}
    for d_id, what in (("D7", "RC and CCAP number the CCTP annexes differently"),
                       ("D8", "group 4's weights sum to 52")):
        gold[d_id] = {"reading": what, "pieces": [], "owner": None, "candidates": [],
                      "status": "outside the 22-CCTP frame"}

    # ---- outputs ------------------------------------------------------------------------------
    a.out.mkdir(parents=True, exist_ok=True)
    dump(a.out / "partition.json", partition)
    dump(a.out / "orphans.json", {
        "declared": {"few": FEW, "vocab_min_len": VOCAB_MIN_LEN, "uvoc_min_sentences": UVOC_MIN_SENT,
                     "rules": {"O1": "names a lot mapped to another piece", "O2": "title vocabulary of another piece",
                               "O3": "unique vocabulary owned by another piece, on a shared sentence"}},
        "title_vocabulary": dict(sorted(title_vocab.items())),
        "unique_vocabulary_owner": dict(sorted(owned.items())),
        "counts": {"by_rule": dict(sorted(Counter(r["rule"] for r in orphans).items())),
                   "by_piece": dict(sorted(Counter(r["piece"] for r in orphans).items())),
                   "total": len(orphans)},
        "orphans": orphans,
    })
    dump(a.out / "lots.json", {"source": LOTS_FILE, "doc_id": lot_docs[0][0], "min_jaccard": LOT_MIN_JACCARD,
                               "lots": lots, "mapped": len(lot_piece), "unmapped": len(lots) - len(lot_piece)})
    dump(a.out / "gold.json", gold)
    dump(a.out / "manifest.json", {
        "inputs": {"store_sha256": store_sha, "store_wal_bytes": wal_bytes,
                   "pieces_yaml_sha256": sha256_file(pieces_yaml), "nodes_json_sha256": sha256_file(nodes_json)},
        "script_sha256": sha256_file(Path(__file__)),
        "gates": {"store": True, "frame": True, "rollups": True, "titles": True, "lots_contiguous": True,
                  "spans_reread": spans_checked},
        "counts": {"pieces": len(order), "distinct_sentences": len(df),
                   "sentence_occurrences": sum(len(pieces[no].sentences) for no in order),
                   "fragments": sum(pieces[no].fragments for no in order),
                   "skeleton": len(skeleton), "skeleton21": len(skeleton21),
                   "shared": sum(1 for d in df.values() if 2 <= d < N_CCTP),
                   "unique": sum(1 for d in df.values() if d == 1),
                   "lots": len(lots), "lots_mapped": len(lot_piece), "orphans": len(orphans)},
    })
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
