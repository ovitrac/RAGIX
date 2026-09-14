#!/usr/bin/env python3
"""M5 — the checker of block G's brief: CLAUDE.md §9 rule 9 (no cited claim may refer to evidence absent
from the payload) and rule 2 (no derived object may lose source provenance) (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4); AMENDED TWICE on 2026-09-11: [form] when the blindness list was
lifted, to align with the harvest form's own definitions, which win where they differ; [C6] on coord's
word, to test that an answer to a trap cites that trap's own evidence. The form's PLACEHOLDER,
CRITICAL_TEXT and FORM_VERSION are imported from tender.harvest; run with PYTHONPATH=<lab>/src.
B0 inputs. --brief (UTF-8 Markdown); --claims (one or more JSON-lines files of ClaimRecords, a claim_id met
   twice with different content refuses); --nodes and --knowledge (block A's shape: {dce, documents});
   --traps (the testing seat's traps.json); --store (saqqara.db, opened mode=ro&immutable=1 after its
   sha256 prefix and an empty WAL are checked).
B1 references. [form] A claim placeholder is exactly the form's PLACEHOLDER, '{{claim:<hex>}}' with 8 to
   64 hex digits and no whitespace, the digits a prefix of a claim_id; it resolves to the one claim whose
   id starts with it; a prefix matching two claims or more is an offender ('ambiguous claim prefix'),
   where the form's render() takes the first. A knowledge reference, this checker's own until block G
   states another, is exactly '{{k:<node>:<sentence>[@<hex>]}}': the row numbered <sentence> of node
   <node> ('n_dce' = knowledge.json's dce.summary; other nodes = its documents), <hex> an optional prefix
   of the row's 'source_sha256' or 'source_hash' field, else of the sha256 of its canonical JSON (sorted
   keys, compact, UTF-8). [C6] A trap marker, this checker's own until block G states another, is
   exactly '{{trap:T<nn>}}' and makes its line the answer to that trap. Any other '{{…}}' is malformed.
C1 claims (rule 9). The cited claim resolves; each of its sources with a chunk_id is re-read from the
   store, text[char_start:char_end], and must hash to its span_sha256; an observed or derived claim's
   value.raw must be one of its sources' spans; an aggregated claim's sources that name a claim_id (a
   string or {'claim_id': …}) must each resolve, recursively (cycles refused), to verified spans; an
   interpreted claim may not carry a value of type date, datetime, amount, percentage, duration or
   quantity (ClaimRecord 1.1, D-0022); a claim with no verifiable source is an offender.
C2 knowledge rows (rule 2). Every cited row exists; its hash, when the reference gives one, matches; each
   claim the row lists passes C1; each child is a node of nodes.json or a claim passing C1.
C3 directions. The brief's headings ('#'…) are matched against DIRECTIONS; each of the five must head a
   section, and each section must be backed (at least one claim or knowledge reference that passes C1 or
   C2; [C6] a trap marker alone does not back) or say 'sans preuve'. A heading matching two directions
   is an offender.
C4 typed numbers. In the brief's non-heading text, after removing every '{{…}}' and the identifiers of
   ALLOWED (piece names with their numbers, the consultation's own reference), any run of digits.
C4b [form] critical text. In the same text with every '{{…}}' replaced by a space (as the form does),
   every hit of the form's CRITICAL_TEXT (a date, a clock time, an amount, a percentage, a duration).
C5 sources. [form] Every non-heading sentence (a line, list markers stripped, split as the form splits a
   summary: after '.', '!' or '?' followed by whitespace; sentences without a letter skipped) must carry
   a claim or knowledge reference or say 'sans preuve'; [C6] a trap marker is not a source.
C6 [C6] trap answers. A line carrying '{{trap:Tnn}}' must name a trap of traps.json and cite at least one
   claim that passes C1 and one of whose sources overlaps one of that trap's spans. Both are placed in
   the same coordinates, the piece's document roll-up: a source's chunk is the roll-up itself (level 1,
   no parent), a window of it (its meta part.span gives its offset), or a leaf (its offset is the sum of
   the preceding leaves' lengths plus one newline each); the chunk's text must stand at that offset in
   the roll-up, or the source cannot be placed. Overlap is a shared roll-up chunk_id and intersecting
   UTF-8 byte ranges. A trap without spans (outside the 22 CCTPs) is exempt and counted.
C7 [C7] cut values (coord's word, 2026-09-11, from M7's finding; AMENDED the same day to the kernel's
   definition). Every claim the brief leans on — cited directly or through a knowledge row — has each of
   its placed source spans tested in the **roll-up's** coordinates: when the span's first character is a
   digit and `tender.cut.is_cut` holds before it, the number's head lies on the far side of a leaf
   boundary and the span holds its tail only. The predicate is the **kernel's**, imported: a newline
   between the digits, which is what a leaf boundary is and what the lead ruled ("A CR or LF makes sense
   to avoid cutting a §"). The CCAP's roll-up holds 'Forfaitaire\n25\n0,00 €' for a penalty of
   250,00 €, and such a span hashes to its span_sha256 and satisfies C1 while stating a wrong critical
   value. Two notes: (a) this seat's earlier predicate (`clauses.cut_before`, any whitespace) is retired
   — it detected a different defect, a number spaced inside one line, and over the 1 553 values the
   kernel grammar offers its extra reach produced 53 false positives and no true one; (b) a cut span is
   an offender even when every other check passes — that is the point of C7.
Output: --out check_brief.json (inputs' sha256 and the form's version, counts, offenders per check,
sorted); exit 1 on any offender, 2 on a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, NoReturn

try:
    from .form import CRITICAL_TEXT, FORM_VERSION, PLACEHOLDER  # the form's own definitions
except ImportError:  # pragma: no cover - a refusal, not a fallback
    sys.stderr.write("check_brief: REFUSED — tender.harvest not importable; run with PYTHONPATH=<lab>/src\n")
    sys.exit(2)

try:
    from .fr.cut import VERSION as CUT_VERSION       # the project's ONE definition of a cut value
    from .fr.cut import is_cut
except ImportError:  # pragma: no cover - a refusal, not a fallback
    sys.stderr.write("REFUSED — tender.cut not importable: the kernel holds the definition of a cut "
                     "value since 2026-09-11; run with PYTHONPATH=<lab>/src\n")
    sys.exit(2)

#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
STORE_SHA_PREFIX = "53ff3f20f655ad66"
CRITICAL_TYPES = {"date", "datetime", "amount", "percentage", "duration", "quantity"}
DIRECTIONS = {
    "go_no_go": r"go\s*/\s*no[\s-]*go",
    "which_parts": r"quel(?:le)?s?\s+(?:lots?|parties)",
    "with_whom": r"avec\s+qui",
    "risks": r"risques?",
    "scope": r"p[ée]rim[èe]tre|objet\s+du\s+march[ée]",
}
ALLOWED = [r"\b(?:CCTP|CCAP|RC|AE|BPU|BPFU|CCAG(?:[- ]FCS)?|DCE)\s*(?:n°\s*)?\d+(?:\.\d+)*\b", r"\b" + re.escape(CONSULTATION) + r"\b"]
ANY_BRACES = re.compile(r"\{\{.*?\}\}", re.S)
K_RE = re.compile(r"\{\{k:(n_[A-Za-z0-9_]+):(\d+)(?:@([0-9a-f]{8,64}))?\}\}")
TRAP_RE = re.compile(r"\{\{trap:(T\d{2})\}\}")
SANS_PREUVE = re.compile(r"sans\s+preuve", re.IGNORECASE)
DIGITS = re.compile(r"\d+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")   # [form] harvest.validate's split of a summary
GAP = 24                                    # [C7] characters read on either side of a span


def refuse(msg: str) -> NoReturn:
    sys.stderr.write(f"check_brief: REFUSED — {msg}\n")
    sys.exit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def row_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M5 — the checker of block G's brief (S4).")
    ap.add_argument("--brief", required=True, type=Path)
    ap.add_argument("--claims", required=True, type=Path, nargs="+")
    ap.add_argument("--nodes", required=True, type=Path)
    ap.add_argument("--knowledge", required=True, type=Path)
    ap.add_argument("--traps", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    for p in [a.brief, a.nodes, a.knowledge, a.traps, a.store, *a.claims]:
        if not p.exists():
            refuse(f"{p} missing")
    store_sha = sha256_file(a.store)
    wal = Path(str(a.store) + "-wal")
    if not store_sha.startswith(STORE_SHA_PREFIX) or (wal.exists() and wal.stat().st_size):
        refuse(f"store {store_sha[:16]}, WAL not empty or wrong store")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

    claims: dict[str, dict] = {}
    for path in a.claims:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            c = json.loads(line)
            cid = c.get("claim_id")
            if not isinstance(cid, str):
                refuse(f"{path.name}:{n} has no claim_id")
            if cid in claims and claims[cid] != c:
                refuse(f"claim {cid[:12]} appears twice with different content")
            claims[cid] = c
    claim_ids = sorted(claims)
    nodes = json.loads(a.nodes.read_text(encoding="utf-8"))
    knowledge = json.loads(a.knowledge.read_text(encoding="utf-8"))
    traps = {t["id"]: t for t in json.loads(a.traps.read_text(encoding="utf-8"))["traps"]}
    node_ids = {"n_dce"} | set(nodes.get("documents", {}))
    rows: dict[tuple[str, int], dict] = {}
    for r in knowledge.get("dce", {}).get("summary", []):
        rows[("n_dce", int(r["sentence"]))] = r
    for node, rs in knowledge.get("documents", {}).items():
        for r in rs:
            rows[(node, int(r["sentence"]))] = r

    chunk_cache: dict[str, tuple | None] = {}

    def chunk(cid: str) -> tuple | None:
        """(text, level, parent_id, meta, doc_id, seq) of a store chunk."""
        if cid not in chunk_cache:
            chunk_cache[cid] = db.execute("select text, level, parent_id, meta_json, doc_id, seq from chunks "
                                          "where chunk_id=?", (cid,)).fetchone()
        return chunk_cache[cid]

    place_cache: dict[str, tuple[str, int] | None] = {}

    def place(cid: str) -> tuple[str, int] | None:
        """(roll-up chunk_id, character offset of the chunk in it), or None when it cannot be placed."""
        if cid in place_cache:
            return place_cache[cid]
        row = chunk(cid)
        where: tuple[str, int] | None = None
        if row is not None:
            text, level, parent, meta, doc_id, seq = row
            if level == 1 and parent is None:
                where = (cid, 0)
            elif parent is not None:
                span = (json.loads(meta or "{}").get("part") or {}).get("span")
                if span:
                    where = (parent, int(span[0]))
                elif level == 0:
                    lens = db.execute("select length(text) from chunks where doc_id=? and level=0 and seq<? "
                                      "order by seq", (doc_id, seq)).fetchall()
                    where = (parent, sum(n + 1 for (n,) in lens))
            if where is not None:
                roll = chunk(where[0])
                if roll is None or roll[0][where[1]:where[1] + len(text)] != text:
                    where = None
        place_cache[cid] = where
        return where

    verdict: dict[str, str | None] = {}  # claim_id -> None (passes) or the reason it fails

    def check_claim(cid: str, stack: tuple[str, ...] = ()) -> str | None:
        if cid in verdict:
            return verdict[cid]
        if cid in stack:
            return "cycle in aggregated sources"
        c = claims.get(cid)
        if c is None:
            verdict[cid] = "claim not in the payload"
            return verdict[cid]
        prov = c.get("provenance") or {}
        origin = prov.get("origin")
        value = c.get("value") or {}
        reason: str | None = None
        spans: list[str] = []
        children: list[str] = []
        for s in prov.get("sources") or []:
            if isinstance(s, str):
                children.append(s)
            elif isinstance(s, dict) and "claim_id" in s and "chunk_id" not in s:
                children.append(s["claim_id"])
            elif isinstance(s, dict) and "chunk_id" in s:
                row = chunk(s["chunk_id"])
                if row is None:
                    reason = reason or f"source chunk {s['chunk_id'][:12]} not in the store"
                    continue
                sp = row[0][s.get("char_start", 0):s.get("char_end", 0)]
                if hashlib.sha256(sp.encode("utf-8")).hexdigest() != s.get("span_sha256"):
                    reason = reason or f"span in {s['chunk_id'][:12]} does not hash to span_sha256"
                    continue
                spans.append(sp)
            else:
                reason = reason or "a source is neither a span nor a claim reference"
        for child in children:
            sub = check_claim(child, stack + (cid,))
            if sub is not None:
                reason = reason or f"child {child[:12]}: {sub}"
        if not spans and not children:
            reason = reason or "no verifiable source"
        if origin in ("observed", "derived") and value.get("raw") is not None and value.get("raw") not in spans:
            reason = reason or "value.raw is none of its sources' spans"
        if origin == "interpreted" and value.get("type") in CRITICAL_TYPES:
            reason = reason or f"an interpreted claim carries a {value.get('type')} value"
        verdict[cid] = reason
        return reason

    def cut_sources(cid: str) -> list[dict[str, Any]]:
        """[C7] the placed sources of a claim whose number the store's text layer cut."""
        out: list[dict[str, Any]] = []
        for s in (claims[cid].get("provenance") or {}).get("sources") or []:
            if not isinstance(s, dict) or "chunk_id" not in s:
                continue
            where = place(s["chunk_id"])
            roll = chunk(where[0]) if where is not None else None
            if where is None or roll is None:
                continue
            text = roll[0]
            c0, c1 = where[1] + s.get("char_start", 0), where[1] + s.get("char_end", 0)
            span = text[c0:c1]
            if not span:
                continue
            if not (span[0].isdigit() and is_cut(text, c0)):         # the kernel's rule, imported
                continue
            near = text[max(0, c0 - GAP):c0]
            gap = near[len(near.rstrip()):]
            out.append({"claim": cid[:12], "side": "head", "chunk": where[0][:12],
                        "bytes": [len(text[:c0].encode("utf-8")), len(text[:c1].encode("utf-8"))],
                        "span": span[:40], "neighbour": near[-16:],
                        "boundary": "leaf" if "\n" in gap else "space"})
        return out

    def claim_byte_ranges(cid: str) -> list[tuple[str, int, int]]:
        """Each placed source of a claim as (roll-up chunk_id, byte start, byte end)."""
        out = []
        for s in (claims[cid].get("provenance") or {}).get("sources") or []:
            if not isinstance(s, dict) or "chunk_id" not in s:
                continue
            where = place(s["chunk_id"])
            roll = chunk(where[0]) if where is not None else None
            if where is None or roll is None:
                continue
            roll_text = roll[0]
            c0, c1 = where[1] + s.get("char_start", 0), where[1] + s.get("char_end", 0)
            b0 = len(roll_text[:c0].encode("utf-8"))
            out.append((where[0], b0, b0 + len(roll_text[c0:c1].encode("utf-8"))))
        return out

    brief = a.brief.read_text(encoding="utf-8")
    off: dict[str, list[dict[str, Any]]] = {k: [] for k in ("claims", "k_rows", "malformed", "directions",
                                                             "numbers", "critical_text", "sources",
                                                             "trap_answers", "cut_values")}
    n_claim_refs = n_k_refs = n_trap_refs = n_trap_unspanned = 0

    def resolve(token: str, line_no: int) -> tuple[str, str | None]:
        """('claim', resolved id or None) | ('k', ok or None) | ('trap', id) | ('bad', None)."""
        nonlocal n_claim_refs, n_k_refs
        m = PLACEHOLDER.fullmatch(token)
        if m:
            n_claim_refs += 1
            prefix = m.group(1)
            hits = [c for c in claim_ids if c.startswith(prefix)]
            if not hits:
                reason: str | None = "claim not in the payload"
            elif len(hits) > 1:
                reason = f"ambiguous claim prefix ({len(hits)} claims)"
            else:
                reason = check_claim(hits[0])
            if reason:
                off["claims"].append({"line": line_no, "claim": prefix, "reason": reason})
                return ("claim", None)
            return ("claim", hits[0])
        m = K_RE.fullmatch(token)
        if m:
            n_k_refs += 1
            row = rows.get((m.group(1), int(m.group(2))))
            reason = None
            if row is None:
                reason = "knowledge row not in the payload"
            else:
                if m.group(3):
                    ref_hash = row.get("source_sha256") or row.get("source_hash") or row_hash(row)
                    if not ref_hash.startswith(m.group(3)):
                        reason = "knowledge row hash does not match the reference"
                for cid in row.get("claims", []):
                    r = check_claim(cid)
                    if r and not reason:
                        reason = f"row claim {cid[:12]}: {r}"
                for ch in row.get("children", []):
                    if ch in node_ids:
                        continue
                    r = check_claim(ch)
                    if r and not reason:
                        reason = f"row child {ch[:12]}: {r}"
            if reason:
                off["k_rows"].append({"line": line_no, "ref": token, "reason": reason})
                return ("k", None)
            return ("k", "ok")
        m = TRAP_RE.fullmatch(token)
        if m:
            return ("trap", m.group(1))
        off["malformed"].append({"line": line_no, "placeholder": token[:80]})
        return ("bad", None)

    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None
    n_sent = 0
    for line_no, line in enumerate(brief.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            hits = [d for d, rx in DIRECTIONS.items() if re.search(rx, title, re.IGNORECASE)]
            if len(hits) > 1:
                off["directions"].append({"line": line_no, "heading": title, "reason": "matches " + ", ".join(hits)})
                current = None
            elif hits:
                current = hits[0]
                sections.setdefault(current, {"line": line_no, "backed": False, "sans_preuve": False})
            else:
                current = None
            continue
        resolved = [resolve(m.group(0), line_no) for m in ANY_BRACES.finditer(line)]
        good_claims = [v for k, v in resolved if k == "claim" and v]
        backed = bool(good_claims) or any(k == "k" and v for k, v in resolved)
        if current:
            sections[current]["backed"] |= backed
            sections[current]["sans_preuve"] |= bool(SANS_PREUVE.search(line))
        for kind, tid in resolved:  # C6
            if kind != "trap":
                continue
            n_trap_refs += 1
            t = traps.get(tid)
            if t is None:
                off["trap_answers"].append({"line": line_no, "trap": tid, "reason": "trap not in traps.json"})
                continue
            if not t["spans"]:
                n_trap_unspanned += 1
                continue
            if not good_claims:
                off["trap_answers"].append({"line": line_no, "trap": tid, "reason": "no passing claim cited"})
                continue
            targets = [(s["chunk_id"], s["bytes"][0], s["bytes"][1]) for s in t["spans"]]
            hit = any(rc == tc and b0 < t1 and t0 < b1
                      for cid in good_claims for rc, b0, b1 in claim_byte_ranges(cid)
                      for tc, t0, t1 in targets)
            if not hit:
                off["trap_answers"].append({"line": line_no, "trap": tid,
                                            "reason": "no cited claim overlaps a span of this trap"})
        blanked = ANY_BRACES.sub(" ", line)
        for m in CRITICAL_TEXT.finditer(blanked):
            off["critical_text"].append({"line": line_no, "text": m.group(0).strip()})
        prose = blanked
        for rx in ALLOWED:
            prose = re.sub(rx, " ", prose)
        for m in DIGITS.finditer(prose):
            off["numbers"].append({"line": line_no, "digits": m.group()})
        body = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)
        for sent in SENT_SPLIT.split(body.strip()):
            refs = [m.group(0) for m in ANY_BRACES.finditer(sent) if not TRAP_RE.fullmatch(m.group(0))]
            if not re.search(r"[^\W\d_]", ANY_BRACES.sub("", sent)) and not refs:
                continue
            n_sent += 1
            if not refs and not SANS_PREUVE.search(sent):
                off["sources"].append({"line": line_no, "sentence": sent.strip()[:120]})
    for cid, reason in sorted(verdict.items()):  # C7
        if reason is not None:
            continue
        off["cut_values"] += cut_sources(cid)
    for d in DIRECTIONS:
        s = sections.get(d)
        if s is None:
            off["directions"].append({"direction": d, "reason": "absent"})
        elif not s["backed"] and not s["sans_preuve"]:
            off["directions"].append({"direction": d, "line": s["line"], "reason": "neither backed nor 'sans preuve'"})

    for k in off:
        off[k].sort(key=lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False))
    counts = {"claim_refs": n_claim_refs, "k_refs": n_k_refs, "trap_refs": n_trap_refs,
              "trap_refs_unspanned": n_trap_unspanned, "claims_checked": len(verdict),
              "claims_failing": sum(1 for v in verdict.values() if v), "sentences": n_sent,
              "directions_present": sorted(sections), **{f"offenders_{k}": len(v) for k, v in off.items()}}
    total = sum(len(v) for v in off.values())
    report = {"inputs": {"brief": sha256_file(a.brief), "claims": [sha256_file(p) for p in a.claims],
                         "nodes": sha256_file(a.nodes), "knowledge": sha256_file(a.knowledge),
                         "traps": sha256_file(a.traps), "store": store_sha, "form_version": FORM_VERSION},
              "counts": counts, "offenders": off, "pass": total == 0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": total == 0, **{k: v for k, v in counts.items() if k.startswith("offenders")}},
                     sort_keys=True))
    db.close()
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
