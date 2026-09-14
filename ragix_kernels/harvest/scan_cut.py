#!/usr/bin/env python3
"""M7b — the cut-value scan: every number the store's text layer split, in claims and in harvested values.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4), on coord's word of 2026-09-11 (seat S4). It measures, it repairs
nothing, it writes nothing but its report. The finding it counts is M7's: a leaf boundary inside a digit
run makes a grammar reading from a word boundary return the tail of a number — « 0,00 € » where the CCAP
prints 250,00 €. Such a span hashes to its own bytes, so every span check passes and the value is wrong.

S0 inputs. --store (saqqara.db, opened mode=ro&immutable=1 after its sha256 prefix and an empty WAL are
   checked); --claims, zero or more JSON-lines files of ClaimRecords; --nodes-jsonl, zero or more of the
   harvest's record files (row 19's `nodes.jsonl` shape: one object per call, `node` a chunk_id, on
   success `summary` and `values`); --out the report. At least one input of either kind, or it refuses.
S1 the rule, the kernel's and the only one. AMENDED 2026-09-11 on the lead's word — "A CR or LF makes
   sense to avoid cutting a §" — and on the measurement behind it: `tender.cut.is_cut` is the project's
   definition, a value is cut when a **newline** falls between the digits, which is what a leaf boundary
   is. My own `cut_before` (any whitespace) detected a **different** defect, the text layer spacing a
   number inside one line ("202 7MT"), and calling that a cut was a mistake of naming; it is retired.
   The two agree on every case that mattered: all ten register cases are newline cases, and over the
   1 553 values the kernel grammar offers, the broader reach produced 53 false positives and not one true
   one. `boundary` is still recorded, and is now always 'leaf' by construction.
   The mirror (a span *ending* on a digit followed by a digit) was declared and withdrawn the same day on
   those same 53: a grammar that crosses whitespace cannot leave a tail behind.
S2 coordinates. Every span is tested in its document's **roll-up**, never inside a leaf or a window: the
   head of a cut number is in the preceding leaf, so a test inside the chunk cannot see it. A chunk is
   placed as check_brief.py's C6 places one — the roll-up itself (level 1, no parent) at 0, a window at
   its `meta.part.span[0]`, a leaf at the sum of the preceding leaves' lengths plus one newline each —
   and the placement is gated: the chunk's text must stand at that offset in the roll-up, or the span is
   counted `unplaceable` and never silently skipped. The application of S1 to a span mirrors
   check_brief.py's C7 and `run.sh` gates the two against each other on the same claims.
S3 claims. Each source of each claim that carries a chunk_id; the claim's own id, the span, its byte range
   in the roll-up. No claim is judged: a cut span is reported whatever the claim's origin or verdict.
S4 harvested values. For each record with a `node`, the node's text is read from the store and the
   **kernel's own grammar** (`tender.grammars_fr.read_values`, used read-only, never modified) is re-run
   on it, which is what the harvest offered the model: value `v<i+1>` for the i-th value, exactly as
   `demoE2E/19_node_harvest/harvest_nodes.py` numbers them. A value is `cited` when its id appears in a
   placeholder of the record's summary, by the form's own PLACEHOLDER (`tender.harvest`); at
   harvest-form/0.5 that pattern accepts `v3`, so the citation is read as the driver reads it.
Exit: 2 on a refusal; **1 when a cut value is cited** by the summary that carries it — that one reaches a
brief; 0 otherwise, cut values present or not. Counting them is the point, hiding them is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, NoReturn

try:
    from .fr.grammars import read_values          # the kernel's grammar, read-only
    from .form import PLACEHOLDER              # the form's own citation pattern
except ImportError:  # pragma: no cover - a refusal, not a fallback
    sys.stderr.write("scan_cut: REFUSED — tender not importable; run with PYTHONPATH=<lab>/src\n")
    sys.exit(2)

try:
    from .fr.cut import VERSION as CUT_VERSION       # the project's ONE definition of a cut
    from .fr.cut import is_cut
except ImportError:  # pragma: no cover - a refusal, not a fallback
    sys.stderr.write("scan_cut: REFUSED — tender.cut not importable: the kernel holds the "
                     "definition of a cut value since 2026-09-11\n")
    sys.exit(2)

STORE_SHA_PREFIX = "53ff3f20f655ad66"
GAP = 24


def refuse(msg: str) -> NoReturn:
    sys.stderr.write(f"scan_cut: REFUSED — {msg}\n")
    sys.exit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def cut_record(text: str, c0: int, c1: int) -> dict[str, Any] | None:
    """S1 applied to one span of `text`: None, or the offender's evidence."""
    span = text[c0:c1]
    if not span or not span[0].isdigit() or not is_cut(text, c0):
        return None
    near = text[max(0, c0 - GAP):c0]
    gap = near[len(near.rstrip()):]
    return {"side": "head", "span": span[:40], "neighbour": near[-16:],
            "boundary": "leaf" if "\n" in gap else "space",
            "bytes": [len(text[:c0].encode("utf-8")), len(text[:c1].encode("utf-8"))]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M7b — the cut-value scan (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--claims", nargs="*", default=[], type=Path)
    ap.add_argument("--nodes-jsonl", nargs="*", default=[], type=Path, dest="nodes_jsonl")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    if not a.claims and not a.nodes_jsonl:
        refuse("nothing to scan: give --claims and/or --nodes-jsonl")
    for p in [a.store, *a.claims, *a.nodes_jsonl]:
        if not p.exists():
            refuse(f"{p} missing")
    store_sha = sha256_file(a.store)
    wal = Path(str(a.store) + "-wal")
    if not store_sha.startswith(STORE_SHA_PREFIX) or (wal.exists() and wal.stat().st_size):
        refuse(f"store {store_sha[:16]}, WAL not empty or wrong store")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

    chunk_cache: dict[str, tuple | None] = {}

    def chunk(cid: str) -> tuple | None:
        if cid not in chunk_cache:
            chunk_cache[cid] = db.execute("select text, level, parent_id, meta_json, doc_id, seq from chunks "
                                          "where chunk_id=?", (cid,)).fetchone()
        return chunk_cache[cid]

    place_cache: dict[str, tuple[str, int] | None] = {}

    def place(cid: str) -> tuple[str, int] | None:
        """S2: (roll-up chunk_id, character offset of the chunk in it), or None when it cannot be placed."""
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

    counts = {"claims_scanned": 0, "claim_sources": 0, "claim_sources_unplaceable": 0, "claims_cut": 0,
              "nodes_scanned": 0, "nodes_unplaceable": 0, "nodes_without_text": 0, "values_read": 0,
              "values_cut": 0, "values_cut_cited": 0, "nodes_with_cut": 0}
    claims_cut: list[dict[str, Any]] = []
    per_file: dict[str, dict[str, int]] = {}
    for path in a.claims:                                                   # S3
        mine = per_file.setdefault(path.name, {"claims": 0, "sources": 0, "cut": 0})
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            c = json.loads(line)
            counts["claims_scanned"] += 1
            mine["claims"] += 1
            cid = c.get("claim_id") or f"{path.name}:{n}"
            for s in (c.get("provenance") or {}).get("sources") or []:
                if not isinstance(s, dict) or "chunk_id" not in s:
                    continue
                counts["claim_sources"] += 1
                mine["sources"] += 1
                where = place(s["chunk_id"])
                roll = chunk(where[0]) if where is not None else None
                if where is None or roll is None:
                    counts["claim_sources_unplaceable"] += 1
                    claims_cut.append({"claim": cid[:12], "chunk": s["chunk_id"][:12],
                                       "unplaceable": True})
                    continue
                rec = cut_record(roll[0], where[1] + s.get("char_start", 0), where[1] + s.get("char_end", 0))
                if rec is not None:
                    counts["claims_cut"] += 1
                    mine["cut"] += 1
                    claims_cut.append({"claim": cid[:12], "chunk": where[0][:12], "file": path.name, **rec})

    nodes: dict[str, dict[str, Any]] = {}
    for path in a.nodes_jsonl:                                              # S4
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            node = r.get("node")
            if not isinstance(node, str):
                refuse(f"{path.name}:{n} has no node")
            counts["nodes_scanned"] += 1
            row = chunk(node)
            if row is None:
                counts["nodes_without_text"] += 1
                nodes[node] = {"file": path.name, "absent_from_the_store": True}
                continue
            text = row[0]
            where = place(node)
            roll = chunk(where[0]) if where is not None else None
            if where is None or roll is None:
                counts["nodes_unplaceable"] += 1
                nodes[node] = {"file": path.name, "unplaceable": True, "chars": len(text)}
                continue
            body = r.get("summary")
            summary = "\n".join(body) if isinstance(body, list) else (body or "")
            cited_ids = set(PLACEHOLDER.findall(summary))
            values = read_values(text)
            counts["values_read"] += len(values)
            cut: list[dict[str, Any]] = []
            for i, v in enumerate(values):
                rec = cut_record(roll[0], where[1] + v.start, where[1] + v.end)
                if rec is None:
                    continue
                vid = f"v{i + 1}"
                counts["values_cut"] += 1
                counts["values_cut_cited"] += vid in cited_ids
                cut.append({"value_id": vid, "kind": v.kind, "raw": v.raw[:40],
                            "normalized": v.normalized, "cited": vid in cited_ids, **rec})
            if cut:
                counts["nodes_with_cut"] += 1
            nodes[node] = {"file": path.name, "roll_up": where[0][:12], "doc_id": row[4][:12],
                           "chars": len(text), "values": len(values), "cited_values": len(cited_ids),
                           "ok": r.get("ok"), "cut": cut}

    claims_cut.sort(key=lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False))
    report = {"declared": {"rule": f"tender.cut.is_cut ({CUT_VERSION}), the kernel's one definition", "gap": GAP,
                           "coordinates": "the document roll-up", "value_ids": "v<i+1> as row 19 numbers",
                           "exit": "1 when a cut value is cited, else 0"},
              "inputs": {"store": store_sha,
                         "claims": {p.name: sha256_file(p) for p in a.claims},
                         "nodes_jsonl": {p.name: sha256_file(p) for p in a.nodes_jsonl}},
              "counts": counts, "per_claims_file": per_file, "claims_cut": claims_cut,
              "nodes": {k: nodes[k] for k in sorted(nodes)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                     encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))
    db.close()
    return 1 if counts["values_cut_cited"] else 0


if __name__ == "__main__":
    sys.exit(main())
