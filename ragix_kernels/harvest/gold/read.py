#!/usr/bin/env python3
"""demoE2E step 05 — the gold by reading: a read-only navigator over the store, and
the builder that turns a reader's records into a verified gold.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

BLIND BY ITS DECLARED INPUTS. It reads the saqqara store (``mode=ro`` — the
protection is SQLite's, not the reader's), ``pieces.yaml``, ``grid.yaml`` and
``grid_fiche.yaml``, all named in ``INPUTS``; ``build`` refuses if any of them
resolves inside a location the gold must be blind to. That check covers the
DECLARED inputs and nothing more — it cannot prove no other code path exists. The
gold's blindness rests on the reader's discipline and on the log, and says so.
(A first version scanned this file's whole source for forbidden words and refused
on its own docstring: a check that cannot tell prose from an ``open()`` proves
nothing. Replaced 2026-09-10.)

THE READING PATH IS A RECORD, NOT A RECOLLECTION. Every navigation command appends
to ``reading_log.jsonl`` what it OFFERED (a search's hits) and what the reader
OPENED (a chunk, a cell table), tagged by pass and item. The gold's reading path for
an item is read back from that log; it is never written by hand. The log carries
chunk ids and headings — never chunk text.

THE READER SUPPLIES ONLY A CHUNK AND A LITERAL. A record names a chunk id (a prefix
is enough) and the span as the reader copied it; ``build`` resolves the document,
the path, the pages, the section and its heading and the tree nodes from the store
itself, and refuses a span that is not byte-exact in that chunk's text.

    python3 demoE2E/gold/read.py --pass A --item F5 pieces
    python3 demoE2E/gold/read.py --pass A --item F5 outline <doc>
    python3 demoE2E/gold/read.py --pass A --item F5 search "date limite remise" --piece RC
    python3 demoE2E/gold/read.py --pass A --item F5 chunk <chunk> [--at N] [--len M]
    python3 demoE2E/gold/read.py --pass A --item F5 find <chunk> "<literal>"
    python3 demoE2E/gold/read.py --pass A --item F5 cells <doc>
    python3 demoE2E/gold/read.py build --grid fiche --pass A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(os.environ.get("HARVEST_LAB", ".")).resolve()     # the lab root the paths below hang from
HERE = LAB / "demoE2E/gold"
STORE = LAB / "demoE2E/runs/02_collect/20260906T163718/run/saqqara.db"
PIECES = LAB / "demoE2E/03_analyze/pieces.yaml"
GRIDS = {"grid": LAB / "demoE2E/03_analyze/grid.yaml",
         "fiche": LAB / "demoE2E/03_analyze/grid_fiche.yaml"}
LOG = HERE / "reading_log.jsonl"
WORK = HERE / "work"

#: Declared before the first read (runbook row 05): the grid items whose run outcome
#: the reader saw on 2026-09-09, and what was seen. Carried into the gold, so every
#: comparison can be reported with and without them.
PRIOR_EXPOSURE = {
    "Q1": "act B accepted_spans.csv, first row: one accepted span in the RC, pages 1-6",
    "Q13": "G03.6 claim: abstention expected; both arms abstained",
    "Q14": "G03.6 claim: abstention expected; both arms abstained; the outline names it a clause to read by hand",
    "Q15": "G03.6 claim: abstention expected; both arms abstained",
    "Q16": "G03.6 claim: abstention expected; both arms abstained; the only question declaring the annexes",
    "Q21": "G03.6 claim: qwen3:32b supported_with_caveats, mistral-small:24b abstained; a clause to read by hand",
    "Q29": "G03.6 claim: abstention expected; both arms abstained",
    "Q33": "G03.6 claim: qwen3:32b needs_review, mistral-small:24b supported_with_caveats; a clause to read by hand",
    "Q34": "the outline: 'the Q34 artefact' of filter-first retrieval",
}

#: Every file the navigator reads. `build` checks each against the locations the gold
#: is blind to (runbook row 05).
INPUTS = (STORE, PIECES, *GRIDS.values())
BLIND_ROOTS = (LAB / "demoE2E/03_analyze/outputs", LAB / "demoE2E/measurements")
BLIND_PREFIXES = ("DRYRUN_", "FINDINGS_")


def check_inputs_blind() -> None:
    for path in INPUTS:
        rp = path.resolve()
        bad = (any(rp == r or r in rp.parents for r in BLIND_ROOTS)
               or rp.name.startswith(BLIND_PREFIXES) or "traces" in rp.parts)
        if bad:
            raise SystemExit(f"declared input {path} lies inside a location the gold is blind to")

STATUSES = ("answered", "absent", "ambiguous")
CONFIDENCES = ("high", "medium", "low")
MAX_SPAN = 320
# The files whose sha256 row 05 carries. A subset build (--only) may never write one of them.
SEALED = ("gold_grid.yaml", "gold_fiche.yaml", "gold_fiche_passB.yaml")


def db() -> sqlite3.Connection:
    if not STORE.is_file():
        raise SystemExit(f"no store at {STORE}: refusing to create one")
    con = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_pieces() -> dict[str, list[str]]:
    """Any piece name under `pieces:` is accepted — deliberately NOT the hard-coded
    list `analyze.py::load_pieces` matches, which ignores an unknown name silently
    (runbook row 03b, P2). `target_side` is a top-level block of its own; a first
    version read its `paths:` key as a piece called `paths`."""
    pieces, cur, block = {}, None, None
    for line in PIECES.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace():
            block = line.split(":")[0].strip()
            cur = block if block == "target_side" else None
            if cur:
                pieces[cur] = []
            continue
        s = line.strip()
        if block == "pieces":
            m = re.fullmatch(r"([A-Za-z_]+):", s)
            if m and line.startswith("  ") and not line.startswith("    "):
                cur = m.group(1)
                pieces[cur] = []
                continue
        if cur and s.startswith('- "'):
            pieces[cur].append(json.loads(s[2:]))
    return pieces


def corpus_root(con) -> str:
    """Derived from the paths the store holds, never from a fixed depth (the
    executor read's lesson of 2026-09-09: `parts[1]` read `home` for everything)."""
    paths = [r[0] for r in con.execute("select source_path from documents")]
    return os.path.commonpath(paths)


#: The pieces the reader navigates. Each of their paths must resolve to exactly one
#: document of the store, or the navigator stops. Other pieces (the annexes, the
#: response side) are resolved leniently: the store holds 570 of the manifest's 600
#: files — twins deduplicated, unreadable files refused — so an annex path with no
#: document is a known fact of the store, REPORTED with its count and never skipped
#: silently.
STRICT = ("RC", "CCAP", "CCTP")


def piece_docs(con) -> dict[str, list[tuple[str, str]]]:
    root = corpus_root(con)
    docs = [(r["doc_id"], r["source_path"]) for r in con.execute(
        "select doc_id, source_path from documents where trashed = 0")]
    out, unresolved = {}, {}
    for piece, rels in load_pieces().items():
        out[piece] = []
        for rel in rels:
            hit = [d for d, p in docs if p.endswith("/" + rel)]
            if len(hit) != 1:
                if piece in STRICT:
                    raise SystemExit(f"piece {piece}: {rel!r} resolves to {len(hit)} documents")
                unresolved[piece] = unresolved.get(piece, 0) + 1
                continue
            out[piece].append((hit[0], os.path.relpath(
                next(p for d, p in docs if d == hit[0]), root)))
    for piece, n in unresolved.items():
        print(f"(piece {piece}: {n} path(s) of the map hold no document in the store — reported, not read)",
              file=sys.stderr)
    return out


def resolve_doc(con, key: str) -> tuple[str, str]:
    root = corpus_root(con)
    rows = [(r["doc_id"], r["source_path"]) for r in con.execute(
        "select doc_id, source_path from documents")]
    hit = [(d, p) for d, p in rows if d.startswith(key)] or \
          [(d, p) for d, p in rows if key.lower() in os.path.relpath(p, root).lower()]
    if len(hit) != 1:
        raise SystemExit(f"doc {key!r} resolves to {len(hit)} documents: "
                         + "; ".join(os.path.relpath(p, root) for _, p in hit[:6]))
    return hit[0][0], os.path.relpath(hit[0][1], root)


def resolve_chunk(con, key: str) -> sqlite3.Row:
    rows = con.execute("select * from chunks where chunk_id like ?", (key + "%",)).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"chunk {key!r} resolves to {len(rows)} chunks")
    return rows[0]


def log(args, kind: str, rows: list[dict]) -> None:
    if not args.item:
        return
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "pass": args.pass_, "item": args.item, "cmd": args.cmd,
             "args": {k: v for k, v in vars(args).items()
                      if k not in ("cmd", "pass_", "item", "func") and v not in (None, False)},
             kind: rows}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def short(x: str, n: int = 12) -> str:
    return x[:n]


# ----------------------------------------------------------------- navigation

def cmd_pieces(con, args):
    pd = piece_docs(con)
    rows = []
    for piece in ("RC", "CCAP", "CCTP"):
        for d, rel in pd[piece]:
            n0 = con.execute("select count(*) from chunks where doc_id=? and level=0", (d,)).fetchone()[0]
            n1 = con.execute("select count(*) from chunks where doc_id=? and level=1", (d,)).fetchone()[0]
            cls = con.execute("select doc_class from documents where doc_id=?", (d,)).fetchone()[0]
            print(f"{piece:<5} {short(d)} {cls:<5} leaves={n0:<5} sections={n1:<4} {rel}")
            rows.append({"piece": piece, "doc": short(d), "path": rel})
    log(args, "offered", rows)


def cmd_outline(con, args):
    d, rel = resolve_doc(con, args.doc)
    print(f"# {rel}  [{short(d)}]")
    windows = con.execute("select chunk_id, pages_json, length(text) n, "
                          "substr(replace(replace(text, char(10), ' '), char(13), ' '), 1, 90) head "
                          "from chunks where doc_id=? and level=1 order by seq", (d,)).fetchall()
    if windows:
        print(f"  level-1 windows ({len(windows)}) — the store holds no heading path for this "
              f"document, so its outline is read from the windows' opening words:")
        for w in windows:
            pg = json.loads(w["pages_json"])
            pgs = f"p{pg[0]}-{pg[-1]}" if pg else "p-"
            print(f"  {short(w['chunk_id'])} {pgs:<9} chars={w['n']:<6} {w['head']}")
    seen, rows = {}, []
    for r in con.execute("select chunk_id, level, section_path_json, pages_json, length(text) n "
                         "from chunks where doc_id=? order by seq", (d,)):
        sec = tuple(json.loads(r["section_path_json"]))
        pages = json.loads(r["pages_json"])
        e = seen.setdefault(sec, {"first": short(r["chunk_id"]), "pages": set(), "chunks": 0, "chars": 0})
        e["pages"].update(pages)
        e["chunks"] += 1
        e["chars"] += r["n"]
    for sec, e in seen.items():
        pg = sorted(e["pages"])
        pgs = f"p{pg[0]}-{pg[-1]}" if pg else "p-"
        head = " › ".join(sec) if sec else "(no heading)"
        print(f"  {e['first']} {pgs:<9} n={e['chunks']:<3} chars={e['chars']:<6} {head[:150]}")
        rows.append({"section": list(sec), "first_chunk": e["first"], "pages": pg[:1] + pg[-1:]})
    log(args, "offered", [{"doc": short(d), "path": rel, "sections": len(rows)}])
    log(args, "opened", [{"outline": short(d)}])


def fts_query(terms: list[str], any_: bool) -> str:
    q = []
    for t in terms:
        for w in t.split():
            if w in ("OR", "AND"):       # an operator typed by the reader, not a term
                continue
            q.append(w if w.endswith("*") else '"' + w.replace('"', '""') + '"')
    return (" OR " if any_ else " AND ").join(q)


def cmd_search(con, args):
    pd = piece_docs(con)
    allowed = None
    if args.piece:
        allowed = {d for pc in args.piece.split(",") for d, _ in pd[pc]}
    if args.doc:
        allowed = {resolve_doc(con, args.doc)[0]}
    q = fts_query(args.terms, args.any)
    sql = ("select f.chunk_id, snippet(chunks_fts, 1, '⟦', '⟧', '…', 22) snip, "
           "bm25(chunks_fts) score, c.doc_id, c.level, c.section_path_json, c.pages_json "
           "from chunks_fts f join chunks c on c.chunk_id = f.chunk_id "
           "where chunks_fts match ? order by score")
    rows, out = con.execute(sql, (q,)).fetchall(), []
    for r in rows:
        if allowed is not None and r["doc_id"] not in allowed:
            continue
        if args.level is not None and r["level"] != args.level:
            continue
        out.append(r)
        if len(out) >= args.n:
            break
    root_rel = {d: rel for pc in pd.values() for d, rel in pc}
    for r in out:
        sec = " › ".join(json.loads(r["section_path_json"])) or "(no heading)"
        pg = json.loads(r["pages_json"])
        pgs = f"p{pg[0]}" + (f"-{pg[-1]}" if len(pg) > 1 else "") if pg else "p-"
        where = root_rel.get(r["doc_id"], short(r["doc_id"]))
        print(f"{short(r['chunk_id'])} L{r['level']} {pgs:<8} {where[:48]:<48} | {sec[:70]}")
        print(f"    {r['snip']}")
    print(f"-- {len(out)} shown · query {q!r}")
    log(args, "offered", [{"chunk": short(r["chunk_id"]), "level": r["level"],
                           "section": json.loads(r["section_path_json"]),
                           "pages": json.loads(r["pages_json"])} for r in out])


def cmd_chunk(con, args):
    r = resolve_chunk(con, args.chunk)
    t = r["text"]
    a = args.at or 0
    b = len(t) if args.len is None else min(len(t), a + args.len)
    print(f"{short(r['chunk_id'], 16)} L{r['level']} doc {short(r['doc_id'])} "
          f"pages {json.loads(r['pages_json'])} parent {short(r['parent_id'] or '-', 16)}")
    print(f"section: {' › '.join(json.loads(r['section_path_json'])) or '(no heading)'}")
    print(f"chars {len(t)} · showing [{a}:{b}]")
    print(t[a:b])
    log(args, "opened", [{"chunk": short(r["chunk_id"]), "level": r["level"],
                          "section": json.loads(r["section_path_json"]),
                          "pages": json.loads(r["pages_json"]), "window": [a, b]}])


def cmd_page(con, args):
    d, rel = resolve_doc(con, args.doc)
    leaves = [r for r in con.execute("select chunk_id, pages_json, text from chunks "
                                     "where doc_id=? and level=0 order by seq", (d,))
              if args.page in json.loads(r["pages_json"])]
    print(f"# {rel} [{short(d)}] page {args.page}: {len(leaves)} leaves")
    for r in leaves:
        print(f"  {short(r['chunk_id'])}  {r['text'].replace(chr(10), ' ')}")
    log(args, "opened", [{"page": args.page, "doc": short(d), "leaves": len(leaves),
                          "first": short(leaves[0]["chunk_id"]) if leaves else None,
                          "last": short(leaves[-1]["chunk_id"]) if leaves else None}])


def cmd_find(con, args):
    r = resolve_chunk(con, args.chunk)
    t = r["text"]
    hits = [m.start() for m in re.finditer(re.escape(args.literal), t)]
    print(f"{short(r['chunk_id'], 16)}: {len(hits)} exact occurrence(s) at {hits[:10]}")
    log(args, "opened", [{"chunk": short(r["chunk_id"]), "find_hits": len(hits)}])


def cmd_cells(con, args):
    d, rel = resolve_doc(con, args.doc)
    n = 0
    for r in con.execute("select node_id, kind, page, cells_json from objects where doc_id=? "
                         "order by node_id", (d,)):
        if not r["cells_json"]:
            continue
        cells = json.loads(r["cells_json"])
        print(f"== table {r['node_id']} page {r['page']} ({len(cells)} cells)")
        for c in cells[: args.n]:
            print("   ", json.dumps(c, ensure_ascii=False)[:200])
        n += 1
    print(f"-- {n} table(s) with cells in {rel}")
    log(args, "opened", [{"cells": short(d), "tables": n}])


# -------------------------------------------------------------------- build

def yaml_emit(obj, indent: int = 0) -> str:
    """Block YAML with JSON-quoted scalars: deterministic, and readable by any YAML
    parser. No dependency: the lab environment carries none."""
    pad = "  " * indent
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:\n{yaml_emit(v, indent + 1)}")
            else:
                out.append(f"{pad}{k}: {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(out)
    if isinstance(obj, list):
        out = []
        for v in obj:
            if isinstance(v, dict) and v:
                inner = yaml_emit(v, indent + 1).split("\n")
                out.append(f"{pad}- {inner[0].lstrip()}")
                out.extend(inner[1:])
            else:
                out.append(f"{pad}- {json.dumps(v, ensure_ascii=False)}")
        return "\n".join(out)
    return pad + json.dumps(obj, ensure_ascii=False)


def grid_ids(which: str) -> list[str]:
    return [line.split()[-1] for line in GRIDS[which].read_text(encoding="utf-8").splitlines()
            if line.startswith("  - id: ")]


def reading_path(item: str, pass_: str) -> list[dict]:
    if not LOG.is_file():
        return []
    path = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        # `*` is navigation shared by every item of a pass — an outline read once for
        # seventeen facts. It is prepended to each item's path and marked shared, so
        # no single item is credited with a read that served all of them.
        if e["pass"] == pass_ and e["item"] in (item, "*"):
            step = {"cmd": e["cmd"]}
            if e["item"] == "*":
                step["shared"] = True
            if "offered" in e:
                step["offered"] = len(e["offered"])
                if e["cmd"] == "search":
                    step["query"] = " ".join(e["args"].get("terms", []))
                    if "piece" in e["args"]:
                        step["piece"] = e["args"]["piece"]
            if "opened" in e:
                step["opened"] = [
                    {k: v for k, v in o.items()
                     if k in ("chunk", "outline", "cells", "section", "pages", "page", "doc", "leaves")}
                    for o in e["opened"]]
            path.append(step)
    return path


def cmd_build(con, args):
    check_inputs_blind()
    work = WORK / f"{args.grid}_{args.pass_}.jsonl"
    if not work.is_file():
        raise SystemExit(f"no reader records at {work}")
    records = [json.loads(l) for l in work.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = grid_ids(args.grid)
    only = None
    if getattr(args, "only", None):
        # A later pass reads a subset chosen by a script (grid pass 2: select_pass2.py). The
        # sealed passes are never rebuilt through this door, and the gold keeps the grid's
        # order, so the reading order handed to the reader leaves no trace in it.
        name = f"gold_{args.grid}" + ("" if args.pass_ in ("A", "1") else f"_pass{args.pass_}") + ".yaml"
        if name in SEALED:
            raise SystemExit(f"--only never builds {name}: it is sealed in row 05")
        only = Path(args.only).resolve()
        chosen = [l.strip() for l in only.read_text(encoding="utf-8").splitlines() if l.strip()]
        unknown = [q for q in chosen if q not in ids]
        if unknown or len(set(chosen)) != len(chosen):
            raise SystemExit(f"--only: ids not in {args.grid}: {unknown}, or an id repeated")
        ids = [q for q in ids if q in set(chosen)]
    fails = []
    by_id = {}
    for r in records:
        if r["id"] in by_id:
            fails.append(f"{r['id']}: recorded twice")
        by_id[r["id"]] = r
    for q in ids:
        if q not in by_id:
            fails.append(f"{q}: no record — a silent omission")
    for q in by_id:
        if q not in ids:
            fails.append(f"{q}: not a question of {args.grid}")

    root = corpus_root(con)
    trees: dict[str, set[str]] = {}

    def tree_nodes(doc_id: str) -> set[str]:
        """The positional paths of the document's tree. saqqara names a node by its
        position — `1.2` is the third child of the second child of the root — and
        stores no id field; a first draft of this gate looked for one, found none,
        and would have failed every location on a false premise (2026-09-10)."""
        if doc_id not in trees:
            tj = con.execute("select tree_json from documents where doc_id=?", (doc_id,)).fetchone()[0]
            out: set[str] = set()
            stack = [(json.loads(tj)["root"], "")]
            while stack:
                node, path = stack.pop()
                for i, c in enumerate(node.get("children", [])):
                    q = f"{path}.{i}" if path else str(i)
                    out.add(q)
                    stack.append((c, q))
            trees[doc_id] = out
        return trees[doc_id]

    items = []
    counts = {s: 0 for s in STATUSES}
    for q in ids:
        r = by_id.get(q)
        if r is None:
            continue
        st, conf = r.get("status"), r.get("confidence")
        if st not in STATUSES:
            fails.append(f"{q}: status {st!r}")
            continue
        if conf not in CONFIDENCES:
            fails.append(f"{q}: confidence {conf!r}")
        counts[st] += 1
        item = {"id": q, "status": st, "confidence": conf}
        locs = []
        for L in r.get("locations", []):
            try:
                c = resolve_chunk(con, L["chunk"])
            except SystemExit as e:
                fails.append(f"{q}: {e}")
                continue
            span = L["span"]
            if len(span) > MAX_SPAN:
                fails.append(f"{q}: span of {len(span)} characters exceeds {MAX_SPAN}")
            pos = c["text"].find(span)
            if pos < 0:
                fails.append(f"{q}: span not byte-verbatim in chunk {short(c['chunk_id'])}: {span[:50]!r}")
                continue
            nodes = json.loads(c["node_ids_json"] or "[]")
            known = tree_nodes(c["doc_id"])
            if not any(n in known for n in nodes):
                fails.append(f"{q}: chunk {short(c['chunk_id'])} resolves to no node of its document's tree")
            path = con.execute("select source_path from documents where doc_id=?",
                               (c["doc_id"],)).fetchone()[0]
            loc = {"doc_id": c["doc_id"], "path": os.path.relpath(path, root),
                   "pages": json.loads(c["pages_json"]),
                   "section_id": c["parent_id"] or c["chunk_id"],
                   "heading": json.loads(c["section_path_json"]),
                   "chunk_id": c["chunk_id"], "level": c["level"],
                   "node_ids": nodes[:6],
                   "char_offset": pos,
                   "byte_offset": len(c["text"][:pos].encode("utf-8")),
                   "span": span,
                   "span_sha256": hashlib.sha256(span.encode("utf-8")).hexdigest()[:16]}
            if L.get("value") is not None:
                loc["value"] = L["value"]
            if L.get("role"):
                loc["role"] = L["role"]
            locs.append(loc)
        if st == "answered" and not locs:
            fails.append(f"{q}: answered with no location")
        if st == "ambiguous" and len(locs) < 2 and not r.get("conflict"):
            fails.append(f"{q}: ambiguous needs two locations or a stated conflict")
        if st == "absent" and not (r.get("reason") and r.get("pieces_read")):
            fails.append(f"{q}: absent needs a reason and the pieces read")
        if locs:
            item["locations"] = locs
        for k in ("value", "reason", "pieces_read", "conflict", "notes"):
            if r.get(k):
                item[k] = r[k]
        if q in PRIOR_EXPOSURE and args.grid == "grid":
            item["prior_exposure"] = PRIOR_EXPOSURE[q]
        rp = reading_path(q, args.pass_)
        if not rp:
            fails.append(f"{q}: no reading path in the log")
        item["reading_path"] = rp
        items.append(item)

    if fails:
        print(f"BUILD REFUSED — {len(fails)} failure(s):")
        for f in fails:
            print("  ", f)
        return 1

    suffix = "" if args.pass_ in ("A", "1") else f"_pass{args.pass_}"
    out = HERE / f"gold_{args.grid}{suffix}.yaml"
    header = {
        "gold": args.grid, "pass": args.pass_,
        "what": ("an agent's blind reading of the store, not a human's; a reviewer can re-read "
                 "any item from its reading path"),
        "blind_to": ["03_analyze outputs", "the measurements", "the DRYRUN",
                     "the FINDINGS' verdict sections", "every trace"],
        "read_from": ["the saqqara store of 02_collect/20260906T163718, mode=ro",
                      "pieces.yaml", GRIDS[args.grid].name],
        "prior_exposure_items": sorted(PRIOR_EXPOSURE) if args.grid == "grid" else [],
        "inputs_sha256": {
            "store": hashlib.sha256(STORE.read_bytes()).hexdigest(),
            GRIDS[args.grid].name: hashlib.sha256(GRIDS[args.grid].read_bytes()).hexdigest(),
            "pieces.yaml": hashlib.sha256(PIECES.read_bytes()).hexdigest()},
        "counts": counts,
        "gate": "every answered span byte-verbatim in the store; every location resolved to a "
                "tree node; a reading path on every item; one record per question",
    }
    if only is not None:
        header["selected_by"] = {
            "ids_file": str(only.relative_to(LAB)),
            "ids_sha256": hashlib.sha256(only.read_bytes()).hexdigest(),
            "questions": len(ids),
            "the_reader_knows": "the two arms of step 03 agreed on these questions, not on what"}
    text = ("# " + out.name + " — generated by demoE2E/gold/read.py build; edit the reader's "
            "records in work/, never this file\n" + yaml_emit(header) + "\nitems:\n"
            + yaml_emit(items, 1) + "\n")
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"file": str(out.relative_to(LAB)), "items": len(items), "counts": counts,
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="A")
    ap.add_argument("--item", default="")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pieces")
    o = sub.add_parser("outline"); o.add_argument("doc")
    s = sub.add_parser("search"); s.add_argument("terms", nargs="+")
    s.add_argument("--piece"); s.add_argument("--doc"); s.add_argument("--n", type=int, default=12)
    s.add_argument("--any", action="store_true"); s.add_argument("--level", type=int)
    c = sub.add_parser("chunk"); c.add_argument("chunk")
    c.add_argument("--at", type=int); c.add_argument("--len", type=int)
    f = sub.add_parser("find"); f.add_argument("chunk"); f.add_argument("literal")
    g = sub.add_parser("page"); g.add_argument("doc"); g.add_argument("page", type=int)
    e = sub.add_parser("cells"); e.add_argument("doc"); e.add_argument("--n", type=int, default=40)
    b = sub.add_parser("build"); b.add_argument("--grid", choices=list(GRIDS), required=True)
    # `--pass` is accepted after the subcommand too: row 05 and this file's docstring print
    # `build --grid fiche --pass A`, and a first version refused it (2026-09-10). SUPPRESS keeps
    # the top-level value when the option is not repeated here.
    b.add_argument("--pass", dest="pass_", default=argparse.SUPPRESS)
    b.add_argument("--only", default=None,
                   help="a file of ids, one per line: build a later pass over that subset")
    args = ap.parse_args(argv)
    con = db()
    fn = {"pieces": cmd_pieces, "outline": cmd_outline, "search": cmd_search,
          "chunk": cmd_chunk, "find": cmd_find, "page": cmd_page, "cells": cmd_cells,
          "build": cmd_build}[args.cmd]
    return fn(con, args) or 0


if __name__ == "__main__":
    sys.exit(main())
