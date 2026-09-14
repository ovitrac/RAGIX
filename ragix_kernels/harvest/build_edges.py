#!/usr/bin/env python3
"""demoE2E 27 — the graph: derived_edges materialised from what already exists (RUNBOOK row 27).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

CPU only, deterministic, append-only. No model is called and nothing is inferred: every edge is read off a
source the record already holds, by a rule written below, and carries that rule and the source's sha256.

  contains         the store's own tree: a document contains its level-1 roll-ups; a chunk contains the
                   chunks whose parent_id names it.
  summarises       pass 1 (row 22): an abstract is written from these children.
  governed_by      row 14: a deadline claim is governed by each authority that ranks its piece.
  shares_template  M2b: a member span belongs to its template cluster (a hub node per cluster).
  contradicts      M2d: each span a trap sets against the others points to the trap (a hub per trap). A trap
                   is a set of jointly inconsistent spans, not a pair with a direction — T13's twenty-one
                   spans agree with each other and together contradict the AE — so no pairwise edge is
                   asserted. A trap with no span in the record (T15, T16) is anchored to the human reading it
                   was recorded from, and says so; a trap with neither is refused.
  references       the reference grammar over the 24 contractual roll-ups, resolved to the document a
                   reference names (CCTP n, CCAP, RC, AE). Everything else is counted by reason — a page
                   pointer, a code of law, the CCAG, an annex or an article or a BPU without its piece — and
                   nothing is dropped: resolved + self + unresolved = every reference read. An ARTICLE of the
                   CCTP (« article N du CCTP », « article N du présent CCTP ») is never lot N: read in a
                   CCTP it is that CCTP's own article, a reference inside the same document (`self`); read
                   elsewhere it names no lot and is counted unresolved (`resolve_reference`).

Node ids: a chunk is its chunk_id; a document is `doc:<doc_id>`; a claim `claim:<claim_id>`; a span
`span:<chunk_id>@c<start>-<end>` in characters or `@b<start>-<end>` in bytes, as its source records it;
hubs `trap:<id>` and `template:<id>`; a recorded reading `reading:<card>#<ref>`. Block A's pyramid names its
nodes `n_<PIECE>_<doc prefix>`; the summary maps those names to these ids rather than re-keying either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

from .derived import DerivedStore
from .fr.grammars import CHANNEL, read_values

#: the 24 contractual pieces by document id prefix — the declared scope of rows 13 to 20
CORE = {"28b2ce46": "RC", "4dbc921b": "CCAP", "03469319": "17", "0d1bdb93": "10", "0db6cdeb": "13",
        "176401b7": "01", "17c5463b": "14", "200738ee": "09", "2f69201c": "18", "34222d1c": "19",
        "48add6c6": "03", "506606ec": "16", "6b4577fa": "02", "742cb192": "11", "76804a6d": "07",
        "808b166d": "05", "80bcce1b": "04", "86eb27f4": "20", "8ddf9037": "12", "afce1e7c": "00",
        "afed6bfe": "21", "e0e28938": "15", "efe251ec": "06", "f05d9646": "08"}
AE_PREFIX = "b9d02c20"                     # « AE <REF>.docx », read from the store's documents table


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def refuse(message: str) -> None:
    sys.stderr.write(f"build_edges: REFUSED — {message}\n")
    raise SystemExit(2)


def resolve(normalized: str, docs: dict[str, str]) -> tuple[str | None, str]:
    """A reference's target document, or the reason it has none."""
    n = normalized or ""
    if m := re.match(r"^CCTP (\d+)\b", n):
        piece = f"{int(m.group(1)):02d}"
        return (docs.get(piece), "") if piece in docs else (None, "no such CCTP in the DCE")
    for head in ("CCAP", "RC", "AE"):
        if re.match(rf"^{head}\b", n):
            return docs.get(head), ""
    if re.fullmatch(r"\d+/\d+", n):
        return None, "page pointer"
    if re.match(r"^(article )?[LRD]\.?\s?\d", n):
        return None, "external: code of law"
    if n.startswith("CCAG"):
        return None, "external: CCAG"
    for head, why in (("annexe", "an annex without its piece"), ("article", "an article without its piece"),
                      ("BPU", "a BPU without its lot")):
        if n.startswith(head):
            return None, f"ambiguous: {why}"
    return None, "unrecognised form"


#: « article N du CCTP » normalises to « CCTP N », the same form as the lot pointer « CCTP 05 »,
#: and `resolve` read it as lot N; the raw text is what tells an article from a lot. « du présent
#: CCTP » is not part of the grammar's match at all (« article N » is), so it is read from the text
#: that follows the match.
_ARTICLE = re.compile(r"articles?\b", re.IGNORECASE)
_PRESENT_CCTP = re.compile(r"\s+du\s+pr[ée]sent\s+CCTP\b", re.IGNORECASE)


def resolve_reference(v, text: str, citing: str, docs: dict[str, str]) -> tuple[str | None, str]:
    """The document a reference read in `text` points at, or the reason it has none.

    `citing` is the piece of the document the text belongs to (« RC », « CCAP », or a CCTP's two
    digits). An article of the CCTP — « article N du CCTP », « article N du présent CCTP » — is
    never a pointer to lot N: in a CCTP it is that CCTP's own article, the citing document itself;
    anywhere else it names no lot, and is counted unresolved rather than guessed. Every other
    reference goes to `resolve` unchanged.
    """
    if _ARTICLE.match(v.raw or "") and ((v.normalized or "").startswith("CCTP ")
                                        or _PRESENT_CCTP.match(text, v.end)):
        if citing.isdigit():
            return docs.get(citing), ""
        return None, "ambiguous: an article of a CCTP without its lot"
    return resolve(v.normalized, docs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="row 27 — the graph, from what exists")
    for name in ("store", "sidecar", "run-id", "out", "claims", "authority", "pass1", "template", "traps"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)

    store_sha = sha256_file(a.store)
    shas = {"claims": sha256_file(a.claims), "authority": sha256_file(str(Path(a.authority) / "authority.json")),
            "pass1": sha256_file(a.pass1), "template": sha256_file(a.template), "traps": sha256_file(a.traps)}
    db = sqlite3.connect(f"file:{Path(a.store).resolve()}?mode=ro&immutable=1", uri=True)
    doc_ids = [d for (d,) in db.execute("select doc_id from documents")]
    docs = {piece: next(d for d in doc_ids if d.startswith(prefix)) for prefix, piece in CORE.items()}
    docs["AE"] = next((d for d in doc_ids if d.startswith(AE_PREFIX)), None)
    if docs["AE"] is None:
        refuse("the AE is not in the store")

    edges: list[tuple[str, str, str, dict]] = []
    add = lambda kind, src, dst, payload: edges.append((kind, src, dst, payload))   # noqa: E731

    # contains — the store's own tree
    for chunk_id, doc_id, parent in db.execute("select chunk_id, doc_id, parent_id from chunks order by chunk_id"):
        if parent:
            add("contains", parent, chunk_id, {"rule": "store: parent_id", "source_sha256": store_sha})
        else:
            add("contains", f"doc:{doc_id}", chunk_id,
                {"rule": "store: a document contains its level-1 roll-up", "source_sha256": store_sha})

    # summarises — pass 1's children
    for line in Path(a.pass1).read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        for child in r.get("children") or []:
            add("summarises", r["node_id"], child,
                {"rule": "pass 1 (row 22): an abstract is written from these children", "level": r.get("level"),
                 "source_sha256": shas["pass1"]})

    # governed_by — row 14's authorities over the deadline claims
    family_of = {docs["RC"]: "RC", docs["CCAP"]: "CCAP", docs["AE"]: "AE"}
    family_of.update({docs[p]: "CCTP" for p in CORE.values() if p.isdigit()})
    authorities = json.loads((Path(a.authority) / "authority.json").read_text(encoding="utf-8"))["authorities"]
    unranked = 0
    for line in Path(a.claims).read_text(encoding="utf-8").splitlines():
        c = json.loads(line) if line.strip() else None
        if not c or c["field"] != "offer_deadline":
            continue
        family = family_of.get(c["provenance"]["sources"][0]["doc_id"])
        applicable = []
        for au in authorities:
            if au["field"] != c["field"]:
                continue
            # row 14 has two kinds, and each applies differently: a piece ranking applies through its ranks;
            # an own-cover authority governs the whole field. My first rule knew only the first, and dropped
            # RC 6 for all 24 claims without counting it. An unknown kind is refused, never skipped.
            if au["kind"] == "piece_ranking":
                if family in au["ranks"]:
                    applicable.append((au, {"rank": au["ranks"][family]}))
            elif au["kind"] == "own_cover_governs":
                applicable.append((au, {"governs": "the field, through the RC's own cover"}))
            else:
                refuse(f"authority {au['label']} is of kind {au['kind']!r}, which this builder does not know")
        unranked += not applicable
        for au, detail in applicable:
            sp = au["span"]
            add("governed_by", f"claim:{c['claim_id']}", f"span:{sp['chunk_id']}@c{sp['char_start']}-{sp['char_end']}",
                {"rule": f"row 14: a deadline claim is governed by each applicable authority ({au['kind']})",
                 "authority": au["label"], "family": family, **detail,
                 "source_sha256": shas["authority"], "claims_sha256": shas["claims"]})

    # shares_template — M2b's clusters, through a hub per cluster
    for cluster in json.loads(Path(a.template).read_text(encoding="utf-8"))["clusters"]:
        for piece, m in sorted(cluster["members"].items()):
            add("shares_template", f"span:{m['chunk_id']}@b{m['bytes'][0]}-{m['bytes'][1]}", f"template:{cluster['id']}",
                {"rule": "M2b: a member of a template cluster", "piece": piece, "df": cluster.get("df"),
                 "jaccard": m.get("jaccard"), "source_sha256": shas["template"]})

    # contradicts — M2d's traps, through a hub per trap
    traps = json.loads(Path(a.traps).read_text(encoding="utf-8"))["traps"]
    for t in traps:
        anchors = [(f"span:{s['chunk_id']}@b{s['bytes'][0]}-{s['bytes'][1]}",
                    {"rule": "M2d: a span the trap sets against the others", "piece": s.get("piece")})
                   for s in t.get("spans") or []]
        if not anchors:
            anchors = [(f"reading:{e[1]}#{e[2]}", {"rule": "M2d: a trap recorded from a human reading; no span "
                                                            "exists in the record", "anchor": "reading"})
                       for e in t.get("evidence") or [] if e and e[0] == "card"]
        if not anchors:
            refuse(f"trap {t['id']} has neither a span nor a recorded reading: it cannot be present by construction")
        for src, payload in anchors:
            add("contradicts", src, f"trap:{t['id']}", {**payload, "class": t.get("class"), "source_sha256": shas["traps"]})

    # references — the grammar over the 24 roll-ups, resolved where the reference names a document
    refs = {"total": 0, "resolved": 0, "self": 0, "unresolved": {}}
    for chunk_id, doc_id, text in db.execute("select chunk_id, doc_id, text from chunks where level=1 and "
                                             "(parent_id is null or parent_id='') order by chunk_id"):
        if not doc_id.startswith(tuple(CORE)):
            continue
        citing = next(piece for prefix, piece in CORE.items() if doc_id.startswith(prefix))
        for v in read_values(text):
            if v.kind != "reference":
                continue
            refs["total"] += 1
            target, why = resolve_reference(v, text, citing, docs)
            if target is None:
                refs["unresolved"][why] = refs["unresolved"].get(why, 0) + 1
                examples = refs.setdefault("unresolved_examples", {}).setdefault(why, [])
                if len(examples) < 4 and v.normalized not in examples:
                    examples.append(v.normalized)
            elif target == doc_id:
                refs["self"] += 1
            else:
                refs["resolved"] += 1
                add("references", f"span:{chunk_id}@c{v.start}-{v.end}", f"doc:{target}",
                    {"rule": "the reference grammar, resolved to the document the reference names",
                     "raw": v.raw, "normalized": v.normalized, "grammar": CHANNEL, "source_sha256": store_sha})
    db.close()

    Path(a.sidecar).parent.mkdir(parents=True, exist_ok=True)   # a new sidecar's directory; run27 gates block A's
    store = DerivedStore(a.sidecar, source_root=str(Path(a.store).resolve()), source_sha256=store_sha,
                         run_id=a.run_id)
    before = store.conn.total_changes
    for kind, src, dst, payload in edges:
        store.write_edge(kind, src, dst, payload)
    store.commit()
    inserted = store.conn.total_changes - before
    store.close()

    counts: dict[str, int] = {}
    samples: dict[str, dict] = {}
    for kind, src, dst, payload in sorted(edges, key=lambda e: (e[0], e[1], e[2])):
        counts[kind] = counts.get(kind, 0) + 1
        samples.setdefault(kind, {"source": src, "target": dst, "payload": payload})
    summary = {"run_id": a.run_id, "counts": counts, "edges": len(edges), "inserted_this_run": inserted,
               "references": refs, "governed_by_unranked_claims": unranked,
               "trap_hubs": sorted({dst for kind, _, dst, _ in edges if kind == "contradicts"}),
               "samples": samples, "inputs": {"store": store_sha, **shas},
               "namespaces": {"block_A_names": {f"n_{'CCTP' if p.isdigit() else p}_{prefix}": f"doc:{docs[p]}"
                                                for prefix, p in CORE.items()},
                              "note": "block A's pyramid names its nodes n_<PIECE>_<prefix>; this graph uses the "
                                      "store's ids. The map joins them; neither is re-keyed."}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "edges_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"counts": counts, "inserted_this_run": inserted, "references": {k: refs[k] for k in
                                                                                     ("total", "resolved", "self")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
