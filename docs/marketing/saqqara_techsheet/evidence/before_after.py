"""
Before → after, per PDF: the measures a reader change can move, from two bench results.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python before_after.py <before.json> <after.json>

Prints one Markdown table per PDF input, one row per measure, with the before and after values and a
mark where they differ. The measures: text anchors outside the page and sections accepted (per text
reader), headings added by the opt-in outline pass (per reader), and, on the default reader, pictures,
caption binding, drawn regions with their refusals and exclusions, and headings promoted by size; then
the Explorer's native tables and the index counts. Every value is read from the two files.
"""

from __future__ import annotations

import json
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def by_id(rows):
    return {row["id"]: row for row in rows}


def measures(result, ident):
    out = {}
    kernel = by_id(result["parts"]["kernel"]["files"])[ident]
    for reader, x in kernel["readers"].items():
        tree = x["runs"][0]["tree"]
        anchors = tree["text_anchors"]
        sec = tree["sections_analyzer"]
        out[f"{reader}: anchors outside the page"] = f"{anchors['outside_page']} / {anchors['anchors']}"
        out[f"{reader}: paragraphs"] = tree["kinds"].get("paragraph", 0)
        out[f"{reader}: sections accepted / candidates"] = f"{sec['accepted']} / {sec['candidates']}"
        out[f"{reader}: sections rejected"] = ", ".join(f"{k} {v}" for k, v in sec["rejected"].items())
        out[f"{reader}: outline pass, inferred headings"] = x["promote_outline"]["inferred_headings"]
        out[f"{reader}: abstentions"] = ", ".join(f"{k} {v}" for k, v in x["runs"][0]["abstentions"].items()) or "0"
    objects = by_id(result["parts"]["objects"]["files"]).get(ident, {})
    if "vector_regions" in objects:
        v, c, h = objects["vector_regions"], objects["caption_binding"], objects["format_headings"]
        out["pictures"] = objects["figures"]
        out["caption binding: bound / figures"] = f"{c['bound']} / {c['figures']}"
        out["caption binding: abstained"] = ", ".join(f"{k} {n}" for k, n in (c.get("abstained") or {}).items()) or "0"
        out["drawn ink: marks / clusters / promoted"] = f"{v['marks']} / {v['clusters']} / {v['promoted']}"
        out["drawn regions refused"] = ", ".join(f"{k} {n}" for k, n in v["refused"].items()) or "0"
        out["drawn marks excluded"] = ", ".join(f"{k} {n}" for k, n in v["excluded"].items()) or "0"
        out["headings by size: promoted"] = h.get("promoted", 0)
        out["headings by size: abstained"] = json.dumps(h.get("abstained")) if h.get("abstained") else "–"
    explorer = by_id(result["parts"]["explorer"]["files"]).get(ident)
    if explorer:
        out["Explorer: tables / cells"] = f"{explorer['tables']} / {explorer['table_cells']}"
    # the bench keeps the per-document index list only while it is short; absent, this row is skipped
    for doc in result["parts"]["cli"]["index"]["data"].get("documents", []):
        if doc["path"] == by_id(result["inputs"])[ident]["path"]:
            out["index: chunks (level 0 / level 1)"] = f"{doc['chunks']} ({doc.get('level_0', 0)} / {doc.get('level_1', 0)})"
    return out


def main() -> None:
    before, after = load(sys.argv[1]), load(sys.argv[2])
    pdfs = [e["id"] for e in before["inputs"] if e["path"].endswith(".pdf")]
    for ident in pdfs:
        b, a = measures(before, ident), measures(after, ident)
        print(f"### {ident}\n")
        print("| measure | before | after | |")
        print("|---|---|---|---|")
        for key in b:
            mark = "" if b[key] == a.get(key) else "moved"
            print(f"| {key} | {b[key]} | {a.get(key)} | {mark} |")
        print()
    print("| input | reader | merkle_root before | merkle_root after | reader version before → after |")
    print("|---|---|---|---|---|")
    fb, fa = by_id(before["parts"]["kernel"]["files"]), by_id(after["parts"]["kernel"]["files"])
    for ident in fb:
        for reader in fb[ident]["readers"]:
            rb, ra = fb[ident]["readers"][reader]["runs"][0], fa[ident]["readers"][reader]["runs"][0]
            print(f"| {ident} | {reader} | {rb['merkle_root'][:16]}… | {ra['merkle_root'][:16]}… "
                  f"| {rb['tree']['reader_kernel_version']} → {ra['tree']['reader_kernel_version']} |")
    for reader in before["parts"]["kernel"]["whole_set"]:
        rb = before["parts"]["kernel"]["whole_set"][reader]["runs"][0]
        ra = after["parts"]["kernel"]["whole_set"][reader]["runs"][0]
        print(f"| whole set | {reader} | {rb['merkle_root'][:16]}… | {ra['merkle_root'][:16]}… | – |")
    print()
    for query in before["parts"]["cli"]["search"]:
        tb = [(t["source_path"].split("/")[-1], t["node_ids"]) for t in before["parts"]["cli"]["search"][query]["top"]]
        ta = [(t["source_path"].split("/")[-1], t["node_ids"]) for t in after["parts"]["cli"]["search"][query]["top"]]
        print(f"search '{query}', top 5 hits identical before and after: {tb == ta}")


if __name__ == "__main__":
    main()
