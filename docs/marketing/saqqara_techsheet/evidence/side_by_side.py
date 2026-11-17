"""
Saqqara and Docling side by side, per input, from one Saqqara bench result and one Docling bench result.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python side_by_side.py <saqqara results.json> <docling results.json> [--rss <saqqara_rss.json>]

Every value is read from the two files. Where the two tools do not measure the same thing, the row says
what each one counts; the table does not convert one into the other. Saqqara's text figures are those of
its default reader (pypdf) unless a row names the PyMuPDF reader; its PDF tables are the Explorer's
(PyMuPDF), its pictures, drawn regions and size-promoted headings are the library passes (K6).
"""

from __future__ import annotations

import argparse
import json


def by_id(rows, key="id"):
    return {r[key]: r for r in rows}


def fmt(values):
    return " / ".join(f"{v:.3f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("saqqara")
    parser.add_argument("docling")
    parser.add_argument("--rss")
    args = parser.parse_args()
    s, d = (json.load(open(p)) for p in (args.saqqara, args.docling))
    rss = by_id(json.load(open(args.rss))["files"]) if args.rss else {}
    kernel = by_id(s["parts"]["kernel"]["files"])
    objects = by_id(s["parts"]["objects"]["files"])
    explorer = by_id(s["parts"]["explorer"]["files"])
    ids = [e["id"] for e in d["inputs"]]
    for ident in ids:
        runs = [r for r in d["runs"] if r["id"] == ident]
        r4 = [r for r in runs if r["threads"] == min(x["threads"] for x in runs)]
        rn = [r for r in runs if r["threads"] == max(x["threads"] for x in runs)]
        first = runs[0]
        k = kernel.get(ident)
        reader = "pypdf" if k and "pypdf" in k["readers"] else "default"
        tree = k["readers"][reader]["runs"][0]["tree"] if k else None
        pages = first["pages"]
        print(f"### {ident}\n")
        print(f"| measure | Saqqara | Docling 2.150.1 |")
        print("|---|---|---|")
        print(f"| pages | {tree['pages'] if tree else '–'} | {pages or '–'} |")
        if tree:
            kinds = ", ".join(f"{a} {b}" for a, b in tree["kinds"].items() if a != "document")
        else:
            kinds = "–"
        print(f"| elements by type | {kinds} | {', '.join(f'{a} {b}' for a, b in first['labels'].items())} |")
        ex = explorer.get(ident)
        stab = (f"{ex['tables']} tables, {ex['table_cells']} cells (Explorer, PyMuPDF); rows per table "
                f"{', '.join(map(str, ex['table_rows'])) or '–'}") if ex else (
            "; ".join(f"{t['rows']}×{t['cols']}, {t['cell_nodes']} cells" for t in tree["tables"]) if tree and tree["tables"]
            else "0 (the Markdown reader types no table)" if ident.endswith("_md") else "0")
        shapes = ", ".join("%d×%d" % (t["rows"], t["cols"]) for t in first["tables"]) or "–"
        dtab = f"{len(first['tables'])} tables, {sum(t['cells'] for t in first['tables'])} cells; rows×cols {shapes}"
        print(f"| tables | {stab} | {dtab} |")
        if tree:
            ob = objects.get(ident, {})
            size = ob.get("format_headings", {}).get("promoted")
            heads = (f"{tree['headings']} declared; +{k['readers'][reader]['promote_outline']['inferred_headings']} "
                     f"by the opt-in outline pass" + (f"; {size} promoted by size (library)" if size is not None else ""))
        else:
            heads = "–"
        print(f"| headings | {heads} | {', '.join(f'{a} {b}' for a, b in first['headings'].items()) or '0'} |")
        if ident in objects and "vector_regions" in objects[ident]:
            ob = objects[ident]
            figs = f"{ob['figures']} pictures; {ob['vector_regions']['promoted']} drawn regions (library passes)"
        else:
            figs = "–"
        print(f"| figures | {figs} | {first['pictures']} pictures |")
        prov = first["provenance"]
        sprov = ("every node: source path, format, locator chain (page; for text the anchor x, y, width and "
                 "font size; for pictures and regions a box), reader and version") if tree else "–"
        print(f"| provenance | {sprov} | {prov['items_with_prov']} of {first['items']} items with prov; "
              f"bbox {prov['with_bbox']}, charspan {prov['with_charspan']}; origin {prov['bbox_origin'] or '–'} |")
        conf = first.get("confidence") or {}
        sab = (", ".join(f"{a} {b}" for a, b in k["readers"][reader]["runs"][0]["abstentions"].items()) or "0") if k else "–"
        print(f"| uncertainty | abstentions: {sab}; read nodes carry confidence 1.0, inferred ones less | "
              f"page grades: mean {conf.get('mean_grade')}, low {conf.get('low_grade')} |")
        if tree and tree["pages"]:
            sp = fmt(k["readers"][reader]["wall_s_per_page"])
        elif k:
            sp = fmt(k["readers"][reader]["wall_s"]) + " per file"
        else:
            sp = "–"
        unit = (lambda r: r["convert_s"] / pages) if pages else (lambda r: r["convert_s"])
        dp4 = fmt([unit(r) for r in r4]) + ("" if pages else " per file")
        dpn = fmt([unit(r) for r in rn]) + ("" if pages else " per file")
        print(f"| seconds per page, run 1 / run 2 | {sp} (one process) | {dp4} at {r4[0]['threads']} threads; "
              f"{dpn} at {rn[0]['threads']} threads (conversion only; model load {fmt([r['init_s'] for r in r4])} s) |")
        det = d["determinism"][ident]
        if k:
            sdet = "roots equal across runs" if k["readers"][reader]["roots_equal_across_runs"] else "roots DIFFER"
        else:
            sdet = "–"
        same = lambda rows: len({r["json_sha256"] for r in rows}) == 1
        verdict = lambda ok: "identical" if ok else "DIFFER"
        print(f"| same output twice | {sdet} | JSON, {r4[0]['threads']} threads, run 1 vs 2: {verdict(same(r4))}; "
              f"{rn[0]['threads']} threads: {verdict(same(rn))}; across thread counts: "
              f"{verdict(same([r4[0], rn[0]]))} (no time stamp in the export) |")
        drss = max(r["peak_rss_kib"] for r in runs)
        srss = f"{rss[ident]['peak_rss_kib']} KiB" if ident in rss else "not measured"
        print(f"| peak resident memory | {srss} | {drss} KiB (largest over runs) |")
        print()


if __name__ == "__main__":
    main()
