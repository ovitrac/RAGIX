"""
Print the Markdown tables of EVIDENCE.md from one or two bench results.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python summarize.py results_m1_x86_64.json [results_m2_aarch64.json]
    python summarize.py --timings <results.json> [<results.json> ...]

Every number in the tables is read from the JSON given; nothing is typed by hand. With two results,
a last table compares the roots and the path-free tree digests host by host.
"""

from __future__ import annotations

import json
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def fmt_s(values):
    return " / ".join(f"{v:.3f}" for v in values)


def kernel_table(result) -> None:
    host = result["host"]
    print(f"Host: {host['machine']}, {host['cpu']}, {host['logical_cpus']} logical CPUs, "
          f"Python {host['versions']['python']}, code digest {result['code']['digest'][:16]}…, "
          f"{result['runs']} runs per file.\n")
    print("| input | reader | pages | nodes | kinds | anchors off page | tables (rows×cols, cells) "
          "| headings (+outline) | sections accepted / candidates | abstentions | wall s (run 1 / 2) "
          "| s per page | roots equal |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in result["parts"]["kernel"]["files"]:
        for reader, x in f["readers"].items():
            first = x["runs"][0]
            tree = first["tree"]
            tables = "; ".join(f"{t['rows']}×{t['cols']}, {t['cell_nodes']} ({t['type']})"
                               for t in tree["tables"]) or "0"
            kinds = ", ".join(f"{k} {v}" for k, v in tree["kinds"].items() if k != "document")
            sec = tree["sections_analyzer"]
            anchors = tree["text_anchors"]
            off = f"{anchors['outside_page']} / {anchors['anchors']}" if anchors["anchors"] else "–"
            abst = ", ".join(f"{k} {v}" for k, v in first["abstentions"].items()) or "0"
            per_page = fmt_s(x["wall_s_per_page"]) if x["wall_s_per_page"] else "–"
            name = f["id"] + (" (SYNTHETIC)" if f["synthetic"] else "")
            print(f"| {name} | {reader} | {tree['pages'] or '–'} | {tree['nodes']} | {kinds} | {off} | {tables} "
                  f"| {tree['headings']} (+{x['promote_outline']['inferred_headings']}) "
                  f"| {sec['accepted']} / {sec['candidates']} | {abst} | {fmt_s(x['wall_s'])} | {per_page} "
                  f"| {'yes' if x['roots_equal_across_runs'] else 'NO'} |")
    print()
    for reader, w in result["parts"]["kernel"]["whole_set"].items():
        r = w["runs"][0]
        print(f"Whole input set, {reader}: {r['summary']} merkle_root {r['merkle_root']}, source_root "
              f"{r['source_root']}, refusals {r['refusals']}, wall s {fmt_s([x['wall_s'] for x in w['runs']])}, "
              f"roots equal across runs: {w['roots_equal_across_runs']}.")
    p = result["parts"]["kernel"]["path_test"]
    print(f"\nSame bytes from another path ({p['moved']}): merkle_root equal {p['merkle_root_equal']}, "
          f"source_root equal {p['source_root_equal']}, path-free tree equal {p['pathfree_tree_equal']}.\n")


def objects_table(result) -> None:
    print("| input | pictures (assets) | drawn ink: marks / clusters / promoted regions | regions refused or excluded "
          "| caption binding: figures / bound / abstained | headings by size: promoted (nodes before → after) |")
    print("|---|---|---|---|---|---|")
    for row in result["parts"]["objects"]["files"]:
        if "vector_regions" not in row:
            continue
        v, c, h = row["vector_regions"], row["caption_binding"], row["format_headings"]
        refused = ", ".join(f"{k} {n}" for k, n in {**v["refused"], **v["excluded"]}.items()) or "0"
        abst = ", ".join(f"{k} {n}" for k, n in (c.get("abstained") or {}).items()) or "0"
        before, after = row["heading_nodes_before_after"]
        print(f"| {row['id']} | {row['figures']} ({row['assets']}) | {v['marks']} / {v['clusters']} / {v['promoted']} "
              f"| {refused} | {c['figures']} / {c['bound']} / {abst} | {h.get('promoted', 0)} ({before} → {after}) |")
    print()


def explorer_table(result) -> None:
    print("| input | extractor | pages | text spans | drawings | native tables | rows per table | cells "
          "| cells without geometry | digest s | explore s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in result["parts"]["explorer"]["files"]:
        print(f"| {row['id']} | {row['extractor']} | {row['pages']} | {row['spans']} | {row['drawings']} "
              f"| {row['tables']} | {', '.join(map(str, row['table_rows'])) or '–'} | {row['table_cells']} "
              f"| {row['cells_missing_geometry']} | {row['digest_wall_s']:.3f} | {row['explore_wall_s']:.3f} |")
    print()


def cli_table(result) -> None:
    cli = result["parts"]["cli"]
    status = cli["index"]["data"]["status"]
    print(f"saqqaractl run: counts {cli['run']['counts']}, merkle_root {cli['run']['merkle_root']}, "
          f"wall {cli['run']['wall_s']:.3f} s. index: {status['documents']} documents, {status['chunks']} chunks, "
          f"{status['objects']} objects, {status['edges']} edges, dense lane '{status['dense']}', "
          f"wall {cli['index']['wall_s']:.3f} s.")
    for query, s in cli["search"].items():
        tops = "; ".join(f"#{t['final_rank']} {t['kind']} p.{t['pages']} «{t['text'][:40]}» "
                         f"({t['source_path'].split('/')[-1]})" for t in s["top"][:3])
        print(f"search '{query}': {s['hits']} hits in {s['wall_s']:.3f} s; top 3: {tops}")
    print()


def compare(a, b) -> None:
    print(f"| input | reader | merkle_root {a['host']['machine']} | merkle_root {b['host']['machine']} | equal "
          f"| path-free tree equal | source_root equal |")
    print("|---|---|---|---|---|---|---|")
    fb = {f["id"]: f for f in b["parts"]["kernel"]["files"]}
    for f in a["parts"]["kernel"]["files"]:
        for reader, x in f["readers"].items():
            y = fb[f["id"]]["readers"][reader]
            ra, rb = x["runs"][0], y["runs"][0]
            print(f"| {f['id']} | {reader} | {ra['merkle_root'][:16]}… | {rb['merkle_root'][:16]}… "
                  f"| {'yes' if ra['merkle_root'] == rb['merkle_root'] else 'NO'} "
                  f"| {'yes' if ra['tree']['pathfree_tree_sha256'] == rb['tree']['pathfree_tree_sha256'] else 'NO'} "
                  f"| {'yes' if ra['source_root'] == rb['source_root'] else 'NO'} |")
    for reader in a["parts"]["kernel"]["whole_set"]:
        ra = a["parts"]["kernel"]["whole_set"][reader]["runs"][0]
        rb = b["parts"]["kernel"]["whole_set"][reader]["runs"][0]
        print(f"| whole set | {reader} | {ra['merkle_root'][:16]}… | {rb['merkle_root'][:16]}… "
              f"| {'yes' if ra['merkle_root'] == rb['merkle_root'] else 'NO'} | – "
              f"| {'yes' if ra['source_root'] == rb['source_root'] else 'NO'} |")
    print(f"\nCode digest equal: {a['code']['digest'] == b['code']['digest']} "
          f"({a['code']['digest'][:16]}… / {b['code']['digest'][:16]}…).")


def timings(results) -> None:
    """Seconds per page (or per file, where a file has no pages), run 1 / run 2, one column per result."""
    head = " | ".join(f"{r['host']['machine']}, load {r['host']['loadavg_at_start'][0]:.2f}→"
                      f"{r['host']['loadavg_at_end'][0]:.2f}" for r in results)
    print(f"| input | reader | {head} |")
    print("|---|---|" + "---|" * len(results))
    for i, f in enumerate(results[0]["parts"]["kernel"]["files"]):
        for reader in f["readers"]:
            cells = []
            for r in results:
                x = r["parts"]["kernel"]["files"][i]["readers"][reader]
                cells.append(fmt_s(x["wall_s_per_page"] or x["wall_s"]) + ("" if x["wall_s_per_page"] else " (per file)"))
            print(f"| {f['id']} | {reader} | " + " | ".join(cells) + " |")
    print("| whole set, s | pymupdf / pypdf | " + " | ".join(
        "; ".join(fmt_s([x["wall_s"] for x in r["parts"]["kernel"]["whole_set"][k]["runs"]]) for k in ("pymupdf", "pypdf"))
        for r in results) + " |")
    print()


def main() -> None:
    if sys.argv[1] == "--timings":
        timings([load(p) for p in sys.argv[2:]])
        return
    results = [load(p) for p in sys.argv[1:]]
    for result in results:
        print(f"## {result['host']['machine']}\n")
        kernel_table(result)
        objects_table(result)
        explorer_table(result)
        cli_table(result)
    if len(results) == 2:
        print("## Across hosts\n")
        compare(*results)


if __name__ == "__main__":
    main()
