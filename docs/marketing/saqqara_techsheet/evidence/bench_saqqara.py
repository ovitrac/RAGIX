"""
Measure Saqqara on the staged tech-sheet inputs: trees, tables, sections, objects, roots and times.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    cd <bench> && python <evidence>/bench_saqqara.py --out <results.json> [--runs 2]

Run from the bench directory that stage_inputs.py filled. Every source path is relative, so the path
a tree records is the same on every host, and `merkle_root` can be compared across hosts.

Four parts, each in its own process. The pdf reader keeps an asset store once it is given one, so the
library-only object pass never shares a process with the kernel runs it is compared to.
  kernel    SaqqaraKernel on each file, per pdf text reader, `--runs` times, and once with
            promote_outline; the whole input set `--runs` times per reader; one file read again
            from a second path (what each root covers)
  objects   the library-only K6 passes: pictures (given an asset store), drawn regions, caption
            binding, headings shown by size or weight
  explorer  the Explorer's PyMuPDF digest of each pdf: native table observations, rows, cells
  cli       saqqaractl run, index and search on the whole input set
Writes one JSON with the host, the versions, a digest of the code, and per part the counts. Wall
times sit under keys named `wall_s`, so everything else compares across hosts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PARTS = ("kernel", "objects", "explorer", "cli")
QUERIES = ("provenance", "merkle_root")


# ----------------------------------------------------------------- helpers

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def manifest() -> dict:
    return json.loads(Path("inputs/MANIFEST.json").read_text())


def check_inputs(entries) -> None:
    for entry in entries:
        got = sha256_bytes(Path(entry["path"]).read_bytes())
        if got != entry["sha256"]:
            raise SystemExit(f"input changed: {entry['path']} {got} != {entry['sha256']}")


def walk(node, acc=None):
    acc = [] if acc is None else acc
    acc.append(node)
    for child in node.get("children", []):
        walk(child, acc)
    return acc


def pathfree_digest(tree) -> str:
    """The tree's canonical JSON with every recorded source path blanked."""
    def strip(value):
        if isinstance(value, dict):
            return {k: ("" if k == "source_path" else strip(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    text = json.dumps(strip(tree), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(text.encode("utf-8"))


def host_info() -> dict:
    models = []
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
            if line.startswith("Model name:"):
                models.append(line.split(":", 1)[1].strip())
    except OSError:
        pass
    model = " + ".join(models)  # a host with two core types lists both
    versions = {}
    import sqlite3
    versions["python"] = platform.python_version()
    versions["sqlite"] = sqlite3.sqlite_version
    for name in ("pypdf", "pymupdf", "docx", "openpyxl", "pptx", "numpy", "PIL"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = None
    try:
        import pypdfium2
        info = pypdfium2.version
        versions["pypdfium2"] = f"{info.PYPDFIUM_INFO}+pdfium{info.PDFIUM_INFO}"
    except ImportError:
        versions["pypdfium2"] = None
    return {"machine": platform.machine(), "platform": platform.platform(), "cpu": model,
            "logical_cpus": os.cpu_count(), "versions": versions,
            "loadavg_at_start": list(os.getloadavg()),
            "environment": os.environ.get("BENCH_ENV_LABEL", "measurement environment")}


def code_info() -> dict:
    """A digest over every Python file of ragix_kernels as imported, path and bytes."""
    import ragix_kernels
    root = Path(ragix_kernels.__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root.parent)).encode() + b"\0" + path.read_bytes() + b"\0")
    git = {}
    top = root.parent
    if (top / ".git").exists():
        # --no-optional-locks: a status never rewrites the index of a checkout this bench only reads
        run = lambda *a: subprocess.run(["git", "--no-optional-locks", "-C", str(top), *a],
                                        capture_output=True, text=True).stdout.strip()
        git = {"head": run("rev-parse", "HEAD"),
               "dirty_ragix_kernels": run("status", "--porcelain", "--", "ragix_kernels") != ""}
    elif os.environ.get("BENCH_CODE_COMMIT"):
        # a copy made by `git archive <commit>` has no .git: the caller names the commit
        git = {"archive_of": os.environ["BENCH_CODE_COMMIT"]}
    return {"python_files": len(files), "digest": h.hexdigest(), "git": git}


# ------------------------------------------------------------- tree measures

def table_measure(node) -> dict:
    facts = node.get("facts", {})
    signals = facts.get("signals", {}) or {}
    cells = [n for n in walk(node)[1:] if n["kind"] == "cell"]
    return {
        "type": facts.get("table_type") or facts.get("block_type"),
        "origin": node.get("origin"),
        "rows": facts.get("n_rows", signals.get("value_rows")),
        "cols": facts.get("n_grid_cols", signals.get("value_columns")),
        "cell_nodes": len(cells),
        "empty_cells": sum(1 for c in cells if (c.get("facts") or {}).get("empty")),
    }


def anchor_measure(pages) -> dict:
    """PDF text anchors (facts x, y) that fall outside their own page box."""
    anchors = outside = 0
    for page in pages:
        facts = page.get("facts") or {}
        width, height = facts.get("width"), facts.get("height")
        if width is None or height is None:
            continue
        for node in walk(page)[1:]:
            f = node.get("facts") or {}
            if "x" in f and "y" in f:
                anchors += 1
                if not (0 <= f["x"] <= width and 0 <= f["y"] <= height):
                    outside += 1
    return {"anchors": anchors, "outside_page": outside}


def tree_measure(document) -> dict:
    tree = document["tree"]
    nodes = walk(tree["root"])
    kinds = collections.Counter(n["kind"] for n in nodes)
    pages = [n for n in nodes if n["kind"] in ("page", "slide")]
    traces = document["traces"]
    sections = traces.get("sections", {})
    return {
        "nodes": len(nodes),
        "kinds": dict(sorted(kinds.items())),
        "pages": len(pages),
        "text_anchors": anchor_measure(pages),
        "pages_without_text": sum(1 for p in pages if (p.get("facts") or {}).get("has_text") is False),
        "image_placements": sum(int((p.get("facts") or {}).get("image_count") or 0) for p in pages),
        "tables": [table_measure(n) for n in nodes if n["kind"] == "table"],
        "headings": kinds.get("heading", 0),
        "inferred_headings": sum(1 for n in nodes if n["kind"] == "heading" and n.get("origin") == "inferred"),
        "sections_analyzer": {
            "candidates": sections.get("candidates"),
            "accepted": sections.get("accepted"),
            "rejected": sections.get("rejected"),
            "baseline": sections.get("baseline"),
            "blind": sections.get("blind"),
            "channels": {k: len(v) for k, v in (sections.get("channels") or {}).items()},
        },
        "grid_tables": {k: traces["grid_tables"].get(k) for k in ("tables", "by_type")},
        "builder": {k: traces["builder"].get(k) for k in ("reader", "reader_version", "observations",
                                                           "nodes", "attached", "dropped")},
        "reader_kernel_version": tree["root"]["provenance"]["kernel_version"],
        "pathfree_tree_sha256": pathfree_digest(tree),
    }


def run_kernel(source: str, workspace: Path, extra: dict) -> tuple[dict, float]:
    from ragix_kernels.base import KernelInput
    from ragix_kernels.saqqara.kernel import SaqqaraKernel

    workspace.mkdir(parents=True, exist_ok=True)
    config = {"source": {"path": source}, **extra}
    start = time.perf_counter()
    out = SaqqaraKernel().run(KernelInput(workspace=workspace, config=config))
    wall = time.perf_counter() - start
    if not out.success:
        raise SystemExit(f"kernel failed on {source} {extra}: {out.errors}")
    return out.data | {"_summary": out.summary}, wall


def run_summary(data: dict) -> dict:
    report = data["report"]
    return {
        "merkle_root": data["merkle_root"],
        "source_root": data["source_root"],
        "counts": report["counts"],
        "refusals": [{"path": r["path"], "reason": r["reason"]} for r in report["refusals"]],
        "abstentions": dict(sorted(collections.Counter(
            f"{a['analyzer']}:{a['reason']}" for a in report["abstentions"]).items())),
        "abstention_records": len(report["abstentions"]),
        "summary": data["_summary"],
    }


READERS = {"pypdf": {}, "pymupdf": {"pdf": {"text_reader": "pymupdf"}}}


def part_kernel(entries, work: Path, runs: int) -> dict:
    files = []
    for entry in entries:
        is_pdf = entry["path"].endswith(".pdf")
        variants = list(READERS) if is_pdf else ["default"]
        per_reader = {}
        for reader in variants:
            extra = READERS.get(reader, {})
            result = {"runs": [], "wall_s": []}
            for k in range(runs):
                data, wall = run_kernel(entry["path"], work / "kernel" / entry["id"] / reader / f"run{k}", extra)
                summary = run_summary(data)
                summary["tree"] = tree_measure(data["documents"][0])
                result["runs"].append(summary)
                result["wall_s"].append(round(wall, 4))
            first = result["runs"][0]
            result["roots_equal_across_runs"] = all(
                r["merkle_root"] == first["merkle_root"] and r["source_root"] == first["source_root"]
                for r in result["runs"])
            pages = first["tree"]["pages"]
            result["wall_s_per_page"] = [round(w / pages, 4) for w in result["wall_s"]] if pages else None
            data, wall = run_kernel(entry["path"], work / "kernel" / entry["id"] / reader / "outline",
                                    {**extra, "promote_outline": True})
            tree = tree_measure(data["documents"][0])
            result["promote_outline"] = {"merkle_root": data["merkle_root"], "headings": tree["headings"],
                                         "inferred_headings": tree["inferred_headings"],
                                         "nodes": tree["nodes"], "wall_s": round(wall, 4)}
            per_reader[reader] = result
        files.append({"id": entry["id"], "path": entry["path"], "synthetic": entry["synthetic"],
                      "readers": per_reader})

    whole = {}
    for reader, extra in READERS.items():
        rows = []
        for k in range(runs):
            data, wall = run_kernel("inputs", work / "kernel" / "_all" / reader / f"run{k}", extra)
            rows.append(run_summary(data) | {"wall_s": round(wall, 4),
                                             "nodes": sum(len(walk(d["tree"]["root"])) for d in data["documents"])})
        whole[reader] = {"runs": rows,
                         "roots_equal_across_runs": len({(r["merkle_root"], r["source_root"]) for r in rows}) == 1}

    # What each root covers: the same bytes read again from another directory, under another name.
    moved = work / "moved" / "renamed_copy.pdf"
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("inputs/public/Adservio-RAGIXTechSheet.pdf", moved)
    original, _ = run_kernel("inputs/public/Adservio-RAGIXTechSheet.pdf", work / "kernel" / "_path" / "original", {})
    elsewhere, _ = run_kernel(str(moved), work / "kernel" / "_path" / "moved", {})
    path_test = {
        "original": "inputs/public/Adservio-RAGIXTechSheet.pdf",
        "moved": str(moved),
        "merkle_root_equal": original["merkle_root"] == elsewhere["merkle_root"],
        "source_root_equal": original["source_root"] == elsewhere["source_root"],
        "pathfree_tree_equal": (pathfree_digest(original["documents"][0]["tree"])
                                == pathfree_digest(elsewhere["documents"][0]["tree"])),
    }
    return {"files": files, "whole_set": whole, "path_test": path_test}


def part_objects(entries, work: Path) -> dict:
    from ragix_kernels.saqqara.adapters import adapter_for, read_path
    from ragix_kernels.saqqara.analyzers.caption_binding import CaptionBindingAnalyzer
    from ragix_kernels.saqqara.analyzers.format_headings import FormatHeadingsAnalyzer
    from ragix_kernels.saqqara.analyzers.vector_regions import VectorRegionAnalyzer
    from ragix_kernels.saqqara.assets import AssetStore
    from ragix_kernels.saqqara.builder import build_tree

    rows = []
    for entry in entries:
        path = Path(entry["path"])
        store = AssetStore(work / "objects" / entry["id"] / "assets")
        start = time.perf_counter()
        adapter = adapter_for(path)
        records = read_path(path, store=store)
        tree = build_tree(records, str(path), adapter.format, adapter.format, adapter.version).tree
        read_wall = time.perf_counter() - start
        kinds = collections.Counter(n["kind"] for n in walk(tree.to_dict()["root"]))
        row = {"id": entry["id"], "figures": kinds.get("figure", 0), "assets": len(store.ids()),
               "read_wall_s": round(read_wall, 4)}
        if adapter.format == "pdf":
            start = time.perf_counter()
            regions = VectorRegionAnalyzer(store=store, keep_raster=False).run(tree)
            row["vector_regions"] = {k: regions.trace.get(k) for k in
                                     ("pages_examined", "marks", "clusters", "grouped", "promoted",
                                      "excluded", "refused", "renderer")}
            row["vector_region_nodes"] = sum(1 for n in walk(regions.tree.to_dict()["root"])
                                             if n["kind"] == "vector_region")
            row["regions_wall_s"] = round(time.perf_counter() - start, 4)
            captions = CaptionBindingAnalyzer().run(regions.tree)
            row["caption_binding"] = {k: v for k, v in captions.trace.items()
                                      if isinstance(v, (int, float, str)) or k in ("abstained", "refused")}
            # A tree of its own, built from the same records: an analyzer may edit the tree it is given.
            plain = build_tree(records, str(path), adapter.format, adapter.format, adapter.version).tree
            count = lambda t: sum(1 for n in walk(t.to_dict()["root"]) if n["kind"] == "heading")
            before = count(plain)
            headings = FormatHeadingsAnalyzer().run(plain)
            row["format_headings"] = {k: v for k, v in headings.trace.items()
                                      if isinstance(v, (int, float, str)) or k in ("abstained", "refused")}
            row["heading_nodes_before_after"] = [before, count(headings.tree)]
        rows.append(row)
    return {"files": rows}


def part_explorer(entries, work: Path) -> dict:
    from ragix_kernels.saqqara.explorer import digest_pdf, explore

    rows = []
    for entry in entries:
        if not entry["path"].endswith(".pdf"):
            continue
        start = time.perf_counter()
        digest = digest_pdf(Path(entry["path"]))
        digest_wall = time.perf_counter() - start
        tables = [t for p in digest.pages for t in p.tables]
        cells = [c for t in tables for r in t.cell_rows for c in r]
        start = time.perf_counter()
        result = explore(digest)
        explore_wall = time.perf_counter() - start
        report = result.report
        rows.append({
            "id": entry["id"], "extractor": f"{digest.extractor} {digest.extractor_version}"
            if hasattr(digest, "extractor") else None,
            "pages": len(digest.pages),
            "spans": sum(len(p.spans) for p in digest.pages),
            "drawings": sum(p.drawing_count for p in digest.pages) if hasattr(digest.pages[0], "drawing_count") else None,
            "tables": len(tables),
            "table_rows": [1 + len(t.rows) for t in tables],
            "table_cells": len(cells),
            "cells_missing_geometry": sum(1 for c in cells if "MISSING_CELL_GEOMETRY" in (c.flags or ())),
            "tables_per_page": dict(sorted(collections.Counter(t.page for t in tables).items())),
            "report_type": type(report).__name__,
            "digest_wall_s": round(digest_wall, 4),
            "explore_wall_s": round(explore_wall, 4),
        })
    return {"files": rows}


def part_cli(work: Path) -> dict:
    ws = work / "cli" / "ws"
    if ws.exists():
        raise SystemExit(f"refused: {ws} exists")
    base = [sys.executable, "-m", "ragix_kernels.saqqara.cli.saqqaractl"]
    out = {}
    for name, argv in (("run", ["run", "inputs", "-w", str(ws), "--json"]),
                       ("index", ["index", str(ws), "--json"])):
        start = time.perf_counter()
        done = subprocess.run(base + argv, capture_output=True, text=True)
        wall = time.perf_counter() - start
        if done.returncode != 0:
            raise SystemExit(f"saqqaractl {name} failed: {done.stderr[-2000:]}")
        data = json.loads(done.stdout)
        (work / "cli" / f"{name}.json").write_text(done.stdout)
        if name == "run":
            out[name] = {"merkle_root": data.get("merkle_root"), "source_root": data.get("source_root"),
                         "counts": (data.get("report") or {}).get("counts"), "wall_s": round(wall, 4)}
        else:
            out[name] = {"data": {k: v for k, v in data.items() if not isinstance(v, (list, dict)) or
                                  len(json.dumps(v)) < 2000}, "wall_s": round(wall, 4)}
    out["search"] = {}
    for query in QUERIES:
        start = time.perf_counter()
        done = subprocess.run(base + ["search", str(ws), query, "-k", "5", "--json"],
                              capture_output=True, text=True)
        wall = time.perf_counter() - start
        if done.returncode != 0:
            raise SystemExit(f"saqqaractl search failed: {done.stderr[-2000:]}")
        hits = json.loads(done.stdout)
        (work / "cli" / f"search_{query}.json").write_text(done.stdout)
        hit_list = hits if isinstance(hits, list) else hits.get("hits", [])
        out["search"][query] = {"hits": len(hit_list), "wall_s": round(wall, 4), "top": [
            {"source_path": (h.get("provenance") or [{}])[0].get("source_path"),
             "kind": (h.get("provenance") or [{}])[0].get("kind"),
             "node_ids": h["chunk"].get("node_ids"), "pages": h["chunk"].get("pages"),
             "level": h["chunk"].get("level"), "text": (h["chunk"].get("text") or "")[:80],
             "lexical_rank": h.get("lexical_rank"), "dense_rank": h.get("dense_rank"),
             "final_rank": h.get("final_rank")}
            for h in hit_list[:5]]}
    return out


# ------------------------------------------------------------------ driver

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--work", default="work")
    parser.add_argument("--part", choices=PARTS)
    parser.add_argument("--part-out")
    args = parser.parse_args()
    entries = manifest()["inputs"]
    work = Path(args.work)

    if args.part:
        check_inputs(entries)
        result = {"kernel": lambda: part_kernel(entries, work, args.runs),
                  "objects": lambda: part_objects(entries, work),
                  "explorer": lambda: part_explorer(entries, work),
                  "cli": lambda: part_cli(work)}[args.part]()
        Path(args.part_out).write_text(json.dumps(result, indent=1, sort_keys=True))
        return

    if work.exists():
        raise SystemExit(f"refused: {work} exists; give another --work")
    work.mkdir(parents=True)
    check_inputs(entries)
    result = {"schema": "saqqara-techsheet-bench/1", "host": host_info(), "code": code_info(),
              "inputs": entries, "runs": args.runs, "parts": {}}
    for part in PARTS:
        part_out = work / f"part_{part}.json"
        start = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--part", part, "--part-out", str(part_out),
                        "--runs", str(args.runs), "--work", str(work)], check=True)
        result["parts"][part] = json.loads(part_out.read_text())
        result["parts"][part]["part_wall_s"] = round(time.perf_counter() - start, 4)
        print(f"part {part} done in {result['parts'][part]['part_wall_s']} s", flush=True)
    result["host"]["loadavg_at_end"] = list(os.getloadavg())
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
