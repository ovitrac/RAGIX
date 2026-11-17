"""
Measure Docling on the staged tech-sheet inputs, beside Saqqara: elements, tables, headings, figures,
provenance, confidence grades, times, peak memory, and whether two runs export the same JSON.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    cd <bench> && <docling env>/bin/python docling_bench.py --models <dir> --out <results.json> \
        --runs 2 --threads 4,20 --ids techsheet_pdf,slides_pdf,... [--markdown-dir <dir>]

Each conversion runs in its own process (`--one`), so its peak resident memory is its own and a model
loaded once is not shared between files. The pipeline: Docling's standard PDF pipeline, OCR off, table
structure on (TableFormer, its default mode), CPU only, `num_threads` and OMP_NUM_THREADS set to the
thread count of the run, models read from `--models` with the Hugging Face hub offline, so a run can
fetch nothing. Markdown inputs go through Docling's Markdown backend.

The comparison object is Docling's lossless JSON export (`DoclingDocument.save_as_json`). For each
input the JSON of every run is hashed as written, and again with every key whose name contains
"time", "date" or "stamp" removed. The Markdown export is written only as an illustration.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

STAMP = re.compile(r"time|date|stamp", re.IGNORECASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_stamps(value):
    if isinstance(value, dict):
        return {k: without_stamps(v) for k, v in value.items() if not STAMP.search(k)}
    if isinstance(value, list):
        return [without_stamps(v) for v in value]
    return value


def stamp_keys(value, path="", out=None):
    out = set() if out is None else out
    if isinstance(value, dict):
        for k, v in value.items():
            if STAMP.search(k):
                out.add(f"{path}/{k}")
            stamp_keys(v, f"{path}/{k}", out)
    elif isinstance(value, list):
        for v in value:
            stamp_keys(v, f"{path}[]", out)
    return out


def one(args) -> None:
    """Convert one file in this process and print one JSON line of measures."""
    threads = int(args.threads_one)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    started = time.perf_counter()
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    import docling, torch
    torch.set_num_threads(threads)
    imported = time.perf_counter()

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True, artifacts_path=args.models,
                                 accelerator_options=AcceleratorOptions(num_threads=threads, device="cpu"))
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.MD],
                                  format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    path = Path(args.one)
    is_pdf = path.suffix.lower() == ".pdf"
    if is_pdf:
        converter.initialize_pipeline(InputFormat.PDF)
    initialized = time.perf_counter()
    result = converter.convert(path)
    converted = time.perf_counter()

    doc = result.document
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(out_json)
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(doc.export_to_markdown())

    labels = collections.Counter()
    with_prov = with_bbox = with_charspan = items = 0
    origins = collections.Counter()
    tables, pictures, headings = [], 0, collections.Counter()
    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        label = getattr(label, "value", str(label))
        labels[label] += 1
        items += 1
        prov = getattr(item, "prov", None) or []
        if prov:
            with_prov += 1
            if all(getattr(p, "bbox", None) is not None for p in prov):
                with_bbox += 1
                origins[str(getattr(prov[0].bbox, "coord_origin", "?"))] += 1
            if all(getattr(p, "charspan", None) is not None for p in prov):
                with_charspan += 1
        if label in ("section_header", "title"):
            headings[label] += 1
        if label == "picture":
            pictures += 1
        if label == "table":
            data = item.data
            tables.append({"page": prov[0].page_no if prov else None, "rows": data.num_rows,
                           "cols": data.num_cols, "cells": len(data.table_cells)})
    confidence = None
    if getattr(result, "confidence", None) is not None:
        confidence = result.confidence.model_dump(mode="json")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record = {
        "status": str(getattr(result.status, "value", result.status)),
        "errors": [str(e) for e in (result.errors or [])],
        "backend": type(result.input._backend).__name__ if getattr(result.input, "_backend", None) else None,
        "docling": getattr(docling, "__version__", None), "torch": torch.__version__,
        "pages": len(doc.pages) if is_pdf else 0,
        "items": items, "labels": dict(sorted(labels.items())),
        "provenance": {"items_with_prov": with_prov, "with_bbox": with_bbox, "with_charspan": with_charspan,
                       "bbox_origin": dict(origins)},
        "tables": tables, "pictures": pictures, "headings": dict(headings),
        "confidence": confidence,
        "json_sha256": sha256(out_json),
        "json_sha256_without_stamps": hashlib.sha256(json.dumps(
            without_stamps(json.loads(out_json.read_text())), sort_keys=True).encode()).hexdigest(),
        "json_stamp_keys": sorted(stamp_keys(json.loads(out_json.read_text())))[:20],
        "json_bytes": out_json.stat().st_size,
        "import_s": round(imported - started, 4), "init_s": round(initialized - imported, 4),
        "convert_s": round(converted - initialized, 4),
        "peak_rss_kib": usage.ru_maxrss,
    }
    print("RESULT " + json.dumps(record, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--out")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--threads", default="4")
    parser.add_argument("--ids", default="")
    parser.add_argument("--work", default="m4_work")
    parser.add_argument("--markdown-dir")
    parser.add_argument("--one")
    parser.add_argument("--threads-one")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()
    if args.one:
        one(args)
        return

    manifest = json.loads(Path("inputs/MANIFEST.json").read_text())
    wanted = [i for i in args.ids.split(",") if i]
    entries = [e for e in manifest["inputs"] if not wanted or e["id"] in wanted]
    for e in entries:
        if sha256(Path(e["path"])) != e["sha256"]:
            raise SystemExit(f"input changed: {e['path']}")
    work = Path(args.work)
    if work.exists():
        raise SystemExit(f"refused: {work} exists")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               DO_NOT_TRACK="1")
    runs = []
    for threads in [int(t) for t in args.threads.split(",")]:
        for k in range(1, args.runs + 1):
            for e in entries:
                tag = f"T{threads}_run{k}"
                md = (str(Path(args.markdown_dir) / f"{e['id']}.md")
                      if args.markdown_dir and k == 1 and threads == int(args.threads.split(",")[0]) else None)
                cmd = [sys.executable, __file__, "--models", args.models, "--one", e["path"],
                       "--threads-one", str(threads), "--json-out", str(work / tag / f"{e['id']}.json")]
                if md:
                    cmd += ["--md-out", md]
                load = os.getloadavg()[0]
                start = time.perf_counter()
                done = subprocess.run(cmd, env=dict(env, OMP_NUM_THREADS=str(threads)), capture_output=True, text=True)
                wall = time.perf_counter() - start
                line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
                if done.returncode != 0 or not line:
                    raise SystemExit(f"conversion failed: {e['id']} {tag}\n{done.stderr[-3000:]}")
                record = json.loads(line[-1][len("RESULT "):])
                record.update({"id": e["id"], "threads": threads, "run": k, "process_wall_s": round(wall, 4),
                               "loadavg_1min_at_start": round(load, 2),
                               "warnings": len([l for l in done.stderr.splitlines() if "warn" in l.lower()])})
                runs.append(record)
                print(f"{tag} {e['id']}: {record['pages']} p, convert {record['convert_s']} s, "
                      f"rss {record['peak_rss_kib']} KiB, json {record['json_sha256'][:12]}", flush=True)

    summary = {}
    for e in entries:
        mine = [r for r in runs if r["id"] == e["id"]]
        summary[e["id"]] = {
            "json_identical_all_runs": len({r["json_sha256"] for r in mine}) == 1,
            "json_identical_without_stamps": len({r["json_sha256_without_stamps"] for r in mine}) == 1,
            "json_sha256_by_run": {f"T{r['threads']}_run{r['run']}": r["json_sha256"] for r in mine},
        }
    import platform
    out = {"schema": "docling-techsheet-bench/1", "machine": platform.machine(),
           "inputs": entries, "runs": runs, "determinism": summary,
           "loadavg_at_end": list(os.getloadavg())}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
