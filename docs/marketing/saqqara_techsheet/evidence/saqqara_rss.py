"""
Peak resident memory of one Saqqara run per input, each in its own process, beside Docling's.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    cd <bench> && python saqqara_rss.py --out <json> --ids a,b,... [--reader pypdf|pymupdf] [--work <dir>]

One `SaqqaraKernel().run(...)` per process, on one file, then `ru_maxrss` of that process: what reading
one document into its tree costs in memory, imports included. The library passes and the Explorer are
not part of it.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def one(path: str, reader: str, workspace: str) -> None:
    from ragix_kernels.base import KernelInput
    from ragix_kernels.saqqara.kernel import SaqqaraKernel

    config = {"source": {"path": path}}
    if reader == "pymupdf" and path.endswith(".pdf"):
        config["pdf"] = {"text_reader": "pymupdf"}
    Path(workspace).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    out = SaqqaraKernel().run(KernelInput(workspace=Path(workspace), config=config))
    wall = time.perf_counter() - start
    if not out.success:
        raise SystemExit(f"failed: {out.errors}")
    print("RESULT " + json.dumps({"wall_s": round(wall, 4),
                                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                  "merkle_root": out.data["merkle_root"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--ids", default="")
    parser.add_argument("--reader", default="pypdf")
    parser.add_argument("--work", default="rss_work")
    parser.add_argument("--one")
    parser.add_argument("--workspace")
    args = parser.parse_args()
    if args.one:
        one(args.one, args.reader, args.workspace)
        return
    manifest = json.loads(Path("inputs/MANIFEST.json").read_text())
    wanted = [i for i in args.ids.split(",") if i]
    rows = []
    for e in manifest["inputs"]:
        if wanted and e["id"] not in wanted:
            continue
        done = subprocess.run([sys.executable, __file__, "--one", e["path"], "--reader", args.reader,
                               "--workspace", str(Path(args.work) / e["id"])], capture_output=True, text=True)
        line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
        if done.returncode != 0 or not line:
            raise SystemExit(f"{e['id']} failed: {done.stderr[-2000:]}")
        row = json.loads(line[-1][7:]) | {"id": e["id"], "reader": args.reader,
                                          "loadavg_1min_at_end": round(os.getloadavg()[0], 2)}
        rows.append(row)
        print(row, flush=True)
    Path(args.out).write_text(json.dumps({"files": rows}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
