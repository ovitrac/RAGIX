"""
Count the pdf text anchors that fall outside their page box, per text reader, for the PDFs given.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python anchor_check.py --work <dir> --out <json> <pdf> [<pdf> ...]

The same measure as `text_anchors` in bench_saqqara.py (an anchor is the facts x, y of a node; it is
outside when it is not within 0..width, 0..height of its page), on PDFs outside the bench inputs. It
reads; it diagnoses nothing and changes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_saqqara import anchor_measure, run_kernel, walk  # noqa: E402

READERS = {"pypdf": {}, "pymupdf": {"pdf": {"text_reader": "pymupdf"}}}


def pdfinfo(path: Path) -> dict:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    keep = ("Producer", "Creator", "Pages", "Page size")
    return {k: v.strip() for k, _, v in (line.partition(":") for line in out.splitlines()) if k in keep}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("pdfs", nargs="+")
    args = parser.parse_args()
    rows = []
    for name in args.pdfs:
        path = Path(name)
        row = {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "pdfinfo": pdfinfo(path), "readers": {}}
        for reader, extra in READERS.items():
            data, wall = run_kernel(name, Path(args.work) / path.stem / reader, extra)
            pages = [n for n in walk(data["documents"][0]["tree"]["root"]) if n["kind"] == "page"]
            row["readers"][reader] = anchor_measure(pages) | {"pages": len(pages), "wall_s": round(wall, 4)}
        rows.append(row)
        print(name, {r: (v["outside_page"], v["anchors"]) for r, v in row["readers"].items()})
    Path(args.out).write_text(json.dumps({"files": rows}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
