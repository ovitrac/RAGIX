#!/usr/bin/env python3
"""Chunk what was read into a store, with no embedder at all.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

`embedder.provider: none` is a choice, not an omission: the store works, it has
one lane, and its status says so rather than leaving you to infer it. No zero
vectors are written — an absent embedding is not a zero one.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ragix_kernels.base import KernelInput  # noqa: E402
from ragix_kernels.saqqara.kernels.saqqara_index import SaqqaraIndexKernel  # noqa: E402

WORKSPACE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./demo-workspace")
CONFIG = Path(__file__).with_name("saqqara.yaml")


def main() -> int:
    read = WORKSPACE / "stage1" / "saqqara.json"
    if not read.is_file():
        print(f"nothing read into {WORKSPACE}; run 01_read_tree.py first")
        return 2

    output = SaqqaraIndexKernel().run(KernelInput(
        workspace=WORKSPACE, config={"config": str(CONFIG)},
        dependencies={"document_tree": read}))
    if not output.success:
        for error in output.errors or []:
            print(f"error: {error}")
        return 1

    print(f"{output.summary}\n")
    status = output.data["status"]
    for key in ("documents", "chunks", "objects", "edges", "embeddings"):
        print(f"  {key:12s} {status[key]}")
    print(f"  {'dense':12s} {status['dense']}")

    print("\nper document:")
    for entry in output.data["documents"]:
        print(f"  {entry['format']:5s} {Path(entry['path']).name:16s} "
              f"chunks={entry['chunks']} refused={entry.get('refused', 0)}")
    print("\n(refused counts containers and empty nodes — every node is accounted for)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
