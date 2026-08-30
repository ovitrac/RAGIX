#!/usr/bin/env python3
"""Read documents into typed trees, and look at what a node actually carries.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The documents are BUILT HERE, by the same generators the gates use. Nothing is
committed to this repository, so everything you see below came from code you can
read — which is also why the demo needs no corpus and no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.base import KernelInput  # noqa: E402
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel  # noqa: E402
from ragix_kernels.saqqara.model import Tree  # noqa: E402

WORKSPACE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./demo-workspace")
CORPUS = {"report.docx": "docx_two_tier", "figures.pdf": "pdf_caption_below",
          "workbook.xlsx": "mixed_workbook", "notes.md": "markdown_document",
          "deck.pptx": "slide_deck"}


def main() -> int:
    source = WORKSPACE / "corpus"
    source.mkdir(parents=True, exist_ok=True)
    for name, fixture in CORPUS.items():
        G.FIXTURES[fixture](source / name)
    print(f"built {len(CORPUS)} documents in {source}")

    output = SaqqaraKernel().run(
        KernelInput(workspace=WORKSPACE, config={"source": {"path": str(source)}}))
    print(f"\n{output.summary}\n")

    for document in output.data["documents"]:
        tree = Tree.from_dict(document["tree"])
        kinds: dict[str, int] = {}
        for node in tree.walk():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        print(f"{document['format']:5s} {Path(document['path']).name:16s} {kinds}")

    # One node, and everything it can tell you about where it came from.
    tree = Tree.from_dict(output.data["documents"][0]["tree"])
    node = next((n for n in tree.walk() if n.text), tree.root)
    print(f"\na node, and its citation:\n  kind       {node.kind}"
          f"\n  origin     {node.origin} (confidence {node.confidence})"
          f"\n  text       {(node.text or '')[:60]!r}"
          f"\n  source     {node.provenance.source_path}"
          f"\n  chain      {[l.to_dict() for l in node.provenance.chain]}")

    print(f"\nmerkle_root {output.data['merkle_root'][:32]}…  (stable across runs)")
    print(f"source_root {output.data['source_root'][:32]}…  (follows the bytes)")
    refusals = (output.data.get("report") or {}).get("refusals") or []
    print(f"refused     {len(refusals)}"
          + (f" — {refusals[0].get('reason')}" if refusals else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
