#!/usr/bin/env python3
"""Search, and follow every hit back to the document it came from.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Runs the lexical lane always. The dense lane needs an embedder, and if none can
be imported this script SKIPS it with a printed reason rather than failing: a
demo that cannot run without a model download is a demo nobody runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ragix_kernels.saqqara.store.config import load_config  # noqa: E402
from ragix_kernels.saqqara.store.ports import build_store  # noqa: E402
from ragix_kernels.saqqara.store.retrieve import Retriever, provenance_of  # noqa: E402

WORKSPACE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./demo-workspace")
QUERY = sys.argv[2] if len(sys.argv) > 2 else "paragraphe"
CONFIG = Path(__file__).with_name("saqqara.yaml")


def _embedder():
    """A real embedder if one is importable, otherwise None and why."""
    from ragix_kernels.saqqara.store.embed import build_embedder

    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        print(f"dense lane SKIPPED: sentence-transformers is not installed ({exc.__class__.__name__})")
        print("  install it with: pip install -e \".[retrieval]\"\n")
        return None, ""
    try:
        return build_embedder("sentence-transformers"), "sentence-transformers"
    except Exception as exc:
        print(f"dense lane SKIPPED: the backend could not start ({exc})\n")
        return None, ""


def main() -> int:
    config = load_config(CONFIG)
    path = WORKSPACE / config.get("store.path")
    if not path.is_file():
        print(f"no store at {path}; run 02_index_store.py first")
        return 2

    store = build_store({**config.section("store"), "path": str(path)})
    embedder, model = _embedder()
    vector = embedder.embed_batch([QUERY])[0] if embedder else None

    retrieval = config.section("retrieval")
    hits = Retriever(store, model=model, rrf_k=retrieval["rrf_k"]).search(
        query=QUERY, vector=vector, top_k=retrieval["top_k"],
        dense_k=retrieval["dense_k"], lexical_k=retrieval["lexical_k"])

    print(f"query {QUERY!r} — {len(hits)} hit(s)")
    if not hits:
        print("  (the demo documents are in French; try 'Titre' or 'Moyens')")
        return 0

    trees = {d.doc_id: d.tree for d in store.list_documents()}
    for hit in hits:
        print(f"\n  dense={hit.dense_rank}  lexical={hit.lexical_rank}  "
              f"final={hit.final_rank}")
        print(f"    {hit.chunk.text[:90]!r}")
        for entry in provenance_of(trees.get(hit.chunk.doc_id), hit.chunk):
            print(f"    <- {entry['kind']:10s} {Path(entry['source_path']).name} "
                  f"{entry['chain']}")

    print("\nEach lane keeps its own rank. A hit found by one lane shows None for")
    print("the other — absent and last are different answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
