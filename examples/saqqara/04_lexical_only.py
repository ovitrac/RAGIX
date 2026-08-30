#!/usr/bin/env python3
"""Two refusals you want to see happen: no zero vectors, and no silent typo.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ragix_kernels.saqqara.store.config import load_config  # noqa: E402
from ragix_kernels.saqqara.store.ports import build_store  # noqa: E402

WORKSPACE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./demo-workspace")
CONFIG = Path(__file__).with_name("saqqara.yaml")


def main() -> int:
    config = load_config(CONFIG)
    path = WORKSPACE / config.get("store.path")
    if not path.is_file():
        print(f"no store at {path}; run 02_index_store.py first")
        return 2

    store = build_store({**config.section("store"), "path": str(path)})
    status = store.status()

    print("1. provider: none writes NOTHING, not a zero vector\n")
    print(f"   chunks     {status['chunks']}")
    print(f"   embeddings {status['embeddings']}")
    print(f"   dense      disabled (no embedder)")
    assert status["embeddings"] == 0, "a vector was written with no embedder"
    print("\n   A zero vector would have a cosine with every query and would make")
    print("   an unembedded chunk look embedded. So none is written at all.\n")

    hits = store.lexical_search("Titre", top_k=3)
    print(f"2. the store still answers: {len(hits)} lexical hit(s)")
    for hit in hits:
        print(f"   lexical={hit.lexical_rank} dense={hit.dense_rank} "
              f"{hit.chunk.text[:50]!r}")

    print("\n3. a typo in the configuration is refused, with its path\n")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "typo.yaml"
        bad.write_text("embedder:\n  privider: ollama\n", encoding="utf-8")
        try:
            load_config(bad)
        except ValueError as exc:
            print(f"   {exc}")
        else:
            print("   NOT REFUSED — this is a defect")
            return 1

    print("\n   The alternative is a run that quietly uses the default embedder")
    print("   and reports success, with the instruction discarded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
