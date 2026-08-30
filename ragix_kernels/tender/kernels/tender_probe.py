"""
tender_probe — stage 3: what does the store actually hold?

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate T0.2, T0.3, T0.4.

It opens a document store, asks it what it holds, and reports that. It computes
nothing of its own — deliberately, because what is being proved by its existence
is the chain around it: `requires=["document_store"]` enforced by the envelope,
discovery by the registry, one shape shared by the CLI and the MCP tool, and a
configuration that refuses what it does not declare.

**It does not write.** Not a journal, not a vacuum, not a timestamp. The gate
asserts that on the database file's sha256 rather than on intent, because "read
only" is a claim about behaviour and a hash is the only thing that checks it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput

from ..config import load_config
from ..models import ProbeResult

__all__ = ["TenderProbeKernel"]


class TenderProbeKernel(Kernel):
    """Report what a document store holds, without altering it."""

    name = "tender_probe"
    version = "0.1.0"
    category = "tender"
    stage = 3
    description = "Report the contents of a document store without modifying it"
    requires: List[str] = ["document_store"]
    provides = ["tender_probe"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        config = load_config(input.config.get("config") or None,
                             **(input.config.get("overrides") or {}))

        declared = Path(config.get("store.path"))
        if not declared.is_absolute():
            declared = Path(input.workspace) / declared

        # Refused rather than created. A probe that builds the store it was asked
        # to inspect reports on itself and calls the corpus empty.
        if not declared.is_file():
            raise FileNotFoundError(
                f"no document store at {declared}: this family reads a store, it "
                "does not build one"
            )

        from ragix_kernels.saqqara.store.ports import build_store

        store = build_store({"provider": "sqlite", "path": str(declared)})
        status = store.status()

        # The counts are the store's own. Counting again here would be a second
        # opinion about one file, and the two would eventually disagree.
        dense = status.get("dense") or (
            f"{len(status.get('models') or [])} model(s), "
            f"{status.get('embeddings', 0)} vector(s)"
            if status.get("embeddings") else "disabled (no embedder)"
        )
        result = ProbeResult(
            store_path=str(declared),
            documents=status.get("documents", 0),
            chunks=status.get("chunks", 0),
            dense_enabled=bool(status.get("embeddings", 0)),
            dense=dense,
        )
        return {"probe": result.to_dict(), "status": status}

    def summarize(self, data: Dict[str, Any]) -> str:
        probe = data.get("probe", {})
        return (
            f"{probe.get('documents', 0)} document(s), {probe.get('chunks', 0)} "
            f"chunk(s) in the store. Dense: {probe.get('dense', 'unknown')}."
        )
