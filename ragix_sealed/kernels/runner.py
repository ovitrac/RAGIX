"""
RAGIX-Sealed — inventory kernel runner/registry (WP §22, Sprint 4).

Backs the safe `sealed.run_inventory_kernel` / `sealed.kernel_list` tool surface. Returns
metrics-only results; no kernel here interprets or emits content.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import CorpusView, InventoryKernel, KernelResult
from .inventory import ALL_INVENTORY_KERNELS

_REGISTRY: Dict[str, InventoryKernel] = {k.name: k for k in ALL_INVENTORY_KERNELS}


class UnknownKernelError(Exception):
    """Raised when an inventory kernel name is not registered."""


def kernel_list() -> List[str]:
    """Names of registered inventory kernels (safe for the orchestrator)."""
    return list(_REGISTRY.keys())


def run_inventory_kernel(name: str, view: CorpusView) -> KernelResult:
    """Run a single named inventory kernel."""
    kernel = _REGISTRY.get(name)
    if kernel is None:
        raise UnknownKernelError(f"unknown inventory kernel: {name!r}")
    return kernel.run(view)


def run_inventory(view: CorpusView, names: Optional[List[str]] = None) -> Dict[str, dict]:
    """Run all (or the named) inventory kernels; return {kernel_name: metrics}."""
    selected = names if names is not None else list(_REGISTRY.keys())
    out: Dict[str, dict] = {}
    for name in selected:
        out[name] = run_inventory_kernel(name, view).metrics
    return out
