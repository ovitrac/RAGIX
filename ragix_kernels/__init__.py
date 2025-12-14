"""
RAGIX Kernels — Kernel-Orchestrated Audit System (KOAS)

This module provides the kernel infrastructure for KOAS:
- Kernels do computation (no LLM inside)
- Produce structured output + summary for LLM consumption
- Fully traceable and reproducible

Following the VirtualHybridLab pattern:
> "Each interaction costs a few seconds; most time is spent inside
>  the scientific kernels, not the LLMs."

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from ragix_core.version import __version__

__all__ = [
    # Base classes
    "Kernel",
    "KernelInput",
    "KernelOutput",
    # Registry
    "KernelRegistry",
    "discover_kernels",
    # Utilities
    "get_kernel",
    "list_kernels",
]


def __getattr__(name):
    """Lazy imports to avoid circular dependencies."""
    if name in ("Kernel", "KernelInput", "KernelOutput"):
        from ragix_kernels.base import Kernel, KernelInput, KernelOutput
        return locals()[name]
    elif name in ("KernelRegistry", "discover_kernels", "get_kernel", "list_kernels"):
        from ragix_kernels.registry import KernelRegistry, discover_kernels, get_kernel, list_kernels
        return locals()[name]
    raise AttributeError(f"module 'ragix_kernels' has no attribute '{name}'")
