"""
RAGIX-Sealed Level-1 inventory kernels (WP §15-§16, Sprint 4).

Self-contained metrics kernels over the cooled corpus — no interpretation, no content,
opaque ids only. Kept inside the sealed subsystem rather than coupled to the KOAS kernel
registry, consistent with the sealed-isolation doctrine.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .base import CorpusView, InventoryKernel, KernelResult
from .inventory import (
    ALL_INVENTORY_KERNELS,
    CorpusMetricsKernel,
    EntityInventoryKernel,
    QualityRiskKernel,
    ReviewQueueKernel,
    TypologyKernel,
)
from .runner import (
    UnknownKernelError,
    kernel_list,
    run_inventory,
    run_inventory_kernel,
)
from .analysis import (
    ALL_ANALYSIS_KERNELS,
    DEFERRED_KERNELS,
    AnalysisError,
    AnalysisKernel,
    AnalysisResult,
    CommitmentKernelV0,
    EntityRoleGraphKernel,
    GapDetectionKernelV0,
    TimelineKernel,
    UnknownAnalysisKernelError,
    analysis_kernel_list,
    run_analysis,
    run_analysis_kernel,
)

__all__ = [
    "CorpusView",
    "InventoryKernel",
    "KernelResult",
    "CorpusMetricsKernel",
    "TypologyKernel",
    "EntityInventoryKernel",
    "QualityRiskKernel",
    "ReviewQueueKernel",
    "ALL_INVENTORY_KERNELS",
    "kernel_list",
    "run_inventory",
    "run_inventory_kernel",
    "UnknownKernelError",
    "AnalysisKernel",
    "AnalysisResult",
    "AnalysisError",
    "TimelineKernel",
    "EntityRoleGraphKernel",
    "CommitmentKernelV0",
    "GapDetectionKernelV0",
    "ALL_ANALYSIS_KERNELS",
    "DEFERRED_KERNELS",
    "analysis_kernel_list",
    "run_analysis",
    "run_analysis_kernel",
    "UnknownAnalysisKernelError",
]
