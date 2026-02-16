"""
RAGIX Shared Utilities

Generic, reusable tools for text processing, file extraction, and chunking.
Used by memory, summary, reviewer, and presenter subsystems.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from ragix_core.shared.text_utils import (
    normalize_whitespace,
    estimate_tokens,
    stable_hash,
    split_paragraphs,
)
from ragix_core.shared.gpu_detect import (
    GPUInfo,
    gpu_info,
    has_gpu,
    has_faiss_gpu,
)

__all__ = [
    "normalize_whitespace",
    "estimate_tokens",
    "stable_hash",
    "split_paragraphs",
    "GPUInfo",
    "gpu_info",
    "has_gpu",
    "has_faiss_gpu",
]
