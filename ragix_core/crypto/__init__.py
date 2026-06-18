"""
RAGIX cryptographic primitives.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

from .sealed_aead import (
    KeyRef,
    SealedBlob,
    SealingAAD,
    SealedAEADError,
    open_bytes,
    seal_bytes,
)

__all__ = [
    "KeyRef",
    "SealedBlob",
    "SealingAAD",
    "SealedAEADError",
    "seal_bytes",
    "open_bytes",
]
