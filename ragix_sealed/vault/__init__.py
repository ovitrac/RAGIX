"""
RAGIX-Sealed vault — RAGIX-owned reversible-redaction interface and native backend.

The interface (``SealedVaultBackend``) is deliberately RAGIX-owned and backend-agnostic
(WP §6, decision D1): RAGIX-Sealed depends only on the Protocol, never on a concrete
backend. The native backend is the chosen default; a CloakMCP adapter is intentionally
NOT built (see base/RAGIX-sealed/SPIKE_NOTES.md for the rationale).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

from .backend import (
    AuthorizationToken,
    EntityMapping,
    PlaceholderPolicy,
    PlaceholderRef,
    ReidentificationPurpose,
    SealedMappingRef,
    SealedVaultBackend,
    VaultAuthorizationError,
    VaultError,
)
from .native import RAGIXSealedVaultBackend

__all__ = [
    "SealedVaultBackend",
    "PlaceholderPolicy",
    "PlaceholderRef",
    "EntityMapping",
    "SealedMappingRef",
    "ReidentificationPurpose",
    "AuthorizationToken",
    "VaultError",
    "VaultAuthorizationError",
    "RAGIXSealedVaultBackend",
]
