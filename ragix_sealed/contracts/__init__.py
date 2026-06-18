"""
RAGIX-Sealed contracts (WP §13, Sprint 1).

Declarative YAML contracts ("the contract everything else obeys") plus a loader/validator
that guarantees their internal consistency. Canonical defaults live here; per-case copies
are instantiated into /var/lib/ragix-sealed/cases/{case_id}/ at ingestion time.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-17
"""

from .loader import (
    CONTRACTS_DIR,
    FILES,
    ContractError,
    SealedContracts,
    load_contracts,
    validate,
)

__all__ = [
    "CONTRACTS_DIR",
    "FILES",
    "ContractError",
    "SealedContracts",
    "load_contracts",
    "validate",
]
