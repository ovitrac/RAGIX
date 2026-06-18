"""
RAGIX-owned sealed vault interface (WP_RAGIX_Sealed_v1.md §6, decision D1).

This module defines the *interface* that RAGIX-Sealed depends on — never a concrete
backend. A native AES-256-GCM+AAD backend implements it (``native.py``); a CloakMCP
adapter could implement it later, but is intentionally not built (independence, WP §3).

Key doctrinal property (WP §1.1): ``resolve_placeholder`` is structurally a
human-authorized, audited operation. There is no API path that returns a raw value into
an LLM prompt — re-identification always serves the authorized human.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ragix_core.crypto.sealed_aead import SealingAAD


class VaultError(Exception):
    """Base error for sealed-vault operations."""


class VaultAuthorizationError(VaultError):
    """Raised when a re-identification request lacks valid human authorization."""


class ReidentificationPurpose(Enum):
    """Purposes for which a placeholder may be resolved back to its raw value.

    Every purpose is a HUMAN purpose. None of these may be used to feed an LLM context —
    that is the whole point of the human <-> LLM boundary (WP §1).
    """

    AUTHORIZED_REVIEW = "authorized_review"
    REPORT_EXPORT_HUMAN = "report_export_human"
    FORMAL_DISCLOSURE = "formal_disclosure"
    AUDIT_INSPECTION = "audit_inspection"


@dataclass(frozen=True)
class PlaceholderPolicy:
    """How a placeholder string is rendered for an entity type (WP §7).

    ``fmt`` is a Python format string taking ``{n}`` (the per-type counter). If
    ``role_visible`` is true, an inferred role label may be exposed *alongside* the
    placeholder to the reasoning layer (e.g. ``[PERSON_001 | primary_party]``), but never the
    raw identity.
    """

    entity_type: str
    fmt: str = "[{etype}_{n:03d}]"
    role_visible: bool = True
    schema_version: str = "placeholder-schema-0.1"

    def render(self, n: int, role: Optional[str]) -> str:
        base = self.fmt.format(etype=self.entity_type, n=n)
        if self.role_visible and role:
            # Insert the role inside the brackets: "[PERSON_001]" -> "[PERSON_001 | role]"
            if base.endswith("]"):
                return f"{base[:-1]} | {role}]"
            return f"{base} | {role}"
        return base


@dataclass(frozen=True)
class PlaceholderRef:
    """Public-facing result of creating a placeholder.

    PUBLIC-FACING CONTRACT (WP §11, K2): this object MUST NOT carry the raw value. It
    carries only the placeholder string, the entity type, an optional role label, and the
    case id. It is safe to hand to an LLM context or the orchestrator.
    """

    placeholder: str
    entity_type: str
    case_id: str
    role: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "placeholder": self.placeholder,
            "entity_type": self.entity_type,
            "case_id": self.case_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class EntityMapping:
    """Internal mapping between a placeholder and its raw value.

    INTERNAL ONLY. This object carries the raw value and must never cross the LLM boundary.
    It exists to be sealed via ``seal_mapping``; it is never returned by a public-facing
    method.
    """

    placeholder: str
    entity_type: str
    raw_value: str = field(repr=False)  # confidential; never shown in repr/logs
    role: Optional[str] = None


@dataclass(frozen=True)
class AuthorizationToken:
    """Represents explicit human authorization for re-identification.

    For the Sprint 0 spike this is a simple in-process token. In production it becomes a
    signed, expiring, purpose-scoped grant tied to an authenticated human identity.
    """

    subject: str  # the authorized human (opaque id), not a raw name
    purpose: ReidentificationPurpose
    token: str = field(repr=False)

    def authorizes(self, purpose: ReidentificationPurpose) -> bool:
        return bool(self.token) and self.purpose == purpose


@dataclass(frozen=True)
class SealedMappingRef:
    """Opaque reference to a sealed entity mapping.

    PUBLIC-FACING CONTRACT: carries no raw value — only the placeholder, a key handle, and
    an AAD digest for audit correlation.
    """

    placeholder: str
    key_id: str
    aad_sha256: str

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "placeholder": self.placeholder,
            "key_id": self.key_id,
            "aad_sha256": self.aad_sha256,
        }


@runtime_checkable
class SealedVaultBackend(Protocol):
    """RAGIX-owned reversible-redaction vault interface (WP §6)."""

    def create_placeholder(
        self,
        case_id: str,
        entity_type: str,
        raw_value: str,
        role: Optional[str],
        policy: PlaceholderPolicy,
    ) -> PlaceholderRef:
        """Assign (or reuse) a stable placeholder for ``raw_value`` within a case.

        Must return a public-facing ``PlaceholderRef`` with no raw value. Repeated calls
        with the same case/entity_type/raw_value must return the same placeholder
        (stability, WP acceptance A).
        """

    def resolve_placeholder(
        self,
        case_id: str,
        placeholder: str,
        purpose: ReidentificationPurpose,
        authorization: AuthorizationToken,
    ) -> str:
        """Resolve a placeholder back to its raw value FOR AN AUTHORIZED HUMAN.

        Must raise ``VaultAuthorizationError`` if ``authorization`` does not authorize
        ``purpose``. Never returns into an LLM context.
        """

    def seal_mapping(
        self,
        case_id: str,
        mapping: EntityMapping,
        aad: SealingAAD,
    ) -> SealedMappingRef:
        """Seal an entity mapping at rest under the given AAD, returning an opaque ref."""
