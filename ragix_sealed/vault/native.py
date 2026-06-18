"""
Native RAGIX-Sealed vault backend (WP_RAGIX_Sealed_v1.md §6, decision D1).

Implements ``SealedVaultBackend`` over the native AES-256-GCM+AAD primitive
(``ragix_core.crypto.sealed_aead``). This is the chosen default; RAGIX-Sealed is
independent of CloakMCP (WP §3).

Storage discipline (Sprint 0 spike — in-memory, no real persistence yet):
- Raw values are NEVER stored in clear. Each raw value is sealed into a ``SealedBlob``
  keyed by its placeholder; resolution opens that blob.
- Placeholder stability / dedup uses an HMAC of the raw value (WP §5.2 opaque-id style),
  so the dedup index itself stores no raw value.
- ``resolve_placeholder`` requires a valid human ``AuthorizationToken`` and serves the
  authorized human only — never an LLM context (WP §1.1).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Optional, Tuple

from ragix_core.crypto.sealed_aead import (
    KeyRef,
    SealedBlob,
    SealingAAD,
    open_bytes,
    seal_bytes,
)

from .backend import (
    AuthorizationToken,
    EntityMapping,
    PlaceholderPolicy,
    PlaceholderRef,
    ReidentificationPurpose,
    SealedMappingRef,
    VaultAuthorizationError,
    VaultError,
)

# Default AAD components for the spike. In production these come from the case manifest
# and policy (WP §5.1); here they make the AAD binding exercised end-to-end.
_SPIKE_POLICY_VERSION = "seal-policy-0.1"
_SPIKE_SCHEMA_VERSION = "placeholder-schema-0.1"
_SPIKE_INGESTION_STATE = "PLACEHOLDERIZED"


class RAGIXSealedVaultBackend:
    """In-memory native vault backend for the Sprint 0 spike.

    One instance manages one or more cases. A per-case secret drives both the opaque
    dedup index and a per-case key. Nothing here is intended to persist yet; the goal is
    to validate the interface, the AAD binding, and the no-raw boundary.
    """

    def __init__(self) -> None:
        # case_id -> case secret (HMAC key for dedup index + doc-id derivation)
        self._case_secrets: Dict[str, bytes] = {}
        # case_id -> encryption key
        self._case_keys: Dict[str, KeyRef] = {}
        # (case_id, entity_type) -> next counter
        self._counters: Dict[Tuple[str, str], int] = {}
        # (case_id, dedup_key_hex) -> placeholder   (dedup_key is an HMAC, never raw)
        self._dedup: Dict[Tuple[str, str], str] = {}
        # (case_id, placeholder) -> (SealedBlob, SealingAAD, role, entity_type)
        self._sealed: Dict[Tuple[str, str], Tuple[SealedBlob, SealingAAD, Optional[str], str]] = {}

    # -- case bootstrap ----------------------------------------------------------------

    def _ensure_case(self, case_id: str) -> None:
        if case_id not in self._case_secrets:
            self._case_secrets[case_id] = os.urandom(32)
            self._case_keys[case_id] = KeyRef.generate(key_id=f"case-key::{case_id}")

    def _dedup_key(self, case_id: str, entity_type: str, raw_value: str) -> str:
        """Opaque, raw-free dedup key: HMAC(case_secret, entity_type | raw_value)."""
        msg = f"{entity_type}\x1f{raw_value}".encode("utf-8")
        return hmac.new(self._case_secrets[case_id], msg, hashlib.sha256).hexdigest()

    def _doc_id_placeholder_anchor(self, case_id: str, placeholder: str) -> str:
        """A stable opaque anchor used as the AAD ``doc_id`` for a per-placeholder seal."""
        msg = f"placeholder\x1f{placeholder}".encode("utf-8")
        return hmac.new(self._case_secrets[case_id], msg, hashlib.sha256).hexdigest()[:24]

    def _aad_for(self, case_id: str, placeholder: str, raw_value: str) -> SealingAAD:
        return SealingAAD(
            case_id=case_id,
            doc_id=self._doc_id_placeholder_anchor(case_id, placeholder),
            raw_sha256=hashlib.sha256(raw_value.encode("utf-8")).hexdigest(),
            policy_version=_SPIKE_POLICY_VERSION,
            placeholder_schema_version=_SPIKE_SCHEMA_VERSION,
            ingestion_state=_SPIKE_INGESTION_STATE,
        )

    # -- SealedVaultBackend interface --------------------------------------------------

    def create_placeholder(
        self,
        case_id: str,
        entity_type: str,
        raw_value: str,
        role: Optional[str],
        policy: PlaceholderPolicy,
    ) -> PlaceholderRef:
        if not raw_value:
            raise VaultError("raw_value must be non-empty")
        self._ensure_case(case_id)

        # Stability / dedup: same raw value in the same case+type reuses its placeholder.
        dkey = self._dedup_key(case_id, entity_type, raw_value)
        existing = self._dedup.get((case_id, dkey))
        if existing is not None:
            _, _, stored_role, _ = self._sealed[(case_id, existing)]
            return PlaceholderRef(
                placeholder=existing,
                entity_type=entity_type,
                case_id=case_id,
                role=stored_role,
            )

        # Allocate a fresh per-type counter and render the placeholder.
        n = self._counters.get((case_id, entity_type), 0) + 1
        self._counters[(case_id, entity_type)] = n
        placeholder = policy.render(n, role)

        # Seal the raw value at rest, bound by AAD; store no clear raw value anywhere.
        aad = self._aad_for(case_id, placeholder, raw_value)
        blob = seal_bytes(raw_value.encode("utf-8"), aad, self._case_keys[case_id])
        self._sealed[(case_id, placeholder)] = (blob, aad, role, entity_type)
        self._dedup[(case_id, dkey)] = placeholder

        return PlaceholderRef(
            placeholder=placeholder,
            entity_type=entity_type,
            case_id=case_id,
            role=role,
        )

    def resolve_placeholder(
        self,
        case_id: str,
        placeholder: str,
        purpose: ReidentificationPurpose,
        authorization: AuthorizationToken,
    ) -> str:
        # Human-authorized path only (WP §1.1). No LLM path exists.
        if authorization is None or not authorization.authorizes(purpose):
            raise VaultAuthorizationError(
                "re-identification denied: missing or insufficient human authorization"
            )
        entry = self._sealed.get((case_id, placeholder))
        if entry is None:
            raise VaultError(f"unknown placeholder for case: {placeholder}")
        blob, aad, _, _ = entry
        plaintext = open_bytes(blob, aad, self._case_keys[case_id])
        return plaintext.decode("utf-8")

    def seal_mapping(
        self,
        case_id: str,
        mapping: EntityMapping,
        aad: SealingAAD,
    ) -> SealedMappingRef:
        self._ensure_case(case_id)
        blob = seal_bytes(
            mapping.raw_value.encode("utf-8"), aad, self._case_keys[case_id]
        )
        return SealedMappingRef(
            placeholder=mapping.placeholder,
            key_id=blob.key_id,
            aad_sha256=blob.aad_sha256,
        )
