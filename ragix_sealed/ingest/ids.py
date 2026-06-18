"""
RAGIX-Sealed — opaque identifiers and case context (WP §5.2, Sprint 2).

Public identifiers are never derived from filenames or case titles. They are HMAC-derived
from the case secret, so they are stable, unlinkable across cases, and reveal nothing.

The ``CaseContext`` owns the case secret and the symmetric key used to seal *original
bytes*. This is deliberately separate from the entity vault's own key (key separation:
sealed originals and the entity vault are different sealing domains, WP §8).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field

from ragix_core.crypto.sealed_aead import KeyRef

DEFAULT_POLICY_VERSION = "seal-policy-0.1"
DEFAULT_SCHEMA_VERSION = "placeholder-schema-0.1"

_SEP = "\x1f"  # unit separator — cannot appear in the hex/ascii parts we join


def raw_sha256(data: bytes) -> str:
    """Hex SHA-256 of raw bytes (the content fingerprint)."""
    return hashlib.sha256(data).hexdigest()


def _hmac_hex(secret: bytes, *parts: str) -> str:
    msg = _SEP.join(parts).encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class CaseContext:
    """Per-case secrets + sealing key for ingestion.

    ``case_secret`` drives opaque-id derivation; ``key`` seals original bytes. Both are
    hidden from ``repr`` so they never land in logs.
    """

    case_id: str
    case_secret: bytes = field(repr=False)
    key: KeyRef = field(repr=False)
    policy_version: str = DEFAULT_POLICY_VERSION
    placeholder_schema_version: str = DEFAULT_SCHEMA_VERSION

    def doc_id(self, raw_hash: str, ingestion_counter: int) -> str:
        """`doc_id = HMAC(case_secret, raw_hash | counter | policy_version)` (§5.2)."""
        h = _hmac_hex(self.case_secret, raw_hash, str(ingestion_counter), self.policy_version)
        return "doc_" + h[:12]

    def chunk_id(self, doc_id: str, page: int, block: int, chunk_version: str = "1") -> str:
        """`chunk_id = HMAC(case_secret, doc_id | page | block | chunk_version)` (§5.2)."""
        h = _hmac_hex(self.case_secret, doc_id, str(page), str(block), chunk_version)
        return "chunk_" + h[:8]


def new_case_context(
    case_id: str,
    policy_version: str = DEFAULT_POLICY_VERSION,
    placeholder_schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> CaseContext:
    """Create a fresh case context with a random secret and a fresh 256-bit sealing key."""
    return CaseContext(
        case_id=case_id,
        case_secret=os.urandom(32),
        key=KeyRef.generate(key_id=f"orig-key::{case_id}"),
        policy_version=policy_version,
        placeholder_schema_version=placeholder_schema_version,
    )
