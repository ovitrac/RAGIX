"""
Sealed AEAD primitive for RAGIX-Sealed (WP_RAGIX_Sealed_v1.md §5).

Authenticated encryption with associated data (AEAD) over AES-256-GCM, used to
seal highly sensitive, confidential material. The associated data (AAD) is *load-bearing*:
it cryptographically binds each ciphertext to its case, document, content hash, policy
version, placeholder-schema version, and ingestion state. An attacker (or an accident)
cannot move ciphertext between cases, roll back the policy, substitute a document, or
replay it under a different state without failing authentication.

Design notes
------------
- AES-256-GCM (256-bit key, 96-bit fresh random nonce per encryption).
- The GCM authentication tag is verified *before* any plaintext is returned; a wrong key,
  wrong AAD, or tampered ciphertext raises ``SealedAEADError`` and yields no plaintext.
- This module is deliberately narrow: it knows nothing about placeholders, vaults, or
  documents. It only seals and opens bytes. Higher layers (``ragix_sealed.vault``) build
  the sealed-document lifecycle on top of it.
- This primitive is *additive*: it does not replace ``ragix_core.secrets_vault`` (Fernet),
  which remains valid for ordinary application secrets. AAD support is the reason Fernet
  is not reused here (Fernet has no AAD).

Threat-model reminder (WP §1): raw content stays visible to the authorized human; this
primitive exists to keep it away from the eyes of any LLM, with much-higher protection.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# AES-256-GCM constants.
KEY_SIZE_BYTES = 32  # 256-bit key
NONCE_SIZE_BYTES = 12  # 96-bit nonce, the GCM standard / recommended size
ALG = "AES-256-GCM"


class SealedAEADError(Exception):
    """Raised when sealing or opening fails.

    On open, this means authentication failed: wrong key, wrong AAD (e.g. a mismatched
    ``case_id`` or ``policy_version``), or tampered ciphertext. No plaintext is ever
    returned in this case.
    """


@dataclass(frozen=True)
class SealingAAD:
    """Associated data bound into every sealed blob (WP §5.1).

    All fields are non-secret identifiers/metadata. They are authenticated but NOT
    encrypted, so they must never carry raw confidential content (no names, no filenames).
    The same ``SealingAAD`` must be reconstructed at open time or authentication fails.
    """

    case_id: str
    doc_id: str
    raw_sha256: str
    policy_version: str
    placeholder_schema_version: str
    ingestion_state: str

    def to_bytes(self) -> bytes:
        """Deterministic, canonical encoding used as the GCM AAD.

        Canonical JSON (sorted keys, no insignificant whitespace) guarantees that the
        exact same byte string is produced on seal and on open, regardless of field
        construction order.
        """
        canonical = json.dumps(
            {
                "case_id": self.case_id,
                "doc_id": self.doc_id,
                "raw_sha256": self.raw_sha256,
                "policy_version": self.policy_version,
                "placeholder_schema_version": self.placeholder_schema_version,
                "ingestion_state": self.ingestion_state,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return canonical.encode("utf-8")


@dataclass(frozen=True)
class KeyRef:
    """A reference to a 256-bit symmetric key.

    For the Sprint 0 spike the key material is held directly. In production this becomes
    an envelope-encryption reference (per-document DEK wrapped by a master/KMS key, WP §5);
    ``key_id`` is the stable, non-secret handle used in audit records.
    """

    key_id: str
    key: bytes = field(repr=False)  # never shown in repr / logs

    def __post_init__(self) -> None:
        if len(self.key) != KEY_SIZE_BYTES:
            raise SealedAEADError(
                f"key must be {KEY_SIZE_BYTES} bytes (AES-256); got {len(self.key)}"
            )

    @classmethod
    def generate(cls, key_id: str) -> "KeyRef":
        """Generate a fresh random 256-bit key."""
        return cls(key_id=key_id, key=AESGCM.generate_key(bit_length=256))


@dataclass(frozen=True)
class SealedBlob:
    """A sealed (encrypted + authenticated) payload.

    Contains only ciphertext and non-secret metadata. It is safe to persist and to expose
    as an opaque object, but it is NOT a public-facing content object: it carries no
    plaintext, no raw values, and no AAD field values beyond ``aad_sha256`` (a digest used
    only for audit correlation).
    """

    alg: str
    key_id: str
    nonce: bytes = field(repr=False)
    ciphertext: bytes = field(repr=False)  # includes the appended GCM tag
    aad_sha256: str  # digest of the AAD, for audit correlation only

    def to_dict(self) -> Dict[str, Any]:
        """Base64 serialization suitable for persistence in the sealed store."""
        return {
            "alg": self.alg,
            "key_id": self.key_id,
            "nonce": base64.b64encode(self.nonce).decode("ascii"),
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "aad_sha256": self.aad_sha256,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SealedBlob":
        return cls(
            alg=d["alg"],
            key_id=d["key_id"],
            nonce=base64.b64decode(d["nonce"]),
            ciphertext=base64.b64decode(d["ciphertext"]),
            aad_sha256=d["aad_sha256"],
        )


def _aad_digest(aad_bytes: bytes) -> str:
    """SHA-256 of the AAD bytes, hex. Used only for audit correlation, never for auth."""
    import hashlib

    return hashlib.sha256(aad_bytes).hexdigest()


def seal_bytes(plaintext: bytes, aad: SealingAAD, key: KeyRef) -> SealedBlob:
    """Seal ``plaintext`` under ``key``, binding it to ``aad``.

    A fresh 96-bit random nonce is generated for every call, so sealing identical
    plaintext twice yields different blobs.

    Raises:
        SealedAEADError: on any encryption failure.
    """
    if not isinstance(plaintext, (bytes, bytearray)):
        raise SealedAEADError("plaintext must be bytes")
    aad_bytes = aad.to_bytes()
    nonce = os.urandom(NONCE_SIZE_BYTES)
    try:
        ciphertext = AESGCM(key.key).encrypt(nonce, bytes(plaintext), aad_bytes)
    except Exception as exc:  # pragma: no cover - defensive
        raise SealedAEADError(f"sealing failed: {exc}") from exc
    return SealedBlob(
        alg=ALG,
        key_id=key.key_id,
        nonce=nonce,
        ciphertext=ciphertext,
        aad_sha256=_aad_digest(aad_bytes),
    )


def open_bytes(sealed: SealedBlob, aad: SealingAAD, key: KeyRef) -> bytes:
    """Open ``sealed`` under ``key`` and ``aad``, returning the plaintext.

    The GCM tag is verified before any plaintext is produced. If the key is wrong, the
    AAD does not match what was sealed (e.g. a different ``case_id`` or ``policy_version``),
    or the ciphertext was tampered with, this raises ``SealedAEADError`` and returns
    nothing.
    """
    if sealed.alg != ALG:
        raise SealedAEADError(f"unsupported algorithm: {sealed.alg!r}")
    if sealed.key_id != key.key_id:
        raise SealedAEADError("key_id mismatch between blob and supplied key")
    aad_bytes = aad.to_bytes()
    try:
        return AESGCM(key.key).decrypt(sealed.nonce, sealed.ciphertext, aad_bytes)
    except InvalidTag as exc:
        raise SealedAEADError(
            "authentication failed: wrong key, mismatched AAD, or tampered ciphertext"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise SealedAEADError(f"opening failed: {exc}") from exc
