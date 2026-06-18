"""
Tests for the sealed AEAD primitive (WP §12).

Covers: round-trip, fresh-nonce non-determinism, and AAD tamper detection (wrong
case_id, wrong policy_version, wrong key, tampered ciphertext). Synthetic data only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

import dataclasses

import pytest

from ragix_core.crypto.sealed_aead import (
    KeyRef,
    SealedAEADError,
    SealedBlob,
    SealingAAD,
    open_bytes,
    seal_bytes,
)

PLAINTEXT = b"synthetic sealed payload \xe2\x80\x94 not a real secret"


def _aad(**overrides) -> SealingAAD:
    base = dict(
        case_id="case_synthetic_001",
        doc_id="doc_synthetic_aaaa",
        raw_sha256="0" * 64,
        policy_version="seal-policy-0.1",
        placeholder_schema_version="placeholder-schema-0.1",
        ingestion_state="PLACEHOLDERIZED",
    )
    base.update(overrides)
    return SealingAAD(**base)


def test_round_trip():
    key = KeyRef.generate("k1")
    aad = _aad()
    blob = seal_bytes(PLAINTEXT, aad, key)
    assert open_bytes(blob, aad, key) == PLAINTEXT


def test_fresh_nonce_non_deterministic():
    key = KeyRef.generate("k1")
    aad = _aad()
    b1 = seal_bytes(PLAINTEXT, aad, key)
    b2 = seal_bytes(PLAINTEXT, aad, key)
    assert b1.nonce != b2.nonce
    assert b1.ciphertext != b2.ciphertext
    # ...yet both open to the same plaintext.
    assert open_bytes(b1, aad, key) == open_bytes(b2, aad, key) == PLAINTEXT


def test_wrong_case_id_fails():
    key = KeyRef.generate("k1")
    blob = seal_bytes(PLAINTEXT, _aad(), key)
    with pytest.raises(SealedAEADError):
        open_bytes(blob, _aad(case_id="case_synthetic_999"), key)


def test_wrong_policy_version_fails():
    key = KeyRef.generate("k1")
    blob = seal_bytes(PLAINTEXT, _aad(), key)
    with pytest.raises(SealedAEADError):
        open_bytes(blob, _aad(policy_version="seal-policy-0.2"), key)


def test_wrong_ingestion_state_fails():
    key = KeyRef.generate("k1")
    blob = seal_bytes(PLAINTEXT, _aad(), key)
    with pytest.raises(SealedAEADError):
        open_bytes(blob, _aad(ingestion_state="RECEIVED"), key)


def test_wrong_key_fails():
    k1 = KeyRef.generate("k1")
    # Same key_id, different key material -> key_id matches but auth must still fail.
    import os

    k1_imposter = KeyRef(key_id="k1", key=os.urandom(32))
    aad = _aad()
    blob = seal_bytes(PLAINTEXT, aad, k1)
    with pytest.raises(SealedAEADError):
        open_bytes(blob, aad, k1_imposter)


def test_tampered_ciphertext_fails():
    key = KeyRef.generate("k1")
    aad = _aad()
    blob = seal_bytes(PLAINTEXT, aad, key)
    tampered = dataclasses.replace(
        blob, ciphertext=bytes([blob.ciphertext[0] ^ 0xFF]) + blob.ciphertext[1:]
    )
    with pytest.raises(SealedAEADError):
        open_bytes(tampered, aad, key)


def test_key_id_mismatch_rejected_early():
    key = KeyRef.generate("k1")
    blob = seal_bytes(PLAINTEXT, _aad(), key)
    other = KeyRef.generate("k2")
    with pytest.raises(SealedAEADError):
        open_bytes(blob, _aad(), other)


def test_bad_key_length_rejected():
    with pytest.raises(SealedAEADError):
        KeyRef(key_id="short", key=b"too-short")


def test_blob_serialization_round_trip():
    key = KeyRef.generate("k1")
    aad = _aad()
    blob = seal_bytes(PLAINTEXT, aad, key)
    restored = SealedBlob.from_dict(blob.to_dict())
    assert open_bytes(restored, aad, key) == PLAINTEXT
