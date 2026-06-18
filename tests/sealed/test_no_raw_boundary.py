"""
Boundary regression tests (WP §12 amendment, §11 / rule K2).

These lock the core doctrine in code: public-facing objects produced by the sealed vault
must never expose raw values, original filenames, plaintext, decrypted content, vault
mappings, prompt text, or source paths — neither as field names nor as field values.

Synthetic fixtures only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

import json

import pytest

from ragix_core.crypto.sealed_aead import KeyRef, SealingAAD, seal_bytes
from ragix_sealed.vault import (
    EntityMapping,
    PlaceholderPolicy,
    RAGIXSealedVaultBackend,
)

# Field names that must never appear in a public-facing object.
FORBIDDEN_KEYS = {
    "raw_value",
    "original_filename",
    "plaintext",
    "decrypted",
    "vault_mapping",
    "prompt_text",
    "source_path",
}

CASE = "case_synthetic_001"
JANE = "Jane Synthetic"
PERSON_POLICY = PlaceholderPolicy(entity_type="PERSON")


def _assert_no_forbidden_keys(obj):
    """Recursively assert no forbidden key appears in a JSON-able structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in FORBIDDEN_KEYS, f"forbidden key leaked: {k}"
            _assert_no_forbidden_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_forbidden_keys(v)


def _assert_raw_absent(blob_str: str, raw: str = JANE):
    assert raw not in blob_str, "raw synthetic value leaked into a public-facing object"


def test_placeholder_ref_has_no_raw():
    v = RAGIXSealedVaultBackend()
    ref = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    d = ref.to_public_dict()
    _assert_no_forbidden_keys(d)
    _assert_raw_absent(json.dumps(d))
    # The role label is intentionally visible; the raw identity is not.
    assert ref.role == "primary_party"


def test_sealed_blob_dict_has_no_raw():
    key = KeyRef.generate("k1")
    aad = SealingAAD(
        case_id=CASE,
        doc_id="doc_synthetic_aaaa",
        raw_sha256="0" * 64,
        policy_version="seal-policy-0.1",
        placeholder_schema_version="placeholder-schema-0.1",
        ingestion_state="PLACEHOLDERIZED",
    )
    blob = seal_bytes(JANE.encode("utf-8"), aad, key)
    d = blob.to_dict()
    _assert_no_forbidden_keys(d)
    # Ciphertext is base64 of encrypted bytes; the raw value must not be recoverable as text.
    _assert_raw_absent(json.dumps(d))


def test_sealed_mapping_ref_has_no_raw():
    v = RAGIXSealedVaultBackend()
    ref = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    aad = SealingAAD(
        case_id=CASE,
        doc_id="doc_synthetic_aaaa",
        raw_sha256="0" * 64,
        policy_version="seal-policy-0.1",
        placeholder_schema_version="placeholder-schema-0.1",
        ingestion_state="PLACEHOLDERIZED",
    )
    mapping = EntityMapping(ref.placeholder, "PERSON", raw_value=JANE, role="primary_party")
    sealed_ref = v.seal_mapping(CASE, mapping, aad)
    d = sealed_ref.to_public_dict()
    _assert_no_forbidden_keys(d)
    _assert_raw_absent(json.dumps(d))


def test_entity_mapping_repr_hides_raw():
    """EntityMapping is internal, but even its repr must not echo the raw value."""
    mapping = EntityMapping("[PERSON_001]", "PERSON", raw_value=JANE, role="primary_party")
    _assert_raw_absent(repr(mapping))


def test_keyref_repr_hides_key_material():
    key = KeyRef.generate("k1")
    r = repr(key)
    # The 32-byte key must not appear in any text form.
    assert key.key.hex() not in r
    assert "key=" not in r or "..." in r or "key_id" in r


def test_authorization_token_repr_hides_token():
    from ragix_sealed.vault import AuthorizationToken, ReidentificationPurpose

    tok = AuthorizationToken("human::reviewer-1", ReidentificationPurpose.AUTHORIZED_REVIEW, token="grant-xyz")
    assert "grant-xyz" not in repr(tok)
