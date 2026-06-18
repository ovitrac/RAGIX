"""
Tests for the native sealed vault backend (WP §12).

Covers placeholder stability/dedup, the human-authorized resolve path, denial without
authorization, and mapping sealing. Synthetic identities only ("Jane Synthetic", etc.).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-16
"""

import pytest

from ragix_core.crypto.sealed_aead import SealingAAD
from ragix_sealed.vault import (
    AuthorizationToken,
    EntityMapping,
    PlaceholderPolicy,
    PlaceholderRef,
    RAGIXSealedVaultBackend,
    ReidentificationPurpose,
    SealedVaultBackend,
    VaultAuthorizationError,
)

CASE = "case_synthetic_001"
PERSON_POLICY = PlaceholderPolicy(entity_type="PERSON")
ORG_POLICY = PlaceholderPolicy(entity_type="ORG")

# Synthetic fixtures — never real identities.
JANE = "Jane Synthetic"
JOHN = "John Synthetic"
ACME = "ACME Synthetic Ltd"


def _vault() -> RAGIXSealedVaultBackend:
    return RAGIXSealedVaultBackend()


def _auth(purpose=ReidentificationPurpose.AUTHORIZED_REVIEW) -> AuthorizationToken:
    return AuthorizationToken(subject="human::reviewer-1", purpose=purpose, token="grant-xyz")


def test_backend_satisfies_protocol():
    assert isinstance(_vault(), SealedVaultBackend)


def test_create_returns_public_placeholder_ref():
    v = _vault()
    ref = v.create_placeholder(CASE, "PERSON", JANE, role="primary_party", policy=PERSON_POLICY)
    assert isinstance(ref, PlaceholderRef)
    assert ref.placeholder == "[PERSON_001 | primary_party]"
    assert ref.entity_type == "PERSON"
    assert ref.role == "primary_party"


def test_placeholder_stability_same_raw_same_placeholder():
    v = _vault()
    r1 = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    r2 = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    assert r1.placeholder == r2.placeholder


def test_distinct_raw_distinct_placeholder():
    v = _vault()
    r1 = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    r2 = v.create_placeholder(CASE, "PERSON", JOHN, "witness", PERSON_POLICY)
    assert r1.placeholder != r2.placeholder
    assert r2.placeholder == "[PERSON_002 | witness]"


def test_counters_are_per_entity_type():
    v = _vault()
    p = v.create_placeholder(CASE, "PERSON", JANE, None, PERSON_POLICY)
    o = v.create_placeholder(CASE, "ORG", ACME, "employer", ORG_POLICY)
    assert p.placeholder == "[PERSON_001]"
    assert o.placeholder == "[ORG_001 | employer]"


def test_resolve_with_authorization_returns_raw_to_human():
    v = _vault()
    ref = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    raw = v.resolve_placeholder(CASE, ref.placeholder, ReidentificationPurpose.AUTHORIZED_REVIEW, _auth())
    assert raw == JANE


def test_resolve_without_authorization_denied():
    v = _vault()
    ref = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    with pytest.raises(VaultAuthorizationError):
        v.resolve_placeholder(CASE, ref.placeholder, ReidentificationPurpose.AUTHORIZED_REVIEW, None)


def test_resolve_with_wrong_purpose_denied():
    v = _vault()
    ref = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    # Token authorizes AUTHORIZED_REVIEW; request FORMAL_DISCLOSURE -> denied.
    auth = _auth(ReidentificationPurpose.AUTHORIZED_REVIEW)
    with pytest.raises(VaultAuthorizationError):
        v.resolve_placeholder(CASE, ref.placeholder, ReidentificationPurpose.FORMAL_DISCLOSURE, auth)


def test_seal_mapping_returns_opaque_ref():
    v = _vault()
    ref = v.create_placeholder(CASE, "PERSON", JANE, "primary_party", PERSON_POLICY)
    aad = SealingAAD(
        case_id=CASE,
        doc_id="doc_synthetic_aaaa",
        raw_sha256="0" * 64,
        policy_version="seal-policy-0.1",
        placeholder_schema_version="placeholder-schema-0.1",
        ingestion_state="PLACEHOLDERIZED",
    )
    mapping = EntityMapping(placeholder=ref.placeholder, entity_type="PERSON", raw_value=JANE, role="primary_party")
    sealed_ref = v.seal_mapping(CASE, mapping, aad)
    assert sealed_ref.placeholder == ref.placeholder
    assert sealed_ref.key_id
    assert sealed_ref.aad_sha256


def test_case_namespaces_are_isolated():
    """Placeholder namespaces are per-case; a placeholder unique to one case is unknown
    in another, and each case resolves its own value."""
    v = _vault()
    v.create_placeholder("case_A", "PERSON", JANE, "primary_party", PERSON_POLICY)
    a_john = v.create_placeholder("case_A", "PERSON", JOHN, "witness", PERSON_POLICY)  # [PERSON_002]
    b_jane = v.create_placeholder("case_B", "PERSON", JANE, "primary_party", PERSON_POLICY)  # [PERSON_001]

    # case_B never minted [PERSON_002]; resolving it under case_B must fail.
    from ragix_sealed.vault import VaultError

    with pytest.raises(VaultError):
        v.resolve_placeholder("case_B", a_john.placeholder, ReidentificationPurpose.AUTHORIZED_REVIEW, _auth())
    # Each case resolves its own.
    assert v.resolve_placeholder("case_A", a_john.placeholder, ReidentificationPurpose.AUTHORIZED_REVIEW, _auth()) == JOHN
    assert v.resolve_placeholder("case_B", b_jane.placeholder, ReidentificationPurpose.AUTHORIZED_REVIEW, _auth()) == JANE


def test_cross_case_key_isolation():
    """Crypto-level guarantee: a sealed blob from one case cannot be opened with another
    case's key (different per-case keys + AAD case_id binding)."""
    from ragix_core.crypto.sealed_aead import SealedAEADError, open_bytes

    v = _vault()
    a = v.create_placeholder("case_A", "PERSON", JANE, "primary_party", PERSON_POLICY)
    v.create_placeholder("case_B", "PERSON", JANE, "primary_party", PERSON_POLICY)

    blob_a, aad_a, _, _ = v._sealed[("case_A", a.placeholder)]  # noqa: SLF001 (spike: assert internal invariant)
    key_b = v._case_keys["case_B"]  # noqa: SLF001
    with pytest.raises(SealedAEADError):
        open_bytes(blob_a, aad_a, key_b)
