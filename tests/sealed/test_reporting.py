"""
Tests for the Sprint 6 reporting kernels (WP §20, §10.1 export modes).

Sanitized memo build, leak-gated sanitized export, human-authorized re-identification
(gated + watermarked), audit-only / orchestrator-metrics shapes. Synthetic only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.reporting import (
    ExportMode,
    ReportAuthorizationError,
    ReportError,
    SealedReport,
    ReportSection,
    build_audit_attestation,
    build_commitment_matrix,
    build_sanitized_memo,
    render,
)
from ragix_sealed.vault.backend import AuthorizationToken, ReidentificationPurpose

SCHEMA = load_contracts().placeholder_schema

INVENTORY = {
    "corpus_metrics": {"documents_total": 3, "documents_indexable": 2, "documents_blocked": 1,
                       "pages_total": 6, "human_review_required": 1},
    "review_queue": {"review_required_docs": ["doc_b"], "blocked_docs": ["doc_c"]},
}
ANALYSIS = {
    "entity_role_graph": [{"nodes": [{"placeholder": "[PERSON_001]", "entity_type": "PERSON",
                                      "role": "primary_party"}], "edges": []}],
    "timeline": [{"date": "[DATE_017]", "citation": "doc_a/page_1/chunk_01", "snippet": "x"}],
    "commitment_v0": [{"candidate": "x", "cues": ["shall"], "entities": ["[ORG_001]"],
                       "citation": "doc_a/page_1/chunk_02", "confidence": 0.4, "requires_review": True}],
    "gap_v0": [{"reference": "[DOCUMENT_004]", "entity_type": "DOCUMENT",
                "citations": ["doc_a/page_1/chunk_02"], "requires_review": True}],
}


def _memo():
    return build_sanitized_memo(INVENTORY, ANALYSIS)


def test_memo_has_sections_and_placeholders():
    md = _memo().to_markdown()
    for heading in ["Corpus Summary", "Key Actors by Role", "Chronology",
                    "Commitments", "Referenced Artifacts", "Review Points"]:
        assert heading in md
    assert "[PERSON_001]" in md and "[DATE_017]" in md


def test_sanitized_render_passes_leak_scan():
    text = render(_memo(), ExportMode.SANITIZED_LLM_SAFE, schema=SCHEMA)
    assert "[PERSON_001]" in text
    assert "@" not in text  # no raw


def test_sanitized_render_blocks_on_residual_raw():
    leaky = SealedReport("Bad", [ReportSection("S", ["- contact raw@example.com"])])
    with pytest.raises(ReportError, match="leak scan"):
        render(leaky, ExportMode.SANITIZED_LLM_SAFE, schema=SCHEMA)


def test_human_authorized_requires_authorization():
    with pytest.raises(ReportAuthorizationError):
        render(_memo(), ExportMode.HUMAN_AUTHORIZED, resolver=lambda t: t)


def test_human_authorized_reidentifies_and_watermarks():
    auth = AuthorizationToken("human::reviewer-1", ReidentificationPurpose.REPORT_EXPORT_HUMAN, token="g")
    mapping = {"[PERSON_001]": "Jane Synthetic", "[DATE_017]": "2024-03-15",
               "[ORG_001]": "ACME Synthetic", "[DOCUMENT_004]": "Annex 4"}
    out = render(_memo(), ExportMode.HUMAN_AUTHORIZED, authorization=auth,
                 resolver=lambda t: mapping.get(t, t))
    assert out.startswith("> [HUMAN-AUTHORIZED")
    assert "Jane Synthetic" in out and "[PERSON_001]" not in out


def test_audit_only_returns_attestation_no_content():
    att = build_audit_attestation("case_1", INVENTORY, "sha256:tip", "seal-policy-0.1",
                                  "placeholder-schema-0.1", {"corpus_metrics": "1.0"})
    out = render(_memo(), ExportMode.AUDIT_ONLY, attestation=att)
    assert out["case_id"] == "case_1" and out["documents_total"] == 3
    assert "snippet" not in str(out) and "[PERSON_001]" not in str(out)


def test_orchestrator_metrics_returns_counts():
    out = render(_memo(), ExportMode.ORCHESTRATOR_METRICS, metrics=INVENTORY["corpus_metrics"])
    assert out["documents_total"] == 3


def test_commitment_matrix_builds():
    md = build_commitment_matrix(ANALYSIS).to_markdown()
    assert "Commitment Matrix" in md and "shall" in md


# -- vault-backed re-identification (end-to-end) -----------------------------------------

def test_vault_resolver_reidentifies_real_placeholder():
    from ragix_sealed.ingest import SealedIngestor, new_case_context
    from ragix_sealed.reporting import render_reidentified
    from ragix_sealed.vault import RAGIXSealedVaultBackend

    contracts = load_contracts()
    vault = RAGIXSealedVaultBackend()
    ing = SealedIngestor(contracts, vault)
    ctx = new_case_context("case_reid_001")
    doc = b"Contact jane.synthetic@example.com about the review."
    status = ing.ingest(ctx, doc, "txt", ingestion_counter=1)
    # The vault now holds [EMAIL_001] -> the raw email for this case.
    report = SealedReport("Memo", [ReportSection("Contacts", ["- [EMAIL_001]"])])

    auth = AuthorizationToken("human::reviewer-1", ReidentificationPurpose.REPORT_EXPORT_HUMAN, token="g")
    out = render_reidentified(report, vault, ctx.case_id, auth)
    assert out.startswith("> [HUMAN-AUTHORIZED")
    assert "jane.synthetic@example.com" in out
    assert "[EMAIL_001]" not in out


def test_vault_reidentify_denied_without_authorization():
    from ragix_sealed.reporting import render_reidentified
    from ragix_sealed.vault import RAGIXSealedVaultBackend

    bad = AuthorizationToken("x", ReidentificationPurpose.AUDIT_INSPECTION, token="g")  # wrong purpose
    report = SealedReport("Memo", [ReportSection("S", ["- [EMAIL_001]"])])
    with pytest.raises(ReportAuthorizationError):
        render_reidentified(report, RAGIXSealedVaultBackend(), "case_x", bad)


def test_vault_resolver_leaves_unknown_placeholder_untouched():
    from ragix_sealed.reporting import vault_resolver
    from ragix_sealed.vault import RAGIXSealedVaultBackend

    auth = AuthorizationToken("h", ReidentificationPurpose.REPORT_EXPORT_HUMAN, token="g")
    resolve = vault_resolver(RAGIXSealedVaultBackend(), "case_empty", auth)
    assert resolve("[PERSON_999]") == "[PERSON_999]"  # unknown -> left placeholderized
