"""
RAGIX-Sealed — red-team & leakage suite (WP §23 Sprint 7, §24 acceptance).

Seeds synthetic canary secrets, runs the full pipeline (ingest → cooled corpus →
analysis → report), and asserts that NO canary leaks into any public surface. Also checks:
the known-entity exact scanner catches an arbitrary registered secret, blocked output when
placeholderization is bypassed, and that the human-authorized path *does* surface values
(the boundary is human↔LLM, not human↔system). All synthetic.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import json

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.corpus import CooledCorpusIndex
from ragix_sealed.corpus.chunking import chunk_cooled
from ragix_sealed.ingest import SealedIngestor, new_case_context
from ragix_sealed.ingest.leak_scan import scan as leak_scan
from ragix_sealed.kernels import run_analysis
from ragix_sealed.reporting import (
    ExportMode,
    build_audit_attestation,
    build_sanitized_memo,
    render,
)
from ragix_sealed.vault import RAGIXSealedVaultBackend
from ragix_sealed.vault.backend import AuthorizationToken, ReidentificationPurpose

# Canary secrets — all detectable by v0 (email/phone/amount/reference/date).
CANARIES = [
    "canary-7f3a@example.invalid",
    "+33 6 11 22 33 44",
    "9 999,99 EUR",
    "2099-12-31",
    "77-8899",
]
DOC = (
    "Project update dated 2099-12-31.\n\n"
    "Email canary-7f3a@example.invalid and phone +33 6 11 22 33 44.\n\n"
    "Budget 9 999,99 EUR under reference n° 77-8899."
).encode("utf-8")

FORBIDDEN_KEYS = {
    "raw_value", "original_filename", "plaintext", "decrypted",
    "vault_mapping", "prompt_text", "source_path",
}


def _assert_no_canary(blob: str, where: str):
    for c in CANARIES:
        assert c not in blob, f"CANARY LEAK in {where}: {c!r}"


def _assert_no_forbidden_keys(obj, where: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in FORBIDDEN_KEYS, f"forbidden key {k!r} in {where}"
            _assert_no_forbidden_keys(v, where)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_forbidden_keys(v, where)


def _pipeline():
    c = load_contracts()
    ing = SealedIngestor(c, RAGIXSealedVaultBackend())
    ctx = new_case_context("case_rt_001")
    status = ing.ingest(ctx, DOC, "txt", ingestion_counter=1)
    cooled = ing.get_cooled(status.doc_id)
    return c, ctx, ing, status, cooled


# -- full-surface non-leakage ------------------------------------------------------------

def test_cooled_text_and_status_have_no_canary():
    c, ctx, ing, status, cooled = _pipeline()
    assert status.state == "COOLED_INDEXABLE"
    _assert_no_canary(cooled.text, "cooled text")
    _assert_no_canary(json.dumps(status.to_public_dict()), "IngestStatus")
    _assert_no_forbidden_keys(status.to_public_dict(), "IngestStatus")


def test_corpus_search_has_no_canary():
    c, ctx, ing, status, cooled = _pipeline()
    idx = CooledCorpusIndex(c)
    idx.add_document(ctx, status.doc_id, cooled.text)
    # search benign terms AND a canary term — neither may echo a raw secret
    for q in ["project update", "budget", "canary", "example.invalid"]:
        for hit in idx.search(q, k=5):
            _assert_no_canary(json.dumps(hit.to_public_dict()), f"search[{q}]")


def test_analysis_findings_have_no_canary():
    c, ctx, ing, status, cooled = _pipeline()
    chunks = chunk_cooled(ctx, status.doc_id, cooled.text)
    results = run_analysis(chunks, c.placeholder_schema)
    _assert_no_canary(json.dumps(results), "analysis findings")


def test_reports_have_no_canary():
    c, ctx, ing, status, cooled = _pipeline()
    chunks = chunk_cooled(ctx, status.doc_id, cooled.text)
    analysis = run_analysis(chunks, c.placeholder_schema)
    inventory = {
        "corpus_metrics": {"documents_total": 1, "documents_indexable": 1,
                           "documents_blocked": 0, "pages_total": 1, "human_review_required": 0},
        "review_queue": {"review_required_docs": [], "blocked_docs": []},
    }
    memo = build_sanitized_memo(inventory, analysis)
    sanitized = render(memo, ExportMode.SANITIZED_LLM_SAFE, schema=c.placeholder_schema)
    _assert_no_canary(sanitized, "sanitized memo")

    att = build_audit_attestation("case_rt_001", inventory, "sha256:tip",
                                  "seal-policy-0.1", "placeholder-schema-0.1")
    _assert_no_canary(json.dumps(att), "audit attestation")


# -- leak scanner mechanism --------------------------------------------------------------

def test_known_entity_scanner_catches_arbitrary_secret():
    """Even an entity v0 cannot detect by pattern (a name) is caught if registered."""
    c = load_contracts()
    secret = "Zzyzx Canary Nomine"
    verdict = leak_scan(f"text mentioning {secret} here", [secret], c.placeholder_schema)
    assert verdict.verdict == "FAIL"


def test_blocked_when_placeholderization_bypassed(monkeypatch):
    c, ctx, ing, _status, _cooled = _pipeline()
    ing2 = SealedIngestor(c, RAGIXSealedVaultBackend())
    monkeypatch.setattr(ing2, "_placeholderize",
                        lambda case_id, text, dets: (text, [], [d.value for d in dets]))
    blocked = ing2.ingest(new_case_context("case_rt_002"), DOC, "txt", ingestion_counter=1)
    assert blocked.state == "BLOCKED"
    assert blocked.leak_verdict == "FAIL"
    assert ing2.get_cooled(blocked.doc_id) is None


# -- the human path still works (boundary is human <-> LLM) ------------------------------

def test_human_authorized_export_can_reidentify():
    c, ctx, ing, status, cooled = _pipeline()
    memo = build_sanitized_memo(
        {"corpus_metrics": {}, "review_queue": {}},
        {"entity_role_graph": [{"nodes": [{"placeholder": "[EMAIL_001]", "entity_type": "EMAIL"}], "edges": []}]},
    )
    auth = AuthorizationToken("human::reviewer-1", ReidentificationPurpose.REPORT_EXPORT_HUMAN, token="g")
    out = render(memo, ExportMode.HUMAN_AUTHORIZED, authorization=auth,
                 resolver=lambda t: "canary-7f3a@example.invalid" if t == "[EMAIL_001]" else t)
    # The authorized human DOES see the real value (and it is watermarked).
    assert "canary-7f3a@example.invalid" in out
    assert out.startswith("> [HUMAN-AUTHORIZED")
