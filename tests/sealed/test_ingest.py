"""
Tests for the Sprint 2 warm ingestion pipeline (WP §8).

End-to-end cooling on synthetic text, illegal-transition guard, leak FAIL ⇒ BLOCKED,
no-raw boundary on the public status and cooled document, and human-only original reopen.

Synthetic fixtures only — no real identities.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import json

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.ingest import (
    ExtractorUnavailable,
    IngestError,
    SealedIngestor,
    detect_entities,
    extract_text,
    new_case_context,
    raw_sha256,
)
from ragix_sealed.ingest.leak_scan import scan
from ragix_sealed.vault import RAGIXSealedVaultBackend

# Synthetic document: contains an email, a phone, a date, an amount, a case number.
SYNTH_DOC = (
    "Memo dated 2024-03-15. Contact jane.synthetic@example.com or +33 1 23 45 67 89. "
    "Invoice total 1 250,00 EUR under case n° 12-3456. End of synthetic memo."
).encode("utf-8")

RAW_BITS = ["jane.synthetic@example.com", "+33 1 23 45 67 89", "2024-03-15", "12-3456"]


def _ingestor():
    return SealedIngestor(load_contracts(), RAGIXSealedVaultBackend())


# -- extraction --------------------------------------------------------------------------

def test_extract_txt():
    assert "synthetic memo" in extract_text(b"a synthetic memo", "txt")


def test_extract_pdf_requires_sealed_extra():
    # Without the optional 'sealed' extra installed, PDF extraction raises a clear hint.
    # (When pypdf/pdfminer.six ARE installed, this path returns text instead.)
    pytest.importorskip  # keep import available; no skip — we assert the unavailable path
    try:
        import pdfminer  # noqa: F401
        import_available = True
    except ImportError:
        import_available = False
    if import_available:
        pytest.skip("pdfminer installed; unavailable-path test not applicable")
    with pytest.raises(ExtractorUnavailable, match="PDF"):
        extract_text(b"%PDF-1.4", "pdf")


# -- detection ---------------------------------------------------------------------------

def test_detect_finds_patterns():
    c = load_contracts()
    dets = detect_entities(SYNTH_DOC.decode(), c.placeholder_schema)
    types = {d.entity_type for d in dets}
    assert {"EMAIL", "PHONE", "DATE", "AMOUNT", "REFERENCE"} <= types


def test_detect_spans_non_overlapping():
    c = load_contracts()
    dets = detect_entities(SYNTH_DOC.decode(), c.placeholder_schema)
    last = -1
    for d in dets:
        assert d.start >= last
        last = d.end


# -- end-to-end cooling ------------------------------------------------------------------

def test_ingest_reaches_cooled_indexable():
    ing = _ingestor()
    ctx = new_case_context("case_synth_001")
    status = ing.ingest(ctx, SYNTH_DOC, "txt", ingestion_counter=1)
    assert status.state == "COOLED_INDEXABLE"
    assert status.leak_verdict == "PASS"
    assert status.entity_counts.get("EMAIL", 0) >= 1
    assert status.doc_id.startswith("doc_")


def test_cooled_text_has_placeholders_not_raw():
    ing = _ingestor()
    ctx = new_case_context("case_synth_001")
    status = ing.ingest(ctx, SYNTH_DOC, "txt", ingestion_counter=1)
    cooled = ing.get_cooled(status.doc_id)
    assert cooled is not None
    for raw in RAW_BITS:
        assert raw not in cooled.text, f"raw value leaked into cooled text: {raw!r}"
    assert "[EMAIL_" in cooled.text


def test_doc_id_stable_and_opaque():
    ctx = new_case_context("case_synth_001")
    rawh = raw_sha256(SYNTH_DOC)
    assert ctx.doc_id(rawh, 1) == ctx.doc_id(rawh, 1)        # stable
    assert ctx.doc_id(rawh, 1) != ctx.doc_id(rawh, 2)        # counter changes id
    assert "synthetic" not in ctx.doc_id(rawh, 1)            # opaque


# -- boundary safety ---------------------------------------------------------------------

def test_status_public_dict_has_no_raw():
    ing = _ingestor()
    ctx = new_case_context("case_synth_001")
    status = ing.ingest(ctx, SYNTH_DOC, "txt", ingestion_counter=1)
    blob = json.dumps(status.to_public_dict())
    for raw in RAW_BITS:
        assert raw not in blob


# -- leak scan FAIL path -----------------------------------------------------------------

def test_leak_scan_fails_on_residual_raw():
    c = load_contracts()
    # Placeholderized text that still leaks a raw email -> FAIL.
    v = scan("contact jane.synthetic@example.com still here", ["jane.synthetic@example.com"], c.placeholder_schema)
    assert v.verdict == "FAIL"


def test_leak_scan_fails_on_residual_pattern():
    c = load_contracts()
    # No known raw value supplied, but a detectable email remains -> FAIL via re-scan.
    v = scan("contact other@example.com", [], c.placeholder_schema)
    assert v.verdict == "FAIL"


def test_ingest_blocks_when_leak_detected(monkeypatch):
    """If placeholderization is bypassed, the leak scanner must block (state=BLOCKED)."""
    ing = _ingestor()
    ctx = new_case_context("case_synth_001")

    # Force placeholderization to a no-op so raw values survive to the leak scan.
    def _noop(case_id, text, detections):
        return text, [], [d.value for d in detections]

    monkeypatch.setattr(ing, "_placeholderize", _noop)
    status = ing.ingest(ctx, SYNTH_DOC, "txt", ingestion_counter=1)
    assert status.state == "BLOCKED"
    assert status.leak_verdict == "FAIL"
    assert ing.get_cooled(status.doc_id) is None


# -- state machine guard -----------------------------------------------------------------

def test_illegal_transition_rejected():
    ing = _ingestor()
    with pytest.raises(IngestError, match="illegal transition"):
        ing._advance("RECEIVED", "COOLED_INDEXABLE")  # not an allowed edge


# -- original reopen (human/internal path) ----------------------------------------------

def test_open_original_round_trips():
    ing = _ingestor()
    ctx = new_case_context("case_synth_001")
    status = ing.ingest(ctx, SYNTH_DOC, "txt", ingestion_counter=1)
    assert ing.open_original(ctx, status.doc_id) == SYNTH_DOC
