"""ClaimRecord 1.0 and Applicability 1.1 (WP_TENDER_LAYERS §I.7, D-0021) — constructor contracts.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Synthetic values only; no corpus text.
"""

from __future__ import annotations

import pytest

from ragix_kernels.tender.domain.claims import ClaimRecord, ClaimValue, Provenance, SourceSpan, sha256_text
from ragix_kernels.tender.domain.records import Applicability

RAW = "Lundi\n2 mars 202\n6 à 9h30"
CHANNEL = "tender.dates_fr 1.0"
SPAN = SourceSpan("d" * 64, "c" * 64, ("0.1", "0.2"), 10, 10 + len(RAW), sha256_text(RAW))
CLAUSE = "au plus tard 3 jours avant"
CLAUSE_SPAN = SourceSpan("d" * 64, "e" * 64, ("0.9",), 5, 5 + len(CLAUSE), sha256_text(CLAUSE))
APP = Applicability(project="P", clause_scope={"pieces": "consultation", "lots": "all"})


def observed(**kw):
    base = dict(field="offer_deadline", value=ClaimValue("datetime", RAW, "2026-03-02T09:30"),
                provenance=Provenance("observed", (SPAN,), None, CHANNEL), applicability=APP)
    base.update(kw)
    return ClaimRecord(**base)


def test_claim_id_is_a_deterministic_sha256():
    c = observed()
    assert len(c.claim_id) == 64 and c.claim_id == observed().claim_id


def test_claim_id_follows_the_sources():
    other = SourceSpan("d" * 64, "c" * 64, ("0.1", "0.2"), 11, 11 + len(RAW), sha256_text(RAW))
    moved = observed(provenance=Provenance("observed", (other,), None, CHANNEL))
    assert moved.claim_id != observed().claim_id


def test_a_given_claim_id_must_match():
    with pytest.raises(ValueError):
        observed(claim_id="0" * 64)


def test_field_outside_the_slice():
    with pytest.raises(ValueError):
        observed(field="award_date")


def test_observed_keeps_exactly_one_span_and_no_derivation():
    with pytest.raises(ValueError):
        Provenance("observed", (SPAN, SPAN), None, CHANNEL)
    with pytest.raises(ValueError):
        Provenance("observed", (SPAN,), "x − 1 day", CHANNEL)


def test_derived_keeps_its_expression_and_its_operands():
    with pytest.raises(ValueError):
        Provenance("derived", (SPAN, CLAUSE_SPAN), None, CHANNEL)
    with pytest.raises(ValueError):
        Provenance("derived", (), "offer_deadline − 3 days", CHANNEL)
    d = ClaimRecord("questions_deadline", ClaimValue("datetime", CLAUSE, "2026-02-27T09:30"),
                    Provenance("derived", (SPAN, CLAUSE_SPAN), "offer_deadline − 3 days", CHANNEL), APP)
    assert d.provenance.origin == "derived" and len(d.provenance.sources) == 2


def test_normalized_is_iso_8601_and_a_calendar_value():
    for rtype, value in (("date", "2 mars 2026"), ("date", "2026-02-30"), ("datetime", "2026-03-02"),
                         ("date", "2026-03-02T09:30"), ("datetime", "2026-03-02T25:00")):
        with pytest.raises(ValueError):
            ClaimValue(rtype, RAW, value)


def test_raw_is_the_exact_text_of_a_source():
    with pytest.raises(ValueError):
        observed(value=ClaimValue("datetime", "Lundi 2 mars 2026 à 9h30", "2026-03-02T09:30"))


def test_channel_and_offsets_are_mandatory():
    with pytest.raises(ValueError):
        Provenance("observed", (SPAN,), None, "")
    with pytest.raises(ValueError):
        SourceSpan("d" * 64, "c" * 64, ("0.1",), 10, 10, sha256_text(""))
    with pytest.raises(ValueError):
        SourceSpan("d" * 64, "c" * 64, (), 10, 12, sha256_text("ab"))


def test_applicability_1_0_is_unchanged():
    a = Applicability(project="P")
    assert (a.clause_scope, a.authority, a.outcome, a.valid_from) == (None, None, "unknown", None)


def test_applicability_1_1_is_validated():
    span = {"doc_id": "d", "chunk_id": "c", "char_start": 0, "char_end": 4, "span_sha256": "f" * 64}
    Applicability(project="P", clause_scope={"pieces": ["d"], "lots": ["1"]},
                  authority={"rank": 1, "established_by": [span]})
    for scope in ({"pieces": [], "lots": "all"}, {"pieces": "consultation", "lots": "some"},
                  {"pieces": "consultation"}):
        with pytest.raises(ValueError):
            Applicability(project="P", clause_scope=scope)
    for authority in ({"rank": 0, "established_by": [span]}, {"rank": True, "established_by": [span]},
                      {"rank": 1, "established_by": []}, {"rank": 1, "established_by": [{"doc_id": "d"}]}):
        with pytest.raises(ValueError):
            Applicability(project="P", authority=authority)
