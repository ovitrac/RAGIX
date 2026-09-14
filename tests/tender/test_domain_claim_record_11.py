"""ClaimRecord 1.1 in code, as signed in WP §2.2 (D-0022), registered as D-0024.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The signed block adds two origins and one prohibition: `aggregated` (a branch's entry references its
children's claims and copies no span of its own), `interpreted` (a statement about children — a conflict,
an authority rank, a coverage judgement, a relevance), and **no interpreted claim may carry a value of
type date, datetime, amount, percentage, duration or quantity**. And 1.0 is a subset of 1.1: every claim
already made must keep its id byte for byte.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragix_kernels.tender.domain.claims import CLAIM_SCHEMA_VERSION, ClaimRecord, ClaimValue, Provenance, SourceSpan
from ragix_kernels.tender.domain.records import Applicability

CLAIMS = Path(__file__).resolve().parent / "fixtures" / "claims_deadline_slice.jsonl"


def _rebuild(d: dict) -> ClaimRecord:
    p = d["provenance"]
    return ClaimRecord(field=d["field"], value=ClaimValue(**d["value"]),
                       provenance=Provenance(origin=p["origin"], derivation=p["derivation"], channel=p["channel"],
                                             sources=tuple(SourceSpan(**{**s, "node_ids": tuple(s["node_ids"])})
                                                           for s in p["sources"])),
                       applicability=Applicability(**d["applicability"]), schema_version=d["schema_version"],
                       claim_id=d["claim_id"])


def _existing() -> list[dict]:
    return [json.loads(l) for l in CLAIMS.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_the_version_is_one_one():
    assert CLAIM_SCHEMA_VERSION == "1.1"


def test_every_claim_already_made_keeps_its_id():
    """1.0 is a subset of 1.1: rebuilding each of the 50 claims recomputes exactly the id it was stored with."""
    rows = _existing()
    assert len(rows) == 50
    for d in rows:
        assert _rebuild(d).claim_id == d["claim_id"]


def _children(n: int = 2) -> list[ClaimRecord]:
    return [_rebuild(d) for d in _existing() if d["field"] == "offer_deadline"][:n]


def test_an_aggregated_claim_references_its_children():
    kids = _children()
    agg = ClaimRecord(field="offer_deadline", value=kids[0].value,
                      provenance=Provenance(origin="aggregated", sources=tuple(k.claim_id for k in kids),
                                            derivation=None, channel="tender.pyramid 1.0"),
                      applicability=kids[0].applicability)
    assert agg.schema_version == "1.1" and len(agg.claim_id) == 64
    with pytest.raises(ValueError):                       # a child reference is a claim id, nothing else
        Provenance(origin="aggregated", sources=("not-an-id",), derivation=None, channel="tender.pyramid 1.0")


def test_an_interpreted_claim_may_state_a_conflict():
    kids = _children()
    flag = ClaimRecord(field="conflict", value=ClaimValue(type="flag", raw="true", normalized="true"),
                       provenance=Provenance(origin="interpreted", sources=tuple(k.claim_id for k in kids),
                                             derivation="two values for one field (row 14)",
                                             channel="tender.authority 1.0"),
                       applicability=kids[0].applicability)
    assert flag.provenance.origin == "interpreted"


@pytest.mark.parametrize("vtype, raw, norm", [("datetime", "18 septembre 2026 à 12h00", "2026-09-18T12:00"),
                                              ("date", "18 septembre 2026", "2026-09-18")])
def test_the_prohibition_an_interpreted_claim_carries_no_critical_value(vtype, raw, norm):
    kids = _children()
    with pytest.raises(ValueError, match="may not carry"):
        ClaimRecord(field="conflict", value=ClaimValue(type=vtype, raw=raw, normalized=norm),
                    provenance=Provenance(origin="interpreted", sources=tuple(k.claim_id for k in kids),
                                          derivation="a judgement", channel="tender.authority 1.0"),
                    applicability=kids[0].applicability)
