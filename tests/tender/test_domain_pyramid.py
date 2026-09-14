"""tender.derived and tender.pyramid — the sidecar store and the three resolutions, on synthetic claims.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

No corpus text. The rule under test is the one the lead signed: an interpreted branch layer carries flags,
labels and references, never a critical value.
"""

from __future__ import annotations

import re

import pytest

from ragix_kernels.tender.domain.claims import ClaimRecord, ClaimValue, Provenance, SourceSpan, sha256_text
try:  # the sidecar derived store belongs to the harvest family, ported on its own branch
    from ragix_kernels.harvest.derived import DETERMINISTIC, DerivedStore, KnowledgeRow, NodeRow
except ImportError:
    DETERMINISTIC = DerivedStore = KnowledgeRow = NodeRow = None
needs_derived = pytest.mark.skipif(DerivedStore is None,
                                   reason="ragix_kernels.harvest.derived is not on this branch")
from ragix_kernels.tender.domain.pyramid import aggregate, dce_node, dce_sentences, document_node, rollup, zoom
from ragix_kernels.tender.domain.records import Applicability

CHANNEL = "tender.dates_fr 1.0"
APP = Applicability(project="P", clause_scope={"pieces": "consultation", "lots": "all"})
NOW = "2026-09-11T10:00:00Z"


def claim(field, value, raw, doc, chunk, start=10, origin="observed", derivation=None, extra=()):
    span = SourceSpan(doc, chunk, ("0.1",), start, start + len(raw), sha256_text(raw))
    sources = (span,) + tuple(extra)
    vtype = "datetime" if "T" in value else "date"
    return ClaimRecord(field, ClaimValue(vtype, raw, value),
                       Provenance(origin, sources, derivation, CHANNEL), APP)


RC = claim("offer_deadline", "2027-10-15T12:00", "Jeudi 15 octobre 2027 a 12 heures 00", "docRC", "cRC")
CCAP = claim("offer_deadline", "2027-10-08T12:00", "Jeudi 8 octobre 2027 a 12 heures 00", "docCCAP", "cCCAP")
CLAUSE = "au plus tard 10 jours avant la date limite"
QUESTIONS = ClaimRecord(
    "questions_deadline", ClaimValue("datetime", CLAUSE, "2027-10-05T12:00"),
    Provenance("derived",
               (SourceSpan("docRC", "cRC", ("0.1",), 10, 10 + len("Jeudi 15 octobre 2027 a 12 heures 00"),
                           sha256_text("Jeudi 15 octobre 2027 a 12 heures 00")),
                SourceSpan("docRC", "cQ", ("0.2",), 50, 50 + len(CLAUSE), sha256_text(CLAUSE))),
               "offer_deadline - 10 days", CHANNEL),
    APP)
END = claim("visit_window_end", "2027-09-28", "28 septembre 2027", "docRC", "cV")


def test_aggregation_keeps_every_claim_by_reference():
    k = aggregate([RC, CCAP])
    assert k["fields"]["offer_deadline"] == {"2027-10-08T12:00": [CCAP.claim_id],
                                             "2027-10-15T12:00": [RC.claim_id]}
    assert sorted(k["claims"]) == sorted([RC.claim_id, CCAP.claim_id])


def build():
    children = {"n_rc": document_node("docRC", "RC", [RC, QUESTIONS, END]),
                "n_ccap": document_node("docCCAP", "CCAP", [CCAP])}
    resolution = {"governing": [RC.claim_id], "verdict": "supported_with_caveats",
                  "rendering": "PRET AVEC RESERVES"}
    dependents = {"questions_deadline": [{"claim_id": QUESTIONS.claim_id, "value": "2027-10-05T12:00"}],
                  "visit_window": {"end": "2027-09-28", "questions_deadline": "2027-10-05T12:00",
                                   "days_end_after_questions": -7}}
    authority = {"authorities": [{"label": "RC 6"}, {"label": "CCAP 2"}],
                 "ranks": {"RC": 1, "CCAP": 3}, "sufficient": True}
    return children, dce_node(children, resolution, dependents, authority)


def test_the_interpreted_layer_carries_no_critical_value():
    _, k = build()
    blob = repr(k["interpreted"])
    assert not re.search(r"\d{4}-\d{2}-\d{2}", blob), blob      # no date, no datetime
    assert not re.search(r"\d+[,.]\d{2}\s*(€|%)", blob), blob    # no amount, no percentage
    assert k["interpreted"]["conflict"] is True
    assert k["interpreted"]["conflicting_value_count"] == 2
    assert k["interpreted"]["governing_claims"] == [RC.claim_id]
    assert k["interpreted"]["displaced_claims"] == [CCAP.claim_id]


def test_a_derived_quantity_keeps_its_expression_and_operands():
    _, k = build()
    days = k["derived"]["days_visit_end_after_questions"]
    assert days["value"] == -7 and "visit_window_end" in days["expression"]
    assert days["operands"]["questions_deadline"] == [QUESTIONS.claim_id]


def test_every_summary_sentence_maps_to_children_and_claims():
    children, k = build()
    by_id = {c.claim_id: c for c in (RC, CCAP, QUESTIONS, END)}
    sentences = dce_sentences(k, by_id, sorted(children))
    assert sentences and all(s.children and s.claims for s in sentences)
    assert "2027-10-15T12:00" in sentences[0].text and "PRET AVEC RESERVES" not in sentences[0].text
    assert all(by_id[c] for s in sentences for c in s.claims)


def test_zoom_reaches_the_spans_and_rollup_walks_back():
    children, k = build()
    by_id = {c.claim_id: c for c in (RC, CCAP, QUESTIONS, END)}
    sentence = dce_sentences(k, by_id, sorted(children))[0]
    trace = zoom("n_dce", sentence, children, by_id)
    assert [s["level"] for s in trace["steps"]] == ["dce", "document", "document", "leaf"]
    spans = trace["steps"][-1]["spans"]
    assert spans and all(len(s["span"]["span_sha256"]) == 64 for s in spans)
    assert sha256_text(spans[0]["raw"]) == spans[0]["span"]["span_sha256"]
    assert rollup([RC.claim_id], children) == ["n_rc"]


def store(tmp_path):
    return DerivedStore(str(tmp_path / "derived.sqlite"), "saqqara.db", "a" * 64, "run-1")


@needs_derived
def test_knowledge_is_append_only_and_versioned(tmp_path):
    s = store(tmp_path)
    s.write_node(NodeRow("n_dce", "dce", None, None, ("n_rc",)), NOW)
    s.write_knowledge(KnowledgeRow("n_dce", "node", {"v": 1}), NOW)
    s.write_knowledge(KnowledgeRow("n_dce", "node", {"v": 2}, model="granite"), NOW)
    latest = s.latest_knowledge("n_dce")
    assert latest["revision"] == 2 and latest["model"] == "granite"
    assert s.counts() == {"nodes": 1, "knowledge": 2, "derived_edges": 0}
    s.close()


@needs_derived
def test_a_sentence_mapping_to_no_child_is_refused():
    with pytest.raises(ValueError):
        KnowledgeRow("n", "node", {}, summary="x", summary_map=({"sentence": 1, "children": []},))


@needs_derived
def test_staleness_follows_the_source_hash(tmp_path):
    s = store(tmp_path)
    s.write_knowledge(KnowledgeRow("cChunk", "chunk", {"k": 1}), NOW)
    assert s.stale("cChunk", "a" * 64) is False
    assert s.stale("cChunk", "b" * 64) is True
    assert s.stale("unknown", "a" * 64) is None
    s.close()


@needs_derived
def test_the_model_is_recorded_as_a_fact_of_the_row(tmp_path):
    s = store(tmp_path)
    s.write_knowledge(KnowledgeRow("n", "node", {}), NOW)
    assert s.latest_knowledge("n")["model"] == DETERMINISTIC
    s.close()


def test_a_derived_claims_raw_text_belongs_to_a_later_source():
    """The bug block A's own gate caught: a derived claim's raw is the clause, not its first operand's span."""
    children, k = build()
    by_id = {c.claim_id: c for c in (RC, CCAP, QUESTIONS, END)}
    sentences = dce_sentences(k, by_id, sorted(children))
    questions = next(s for s in sentences if QUESTIONS.claim_id in s.claims)
    entry = next(e for e in zoom("n_dce", questions, children, by_id)["steps"][-1]["spans"]
                 if e["claim_id"] == QUESTIONS.claim_id)
    assert len(entry["sources"]) == 2
    assert sha256_text(entry["raw"]) != entry["span"]["span_sha256"]          # not the first source
    assert sha256_text(entry["raw"]) in {s["span_sha256"] for s in entry["sources"]}
