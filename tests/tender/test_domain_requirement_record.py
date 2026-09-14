"""RequirementRecord (v0-draft) + coarse vocabulary — constructor contracts.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

GREEN work of mission M-2: the object and its guards. The FREEZE of the schema
and of the vocabulary is the lead's signature (gates GR3-0a/b) — these tests
check that the constructors enforce what the draft says, not that the draft is
final.
"""

from __future__ import annotations

import pytest

from ragix_kernels.tender.domain import vocabulary
from ragix_kernels.tender.domain.records import Applicability
from ragix_kernels.tender.domain.requirements import (MODALITIES, RequirementRecord,
                                 SpanNotVerbatim, span_verbatim)

APP = Applicability(project="PROJECT-A")
PAYLOAD = ("Question: Existe-t-il une politique de scan des équipements ?\n"
           "Section: Gestion des vulnérabilités")
SPAN = "politique de scan des équipements"


def coarse(**kw):
    base = dict(scale="coarse", evidence_span=SPAN, source_kind="question",
                source_id="q_abc123", source_path="PROJECT-A/x.xlsx",
                applicability=APP, channel="vocab-tagger", theme="unknown")
    base.update(kw)
    return RequirementRecord(**base)


def fine(**kw):
    base = dict(scale="fine", evidence_span=SPAN, source_kind="question",
                source_id="q_abc123", source_path="PROJECT-A/x.xlsx",
                applicability=APP, channel="llm-schema",
                predicate="le titulaire scanne les équipements",
                modality="must", parent_rid="r_0123456789abcdef")
    base.update(kw)
    return RequirementRecord(**base)


# ── vocabulary ───────────────────────────────────────────────────────────────

def test_vocabulary_loads_and_is_closed():
    ids = vocabulary.axis_ids()
    assert len(ids) >= 50                      # the mined layer-A core
    assert all(i.startswith("AX-") for i in ids)
    assert vocabulary.version()                # stamped on every record
    assert "unknown" not in ids                # abstention is not an axis
    assert vocabulary.is_valid_theme("unknown")


def test_vocabulary_resolves_aliases_and_spelling_variants():
    # canonical, accent variant and plural must land on the same axis
    axis = vocabulary.resolve("pénalités")
    assert axis and axis == vocabulary.resolve("penalites")
    assert vocabulary.resolve("Propriété Intellectuelle") == \
        vocabulary.resolve("droits de propriété intellectuelle")
    assert vocabulary.resolve("assurances") == vocabulary.resolve("assurance")


def test_vocabulary_abstains_outside_the_list():
    assert vocabulary.resolve("modalités de livraison des poneys") is None
    assert not vocabulary.is_valid_theme("AX-not-a-real-axis")


def test_prompt_names_can_drop_furniture():
    everything = vocabulary.prompt_names()
    no_furniture = vocabulary.prompt_names(requirement_only=True)
    assert "sommaire" in everything and "sommaire" not in no_furniture
    assert len(no_furniture) < len(everything)


# ── coarse scale: cannot be split by construction ────────────────────────────

def test_coarse_record_is_valid_and_stamps_the_vocabulary_version():
    r = coarse(theme=vocabulary.resolve("pénalités"))
    assert r.rid.startswith("r_")
    assert r.kernel_versions["axes_coarse"] == vocabulary.version()
    assert r.kernel_versions["requirement_schema"] == r.schema_version
    assert r.to_dict()["theme"] == r.theme


def test_coarse_refuses_a_predicate_or_a_parent():
    with pytest.raises(ValueError, match="cannot be split"):
        coarse(predicate="quelque chose")
    with pytest.raises(ValueError, match="parent_rid"):
        coarse(parent_rid="r_deadbeefdeadbeef")


def test_coarse_requires_an_explicit_theme_even_to_abstain():
    with pytest.raises(ValueError, match="without theme"):
        coarse(theme="")
    assert coarse(theme="unknown").theme == "unknown"


def test_theme_outside_the_closed_vocabulary_is_refused():
    with pytest.raises(ValueError, match="outside the closed"):
        coarse(theme="AX-invented-by-the-model")


# ── fine scale: predicate + mandatory parent ─────────────────────────────────

def test_fine_record_is_valid():
    r = fine()
    assert r.parent_rid and r.predicate and r.modality == "must"


def test_fine_requires_predicate_and_parent():
    with pytest.raises(ValueError, match="without predicate"):
        fine(predicate="   ")
    with pytest.raises(ValueError, match="without parent_rid"):
        fine(parent_rid=None)


# ── closed vocabularies and provenance (rule 2) ──────────────────────────────

@pytest.mark.parametrize("kw", [
    {"scale": "medium"}, {"modality": "maybe"},
    {"channel": "twin-diff-chains"},           # a QuestionRecord channel
    {"source_kind": "rumour"},
])
def test_closed_vocabularies_are_enforced(kw):
    with pytest.raises(ValueError, match="outside vocabulary"):
        coarse(**kw)


@pytest.mark.parametrize("kw,msg", [
    ({"source_id": ""}, "source_id"),
    ({"source_path": ""}, "source_path"),
    ({"applicability": {"project": "X"}}, "Applicability"),
    ({"evidence_span": "  "}, "evidence span"),
])
def test_provenance_and_span_are_mandatory(kw, msg):
    with pytest.raises(ValueError, match=msg):
        coarse(**kw)


def test_all_modalities_are_constructible():
    for m in MODALITIES:
        assert fine(modality=m).modality == m


# ── rule 9 by construction: the verbatim span guard ──────────────────────────

def test_span_verbatim_tolerates_whitespace_and_case_only():
    assert span_verbatim("Politique   de\nscan des ÉQUIPEMENTS".replace(
        "ÉQUIPEMENTS", "équipements"), PAYLOAD)
    assert not span_verbatim("politique de scan des serveurs", PAYLOAD)
    assert not span_verbatim("", PAYLOAD)


def test_from_extraction_accepts_a_verbatim_span_and_traces_it():
    r = RequirementRecord.from_extraction(
        payload=PAYLOAD, scale="coarse", evidence_span=SPAN,
        source_kind="question", source_id="q_abc123",
        source_path="PROJECT-A/x.xlsx", applicability=APP,
        channel="llm-schema", theme="unknown")
    assert r.trace["span_guard"] == "verified"
    assert len(r.trace["payload_sha256"]) == 16


def test_from_extraction_refuses_a_reformulated_span():
    with pytest.raises(SpanNotVerbatim):
        RequirementRecord.from_extraction(
            payload=PAYLOAD, scale="coarse",
            evidence_span="une politique de scan est-elle en place",
            source_kind="question", source_id="q_abc123",
            source_path="PROJECT-A/x.xlsx", applicability=APP,
            channel="llm-schema", theme="unknown")


# ── identity ─────────────────────────────────────────────────────────────────

def test_rid_is_derived_and_discriminates_scale_span_and_predicate():
    a, b = coarse(), coarse()
    assert a.rid == b.rid
    assert coarse(evidence_span="Gestion des vulnérabilités").rid != a.rid
    assert fine().rid != a.rid
    assert fine(predicate="le titulaire documente la fréquence").rid != fine().rid
