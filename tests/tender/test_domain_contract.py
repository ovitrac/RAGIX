"""The producer contract — rules 8 and 9 made into a shape.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

`tests/test_contract.py` remains the freeze of the orchestrator contract that
stays with the quarantined package. This file is the freeze of the one written
fresh for the lab (R4), and it asserts the two rules rather than the shape:

  8. the composing model sees the selected evidence and nothing else;
  9. a citation resolves inside the payload it was produced from — or the pair
     is refused, never quietly mended.

Every fixture here is built by code, and the rule-8 tests carry text that must
be ABSENT: a test that only checks what is present would pass against a payload
that also carried the whole corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ragix_kernels.saqqara.store.records import ChunkRecord          # noqa: E402

from ragix_kernels.tender.domain.contract import (                                        # noqa: E402
    ABSTAINING,
    ASSERTING,
    HUMAN_RENDERING,
    VERDICTS,
    Citation,
    ComposedAnswer,
    ContractViolation,
    EvidenceItem,
    ProducerPayload,
    payload_from_retrieval,
    verify,
)

SELECTED = "Adservio applique une politique de revue par les pairs sur chaque livrable."
PARENT = "SECTION CONFIDENTIELLE QUE LE MODELE NE DOIT PAS VOIR."
RELATED = "AUTRE TEXTE HORS SELECTION."


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(chunk_id=chunk_id, doc_id="doc-1", seq=0, text=text,
                       level=1, node_ids=[f"n-{chunk_id}"],
                       section_path=["Qualite"], pages=[2])


class _Hit:
    def __init__(self, chunk: ChunkRecord) -> None:
        self.chunk = chunk


class _Result:
    """A retrieval result whose non-selected lanes are deliberately non-empty."""

    def __init__(self) -> None:
        self.selected = [_Hit(_chunk("c-1", SELECTED))]
        self.parents = [_chunk("c-parent", PARENT)]
        self.related = [_chunk("c-related", RELATED)]
        self.objects = []
        self.trace = {}


def _payload() -> ProducerPayload:
    return payload_from_retrieval("Comment la qualite est-elle assuree ?", _Result())


# ── rule 8: only the selected evidence ───────────────────────────────────────

def test_the_model_sees_the_selected_evidence_and_nothing_else():
    """The absent halves are the assertions. `parents` and `related` are real
    text on the result object, and a payload builder that swept them in would
    pass a test that only looked for what should be there."""
    payload = _payload()
    wire = json.dumps(payload.for_model(), ensure_ascii=False)

    assert SELECTED in wire
    assert PARENT not in wire, "a parent chunk reached the model (rule 8)"
    assert RELATED not in wire, "a related chunk reached the model (rule 8)"


def test_the_payload_carries_values_and_no_handle_back_to_the_corpus():
    """Rule 8 as a property of the object: what the model gets is serialisable,
    which a store, a tree or a retriever would not be."""
    payload = _payload()

    json.dumps(payload.for_model())                     # raises if a handle got in
    assert set(payload.__dataclass_fields__) == {"question", "evidence"}


def test_the_evidence_the_model_sees_carries_no_provenance_it_could_leak():
    """The item's own fields keep the provenance for the citation to resolve
    against; what crosses to the model is the handle and the text."""
    item = _payload().evidence[0]

    assert set(item.for_model()) == {"evidence_id", "text", "section_path"}
    assert item.doc_id and item.chunk_id and item.node_ids


# ── rule 9: a citation resolves inside its own payload ───────────────────────

def test_a_citation_to_evidence_outside_the_payload_is_refused():
    payload = _payload()
    answer = ComposedAnswer(text="Oui.", verdict="supported",
                            citations=(Citation(evidence_id="e_nowhere",
                                                quote=SELECTED),))

    with pytest.raises(ContractViolation, match="not in the producer payload"):
        verify(answer, payload)


def test_a_quote_absent_from_the_evidence_it_names_is_refused():
    """The pointer resolves and the span does not — a paraphrase presented as a
    quotation, which is the failure rule 9 is written against."""
    payload = _payload()
    answer = ComposedAnswer(
        text="Oui.", verdict="supported",
        citations=(Citation(evidence_id=payload.evidence[0].evidence_id,
                            quote="Adservio fait relire ses livrables."),))

    with pytest.raises(ContractViolation, match="the span does not"):
        verify(answer, payload)


def test_a_reflowed_quote_is_accepted_because_the_rule_says_so():
    """The control on the comparison rule itself. Line breaks are collapsed by
    declaration; if this failed, the contract would be stricter than it states
    and honest citations would be refused."""
    payload = _payload()
    reflowed = SELECTED.replace(" ", "\n  ", 1)
    answer = ComposedAnswer(
        text="Oui.", verdict="supported",
        citations=(Citation(evidence_id=payload.evidence[0].evidence_id,
                            quote=reflowed),))

    verify(answer, payload)          # raises if the normalisation is not applied


def test_an_answer_claiming_support_must_name_what_supports_it():
    payload = _payload()
    answer = ComposedAnswer(text="Oui, tout est conforme.", verdict="supported")

    with pytest.raises(ContractViolation, match="no citation"):
        verify(answer, payload)


def test_every_violation_is_reported_not_only_the_first():
    """A caller routing to human review needs the whole account, not the first
    thing that went wrong."""
    payload = _payload()
    answer = ComposedAnswer(
        text="Oui.", verdict="supported",
        citations=(Citation(evidence_id="e_nowhere", quote="x"),
                   Citation(evidence_id=payload.evidence[0].evidence_id,
                            quote="texte qui n'existe pas")))

    with pytest.raises(ContractViolation) as raised:
        verify(answer, payload)

    message = str(raised.value)
    assert "not in the producer payload" in message
    assert "the span does not" in message


def test_a_well_formed_answer_passes():
    """Without this, every test above would also pass against a `verify` that
    raised on everything."""
    payload = _payload()
    answer = ComposedAnswer(
        text="Oui, revue par les pairs.", verdict="supported_with_caveats",
        citations=(Citation(evidence_id=payload.evidence[0].evidence_id,
                            quote="revue par les pairs"),))

    verify(answer, payload)


# ── the states, and the vocabulary ───────────────────────────────────────────

def test_an_abstention_carrying_citations_is_refused_at_construction():
    """An abstention is a state of the evidence, not an answer with sources."""
    with pytest.raises(ValueError, match="abstention"):
        ComposedAnswer(text="", verdict="abstain_no_evidence",
                       citations=(Citation(evidence_id="e_1", quote="x"),))


def test_an_abstention_verifies_and_is_a_successful_outcome():
    verify(ComposedAnswer(text="", verdict="abstain_no_evidence"), _payload())
    verify(ComposedAnswer(text="", verdict="abstain_conflict"), _payload())


def test_the_verdicts_map_one_to_one_onto_the_human_rendering():
    """§5.2 and §11: one internal model, one external rendering, no third
    taxonomy and no two verdicts sharing a label."""
    assert set(HUMAN_RENDERING) == set(VERDICTS)
    assert len(set(HUMAN_RENDERING.values())) == len(VERDICTS)
    assert set(ASSERTING) | set(ABSTAINING) < set(VERDICTS)
    assert ComposedAnswer(text="", verdict="needs_review").rendering == "À VÉRIFIER"


def test_a_verdict_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="outside"):
        ComposedAnswer(text="x", verdict="probably_fine")


# ── provenance on the evidence itself ────────────────────────────────────────

def test_evidence_without_provenance_is_refused():
    with pytest.raises(ValueError, match="provenance"):
        EvidenceItem(text="t", doc_id="", chunk_id="c", node_ids=("n",))
    with pytest.raises(ValueError, match="naming no node"):
        EvidenceItem(text="t", doc_id="d", chunk_id="c", node_ids=())


def test_the_same_chunk_yields_the_same_evidence_id_in_every_payload():
    """A citation read back from a trace months later still points somewhere."""
    first = EvidenceItem.of_chunk(_chunk("c-1", SELECTED))
    second = EvidenceItem.of_chunk(_chunk("c-1", SELECTED))
    other = EvidenceItem.of_chunk(_chunk("c-2", SELECTED))

    assert first.evidence_id == second.evidence_id
    assert first.evidence_id != other.evidence_id


def test_a_citation_without_a_quote_is_refused():
    with pytest.raises(ValueError, match="without a quote"):
        Citation(evidence_id="e_1", quote="  ")
