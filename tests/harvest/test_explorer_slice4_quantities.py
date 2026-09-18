"""E11 independently specified exact-span fixtures, before reader changes.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import json
from pathlib import Path
import pytest
from ragix_kernels.harvest.quantitative import harvest, roots

CASES = json.loads(
    Path(__file__).with_name("fixtures_explorer_slice4_quantities.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("token_locale", [False, True])
def test_e11_exact_span(case, token_locale):
    candidates = harvest(
        case["text"],
        source_id="synthetic-e11",
        node_id=case["id"],
        classification="CONTENT",
        token_locale=token_locale,
    )
    found = [
        c
        for c in roots(candidates)
        if (c.start, c.end, c.raw, c.kind)
        == (case["start"], case["end"], case["raw"], case["kind"])
    ]
    assert len(found) == 1, [(c.raw, c.kind, c.start, c.end) for c in roots(candidates)]
    for candidate in candidates:
        assert case["text"][candidate.start : candidate.end] == candidate.raw
    if "NOMINAL_ABSENT" in case.get("note", ""):
        assert found[0].nominal is None and "NOMINAL_ABSENT" in found[0].flags
    if "GLUED" in case.get("note", ""):
        assert "GLUED" in found[0].flags
    if "UNIT_SHARED" in case.get("note", ""):
        assert "UNIT_SHARED" in found[0].flags


@pytest.mark.parametrize(
    "text",
    [
        "± 3 nm",
        "3 h 17 min",
        "115 V AC +/- 5 %",
        "+29°C+/-3°C",
        "entre 15 et 25 °C",
        "twice per week",
    ],
)
def test_e11_children_and_literal_provenance(text):
    found = harvest(
        text,
        source_id="synthetic",
        node_id="node",
        classification="UNKNOWN",
        uncertainty=("DIGIT_JOIN",),
    )
    ids = {c.candidate_id for c in found}
    for c in found:
        assert c.raw == text[c.start : c.end]
        assert "DIGIT_JOIN" in c.flags and "CLASSIFICATION_UNKNOWN" in c.flags
        assert all(m.candidate_id in ids for m in c.members)
        if c.unit_start is not None:
            assert text[c.unit_start : c.unit_end] == c.unit_raw
        assert c.needs_review


def test_e11_compound_duration_keeps_units_without_conversion():
    found = harvest(
        "3 h 17 min 4 s", source_id="synthetic", node_id="node", classification="CONTENT"
    )
    (root,) = roots(found)
    by_id = {c.candidate_id: c for c in found}
    assert root.kind == "duration" and root.number is None
    assert [(by_id[m.candidate_id].number, by_id[m.candidate_id].unit) for m in root.members] == [
        ("3", "h"),
        ("17", "min"),
        ("4", "s"),
    ]


def test_e11_missing_nominal_and_unresolved_locale_never_become_ready():
    for text in ("± 3 nm", "3 h 1.234 min"):
        found = harvest(text, source_id="synthetic", node_id="node", classification="CONTENT")
        (root,) = roots(found)
        assert root.needs_review
        if text.startswith("±"):
            assert root.nominal is None
        else:
            assert root.normalization_status == "unparsed"


@pytest.mark.parametrize("text", ["110 V - 50 Hz", "110 V – 50 Hz", "110 V — 50 Hz"])
def test_e11_different_units_are_not_a_dash_interval(text):
    found = roots(harvest(text, source_id="synthetic", node_id="node", classification="CONTENT"))
    assert [(c.number, c.unit) for c in found] == [("110", "V"), ("50", "Hz")]
    assert all(c.kind == "scalar" for c in found)
