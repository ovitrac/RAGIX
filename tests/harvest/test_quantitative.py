"""Synthetic literal, composite and routing falsifiers.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

from dataclasses import FrozenInstanceError
import pytest

from ragix_kernels.harvest.quantitative import harvest, roots, route


def read(text, **kwargs):
    return harvest(text, source_id="synthetic-copy", node_id="node-a", classification="CONTENT", **kwargs)


@pytest.mark.parametrize("text,number,unit", [
    ("-7 °C", "-7", "degC"), ("0.2 %", "0.2", "%"), ("0,0007 %", "0.0007", "%"),
    ("+12 V", "12", "V"), ("1 234,56 W", "1234.56", "W"), ("0.1234 mm", "0.1234", "mm"),
])
def test_literal_exactness(text, number, unit):
    candidate, = read(text)
    assert (candidate.number, candidate.unit) == (number, unit)
    assert text[candidate.start:candidate.end] == candidate.raw
    assert text[candidate.unit_start:candidate.unit_end] == candidate.unit_raw


@pytest.mark.parametrize("text,expected", [
    ("8 °C à 19 °C", ("8", "19", True, True)),
    ("8 °C < T <= 19 °C", ("8", "19", False, True)),
    ("19 °C >= T > 8 °C", ("8", "19", False, True)),
    ("8 à 19 °C", ("8", "19", True, True)),
    ("8..19 °C", ("8", "19", True, True)),
])
def test_composites(text, expected):
    candidates = read(text)
    composite, = roots(candidates)
    assert composite.kind == "interval"
    assert (composite.lower, composite.upper, composite.lower_inclusive, composite.upper_inclusive) == expected
    assert len(composite.members) == 2 and len(candidates) == 3
    assert composite.normalization_status == "parsed"


@pytest.mark.parametrize("text", ["11 °C ± 2 °C", "11 ± 2 °C", "11 °C +/- 2 %", "11 V +/-2%", "11+/-2°C"])
def test_tolerance(text):
    composite, = roots(read(text))
    assert (composite.kind, composite.nominal, composite.tolerance) == ("tolerance", "11", "2")


@pytest.mark.parametrize("text", ["19 °C à 8 °C", "8 °C à 19 V", "11 °C ± -2 °C", "8 °C < T > 19 °C"])
def test_invalid_composite_retained_unresolved(text):
    composite, = roots(read(text))
    assert composite.normalization_status == "unparsed"
    assert "COMPOSITE_UNRESOLVED" in composite.flags


def test_direction_not_invented():
    candidate, = read("allant jusqu'à 19 °C")
    assert candidate.comparator_raw == "allant jusqu'à"
    assert candidate.comparator_normalized is None and candidate.direction_status == "unresolved"


@pytest.mark.parametrize("text,kind", [
    ("4 cycles par jour", "rate"), ("4 alarms per day", "rate"), ("<= ambient", "symbolic_bound"),
    ("3 heures", "duration"), ("6 essais", "cardinality"),
])
def test_additional_kinds(text, kind):
    candidate, = roots(read(text))
    assert candidate.kind == kind


@pytest.mark.parametrize("text", ["ABC-SPEC-029", "RX-12V", "version 2.7", "edition R", "p/n XY29", "15/04/2025"])
def test_nonquantitative_routing(text):
    candidates = read(text)
    assert candidates and not any(c.quantitative for c in candidates)
    assert route(candidates, semantic_cue=False) == "S-/Q-"
    assert route(candidates, semantic_cue=True) == "S+/Q-"


def test_classification_and_replay():
    args = dict(source_id="copy", node_id="node", classification="UNKNOWN")
    a = harvest("8 °C", **args)
    assert a == harvest("8 °C", **args)
    assert "CLASSIFICATION_UNKNOWN" in a[0].flags
    assert a != harvest("8 °C", **{**args, "source_id": "other"})
    assert harvest("8 °C", **{**args, "classification": "FURNITURE"}) == ()
    with pytest.raises(ValueError):
        harvest("8 °C", **{**args, "classification": "UNCLASSIFIED"})
    with pytest.raises(FrozenInstanceError):
        a[0].number = "9"


@pytest.mark.parametrize("text", ["8 à 19 °C", "8..19 °C", "11 ± 2 °C", "Test 3 à 37 °C"])
def test_an_inherited_unit_is_flagged_not_asserted(text):
    candidates = read(text)
    composite, = roots(candidates)
    first = next(c for c in candidates if c.candidate_id == composite.members[0].candidate_id)
    assert first.unit_start >= first.end        # its unit is not inside its own span
    assert "UNIT_INHERITED" in first.flags and "UNIT_INHERITED" in composite.flags


@pytest.mark.parametrize("text", ["8 °C à 19 °C", "11 °C ± 2 °C"])
def test_a_written_unit_is_not_flagged_inherited(text):
    assert not any("UNIT_INHERITED" in c.flags for c in read(text))


@pytest.mark.parametrize("text", ["10 K€", "2 à 5 K€", "3 m$"])
def test_money_is_not_a_physical_quantity(text):
    assert not any(c.quantitative for c in read(text))


def test_ambiguous_separator_is_not_normalized():
    candidate, = read("1.234 V")
    assert candidate.number is None and candidate.normalization_status == "unparsed"
    assert read("1.234 V", decimal_separator=".")[0].number == "1.234"
