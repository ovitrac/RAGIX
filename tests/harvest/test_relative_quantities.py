"""Relative shapes are observations, never reference-parameter assignments.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import pytest
from .test_header_units import context, read
from ragix_kernels.harvest.quantitative import roots
from ragix_kernels.harvest.relative_quantitative import RelativeQuantityCandidate


@pytest.mark.parametrize(
    "text,numbers",
    [
        ("origin - 13 au origin - 31", ("-13", "-31")),
        ("datum +7 to datum −11", ("7", "-11")),
        ("référence locale -17 à référence locale +23", ("-17", "23")),
        ("(datum -13 au datum -31)", ("-13", "-31")),
        ("datum -13", ("-13",)),
    ],
)
def test_exact_anchor_and_signed_offset_spans(text, numbers):
    c = context(text=text)
    (q,) = read(c)
    assert isinstance(q, RelativeQuantityCandidate) and q.quantitative and q.needs_review
    assert tuple(o.number for o in q.offsets) == numbers
    assert q.unit == "mm" and q.unit_source == "INHERITED"
    assert q.number is None and q.lower is None and q.upper is None and q.nominal is None
    for offset in q.offsets:
        for span in (offset.span, offset.anchor):
            assert c.value.text[span.start : span.end] == span.raw and span.cell_id == "v"
    assert (
        not {"parameter", "reference_parameter", "applicability", "absolute_range"}
        & asdict(q).keys()
    )
    if len(numbers) == 2:
        assert float(q.offset_lower) == min(map(float, numbers))
        assert float(q.offset_upper) == max(map(float, numbers))


@pytest.mark.parametrize(
    "text", ["45 - 13", "origin — 13", "origin – 13", "RX-13", "well-known 13"]
)
def test_subtraction_dashes_and_identifiers_are_not_anchored_offsets(text):
    assert not any(isinstance(q, RelativeQuantityCandidate) for q in read(context(text=text)))


def test_header_anchor_is_exact_and_never_assumed_from_an_ordinary_label():
    c = context(text="-13 to -31", header="Relative to datum [mm]")
    (q,) = read(c)
    assert q.anchor_source == "CONTEXT"
    assert all(o.anchor.cell_id == "h" and o.anchor.raw == "datum" for o in q.offsets)
    assert c.column_headers[0].text[q.offsets[0].anchor.start : q.offsets[0].anchor.end] == "datum"
    assert not any(
        isinstance(q, RelativeQuantityCandidate) for q in read(context(text="-31 to -13"))
    )


def test_conflicting_anchors_do_not_form_one_offset_range():
    (q,) = read(context(text="datum -13 to origin -31"))
    assert "RELATIVE_ANCHOR_MISMATCH" in q.flags
    assert q.offset_lower is None and q.offset_upper is None


def test_plus_minus_is_preserved_as_branching_not_an_arbitrary_sign():
    (q,) = read(context(text="datum ±13 au datum ±31"))
    assert [o.operator for o in q.offsets] == ["±", "±"]
    assert all(o.number is None for o in q.offsets)
    assert q.offset_lower is None and "RELATIVE_BRANCHING" in q.flags


def test_inline_relative_units_and_conflicts_are_preserved():
    (q,) = read(context(text="datum -13 V to datum -31 V", header="[V]"))
    assert q.unit == "V" and q.unit_source == "INLINE"
    assert all(m.span.cell_id == "v" for m in q.unit_evidence)
    (q,) = read(context(text="datum -13 V to datum -31 mm"))
    assert q.unit is None and q.unit_reason == "RELATIVE_UNITS_CONFLICT"
    assert q.offset_lower is None


def test_relative_without_unit_keeps_shape_and_reason():
    (q,) = read(context(text="datum -13 to datum -31", header="", label=""))
    assert q.unit is None and q.unit_reason == "NO_UNIT_IN_ASSOCIATED_CELLS"
    assert [o.number for o in q.offsets] == ["-13", "-31"]


def test_ambiguous_header_anchor_retains_both_source_alternatives():
    c = context(text="-13 to -31", header="relative to origin [mm]", label="relative to datum [mm]")
    (q,) = read(c)
    assert "ANCHOR_CONTEXT_AMBIGUOUS" in q.flags
    assert {a.cell_id for a in q.anchor_candidates} == {"h", "l"}
    assert all(o.anchor is None for o in q.offsets)
    assert q.offset_lower is None


@pytest.mark.parametrize("raw,number", [("13,25", "-13.25"), ("0.125", "-0.125"), ("1.234", None)])
def test_relative_offsets_keep_locale_uncertainty(raw, number):
    (q,) = read(context(text=f"datum -{raw}"))
    assert q.offsets[0].number == number
    if number is None:
        assert "SEPARATOR_AMBIGUOUS" in q.flags


def test_shared_inline_unit_keeps_its_actual_source_span():
    c = context(text="datum -13 to datum -31 mm", header="")
    (q,) = read(c)
    assert q.unit == "mm" and q.unit_source == "INHERITED"
    assert q.unit_evidence[0].span.cell_id == "v"
    assert c.value.text[q.unit_evidence[0].span.start : q.unit_evidence[0].span.end] == "mm"


def test_furniture_never_produces_relative_candidates():
    from ragix_kernels.harvest.quantitative import harvest

    c = context(text="datum -13 to datum -31")
    assert (
        harvest(
            c.value.text, source_id="synthetic", node_id="v", classification="FURNITURE", context=c
        )
        == ()
    )


def test_conflicting_unit_context_does_not_produce_ordered_offset_bounds():
    (q,) = read(context(text="datum -13 V to datum -31 V", header="[mm]"))
    assert q.unit == "V" and "UNIT_CONTEXT_CONFLICT" in q.flags
    assert q.offset_lower is None and q.offset_upper is None


def test_unitless_relative_offsets_are_not_dimensionally_ordered():
    (q,) = read(context(text="datum -13 to datum -31", header="", label=""))
    assert q.offset_lower is None and q.offset_upper is None


@pytest.mark.parametrize("text", ["datum -13 to datum -31 approximately", "datum -13 widgets"])
def test_partial_relative_match_cannot_become_absolute_scalars(text):
    (q,) = read(context(text=text))
    assert q.kind == "relative_unparsed" and q.normalization_status == "unparsed"
    assert q.number is None and q.offsets == ()
    assert q.unit_reason == "UNSUPPORTED_RELATIVE_SHAPE" and q.needs_review


def test_unsigned_value_under_explicit_relative_header_is_unparsed():
    (q,) = read(context(text="13 to 31", header="Relative to datum [mm]"))
    assert q.kind == "relative_unparsed" and q.number is None


def test_relative_header_does_not_turn_date_into_an_offset():
    qs = read(context(text="2033-07-19", header="Relative to datum [mm]"))
    assert not any(isinstance(q, RelativeQuantityCandidate) for q in qs)
