"""Unit provenance falsifiers with independent synthetic cell values.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace, asdict
import pytest
from ragix_kernels.harvest.table_context import Cell, CellContext
from ragix_kernels.harvest.quantitative import harvest, roots


def context(text="23", header="Travel [mm]", label="Axis A"):
    def c(id, r, col, t):
        return Cell(
            "synthetic", "t", id, 1, r, col, t, (col * 80, r * 20, col * 80 + 75, r * 20 + 15)
        )

    return CellContext(
        c("v", 1, 1, text),
        (c("h", 0, 1, header),) if header else (),
        (c("l", 1, 0, label),) if label else (),
    )


def read(c, **kwargs):
    return harvest(
        c.value.text,
        source_id=c.value.source_id,
        node_id=c.value.cell_id,
        classification="CONTENT",
        context=c,
        token_locale=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "header,unit",
    [
        ("Temperature [°C]", "degC"),
        ("Temperature [° C]", "degC"),
        ("Supply V", "V"),
        ("V", "V"),
        ("Travel (mm)", "mm"),
    ],
)
def test_exact_inherited_span(header, unit):
    c = context(header=header)
    (q,) = read(c)
    assert q.unit == unit and q.unit_source == "INHERITED" and q.needs_review
    assert q.raw == "23" and q.unit_start is None and q.unit_end is None
    (proof,) = q.unit_evidence
    assert c.column_headers[0].text[proof.span.start : proof.span.end] == proof.span.raw
    assert proof.span.cell_id == "h"


def test_row_label_and_multirow_header_units():
    c = context(header="", label="Travel [mm]")
    (q,) = read(c)
    assert q.unit == "mm" and q.unit_evidence[0].association == "row_label"
    c = context(header="Travel")
    h = replace(c.column_headers[0], cell_id="h2", text="[mm]")
    (q,) = read(replace(c, column_headers=c.column_headers + (h,)))
    assert q.unit == "mm" and q.unit_evidence[0].span.cell_id == "h2"


def test_conflicting_context_never_votes_or_cross_inherits():
    (q,) = read(context(header="[mm]", label="Axis [V]"))
    assert q.unit is None and q.unit_source == "NONE"
    assert q.unit_reason == "AMBIGUOUS_ASSOCIATED_UNITS" and q.number == "23"
    assert {m.unit for m in q.unit_evidence} == {"mm", "V"}


def test_inline_unit_remains_the_observation():
    (q,) = read(context(text="23 V", header="[mm]"))
    assert q.unit == "V" and q.unit_source == "INLINE"
    assert "UNIT_CONTEXT_CONFLICT" in q.flags
    assert q.unit_evidence[0].span.cell_id == "v"


def test_no_context_unit_or_uncertain_header_keeps_reason():
    (q,) = read(context(header="Travel", label="Axis A"))
    assert q.unit is None and q.unit_reason == "NO_UNIT_IN_ASSOCIATED_CELLS"
    c = context()
    c = replace(c, column_headers=(replace(c.column_headers[0], flags=("OCR",)),))
    (q,) = read(c)
    assert q.unit is None and q.unit_reason == "UNCERTAIN_CONTEXT"


@pytest.mark.parametrize("text", ["13 à 29", "13 to 29", "23 ± 3"])
def test_unitless_composite_keeps_original_text_and_child_ids(text):
    candidates = read(context(text=text))
    (q,) = roots(candidates)
    assert q.raw == text and q.unit == "mm"
    assert all(m.candidate_id in {c.candidate_id for c in candidates} for m in q.members)
    assert all(text[c.start : c.end] == c.raw for c in candidates)


def test_a_neighboring_numeric_value_is_not_a_unit_declaration():
    (q,) = read(context(header="", label="17 V"))
    assert q.unit is None


def test_stale_cross_node_context_rejected():
    with pytest.raises(ValueError):
        harvest(
            "29", source_id="synthetic", node_id="v", classification="CONTENT", context=context()
        )


def test_no_context_preserves_existing_candidate_contract():
    (q,) = harvest("23 mm", source_id="synthetic", node_id="v", classification="CONTENT")
    assert "unit_source" not in asdict(q)


def test_numeric_neighbor_with_bracket_unit_is_not_a_label_declaration():
    (q,) = read(context(header="", label="17 [V]"))
    assert q.unit is None


def test_context_does_not_assign_one_unit_to_a_mixed_duration():
    candidates = read(context(text="1 h 17 min", header="[V]"))
    (q,) = roots(candidates)
    assert q.kind == "duration" and q.unit is None
    ids = {c.candidate_id for c in candidates}
    assert all(m.candidate_id in ids for c in candidates for m in c.members)


def test_explicit_composite_proof_and_member_graph_remain_traceable():
    candidates = read(context(text="13 mm to 29 mm"))
    (q,) = roots(candidates)
    assert q.unit == "mm" and q.unit_evidence
    assert all(m.candidate_id in {c.candidate_id for c in candidates} for m in q.members)
    for proof in q.unit_evidence:
        assert (
            context(text="13 mm to 29 mm").value.text[proof.span.start : proof.span.end]
            == proof.span.raw
        )


def test_contaminated_value_is_never_promoted():
    c = context(text="23 mm")
    c = replace(c, value=replace(c.value, flags=("LIFECYCLE_GLYPH_SUSPECTED",)))
    (q,) = read(c)
    assert "LIFECYCLE_GLYPH_SUSPECTED" in q.flags and q.needs_review


@pytest.mark.parametrize("text", ["23 furlong", "23 €", "23 widgets"])
def test_unknown_inline_unit_cannot_be_overwritten_by_header(text):
    (q,) = read(context(text=text))
    assert q.unit is None and q.unit_reason == "UNSUPPORTED_VALUE_SUFFIX"


def test_legacy_locale_context_preserves_bare_comparator():
    c = context(text="<= 23")
    (q,) = harvest(
        c.value.text, source_id="synthetic", node_id="v", classification="CONTENT", context=c
    )
    assert q.kind == "inequality" and q.comparator_normalized == "<=" and q.raw == "<= 23"
