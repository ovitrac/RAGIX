"""Generated geometry exercises cell boundaries without document fixtures.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

from dataclasses import replace
import pytest
from ragix_kernels.saqqara.field_views import TextSpan, VerticalRule, assemble, line_views, split_at_rules


def span(text="AB", x=10, ident="s1", **kw):
    boxes = tuple((x + i * 5, 10, x + (i + 1) * 5, 20) for i in range(len(text)))
    return TextSpan("source", ident, 1, text, (x, 10, x + len(text) * 5, 20),
                    boxes, (x, 18), font_size=10, state="CONTENT", **kw)


def test_join_and_inserted_separator_keep_character_provenance():
    result, = line_views([span(), span("CD", 23, "s2")])
    assert result.text == "AB CD"
    assert result.mapping[2] is None
    assert [(r.span_id, r.offset) for r in result.source_refs(0, 5)] == [("s1", 0), ("s1", 1), ("s2", 0), ("s2", 1)]


def test_drawn_rule_blocks_otherwise_eligible_join():
    assert len(line_views([span(), span("CD", 23, "s2")], [VerticalRule(21, 0, 30)])) == 2


def test_native_coalescence_is_split_without_losing_original_offsets():
    parts = split_at_rules(span("ABCD"), [VerticalRule(20, 0, 30)])
    assert [p.text for p in parts] == ["AB", "CD"]
    assert [p.source_offset for p in parts] == [0, 2]
    assert len(line_views([span("ABCD")], [VerticalRule(20, 0, 30)])) == 2


def test_rule_through_glyph_is_not_silently_split():
    result, = split_at_rules(span(), [VerticalRule(12, 0, 30)])
    assert result.text == "AB" and "RULE_INTERSECTS_GLYPH" in result.flags


def test_missing_glyph_geometry_remains_uncertain():
    result, = split_at_rules(replace(span(), glyph_boxes=()), [VerticalRule(15, 0, 30)])
    assert "CELL_BOUNDARY_UNRESOLVED" in result.flags


def test_classification_and_digit_uncertainty_survive():
    result = assemble([replace(span("1"), state="UNKNOWN"), span("2", 16, "s2")], [""])
    assert result.text == "12" and result.state == "UNKNOWN"
    assert set(result.flags) == {"DIGIT_JOIN", "CLASSIFICATION_UNKNOWN"}
    with pytest.raises(ValueError):
        assemble([replace(span(), state="FURNITURE")])


def test_scope_and_mapping_fail_closed():
    with pytest.raises(ValueError):
        assemble([span(), replace(span(), page=2)])
    with pytest.raises(ValueError):
        assemble([span()]).source_refs(0, 100)
    with pytest.raises(ValueError):
        replace(span(), glyph_boxes=((0, 0, 1, 1),))


def test_rotated_text_is_not_joined():
    assert len(line_views([span(direction=(0, 1)), span("CD", 23, "s2")])) == 2


def test_existing_observations_are_not_changed():
    original = span("ABCD")
    views = line_views([original], [VerticalRule(20, 0, 30)])
    assert original.text == "ABCD" and original.source_offset == 0
    assert [v.text for v in views] == ["AB", "CD"]
    assert views == line_views([original], [VerticalRule(20, 0, 30)])
