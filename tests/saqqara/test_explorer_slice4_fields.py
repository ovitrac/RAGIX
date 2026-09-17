"""Slice-4 reference-field gates on synthetic geometry (A-series).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Each gate runs at every corner of the declared geometry ranges: a result that
changes between corners is a hidden layout rule, not a pass. A gate marked
`xfail(strict=True)` states a defect measured on the frozen reader; the commit that
repairs it removes the mark, and an unexpected pass fails the suite.
"""

import pytest

from ragix_kernels.saqqara.census import CensusConfig
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.saqqara.value_windows import ContinuationPolicy

from . import fixtures_explorer_slice4_fields as fx

LABELS = {fx.LABEL_EN, fx.LABEL_FR}
corners = pytest.mark.parametrize("g", fx.GEOMETRIES, ids=lambda g: g.name)


def known(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def reference_policy(**values):
    """Policy data of the reference-field reader; resolved late so the gates collect."""
    from ragix_kernels.saqqara.value_windows import ReferencePolicy

    return CensusConfig(reference_policy=ReferencePolicy(**values))


def reference_fields(result):
    return [f for f in result.reading.fields if f.label in LABELS]


def reference_windows(result):
    return [w for w in result.census.windows if w.label in LABELS]


def own_identifiers(fields):
    return [[t.raw for t in f.targets] for f in fields]


def sections(target):
    return [(s.kind, s.key_from, s.key_to, s.connector) for s in target.sections]


def label_in_prose(result):
    return result.profile.fields["reference_fields"].diagnostics["label_in_prose"]


@corners
def test_a1_a4_underline_and_adjacent_value_span_are_not_stops(g):
    result = explore(fx.a1(g))
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(f.status == "READ" and not f.flags for f in fields)
    assert all(w.value_position == "same_line" for w in reference_windows(result))


UNDERLINE_ON_THE_EDGE = (
    "an underline on the label's bottom edge is read as a rule between two lines"
)
# Where the next line touches the label line, the underline lies exactly between them.
corners_underlined = pytest.mark.parametrize(
    "g",
    [
        pytest.param(g, marks=known(UNDERLINE_ON_THE_EDGE)) if g is fx.LOW2 else g
        for g in fx.GEOMETRIES
    ],
    ids=lambda g: g.name,
)


@corners_underlined
@pytest.mark.parametrize("wrap", ["value", "label"])
def test_a2_continuation_inside_the_value_cell(g, wrap):
    result = explore(fx.a2(g, wrap))
    fields = reference_fields(result)
    assert len(fields) == len(fx.PAGES)
    assert all(len(w.views) == 3 and w.stop_reason == "label" for w in reference_windows(result))
    for page, field in zip(fx.PAGES, fields):
        assert [t.raw for t in field.targets] == [fx.identifier(page), "AB-FORM-900002"]
        keys = [k for s in field.targets[0].sections for k in (s.key_from, s.key_to) if k]
        assert keys == ["4.1", "4.2", "4.3", "4.4", "5.1", "5.4"]
        assert "titre de formulaire" in field.view.text
        assert field.status == "READ"


@corners_underlined
@pytest.mark.parametrize("wrap", ["value", "label"])
def test_a2_open_range_is_closed_by_the_continuation_line(g, wrap):
    for field in reference_fields(explore(fx.a2(g, wrap))):
        assert sections(field.targets[0]) == [
            ("single", "4.1", None, None),
            ("single", "4.2", None, None),
            ("single", "4.3", None, None),
            ("single", "4.4", None, None),
            ("range", "5.1", "5.4", "à"),
        ]


@known(
    "an underline on the label's bottom edge closes the window; a word before an identifier is read as a role line"
)
@corners
def test_a3_value_starting_the_next_line(g):
    result = explore(fx.a3(g))
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(w.value_position == "next_line" for w in reference_windows(result))
    assert all(f.status == "READ" and "ROLE_LINE_UNDECIDABLE" not in f.flags for f in fields)


@known("label words inside prose are not counted")
@corners
@pytest.mark.parametrize("build", [fx.a5, fx.a6], ids=["a5", "a6"])
def test_a5_a6_label_words_in_prose_make_no_field(g, build):
    result = explore(build(g))
    assert own_identifiers(reference_fields(result)) == [[fx.identifier(p)] for p in fx.PAGES]
    assert len(result.reading.fields) == len(fx.PAGES)
    mentions = label_in_prose(result)
    assert [m["page"] for m in mentions] == list(fx.PAGES)
    assert all(m["reason"] == "NO_REFERENCE_CONTENT" and m["span_id"] for m in mentions)


@known("a colon-less label followed by reference content on its own line is never a candidate")
@corners
@pytest.mark.parametrize("evidence", ["declared", "colon_elsewhere"])
def test_a7_colonless_label_that_introduces_a_value(g, evidence):
    if evidence == "declared":
        result = explore(fx.a7(g), census_config=reference_policy(labels=(fx.LABEL_EN,)))
        expected = [[fx.identifier(p)] for p in fx.PAGES]
    else:
        result = explore(fx.a7(g, colon_evidence=True))
        expected = [[fx.identifier(p, family)] for p in fx.PAGES for family in ("CDE", "FGH")]
    assert own_identifiers(reference_fields(result)) == expected
    assert label_in_prose(result) == []


@known("type words are consumer policy data, absent from the frozen reader")
@corners
def test_a7_type_word_is_reference_content_only_when_declared(g):
    document = fx.a7(g, typed=True)
    declared = explore(
        document, census_config=reference_policy(labels=(fx.LABEL_EN,), type_words=(fx.TYPE_WORD,))
    )
    assert own_identifiers(reference_fields(declared)) == [[fx.identifier(p)] for p in fx.PAGES]
    undeclared = explore(document, census_config=reference_policy(labels=(fx.LABEL_EN,)))
    assert reference_fields(undeclared) == []
    assert [m["page"] for m in label_in_prose(undeclared)] == list(fx.PAGES)


IDEAL = [("shared", 0.0, False), ("per_cell", 0.0, False), ("per_cell", 0.3, False)]
PAINTED = [
    pytest.param(
        "shared", 0.0, True, marks=known("a label underline is taken as the cell's bottom edge")
    ),
    pytest.param(
        "per_cell", 0.3, True, marks=known("a label underline is taken as the cell's bottom edge")
    ),
]


@corners
@pytest.mark.parametrize("position", ["right", "below"])
@pytest.mark.parametrize("borders,jitter,underlined", [*IDEAL, *PAINTED])
def test_a8_label_cell_takes_its_adjacent_value_cell(g, position, borders, jitter, underlined):
    result = explore(fx.a8(g, position, borders, jitter, underlined))
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(f.status == "READ" for f in fields)
    windows = reference_windows(result)
    assert all(
        w.value_position == "next_cell" and w.stop_reason == "cell_boundary" for w in windows
    )
    assert not [f for f in result.reading.findings if f.reason == "RULE_STOP"]


@known("role-word lines are attached or become labels")
@corners
def test_a9_role_word_lines_are_listed_never_read(g):
    result = explore(fx.a9(g))
    fields = reference_fields(result)
    assert len(result.reading.fields) == len(fields) == len(fx.PAGES)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(f.status == "UNDECIDABLE" and "ROLE_LINE_UNDECIDABLE" in f.flags for f in fields)
    read = {t.raw for f in result.reading.fields for t in f.targets}
    assert not read & set(fx.ROLE_IDENTIFIERS)
    for window in reference_windows(result):
        assert window.stop_reason == "table_caption" and len(window.views) == 1
        assert [v.text for v in window.undecidable] == list(fx.ROLE_LINES)


@corners
@pytest.mark.parametrize("marker", ["§", ""], ids=["marked", "bare"])
@pytest.mark.parametrize("connector", ["à", "to"])
def test_a10_range_stays_one_relation(g, connector, marker):
    fields = reference_fields(explore(fx.a10(g, connector, marker)))
    assert len(fields) == len(fx.PAGES)
    for field in fields:
        assert sections(field.targets[0]) == [("range", "4.1", "4.12", connector)]


@corners
def test_a11_field_stops_at_the_page_break(g):
    result = explore(fx.a11(g))
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(sections(f.targets[0]) == [("single", "4.1", None, None)] for f in fields)
    assert all(len({v.page for v in w.views}) == 1 for w in result.census.windows)


@corners
def test_a12_gap_bound_is_derived_and_stops_the_field(g):
    result = explore(fx.a12(g))
    policy = result.census.continuation_policy
    assert policy.source == "derived" and policy.derivation_reason is None
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(sections(f.targets[0]) == [("single", "4.1", None, None)] for f in fields)
    assert all(len(w.views) == 1 for w in reference_windows(result))


@known("a gap stop carries the guard flag whatever the provenance of its bound")
@corners
def test_a12_derived_bound_is_structure_not_a_guard(g):
    result = explore(fx.a12(g))
    for window in reference_windows(result):
        assert window.stop_reason == "derived_gap" and not window.flags
        assert window.policy.gap_histogram and window.policy.valley_margin > 1
    assert all(f.status == "READ" and not f.needs_review for f in reference_fields(result))


@corners
def test_a12c_without_body_lines_nothing_is_derived(g):
    policy = explore(fx.a12(g, body_lines=0)).census.continuation_policy
    assert policy.source == "default" and policy.derivation_reason == "too_few_gaps"


@corners
def test_a13_numbered_heading_is_a_stop(g):
    result = explore(fx.a13(g))
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(sections(f.targets[0]) == [("single", "4.1", None, None)] for f in fields)
    assert all(w.stop_reason == "numbered_heading" for w in reference_windows(result))


BUILDS = {
    "a1": fx.a1,
    "a2": fx.a2,
    "a3": fx.a3,
    "a8_right_painted": lambda g: fx.a8(g, "right", "per_cell", 0.3),
    "a8_below_painted": lambda g: fx.a8(g, "below", "per_cell", 0.3),
    "a8_right": lambda g: fx.a8(g, "right", "per_cell", 0.3, True),
    "a8_below": lambda g: fx.a8(g, "below", "per_cell", 0.3, True),
    "a9": fx.a9,
    "a10": fx.a10,
    "a11": fx.a11,
    "a12": fx.a12,
    "a13": fx.a13,
}


def reading(result):
    return [
        (f.label, f.status, f.flags, [(t.raw, t.revision_raw, sections(t)) for t in f.targets])
        for f in result.reading.fields
    ]


# Swept once a fixture reads correctly: each repair extends this list, so the sweep
# never certifies an invariantly wrong reading.
SWEPT = ["a1", "a10", "a11", "a13", "a8_right_painted", "a8_below_painted"]


@corners
@pytest.mark.parametrize("name", SWEPT)
def test_sensitivity_fallback_bounds_never_change_a_reading(g, name):
    document = BUILDS[name](g)
    reference = reading(explore(document))
    assert len([f for f in reference if f[0] in LABELS]) >= len(fx.PAGES)
    for gap in (1.5, 2.5, 4.0):
        for cap in (6, 8, 16):
            config = CensusConfig(
                continuation_policy=ContinuationPolicy(max_gap_ratio=gap, max_lines=cap)
            )
            assert reading(explore(document, census_config=config)) == reference


def cells(result):
    return [(w.stop_reason, w.value_position, w.grid_cells) for w in reference_windows(result)]


@corners
@pytest.mark.parametrize("position", ["right", "below"])
@pytest.mark.parametrize("tolerance", [0.5, 1.0, 2.0])
def test_a8_rule_tolerance_is_a_guard_not_a_layout(g, position, tolerance):
    document = fx.a8(g, position, "per_cell", 0.3)
    declared = explore(document)
    swept = explore(document, census_config=reference_policy(rule_tolerance=tolerance))
    assert reading(swept) == reading(declared) and cells(swept) == cells(declared)
    assert swept.census.reference_policy.rule_tolerance == tolerance


def test_painted_rules_cluster_into_edges_without_drift():
    from ragix_kernels.saqqara.value_windows import _edges

    assert _edges([220.3, 60.0, 220.0, 480.3], 1.0) == [
        (60.0, 60.0),
        (220.0, 220.3),
        (480.3, 480.3),
    ]
    # A run of close rules never chains into one wide edge.
    assert _edges([0.0, 0.9, 1.8], 1.0) == [(0.0, 0.9), (1.8, 1.8)]
    # A column as narrow as one short line stays a column at the widest swept tolerance.
    assert _edges([100.0, 112.0], 2.0) == [(100.0, 100.0), (112.0, 112.0)]
    assert _edges([100.0, 100.3], 0.0) == [(100.0, 100.0), (100.3, 100.3)]


def test_reference_policy_is_sealed_in_the_census_and_refuses_invalid_values():
    from dataclasses import asdict
    from ragix_kernels.saqqara.census import census_from_dict
    from ragix_kernels.saqqara.value_windows import ReferencePolicy

    result = explore(fx.a1(fx.MID), census_config=reference_policy(rule_tolerance=0.5))
    assert census_from_dict(asdict(result.census)) == result.census
    for value in (-0.1, float("nan"), True):
        with pytest.raises(ValueError):
            ReferencePolicy(rule_tolerance=value)


def test_rows_are_shared_by_centre_not_by_overlap():
    from types import SimpleNamespace as box
    from ragix_kernels.saqqara.value_windows import _same_row

    line = box(bbox=(60.0, 100.0, 300.0, 112.0))
    assert _same_row(line, box(bbox=(310.0, 100.0, 400.0, 112.0)))
    assert _same_row(line, box(bbox=(310.0, 103.0, 330.0, 109.0)))  # a smaller span on the row
    assert not _same_row(line, box(bbox=(60.0, 109.5, 300.0, 121.5)))  # next line, boxes overlap
    assert not _same_row(line, box(bbox=(60.0, 112.0, 300.0, 124.0)))


def test_connector_is_classified_without_the_marker_of_the_next_number():
    from ragix_kernels.saqqara.census import bare_connector

    assert bare_connector(" à § ", ["", "§"]) == "à"
    assert bare_connector("à\n§", ["§"]) == "à"
    assert bare_connector(", §", ["§"]) == ","
    assert bare_connector("§", ["§"]) == ""
    assert bare_connector("à §", [""]) == "à §"  # an unobserved marker is never assumed


def numbering_records(result):
    return {
        r.literal: dict(r.attributes)["marker"]
        for r in result.census.records
        if r.category == "numbering"
    }


def test_section_sign_marks_every_number_but_a_word_marks_only_the_first():
    marked = numbering_records(explore(fx.a10(fx.MID, "à", "§")))
    assert marked["4.1"] == marked["4.12"] == "§"
    bare = numbering_records(explore(fx.a10(fx.MID, "à", "")))
    assert bare["4.12"] == ""  # the word before it is a connector


def test_connector_closing_a_wrapped_line_is_observed_on_its_own_line():
    result = explore(fx.a2(fx.MID))
    across = [
        r
        for r in result.census.records
        if r.category == "connector" and dict(r.attributes)["pattern"] == "across_lines"
    ]
    assert [(r.literal, r.count) for r in across] == [("à", len(fx.PAGES))]
    assert all(e.literal == "à" and e.end - e.start == 1 for r in across for e in r.evidence)
    assert result.profile.fields["numbering_style"].value["range_connectors"] == ["à"]
