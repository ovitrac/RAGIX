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


@known("a line box overlapping the previous one is read as the same row: column_break")
@corners
@pytest.mark.parametrize("wrap", ["value", "label"])
def test_a2_continuation_inside_the_value_cell(g, wrap):
    result = explore(fx.a2(g, wrap))
    fields = reference_fields(result)
    assert len(fields) == len(fx.PAGES)
    assert all(len(w.views) == 3 for w in reference_windows(result))
    for page, field in zip(fx.PAGES, fields):
        assert [t.raw for t in field.targets] == [fx.identifier(page), "AB-FORM-900002"]
        assert sections(field.targets[0]) == [
            ("single", "4.1", None, None),
            ("single", "4.2", None, None),
            ("single", "4.3", None, None),
            ("single", "4.4", None, None),
            ("range", "5.1", "5.4", "à"),
        ]
        assert "titre de formulaire" in field.view.text
        assert field.status == "READ"


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


IDEAL = [("shared", 0.0, False), ("per_cell", 0.0, False)]
PAINTED = [
    pytest.param(
        "shared", 0.0, True, marks=known("a label underline is taken as the cell's bottom edge")
    ),
    pytest.param(
        "per_cell", 0.3, False, marks=known("rule coincidence is tested by exact equality")
    ),
    pytest.param(
        "per_cell", 0.3, True, marks=known("rule coincidence is tested by exact equality")
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


@known(
    "the census keeps the marker inside the connector and the profile classifies bare connectors only"
)
@corners
@pytest.mark.parametrize("connector", ["à", "to"])
def test_a10_range_stays_one_relation(g, connector):
    fields = reference_fields(explore(fx.a10(g, connector)))
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
SWEPT = ["a1", "a11", "a13"]


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
