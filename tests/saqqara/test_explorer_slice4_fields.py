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


# Both remaining expected failures await one ruling on bounds derived from the gap
# histogram; until then a derived bound stops as a guard and keeps its review flag.
WEAK_VALLEY = "a bound derived between two leading values closes the window; awaits a ruling"


def corners_with(reasons):
    """Geometry corners, those named carrying the defect that still blocks them."""
    return pytest.mark.parametrize(
        "g",
        [
            pytest.param(g, marks=known(reasons[g.name])) if g.name in reasons else g
            for g in fx.GEOMETRIES
        ],
        ids=lambda g: g.name,
    )


@corners
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


@corners
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


@corners
def test_a3_value_starting_the_next_line(g):
    result = explore(fx.a3(g))
    fields = reference_fields(result)
    assert own_identifiers(fields) == [[fx.identifier(p)] for p in fx.PAGES]
    assert all(w.value_position == "next_line" for w in reference_windows(result))
    assert all(f.status == "READ" and "ROLE_LINE_UNDECIDABLE" not in f.flags for f in fields)


@corners
@pytest.mark.parametrize("build", [fx.a5, fx.a6], ids=["a5", "a6"])
def test_a5_a6_label_words_in_prose_make_no_field(g, build):
    result = explore(build(g))
    assert own_identifiers(reference_fields(result)) == [[fx.identifier(p)] for p in fx.PAGES]
    assert len(result.reading.fields) == len(fx.PAGES)
    mentions = label_in_prose(result)
    # Label words in prose are observed, and they never vote for or against the label.
    field = result.profile.fields["reference_fields"]
    assert field.status == "PROBED" and field.confidence == 1
    assert [
        (c["positives"], c["non_empty_negatives"])
        for c in field.diagnostics["reference_counts"]
        if c["label"] == fx.LABEL_FR
    ] == [(len(fx.PAGES), 0)]
    assert [m["page"] for m in mentions] == list(fx.PAGES)
    assert all(m["reason"] == "NO_REFERENCE_CONTENT" and m["span_id"] for m in mentions)


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


GRIDS = [
    ("shared", 0.0, False),
    ("per_cell", 0.0, False),
    ("per_cell", 0.3, False),
    ("shared", 0.0, True),
    ("per_cell", 0.3, True),
]


@corners
@pytest.mark.parametrize("position", ["right", "below"])
@pytest.mark.parametrize("borders,jitter,underlined", GRIDS)
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


@corners_with({"mid": WEAK_VALLEY})
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


@known("a derived bound stops as a guard and keeps its flag; structure awaits a ruling")
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
    "a5": fx.a5,
    "a7": lambda g: fx.a7(g, colon_evidence=True),
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
SWEPT = [
    "a1",
    "a2",
    "a3",
    "a5",
    "a7",
    "a8_right",
    "a8_below",
    "a8_right_painted",
    "a8_below_painted",
    "a10",
    "a11",
    "a13",
]


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


def test_reference_content_is_what_a_label_may_introduce():
    from ragix_kernels.saqqara.census import identifiers
    from ragix_kernels.saqqara.value_windows import begins_with_reference

    for text in (" AB-CDE-900001 V 1.0", "§4.1", "§ 4.1", "4.1.2 et 4.3", "« titre »", "“title”"):
        assert begins_with_reference(text, identifiers)
    for text in (
        "par le laboratoire",
        "ensuite AB-CDE-900001",
        "4 essais",
        "",
        "ZRS AB-CDE-900001",
    ):
        assert not begins_with_reference(text, identifiers)
    for text in ("ZRS AB-CDE-900001", "zrs n° AB-CDE-900001", "ZRS : AB-CDE-900001"):
        assert begins_with_reference(text, identifiers, ("ZRS",))
    assert not begins_with_reference("ZRSX AB-CDE-900001", identifiers, ("ZRS",))
    assert not begins_with_reference("ZRS puis AB-CDE-900001", identifiers, ("ZRS",))
    # Shipped type words are language-generic document nouns.
    assert begins_with_reference("Spécification AB-CDE-900001", identifiers, reference_words())


def reference_words():
    from ragix_kernels.saqqara.value_windows import DEFAULT_REFERENCE

    return DEFAULT_REFERENCE.type_words


def test_fold_keeps_the_source_index_of_every_character():
    from ragix_kernels.saqqara.value_windows import fold

    folded, index = fold("Référence ÉTÉ")
    assert folded == "reference ete" and index == list(range(13))
    assert fold("ﬁn")[1] == [0, 0, 1]  # one ligature, two folded characters


@corners
def test_a_longer_word_sharing_the_label_prefix_is_not_the_label(g):
    """Plural or derived forms are not folded: a consumer declares the forms it uses."""
    document = fx.a7(g)
    result = explore(document, census_config=reference_policy(labels=("Tested referenc",)))
    assert reference_fields(result) == [] and label_in_prose(result) == []


def test_declared_words_are_validated_and_sealed():
    from dataclasses import asdict
    from ragix_kernels.saqqara.census import census_from_dict
    from ragix_kernels.saqqara.value_windows import ReferencePolicy

    config = reference_policy(labels=(fx.LABEL_EN,), type_words=(fx.TYPE_WORD,))
    result = explore(fx.a7(fx.MID, typed=True), census_config=config)
    assert census_from_dict(asdict(result.census)) == result.census
    assert result.census.reference_policy.labels == (fx.LABEL_EN,)
    for bad in ([fx.LABEL_EN], ("",), (" ",), (1,)):
        with pytest.raises(ValueError):
            ReferencePolicy(labels=bad)
        with pytest.raises(ValueError):
            ReferencePolicy(type_words=bad)


def test_role_line_opens_with_a_declared_word_directly_before_an_identifier():
    from ragix_kernels.saqqara.census import identifiers
    from ragix_kernels.saqqara.value_windows import DEFAULT_REFERENCE, role_line

    words = DEFAULT_REFERENCE.role_words
    for text in (
        "Procédure n° AB-SOP-900005 « Titre »",
        "PROCEDURE AB-SOP-900005",
        "Rapport : AB-OPE-900006",
        "• Voir AB-OPE-900006",
        "cf. AB-OPE-900006",
        "Forms # AB-FRM-900008",
    ):
        assert role_line(text, identifiers, words), text
    for text in (
        "Procédure générale de maintenance",  # no identifier
        "Procédure de AB-SOP-900005",  # not directly before it
        "Procedural AB-SOP-900005",  # a longer word
        "ZRS AB-CDE-900001",  # a type word is not a role word
        "AB-SOP-900005 procédure",
    ):
        assert not role_line(text, identifiers, words), text
    assert role_line("ZRS AB-CDE-900001", identifiers, ("ZRS",))


def test_a_word_cannot_be_both_a_type_word_and_a_role_word():
    from ragix_kernels.saqqara.value_windows import ReferencePolicy

    with pytest.raises(ValueError):
        ReferencePolicy(type_words=("Procédure",))  # shipped as a role word, accents aside
    assert ReferencePolicy(type_words=("ZRS",), role_words=("voir",)).role_words == ("voir",)


def test_listed_lines_survive_the_sealed_census_and_explain_an_empty_field():
    from dataclasses import asdict
    from ragix_kernels.saqqara.census import census_from_dict
    from ragix_kernels.saqqara.value_windows import unresolved_reason

    result = explore(fx.a9(fx.HIGH))
    assert census_from_dict(asdict(result.census)) == result.census
    # Nothing but a role line after the label: the occurrence is explained, not read.
    document = fx.role_only(fx.HIGH)
    result = explore(document)
    empty = [
        w for w in result.census.windows if w.label == fx.LABEL_FR and not w.following_text.strip()
    ]
    assert [unresolved_reason(w) for w in empty] == ["ROLE_LINE_UNDECIDABLE"] * len(fx.PAGES)
    assert all(len(w.undecidable) == 1 for w in empty)
    assert own_identifiers(reference_fields(result)) == [[fx.identifier(p)] for p in fx.PAGES]
    assert not {t.raw for f in result.reading.fields for t in f.targets} & set(fx.ROLE_IDENTIFIERS)


UNDERLINED = {
    "a1": fx.a1,
    "a2": fx.a2,
    "a3": fx.a3,
    "a8_right": BUILDS["a8_right"],
    "a8_below": BUILDS["a8_below"],
}


@corners
@pytest.mark.parametrize("name", sorted(UNDERLINED))
@pytest.mark.parametrize("margin", [0.3, 0.5, 1.0])
def test_underline_margin_is_a_guard_not_a_layout(g, name, margin):
    document = UNDERLINED[name](g)
    declared = explore(document)
    swept = explore(document, census_config=reference_policy(underline_margin=margin))
    assert reading(swept) == reading(declared) and cells(swept) == cells(declared)
    assert own_identifiers(reference_fields(swept)) == [
        [fx.identifier(p), *(["AB-FORM-900002"] if name == "a2" else [])] for p in fx.PAGES
    ]


@corners
@pytest.mark.parametrize("margin", [0.3, 0.5, 1.0])
def test_a_rule_running_past_the_line_is_an_edge_even_on_its_bottom_edge(g, margin):
    """The trap of I-12: an underline is emphasis, a table's top edge is a stop."""
    result = explore(fx.a3(g, edge=True), census_config=reference_policy(underline_margin=margin))
    assert reference_fields(result) == []
    assert all(w.stop_reason == "horizontal_rule" for w in reference_windows(result))
    assert [f.reason for f in result.reading.findings] == ["RULE_STOP"] * len(fx.PAGES)


def test_underline_is_told_from_an_edge_by_its_extent():
    from types import SimpleNamespace as view
    from ragix_kernels.saqqara.field_views import HorizontalRule
    from ragix_kernels.saqqara.value_windows import DEFAULT_REFERENCE, underlines

    line = view(bbox=(60.0, 100.0, 180.0, 112.0))
    for rule in (
        HorizontalRule(112.0, 60.0, 180.0),  # on the bottom edge
        HorizontalRule(109.5, 57.0, 183.0),  # inside the box, a little wider
        HorizontalRule(112.8, 60.0, 180.0),  # painted just under the box
        HorizontalRule(111.0, 60.0, 120.0),  # under part of the line
    ):
        assert underlines(rule, line, DEFAULT_REFERENCE), rule
    for rule in (
        HorizontalRule(112.0, 60.0, 480.0),  # runs past the text: an edge
        HorizontalRule(112.0, 20.0, 180.0),
        HorizontalRule(118.0, 60.0, 180.0),  # below the line: between two lines
        HorizontalRule(103.0, 60.0, 180.0),  # upper half: not an underline
    ):
        assert not underlines(rule, line, DEFAULT_REFERENCE), rule
