"""E7.9: occurrence-level reference evidence and bounded grid adjacency.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
from collections import Counter
import pytest
from ragix_kernels.saqqara.census import DocumentDigest, PageDigest, census
from ragix_kernels.saqqara.field_views import TextSpan, VerticalRule, HorizontalRule
from ragix_kernels.saqqara.profile import ProfileConfig, derive_profile
from ragix_kernels.saqqara.profile_readers import read_document
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.harvest.report import render_report


def text(value, page, ident, x=40, y=100):
    boxes = tuple((x + 4 * i, y, x + 4 * (i + 1), y + 10) for i in range(len(value)))
    return TextSpan(
        "synthetic-occurrences",
        f"s:{page}:{ident}",
        page,
        value,
        (x, y, x + 4 * len(value), y + 10),
        boxes,
        (x, y + 10),
        font_size=10,
    )


def occurrence_document():
    pages = []
    for p in range(1, 22):
        spans = [text("Repeated header", p, "header", y=20), text("Link:", p, "label")]
        vertical = ()
        horizontal = ()
        if p <= 12 or p in (16, 17):
            vertical = tuple(VerticalRule(x, 90, 130) for x in (20, 200, 400))
            horizontal = tuple(HorizontalRule(y, 20, 400) for y in (90, 130))
            if p <= 12:
                spans.append(text(f"ZX-R-{p:03} §4.2", p, "value", x=220))
        elif p == 13:
            spans[-1] = text("Link: ZX-R-013 §4.2", p, "label")
        elif p in (14, 15):
            spans.append(text("ZX-R-800", p, "outside", y=300))
        elif p in (18, 19):
            horizontal = (HorizontalRule(125, 20, 400),)
            spans.append(text("ZX-R-801", p, "outside", y=140))
        pages.append(PageDigest(p, 600, 800, tuple(spans), vertical, horizontal_rules=horizontal))
    return DocumentDigest("synthetic-occurrences", tuple(pages), "synthetic", "1")


@pytest.mark.parametrize("minimum", range(2, 6))
def test_e7_9a_d_found_fields_survive_empty_windows(minimum):
    doc = occurrence_document()
    c = census(doc)
    p = derive_profile(c, ProfileConfig(minimum_occurrences=minimum))
    r = read_document(doc, c, p)
    field = p.fields["reference_fields"]
    assert field.status == "PROBED" and field.confidence == 1
    stats = next(s for s in field.diagnostics["reference_counts"] if s["label"] == "Link")
    assert (stats["positives"], stats["non_empty_negatives"], stats["empty"]) == (13, 0, 8)
    assert len(field.diagnostics["unresolved_occurrences"]) == 8
    assert len(r.fields) == 13 and len(r.findings) == 8
    assert Counter(f.reason for f in r.findings) == {
        "WINDOW_BOUND_HIT": 2,
        "EMPTY_CELL": 2,
        "RULE_STOP": 2,
        "NO_TEXT": 2,
    }
    assert all(f.page and f.span_id for f in r.findings)
    assert Counter(w.value_position for w in c.windows) == {
        "next_cell": 12,
        "same_line": 1,
        "none": 8,
    }
    assert all(len(f.targets) == 1 for f in r.fields)


def class_document(values):
    return DocumentDigest(
        "synthetic-occurrences",
        tuple(
            PageDigest(
                p, 600, 800, (text("Repeated header", p, "h", y=20), text("Link: " + value, p, "l"))
            )
            for p, value in enumerate(values, 1)
        ),
        "synthetic",
        "1",
    )


def test_e7_9b_free_text_plurality_is_unknown_but_candidates_retained():
    result = explore(class_document(["ZX-R-001", "ZX-R-002", *("some words" for _ in range(6))]))
    field = result.profile.fields["reference_fields"]
    assert field.status == "UNKNOWN" and field.value is None and field.confidence == 0.25
    stats = field.diagnostics["reference_counts"][0]
    assert (stats["positives"], stats["non_empty_negatives"], stats["empty"]) == (2, 6, 0)
    assert len(result.reading.fields) == 2 and all(
        f.needs_review and f.status == "UNDECIDABLE" for f in result.reading.fields
    )
    assert len([f for f in result.reading.findings if f.reason == "NON_IDENTIFIER_VALUE"]) == 6


def test_plurality_does_not_hide_a_majority_fraction_rule():
    result = explore(
        class_document(["ZX-R-001", "ZX-R-002", "ZX-R-003", "some words", "other words", "7", "8"])
    )
    field = result.profile.fields["reference_fields"]
    assert field.status == "PROBED" and field.confidence == 3 / 7
    with pytest.raises(TypeError):
        ProfileConfig(reference_fraction=0.8)


@pytest.mark.parametrize("position", ["right", "below"])
def test_e7_9c_cell_rules_define_the_window(position):
    pages = []
    for p in (1, 2):
        values = [text("Link:", p, "l")]
        values.append(
            text(
                "ZX-R-923 §4.2",
                p,
                "v",
                x=220 if position == "right" else 40,
                y=100 if position == "right" else 150,
            )
        )
        vertical = tuple(VerticalRule(x, 90, 180) for x in (20, 200, 400))
        horizontal = tuple(HorizontalRule(y, 20, 400) for y in (90, 130, 180)) + (
            HorizontalRule(115, 450, 590),
        )
        pages.append(PageDigest(p, 600, 800, tuple(values), vertical, horizontal_rules=horizontal))
    result = explore(DocumentDigest("synthetic-occurrences", tuple(pages), "synthetic", "1"))
    assert len(result.reading.fields) == 2
    assert all(f.targets[0].raw == "ZX-R-923" for f in result.reading.fields)
    assert all(
        w.stop_reason == "cell_boundary" and w.value_position == "next_cell"
        for w in result.census.windows
    )
    assert all(w.grid_cells for w in result.census.windows)


def test_e7_9e_default_reason_is_in_profile_and_report():
    for ys, reason in [
        ([], "no_body_lines"),
        ([100, 116], "too_few_gaps"),
        ([100, 116, 132, 148, 164], "unimodal"),
    ]:
        spans = tuple(text("ordinary words", 1, i, y=y) for i, y in enumerate(ys))
        result = explore(
            DocumentDigest(
                "synthetic-occurrences", (PageDigest(1, 600, 800, spans),), "synthetic", "1"
            )
        )
        assert result.profile.continuation_policy.source == "default"
        assert result.profile.continuation_policy.derivation_reason == reason
        assert reason in render_report(
            result.report, {"field": "field", "quantity": "quantity", "table_row": "row"}
        )
    spans = tuple(
        text("ordinary words", 1, i, y=y) for i, y in enumerate((100, 116, 180, 196, 260, 276))
    )
    result = explore(
        DocumentDigest("synthetic-occurrences", (PageDigest(1, 600, 800, spans),), "synthetic", "1")
    )
    assert result.profile.continuation_policy.source == "derived"
    assert result.profile.continuation_policy.derivation_reason is None


def test_tied_plurality_is_unknown_and_counts_stay_visible():
    result = explore(class_document(["ZX-R-001", "ZX-R-002", "some words", "other words"]))
    field = result.profile.fields["reference_fields"]
    assert field.status == "UNKNOWN" and field.confidence == 0.5
    assert field.diagnostics["reference_counts"][0]["reason"] == "COMPETING_PLURALITY"
    assert len(result.reading.fields) == 2


def test_occurrence_findings_are_counted_in_coverage():
    result = explore(occurrence_document())
    assert result.report.coverage.unresolved_occurrences == 8
    assert result.report.coverage.unknown_template_fields == 0
