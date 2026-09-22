"""Whitespace, occurrence and partition falsifiers for content ledger v2.2.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace

import pytest

from ragix_kernels.harvest.region_types import BoundaryPolicy, RegionMember, RegionRefused
from ragix_kernels.harvest.regions import region
from ragix_kernels.saqqara.context_groups import (
    ContextGroupPolicy,
    cell_inventory,
    context_groups,
)
from ragix_kernels.saqqara.content_ledger import (
    NORMALISATION_RULE,
    NormalisedRange,
    OrderBreak,
    TextCarrier,
    TextEntry,
    TextLedger,
    text_ledger,
)
from ragix_kernels.saqqara.field_views import CharacterRef, TextView

from .test_context_groups import SOURCE, cell, fixture, span


def view(ident, text, mappings, *, top=10):
    refs = tuple(
        None if item is None else CharacterRef(item[0], item[1], (10 + i, top, 11 + i, top + 8))
        for i, item in enumerate(mappings)
    )
    return TextView(
        ident,
        SOURCE,
        1,
        text,
        refs,
        "UNKNOWN",
        ("CLASSIFICATION_UNKNOWN",),
        (10, top, 10 + max(len(text), 1), top + 8),
    )


def ledger_with_views(monkeypatch, spans, views, *, cells=None):
    import ragix_kernels.saqqara.content_ledger as module

    result, index = fixture(spans=spans, cells=cells)
    members = tuple(
        RegionMember(
            item.view_id,
            SOURCE,
            1,
            "LINE",
            item.text,
            item.bbox,
            tuple(dict.fromkeys(ref.span_id for ref in item.mapping if ref is not None)),
        )
        for item in views
    )
    index.regions = (region("PROSE", members, source_id=SOURCE, policy=BoundaryPolicy()),)
    monkeypatch.setattr(module, "page_lines", lambda page: list(views))
    groups = context_groups(result, index, source_id=SOURCE)
    return (
        text_ledger(result, index, groups, source_id=SOURCE),
        cell_inventory(result, groups, source_id=SOURCE),
        groups,
    )


def by_id(ledger):
    return {entry.span_id: entry for entry in ledger.entries}


def test_w1_lost_separator_across_spans_is_missing(monkeypatch):
    spans = (span("a", "10 ", (10, 10, 30, 18)), span("b", "43", (30, 10, 50, 18)))
    joined = view("line", "1043", (("a", 0), ("a", 1), ("b", 0), ("b", 1)))
    ledger, _, _ = ledger_with_views(monkeypatch, spans, (joined,))
    entry = by_id(ledger)["a"]
    assert entry.status == "PARTIAL"
    assert entry.carried_count == 2 and entry.normalised == ()
    assert entry.missing == ((2, 3),) and not ledger.passes


def test_w1_exact_cell_fallback_also_detects_the_lost_seam():
    spans = (span("a", "10 ", (10, 10, 30, 18)), span("b", "43", (30, 10, 50, 18)))
    result, index = fixture(
        spans=spans,
        cells=[cell("joined", "1043", (0, 0, 100, 40))],
    )
    index.regions = ()
    groups = context_groups(result, index, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    assert by_id(ledger)["a"].missing == ((2, 3),)
    assert by_id(ledger)["b"].status == "CARRIED_EXACT"
    assert not ledger.passes


def test_w2_inserted_separator_is_accepted_normalisation(monkeypatch):
    spans = (span("a", "10 ", (10, 10, 30, 18)), span("b", "43", (30, 10, 50, 18)))
    joined = view(
        "line",
        "10 43",
        (("a", 0), ("a", 1), None, ("b", 0), ("b", 1)),
    )
    ledger, _, _ = ledger_with_views(monkeypatch, spans, (joined,))
    entry = by_id(ledger)["a"]
    assert entry.status == "CARRIED_NORMALISED"
    assert entry.carried_count == 2
    assert entry.normalised == (NormalisedRange(2, 3),)
    assert entry.missing == () and ledger.passes


def test_w2_mapped_separator_remains_carried(monkeypatch):
    source = span("a", "10 43", (10, 10, 50, 18))
    joined = view(
        "line",
        "10 43",
        tuple(("a", offset) for offset in range(5)),
    )
    ledger, _, _ = ledger_with_views(monkeypatch, (source,), (joined,))
    entry = ledger.entries[0]
    assert entry.status == "CARRIED" and entry.carried_count == 5
    assert entry.normalised == () and entry.missing == ()


def test_w3_different_line_members_preserve_the_boundary(monkeypatch):
    spans = (span("a", "10 ", (10, 10, 30, 18)), span("b", "43", (10, 30, 30, 38)))
    lines = (
        view("line-a", "10", (("a", 0), ("a", 1)), top=10),
        view("line-b", "43", (("b", 0), ("b", 1)), top=30),
    )
    ledger, _, _ = ledger_with_views(monkeypatch, spans, lines)
    entry = by_id(ledger)["a"]
    assert entry.status == "CARRIED_NORMALISED"
    assert entry.normalised == (NormalisedRange(2, 3),)
    assert ledger.passes


def test_w3_different_cells_preserve_the_boundary():
    spans = (span("a", "10 ", (10, 10, 30, 18)), span("b", "43", (40, 10, 60, 18)))
    result, index = fixture(
        spans=spans,
        cells=[
            cell("left", "10", (0, 0, 35, 30)),
            cell("right", "43", (35, 0, 80, 30)),
        ],
    )
    index.regions = ()
    groups = context_groups(result, index, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    assert by_id(ledger)["a"].status == "CARRIED_NORMALISED"
    assert ledger.passes


def test_w4_entire_separator_run_is_missing(monkeypatch):
    spans = (span("a", "10  ", (10, 10, 30, 18)), span("b", "43", (30, 10, 50, 18)))
    joined = view("line", "1043", (("a", 0), ("a", 1), ("b", 0), ("b", 1)))
    ledger, _, _ = ledger_with_views(monkeypatch, spans, (joined,))
    entry = by_id(ledger)["a"]
    assert entry.status == "PARTIAL"
    assert entry.missing == ((2, 4),)
    assert (
        entry.carried_count
        + sum(item.end - item.start for item in entry.normalised)
        + sum(end - start for start, end in entry.missing)
        == entry.span_length
    )


def test_w5_trailing_whitespace_normalises_when_it_separates_nothing(monkeypatch):
    source = span("a", "10 ", (10, 10, 30, 18))
    line = view("line", "10", (("a", 0), ("a", 1)))
    ledger, _, _ = ledger_with_views(monkeypatch, (source,), (line,))
    entry = ledger.entries[0]
    assert entry.status == "CARRIED_NORMALISED"
    assert entry.normalised == (NormalisedRange(2, 3),)
    assert ledger.passes


def test_w6_whitespace_elsewhere_cannot_pay_for_the_first_gap(monkeypatch):
    source = span("a", "a b c", (10, 10, 60, 18))
    line = view(
        "line",
        "ab c",
        (("a", 0), ("a", 2), ("a", 3), ("a", 4)),
    )
    ledger, _, _ = ledger_with_views(monkeypatch, (source,), (line,))
    entry = ledger.entries[0]
    assert entry.status == "PARTIAL"
    assert entry.carried_count == 4
    assert entry.missing == ((1, 2),) and entry.normalised == ()


def test_w7_reordered_mapped_characters_are_an_order_break(monkeypatch):
    source = span("a", "12", (10, 10, 30, 18))
    line = view("line", "21", (("a", 1), ("a", 0)))
    ledger, _, _ = ledger_with_views(monkeypatch, (source,), (line,))
    assert ledger.entries[0].status == "CARRIED"
    assert len(ledger.order_breaks) == 1
    assert ledger.order_breaks[0].positions == ((0, 1), (1, 0))
    assert not ledger.passes


def test_w8_one_exact_occurrence_never_pays_for_two_sources():
    spans = (span("a", "10", (10, 10, 30, 18)), span("b", "10", (10, 10, 30, 18)))
    result, index = fixture(
        spans=spans,
        cells=[cell("survivor", "10", (0, 0, 100, 40))],
    )
    index.regions = ()
    groups = context_groups(result, index, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    counts = {
        status: sum(entry.status == status for entry in ledger.entries)
        for status in ("CARRIED_EXACT", "NOT_CARRIED")
    }
    assert counts == {"CARRIED_EXACT": 1, "NOT_CARRIED": 1}
    assert not ledger.passes


def test_w9_unmapped_non_whitespace_remains_missing(monkeypatch):
    source = span("a", "A!", (10, 10, 30, 18))
    line = view("line", "A", (("a", 0),))
    ledger, _, _ = ledger_with_views(monkeypatch, (source,), (line,))
    entry = ledger.entries[0]
    assert entry.status == "PARTIAL"
    assert entry.missing == ((1, 2),) and entry.normalised == ()


def test_w10_unreadable_cell_never_becomes_empty_success(monkeypatch):
    source = span("a", "text", (10, 10, 40, 18))
    line = view("line", "text", tuple(("a", offset) for offset in range(4)))
    ledger, inventory, _ = ledger_with_views(
        monkeypatch,
        (source,),
        (line,),
        cells=[cell("unreadable", None, (0, 0, 100, 40))],
    )
    assert ledger.passes and ledger.entries[0].status == "CARRIED"
    assert inventory[0].readability == "UNREADABLE"
    assert inventory[0].text is None


def test_w11_partition_arithmetic_is_enforced():
    carrier = (TextCarrier("MEMBER", "line", "MAPPED"),)
    valid = TextEntry(
        "span",
        1,
        "10 ",
        3,
        "CARRIED_NORMALISED",
        carrier,
        2,
        (NormalisedRange(2, 3),),
    )
    assert valid.carried_count == 2
    with pytest.raises(RegionRefused, match="INVALID_TEXT_PARTITION"):
        replace(valid, carried_count=3)
    with pytest.raises(RegionRefused, match="INVALID_TEXT_PARTITION"):
        replace(valid, missing=((2, 3),), status="PARTIAL")
    with pytest.raises(RegionRefused, match="INVALID_TEXT_STATUS"):
        replace(valid, status="CARRIED")


def test_summary_separates_passed_normalised_missing_and_excluded(monkeypatch):
    spans = (
        span("a", "10 ", (10, 10, 30, 18)),
        span("b", "43", (30, 10, 50, 18)),
    )
    joined = view(
        "line",
        "10 43",
        (("a", 0), ("a", 1), None, ("b", 0), ("b", 1)),
    )
    ledger, _, _ = ledger_with_views(monkeypatch, spans, (joined,))
    summary = ledger.summary()
    assert summary["rule"] == "text-occurrence-ledger/2"
    assert summary["categories"] == {
        "passed": 1,
        "accepted_normalisation": 1,
        "missing_content": 0,
        "excluded": 0,
    }
    assert summary["page_categories"][1] == summary["categories"]
    assert summary["order_breaks"] == 0


def test_p1_final_groups_compose_envelope_chunks_inventory_and_ledger():
    spans = (
        span("header", "Unit V", (10, 10, 50, 18)),
        span("value-a", "10", (10, 30, 30, 38)),
        span("value-b", "43", (40, 30, 60, 38)),
    )
    cells = [
        cell("header-cell", "Unit V", (0, 0, 70, 20), spans=("header",)),
        cell("value-cell-a", "10", (0, 20, 35, 50), spans=("value-a",)),
        cell("value-cell-b", "43", (35, 20, 70, 50), spans=("value-b",)),
        cell(
            "placeholder",
            None,
            (0, 0, 900, 19000),
            flags=("MISSING_CELL_GEOMETRY",),
        ),
    ]
    result, index = fixture(spans=spans, cells=cells)
    policy = ContextGroupPolicy(max_group_members=2, max_group_cells=2)
    groups = context_groups(result, index, source_id=SOURCE, policy=policy)
    inventory = cell_inventory(result, groups, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    assert len(groups) == 2
    assert all(group.candidate_id == "candidate" and group.chunk_count == 2 for group in groups)
    assert groups[0].next_group_id == groups[1].group_id
    assert groups[1].prev_group_id == groups[0].group_id
    assert all(group.envelope_box == (0, 0, 70, 50) for group in groups)
    assert len(inventory) == len({entry.cell_id for entry in inventory}) == 4
    placeholder = next(entry for entry in inventory if entry.cell_id == "placeholder")
    assert placeholder.observed_bbox is None and placeholder.readability == "UNREADABLE"
    assert ledger.passes
    assert {entry.status for entry in ledger.entries} == {"CARRIED"}
    assert not any(hasattr(group, "semantic_binding") for group in groups)


def test_w11_ranges_and_order_breaks_stay_inside_the_source_span():
    carrier = (TextCarrier("MEMBER", "line", "MAPPED"),)
    valid = TextEntry("span", 1, "abc", 3, "PARTIAL", carrier, 2, (), ((2, 3),))
    with pytest.raises(RegionRefused, match="INVALID_TEXT_PARTITION"):
        replace(valid, missing=((3, 4),))


def test_w8_two_exact_occurrences_pay_for_two_sources():
    spans = (span("a", "10", (10, 10, 30, 18)), span("b", "10", (10, 10, 30, 18)))
    result, index = fixture(
        spans=spans,
        cells=[cell("survivor", "10 10", (0, 0, 100, 40))],
    )
    index.regions = ()
    groups = context_groups(result, index, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    assert [entry.status for entry in ledger.entries] == ["CARRIED_EXACT", "CARRIED_EXACT"]
    assert ledger.passes


def test_w11_order_break_offsets_stay_inside_the_entry():
    carrier = (TextCarrier("MEMBER", "line", "MAPPED"),)
    entry = TextEntry("span", 1, "abc", 3, "CARRIED", carrier, 3)
    broken = OrderBreak("span", "line", ((0, 0), (1, 3), (2, 2)))
    with pytest.raises(RegionRefused, match="INVALID_TEXT_LEDGER"):
        TextLedger(SOURCE, (entry,), (broken,))


def test_whitespace_only_span_without_a_carrier_is_normalised():
    result, index = fixture(spans=[span("space", " \t", (10, 10, 20, 18))])
    index.regions = ()
    groups = context_groups(result, index, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    entry = ledger.entries[0]
    assert entry.status == "CARRIED_NORMALISED"
    assert entry.carried_count == 0
    assert entry.normalised == (NormalisedRange(0, 2),)
    assert ledger.passes


def test_edge_whitespace_never_hides_uncarried_non_whitespace():
    result, index = fixture(spans=[span("lost", " missing ", (10, 10, 70, 18))])
    index.regions = ()
    groups = context_groups(result, index, source_id=SOURCE)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    entry = ledger.entries[0]
    assert entry.status == "PARTIAL"
    assert entry.normalised == (NormalisedRange(0, 1), NormalisedRange(8, 9))
    assert entry.missing == ((1, 8),)
    assert not ledger.passes
