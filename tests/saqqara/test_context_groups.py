"""Synthetic content-retention falsifiers, independent of table acceptance.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from ragix_kernels.harvest.region_types import (
    BoundaryPolicy,
    RegionMember,
    RegionRefused,
    TableRegionRefusal,
)
from ragix_kernels.harvest.regions import region
from ragix_kernels.saqqara.census import (
    DocumentDigest,
    Evidence,
    PageDigest,
    TableObservation,
    page_lines,
)
from ragix_kernels.saqqara.context_groups import ContextGroupPolicy, cell_inventory, context_groups
from ragix_kernels.saqqara.content_ledger import text_ledger
from ragix_kernels.saqqara.explorer import ExplorerResult, explore
from ragix_kernels.saqqara.field_views import TextSpan, VerticalRule
from ragix_kernels.saqqara.renderable_regions import regions_from_explorer
from ragix_kernels.saqqara.table_views import TableCell

SOURCE = "synthetic-context"


def cell(ident="c", text="entry", box=(0, 0, 200, 200), *, flags=(), spans=()):
    return TableCell(ident, text, box, spans, flags, "cell_box")


def span(ident="s", text="entry", box=(20, 20, 50, 30), *, glyphs=()):
    return TextSpan(SOURCE, ident, 1, text, box, glyphs, (box[0], box[3]))


def table(ident, cells):
    return TableObservation(
        ident,
        1,
        ("observed",),
        tuple((c.text,) for c in cells[1:]),
        tuple(
            Evidence(SOURCE, 1, c.cell_id, 0, len(c.text or ""), c.text or "", c.bbox)
            for c in cells
        ),
        cell_rows=tuple((c,) for c in cells),
    )


def fixture(
    *, spans=(), cells=None, tables=None, region_sizes=None, rules=(), furniture=(), omit=()
):
    """Explicit observations and independent region partition; no table inference."""
    if tables is None:
        tables = (table("candidate", [cell()] if cells is None else cells),)
    page = PageDigest(1, 1000, 20000, tuple(spans), tuple(rules), tuple(tables))
    document = DocumentDigest(SOURCE, (page,), "synthetic", "1")
    views = page_lines(page)
    members = tuple(
        RegionMember(
            v.view_id,
            SOURCE,
            1,
            "LINE",
            v.text,
            v.bbox,
            tuple(dict.fromkeys(r.span_id for r in v.mapping if r is not None)),
        )
        for v in views
        if v.view_id not in omit and v.view_id not in furniture
    )
    sizes = region_sizes if region_sizes is not None else [1] * len(members)
    assert sum(sizes) == len(members)
    regions, start = [], 0
    for size in sizes:
        regions.append(
            region(
                "PROSE", members[start : start + size], source_id=SOURCE, policy=BoundaryPolicy()
            )
        )
        start += size
    index = SimpleNamespace(
        source_id=SOURCE,
        regions=tuple(regions),
        refusals=tuple(
            TableRegionRefusal(
                SOURCE,
                t.table_id,
                (1,),
                "STRADDLING_OR_OUTSIDE_BANDS",
                tuple(c.cell_id for row in t.cell_rows for c in row),
            )
            for t in tables
        ),
    )
    result = ExplorerResult(
        document,
        None,
        None,
        SimpleNamespace(source_id=SOURCE, furniture=tuple(furniture)),
        SimpleNamespace(status="READY"),
    )
    return result, index


def groups_for(result, index, **policy):
    return context_groups(result, index, source_id=SOURCE, policy=ContextGroupPolicy(**policy))


def entry(result, index):
    groups = groups_for(result, index)
    ledger = text_ledger(result, index, groups, source_id=SOURCE)
    assert len(ledger.entries) == 1
    return ledger, ledger.entries[0]


def test_twenty_lines_are_grouped_without_structure_or_new_text():
    result, index = fixture(
        spans=[span(str(i), f"line {i}", (20, 2 + i * 8, 90, 8 + i * 8)) for i in range(20)]
    )
    before = repr(result), repr(index)
    groups = groups_for(result, index)
    assert len(groups) == 1
    assert groups[0].member_ids == tuple(m.member_id for r in index.regions for m in r.members)
    assert groups[0].cell_ids == ("c",)
    assert "text" not in asdict(groups[0])
    assert not {"row", "column", "row_span", "column_span"} & asdict(groups[0]).keys()
    assert (repr(result), repr(index)) == before
    assert text_ledger(result, index, groups, source_id=SOURCE).passes


@pytest.mark.parametrize(
    "text,state",
    [(None, "UNREADABLE"), ("", "READABLE_EMPTY"), (" ", "READABLE"), ("entry", "READABLE")],
)
def test_cell_readability_is_independent_of_native_spans(text, state):
    result, index = fixture(cells=[cell(text=text)])
    groups = groups_for(result, index)
    inventory = cell_inventory(result, groups, source_id=SOURCE)
    assert inventory[0].text == text and inventory[0].readability == state
    assert ("CONTAINS_UNREADABLE" in groups[0].flags) == (text is None)
    assert text_ledger(result, index, groups, source_id=SOURCE).entries == ()


@pytest.mark.parametrize(
    "sizes,expected,split",
    [
        ([300, 300, 300], [300, 300, 300], False),
        ([401], [400, 1], True),
        ([300, 200, 400], [300, 200, 400], False),
    ],
)
def test_member_chunks_respect_region_boundaries_or_explicitly_split(sizes, expected, split):
    n = sum(sizes)
    spans = [span(str(i), f"line {i}", (10, 1 + 10 * i, 100, 8 + 10 * i)) for i in range(n)]
    result, index = fixture(spans=spans, cells=[cell(box=(0, 0, 200, 10000))], region_sizes=sizes)
    groups = groups_for(result, index)
    assert [len(g.member_ids) for g in groups] == expected
    ids = [mid for g in groups for mid in g.member_ids]
    assert len(ids) == len(set(ids)) == n
    assert all(("REGION_SPLIT" in g.flags) == split for g in groups)
    assert groups[0].prev_group_id is None and groups[-1].next_group_id is None
    for a, b in zip(groups, groups[1:]):
        assert a.next_group_id == b.group_id and b.prev_group_id == a.group_id


def test_cell_only_chunks_are_bounded_unique_and_linked():
    cells = [cell(str(i), str(i), (0, i * 2, 10, i * 2 + 1)) for i in range(900)]
    result, index = fixture(cells=cells)
    groups = groups_for(result, index)
    assert [len(g.cell_ids) for g in groups] == [400, 400, 100]
    assert all(g.member_ids == () for g in groups)
    assert len({g.group_id for g in groups}) == 3
    inventory = cell_inventory(result, groups, source_id=SOURCE)
    assert len(inventory) == len({e.cell_id for e in inventory}) == 900
    assert groups[0].next_group_id == groups[1].group_id


def test_page_scale_and_shared_members_are_explicit():
    candidates = (
        table("a", [cell("a0", box=(0, 0, 1000, 18000))]),
        table("b", [cell("b0", box=(10, 10, 900, 16000))]),
    )
    result, index = fixture(spans=[span()], tables=candidates)
    before = repr(index)
    groups = groups_for(result, index)
    assert len(groups) == 2 and groups[0].member_ids == groups[1].member_ids
    assert all({"PAGE_SCALE_CANDIDATE", "SHARES_MEMBERS"} <= set(g.flags) for g in groups)
    assert repr(index) == before
    assert len(text_ledger(result, index, groups, source_id=SOURCE).entries) == 1


def test_partial_geometry_does_not_inflate_envelope_with_placeholder():
    cells = [
        cell("located", box=(10, 10, 80, 80)),
        cell("missing", box=(0, 0, 900, 19000), flags=("MISSING_CELL_GEOMETRY",)),
    ]
    result, index = fixture(spans=[span()], cells=cells)
    groups = groups_for(result, index)
    assert groups[0].envelope_box == (10, 10, 80, 80)
    assert groups[0].flags == ("PARTIAL_GEOMETRY",)
    records = {e.cell_id: e for e in cell_inventory(result, groups, source_id=SOURCE)}
    assert records["missing"].observed_bbox is None
    assert records["missing"].container_bbox == (0, 0, 900, 19000)
    assert records["missing"].flags == ("MISSING_CELL_GEOMETRY",)


def test_all_geometry_missing_preserves_text_without_claiming_an_envelope():
    result, index = fixture(cells=[cell(flags=("MISSING_CELL_GEOMETRY",))])
    groups = groups_for(result, index)
    assert len(groups) == 1 and groups[0].envelope_box is None and groups[0].member_ids == ()
    assert groups[0].flags == ("GEOMETRY_UNAVAILABLE",)
    assert cell_inventory(result, groups, source_id=SOURCE)[0].text == "entry"


def test_missing_geometry_cannot_prove_exact_retention():
    result, index = fixture(spans=[span()], cells=[cell(flags=("MISSING_CELL_GEOMETRY",))])
    index.regions = ()
    ledger, record = entry(result, index)
    assert record.status == "NOT_CARRIED" and not ledger.passes


def test_exact_cell_retention_requires_literal_and_containment():
    result, index = fixture(spans=[span()], cells=[cell(text="prefix entry suffix")])
    index.regions = ()
    ledger, record = entry(result, index)
    assert record.status == "CARRIED_EXACT" and ledger.passes
    assert record.carriers[0].kind == "CELL" and record.carriers[0].how == "EXACT"
    result, index = fixture(spans=[span()], cells=[cell(text="entry", box=(500, 500, 600, 600))])
    index.regions = ()
    assert entry(result, index)[1].status == "NOT_CARRIED"


def test_span_id_alone_does_not_cover_a_missing_sign():
    result, index = fixture(spans=[span(text="-37")], cells=[cell(text="37", spans=("s",))])
    index.regions = ()
    ledger, record = entry(result, index)
    assert record.status == "NOT_CARRIED" and not ledger.passes


def test_two_partial_cells_do_not_claim_exact_whole_span():
    result, index = fixture(
        spans=[span(text="ABCD", box=(0, 0, 40, 10))],
        cells=[cell("left", "AB", (0, 0, 20, 10)), cell("right", "CD", (20, 0, 40, 10))],
    )
    index.regions = ()
    assert entry(result, index)[1].status == "NOT_CARRIED"


def test_partial_character_mapping_records_missing_sign_offsets():
    native = span(
        text="-37",
        box=(0, 10, 30, 20),
        glyphs=((0, 10, 10, 20), (10, 10, 20, 20), (20, 10, 30, 20)),
    )
    result, index = fixture(
        spans=[native], cells=[cell(text="37")], rules=[VerticalRule(10, 0, 30)]
    )
    # Keep the right source fragment only; its mapping starts at source offset 1.
    index.regions = tuple(r for r in index.regions if r.members[0].text == "37")
    ledger, record = entry(result, index)
    assert record.status == "PARTIAL" and record.missing == ((0, 1),) and not ledger.passes


def test_literal_substitution_under_same_member_id_is_refused():
    result, index = fixture(spans=[span(text="-37")])
    r = index.regions[0]
    index.regions = (replace(r, members=(replace(r.members[0], text="+37"),)),)
    with pytest.raises(RegionRefused, match="STALE_CONTEXT_MEMBER_TEXT"):
        entry(result, index)


def test_changed_character_mapping_is_refused(monkeypatch):
    import ragix_kernels.saqqara.content_ledger as module

    result, index = fixture(spans=[span(text="-37")])
    views = page_lines(result.document.pages[0])
    bad = replace(views[0], mapping=(views[0].mapping[1], *views[0].mapping[1:]))
    monkeypatch.setattr(module, "page_lines", lambda page: [bad])
    with pytest.raises(RegionRefused, match="CONTEXT_MAPPING_CHARACTER_MISMATCH"):
        entry(result, index)


def test_native_offset_is_not_assumed_zero():
    result, index = fixture(spans=[replace(span(), source_offset=7)])
    ledger, record = entry(result, index)
    assert record.status == "CARRIED" and ledger.passes


def test_unreadable_cell_does_not_erase_readable_span():
    result, index = fixture(spans=[span()], cells=[cell(text=None)])
    groups = groups_for(result, index)
    assert cell_inventory(result, groups, source_id=SOURCE)[0].readability == "UNREADABLE"
    assert text_ledger(result, index, groups, source_id=SOURCE).entries[0].status == "CARRIED"


def test_equal_strings_at_distinct_positions_keep_occurrences():
    result, index = fixture(spans=[span("first"), span("second", box=(20, 50, 50, 60))])
    ledger = text_ledger(result, index, groups_for(result, index), source_id=SOURCE)
    assert {e.span_id for e in ledger.entries} == {"first", "second"}
    assert ledger.summary()["counts"]["CARRIED"] == 2


def test_only_existing_furniture_classification_can_exclude():
    result, index = fixture(spans=[span(text="1")])
    vid = index.regions[0].members[0].member_id
    result = replace(result, reading=SimpleNamespace(source_id=SOURCE, furniture=(vid,)))
    index.regions = ()
    ledger, record = entry(result, index)
    assert record.status == "EXCLUDED" and record.rule == "reader-furniture/1" and ledger.passes
    result = replace(result, reading=SimpleNamespace(source_id=SOURCE, furniture=()))
    assert entry(result, index)[1].status == "NOT_CARRIED"


def test_all_functions_are_deterministic_and_leave_inputs_unchanged():
    result, index = fixture(spans=[span()], cells=[cell(text=None)])
    before = repr(result), repr(index)

    def output():
        groups = groups_for(result, index)
        return json.dumps(
            [
                *[asdict(g) for g in groups],
                asdict(text_ledger(result, index, groups, source_id=SOURCE)),
                *[asdict(e) for e in cell_inventory(result, groups, source_id=SOURCE)],
            ],
            sort_keys=True,
        )

    assert output() == output()
    assert before == (repr(result), repr(index))


@pytest.mark.parametrize(
    "policy",
    [
        {"max_group_members": 0},
        {"max_group_members": True},
        {"max_group_cells": -1},
        {"page_scale_fraction": float("nan")},
        {"page_scale_fraction": 2},
    ],
)
def test_invalid_limits_are_refused(policy):
    with pytest.raises(RegionRefused, match="INVALID_CONTEXT_GROUP_POLICY"):
        ContextGroupPolicy(**policy)


def test_identity_checks_apply_to_evidence_and_index():
    result, index = fixture()
    for changed in ("argument", "index", "evidence"):
        if changed == "argument":
            with pytest.raises(RegionRefused, match="SOURCE_MISMATCH"):
                context_groups(result, index, source_id="other")
        elif changed == "index":
            index.source_id = "other"
            with pytest.raises(RegionRefused, match="SOURCE_MISMATCH"):
                groups_for(result, index)
            index.source_id = SOURCE
        else:
            page = result.document.pages[0]
            t = page.tables[0]
            t = replace(t, evidence=tuple(replace(e, source_id="other") for e in t.evidence))
            result = replace(
                result, document=replace(result.document, pages=(replace(page, tables=(t,)),))
            )
            with pytest.raises(RegionRefused, match="SOURCE_MISMATCH"):
                groups_for(result, index)


def test_omitted_candidate_fails_document_coverage_gate():
    result, index = fixture()
    with pytest.raises(RegionRefused, match="CONTEXT_REFUSAL_NOT_GROUPED"):
        text_ledger(result, index, (), source_id=SOURCE)


def test_missing_cell_assignment_is_refused():
    result, index = fixture(cells=[cell("a"), cell("b")])
    groups = groups_for(result, index, max_group_cells=1)
    with pytest.raises(RegionRefused, match="CONTEXT_CELL_NOT_ASSIGNED"):
        cell_inventory(result, groups[:1], source_id=SOURCE)


@pytest.mark.parametrize(
    "shape", [None, "gap", "overlap", "merged", "empty-row", "missing-geometry"]
)
def test_s0_canonical_snapshots_unchanged_for_existing_refusal_fixtures(shape):
    from .test_region_table_refusals import document_with_table

    result = explore(document_with_table(shape))
    index = regions_from_explorer(result)
    before = json.dumps(
        {"regions": [asdict(r) for r in index.regions], "refusals": index.refusal_report()},
        sort_keys=True,
    )
    groups = context_groups(result, index, source_id=result.document.source_id)
    cell_inventory(result, groups, source_id=result.document.source_id)
    text_ledger(result, index, groups, source_id=result.document.source_id)
    after = regions_from_explorer(result)
    assert (
        json.dumps(
            {"regions": [asdict(r) for r in index.regions], "refusals": index.refusal_report()},
            sort_keys=True,
        )
        == before
    )
    assert (
        json.dumps(
            {"regions": [asdict(r) for r in after.regions], "refusals": after.refusal_report()},
            sort_keys=True,
        )
        == before
    )


@pytest.mark.parametrize("columns", [12, 13, 14, 15])
def test_s0_existing_multi_page_table_fixture(columns):
    from .fixtures_explorer_slice4 import realistic_table, TableGeometry

    result = explore(realistic_table(TableGeometry(columns, 100, 32, 60, 45)).document)
    index = regions_from_explorer(result)
    before = [asdict(r) for r in index.regions], index.refusal_report()
    groups = context_groups(result, index, source_id=result.document.source_id)
    ledger = text_ledger(result, index, groups, source_id=result.document.source_id)
    assert len(ledger.entries) == sum(len(p.spans) for p in result.document.pages)
    assert ([asdict(r) for r in index.regions], index.refusal_report()) == before


@pytest.mark.parametrize(
    "text,cell_text,expected",
    [("e\u0301", "é", "NOT_CARRIED"), ("e\u0301", "e\u0301", "CARRIED_EXACT")],
)
def test_exact_means_original_code_points_not_normalization(text, cell_text, expected):
    result, index = fixture(spans=[span(text=text)], cells=[cell(text=cell_text)])
    index.regions = ()
    assert entry(result, index)[1].status == expected


def test_stale_member_geometry_is_refused():
    result, index = fixture(spans=[span()])
    r = index.regions[0]
    index.regions = (replace(r, members=(replace(r.members[0], bbox=(0, 0, 10, 10)),)),)
    with pytest.raises(RegionRefused, match="STALE_CONTEXT_MEMBER_GEOMETRY_OR_REFS"):
        entry(result, index)


def test_refusal_member_must_resolve_to_an_observation():
    result, index = fixture()
    index.refusals = (replace(index.refusals[0], member_ids=("not-observed",)),)
    with pytest.raises(RegionRefused, match="REFUSAL_SOURCE_MEMBER_MISSING"):
        groups_for(result, index)


def test_recovered_refusal_projects_to_each_native_candidate():
    result, index = fixture(tables=(table("first", [cell("a")]), table("second", [cell("b")])))
    index.refusals = (
        TableRegionRefusal(SOURCE, "composite", (1,), "TABLE_SOURCE_MEMBER_MISSING", ("a", "b")),
    )
    groups = groups_for(result, index)
    assert {g.candidate_id for g in groups} == {"first", "second"}
    assert {e.cell_id for e in cell_inventory(result, groups, source_id=SOURCE)} == {"a", "b"}


def test_corrupt_ledger_state_cannot_be_reported_as_passing():
    ledger, record = entry(*fixture(spans=[span()]))
    with pytest.raises(RegionRefused, match="INVALID_TEXT_ENTRY"):
        replace(record, status="UNKNOWN")
    with pytest.raises(RegionRefused, match="INVALID_TEXT_LEDGER"):
        replace(ledger, entries=(record, record))
    with pytest.raises(RegionRefused, match="TEXT_CARRIER_REQUIRED"):
        replace(record, carriers=())


def test_mixed_furniture_does_not_hide_missing_body_fragment():
    native = span(text="AB", box=(0, 10, 20, 20), glyphs=((0, 10, 10, 20), (10, 10, 20, 20)))
    result, index = fixture(spans=[native], cells=[cell(text="")], rules=[VerticalRule(10, 0, 30)])
    views = page_lines(result.document.pages[0])
    result = replace(
        result, reading=SimpleNamespace(source_id=SOURCE, furniture=(views[0].view_id,))
    )
    index.regions = ()
    ledger, record = entry(result, index)
    assert record.status == "NOT_CARRIED" and not ledger.passes


def test_one_source_member_may_have_line_and_caption_presentations():
    result, index = fixture(spans=[span()])
    original = index.regions[0]
    caption = replace(original.members[0], kind="CAPTION", flags=("CAPTION_PRESENTATION",))
    index.regions = (
        *index.regions,
        replace(
            original,
            region_id="caption-region",
            kind="FIGURE",
            members=(caption,),
            figure_id="synthetic-figure",
            figure_bbox=original.bbox,
            image_reason="NO_RASTER",
        ),
    )
    groups = groups_for(result, index)
    assert len(groups[0].member_ids) == 1
    assert len(groups[0].region_ids) == 2
    assert text_ledger(result, index, groups, source_id=SOURCE).passes


def test_shared_presentations_cannot_disagree_on_source_text():
    result, index = fixture(spans=[span()])
    original = index.regions[0]
    caption = replace(original.members[0], kind="CAPTION", text="different")
    index.regions = (
        *index.regions,
        replace(
            original,
            region_id="caption-region",
            kind="FIGURE",
            members=(caption,),
            figure_id="synthetic-figure",
            figure_bbox=original.bbox,
            image_reason="NO_RASTER",
        ),
    )
    with pytest.raises(RegionRefused, match="CONFLICTING_CONTEXT_MEMBER"):
        groups_for(result, index)
