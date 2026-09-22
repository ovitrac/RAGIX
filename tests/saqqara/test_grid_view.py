"""Synthetic geometry falsifiers for the optional native-cell grid view.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import json

import pytest

from ragix_kernels.harvest.region_types import RegionRefused
from ragix_kernels.saqqara.census import Evidence, TableObservation
from ragix_kernels.saqqara.grid_view import grid_view
from ragix_kernels.saqqara.table_views import TableCell

SOURCE = "synthetic-grid"


def cell(ident, text, box, *, flags=(), geometry="cell_box", spans=()):
    return TableCell(ident, text, box, spans, flags, geometry)


def observation(cells, *, ident="candidate", source=SOURCE, evidence_cells=None):
    evidence_cells = tuple(cells if evidence_cells is None else evidence_cells)
    evidence = tuple(
        Evidence(source, 1, item.cell_id, 0, len(item.text or ""), item.text or "", item.bbox)
        for item in evidence_cells
    )
    return TableObservation(
        ident,
        1,
        ("observed",),
        (),
        evidence,
        cell_rows=tuple((item,) for item in cells),
    )


def tile(view, x, y):
    return view.tiles[y][x]


def states(view):
    return tuple(tuple(entry.state for entry in row) for row in view.tiles)


def cells_by_id(view):
    return {entry.cell_id: entry for entry in view.cells}


def test_merged_value_is_one_observation_with_full_scope():
    cells = (
        cell("h-label", "label", (0, 0, 10, 10)),
        cell("h-a", "A", (10, 0, 20, 10)),
        cell("h-b", "B", (20, 0, 30, 10)),
        cell("h-c", "C", (30, 0, 40, 10)),
        cell("v-label", "setting", (0, 10, 10, 20)),
        cell("v-merged", "shared", (10, 10, 40, 20)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    merged = cells_by_id(view)["v-merged"]
    assert merged.x_range == (1, 4) and merged.y_range == (1, 2)
    assert [entry.cell_id for entry in view.cells].count("v-merged") == 1
    assert view.status == "COMPLETE" and len(view.components) == 1
    assert len(view.header_proposals) == 1
    assert view.header_proposals[0].cell_ids == ("h-label", "h-a", "h-b", "h-c")


def test_cross_row_overlap_is_conflict_and_breaks_connectivity():
    cells = (
        cell("top", "top", (0, 0, 10, 10)),
        cell("tall", "first", (0, 10, 10, 30)),
        cell("late", "second", (0, 20, 10, 30)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    assert tile(view, 0, 2).state == "CONFLICT"
    assert tile(view, 0, 2).owners == ("late", "tall")
    assert {entry.cell_id for entry in view.cells} == {"top", "tall", "late"}
    assert view.status == "PARTIAL" and len(view.components) == 1
    tall = next(entry for entry in view.components[0].cells if entry.cell_id == "tall")
    assert tall.flags == ("EXTENDS_BEYOND_COMPONENT",)


def test_hole_is_unresolved_while_other_cells_survive():
    view = grid_view(
        observation(
            (
                cell("left", "L", (0, 0, 10, 10)),
                cell("right", "R", (10, 10, 20, 20)),
            )
        ),
        source_id=SOURCE,
    )
    assert states(view) == (("OWNED", "UNRESOLVED"), ("UNRESOLVED", "OWNED"))
    assert view.status == "PARTIAL" and len(view.components) == 2
    assert {entry.text for entry in view.cells} == {"L", "R"}


@pytest.mark.parametrize(
    "text,readability",
    [(None, "UNREADABLE"), ("", "READABLE_EMPTY"), (" ", "READABLE"), ("value", "READABLE")],
)
def test_unreadable_and_empty_cells_still_own_geometry(text, readability):
    view = grid_view(observation((cell("only", text, (0, 0, 10, 4)),)), source_id=SOURCE)
    assert cells_by_id(view)["only"].readability == readability
    assert tile(view, 0, 0).owners == ("only",)
    assert view.status == "COMPLETE"


def test_missing_geometry_is_retained_but_owns_no_tile():
    cells = (
        cell("located", "value", (0, 0, 10, 10)),
        cell("missing", "retained", (0, 0, 100, 100), flags=("MISSING_CELL_GEOMETRY",)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    missing = cells_by_id(view)["missing"]
    assert missing.original_bbox == (0, 0, 100, 100)
    assert missing.rounded_bbox is None and missing.x_range is None and missing.y_range is None
    assert [asdict(issue) for issue in view.unlocated_cells] == [
        {"cell_id": "missing", "reason": "MISSING_CELL_GEOMETRY"}
    ]
    assert tile(view, 0, 0).owners == ("located",)
    assert view.status == "PARTIAL"


def test_all_missing_geometry_yields_none_without_dropping_cells():
    cells = (
        cell("a", None, (0, 0, 100, 100), flags=("MISSING_CELL_GEOMETRY",)),
        cell("b", "text", (0, 0, 100, 100), flags=("MISSING_CELL_GEOMETRY",)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    assert view.status == "NONE" and view.x_edges == view.y_edges == () and view.tiles == ()
    assert {entry.cell_id for entry in view.cells} == {"a", "b"}
    assert {issue.cell_id for issue in view.unlocated_cells} == {"a", "b"}


def test_non_native_geometry_is_retained_as_unlocated():
    view = grid_view(
        observation((cell("text-box", "text", (0, 0, 10, 10), geometry="text_box"),)),
        source_id=SOURCE,
    )
    assert view.status == "NONE"
    assert view.unlocated_cells[0].reason == "GEOMETRY_KIND_UNAVAILABLE"


def test_source_identity_and_cell_evidence_fail_closed():
    source = observation((cell("a", "A", (0, 0, 10, 10)),))
    with pytest.raises(RegionRefused, match="GRID_SOURCE_MISMATCH"):
        grid_view(source, source_id="other")
    missing = observation(
        (cell("a", "A", (0, 0, 10, 10)), cell("b", "B", (10, 0, 20, 10))),
        evidence_cells=(cell("a", "A", (0, 0, 10, 10)),),
    )
    with pytest.raises(RegionRefused, match="GRID_CELL_EVIDENCE_MISSING"):
        grid_view(missing, source_id=SOURCE)


def test_duplicate_cell_identity_is_refused():
    duplicate = cell("same", "A", (0, 0, 10, 10))
    source = observation((duplicate, replace(duplicate, text="B", bbox=(10, 0, 20, 10))))
    with pytest.raises(RegionRefused, match="DUPLICATE_GRID_CELL"):
        grid_view(source, source_id=SOURCE)


def test_identical_boxes_remain_conflicting_observations():
    source = observation((cell("a", "same", (0, 0, 10, 10)), cell("b", "same", (0, 0, 10, 10))))
    view = grid_view(source, source_id=SOURCE)
    assert tile(view, 0, 0).state == "CONFLICT"
    assert tile(view, 0, 0).owners == ("a", "b")
    assert view.status == "NONE" and view.components == ()


def test_l_shape_is_one_connected_nonrectangular_component():
    cells = (
        cell("top-left", "A", (0, 0, 10, 10)),
        cell("top-right", "B", (10, 0, 20, 10)),
        cell("bottom-left", "C", (0, 10, 10, 20)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    assert len(view.components) == 1
    assert view.components[0].tiles == ((0, 0), (1, 0), (0, 1))
    assert tile(view, 1, 1).state == "UNRESOLVED"


def test_t_shape_component_identity_and_order_are_deterministic():
    cells = (
        cell("a", "A", (0, 0, 10, 10)),
        cell("b", "B", (10, 0, 20, 10)),
        cell("c", "C", (20, 0, 30, 10)),
        cell("d", "D", (10, 10, 20, 20)),
    )
    source = observation(cells)
    one = grid_view(source, source_id=SOURCE)
    two = grid_view(source, source_id=SOURCE)
    assert one.components == two.components
    assert one.components[0].tiles == ((0, 0), (1, 0), (2, 0), (1, 1))


def test_cell_crossing_owned_and_conflicting_tiles_is_not_split():
    source = observation((cell("wide", "W", (0, 0, 20, 10)), cell("overlap", "X", (10, 0, 20, 10))))
    view = grid_view(source, source_id=SOURCE)
    assert len(view.components) == 1
    assert tile(view, 0, 0).state == "OWNED" and tile(view, 1, 0).state == "CONFLICT"
    assert [entry.cell_id for entry in view.cells].count("wide") == 1
    link = view.components[0].cells[0]
    assert link.cell_id == "wide" and link.tiles == ((0, 0),)
    assert link.flags == ("EXTENDS_BEYOND_COMPONENT",)


def test_empty_geometric_first_row_blocks_header_without_fallthrough():
    cells = (
        cell("empty", "", (0, 0, 10, 4)),
        cell("a", "A", (0, 4, 10, 14)),
        cell("b", "B", (10, 4, 20, 14)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    assert len(view.components) == 1
    assert view.header_proposals == ()


def test_merged_first_row_is_not_proposed_as_header():
    cells = (
        cell("merged", "A B", (0, 0, 20, 10)),
        cell("a", "A", (0, 10, 10, 20)),
        cell("b", "B", (10, 10, 20, 20)),
    )
    view = grid_view(observation(cells), source_id=SOURCE)
    assert view.header_proposals == ()


def test_one_column_candidate_claims_only_geometry_and_a_header_hypothesis():
    view = grid_view(
        observation((cell("caption", "drawing caption", (0, 0, 20, 10)),)),
        source_id=SOURCE,
    )
    assert view.status == "COMPLETE" and view.rule == "table-grid-view/1"
    assert view.header_proposals[0].rule == "header-proposal/1"
    assert not hasattr(view, "is_table") and not hasattr(view, "applicability")


def test_rounding_collapse_and_boundary_straddle_are_distinct():
    collapsed = grid_view(
        observation((cell("tiny", "T", (0.0, 0.0, 0.0004, 1.0)),)),
        source_id=SOURCE,
    )
    assert collapsed.status == "NONE"
    assert collapsed.degenerate_cells[0].reason == "DEGENERATE_AFTER_ROUNDING"
    assert cells_by_id(collapsed)["tiny"].rounded_bbox == (0.0, 0.0, 0.0, 1.0)
    straddled = grid_view(
        observation((cell("tiny", "T", (0.0004, 0.0, 0.0008, 1.0)),)),
        source_id=SOURCE,
    )
    assert straddled.status == "COMPLETE" and straddled.degenerate_cells == ()
    assert cells_by_id(straddled)["tiny"].rounded_bbox == (0.0, 0.0, 0.001, 1.0)


def test_edge_rounding_is_applied_once_and_owner_ids_are_sorted():
    source = observation(
        (
            cell("z", "Z", (0.00049, 0, 1.00049, 1)),
            cell("a", "A", (0.00049, 0, 1.00049, 1)),
        )
    )
    view = grid_view(source, source_id=SOURCE)
    assert view.x_edges == (0.0, 1.0)
    assert tile(view, 0, 0).owners == ("a", "z")


def test_grid_view_is_byte_stable_and_does_not_mutate_observation():
    source = observation((cell("a", "A", (0, 0, 10, 10)), cell("b", None, (10, 0, 20, 10))))
    before = repr(source)
    one = json.dumps(asdict(grid_view(source, source_id=SOURCE)), sort_keys=True)
    two = json.dumps(asdict(grid_view(source, source_id=SOURCE)), sort_keys=True)
    assert one == two and repr(source) == before


@pytest.mark.parametrize("shape", ["gap", "overlap", "merged", "empty-row", "missing-geometry"])
def test_s0_grid_view_does_not_change_canonical_region_outputs(shape):
    from ragix_kernels.saqqara.explorer import explore
    from ragix_kernels.saqqara.renderable_regions import regions_from_explorer
    from .test_region_table_refusals import defective_table, document_with_table

    result = explore(document_with_table(shape))
    index = regions_from_explorer(result)
    before = json.dumps(
        {
            "regions": [asdict(region) for region in index.regions],
            "refusals": index.refusal_report(),
        },
        sort_keys=True,
    )
    view = grid_view(defective_table(shape), source_id="synthetic")
    assert view.candidate_id == "bad"
    after = regions_from_explorer(result)
    assert (
        json.dumps(
            {
                "regions": [asdict(region) for region in after.regions],
                "refusals": after.refusal_report(),
            },
            sort_keys=True,
        )
        == before
    )


def test_grid_view_rejects_corrupt_status_and_tile_ownership():
    view = grid_view(observation((cell("a", "A", (0, 0, 10, 10)),)), source_id=SOURCE)
    with pytest.raises(RegionRefused, match="INVALID_GRID_STATUS"):
        replace(view, status="PARTIAL")
    with pytest.raises(RegionRefused, match="INVALID_GRID_TILE"):
        replace(view.tiles[0][0], owners=())
    bad_tile = replace(view.tiles[0][0], owners=("ghost",))
    with pytest.raises(RegionRefused, match="INVALID_GRID_TILE_OWNERS"):
        replace(view, tiles=((bad_tile,),))


def test_grid_view_rejects_dropped_or_overlapping_component_coverage():
    view = grid_view(
        observation(
            (
                cell("left", "L", (0, 0, 10, 10)),
                cell("right", "R", (20, 0, 30, 10)),
            )
        ),
        source_id=SOURCE,
    )
    assert len(view.components) == 2
    with pytest.raises(RegionRefused, match="INVALID_GRID_COMPONENT_COVERAGE"):
        replace(view, components=view.components[:1], header_proposals=view.header_proposals[:1])
    duplicate = replace(
        view.components[1],
        component_id=view.components[0].component_id,
        tiles=view.components[0].tiles,
        cells=view.components[0].cells,
    )
    with pytest.raises(RegionRefused, match="INVALID_GRID_COMPONENT_SCOPE"):
        replace(view, components=(view.components[0], duplicate))


def test_grid_view_rejects_unlocated_cell_promoted_to_geometry():
    view = grid_view(
        observation((cell("missing", "text", (0, 0, 10, 10), flags=("MISSING_CELL_GEOMETRY",)),)),
        source_id=SOURCE,
    )
    promoted = replace(
        view.cells[0],
        rounded_bbox=(0, 0, 10, 10),
        x_range=(0, 1),
        y_range=(0, 1),
    )
    with pytest.raises(RegionRefused, match="INVALID_UNLOCATED_GRID_CELL"):
        replace(view, cells=(promoted,))


def test_cell_evidence_literal_and_box_must_match_observation():
    native = cell("a", "A", (0, 0, 10, 10))
    source = observation((native,))
    wrong_literal = replace(source.evidence[0], literal="B")
    with pytest.raises(RegionRefused, match="GRID_CELL_EVIDENCE_MISMATCH"):
        grid_view(replace(source, evidence=(wrong_literal,)), source_id=SOURCE)
    wrong_box = replace(source.evidence[0], bbox=(0, 0, 20, 10))
    with pytest.raises(RegionRefused, match="GRID_CELL_EVIDENCE_MISMATCH"):
        grid_view(replace(source, evidence=(wrong_box,)), source_id=SOURCE)
    duplicate = (source.evidence[0], source.evidence[0])
    with pytest.raises(RegionRefused, match="DUPLICATE_GRID_CELL_EVIDENCE"):
        grid_view(replace(source, evidence=duplicate), source_id=SOURCE)
