"""Geometry-only connectivity views over native table cell observations.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The canonical table reader remains unchanged.  This module exposes what native
cell rectangles establish, including holes and conflicts, without asserting that
the observation is a semantic table, that a header is authoritative, or that a
value applies to any model or requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..harvest.region_types import RegionRefused, check_box, identity

Box = tuple[float, float, float, float]
IndexRange = tuple[int, int]
TileCoordinate = tuple[int, int]
RULE = "table-grid-view/1"
COMPONENT_RULE = "component/1"
HEADER_RULE = "header-proposal/1"
TILE_STATES = frozenset({"UNRESOLVED", "OWNED", "CONFLICT"})
CELL_STATES = frozenset({"READABLE", "READABLE_EMPTY", "UNREADABLE"})
VIEW_STATES = frozenset({"COMPLETE", "PARTIAL", "NONE"})


def _valid_range(value: IndexRange | None) -> bool:
    return (
        value is None
        or isinstance(value, tuple)
        and len(value) == 2
        and all(type(n) is int for n in value)
        and 0 <= value[0] < value[1]
    )


@dataclass(frozen=True)
class GridCell:
    """One native cell observation, retained once with its full scope."""

    cell_id: str
    text: str | None
    readability: str
    geometry_kind: str
    flags: tuple[str, ...]
    original_bbox: Box
    rounded_bbox: Box | None
    x_range: IndexRange | None
    y_range: IndexRange | None
    source_spans: tuple[str, ...]

    def __post_init__(self):
        if (
            not self.cell_id
            or self.readability not in CELL_STATES
            or self.geometry_kind not in {"text_box", "cell_box"}
            or not isinstance(self.flags, tuple)
            or not isinstance(self.source_spans, tuple)
            or any(not isinstance(value, str) or not value for value in self.source_spans)
            or not _valid_range(self.x_range)
            or not _valid_range(self.y_range)
            or (self.x_range is None) != (self.y_range is None)
        ):
            raise RegionRefused("INVALID_GRID_CELL")
        check_box(self.original_bbox)
        if self.rounded_bbox is not None and (
            not isinstance(self.rounded_bbox, tuple)
            or len(self.rounded_bbox) != 4
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in self.rounded_bbox
            )
        ):
            raise RegionRefused("INVALID_GRID_CELL")
        if self.rounded_bbox is not None and self.x_range is not None:
            check_box(self.rounded_bbox)
        if self.readability == "UNREADABLE" and self.text is not None:
            raise RegionRefused("INVALID_GRID_CELL_READABILITY")
        if self.readability == "READABLE_EMPTY" and self.text != "":
            raise RegionRefused("INVALID_GRID_CELL_READABILITY")
        if self.readability == "READABLE" and (not isinstance(self.text, str) or self.text == ""):
            raise RegionRefused("INVALID_GRID_CELL_READABILITY")


@dataclass(frozen=True)
class GridCellIssue:
    cell_id: str
    reason: str

    def __post_init__(self):
        if not self.cell_id or self.reason not in {
            "MISSING_CELL_GEOMETRY",
            "GEOMETRY_KIND_UNAVAILABLE",
            "DEGENERATE_AFTER_ROUNDING",
        }:
            raise RegionRefused("INVALID_GRID_CELL_ISSUE")


@dataclass(frozen=True)
class GridTile:
    """One elementary rectangle; indices are (x, y), storage is row-major."""

    x_index: int
    y_index: int
    bbox: Box
    owners: tuple[str, ...]
    state: str

    def __post_init__(self):
        if (
            type(self.x_index) is not int
            or type(self.y_index) is not int
            or min(self.x_index, self.y_index) < 0
            or self.state not in TILE_STATES
            or self.owners != tuple(sorted(set(self.owners)))
            or (self.state == "UNRESOLVED") != (len(self.owners) == 0)
            or (self.state == "OWNED") != (len(self.owners) == 1)
            or (self.state == "CONFLICT") != (len(self.owners) >= 2)
        ):
            raise RegionRefused("INVALID_GRID_TILE")
        check_box(self.bbox)


@dataclass(frozen=True)
class ComponentCell:
    cell_id: str
    tiles: tuple[TileCoordinate, ...]
    flags: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not self.cell_id
            or not self.tiles
            or self.tiles != tuple(sorted(set(self.tiles), key=lambda tile: (tile[1], tile[0])))
            or set(self.flags) - {"EXTENDS_BEYOND_COMPONENT"}
        ):
            raise RegionRefused("INVALID_COMPONENT_CELL")


@dataclass(frozen=True)
class GridComponent:
    component_id: str
    rule: str
    tiles: tuple[TileCoordinate, ...]
    cells: tuple[ComponentCell, ...]

    def __post_init__(self):
        if (
            not self.component_id
            or self.rule != COMPONENT_RULE
            or not self.tiles
            or self.tiles != tuple(sorted(set(self.tiles), key=lambda tile: (tile[1], tile[0])))
            or not isinstance(self.cells, tuple)
            or any(not isinstance(cell, ComponentCell) for cell in self.cells)
            or len({cell.cell_id for cell in self.cells}) != len(self.cells)
        ):
            raise RegionRefused("INVALID_GRID_COMPONENT")


@dataclass(frozen=True)
class HeaderProposal:
    proposal_id: str
    rule: str
    component_id: str
    y_index: int
    cell_ids: tuple[str, ...]
    evidence_tiles: tuple[TileCoordinate, ...]

    def __post_init__(self):
        if (
            not self.proposal_id
            or self.rule != HEADER_RULE
            or not self.component_id
            or type(self.y_index) is not int
            or self.y_index < 0
            or not self.cell_ids
            or len(set(self.cell_ids)) != len(self.cell_ids)
            or not self.evidence_tiles
        ):
            raise RegionRefused("INVALID_HEADER_PROPOSAL")


@dataclass(frozen=True)
class GridView:
    candidate_id: str
    source_id: str
    page: int
    rule: str
    x_edges: tuple[float, ...]
    y_edges: tuple[float, ...]
    tiles: tuple[tuple[GridTile, ...], ...]
    cells: tuple[GridCell, ...]
    unlocated_cells: tuple[GridCellIssue, ...]
    degenerate_cells: tuple[GridCellIssue, ...]
    components: tuple[GridComponent, ...]
    header_proposals: tuple[HeaderProposal, ...]
    status: str

    def __post_init__(self):
        if (
            not self.candidate_id
            or not self.source_id
            or type(self.page) is not int
            or self.page < 1
            or self.rule != RULE
            or self.status not in VIEW_STATES
            or self.x_edges != tuple(sorted(set(self.x_edges)))
            or self.y_edges != tuple(sorted(set(self.y_edges)))
            or any(not math.isfinite(value) for value in (*self.x_edges, *self.y_edges))
            or len({cell.cell_id for cell in self.cells}) != len(self.cells)
            or any(not isinstance(cell, GridCell) for cell in self.cells)
            or any(not isinstance(issue, GridCellIssue) for issue in self.unlocated_cells)
            or any(not isinstance(issue, GridCellIssue) for issue in self.degenerate_cells)
            or any(not isinstance(component, GridComponent) for component in self.components)
            or any(not isinstance(proposal, HeaderProposal) for proposal in self.header_proposals)
        ):
            raise RegionRefused("INVALID_GRID_VIEW")
        width = max(0, len(self.x_edges) - 1)
        height = max(0, len(self.y_edges) - 1)
        if len(self.tiles) != height or any(len(row) != width for row in self.tiles):
            raise RegionRefused("INVALID_GRID_SHAPE")
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if tile.x_index != x or tile.y_index != y:
                    raise RegionRefused("INVALID_GRID_TILE_ORDER")
        cell_by_id = {cell.cell_id: cell for cell in self.cells}
        unlocated = tuple(issue.cell_id for issue in self.unlocated_cells)
        degenerate = tuple(issue.cell_id for issue in self.degenerate_cells)
        if (
            len(set(unlocated)) != len(unlocated)
            or len(set(degenerate)) != len(degenerate)
            or set(unlocated) & set(degenerate)
            or not (set(unlocated) | set(degenerate)) <= cell_by_id.keys()
        ):
            raise RegionRefused("INVALID_GRID_CELL_ISSUES")
        for cell_id, cell in cell_by_id.items():
            if cell_id in unlocated:
                if cell.rounded_bbox is not None or cell.x_range is not None:
                    raise RegionRefused("INVALID_UNLOCATED_GRID_CELL")
            elif cell_id in degenerate:
                if cell.rounded_bbox is None or cell.x_range is not None:
                    raise RegionRefused("INVALID_DEGENERATE_GRID_CELL")
                if (
                    cell.rounded_bbox[2] > cell.rounded_bbox[0]
                    and cell.rounded_bbox[3] > cell.rounded_bbox[1]
                ):
                    raise RegionRefused("INVALID_DEGENERATE_GRID_CELL")
            elif (
                cell.rounded_bbox is None
                or cell.x_range is None
                or cell.x_range[1] > width
                or cell.y_range[1] > height
            ):
                raise RegionRefused("INVALID_LOCATED_GRID_CELL")
        for row in self.tiles:
            for tile in row:
                expected = tuple(
                    sorted(
                        cell.cell_id
                        for cell in self.cells
                        if cell.x_range is not None
                        and cell.x_range[0] <= tile.x_index < cell.x_range[1]
                        and cell.y_range[0] <= tile.y_index < cell.y_range[1]
                    )
                )
                if tile.owners != expected:
                    raise RegionRefused("INVALID_GRID_TILE_OWNERS")
        owned = {
            (tile.x_index, tile.y_index)
            for row in self.tiles
            for tile in row
            if tile.state == "OWNED"
        }
        tile_by_coordinate = {
            (tile.x_index, tile.y_index): tile for row in self.tiles for tile in row
        }
        seen = set()
        for component in self.components:
            component_tiles = set(component.tiles)
            if (
                component.component_id
                != identity(COMPONENT_RULE, self.candidate_id, component.tiles)
                or not component_tiles <= owned
                or seen & component_tiles
            ):
                raise RegionRefused("INVALID_GRID_COMPONENT_SCOPE")
            pending = [component.tiles[0]]
            connected = set()
            while pending:
                current = pending.pop()
                if current in connected or current not in component_tiles:
                    continue
                connected.add(current)
                x, y = current
                pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
            if connected != component_tiles:
                raise RegionRefused("INVALID_GRID_COMPONENT_CONNECTIVITY")
            expected_cells = {
                tile_by_coordinate[coordinate].owners[0] for coordinate in component_tiles
            }
            if {link.cell_id for link in component.cells} != expected_cells:
                raise RegionRefused("INVALID_GRID_COMPONENT_CELL")
            seen.update(component_tiles)
            for link in component.cells:
                cell = cell_by_id.get(link.cell_id)
                if cell is None:
                    raise RegionRefused("INVALID_GRID_COMPONENT_CELL")
                scope = set(_cell_tiles(cell))
                shared = tuple(sorted(scope & component_tiles, key=lambda tile: (tile[1], tile[0])))
                flags = ("EXTENDS_BEYOND_COMPONENT",) if not scope <= component_tiles else ()
                if link.tiles != shared or link.flags != flags:
                    raise RegionRefused("INVALID_GRID_COMPONENT_CELL")
        if seen != owned:
            raise RegionRefused("INVALID_GRID_COMPONENT_COVERAGE")
        expected_proposals = tuple(
            proposal
            for component in self.components
            for proposal in (_header_proposal(self.candidate_id, component, self.cells),)
            if proposal is not None
        )
        if self.header_proposals != expected_proposals:
            raise RegionRefused("INVALID_HEADER_PROPOSALS")
        flat = tuple(tile for row in self.tiles for tile in row)
        expected_status = (
            "NONE"
            if not self.components
            else (
                "COMPLETE"
                if len(self.components) == 1
                and flat
                and all(tile.state == "OWNED" for tile in flat)
                and not self.unlocated_cells
                and not self.degenerate_cells
                else "PARTIAL"
            )
        )
        if self.status != expected_status:
            raise RegionRefused("INVALID_GRID_STATUS")


def _round_box(box: Box) -> tuple[float, float, float, float]:
    return tuple(round(value, 3) for value in box)


def _readability(text: str | None) -> str:
    return "UNREADABLE" if text is None else "READABLE_EMPTY" if text == "" else "READABLE"


def _source(observation, source_id: str):
    if not isinstance(source_id, str) or not source_id or not observation.table_id:
        raise RegionRefused("GRID_SOURCE_REQUIRED")
    if type(observation.page) is not int or observation.page < 1 or not observation.evidence:
        raise RegionRefused("GRID_SOURCE_EVIDENCE_REQUIRED")
    if any(
        evidence.source_id != source_id or evidence.page != observation.page
        for evidence in observation.evidence
    ):
        raise RegionRefused("GRID_SOURCE_MISMATCH")
    cells = tuple(cell for row in observation.cell_rows for cell in row)
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise RegionRefused("DUPLICATE_GRID_CELL")
    evidence_by_id = {}
    for evidence in observation.evidence:
        if evidence.span_id in evidence_by_id:
            raise RegionRefused("DUPLICATE_GRID_CELL_EVIDENCE")
        evidence_by_id[evidence.span_id] = evidence
    for cell in cells:
        evidence = evidence_by_id.get(cell.cell_id)
        if evidence is None:
            raise RegionRefused("GRID_CELL_EVIDENCE_MISSING")
        if evidence.literal != (cell.text or "") or evidence.bbox != cell.bbox:
            raise RegionRefused("GRID_CELL_EVIDENCE_MISMATCH")
    return cells


def _located(cell) -> bool:
    return cell.geometry_kind == "cell_box" and "MISSING_CELL_GEOMETRY" not in cell.flags


def _range(box: Box, x_edges: tuple[float, ...], y_edges: tuple[float, ...]):
    return (
        (x_edges.index(box[0]), x_edges.index(box[2])),
        (y_edges.index(box[1]), y_edges.index(box[3])),
    )


def _cell_tiles(cell: GridCell) -> tuple[TileCoordinate, ...]:
    if cell.x_range is None or cell.y_range is None:
        return ()
    return tuple(
        (x, y)
        for y in range(cell.y_range[0], cell.y_range[1])
        for x in range(cell.x_range[0], cell.x_range[1])
    )


def _components(candidate_id: str, tiles, cells):
    owned = {
        (tile.x_index, tile.y_index): tile.owners[0]
        for row in tiles
        for tile in row
        if tile.state == "OWNED"
    }
    remaining = set(owned)
    components = []
    cells_by_id = {cell.cell_id: cell for cell in cells}
    while remaining:
        seed = min(remaining, key=lambda tile: (tile[1], tile[0]))
        pending, found = [seed], set()
        while pending:
            current = pending.pop()
            if current in found or current not in remaining:
                continue
            found.add(current)
            x, y = current
            pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
        remaining.difference_update(found)
        ordered = tuple(sorted(found, key=lambda tile: (tile[1], tile[0])))
        component_cells = []
        for cell_id in sorted({owned[tile] for tile in found}):
            scope = set(_cell_tiles(cells_by_id[cell_id]))
            shared = tuple(sorted(scope & found, key=lambda tile: (tile[1], tile[0])))
            flags = ("EXTENDS_BEYOND_COMPONENT",) if not scope <= found else ()
            component_cells.append(ComponentCell(cell_id, shared, flags))
        components.append(
            GridComponent(
                identity(COMPONENT_RULE, candidate_id, ordered),
                COMPONENT_RULE,
                ordered,
                tuple(component_cells),
            )
        )
    return tuple(components)


def _header_proposal(candidate_id: str, component: GridComponent, cells):
    by_id = {cell.cell_id: cell for cell in cells}
    top = min(y for _, y in component.tiles)
    top_tiles = tuple(tile for tile in component.tiles if tile[1] == top)
    component_tiles = set(component.tiles)
    owners = []
    for tile in top_tiles:
        owner = next((entry.cell_id for entry in component.cells if tile in entry.tiles), None)
        if owner is None:
            return None
        cell = by_id[owner]
        scope = set(_cell_tiles(cell))
        if (
            cell.readability != "READABLE"
            or cell.x_range is None
            or cell.x_range[1] - cell.x_range[0] != 1
            or not scope <= component_tiles
        ):
            return None
        owners.append(owner)
    cell_ids = tuple(dict.fromkeys(owners))
    return HeaderProposal(
        identity(HEADER_RULE, candidate_id, component.component_id, top, cell_ids, top_tiles),
        HEADER_RULE,
        component.component_id,
        top,
        cell_ids,
        top_tiles,
    )


def grid_view(observation, *, source_id: str) -> GridView:
    """Build a deterministic geometry view without changing the observation.

    The nested tile matrix is stored as ``tiles[y][x]``.  Coordinates and
    half-open ranges describe rounded geometry only; original boxes remain on
    each cell.  Missing and collapsed geometry stays in the cell inventory and
    owns no tile.
    """
    native = _source(observation, source_id)
    prepared = []
    unlocated = []
    degenerate = []
    for cell in native:
        rounded = _round_box(cell.bbox)
        if not _located(cell):
            reason = (
                "MISSING_CELL_GEOMETRY"
                if "MISSING_CELL_GEOMETRY" in cell.flags
                else "GEOMETRY_KIND_UNAVAILABLE"
            )
            prepared.append((cell, None, None, reason))
            unlocated.append(GridCellIssue(cell.cell_id, reason))
        elif rounded[2] <= rounded[0] or rounded[3] <= rounded[1]:
            prepared.append((cell, None, rounded, "DEGENERATE_AFTER_ROUNDING"))
            degenerate.append(GridCellIssue(cell.cell_id, "DEGENERATE_AFTER_ROUNDING"))
        else:
            prepared.append((cell, rounded, rounded, None))
    x_edges = tuple(
        sorted(
            {value for _, usable, _, _ in prepared if usable for value in (usable[0], usable[2])}
        )
    )
    y_edges = tuple(
        sorted(
            {value for _, usable, _, _ in prepared if usable for value in (usable[1], usable[3])}
        )
    )
    cells = []
    for native_cell, usable, rounded, reason in prepared:
        xr, yr = _range(usable, x_edges, y_edges) if usable else (None, None)
        cells.append(
            GridCell(
                native_cell.cell_id,
                native_cell.text,
                _readability(native_cell.text),
                native_cell.geometry_kind,
                native_cell.flags,
                native_cell.bbox,
                rounded,
                xr,
                yr,
                native_cell.source_spans,
            )
        )
    cells = tuple(sorted(cells, key=lambda cell: cell.cell_id))
    tiles = []
    for y in range(max(0, len(y_edges) - 1)):
        row = []
        for x in range(max(0, len(x_edges) - 1)):
            owners = tuple(
                sorted(
                    cell.cell_id
                    for cell in cells
                    if cell.x_range is not None
                    and cell.x_range[0] <= x < cell.x_range[1]
                    and cell.y_range[0] <= y < cell.y_range[1]
                )
            )
            state = "UNRESOLVED" if not owners else "OWNED" if len(owners) == 1 else "CONFLICT"
            row.append(
                GridTile(
                    x, y, (x_edges[x], y_edges[y], x_edges[x + 1], y_edges[y + 1]), owners, state
                )
            )
        tiles.append(tuple(row))
    tiles = tuple(tiles)
    components = _components(observation.table_id, tiles, cells)
    proposals = tuple(
        proposal
        for component in components
        for proposal in (_header_proposal(observation.table_id, component, cells),)
        if proposal is not None
    )
    flat = tuple(tile for row in tiles for tile in row)
    if not components:
        status = "NONE"
    elif (
        len(components) == 1
        and flat
        and all(tile.state == "OWNED" for tile in flat)
        and not unlocated
        and not degenerate
    ):
        status = "COMPLETE"
    else:
        status = "PARTIAL"
    return GridView(
        observation.table_id,
        source_id,
        observation.page,
        RULE,
        x_edges,
        y_edges,
        tiles,
        cells,
        tuple(sorted(unlocated, key=lambda issue: issue.cell_id)),
        tuple(sorted(degenerate, key=lambda issue: issue.cell_id)),
        components,
        proposals,
        status,
    )
