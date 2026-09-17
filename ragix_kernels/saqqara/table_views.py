"""Recurrence-filtered physical table blocks and traced column reconstruction.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, replace, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .census import PageDigest, TableObservation
from statistics import median
import re
import math
from .field_views import stable_id, union_box
from .privacy import TableStampLine, table_row_stamps, table_stamp_from_dict


@dataclass(frozen=True)
class TableCell:
    cell_id: str
    text: str | None
    bbox: tuple[float, float, float, float]
    source_spans: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    geometry_kind: str = "text_box"

    def __post_init__(self):
        if (
            not self.cell_id
            or self.geometry_kind not in {"text_box", "cell_box"}
            or self.text is not None
            and not isinstance(self.text, str)
            or len(self.bbox) != 4
            or any(
                not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                for v in self.bbox
            )
            or self.bbox[2] <= self.bbox[0]
            or self.bbox[3] <= self.bbox[1]
        ):
            raise ValueError("invalid physical table cell")


@dataclass(frozen=True)
class TablePolicy:
    x_tolerance_factor: float = 0.5
    minimum_rows: int = 2
    fallback_cell_width: float = 12
    version: str = "table-policy/0.1"

    def __post_init__(self):
        if (
            isinstance(self.x_tolerance_factor, bool)
            or not isinstance(self.x_tolerance_factor, (int, float))
            or not math.isfinite(self.x_tolerance_factor)
            or not 0 < self.x_tolerance_factor <= 1
            or type(self.minimum_rows) is not int
            or self.minimum_rows < 2
            or not math.isfinite(self.fallback_cell_width)
            or self.fallback_cell_width <= 0
            or self.version != "table-policy/0.1"
        ):
            raise ValueError("invalid table geometry policy")


@dataclass(frozen=True)
class TableDiagnostics:
    """Literal-free geometry counts at the exact refusal stage."""

    stage: str = "candidate"
    raw_row_cell_counts: tuple[int, ...] = ()
    retained_row_cell_counts: tuple[int, ...] = ()
    nonempty_row_cell_counts: tuple[int, ...] = ()
    unreadable_row_cell_counts: tuple[int, ...] = ()
    table_sized_unreadable_cells: int = 0
    unlocated_slot_count: int = 0
    route: str = "undetermined"
    full_height_rule_count: int = 0
    row_band_counts: tuple[int, ...] = ()
    inferred_band_count: int | None = None
    bands: tuple[tuple[float, float], ...] = ()
    mapped_band_support: tuple[int, ...] = ()
    mapped_nonempty_support: tuple[int, ...] = ()
    continuation_fragments: int = 0
    pooled_rows: int = 0
    pages_spanned: tuple[int, ...] = ()
    observed_band_nonempty_counts: tuple[int, ...] = ()
    populated_band_count: int | None = None

    def __post_init__(self):
        counts = (
            self.table_sized_unreadable_cells,
            self.unlocated_slot_count,
            self.full_height_rule_count,
            self.continuation_fragments,
            self.pooled_rows,
            *self.raw_row_cell_counts,
            *self.retained_row_cell_counts,
            *self.nonempty_row_cell_counts,
            *self.unreadable_row_cell_counts,
            *self.row_band_counts,
            *self.mapped_band_support,
            *self.mapped_nonempty_support,
            *self.observed_band_nonempty_counts,
        )
        if (
            self.stage
            not in {"candidate", "bands", "mapping", "header", "role", "continuation", "support"}
            or self.route
            not in {"undetermined", "grid", "row_supported_bands", "native_cell_bounds"}
            or any(type(n) is not int or n < 0 for n in counts)
            or any(type(n) is not int or n < 1 for n in self.pages_spanned)
            or self.inferred_band_count is not None
            and (
                type(self.inferred_band_count) is not int
                or self.inferred_band_count != len(self.bands)
            )
            or self.populated_band_count is not None
            and (
                type(self.populated_band_count) is not int
                or self.populated_band_count
                != sum(n > 0 for n in self.observed_band_nonempty_counts)
            )
            or any(
                len(b) != 2 or not all(math.isfinite(v) for v in b) or b[1] <= b[0]
                for b in self.bands
            )
        ):
            raise ValueError("invalid table geometry diagnostics")


@dataclass(frozen=True)
class TableFinding:
    record_id: str
    table_id: str
    page: int
    reason: str
    inspected_count: int
    evidence_ids: tuple[str, ...]
    outcome: str = "TABLE_UNRESOLVED"
    diagnostics: TableDiagnostics | None = None

    def __post_init__(self):
        if (
            not self.record_id
            or not self.table_id
            or self.page < 1
            or not self.reason
            or type(self.inspected_count) is not int
            or self.inspected_count < 1
            or self.outcome != "TABLE_UNRESOLVED"
        ):
            raise ValueError("table refusal requires scope, reason and count")


@dataclass(frozen=True)
class TableRow:
    record_id: str
    page: int
    cells: tuple[str | None, ...]
    members: tuple[tuple[str, ...], ...]
    bbox: tuple[float, float, float, float]
    flags: tuple[str, ...] = ()
    cell_flags: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self):
        members = {ident for column in self.members for ident in column}
        ids = [ident for ident, _ in self.cell_flags]
        if len(ids) != len(set(ids)) or any(ident not in members for ident in ids):
            raise ValueError("cell flags require unique source members")


@dataclass(frozen=True)
class RecoveredTable:
    table_id: str
    headers: tuple[str, ...]
    roles: tuple[str, ...]
    rows: tuple[TableRow, ...]
    fragments: tuple[str, ...]
    pages: tuple[int, ...]
    continued_on: int
    repetition_count: int
    bands: tuple[tuple[float, float], ...]
    policy: dict
    evidence_ids: tuple[str, ...]
    continuation_rule: str = "repeated-header-geometry-adjacent-page/1"
    flags: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not self.table_id
            or len(self.headers) != len(self.roles)
            or len(self.bands) != len(self.headers)
            or any(
                r not in {"id", "free_text", "short_code", "empty", "unknown"} for r in self.roles
            )
            or len(self.pages) != self.continued_on
            or len(self.fragments) != self.continued_on
            or self.repetition_count != self.continued_on - 1
            or any(
                len(r.cells) != len(self.headers) or len(r.members) != len(self.headers)
                for r in self.rows
            )
        ):
            raise ValueError("invalid reconstructed table topology")


@dataclass(frozen=True)
class TableAnalysis:
    tables: tuple[RecoveredTable, ...] = ()
    findings: tuple[TableFinding, ...] = ()
    excluded: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    version: str = "table-analysis/0.1"
    stamp_lines: tuple[TableStampLine, ...] = ()


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _join(cells):
    if any(c.text is None for c in cells):
        return None
    rows = []
    for c in sorted(cells, key=lambda c: (c.bbox[1], c.bbox[0], c.cell_id)):
        if (
            rows
            and c.bbox[1] - rows[-1][0].bbox[1]
            <= min(c.bbox[3] - c.bbox[1], rows[-1][0].bbox[3] - rows[-1][0].bbox[1]) * 0.1
        ):
            rows[-1].append(c)
        else:
            rows.append([c])
    return "\n".join(
        " ".join(c.text for c in sorted(row, key=lambda c: c.bbox[0]) if c.text) for row in rows
    ).strip()


def _bands(rows, tolerance):
    intervals = sorted((round(c.bbox[0], 3), round(c.bbox[2], 3)) for row in rows for c in row)
    groups = []
    for left, right in intervals:
        if groups and left - groups[-1][1] <= tolerance:
            groups[-1] = (groups[-1][0], max(right, groups[-1][1]))
        else:
            groups.append((left, right))
    return tuple(groups)


def _map(cells, bands, tolerance):
    columns = [[] for _ in bands]
    for cell in cells:
        box = tuple(round(v, 3) for v in cell.bbox)
        bands = tuple((round(a, 3), round(b, 3)) for a, b in bands)
        overlaps = [i for i, (a, b) in enumerate(bands) if min(box[2], b) > max(box[0], a)]
        if len(overlaps) != 1:
            raise ValueError("STRADDLING_OR_OUTSIDE_BANDS")
        i = overlaps[0]
        a, b = bands[i]
        if box[0] < a - tolerance or box[2] > b + tolerance:
            raise ValueError("STRADDLING_OR_OUTSIDE_BANDS")
        columns[i].append(cell)
    return columns


def _covering_rule_positions(rules, top, bottom):
    """Union observed collinear segments; never bridge a positive gap."""
    by_x = defaultdict(list)
    for rule in rules:
        by_x[round(rule.x, 3)].append((round(rule.top, 3), round(rule.bottom, 3)))
    positions = []
    for x, segments in sorted(by_x.items()):
        covered = top
        for start, end in sorted(segments):
            if end <= covered:
                continue
            if start > covered:
                break
            covered = end
            if covered >= bottom:
                positions.append(x)
                break
    return positions


def _native_header_bands(header, body):
    """Coarsen observed partitions while keeping header labels distinct."""
    partitions = [
        tuple(sorted((round(c.bbox[0], 3), round(c.bbox[2], 3)) for c in row))
        for row in (header, *body)
    ]
    for partition in partitions:
        if any(a[1] != b[0] for a, b in zip(partition, partition[1:])):
            raise ValueError("STRADDLING_OR_OUTSIDE_BANDS")
    if len({(p[0][0], p[-1][1]) for p in partitions}) != 1:
        raise ValueError("STRADDLING_OR_OUTSIDE_BANDS")
    edges = sorted(set.intersection(*(set(v for pair in p for v in pair) for p in partitions)))
    bands = tuple(zip(edges, edges[1:]))
    grouped = _map(header, bands, 0)
    if any(sum(bool(c.text) for c in group) > 1 for group in grouped):
        raise ValueError("BAND_COUNT_VARIES")
    return bands


def _column_role(values, identifiers, numbering, minimum):
    nonempty = [v for v in values if v]
    id_count = sum(bool(identifiers(v)) or bool(numbering.fullmatch(v)) for v in nonempty)
    if id_count >= minimum:
        return "id"
    if all(v == "" for v in values):
        return "empty"
    if any(v is None for v in values):
        return "unknown"
    if nonempty and all(re.fullmatch(r"[A-Z0-9]{1,8}", v) for v in nonempty):
        return "short_code"
    return "free_text"


@dataclass(frozen=True)
class _Candidate:
    page: "PageDigest"
    table: "TableObservation"
    rows: tuple[tuple[TableCell, ...], ...]
    all_cells: tuple[TableCell, ...]
    native_cells: bool


@dataclass(frozen=True)
class _Fragment:
    signature: tuple[str, ...]
    table: "TableObservation"
    headers: tuple[str, ...]
    roles: tuple[str, ...]
    rows: tuple[TableRow, ...]
    bands: tuple[tuple[float, float], ...]
    policy: dict
    evidence_ids: tuple[str, ...]
    flags: tuple[str, ...]
    header_cells: tuple[TableCell, ...] = ()
    header_source_id: str = ""


@dataclass
class _RecoveryState:
    source_id: str
    findings: list[TableFinding] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    diagnostics: dict[str, TableDiagnostics] = field(default_factory=dict)

    def trace(self, table, **values):
        self.diagnostics[table.table_id] = replace(
            self.diagnostics.get(table.table_id, TableDiagnostics()), **values
        )

    def fail(self, table, reason):
        cells = tuple(c for row in table.cell_rows for c in row)
        self.findings.append(
            TableFinding(
                stable_id("table-finding", self.source_id, table.table_id, reason),
                table.table_id,
                table.page,
                reason,
                max(1, len(cells)),
                tuple(c.cell_id for c in cells),
                diagnostics=self.diagnostics.get(table.table_id),
            )
        )


def _is_furniture(cell, boxes, furniture_span_ids):
    return any(_overlap(cell.bbox, b) for b in boxes) or (
        cell.source_spans and set(cell.source_spans) <= furniture_span_ids
    )


def _table_candidacy(page, table, furniture_boxes, furniture_span_ids, state):
    if not table.cell_rows:
        return None  # Already-declared logical tables keep their original path.
    boxes = furniture_boxes.get(page.page, ())
    all_cells = tuple(c for row in table.cell_rows for c in row)
    # Furniture classification precedes candidacy. A native rectangle is
    # topology, not a text span: marks crossing an empty cell cannot
    # delete a column. Entire furniture blocks still leave candidacy.
    native_cells = all(c.geometry_kind == "cell_box" for c in all_cells)

    if native_cells:
        content_cells = [c for c in all_cells if c.text]
        if content_cells and all(
            _is_furniture(c, boxes, furniture_span_ids) for c in content_cells
        ):
            state.excluded.append(table.table_id)
            return None
        rows = table.cell_rows
    else:
        rows = tuple(
            tuple(c for c in row if not _is_furniture(c, boxes, furniture_span_ids))
            for row in table.cell_rows
        )
    rows = tuple(row for row in rows if row)
    if not rows:
        state.excluded.append(table.table_id)
        return None
    state.candidates.append(table.table_id)
    table_box = union_box(c.bbox for c in all_cells)
    state.trace(
        table,
        pages_spanned=(table.page,),
        raw_row_cell_counts=tuple(len(row) for row in table.cell_rows),
        retained_row_cell_counts=tuple(len(row) for row in rows),
        nonempty_row_cell_counts=tuple(sum(bool(c.text) for c in row) for row in rows),
        unreadable_row_cell_counts=tuple(sum(c.text is None for c in row) for row in rows),
        table_sized_unreadable_cells=sum(c.text is None and c.bbox == table_box for c in all_cells),
    )
    if any("MISSING_CELL_GEOMETRY" in c.flags for c in all_cells):
        state.fail(table, "MISSING_CELL_GEOMETRY")
        return None
    if native_cells:
        state.trace(
            table,
            raw_row_cell_counts=(len(table.headers), *(len(r) for r in table.rows)),
            unlocated_slot_count=len(table.headers) + sum(map(len, table.rows)) - len(all_cells),
        )
    return _Candidate(page, table, rows, all_cells, native_cells)


@dataclass(frozen=True)
class _FragmentGeometry:
    bands: tuple[tuple[float, float], ...]
    width: float
    tolerance: float
    derived: bool
    ruled: bool


def _fragment_geometry(candidate, header, body, policy, state):
    page, table = candidate.page, candidate.table
    rows, native_cells = candidate.rows, candidate.native_cells
    widths = [round(c.bbox[2] - c.bbox[0], 3) for row in body for c in row if c.text]
    width = median(widths) if widths else policy.fallback_cell_width
    tolerance = round(width * policy.x_tolerance_factor, 3)
    box = tuple(round(v, 3) for v in union_box(c.bbox for row in rows for c in row))
    possible = _covering_rule_positions(page.rules, box[1], box[3])
    state.trace(table, stage="bands", full_height_rule_count=len(possible))
    left = [x for x in possible if x <= box[0]]
    right = [x for x in possible if x >= box[2]]
    rules = [x for x in possible if left[-1] <= x <= right[0]] if left and right else []
    ruled = len(rules) >= 2
    if native_cells:
        # Native rectangles are cell extents, not glyph extents. Touching
        # rectangles are separate columns; header subcells map to them.
        row_bands = [
            tuple(sorted((round(c.bbox[0], 3), round(c.bbox[2], 3)) for c in row)) for row in body
        ]
        state.trace(table, route="native_cell_bounds", row_band_counts=tuple(map(len, row_bands)))
        try:
            bands = _native_header_bands(header, body)
        except ValueError as error:
            state.fail(table, str(error))
            return None
    elif ruled:
        state.trace(table, route="grid")
        bands = tuple(zip(rules, rules[1:]))
    else:
        state.trace(table, route="row_supported_bands")
        row_bands = [_bands((row,), tolerance) for row in body]
        state.trace(table, row_band_counts=tuple(len(b) for b in row_bands))
        if len({len(b) for b in row_bands}) != 1:
            state.fail(table, "BAND_COUNT_VARIES")
            return None
        starts = [min(b[i][0] for b in row_bands) for i in range(len(row_bands[0]))]
        pitch = median(b - a for a, b in zip(starts, starts[1:])) if len(starts) > 1 else width
        supported_right = max(b[-1][1] for b in row_bands)
        right = min(box[2], max(supported_right, starts[-1] + pitch))
        edges = [min(box[0], starts[0]), *starts[1:], right]
        bands = tuple(zip(edges, edges[1:]))
    observed_counts = tuple(
        sum(
            any(
                c.text and min(round(c.bbox[2], 3), right) > max(round(c.bbox[0], 3), left)
                for c in row
            )
            for row in body
        )
        for left, right in bands
    )
    state.trace(
        table,
        inferred_band_count=len(bands),
        bands=bands,
        observed_band_nonempty_counts=observed_counts,
        populated_band_count=sum(n > 0 for n in observed_counts),
    )
    if len(bands) < 2:
        state.fail(table, "INSUFFICIENT_BANDS")
        return None
    return _FragmentGeometry(bands, width, tolerance, bool(widths), ruled)


def _assemble_table_rows(mapped_rows, candidate, marks, state):
    page, table = candidate.page, candidate.table
    output_rows = []
    for index, row in enumerate(mapped_rows):
        cells = tuple(c for group in row for c in group)
        rowbox = union_box(c.bbox for c in cells)
        flags = {f for c in cells for f in c.flags}
        for group in row:
            ordered = sorted(group, key=lambda c: (c.bbox[1], c.bbox[0]))
            if any(
                (a.text or "")[-1:].isdigit() and (b.text or "")[:1].isdigit()
                for a, b in zip(ordered, ordered[1:])
            ):
                flags.add("DIGIT_JOIN")
        if any(c.text is None for c in cells):
            flags.add("UNREADABLE_CELL")
        if any(_overlap(s.bbox, rowbox) for s in marks):
            flags.add("LIFECYCLE_GLYPH_SUSPECTED")
        cell_flags = []
        for cell in cells:
            observed = set(cell.flags)
            if cell.text is None:
                observed.add("UNREADABLE_CELL")
            if any(_overlap(mark.bbox, cell.bbox) for mark in marks):
                observed.add("LIFECYCLE_GLYPH_SUSPECTED")
            cell_flags.append((cell.cell_id, tuple(sorted(observed))))
        output_rows.append(
            TableRow(
                stable_id("table-row", state.source_id, table.table_id, index),
                page.page,
                tuple(_join(group) for group in row),
                tuple(tuple(c.cell_id for c in group) for group in row),
                rowbox,
                tuple(sorted(flags)),
                tuple(cell_flags),
            )
        )
    return tuple(output_rows)


def _collapse_fragment(candidate, identifiers, numbering, policy, state, header_profile=None):
    page, table = candidate.page, candidate.table
    rows, all_cells, native_cells = candidate.rows, candidate.all_cells, candidate.native_cells
    if header_profile is None:
        if len(rows) < 2:
            state.fail(table, "TOO_FEW_ROWS")
            return None
        header, *body = rows
    else:
        header, body = header_profile.header_cells, rows
    geometry = _fragment_geometry(candidate, header, body, policy, state)
    if geometry is None:
        return None
    bands, tolerance, ruled = geometry.bands, geometry.tolerance, geometry.ruled
    try:
        state.trace(table, stage="mapping")
        mapped = [
            _map(row, bands, 0 if ruled or native_cells else tolerance) for row in (header, *body)
        ]
        state.trace(
            table,
            mapped_band_support=tuple(
                sum(bool(row[i]) for row in mapped[1:]) for i in range(len(bands))
            ),
            mapped_nonempty_support=tuple(
                sum(any(c.text for c in row[i]) for row in mapped[1:]) for i in range(len(bands))
            ),
        )
        if any(any(not cells for cells in row) for row in mapped):
            raise ValueError("BAND_COUNT_VARIES")
        state.trace(table, stage="header")
        headers = tuple(_join(cells) for cells in mapped[0])
        if any(h is None or not h for h in headers):
            raise ValueError("HEADER_UNREADABLE")
        # One observed identifier locates a provisional column. Acceptance
        # still requires minimum_rows observations in the continuation group.
        roles = [
            _column_role([_join(row[i]) for row in mapped[1:]], identifiers, numbering, 1)
            for i in range(len(bands))
        ]
        state.trace(table, stage="role")
        if "id" not in roles:
            raise ValueError("NO_ID_LIKE_COLUMN")
        marks = [s for s in page.spans if abs(s.direction[1]) > 0.1]
        if header_profile is None:
            header_flags = {f for c in header for f in c.flags}
            if any(_overlap(s.bbox, c.bbox) for s in marks for c in header):
                header_flags.add("LIFECYCLE_GLYPH_SUSPECTED")
        else:
            header_flags = set(header_profile.flags)
        output_rows = _assemble_table_rows(mapped[1:], candidate, marks, state)
        normalized = tuple(re.sub(r"\s+", " ", h).strip().casefold() for h in headers)
        return _Fragment(
            normalized,
            table,
            headers,
            tuple(roles),
            tuple(output_rows),
            bands,
            {
                **asdict(policy),
                "median_cell_width": geometry.width,
                "x_tolerance": tolerance,
                "source": "derived" if geometry.derived else "default",
                "route": (
                    "native_cell_bounds"
                    if native_cells
                    else "grid" if ruled else "row_supported_bands"
                ),
            },
            tuple(c.cell_id for c in all_cells),
            tuple(sorted(header_flags)),
            tuple(header),
            table.table_id if header_profile is None else header_profile.header_source_id,
        )
    except ValueError as error:
        state.fail(table, str(error))
        return None


def _continuation_key(item):
    return (
        item.signature,
        tuple(i for i, role in enumerate(item.roles) if role == "id"),
        _relative_bands(item.bands),
    )


def _continue_fragments(provisional, state):
    counts = Counter(p.signature for p in provisional if p.header_source_id == p.table.table_id)

    per_page = Counter((_continuation_key(item), item.table.page) for item in provisional)
    groups = []
    for item in sorted(provisional, key=lambda p: (p.table.page, p.table.table_id)):
        signature, table = item.signature, item.table
        state.trace(table, stage="continuation")
        if counts[signature] < 2:
            state.fail(table, "HEADER_NOT_REPEATED")
            continue
        # A repeated header supplies a continuation observation. Ambiguous same-
        # page blocks stay separate; no row is deleted merely for equal content.
        matches = [
            g
            for g in groups
            if g[-1].table.page == table.page - 1
            and _continuation_key(g[-1]) == _continuation_key(item)
            and per_page[(_continuation_key(item), table.page)] == 1
            and per_page[(_continuation_key(item), table.page - 1)] == 1
        ]
        if len(matches) == 1:
            matches[0].append(item)
        else:
            groups.append([item])
    return groups


def _support_groups(groups, identifiers, numbering, policy, state):
    recovered = []
    for group in groups:
        first = group[0]
        rows = tuple(r for item in group for r in item.rows)
        for item in group:
            state.trace(
                item.table,
                stage="support",
                continuation_fragments=len(group),
                pooled_rows=len(rows),
                pages_spanned=tuple(member.table.page for member in group),
            )
        # Supporting-row evidence belongs to the continued table, not to an
        # arbitrary page break. Every mapped row has a physical cell per band.
        if len(rows) < policy.minimum_rows:
            for item in group:
                state.fail(item.table, "TOO_FEW_ROWS")
            continue
        roles = tuple(
            _column_role([r.cells[i] for r in rows], identifiers, numbering, policy.minimum_rows)
            for i in range(len(first.roles))
        )
        if "id" not in roles:
            for item in group:
                state.fail(item.table, "NO_ID_LIKE_COLUMN")
            continue
        inherited_headers = {
            item.table.table_id: item.header_source_id
            for item in group
            if item.header_source_id != item.table.table_id
        }
        group_policy = (
            {**first.policy, "inherited_headers": inherited_headers}
            if inherited_headers
            else first.policy
        )
        recovered.append(
            RecoveredTable(
                stable_id("continued-table", state.source_id, first.table.table_id),
                first.headers,
                roles,
                rows,
                tuple(item.table.table_id for item in group),
                tuple(item.table.page for item in group),
                len(group),
                len(group) - 1,
                first.bands,
                group_policy,
                tuple(cid for item in group for cid in item.evidence_ids),
                flags=tuple(sorted({f for item in group for f in item.flags})),
            )
        )
    return recovered


def _row_has_identifier(row, identifiers, numbering):
    return any(c.text and (identifiers(c.text) or numbering.fullmatch(c.text)) for c in row)


def _inherit_fragment_headers(pending, fragments, identifiers, numbering, policy, state):
    """Use only repeated profiles on adjacent pages; keep all physical data rows."""
    repeated = Counter(f.signature for f in fragments)
    by_page = defaultdict(list)
    for fragment in fragments:
        if repeated[fragment.signature] >= 2:
            by_page[fragment.table.page].append(fragment)
    for candidate in pending:
        table = candidate.table
        proposals = []
        for number in (table.page - 1, table.page + 1):
            neighbors = by_page[number]
            unique = Counter(_continuation_key(f) for f in neighbors)
            proposals.extend(f for f in neighbors if unique[_continuation_key(f)] == 1)
        tested = {}
        attempts = []
        for profile in proposals:
            key = _continuation_key(profile)
            if key in tested:
                continue
            local = _RecoveryState(
                state.source_id, diagnostics={table.table_id: state.diagnostics[table.table_id]}
            )
            fragment = _collapse_fragment(candidate, identifiers, numbering, policy, local, profile)
            attempts.append(local)
            tested[key] = (
                (fragment, local)
                if fragment is not None and _continuation_key(fragment) == key
                else None
            )
        matches = [f for f in tested.values() if f is not None]
        if len(matches) == 1:
            fragment, local = matches[0]
            state.diagnostics[table.table_id] = local.diagnostics[table.table_id]
            fragments.append(fragment)
            by_page[table.page].append(fragment)
            # The selected header source is retained in the logical table policy.
            state.trace(table, stage="continuation")
        elif len(attempts) == 1 and attempts[0].findings:
            state.diagnostics[table.table_id] = attempts[0].diagnostics[table.table_id]
            state.findings.extend(attempts[0].findings)
        else:
            state.trace(table, stage="continuation")
            state.fail(table, "HEADER_NOT_REPEATED")


def recover_tables(
    document,
    *,
    furniture_boxes,
    furniture_span_ids=frozenset(),
    identifiers,
    numbering,
    policy=TablePolicy(),
):
    """Run candidacy, collapse, continuation and support with ordered evidence."""
    state = _RecoveryState(document.source_id)
    fragments = []
    pending = []
    for page in document.pages:
        for table in page.tables:
            candidate = _table_candidacy(page, table, furniture_boxes, furniture_span_ids, state)
            if candidate is None:
                continue
            if _row_has_identifier(candidate.rows[0], identifiers, numbering):
                pending.append(candidate)
                continue
            fragment = _collapse_fragment(candidate, identifiers, numbering, policy, state)
            if fragment is not None:
                fragments.append(fragment)
    _inherit_fragment_headers(pending, fragments, identifiers, numbering, policy, state)
    groups = _continue_fragments(fragments, state)
    recovered = _support_groups(groups, identifiers, numbering, policy, state)
    return TableAnalysis(
        tuple(recovered),
        tuple(state.findings),
        tuple(state.excluded),
        tuple(state.candidates),
        stamp_lines=table_row_stamps(document),
    )


def analysis_from_dict(data):
    return TableAnalysis(
        **{
            **data,
            "tables": tuple(
                RecoveredTable(
                    **{
                        **t,
                        "headers": tuple(t["headers"]),
                        "roles": tuple(t["roles"]),
                        "flags": tuple(t.get("flags", ())),
                        "rows": tuple(
                            TableRow(
                                **{
                                    **r,
                                    "cells": tuple(r["cells"]),
                                    "members": tuple(tuple(m) for m in r["members"]),
                                    "bbox": tuple(r["bbox"]),
                                    "flags": tuple(r["flags"]),
                                    "cell_flags": tuple(
                                        (ident, tuple(flags))
                                        for ident, flags in r.get("cell_flags", ())
                                    ),
                                }
                            )
                            for r in t["rows"]
                        ),
                        "fragments": tuple(t["fragments"]),
                        "pages": tuple(t["pages"]),
                        "bands": tuple(tuple(b) for b in t["bands"]),
                        "evidence_ids": tuple(t["evidence_ids"]),
                    }
                )
                for t in data["tables"]
            ),
            "findings": tuple(
                TableFinding(
                    **{
                        **f,
                        "evidence_ids": tuple(f["evidence_ids"]),
                        "diagnostics": _diagnostics_from_dict(f.get("diagnostics")),
                    }
                )
                for f in data["findings"]
            ),
            "excluded": tuple(data["excluded"]),
            "candidate_ids": tuple(data["candidate_ids"]),
            "stamp_lines": tuple(
                table_stamp_from_dict(line) for line in data.get("stamp_lines", ())
            ),
        }
    )


def _relative_bands(bands):
    left = bands[0][0]
    width = bands[-1][1] - left
    return tuple((round((a - left) / width, 3), round((b - left) / width, 3)) for a, b in bands)


def _diagnostics_from_dict(data):
    if data is None:
        return None
    return TableDiagnostics(
        **{
            key: (
                tuple(tuple(b) for b in value)
                if key == "bands"
                else tuple(value) if isinstance(value, list) else value
            )
            for key, value in data.items()
        }
    )
