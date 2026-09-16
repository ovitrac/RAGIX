"""Recurrence-filtered physical table blocks and traced column reconstruction.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from statistics import median
import re
import math
from .field_views import stable_id, union_box


@dataclass(frozen=True)
class TableCell:
    cell_id: str
    text: str | None
    bbox: tuple[float, float, float, float]
    source_spans: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not self.cell_id
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
class TableFinding:
    record_id: str
    table_id: str
    page: int
    reason: str
    inspected_count: int
    evidence_ids: tuple[str, ...]
    outcome: str = "TABLE_UNRESOLVED"

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


def recover_tables(
    document,
    *,
    furniture_boxes,
    furniture_span_ids=frozenset(),
    identifiers,
    numbering,
    policy=TablePolicy(),
):
    provisional = []
    findings = []
    excluded = []
    candidates = []

    def fail(table, reason):
        cells = tuple(c for row in table.cell_rows for c in row)
        findings.append(
            TableFinding(
                stable_id("table-finding", document.source_id, table.table_id, reason),
                table.table_id,
                table.page,
                reason,
                max(1, len(cells)),
                tuple(c.cell_id for c in cells),
            )
        )

    for page in document.pages:
        for table in page.tables:
            if not table.cell_rows:
                continue  # Already-declared logical tables keep their original path.
            boxes = furniture_boxes.get(page.page, ())
            all_cells = tuple(c for row in table.cell_rows for c in row)
            # Filtering precedes even the attempt to call a first row a header.
            rows = tuple(
                tuple(
                    c
                    for c in row
                    if not any(_overlap(c.bbox, b) for b in boxes)
                    and not (c.source_spans and set(c.source_spans) <= furniture_span_ids)
                )
                for row in table.cell_rows
            )
            rows = tuple(row for row in rows if row)
            if not rows:
                excluded.append(table.table_id)
                continue
            candidates.append(table.table_id)
            if len(rows) < 2:
                fail(table, "TOO_FEW_ROWS")
                continue
            header, *body = rows
            widths = [round(c.bbox[2] - c.bbox[0], 3) for row in body for c in row if c.text]
            width = median(widths) if widths else policy.fallback_cell_width
            tolerance = round(width * policy.x_tolerance_factor, 3)
            box = tuple(round(v, 3) for v in union_box(c.bbox for row in rows for c in row))
            possible = _covering_rule_positions(page.rules, box[1], box[3])
            left = [x for x in possible if x <= box[0]]
            right = [x for x in possible if x >= box[2]]
            rules = [x for x in possible if left[-1] <= x <= right[0]] if left and right else []
            ruled = len(rules) >= 2
            if ruled:
                bands = tuple(zip(rules, rules[1:]))
            else:
                row_bands = [_bands((row,), tolerance) for row in body]
                if len({len(b) for b in row_bands}) != 1:
                    fail(table, "BAND_COUNT_VARIES")
                    continue
                starts = [min(b[i][0] for b in row_bands) for i in range(len(row_bands[0]))]
                pitch = (
                    median(b - a for a, b in zip(starts, starts[1:])) if len(starts) > 1 else width
                )
                supported_right = max(b[-1][1] for b in row_bands)
                right = min(box[2], max(supported_right, starts[-1] + pitch))
                edges = [min(box[0], starts[0]), *starts[1:], right]
                bands = tuple(zip(edges, edges[1:]))
            if len(bands) < 2:
                fail(table, "INSUFFICIENT_BANDS")
                continue
            try:
                mapped = [_map(row, bands, 0 if ruled else tolerance) for row in rows]
                if any(any(not cells for cells in row) for row in mapped):
                    raise ValueError("BAND_COUNT_VARIES")
                headers = tuple(_join(cells) for cells in mapped[0])
                if any(h is None or not h for h in headers):
                    raise ValueError("HEADER_UNREADABLE")
                # One observed identifier locates a provisional column. Acceptance
                # still requires minimum_rows observations in the continuation group.
                roles = [
                    _column_role([_join(row[i]) for row in mapped[1:]], identifiers, numbering, 1)
                    for i in range(len(bands))
                ]
                if "id" not in roles:
                    raise ValueError("NO_ID_LIKE_COLUMN")
                output_rows = []
                marks = [s for s in page.spans if abs(s.direction[1]) > 0.1]
                header_flags = {f for c in header for f in c.flags}
                if any(_overlap(s.bbox, c.bbox) for s in marks for c in header):
                    header_flags.add("LIFECYCLE_GLYPH_SUSPECTED")
                for index, row in enumerate(mapped[1:]):
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
                    output_rows.append(
                        TableRow(
                            stable_id("table-row", document.source_id, table.table_id, index),
                            page.page,
                            tuple(_join(group) for group in row),
                            tuple(tuple(c.cell_id for c in group) for group in row),
                            rowbox,
                            tuple(sorted(flags)),
                        )
                    )
                normalized = tuple(re.sub(r"\s+", " ", h).strip().casefold() for h in headers)
                provisional.append(
                    (
                        normalized,
                        table,
                        headers,
                        tuple(roles),
                        tuple(output_rows),
                        bands,
                        {
                            **asdict(policy),
                            "median_cell_width": width,
                            "x_tolerance": tolerance,
                            "source": "derived" if widths else "default",
                            "route": "grid" if ruled else "row_supported_bands",
                        },
                        tuple(c.cell_id for c in all_cells),
                        tuple(sorted(header_flags)),
                    )
                )
            except ValueError as error:
                fail(table, str(error))
    counts = Counter(p[0] for p in provisional)

    def continuation_key(item):
        return (
            item[0],
            tuple(i for i, role in enumerate(item[3]) if role == "id"),
            _relative_bands(item[5]),
        )

    per_page = Counter((continuation_key(item), item[1].page) for item in provisional)
    groups = []
    for item in sorted(provisional, key=lambda p: (p[1].page, p[1].table_id)):
        signature, table, *_ = item
        if counts[signature] < 2:
            fail(table, "HEADER_NOT_REPEATED")
            continue
        # A repeated header supplies a continuation observation. Ambiguous same-
        # page blocks stay separate; no row is deleted merely for equal content.
        matches = [
            g
            for g in groups
            if g[-1][1].page == table.page - 1
            and continuation_key(g[-1]) == continuation_key(item)
            and per_page[(continuation_key(item), table.page)] == 1
            and per_page[(continuation_key(item), table.page - 1)] == 1
        ]
        if len(matches) == 1:
            matches[0].append(item)
        else:
            groups.append([item])
    recovered = []
    for group in groups:
        first = group[0]
        rows = tuple(r for item in group for r in item[4])
        # Supporting-row evidence belongs to the continued table, not to an
        # arbitrary page break. Every mapped row has a physical cell per band.
        if len(rows) < policy.minimum_rows:
            for item in group:
                fail(item[1], "TOO_FEW_ROWS")
            continue
        roles = tuple(
            _column_role([r.cells[i] for r in rows], identifiers, numbering, policy.minimum_rows)
            for i in range(len(first[3]))
        )
        if "id" not in roles:
            for item in group:
                fail(item[1], "NO_ID_LIKE_COLUMN")
            continue
        recovered.append(
            RecoveredTable(
                stable_id("continued-table", document.source_id, first[1].table_id),
                first[2],
                roles,
                rows,
                tuple(item[1].table_id for item in group),
                tuple(item[1].page for item in group),
                len(group),
                len(group) - 1,
                first[5],
                first[6],
                tuple(cid for item in group for cid in item[7]),
                flags=tuple(sorted({f for item in group for f in item[8]})),
            )
        )
    return TableAnalysis(tuple(recovered), tuple(findings), tuple(excluded), tuple(candidates))


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
                TableFinding(**{**f, "evidence_ids": tuple(f["evidence_ids"])})
                for f in data["findings"]
            ),
            "excluded": tuple(data["excluded"]),
            "candidate_ids": tuple(data["candidate_ids"]),
        }
    )


def _relative_bands(bands):
    left = bands[0][0]
    width = bands[-1][1] - left
    return tuple((round((a - left) / width, 3), round((b - left) / width, 3)) for a, b in bands)
