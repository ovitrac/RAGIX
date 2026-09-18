"""Structural cell-context adapters over observed and recovered table topology.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import dataclass, replace
from ..harvest.table_context import Cell, CellContext, associate
from .table_views import _map, _join, _native_header_bands


@dataclass(frozen=True)
class ContextFinding:
    table_id: str
    page: int
    reason: str
    inspected_count: int


def _cell(source, table_id, page, row, column, observed, flags=()):
    return Cell(
        source,
        table_id,
        observed.cell_id,
        page,
        row,
        column,
        observed.text,
        observed.bbox,
        observed.source_spans,
        tuple(sorted(set(observed.flags) | set(flags))),
    )


def _headers(observation, expected, bands):
    """Only the observed header row qualifies; never consume a neighboring row."""
    if not observation.cell_rows or not any(expected):
        return ()
    header = observation.cell_rows[0]
    mapped = _map(header, bands, 0)
    if tuple(_join(group) for group in mapped) != tuple(expected):
        raise ValueError("HEADER_ASSOCIATION_UNRESOLVED")
    if any(c.text is None for group in mapped for c in group):
        raise ValueError("HEADER_UNREADABLE")
    return mapped


def native_contexts(source_id, observation, *, row_label_columns=(0,)):
    """Read native cell topology even when it has no identifier-like column.

    The leading stub is exposed as positional row context by default, not as a
    semantic label. Callers may declare different stub columns or none. No value
    or unit is parsed here. Unlocated/fallback cell rectangles cannot supply a unit.
    """
    if not observation.cell_rows:
        return (), (
            ContextFinding(observation.table_id, observation.page, "NO_EXACT_CELL_TOPOLOGY", 1),
        )
    rows = observation.cell_rows
    try:
        if any(
            c.geometry_kind != "cell_box" or "MISSING_CELL_GEOMETRY" in c.flags
            for row in rows
            for c in row
        ):
            raise ValueError("NO_EXACT_CELL_TOPOLOGY")
        bands = _native_header_bands(rows[0], rows[1:])
        if not bands:
            raise ValueError("COLUMN_ASSOCIATION_UNRESOLVED")
        has_header = any(observation.headers)
        expected = tuple(h for h in observation.headers if h)
        headers = _headers(observation, expected, bands) if has_header else ()
        mapped = [_map(row, bands, 0) for row in rows]
        cells = tuple(
            _cell(source_id, observation.table_id, observation.page, r, col, c)
            for r, row in enumerate(mapped)
            for col, group in enumerate(row)
            for c in group
            if c.text is not None
        )
        values = [
            c for c in cells if (not has_header or c.row > 0) and c.column not in row_label_columns
        ]
        contexts = tuple(
            associate(
                c, cells, header_rows=(0,) if headers else (), row_label_columns=row_label_columns
            )
            for c in values
        )
        return contexts, ()
    except ValueError as error:
        return (), (
            ContextFinding(
                observation.table_id, observation.page, str(error), max(1, sum(map(len, rows)))
            ),
        )


def recovered_contexts(document, table, *, row_label_columns=()):
    """Use recovered member ids and observed header cells; no concatenated context."""
    observations = {t.table_id: t for p in document.pages for t in p.tables}
    physical = {
        c.cell_id: (t, c)
        for p in document.pages
        for t in p.tables
        for row in t.cell_rows
        for c in row
    }
    contexts, findings = [], []
    for row_index, row in enumerate(table.rows, 1):
        try:
            owners = {physical[ident][0].table_id for group in row.members for ident in group}
            if len(owners) != 1:
                raise ValueError("AMBIGUOUS_SOURCE_FRAGMENT")
            owner = observations[next(iter(owners))]
            header_id = table.policy.get("inherited_headers", {}).get(
                owner.table_id, owner.table_id
            )
            header = observations[header_id]
            source_cells = header.cell_rows[0]
            left, right = min(c.bbox[0] for c in source_cells), max(c.bbox[2] for c in source_cells)
            a, b = table.bands[0][0], table.bands[-1][1]
            bands = tuple(
                (
                    left + (x - a) / (b - a) * (right - left),
                    left + (y - a) / (b - a) * (right - left),
                )
                for x, y in table.bands
            )
            mapped = _headers(header, table.headers, bands)
            hc = tuple(
                _cell(document.source_id, table.table_id, header.page, 0, col, c, table.flags)
                for col, group in enumerate(mapped)
                for c in group
            )
            flags = dict(row.cell_flags)
            vc = tuple(
                _cell(
                    document.source_id,
                    table.table_id,
                    row.page,
                    row_index,
                    col,
                    physical[ident][1],
                    flags.get(ident, ()),
                )
                for col, group in enumerate(row.members)
                for ident in group
                if physical[ident][1].text is not None
            )
            for value in vc:
                if value.column in row_label_columns or table.roles[value.column] == "id":
                    continue
                contexts.append(
                    associate(value, hc + vc, header_rows=(0,), row_label_columns=row_label_columns)
                )
        except (ValueError, KeyError, IndexError) as error:
            findings.append(
                ContextFinding(
                    table.table_id,
                    row.page,
                    str(error) if isinstance(error, ValueError) else "MISSING_SOURCE_CELL",
                    max(1, sum(map(len, row.members))),
                )
            )
    return tuple(contexts), tuple(findings)


def document_contexts(document, analysis):
    contexts, findings = [], []
    recovered = {ident for table in analysis.tables for ident in table.fragments}
    for table in analysis.tables:
        # A leading identifier column is not a row label. Other stub policies are
        # explicit caller choices through recovered_contexts().
        a, b = recovered_contexts(document, table)
        contexts.extend(a)
        findings.extend(b)
    for page in document.pages:
        for table in page.tables:
            if table.table_id in recovered or table.table_id in analysis.excluded:
                continue
            if not table.cell_rows:
                continue  # Legacy flattened observations retain their existing reader.
            a, b = native_contexts(document.source_id, table)
            contexts.extend(a)
            findings.extend(b)
    return tuple(contexts), tuple(findings)
