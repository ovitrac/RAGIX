"""Exact cell spans and declared structural associations, without interpretation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Cell:
    source_id: str
    table_id: str
    cell_id: str
    page: int
    row: int
    column: int
    text: str
    bbox: tuple[float, float, float, float]
    source_spans: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    column_span: int = 1
    row_span: int = 1

    def __post_init__(self):
        if (
            not all((self.source_id, self.table_id, self.cell_id))
            or not isinstance(self.text, str)
            or any(
                type(n) is not int
                for n in (self.page, self.row, self.column, self.column_span, self.row_span)
            )
            or self.page < 1
            or min(self.row, self.column) < 0
            or min(self.column_span, self.row_span) < 1
            or len(self.bbox) != 4
            or not all(math.isfinite(v) for v in self.bbox)
            or self.bbox[2] <= self.bbox[0]
            or self.bbox[3] <= self.bbox[1]
        ):
            raise ValueError("invalid located table cell")

    def span(self, start=0, end=None):
        end = len(self.text) if end is None else end
        if not 0 <= start <= end <= len(self.text):
            raise ValueError("span outside source cell")
        return CellSpan(
            self.source_id,
            self.table_id,
            self.cell_id,
            self.page,
            start,
            end,
            self.text[start:end],
            self.bbox,
            self.source_spans,
        )


@dataclass(frozen=True)
class CellSpan:
    source_id: str
    table_id: str
    cell_id: str
    page: int
    start: int
    end: int
    raw: str
    bbox: tuple[float, float, float, float]
    source_spans: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            not all((self.source_id, self.table_id, self.cell_id))
            or self.page < 1
            or self.start < 0
            or self.end - self.start != len(self.raw)
        ):
            raise ValueError("invalid exact cell span")


@dataclass(frozen=True)
class CellContext:
    value: Cell
    column_headers: tuple[Cell, ...] = ()
    row_labels: tuple[Cell, ...] = ()
    flags: tuple[str, ...] = ()
    rule: str = "declared-table-topology/1"

    def __post_init__(self):
        refs = self.column_headers + self.row_labels
        if any(
            c.source_id != self.value.source_id
            or c.table_id != self.value.table_id
            or c.cell_id == self.value.cell_id
            for c in refs
        ):
            raise ValueError("context must belong to the same source table")
        if len({c.cell_id for c in refs}) != len(refs):
            raise ValueError("duplicate context cell")
        for c in self.column_headers:
            if (
                not c.column
                <= self.value.column
                < self.value.column + self.value.column_span
                <= c.column + c.column_span
            ):
                raise ValueError("header does not cover value column")
        for c in self.row_labels:
            if c.page != self.value.page or not c.row <= self.value.row < c.row + c.row_span:
                raise ValueError("label does not cover value row")


def associate(value, cells, *, header_rows=(), row_label_columns=()):
    """Select only declared header rows and label columns; never adjacent prose.

    A spanning header must cover the whole value column interval. Multi-line or
    multi-row headers stay separate source cells. With no declared header, none
    is inferred from the first data row.
    """
    cells = tuple(cells)
    if len({c.cell_id for c in cells}) != len(cells) or value not in cells:
        raise ValueError("unique observed cells including the value are required")
    if any((c.source_id, c.table_id) != (value.source_id, value.table_id) for c in cells):
        raise ValueError("mixed source tables")
    headers = tuple(
        c
        for c in cells
        if c.row in header_rows
        and c.cell_id != value.cell_id
        and c.column <= value.column
        and value.column + value.column_span <= c.column + c.column_span
    )
    labels = tuple(
        c
        for c in cells
        if c.row not in header_rows
        and c.column in row_label_columns
        and c.cell_id != value.cell_id
        and c.page == value.page
        and c.row <= value.row < c.row + c.row_span
    )
    return CellContext(value, headers, labels, () if headers else ("NO_COLUMN_HEADER",))
