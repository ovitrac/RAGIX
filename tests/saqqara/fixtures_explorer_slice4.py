"""Generic Slice 4 table construction, specified independently before implementation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import dataclass
from math import cos, sin, radians
from ragix_kernels.saqqara.census import DocumentDigest, PageDigest, TableObservation, Evidence
from ragix_kernels.saqqara.field_views import TextSpan
from ragix_kernels.saqqara.table_views import TableCell

SOURCE = "synthetic-explorer-slice4-T"
HEADERS = ("Ref", "Requirement", "Criticality", "Spec ref", "Test ref")
ROW_COUNTS = (1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 5, 4, 2, 1, 5, 6, 4, 3, 5, 6, 4, 2, 5, 6, 3, 4)
GEOMETRY_RANGES = {
    "native_columns": (12, 13, 14, 15),
    "column_width": (80.0, 120.0),
    "row_height": (24.0, 40.0),
    "watermark_points": (50.0, 90.0),
    "watermark_angle": (30.0, 60.0),
}


@dataclass(frozen=True)
class TableGeometry:
    native_columns: int
    column_width: float
    row_height: float
    watermark_points: float
    watermark_angle: float

    def __post_init__(self):
        if self.native_columns not in GEOMETRY_RANGES["native_columns"]:
            raise ValueError("native columns outside fixture range")
        for key in ("column_width", "row_height", "watermark_points", "watermark_angle"):
            lo, hi = GEOMETRY_RANGES[key]
            if not lo <= getattr(self, key) <= hi:
                raise ValueError("geometry outside declared fixture range")


@dataclass(frozen=True)
class TableFixture:
    document: DocumentDigest
    expected_rows: tuple[tuple[str, ...], ...]
    expected_keys: tuple[str, ...]
    fragment_ids: tuple[str, ...]
    headerless_pages: tuple[int, ...]
    watermark_cells: tuple[str, ...]


def _cell(page, ident, text, box):
    return TableCell(ident, text, box, (ident + ":span",), geometry_kind="cell_box")


def _table(page, ident, physical_rows, slots=None):
    # The raw matrix exposes empty placeholders for spans across physical slots.
    slots = slots or max(map(len, physical_rows))
    matrix = [tuple(c.text for c in row) + (None,) * (slots - len(row)) for row in physical_rows]
    evidence = tuple(
        Evidence(SOURCE, page, c.cell_id, 0, len(c.text or ""), c.text or "", c.bbox)
        for row in physical_rows
        for c in row
    )
    return TableObservation(
        ident,
        page,
        tuple(v or "" for v in matrix[0]),
        tuple(matrix[1:]),
        evidence,
        cell_rows=tuple(tuple(r) for r in physical_rows),
    )


def _spans(page, tables):
    return tuple(
        TextSpan(
            SOURCE,
            c.source_spans[0],
            page,
            c.text,
            c.bbox,
            origin=(c.bbox[0], c.bbox[1] + 10),
            font_size=10,
        )
        for table in tables
        for row in table.cell_rows
        for c in row
        if c.text
    )


def realistic_table(geometry: TableGeometry, *, all_controls=False, removed_page=None):
    """Fixture T: 32 pages, 108 rows, padding, control and headerless pages."""
    pages, expected, fragments, overlapped = [], [], [], []
    serial = 0
    width = geometry.column_width
    for page in range(1, 33):
        tables = []
        for ident, columns, count, y in (("top", 4, 2, 12), ("bottom", 2, 1, 950)):
            rows = []
            for row in range(count):
                rows.append(
                    tuple(
                        _cell(
                            page,
                            f"{ident}:{page}:{row}:{col}",
                            ("Running", "RUN-X-901", "Revision", "A")[col],
                            (
                                20 + col * width,
                                y + row * 16,
                                20 + (col + 1) * width,
                                y + (row + 1) * 16,
                            ),
                        )
                        for col in range(columns)
                    )
                )
            tables.append(_table(page, f"furniture:{ident}:{page}", rows))
        if page == 1:
            values = (
                ("Role", "Person", "Timestamp"),
                ("Prepared", "Nom Prénom-01", "01/01/2000 10:00"),
                ("Checked", "Nom Prénom-02", "01/01/2000 11:00"),
                ("Approved", "Nom Prénom-03", "01/01/2000 12:00"),
            )
            rows = [
                tuple(
                    _cell(
                        page,
                        f"approval:{row}:{col}",
                        value,
                        (20 + col * width, 180 + row * 24, 20 + (col + 1) * width, 204 + row * 24),
                    )
                    for col, value in enumerate(values_row)
                )
                for row, values_row in enumerate(values)
            ]
            tables.append(_table(page, "approval:1", rows))
        count = ROW_COUNTS[page - 5] if 5 <= page <= 30 else 0
        mark_box = (20 + 1.1 * width, 224, 20 + 2.9 * width, 224 + geometry.row_height)
        if count:
            controls = all_controls or page in (9, 21)
            parts = (
                (1,) * 5
                if controls
                else tuple(2 + int(col < geometry.native_columns - 10) for col in range(5))
            )
            physical = []
            if page not in (14, 25):
                physical.append(
                    tuple(
                        _cell(
                            page,
                            f"header:{page}:{col}",
                            label,
                            (20 + col * width, 180, 20 + (col + 1) * width, 212),
                        )
                        for col, label in enumerate(HEADERS)
                    )
                )
            for row in range(count):
                serial += 1
                # Four depth-4 keys extend one depth-3 parent; all keys are planted.
                key = f"7.1.10.{serial-10}" if 11 <= serial <= 14 else f"7.1.{serial}"
                values = (
                    key,
                    f"Condition row {serial:03d}",
                    "C",
                    f"Section 2.{serial}",
                    f"Section 3.{serial}",
                )
                raw = []
                logical = []
                y = 220 + row * geometry.row_height
                for col, (value, n) in enumerate(zip(values, parts)):
                    chunks = [value]
                    if n > 1 and (
                        col in (1, 3)
                        or col == 4
                        and (page % 2 == 0 or geometry.native_columns < 14)
                    ):
                        chunks = value.split(" ", 1)
                    cells = []
                    for sub in range(n):
                        text = chunks[sub] if sub < len(chunks) else ""
                        x = 20 + col * width
                        box = (
                            x + width * sub / n,
                            y,
                            x + width * (sub + 1) / n,
                            y + geometry.row_height,
                        )
                        ident = f"body:{page}:{row}:{col}:{sub}"
                        if min(box[2], mark_box[2]) > max(box[0], mark_box[0]) and min(
                            box[3], mark_box[3]
                        ) > max(box[1], mark_box[1]):
                            overlapped.append(ident)
                        cells.append(_cell(page, ident, text, box))
                    raw.extend(cells)
                    logical.append(" ".join(c.text for c in cells if c.text))
                physical.append(tuple(raw))
                expected.append(tuple(logical))
            ident = f"body-fragment:{page}"
            fragments.append(ident)
            tables.append(_table(page, ident, physical, slots=sum(parts)))
        if page in (31, 32):
            rows = [
                tuple(
                    _cell(
                        page,
                        f"free:{page}:{row}:{col}",
                        value,
                        (20 + col * width, 180 + row * 24, 20 + (col + 1) * width, 204 + row * 24),
                    )
                    for col, value in enumerate(values)
                )
                for row, values in enumerate(
                    (
                        ("Category", "Description", "Notes"),
                        ("alpha", "plain text", "plain text"),
                        ("beta", "plain text", "plain text"),
                    )
                )
            ]
            tables.append(_table(page, f"free:{page}", rows))
        angle = radians(geometry.watermark_angle if page % 2 else 90 - geometry.watermark_angle)
        mark = TextSpan(
            SOURCE,
            f"watermark:{page}",
            page,
            "RETIRED",
            mark_box,
            direction=(cos(angle), -sin(angle)),
            font_size=geometry.watermark_points,
        )
        spans = (*_spans(page, tables), mark)
        if page in (2, 3, 4, 31, 32):
            spans += (
                TextSpan(
                    SOURCE,
                    f"prose:{page}",
                    page,
                    f"{page}.1 Synthetic heading",
                    (20, 150, 20 + 3 * width, 150 + geometry.row_height / 2),
                    font_size=10,
                ),
            )
        if page == removed_page:
            # Keep the physical page inventory; only remove its table fragments.
            tables = [t for t in tables if t.table_id != f"body-fragment:{page}"]
            spans = _spans(page, tables)
        pages.append(PageDigest(page, 40 + 5 * width, 1000, spans, tables=tuple(tables)))
    return TableFixture(
        DocumentDigest(SOURCE, tuple(pages), "synthetic", "slice4/0"),
        tuple(expected),
        tuple(row[0] for row in expected),
        tuple(fragments),
        (14, 25),
        tuple(overlapped),
    )
