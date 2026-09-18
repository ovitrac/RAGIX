"""Slice 4 realistic-table gates; fixture precedes continuation changes.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import pytest
from .fixtures_explorer_slice4 import TableGeometry, realistic_table
from ragix_kernels.saqqara.explorer import explore


@pytest.mark.parametrize("native", [12, 13, 14, 15])
def test_t_header_collapse_and_headerless_continuation(native):
    fixture = realistic_table(TableGeometry(native, 100, 32, 60, 45))
    result = explore(fixture.document)
    (table,) = result.census.table_analysis.tables
    assert table.headers == ("Ref", "Requirement", "Criticality", "Spec ref", "Test ref")
    assert tuple(r.cells for r in table.rows) == fixture.expected_rows
    assert tuple(r.cells[0] for r in table.rows) == fixture.expected_keys
    assert table.continued_on == 26 and len(table.rows) == 108
    assert set(table.fragments) == set(fixture.fragment_ids)
    assert len(result.reading.tables) == 108
    assert not any(r["page"] == 1 for r in result.reading.tables)


def test_t_five_column_controls_preserve_every_value():
    geometry = TableGeometry(15, 100, 32, 60, 45)
    padded = explore(realistic_table(geometry).document)
    control = explore(realistic_table(geometry, all_controls=True).document)
    assert [r["cells"] for r in padded.reading.tables] == [
        r["cells"] for r in control.reading.tables
    ]
    assert len(padded.reading.tables) == 108


@pytest.mark.parametrize("native", [12, 13, 14, 15])
@pytest.mark.parametrize("angle,points", [(30, 50), (30, 90), (60, 50), (60, 90)])
def test_t_geometry_ranges_and_exact_cell_flags(native, angle, points):
    from dataclasses import asdict
    import json
    from ragix_kernels.saqqara.table_views import analysis_from_dict

    width, height = (80, 24) if angle == 30 else (120, 40)
    f = realistic_table(TableGeometry(native, width, height, points, angle))
    r = explore(f.document)
    (table,) = r.census.table_analysis.tables
    assert tuple(row.cells for row in table.rows) == f.expected_rows
    marked = {
        ident
        for row in table.rows
        for ident, flags in row.cell_flags
        if "LIFECYCLE_GLYPH_SUSPECTED" in flags
    }
    assert marked == set(f.watermark_cells)
    assert len(table.rows) == 108 and len(table.headers) == 5
    assert (
        analysis_from_dict(json.loads(json.dumps(asdict(r.census.table_analysis))))
        == r.census.table_analysis
    )


def test_t_missing_fragment_never_bridges_the_gap():
    f = realistic_table(TableGeometry(15, 100, 32, 60, 45), removed_page=17)
    r = explore(f.document)
    assert len(r.census.table_analysis.tables) == 2
    assert not any(min(t.pages) < 17 < max(t.pages) for t in r.census.table_analysis.tables)
    assert all(finding.diagnostics.pages_spanned for finding in r.census.table_analysis.findings)
    assert all(
        finding.diagnostics.inferred_band_count is not None
        for finding in r.census.table_analysis.findings
    )


def test_t_unmatched_headerless_geometry_refuses():
    from dataclasses import replace

    f = realistic_table(TableGeometry(15, 100, 32, 60, 45))
    page = f.document.pages[13]
    tables = []
    for table in page.tables:
        if table.table_id != "body-fragment:14":
            tables.append(table)
            continue
        rows = tuple(
            tuple(
                replace(c, bbox=(c.bbox[0] + 10, c.bbox[1], c.bbox[2] + 10, c.bbox[3])) for c in row
            )
            for row in table.cell_rows
        )
        tables.append(replace(table, cell_rows=rows))
    doc = replace(
        f.document,
        pages=(*f.document.pages[:13], replace(page, tables=tuple(tables)), *f.document.pages[14:]),
    )
    r = explore(doc)
    assert not any(row.page == 14 for t in r.census.table_analysis.tables for row in t.rows)
    assert any(finding.page == 14 for finding in r.census.table_analysis.findings)


def test_t_conflicting_neighbor_headers_do_not_choose_one():
    from dataclasses import replace

    f = realistic_table(TableGeometry(15, 100, 32, 60, 45))
    pages = []
    for page in f.document.pages:
        if page.page not in (15, 16):
            pages.append(page)
            continue
        tables = []
        for table in page.tables:
            if not table.table_id.startswith("body-fragment:"):
                tables.append(table)
                continue
            header = tuple(
                replace(c, text="Alternative") if c.text == "Requirement" else c
                for c in table.cell_rows[0]
            )
            tables.append(replace(table, cell_rows=(header, *table.cell_rows[1:])))
        pages.append(replace(page, tables=tuple(tables)))
    r = explore(replace(f.document, pages=tuple(pages)))
    assert not any(row.page == 14 for t in r.census.table_analysis.tables for row in t.rows)
    assert any(
        finding.page == 14 and finding.reason == "HEADER_NOT_REPEATED"
        for finding in r.census.table_analysis.findings
    )


@pytest.mark.parametrize(
    "native,anchors",
    [
        (12, (0, 3, 6, 8, 10)),
        (13, (0, 3, 6, 9, 11)),
        (14, (0, 3, 6, 9, 12)),
        (15, (0, 3, 6, 9, 12)),
    ],
)
def test_t_raw_header_slots_follow_native_spanning_geometry(native, anchors):
    f = realistic_table(TableGeometry(native, 100, 32, 60, 45))
    table = next(t for t in f.document.pages[4].tables if t.table_id == "body-fragment:5")
    assert len(table.headers) == native
    assert tuple(i for i, text in enumerate(table.headers) if text) == anchors
    assert tuple(table.headers[i] for i in anchors) == (
        "Ref",
        "Requirement",
        "Criticality",
        "Spec ref",
        "Test ref",
    )
    assert len(table.cell_rows[0]) == 5
