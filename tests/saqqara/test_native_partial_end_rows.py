"""X24: a native table row may lack cells at its ends where the source draws none (K9.24).

A ruled table often draws no cell over a blank corner (an empty first row over the value columns only) or over the
label column of a band row. PyMuPDF then reports that row without those cells. Every present cell still lies on the
grid the full-width rows draw, yet `_native_header_bands` refused the whole table because the row's outer extent
differed, so its cells fell back to line members and the table was read as small prose fragments.

A row lacking end cells is kept when its present cells are contiguous and it begins and ends on edges the full-width
rows draw; it takes no part in the band-edge intersection, and its cells are mapped to bands by their edges. Refused as
before: a partial row ending off that grid, a table with no full-width row, a row with a gap between present cells.
A table whose rows are all full width keeps exactly the bands it had.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import importlib.util
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from ragix_kernels.saqqara.table_views import RegionRefused, _native_header_bands

Cell = namedtuple("Cell", "bbox text")
XS = (50, 190, 260, 330, 400, 470, 550)


def row(columns, texts=None, xs=XS):
    """Cells over the given (start, end) column spans, contiguous by construction."""
    return tuple(Cell((xs[a], 0, xs[b], 10), (texts or {}).get(a, "")) for a, b in columns)


FULL = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6))
MERGED = ((0, 1), (1, 6))


# X24a -- the rule, on cells ----------------------------------------------------------------------------------

def test_x24a_rows_lacking_end_cells_on_the_grid_are_kept_and_left_out_of_the_intersection():
    header = row(((1, 2), (2, 3), (3, 4), (4, 5)))                  # blank corner: no label cell, no last cell
    body = (row(FULL), row(FULL), row(MERGED), row(((1, 6),)), row(FULL))   # (1, 6): a band with no label cell
    assert _native_header_bands(header, body) == ((50, 190), (190, 550)), "the full rows' coarsened bands"


def test_x24a_a_table_whose_rows_are_all_full_width_keeps_the_bands_it_had():
    rows = (row(FULL), row(FULL), row(MERGED), row(FULL))
    edges = sorted(set.intersection(*({v for c in r for v in (c.bbox[0], c.bbox[2])} for r in rows)))
    assert _native_header_bands(rows[0], rows[1:]) == tuple(zip(edges, edges[1:]))
    fine = (row(FULL), row(FULL))
    assert _native_header_bands(fine[0], fine[1:]) == tuple((XS[i], XS[i + 1]) for i in range(6))


@pytest.mark.parametrize("header, body", [
    (row(((1, 2), (2, 3))), (row(FULL), row(((0, 1), (1, 2)), xs=(50, 190, 275)))),
    ((Cell((75, 0, 190, 10), ""),) + row(((1, 2),)), (row(FULL), row(FULL))),
    (row(((0, 1), (1, 5))), (row(((1, 2), (2, 6))), row(((0, 1), (1, 5))))),
    (row(FULL), (row(((0, 1), (1, 2), (3, 4), (4, 6))), row(FULL)))],
    ids=["partial-row-ends-off-grid", "partial-row-starts-off-grid", "no-full-width-row", "gap-between-cells"])
def test_x24a_refused_as_before(header, body):
    with pytest.raises(RegionRefused) as refused:
        _native_header_bands(header, body)
    assert refused.value.code == "STRADDLING_OR_OUTSIDE_BANDS"


def test_x24a_a_labelled_header_with_two_labels_in_one_band_is_still_refused():
    header = row(((0, 1), (1, 2), (2, 3)), {1: "M1", 2: "M2"}, xs=(50, 190, 260, 330))
    body = (row(((0, 1), (1, 3)), xs=(50, 190, 260, 330)),)
    with pytest.raises(RegionRefused) as refused:
        _native_header_bands(header, body)
    assert refused.value.code == "BAND_COUNT_VARIES"


# X24b -- through the reader, on a synthetic PDF ------------------------------------------------------------

def _run(name, tmp_path):
    """Optional imports stay in a subprocess; default-route import guards (K6.8, K6.16) remain meaningful."""
    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional reader not installed")
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; from pathlib import Path; "
            "runpy.run_path(sys.argv[1])[sys.argv[2]](Path(sys.argv[3]))",
            str(Path(__file__).resolve()),
            name,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode:
        pytest.fail(done.stderr[-2000:])


def _partial_end_table_pdf(path, *, off_grid=False):
    """Synthetic ruled table, 6 columns: a blank first row drawn over the four middle columns only, a header row, data
    rows, rows whose value cells are merged, and a band row with no label cell. `off_grid` moves the blank row's left
    edge off the grid."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    kinds = ("blank", "header", "data", "data", "merged", "nolabel", "data", "merged")
    ys = [100 + 30 * i for i in range(len(kinds) + 1)]
    for i, kind in enumerate(kinds):
        top, bottom = ys[i], ys[i + 1]
        if kind == "blank":
            first = XS[1] + (20 if off_grid else 0)
            verts, left, right = (first,) + XS[2:6], first, XS[5]
        elif kind == "merged":
            verts, left, right = (XS[0], XS[1], XS[6]), XS[0], XS[6]
        elif kind == "nolabel":
            verts, left, right = (XS[1], XS[6]), XS[0], XS[6]
        else:
            verts, left, right = XS, XS[0], XS[6]
        page.draw_line((left, top), (right, top), width=0.8)
        page.draw_line((left, bottom), (right, bottom), width=0.8)
        for x in verts:
            page.draw_line((x, top), (x, bottom), width=0.8)
        texts = {"header": ["Model"] + ["M%d" % c for c in range(1, 6)],
                 "data": ["Param %d" % i] + ["%d.%d" % (i, c) for c in range(1, 6)],
                 "merged": ["Shared %d" % i, "common value %d" % i],
                 "nolabel": [None, "section band %d" % i]}.get(kind, [])
        for c, text in enumerate(texts):
            if text:
                page.insert_text((XS[c] + 4, top + 20), text, fontsize=8)
    doc.save(str(path))
    doc.close()


def _regions(path):
    from ragix_kernels.saqqara.explorer import digest_pdf, explore, imported_provenance
    from ragix_kernels.saqqara.renderable_regions import regions_from_explorer

    digest = digest_pdf(str(path))
    return digest, regions_from_explorer(explore(digest, provenance=imported_provenance(require_clean=False)))


def _check_one_table_region(tmp_path):
    path = tmp_path / "partial.pdf"
    _partial_end_table_pdf(path)
    digest, index = _regions(path)
    (table,) = [t for p in digest.pages for t in p.tables]
    counts = [len(r) for r in table.cell_rows]
    assert counts[0] == 4 and 1 in counts, "the native rows lack end cells, as the fixture draws them"
    assert index.refusals == ()
    (region,) = [r for r in index.regions if r.kind == "TABLE"]
    assert [r.kind for r in index.regions if r.kind != "TABLE"] == []
    cells = [m for m in region.members if m.kind == "CELL"]
    assert len(cells) == sum(counts), "every present cell is a member"
    band_row = [m for m in cells if m.row == counts.index(1)]
    assert [(m.column, m.bbox[0], m.bbox[2]) for m in band_row] == [(1, XS[1], XS[6])], \
        "the band row's cell lands in the value columns it spans, by its edges"
    blank = [m for m in cells if m.row == 0]
    assert len(blank) == 4 and all(not (m.text or "").strip() and not m.is_header for m in blank), \
        "the blank first row carries no text and is never made a header"
    assert not any(m.is_header and not (m.text or "").strip() for m in cells), "no header label is invented"


def _check_off_grid_still_refused(tmp_path):
    path = tmp_path / "off_grid.pdf"
    _partial_end_table_pdf(path, off_grid=True)
    _, index = _regions(path)
    assert [r.code for r in index.refusals] == ["STRADDLING_OR_OUTSIDE_BANDS"]
    assert not [r for r in index.regions if r.kind == "TABLE"]


def test_x24b_partial_end_rows_give_one_table_region_with_edge_mapped_cells(tmp_path):
    _run("_check_one_table_region", tmp_path)


def test_x24b_a_partial_row_off_the_grid_is_still_refused(tmp_path):
    _run("_check_off_grid_still_refused", tmp_path)
