"""X22: the explorer reads PyMuPDF's Table.rows once per table, with unchanged cell boxes (K9.22).

`Table.rows` is a property that re-sorts the cells and rebuilds every row on each access. The explorer used to read
it once per cell and once per span test, which dominated the parse of table-heavy PDFs. It is a pure function of
`table.cells`, so one read per table must give exactly the boxes the per-cell reads gave.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


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


def _ruled_table_pdf(path, tables=2):
    """Synthetic PDF: `tables` ruled grids of 4 rows x 3 columns, one cell left blank in each."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for t in range(tables):
        top = 80 + t * 300
        xs, ys = (60, 200, 340, 480), tuple(top + 40 * i for i in range(5))
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]), width=0.8)
        for y in ys:
            page.draw_line((xs[0], y), (xs[-1], y), width=0.8)
        for r in range(4):
            for c in range(3):
                if (r, c) == (2, 1):
                    continue
                label = "Col%d" % c if r == 0 else "T%dR%dC%d" % (t, r, c)
                page.insert_text((xs[c] + 6, ys[r] + 25), label, fontsize=10)
    doc.save(str(path))
    doc.close()


def _native_boxes(path):
    """Every (table ident, row, column) -> box, read independently with PyMuPDF, as the explorer defines them."""
    import pymupdf

    out = {}
    with pymupdf.open(str(path)) as doc:
        for number, page in enumerate(doc, 1):
            for index, table in enumerate(page.find_tables().tables):
                rows = table.extract()
                if not rows:
                    continue
                for r, row in enumerate(rows):
                    for c, cell in enumerate(row):
                        native = table.rows[r].cells[c]
                        if native is not None or cell:
                            out[(f"table:{number}:{index}", r, c)] = tuple(native or table.bbox)
    return out


def _exercise_boxes(tmp_path):
    from ragix_kernels.saqqara.explorer import digest_pdf

    path = tmp_path / "ruled.pdf"
    _ruled_table_pdf(path)
    expected = _native_boxes(path)
    assert expected, "the fixture must yield native table cells"
    cells, evidence = {}, {}
    for page in digest_pdf(path).pages:
        for table in page.tables:
            for row in table.cell_rows:
                for cell in row:
                    ident, r, c = cell.cell_id.rsplit(":", 2)
                    cells[(ident, int(r), int(c))] = tuple(cell.bbox)
            for ev in table.evidence:
                ident, r, c = ev.span_id.rsplit(":", 2)
                evidence[(ident, int(r), int(c))] = tuple(ev.bbox)
    assert cells == expected
    assert evidence == expected


def _exercise_reads(tmp_path):
    import pymupdf
    from pymupdf.table import Table

    from ragix_kernels.saqqara.explorer import digest_pdf

    path = tmp_path / "ruled.pdf"
    _ruled_table_pdf(path, tables=2)
    original = Table.__dict__["rows"].fget
    calls = []

    def counting(self):
        calls.append(id(self))
        return original(self)

    Table.rows = property(counting)
    # PyMuPDF reads Table.rows itself inside find_tables() and extract(); count those reads first, on the same file,
    # so the assertion isolates the explorer's own reads. The baseline calls find_tables() with its defaults, as
    # digest_pdf does today: if digest_pdf ever passes find_tables() options, pass the same options here.
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            for table in page.find_tables().tables:
                table.extract()
    internal = len(calls)
    del calls[:]
    digest = digest_pdf(path)
    tables = sum(len(p.tables) for p in digest.pages)
    assert tables >= 1
    own = len(calls) - internal
    assert own == tables, "the explorer read Table.rows %d times for %d tables" % (own, tables)


def test_x22a_cell_boxes_equal_the_native_rows(tmp_path):
    _run("_exercise_boxes", tmp_path)


def test_x22b_table_rows_is_read_once_per_table(tmp_path):
    _run("_exercise_reads", tmp_path)
