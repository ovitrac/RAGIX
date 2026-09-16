"""X13: planted word cells, stable columns, recurrence ordering and refusals.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace
from itertools import product
import pytest
from ragix_kernels.saqqara.census import (
    DocumentDigest,
    PageDigest,
    TableObservation,
    Evidence,
    CensusConfig,
)
from ragix_kernels.saqqara.field_views import TextSpan, VerticalRule
from ragix_kernels.saqqara.table_views import TableCell, TablePolicy
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.saqqara.profile import ProfileConfig


def cell(value, page, ident, x, y, width=None, flags=()):
    return TableCell(
        f"cell:{page}:{ident}",
        value,
        (x, y, x + (width or max(8, 4 * len(value or ""))), y + 8),
        (f"s:{page}:{ident}",),
        flags,
    )


def observations(cells, page):
    return tuple(
        TextSpan(
            "synthetic-tables",
            c.source_spans[0],
            page,
            c.text or "",
            c.bbox,
            origin=(c.bbox[0], c.bbox[3]),
            font_size=8,
        )
        for c in cells
    )


def block(ident, page, rows):
    evidence = tuple(
        Evidence("synthetic-tables", page, c.cell_id, 0, len(c.text or ""), c.text or "", c.bbox)
        for row in rows
        for c in row
    )
    return TableObservation(
        ident,
        page,
        tuple(c.text or "" for c in rows[0]),
        (),
        evidence,
        cell_rows=tuple(tuple(r) for r in rows),
    )


def fixture(columns=5, ruled=True, language="en", running=True, order=None, contamination=False):
    pages = []
    expected = []
    offset = 0
    order = tuple(range(columns)) if order is None else tuple(order)
    for page, count in enumerate((14, 13, 13), 1):
        header = []
        rows = []
        spans = []
        tables = []
        rules = []
        for col, original in enumerate(order):
            for wordno, word in enumerate(
                (
                    "Col" if language == "en" else "Champ",
                    chr(65 + original),
                    "Title" if language == "en" else "Titre",
                )
            ):
                header.append(
                    cell(word, page, f"h:{col}:{wordno}", 20 + col * 120 + 8 + wordno * 28, 200)
                )
        for i in range(count):
            raw = []
            want = []
            for col, original in enumerate(order):
                x = 20 + col * 120 + 8
                y = 230 + i * 30
                name = f"r:{i}:{col}"
                if original == 0:
                    value = f"8.{offset+i+1}"
                    raw.append(cell(value, page, name, x, y))
                    want.append(value)
                elif original == columns - 1:
                    raw.append(cell("", page, name, x, y, 28))
                    want.append("")
                elif original == columns - 2 and columns >= 4:
                    raw.append(cell("OK", page, name, x, y))
                    want.append("OK")
                else:
                    raw.extend(
                        (
                            cell("alpha", page, name + ":a", x, y),
                            cell("beta", page, name + ":b", x + 24, y),
                        )
                    )
                    value = "alpha beta"
                    if (offset + i) % 7 == 0:
                        raw.append(cell("gamma", page, name + ":c", x, y + 14))
                        value += "\ngamma"
                    if contamination and page == 2 and i == 1:
                        raw[-1] = replace(
                            raw[-1],
                            text=raw[-1].text + " MARK",
                            flags=("LIFECYCLE_GLYPH_SUSPECTED",),
                        )
                    want.append(value)
            rows.append(raw)
            expected.append(tuple(want))
        body = block(f"body:{page}", page, [header, *rows])
        tables.append(body)
        all_cells = [c for r in body.cell_rows for c in r]
        spans.extend(observations(all_cells, page))
        if ruled:
            rules.extend(
                VerticalRule(20 + i * 120, 190, 230 + count * 30) for i in range(columns + 1)
            )
        if running:
            fr = [[cell("Header", page, "fh:0", 20, 10), cell("Block", page, "fh:1", 150, 10)]]
            for i in range(5):
                fr.append(
                    [
                        cell("DOC-F-731", page, f"f:{i}:0", 20, 20 + i * 10),
                        cell("rev", page, f"f:{i}:1", 150, 20 + i * 10),
                    ]
                )
            furniture = block(f"furniture:{page}", page, fr)
            tables.insert(0, furniture)
            spans.extend(observations([c for r in fr for c in r], page))
        pages.append(
            PageDigest(page, columns * 120 + 60, 1000, tuple(spans), tuple(rules), tuple(tables))
        )
        offset += count
    return DocumentDigest("synthetic-tables", tuple(pages), "synthetic", "1"), tuple(expected)


@pytest.mark.parametrize("columns", [3, 5, 8])
@pytest.mark.parametrize("ruled", [False, True])
@pytest.mark.parametrize("language", ["fr", "en"])
def test_e9_1_2_3_5_planted_cells_and_continuation(columns, ruled, language):
    doc, expected = fixture(columns, ruled, language)
    result = explore(doc)
    assert not result.census.table_analysis.findings
    (table,) = result.census.table_analysis.tables
    assert (
        len(table.headers) == columns and len(doc.pages[0].tables[-1].cell_rows[0]) == 3 * columns
    )
    assert tuple(row.cells for row in table.rows) == expected
    assert len(table.rows) == 40 and table.continued_on == 3 and table.repetition_count == 2
    assert set(result.census.table_analysis.excluded) == {
        "furniture:1",
        "furniture:2",
        "furniture:3",
    }
    assert len(result.reading.tables) == 40 and result.report.coverage.logical_tables == 1
    assert (
        result.report.coverage.tables_recovered == 3
        and result.report.coverage.tables_not_recovered == 0
    )


def test_e9_4_straddling_and_varying_band_counts_refuse_without_cleaning():
    doc, _ = fixture(5, True)
    page = doc.pages[0]
    table = page.tables[-1]
    rows = list(table.cell_rows)
    row = list(rows[2])
    row[0] = replace(row[0], bbox=(28, row[0].bbox[1], 170, row[0].bbox[3]))
    rows[2] = tuple(row)
    changed = replace(table, cell_rows=tuple(rows))
    mutated = replace(doc, pages=(replace(page, tables=(page.tables[0], changed)), *doc.pages[1:]))
    result = explore(mutated)
    assert any(
        f.reason == "STRADDLING_OR_OUTSIDE_BANDS" for f in result.census.table_analysis.findings
    )
    assert not any(r.page == 1 for t in result.census.table_analysis.tables for r in t.rows)
    assert result.document.pages[0].spans == doc.pages[0].spans
    doc, _ = fixture(5, False)
    page = doc.pages[0]
    table = page.tables[-1]
    rows = list(table.cell_rows)
    rows[2] = tuple(c for c in rows[2] if c.bbox[0] < 500)
    result = explore(
        replace(
            doc,
            pages=(
                replace(page, tables=(page.tables[0], replace(table, cell_rows=tuple(rows)))),
                *doc.pages[1:],
            ),
        )
    )
    assert any(f.reason == "BAND_COUNT_VARIES" for f in result.census.table_analysis.findings)


def test_e9_6_reordering_and_contamination_are_explicit():
    doc, expected = fixture(5, True, order=(4, 0, 2, 1, 3))
    result = explore(doc)
    assert tuple(r.cells for t in result.census.table_analysis.tables for r in t.rows) == expected
    assert result.census.table_analysis.tables[0].roles[1] == "id"
    doc, _ = fixture(5, True, contamination=True)
    result = explore(doc)
    rows = [r for t in result.census.table_analysis.tables for r in t.rows if r.flags]
    assert rows and any("MARK" in (c or "") for r in rows for c in r.cells)
    assert all("LIFECYCLE_GLYPH_SUSPECTED" in r.flags for r in rows)


@pytest.mark.parametrize(
    "factor,minimum,share", list(product((0.25, 0.5, 1), (2, 3, 4, 5), (0.4, 0.5, 0.6)))
)
def test_e9_7_sensitivity(factor, minimum, share):
    doc, expected = fixture(5, False)
    result = explore(
        doc,
        census_config=CensusConfig(
            recurrence_fraction=share, table_policy=TablePolicy(factor, minimum)
        ),
        profile_config=ProfileConfig(recurrence_fraction=share),
    )
    (table,) = result.census.table_analysis.tables
    assert tuple(r.cells for r in table.rows) == expected
    assert len(table.headers) == 5 and len(result.census.table_analysis.excluded) == 3
    assert table.policy["source"] == "derived" and table.policy["x_tolerance_factor"] == factor


def test_unruled_header_only_band_cannot_be_merged_into_last_column():
    doc, _ = fixture(5, False)
    pages = []
    for page in doc.pages:
        table = page.tables[-1]
        rows = (
            table.cell_rows[0],
            *(tuple(c for c in row if c.bbox[0] < 500) for row in table.cell_rows[1:]),
        )
        pages.append(replace(page, tables=(page.tables[0], replace(table, cell_rows=rows))))
    result = explore(replace(doc, pages=tuple(pages)))
    assert not result.census.table_analysis.tables
    assert len(result.census.table_analysis.findings) == 3


def test_duplicate_header_literals_do_not_overwrite_cells():
    doc, expected = fixture(5, True)
    pages = []
    for page in doc.pages:
        table = page.tables[-1]
        header = list(table.cell_rows[0])
        for i in range(3):
            header[3 + i] = replace(header[3 + i], text=header[i].text)
        pages.append(
            replace(
                page,
                tables=(
                    page.tables[0],
                    replace(table, cell_rows=(tuple(header), *table.cell_rows[1:])),
                ),
            )
        )
    result = explore(replace(doc, pages=tuple(pages)))
    assert len(result.reading.tables) == 40
    for row, want in zip(result.reading.tables, expected):
        assert len(row["cells"]) == 5 and tuple(row["cells"].values()) == want
        assert row["columns"][0]["header"] == row["columns"][1]["header"]


def test_raw_table_roundtrip_and_kernel_library_parity(tmp_path):
    from dataclasses import asdict
    from ragix_kernels.saqqara.census import digest_from_dict
    from ragix_kernels.saqqara.kernels.explorer import (
        CensusKernel,
        ProfileKernel,
        ReadKernel,
        ReportKernel,
    )
    from ragix_kernels.base import KernelInput
    from ragix_kernels.harvest.report import canonical_json

    doc, _ = fixture(3, False)
    assert digest_from_dict(asdict(doc)) == doc
    previous = None
    for cls in (CensusKernel, ProfileKernel, ReadKernel, ReportKernel):
        config = {"documents": [asdict(doc)]} if previous is None else {}
        deps = {} if previous is None else {cls.requires[0]: previous.output_file}
        previous = cls().run(KernelInput(tmp_path, config, deps))
        assert previous.success, previous.errors
    assert canonical_json(previous.data["reports"][0]) == canonical_json(explore(doc).report)


def test_header_contamination_is_retained_and_propagated():
    doc, _ = fixture(5, True)
    pages = []
    for page in doc.pages:
        table = page.tables[-1]
        header = list(table.cell_rows[0])
        header[0] = replace(
            header[0], text=header[0].text + " MARK", flags=("LIFECYCLE_GLYPH_SUSPECTED",)
        )
        pages.append(
            replace(
                page,
                tables=(
                    page.tables[0],
                    replace(table, cell_rows=(tuple(header), *table.cell_rows[1:])),
                ),
            )
        )
    result = explore(replace(doc, pages=tuple(pages)))
    (table,) = result.census.table_analysis.tables
    assert "MARK" in table.headers[0] and "LIFECYCLE_GLYPH_SUSPECTED" in table.flags
    assert all("LIFECYCLE_GLYPH_SUSPECTED" in row["flags"] for row in result.reading.tables)


def test_table_geometry_replay_normalizes_small_extractor_jitter():
    from ragix_kernels.harvest.report import replay_digest

    doc, _ = fixture(5, False)

    def perturb(epsilon):
        return replace(
            doc,
            pages=tuple(
                replace(
                    page,
                    tables=tuple(
                        replace(
                            table,
                            cell_rows=tuple(
                                tuple(
                                    replace(c, bbox=tuple(float(v) + epsilon for v in c.bbox))
                                    for c in row
                                )
                                for row in table.cell_rows
                            ),
                        )
                        for table in page.tables
                    ),
                )
                for page in doc.pages
            ),
        )

    a = explore(perturb(0.00001))
    b = explore(perturb(0.00002))
    assert replay_digest([a.census]) == replay_digest([b.census])
    assert a.report.replay_digest == b.report.replay_digest


def test_table_record_and_policy_fail_closed():
    from ragix_kernels.saqqara.table_views import TableFinding, analysis_from_dict
    from dataclasses import asdict

    with pytest.raises(ValueError):
        TablePolicy(x_tolerance_factor=True)
    with pytest.raises(ValueError):
        TableFinding("f", "t", 1, "missing", 0, ())
    doc, _ = fixture(3, True)
    data = asdict(explore(doc).census.table_analysis)
    data["tables"][0]["roles"] = ("unsupported", "free_text", "empty")
    with pytest.raises(ValueError):
        analysis_from_dict(data)


def test_native_pdf_tables_and_privacy_work_together(tmp_path):
    import importlib.util, json, subprocess, sys

    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional PDF reader")
    code = """import json,sys,pymupdf
from pathlib import Path
from ragix_kernels.saqqara.explorer import digest_pdf,explore
from ragix_kernels.harvest.report import render_report
pdf=pymupdf.open()
for p in range(2):
 page=pdf.new_page()
 page.insert_text((30,25),"Export: 2033-07-19 Aster Quorin",fontsize=10)
 for x in (20,100,360,440): page.draw_line((x,180),(x,390))
 for y in range(180,391,30): page.draw_line((20,y),(440,y))
 for x,t in zip((28,108,368),("Key","Description","Result")): page.insert_text((x,200),t,fontsize=10)
 for r in range(6):
  page.insert_text((28,230+r*30),f"8.{p*6+r+1}",fontsize=10)
  page.insert_text((108,230+r*30),"plain text",fontsize=10)
path=Path(sys.argv[1]);pdf.save(path,no_new_id=True);pdf.close()
result=explore(digest_pdf(path)); html=render_report(result.report,{"field":"field","quantity":"quantity","table_row":"row"})
print(json.dumps({"tables":len(result.census.table_analysis.tables),"rows":len(result.reading.tables),"masked":result.report.provenance["masked_lines"],"leak":"Quorin" in html}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "synthetic.pdf")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "tables": 1,
        "rows": 12,
        "masked": 2,
        "leak": False,
    }


def test_furniture_word_mutation_leaves_body_table_profile_and_rows_unchanged():
    doc, expected = fixture(5, True)
    original = explore(doc)
    pages = []

    def renamed(value):
        return (
            value.replace("Header", "Banner")
            .replace("Block", "Frame")
            .replace("DOC-F-731", "RUN-G-942")
        )

    for page in doc.pages:
        old = page.tables[0]
        rows = tuple(tuple(replace(c, text=renamed(c.text)) for c in row) for row in old.cell_rows)
        table = replace(
            old,
            headers=tuple(c.text for c in rows[0]),
            cell_rows=rows,
            evidence=tuple(
                replace(e, literal=renamed(e.literal), end=len(renamed(e.literal)))
                for e in old.evidence
            ),
        )
        spans = tuple(replace(s, text=renamed(s.text)) if s.bbox[1] < 80 else s for s in page.spans)
        pages.append(replace(page, spans=spans, tables=(table, page.tables[-1])))
    changed = explore(replace(doc, pages=tuple(pages)))
    assert tuple(r.cells for t in changed.census.table_analysis.tables for r in t.rows) == expected
    assert (
        original.profile.fields["id_row_tables"].value
        == changed.profile.fields["id_row_tables"].value
    )
    assert original.profile.fields["furniture"].value != changed.profile.fields["furniture"].value


def test_continuation_survives_observed_empty_to_code_cells():
    doc, _ = fixture(5, True)
    page = doc.pages[1]
    table = page.tables[-1]
    rows = (
        table.cell_rows[0],
        *(
            tuple(replace(c, text="OK") if c.text == "" else c for c in row)
            for row in table.cell_rows[1:]
        ),
    )
    doc = replace(
        doc,
        pages=(
            doc.pages[0],
            replace(page, tables=(page.tables[0], replace(table, cell_rows=rows))),
            doc.pages[2],
        ),
    )
    result = explore(doc)
    (recovered,) = result.census.table_analysis.tables
    assert recovered.continued_on == 3 and recovered.roles[-1] == "short_code"
    assert sum(r.cells[-1] == "OK" for r in recovered.rows) == 13
    assert sum(r.cells[-1] == "" for r in recovered.rows) == 27
