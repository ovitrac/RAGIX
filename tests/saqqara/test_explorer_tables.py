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


def fixture(
    columns=5,
    ruled=True,
    language="en",
    running=True,
    order=None,
    contamination=False,
    row_counts=(14, 13, 13),
):
    pages = []
    expected = []
    offset = 0
    order = tuple(range(columns)) if order is None else tuple(order)
    for page, count in enumerate(row_counts, 1):
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


def short_fragments(ruled=True, segmented=False, row_counts=(1, 1, 1, 1, 1, 1)):
    """Independent short-fragment control; no consumer text or geometry."""
    doc, expected = fixture(3, ruled, row_counts=row_counts)
    if segmented:
        pages = []
        for page in doc.pages:
            rules = tuple(
                segment
                for rule in page.rules
                for segment in (
                    replace(rule, bottom=(rule.top + rule.bottom) / 2),
                    replace(rule, top=(rule.top + rule.bottom) / 2),
                )
            )
            pages.append(replace(page, rules=rules))
        doc = replace(doc, pages=tuple(pages))
    return doc, expected


@pytest.mark.parametrize("ruled,segmented", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "factor,minimum,share", list(product((0.25, 0.5, 1), (2, 3, 4, 5), (0.4, 0.5, 0.6)))
)
def test_short_fragments_pool_support_after_continuation(ruled, segmented, factor, minimum, share):
    doc, expected = short_fragments(ruled, segmented)
    result = explore(
        doc,
        census_config=CensusConfig(
            recurrence_fraction=share, table_policy=TablePolicy(factor, minimum)
        ),
        profile_config=ProfileConfig(recurrence_fraction=share),
    )
    assert not result.census.table_analysis.findings
    (table,) = result.census.table_analysis.tables
    assert tuple(r.cells for r in table.rows) == expected
    assert table.continued_on == 6 and table.repetition_count == 5
    assert len(result.reading.tables) == 6
    assert len({r.record_id for r in table.rows}) == 6
    assert all(m for r in table.rows for m in r.members)


def test_short_fragments_do_not_borrow_support_across_missing_pages():
    doc, _ = short_fragments()
    # Each adjacent chain has fewer supporting rows than the configured minimum.
    doc = replace(
        doc,
        pages=tuple(
            replace(p, tables=(), spans=(), rules=()) if p.page in (3, 6) else p for p in doc.pages
        ),
    )
    result = explore(doc, census_config=CensusConfig(table_policy=TablePolicy(minimum_rows=3)))
    assert not result.census.table_analysis.tables
    assert len(result.census.table_analysis.findings) == 4
    assert {f.reason for f in result.census.table_analysis.findings} == {"TOO_FEW_ROWS"}


def test_segmented_grid_recovers_touching_physical_cells():
    doc, _ = short_fragments(segmented=True, row_counts=(3, 3))
    pages = []
    expected = []
    for page in doc.pages:
        table = page.tables[-1]
        rows = []
        for i, source in enumerate(table.cell_rows):
            row = []
            for col in range(3):
                members = [c for c in source if 20 + col * 120 <= c.bbox[0] < 20 + (col + 1) * 120]
                text = " ".join(c.text for c in members)
                y = 190 + i * 20
                row.append(
                    TableCell(
                        f"physical:{page.page}:{i}:{col}",
                        text,
                        (20 + col * 120, y, 20 + (col + 1) * 120, y + 20),
                        tuple(s for c in members for s in c.source_spans),
                    )
                )
            rows.append(tuple(row))
            if i:
                expected.append(tuple(c.text for c in row))
        pages.append(replace(page, tables=(page.tables[0], replace(table, cell_rows=tuple(rows)))))
    result = explore(replace(doc, pages=tuple(pages)))
    assert not result.census.table_analysis.findings
    (table,) = result.census.table_analysis.tables
    assert table.policy["route"] == "grid"
    assert tuple(r.cells for r in table.rows) == tuple(expected)


def test_short_fragments_do_not_borrow_identifier_support_from_other_columns():
    doc, _ = short_fragments()
    pages = []
    for page in doc.pages:
        table = page.tables[-1]
        rows = list(table.cell_rows)
        if page.page % 2 == 0:
            row = list(rows[1])
            value = row[0].text
            row[0] = replace(row[0], text="words")
            row[-1] = replace(row[-1], text=value)
            rows[1] = tuple(row)
        pages.append(replace(page, tables=(page.tables[0], replace(table, cell_rows=tuple(rows)))))
    result = explore(replace(doc, pages=tuple(pages)))
    assert not result.census.table_analysis.tables
    assert len(result.census.table_analysis.findings) == 6


@pytest.mark.parametrize("ruled", [True, False])
def test_mixed_length_fragments_retain_every_physical_row(ruled):
    doc, expected = short_fragments(ruled, row_counts=(1, 4, 2, 1, 3, 2, 4, 1, 2))
    result = explore(doc, census_config=CensusConfig(table_policy=TablePolicy(minimum_rows=5)))
    (table,) = result.census.table_analysis.tables
    assert table.continued_on == 9
    assert tuple(row.cells for row in table.rows) == expected
    assert len(result.reading.tables) == len(expected)


def test_short_fragments_refuse_ambiguous_same_page_continuations():
    doc, _ = short_fragments(row_counts=(1, 1))
    page = doc.pages[1]
    table = page.tables[-1]
    duplicate = replace(table, table_id="other-body:2")
    result = explore(
        replace(doc, pages=(doc.pages[0], replace(page, tables=(*page.tables, duplicate))))
    )
    assert not result.census.table_analysis.tables
    assert len(result.census.table_analysis.findings) == 3


def test_collinear_rule_union_does_not_bridge_a_gap():
    from ragix_kernels.saqqara.table_views import _covering_rule_positions

    assert _covering_rule_positions((VerticalRule(10, 0, 15), VerticalRule(10, 15, 30)), 5, 25) == [
        10
    ]
    assert (
        _covering_rule_positions((VerticalRule(10, 0, 15), VerticalRule(10, 16, 30)), 5, 25) == []
    )
    assert _covering_rule_positions((VerticalRule(10, 0, 17), VerticalRule(10, 15, 30)), 5, 25) == [
        10
    ]


def test_native_pdf_one_row_fragments_with_cell_rectangle_borders(tmp_path):
    import importlib.util, json, subprocess, sys

    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional PDF reader")
    code = """import json,sys,pymupdf
from pathlib import Path
from ragix_kernels.saqqara.explorer import digest_pdf,explore
from ragix_kernels.saqqara.census import CensusConfig
from ragix_kernels.saqqara.table_views import TablePolicy
pdf=pymupdf.open()
for p in range(6):
 page=pdf.new_page()
 for y in (180,210):
  for left,right in ((20,100),(100,360),(360,440)):
   page.draw_rect((left,y,right,y+30))
 for x,t in zip((28,108,368),("Key","Description","Result")): page.insert_text((x,200),t,fontsize=10)
 page.insert_text((28,230),f"9.{p+1}",fontsize=10)
 page.insert_text((108,230),"plain text",fontsize=10)
path=Path(sys.argv[1]);pdf.save(path,no_new_id=True);pdf.close()
result=explore(digest_pdf(path),census_config=CensusConfig(table_policy=TablePolicy(minimum_rows=5)))
print(json.dumps({"tables":len(result.census.table_analysis.tables),"rows":len(result.reading.tables),
 "findings":[f.reason for f in result.census.table_analysis.findings],
 "cells":[r.cells for t in result.census.table_analysis.tables for r in t.rows],
 "pages":[t.pages for t in result.census.table_analysis.tables]}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fragmented.pdf")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "tables": 1,
        "rows": 6,
        "findings": [],
        "pages": [[1, 2, 3, 4, 5, 6]],
        "cells": [[f"9.{i}", "plain text", ""] for i in range(1, 7)],
    }


def padded_header_pdf(path, columns=3, pages=6):
    """Native cells: header padding creates subslots; body cells span them."""
    import pymupdf

    pdf = pymupdf.open()
    for p in range(pages):
        page = pdf.new_page(width=columns * 120 + 40, height=600)
        for col in range(columns):
            x = 20 + col * 120
            for left, right in ((x, x + 12), (x + 12, x + 108), (x + 108, x + 120)):
                page.draw_rect((left, 180, right, 210))
            page.draw_rect((x, 210, x + 120, 270))
            page.insert_text((x + 18, 200), f"Column {col}", fontsize=9)
            text = f"6.{p+1}" if col == 0 else "" if col == columns - 1 else "body text"
            if text:
                page.insert_text((x + 18, 230), text, fontsize=9)
    pdf.save(path, no_new_id=True)
    pdf.close()


@pytest.mark.parametrize("columns", [3, 5, 8])
def test_native_padded_header_collapse_preserves_real_empty_cells(tmp_path, columns):
    import importlib.util, json, subprocess, sys

    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional PDF reader")
    code = """import json,sys,importlib.util
from pathlib import Path
from dataclasses import asdict
from ragix_kernels.saqqara.explorer import digest_pdf,explore
from ragix_kernels.saqqara.census import digest_from_dict
spec=importlib.util.spec_from_file_location('fixture',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
path=Path(sys.argv[2]);columns=int(sys.argv[3]);m.padded_header_pdf(path,columns)
doc=digest_pdf(path)
assert digest_from_dict(asdict(doc))==doc
raw=doc.pages[0].tables[0]
result=explore(doc)
print(json.dumps({'raw_header':len(raw.headers),'raw_body':len(raw.rows[0]),
 'physical_header':len(raw.cell_rows[0]),'physical_body':len(raw.cell_rows[1]),
 'unlocated':sum(v is None for v in raw.rows[0]),
 'rows':[r.cells for t in result.census.table_analysis.tables for r in t.rows],
 'tables':len(result.census.table_analysis.tables),
 'findings':[f.reason for f in result.census.table_analysis.findings],
 'flags':[r.flags for t in result.census.table_analysis.tables for r in t.rows]}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, __file__, str(tmp_path / "padded.pdf"), str(columns)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout.splitlines()[-1])
    assert data["raw_header"] == data["raw_body"] == data["physical_header"] == columns * 3
    assert data["unlocated"] == columns * 2
    assert data["physical_body"] == columns
    assert data["tables"] == 1 and data["findings"] == []
    assert data["rows"] == [[f"6.{i}", *(["body text"] * (columns - 2)), ""] for i in range(1, 7)]
    assert all("UNREADABLE_CELL" not in flags for flags in data["flags"])


def cell_box_fixture(columns=5):
    doc, expected = short_fragments()
    pages = []
    expected = []
    for page in doc.pages:
        header, body = [], []
        raw_body = []
        for col in range(columns):
            x = 20 + col * 120
            for part, (left, right, text) in enumerate(
                ((x, x + 12, ""), (x + 12, x + 108, f"Column {col}"), (x + 108, x + 120, ""))
            ):
                header.append(
                    TableCell(
                        f"hc:{page.page}:{col}:{part}",
                        text,
                        (left, 180, right, 210),
                        (f"hs:{page.page}:{col}:{part}",),
                        geometry_kind="cell_box",
                    )
                )
            value = f"6.{page.page}" if col == 0 else "" if col == columns - 1 else "body text"
            body.append(
                TableCell(
                    f"bc:{page.page}:{col}",
                    value,
                    (x, 210, x + 120, 270),
                    (f"bs:{page.page}:{col}",),
                    geometry_kind="cell_box",
                )
            )
            raw_body.extend((value, None, None))
        table = block(f"native:{page.page}", page.page, (header, body))
        table = replace(table, rows=(tuple(raw_body),))
        pages.append(
            replace(
                page,
                width=columns * 120 + 60,
                tables=(table,),
                spans=observations(header + body, page.page),
                rules=(),
            )
        )
        expected.append(tuple(c.text for c in body))
    return replace(doc, pages=tuple(pages)), tuple(expected)


@pytest.mark.parametrize(
    "factor,minimum,share", list(product((0.25, 0.5, 1), (2, 3, 4, 5), (0.4, 0.5, 0.6)))
)
def test_native_cell_collapse_sensitivity(factor, minimum, share):
    doc, expected = cell_box_fixture()
    result = explore(
        doc,
        census_config=CensusConfig(
            recurrence_fraction=share, table_policy=TablePolicy(factor, minimum)
        ),
        profile_config=ProfileConfig(recurrence_fraction=share),
    )
    (table,) = result.census.table_analysis.tables
    assert tuple(r.cells for r in table.rows) == expected
    assert table.policy["route"] == "native_cell_bounds"
    assert len(table.headers) == 5 and all(len(row.members) == 5 for row in table.rows)


def test_native_cell_unknown_text_remains_unreadable_not_an_empty_slot():
    doc, _ = cell_box_fixture()
    pages = []
    for page in doc.pages:
        table = page.tables[0]
        row = (*table.cell_rows[1][:-1], replace(table.cell_rows[1][-1], text=None))
        pages.append(replace(page, tables=(replace(table, cell_rows=(table.cell_rows[0], row)),)))
    result = explore(replace(doc, pages=tuple(pages)))
    (table,) = result.census.table_analysis.tables
    assert len(table.rows) == 6 and table.roles[-1] == "unknown"
    assert all(r.cells[-1] is None and "UNREADABLE_CELL" in r.flags for r in table.rows)


def test_native_header_straddling_refuses_with_band_diagnostics_and_roundtrip():
    from dataclasses import asdict
    from ragix_kernels.saqqara.table_views import analysis_from_dict

    doc, _ = cell_box_fixture()
    pages = []
    for page in doc.pages:
        table = page.tables[0]
        header = list(table.cell_rows[0])
        header[1] = replace(header[1], bbox=(32, 180, 150, 210))
        pages.append(
            replace(page, tables=(replace(table, cell_rows=(tuple(header), *table.cell_rows[1:])),))
        )
    result = explore(replace(doc, pages=tuple(pages)))
    analysis = result.census.table_analysis
    assert not analysis.tables and len(analysis.findings) == 6
    for finding in analysis.findings:
        assert finding.reason == "STRADDLING_OR_OUTSIDE_BANDS"
        d = finding.diagnostics
        assert d.stage == "bands" and d.route == "native_cell_bounds"
        assert d.raw_row_cell_counts == (15, 15) and d.retained_row_cell_counts == (15, 5)
        assert d.unlocated_slot_count == 10 and d.inferred_band_count is None
        assert d.row_band_counts == (5,)
    import json

    assert analysis_from_dict(json.loads(json.dumps(asdict(analysis)))) == analysis


def test_no_geometry_with_nonempty_text_is_counted_and_refused():
    doc, _ = cell_box_fixture()
    page = doc.pages[0]
    table = page.tables[0]
    row = (
        *table.cell_rows[1][:-1],
        replace(table.cell_rows[1][-1], text="unlocated", flags=("MISSING_CELL_GEOMETRY",)),
    )
    changed = replace(table, cell_rows=(table.cell_rows[0], row))
    result = explore(replace(doc, pages=(replace(page, tables=(changed,)), *doc.pages[1:])))
    (failure,) = result.census.table_analysis.findings
    assert failure.page == 1 and failure.reason == "MISSING_CELL_GEOMETRY"
    assert failure.diagnostics.stage == "candidate"
    assert result.document.pages[0].tables[0].cell_rows[1][-1].text == "unlocated"


def test_old_table_findings_without_diagnostics_still_load():
    from dataclasses import asdict
    from ragix_kernels.saqqara.table_views import TableFinding, TableAnalysis, analysis_from_dict

    finding = TableFinding("f", "t", 1, "TOO_FEW_ROWS", 1, ("c",))
    data = asdict(TableAnalysis(findings=(finding,)))
    del data["findings"][0]["diagnostics"]
    assert analysis_from_dict(data).findings == (finding,)


def test_native_empty_cell_geometry_survives_overlapping_rotated_furniture():
    doc, expected = cell_box_fixture()
    pages = []
    for page in doc.pages:
        table = page.tables[0]
        empty = table.cell_rows[1][-1]
        mark = TextSpan(
            doc.source_id,
            f"mark:{page.page}",
            page.page,
            "DRAFT",
            (empty.bbox[0] + 20, 220, empty.bbox[2] - 20, 245),
            direction=(0.6, -0.8),
            font_size=48,
        )
        row = (*table.cell_rows[1][:-1], replace(empty, source_spans=(mark.span_id,)))
        pages.append(
            replace(
                page,
                tables=(replace(table, cell_rows=(table.cell_rows[0], row)),),
                spans=(*page.spans, mark),
            )
        )
    result = explore(replace(doc, pages=tuple(pages)))
    (table,) = result.census.table_analysis.tables
    assert tuple(r.cells for r in table.rows) == expected
    assert all("LIFECYCLE_GLYPH_SUSPECTED" in r.flags for r in table.rows)


def test_whole_native_furniture_blocks_remain_excluded():
    doc, expected = cell_box_fixture()
    pages = []
    for page in doc.pages:
        rows = []
        for i in range(3):
            rows.append(
                tuple(
                    TableCell(
                        f"f:{page.page}:{i}:{j}",
                        text,
                        (20 + j * 120, 10 + i * 20, 140 + j * 120, 30 + i * 20),
                        (f"fs:{page.page}:{i}:{j}",),
                        geometry_kind="cell_box",
                    )
                    for j, text in enumerate(("Banner", "RUN-Z-004", ""))
                )
            )
        furniture = block(f"furniture:{page.page}", page.page, rows)
        furniture = replace(furniture, rows=tuple(tuple(c.text for c in row) for row in rows[1:]))
        pages.append(
            replace(
                page,
                tables=(furniture, *page.tables),
                spans=(*page.spans, *observations([c for row in rows for c in row], page.page)),
            )
        )
    result = explore(replace(doc, pages=tuple(pages)))
    assert set(result.census.table_analysis.excluded) == {f"furniture:{p}" for p in range(1, 7)}
    assert tuple(r.cells for t in result.census.table_analysis.tables for r in t.rows) == expected


def test_table_geometry_diagnostics_reject_invalid_counts_and_stages():
    from ragix_kernels.saqqara.table_views import TableDiagnostics

    for kwargs in (
        {"pooled_rows": -1},
        {"stage": "guess"},
        {"route": "guess"},
        {"inferred_band_count": 3},
        {"raw_row_cell_counts": (True,)},
        {"inferred_band_count": 1, "bands": ((20, 10),)},
        {"pages_spanned": (0,)},
    ):
        with pytest.raises(ValueError):
            TableDiagnostics(**kwargs)


@pytest.mark.parametrize("pages", [15, 23, 30])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("ruled", [False, True])
def test_e9_5b_long_continuations(pages, mixed, ruled):
    counts = tuple(1 + i % 5 if mixed else 1 for i in range(pages))
    doc, expected = short_fragments(ruled, row_counts=counts)
    result = explore(doc, census_config=CensusConfig(table_policy=TablePolicy(minimum_rows=5)))
    (table,) = result.census.table_analysis.tables
    assert table.continued_on == pages and tuple(r.cells for r in table.rows) == expected


def varying_native_subcells():
    doc, expected = cell_box_fixture()
    pages = []
    patterns = ((2, 2, 2, 3, 3), (4, 4, 4, 3, 3))
    for page in doc.pages:
        table = page.tables[0]
        body = []
        for col, original in enumerate(table.cell_rows[1]):
            left, top, right, bottom = original.bbox
            count = patterns[(page.page - 1) % 2][col]
            cuts = {
                2: (left, left + 7, right),
                3: (left, left + 7, right - 9, right),
                4: (left, left + 7, right - 17, right - 9, right),
            }[count]
            for part, (a, b) in enumerate(zip(cuts, cuts[1:])):
                body.append(
                    replace(
                        original,
                        cell_id=original.cell_id + f":part:{part}",
                        text=original.text if part == 1 else "",
                        bbox=(a, top, b, bottom),
                    )
                )
        # Raw matrix slots remain rectangular, independently of physical cells.
        headers = tuple(c.text for c in table.cell_rows[0])
        slots = max(len(headers), len(body))
        changed = replace(
            table,
            cell_rows=(table.cell_rows[0], tuple(body)),
            headers=(*headers, *("" for _ in range(slots - len(headers)))),
            rows=((*(c.text for c in body), *(None for _ in range(slots - len(body)))),),
        )
        pages.append(replace(page, tables=(changed,)))
    return replace(doc, pages=tuple(pages)), expected


def test_header_collapse_precedes_continuation_for_varying_subcells():
    doc, expected = varying_native_subcells()
    result = explore(doc)
    (table,) = result.census.table_analysis.tables
    assert tuple(r.cells for r in table.rows) == expected
    assert len(table.headers) == 5 and table.continued_on == 6
    assert len(table.rows[0].members[0]) == 2 and len(table.rows[1].members[0]) == 4


def test_native_collapse_cannot_merge_distinct_header_labels():
    doc, _ = cell_box_fixture()
    pages = []
    for page in doc.pages:
        table = page.tables[0]
        first, second, *rest = table.cell_rows[1]
        merged = replace(
            first,
            text=first.text + " " + second.text,
            bbox=(first.bbox[0], first.bbox[1], second.bbox[2], first.bbox[3]),
        )
        pages.append(
            replace(page, tables=(replace(table, cell_rows=(table.cell_rows[0], (merged, *rest))),))
        )
    result = explore(replace(doc, pages=tuple(pages)))
    assert not result.census.table_analysis.tables
    assert len(result.census.table_analysis.findings) == 6
    assert all(f.reason == "BAND_COUNT_VARIES" for f in result.census.table_analysis.findings)


def test_unresolved_continuation_reports_bands_support_and_pages():
    doc, _ = short_fragments(row_counts=(1, 1))
    result = explore(doc, census_config=CensusConfig(table_policy=TablePolicy(minimum_rows=3)))
    assert len(result.census.table_analysis.findings) == 2
    for finding in result.census.table_analysis.findings:
        d = finding.diagnostics
        assert d.stage == "support" and d.inferred_band_count == 3 and d.populated_band_count == 2
        assert d.pooled_rows == 2 and d.pages_spanned == (1, 2) and d.continuation_fragments == 2
        assert d.mapped_band_support == (1, 1, 1) and d.mapped_nonempty_support == (1, 1, 0)
    unresolved = result.profile.fields["id_row_tables"].value["unresolved"]
    assert all(f["diagnostics"]["pages_spanned"] == (1, 2) for f in unresolved)
