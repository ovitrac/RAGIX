"""Cell topology and reader exposure on independent synthetic tables.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace
from ragix_kernels.saqqara.census import DocumentDigest, PageDigest, TableObservation, Evidence
from ragix_kernels.saqqara.table_views import TableCell
from ragix_kernels.saqqara.table_context import native_contexts
from ragix_kernels.saqqara.explorer import explore


def table_fixture(*, headers=("Channel", "Travel [mm]", "Supply [V]"), rows=None):
    rows = rows or (
        ("Axis A", "23", "17"),
        ("Axis B", "31", "29"),
        ("Axis C", "47", "11"),
        ("Axis D", "53", "19"),
    )
    values = (headers, *rows)
    physical = tuple(
        tuple(
            TableCell(
                f"t:{r}:{c}",
                text,
                (30 + c * 100, 90 + r * 30, 130 + c * 100, 120 + r * 30),
                (f"s:{r}:{c}",),
                geometry_kind="cell_box",
            )
            for c, text in enumerate(row)
        )
        for r, row in enumerate(values)
    )
    ev = tuple(
        Evidence("synthetic", 1, c.cell_id, 0, len(c.text), c.text, c.bbox)
        for row in physical
        for c in row
    )
    table = TableObservation("t", 1, headers, rows, ev, cell_rows=physical)
    doc = DocumentDigest(
        "synthetic", (PageDigest(1, 600, 800, (), tables=(table,)),), "synthetic", "1"
    )
    return doc, table


def test_native_table_with_no_identifier_still_exposes_context():
    doc, table = table_fixture()
    contexts, failures = native_contexts(doc.source_id, table)
    assert not failures and len(contexts) == 8
    first = contexts[0]
    assert first.value.text == "23"
    assert [c.text for c in first.column_headers] == ["Travel [mm]"]
    assert [c.text for c in first.row_labels] == ["Axis A"]
    result = explore(doc)
    assert len(result.reading.cell_contexts) == 8
    assert len(result.report.cell_contexts) == 8


def test_multiline_header_keeps_exact_cell_text():
    doc, table = table_fixture(headers=("Channel", "Travel\n[mm]", "Supply\n[V]"))
    contexts, failures = native_contexts(doc.source_id, table)
    assert not failures
    assert contexts[0].column_headers[0].text == "Travel\n[mm]"


def test_header_split_into_unresolved_topology_is_reported():
    doc, table = table_fixture()
    broken = replace(table, headers=("Channel", "Travel\n[mm]", "Supply [V]"))
    contexts, failures = native_contexts(doc.source_id, broken)
    assert not contexts and failures[0].reason == "HEADER_ASSOCIATION_UNRESOLVED"


def test_headerless_table_never_promotes_first_data_row():
    doc, table = table_fixture(headers=("", "", ""))
    contexts, failures = native_contexts(doc.source_id, table)
    assert not failures
    assert all(not c.column_headers and "NO_COLUMN_HEADER" in c.flags for c in contexts)


def test_unlocated_geometry_cannot_supply_context():
    doc, table = table_fixture()
    rows = list(table.cell_rows)
    rows[0] = tuple(replace(c, flags=("MISSING_CELL_GEOMETRY",)) for c in rows[0])
    contexts, failures = native_contexts(doc.source_id, replace(table, cell_rows=tuple(rows)))
    assert not contexts and failures[0].reason == "NO_EXACT_CELL_TOPOLOGY"


def test_reader_harvests_each_column_with_its_own_unit():
    doc, _ = table_fixture()
    result = explore(doc)
    quantities = result.reading.quantities
    assert len(quantities) == 8
    for q in quantities:
        assert q["unit_source"] == "INHERITED" and q["needs_review"]
        proof = q["unit_evidence"][0]["span"]
        assert q["unit"] == ("mm" if q["node_id"].endswith(":1") else "V")
        assert proof["cell_id"] == ("t:0:1" if q["unit"] == "mm" else "t:0:2")
        assert q["unit_start"] is None


def test_reader_recovers_row_label_units_without_a_column_unit():
    rows = tuple(
        (f"Axis {letter} [mm]", str(value), "plain")
        for letter, value in zip("ABCD", (23, 31, 47, 53))
    )
    doc, _ = table_fixture(headers=("Channel", "Reading", "Note"), rows=rows)
    quantities = explore(doc).reading.quantities
    assert len(quantities) == 4
    assert all(
        q["unit"] == "mm" and q["unit_evidence"][0]["association"] == "row_label"
        for q in quantities
    )
