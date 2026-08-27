"""
Gate K2 — the readers: raw facts, one reader per format.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K2.1-K2.12, K2.15 and K2.16. K2.13 and K2.14 belong to the laid-out-document reader
and are carried by test_k2p_pdf.py.

Every fixture is built by the generators at test time. What the readers are checked against is
what the builders were told to write, not what the readers happen to produce.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.adapters import (  # noqa: E402
    UnreadableFile,
    UnsupportedFormat,
    adapter_for,
    read_path,
    read_paths,
    registered_adapters,
)
from ragix_kernels.saqqara.adapters.docx import CELL_FACTS as DOCX_CELL_FACTS  # noqa: E402
from ragix_kernels.saqqara.adapters.xlsx import CELL_FACTS as XLSX_CELL_FACTS  # noqa: E402


def _cells(records, flow=None):
    out = [r for r in records if r.kind == "cell"]
    return [r for r in out if flow is None or r.locator.flow == flow]


def _at(records, row, col, flow="body"):
    for record in _cells(records, flow):
        if record.locator.row == row and record.locator.col == col:
            return record
    return None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every fixture this gate needs, built once."""
    root = tmp_path_factory.mktemp("k2")
    suffix = {
        "mixed_workbook": ".xlsx", "two_tier_header": ".xlsx", "numeric_bold_header": ".xlsx",
        "slide_deck": ".pptx", "markdown_document": ".md", "duplicate_pair": ".md",
        "unsupported_format": ".tmp", "empty_string_cells": ".xlsx",
    }
    return {
        name: G.FIXTURES[name](root / f"{name}{suffix.get(name, '.docx')}")
        for name in (
            "numeric_bold_header", "two_tier_header", "mixed_workbook",
            "docx_two_tier", "docx_label_tiling", "docx_layout_prose", "docx_markers",
            "docx_header_stream", "docx_nested", "docx_twin_pair",
            "slide_deck", "markdown_document", "unsupported_format", "duplicate_pair",
            "empty_string_cells",
        )
    }


# ------------------------------------------------- K2.1 the declared fact set

def test_k2_1_every_cell_carries_exactly_the_declared_facts(built):
    records = [r for r in read_path(built["numeric_bold_header"]) if r.kind == "cell"]
    assert records
    for record in records:
        assert tuple(sorted(record.facts)) == tuple(sorted(XLSX_CELL_FACTS))


def test_k2_1_geometric_keys_live_in_the_locator_not_the_facts(built):
    for record in read_path(built["numeric_bold_header"]):
        assert not {"cell", "row", "col", "sheet"} & set(record.facts)
        if record.kind == "cell":
            assert record.locator.cell and record.locator.row and record.locator.col


def test_k2_1_borders_are_their_own_record_not_a_cell_fact(built):
    """Keeping borders out of the cell facts is what keeps that set exact."""
    records = read_path(built["mixed_workbook"])
    borders = [r for r in records if r.kind == "border"]
    assert borders, "the workbook fixture has bordered cells"
    for record in borders:
        assert tuple(sorted(record.facts)) == ("bottom", "left", "right", "top")
    for record in (r for r in records if r.kind == "cell"):
        assert "left" not in record.facts


# --------------------------------------------- K2.2 read, never interpreted

def test_k2_2_a_numeric_bold_header_records_both_facts(built):
    records = read_path(built["numeric_bold_header"])
    years = [r for r in _cells_xlsx(records) if r.locator.row == 1 and 1 < r.locator.col < 5]
    assert len(years) == 3
    for record in years:
        assert record.facts["dtype"] == "n", "a year is numeric and stays numeric"
        assert record.facts["bold"] is True
        assert record.text in {"2024", "2025", "2026"}


def _cells_xlsx(records):
    return [r for r in records if r.kind == "cell"]


def test_k2_2_number_format_and_lock_are_reported_as_found(built):
    records = _cells_xlsx(read_path(built["numeric_bold_header"]))
    budget = [r for r in records if r.locator.row == 3 and 1 < r.locator.col < 5]
    assert budget and all(r.facts["number_format"] == "#,##0.00" for r in budget)
    assert all(isinstance(r.facts["locked"], bool) for r in records)


def test_k2_2_a_formula_is_reported_as_a_formula(built):
    records = _cells_xlsx(read_path(built["numeric_bold_header"]))
    formulas = [r for r in records if r.facts["formula"] is not None]
    assert len(formulas) == 1
    assert formulas[0].facts["dtype"] == "f"
    assert formulas[0].facts["formula"].startswith("=")


# ------------------------------------------------------ K2.3 merges anchor

def test_k2_3_a_merge_is_recorded_once_at_its_anchor(built):
    records = _cells_xlsx(read_path(built["two_tier_header"]))
    anchors = [r for r in records if r.facts["merged"]]
    assert {r.locator.merged_range for r in anchors} == {"A2:A3", "B2:C2"}


def test_k2_3_continuation_positions_are_not_emitted_at_all(built):
    """A continuation carrying a copy of the anchor's facts would double the evidence."""
    positions = {
        (r.locator.row, r.locator.col) for r in _cells_xlsx(read_path(built["two_tier_header"]))
    }
    assert (2, 1) in positions, "the anchor of A2:A3 is emitted"
    assert (3, 1) not in positions, "its continuation is not"
    assert (2, 3) not in positions, "nor is the continuation of B2:C2"


# ------------------------------------------------------ K2.4 JSON primitives

@pytest.mark.parametrize(
    "name", ["numeric_bold_header", "mixed_workbook", "docx_two_tier", "slide_deck",
             "markdown_document"]
)
def test_k2_4_every_record_serialises(built, name):
    payload = [r.to_dict() for r in read_path(built[name])]
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    assert json.loads(text) == payload


# -------------------------------------------------- K2.5 declared versions

def test_k2_5_each_reader_declares_a_version_and_its_fact_set():
    """Pinned here, so changing what a reader emits without bumping it fails."""
    pinned = {
        "xlsx": ("0.3.0", ("dtype", "bold", "number_format", "locked", "formula", "merged")),
        "docx": ("0.2.0", ("span", "vmerge", "empty", "fillable", "marker", "bold", "shaded")),
        "pptx": ("0.2.0", ("span", "vmerge", "empty", "fillable", "marker", "bold", "shaded")),
        "md": ("0.1.0", ("level", "kind_hint")),
        "pdf": ("0.1.0", ("has_text", "image_count", "needs_ocr")),
    }
    seen = {a.format: a for a in registered_adapters().values()}
    assert set(seen) >= set(pinned)
    for fmt, (version, facts) in pinned.items():
        assert seen[fmt].version == version, f"{fmt}: version moved without updating this pin"
        assert seen[fmt].fact_set == facts, f"{fmt}: fact set changed — bump the version"


def test_k2_5_xlsx_and_docx_fact_sets_match_their_modules():
    seen = {a.format: a for a in registered_adapters().values()}
    assert seen["xlsx"].fact_set == XLSX_CELL_FACTS
    assert seen["docx"].fact_set == DOCX_CELL_FACTS


# ------------------------------------- K2.6 resolved grid, every table flow

def test_k2_6_the_grid_is_resolved_not_ragged(built):
    records = read_path(built["docx_two_tier"])
    table = next(r for r in records if r.kind == "table")
    assert table.facts["n_rows"] == 4
    assert table.facts["n_grid_cols"] == 4
    assert table.facts["ragged"] is False


def test_k2_6_span_and_vertical_merge_are_reported(built):
    records = read_path(built["docx_two_tier"])
    assert _at(records, 0, 1).facts["span"] == 3
    assert _at(records, 0, 0).facts["vmerge"] == "restart"
    assert _at(records, 1, 0).facts["vmerge"] == "continue"
    assert _at(records, 0, 1).text == "Moyens mobilises"


def test_k2_6_style_facts_are_reported(built):
    records = read_path(built["docx_two_tier"])
    assert _at(records, 0, 0).facts["shaded"] is True
    assert _at(records, 0, 0).facts["bold"] is True
    assert _at(records, 2, 0).facts["shaded"] is False


def test_k2_6_a_header_flow_table_is_read(built):
    records = read_path(built["docx_header_stream"])
    flows = {r.locator.flow for r in records if r.kind == "table"}
    assert "header:0" in flows, "a table in the page header is invisible to a body walk"
    assert "body" in flows


def test_k2_6_a_nested_table_is_read(built):
    records = read_path(built["docx_nested"])
    nested = [r for r in records if r.kind == "table" and r.locator.flow.startswith("nested:")]
    assert len(nested) == 1
    assert nested[0].facts["n_rows"] == 2 and nested[0].facts["n_grid_cols"] == 2


def test_k2_6_label_tiling_merges_are_reported(built):
    records = read_path(built["docx_label_tiling"])
    column = {(r.locator.row, r.facts["vmerge"]) for r in _cells(records) if r.locator.col == 0}
    assert column == {(0, "restart"), (1, "continue"), (2, "restart"), (3, "continue")}


def test_k2_6_layout_tables_are_read_without_being_judged(built):
    """The reader reports two grids of prose. Whether they are tables is K3's question."""
    records = read_path(built["docx_layout_prose"])
    tables = [r for r in records if r.kind == "table"]
    assert len(tables) == 2
    assert all("layout" not in (t.facts.get("style") or "") for t in tables)


# ----------------------------------------------------- K2.7 no nested bleed

def test_k2_7_nested_content_stays_out_of_the_parent_cell(built):
    records = read_path(built["docx_nested"])
    parent = _at(records, 0, 1, flow="body")
    assert parent is not None
    assert parent.text is None, "the parent cell holds no text of its own"
    nested_text = {
        r.text for r in _cells(records) if r.locator.flow.startswith("nested:")
    }
    assert nested_text == {"n00", "n01", "n10", "n11"}
    for record in _cells(records, flow="body"):
        for value in nested_text:
            assert value not in (record.text or "")


# ------------------------------------------------- K2.8 markers, told apart

def test_k2_8_in_cell_markers_are_cell_facts(built):
    records = read_path(built["docx_markers"])
    assert _at(records, 0, 1).facts["marker"] == "form-text"
    assert _at(records, 1, 1).facts["marker"] == "content-control"
    assert _at(records, 0, 1).facts["fillable"] is True


def test_k2_8_a_marker_outside_any_table_is_its_own_record(built):
    records = read_path(built["docx_markers"])
    loose = [r for r in records if r.kind == "marker"]
    assert len(loose) == 1
    assert loose[0].facts["in_table"] is False
    assert loose[0].facts["marker"] == "form-checkbox"
    assert loose[0].locator.paragraph is not None and loose[0].locator.table_index is None


def test_k2_8_a_plain_empty_slot_carries_no_marker(built):
    records = read_path(built["docx_markers"])
    plain = _at(records, 2, 1)
    assert plain.facts["empty"] is True
    assert plain.facts["fillable"] is False and plain.facts["marker"] is None


# ------------------------------------------------------ K2.9 slot arithmetic

def test_k2_9_blank_minus_filled_equals_the_answers(built):
    blank = built["docx_twin_pair"]
    filled = blank.with_name(f"{blank.stem}_filled{blank.suffix}")

    def slots(path):
        return [
            r for r in _cells(read_path(path))
            if r.facts["empty"] and r.facts["vmerge"] is None
        ]

    answers = len(slots(blank)) - len(slots(filled))
    assert answers == G.EXPECTED_DOCX["docx_twin_pair"]["answers"]


def test_k2_9_continuations_are_excluded_from_the_count(built):
    """Counting continuations as blanks overstates the slots on every merged form."""
    records = read_path(built["docx_label_tiling"])
    naive = [r for r in _cells(records) if r.facts["empty"]]
    honest = [r for r in naive if r.facts["vmerge"] is None]
    assert len(naive) > len(honest), "this fixture has continuations, by construction"
    assert len(honest) == G.EXPECTED_DOCX["docx_label_tiling"]["n_empty"]


# ------------------------------------------------- K2.10 index within a flow

def test_k2_10_indices_restart_in_each_flow(built):
    records = read_path(built["docx_header_stream"])
    per_flow = {}
    for record in (r for r in records if r.kind == "table"):
        per_flow.setdefault(record.locator.flow, []).append(record.locator.table_index)
    assert per_flow["body"] == [0]
    assert per_flow["header:0"] == [0], "a header table is table 0 of its own flow"


def test_k2_10_a_nested_flow_names_its_parent_position(built):
    records = read_path(built["docx_nested"])
    nested = next(r for r in records if r.kind == "table" and r.locator.flow.startswith("nested:"))
    assert nested.locator.flow == "nested:body:0:0:1"
    assert nested.locator.table_index == 0


# --------------------------------------------------------- K2.11 fail closed

def test_k2_11_an_unclaimed_format_raises(built):
    with pytest.raises(UnsupportedFormat):
        read_path(built["unsupported_format"])
    assert adapter_for(built["unsupported_format"]) is None


def test_k2_11_a_claimed_but_broken_file_raises(tmp_path):
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"this is not a document")
    with pytest.raises(UnreadableFile):
        read_path(broken)


def test_k2_11_refusals_are_counted_not_dropped(built, tmp_path):
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"neither is this")
    facts, report = read_paths(
        [built["markdown_document"], built["unsupported_format"], broken]
    )
    assert report.counts == {"read": 1, "refused": 2, "duplicate": 0}
    assert {r.reason for r in report.refusals} == {"unsupported-format", "unreadable-file"}
    assert facts, "the readable file still produced its records"


def test_k2_11_a_missing_file_is_a_refusal_not_a_crash(tmp_path):
    facts, report = read_paths([tmp_path / "absent.md"])
    assert facts == []
    assert report.counts["refused"] == 1


# ------------------------------------------------------- K2.12 duplicates

def test_k2_12_identical_bytes_are_read_once_and_counted(built):
    original = built["duplicate_pair"]
    copy = original.with_name(f"{original.stem}_copy{original.suffix}")
    assert copy.read_bytes() == original.read_bytes()
    facts, report = read_paths([original, copy])
    assert report.counts == {"read": 1, "refused": 0, "duplicate": 1}
    assert report.duplicates == [str(copy)]


def test_k2_12_a_duplicate_never_counts_as_corroboration(built):
    original = built["duplicate_pair"]
    copy = original.with_name(f"{original.stem}_copy{original.suffix}")
    once, _ = read_paths([original])
    twice, _ = read_paths([original, copy])
    assert len(twice) == len(once)


# ------------------------------------------------------------ K2.15 slides

def test_k2_15_slides_are_numbered_as_a_reader_counts_them(built):
    records = read_path(built["slide_deck"])
    slides = [r for r in records if r.kind == "slide"]
    assert [r.locator.slide for r in slides] == [1, 2], "a citation to slide 7 must mean slide 7"


def test_k2_15_shapes_are_emitted_with_their_slide(built):
    records = read_path(built["slide_deck"])
    shapes = [r for r in records if r.kind == "shape"]
    assert shapes
    assert {r.locator.slide for r in shapes} == {1, 2}
    assert any(r.facts["is_title"] for r in shapes)


def test_k2_15_notes_are_kept_off_the_slide(built):
    records = read_path(built["slide_deck"])
    notes = [r for r in records if r.kind == "notes"]
    assert len(notes) == 2
    assert all(r.facts["on_slide"] is False for r in notes)
    shown = " ".join(r.text or "" for r in records if r.kind in ("slide", "shape"))
    for record in notes:
        assert record.text not in shown, "what the audience never saw is not slide text"


# ---------------------------------------------------------- K2.16 markdown

def test_k2_16_front_matter_is_metadata_not_prose(built):
    records = read_path(built["markdown_document"])
    metadata = [r for r in records if r.kind == "metadata"]
    assert len(metadata) == 1
    assert metadata[0].facts["title"] == "Document de controle"
    assert metadata[0].facts["language"] == "fr"
    for record in records:
        if record.kind == "paragraph":
            assert "title:" not in (record.text or "")


def test_k2_16_every_block_knows_the_line_it_starts_on(built):
    records = read_path(built["markdown_document"])
    body = [r for r in records if r.kind in ("heading", "paragraph")]
    assert body
    lines = [r.locator.line for r in body]
    assert lines == sorted(lines) and all(line > 0 for line in lines)
    heading = next(r for r in body if r.kind == "heading")
    assert heading.facts["level"] == 1 and heading.text == "Titre principal"


# ----------------------------------------- K2.17 a whitespace-only value is blank

def test_k2_17_an_empty_string_is_blank_not_content(built):
    """It looks blank on screen; reading it as content changes a table's shape."""
    cells = {
        r.locator.cell: r for r in read_path(built["empty_string_cells"]) if r.kind == "cell"
    }
    for ref, expected in G.EXPECTED_EMPTY_STRINGS.items():
        assert ref in cells, ref
        assert cells[ref].text == expected["text"], ref


def test_k2_17_an_empty_string_stays_distinguishable_from_an_absent_value(built):
    """All three are blank. The data type still says which ones held a string."""
    cells = {
        r.locator.cell: r for r in read_path(built["empty_string_cells"]) if r.kind == "cell"
    }
    for ref, expected in G.EXPECTED_EMPTY_STRINGS.items():
        assert cells[ref].facts["dtype"] == expected["dtype"], ref
        held = cells[ref].facts["dtype"] in ("s", "inlineStr", "str")
        assert held is expected["held_a_string"], ref
    assert cells["B2"].facts["dtype"] != cells["B4"].facts["dtype"]


def test_k2_17_a_ruled_blank_is_still_an_addressable_slot(built):
    cells = {
        r.locator.cell: r for r in read_path(built["empty_string_cells"]) if r.kind == "cell"
    }
    for ref in ("B2", "B3", "B4"):
        assert cells[ref].facts["dtype"] is not None or cells[ref].text is None
    borders = {
        r.locator.cell for r in read_path(built["empty_string_cells"]) if r.kind == "border"
    }
    assert {"B2", "B3", "B4"} <= borders, "a cleared answer cell keeps its rule"


def test_k2_17_the_blank_does_not_count_as_a_value_downstream(built):
    """The reach of the defect: it decided a block's shape, not just a count."""
    from ragix_kernels.saqqara.analyzers import pipeline
    from ragix_kernels.saqqara.builder import build_tree

    path = built["empty_string_cells"]
    adapter = adapter_for(path)
    tree = build_tree(read_path(path), str(path), adapter.format, adapter.format,
                      adapter.version).tree
    tree, _ = pipeline(tree)
    block = next(n for n in tree.walk() if n.facts.get("block_type") == "table")
    header = block.facts["header"]
    assert header["uncertain"] is False
    assert header["label_cols"] == ["A"], "the answer column is a slot column, not a value column"
