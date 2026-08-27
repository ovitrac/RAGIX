"""
Gate K3.f — one grid core, two formats.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K3.29-K3.34.

This is the headline claim of the whole layer: a spreadsheet and a word-processing table are the
same object seen through two file formats, and one set of rules reads both. The claim is easy to
assert and easy to fake — two traces that look similar prove nothing. So the central test builds
one logical table twice, once in each format, and asserts the two analyses are *equal*: same core,
same header rows, same label columns, same chains. Not similar. Equal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.analyzers import (  # noqa: E402
    DOCX_TYPES,
    SelfReferenceError,
    anchors,
    pipeline,
)
from ragix_kernels.saqqara.analyzers.grid import (  # noqa: E402
    DOCX_MAPPING,
    XLSX_MAPPING,
    grid_cells,
)
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402

DOCX_CASES = {
    "docx_two_tier": "data-form",
    "docx_label_tiling": "data-form",
    "docx_markers": "data-form",
    "docx_layout_prose": "layout",
}


def _analyze(path):
    adapter = adapter_for(path)
    tree = build_tree(
        read_path(path), str(path), adapter.format, adapter.format, adapter.version
    ).tree
    return pipeline(tree)


def _tables(tree, fmt=None):
    return [
        n for n in tree.walk()
        if n.kind == "table" and (fmt is None or n.provenance.source_format == fmt)
    ]


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    root = tmp_path_factory.mktemp("k3f")
    out = {}
    for name in (*DOCX_CASES, "docx_header_stream", "docx_nested"):
        out[name] = _analyze(G.FIXTURES[name](root / f"{name}.docx"))
    return out


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    root = tmp_path_factory.mktemp("k3f_twin")
    out = {}
    for name, suffix in (("twin_grid_xlsx", ".xlsx"), ("twin_grid_docx", ".docx"),
                         ("twin_grid_pptx", ".pptx")):
        tree, traces = _analyze(G.FIXTURES[name](root / f"{name}{suffix}"))
        block = next(n for n in tree.walk() if n.facts.get("block_type") == "table")
        out[name] = (tree, traces, block)
    return out


def _analysis(block):
    """The comparable shape of one block: geometry and chains, nothing else."""
    header = block.facts["header"]
    return {
        "core": header["core"],
        "header_rows": header["header_rows"],
        "label_cols": header["label_cols"],
        "title": header["title"],
        "section_rows": header["section_rows"],
        "chains": [
            (
                sample["cell"],
                tuple(anchors(block, sample["cell"])["col_chain"]),
                tuple(anchors(block, sample["cell"])["row_chain"]),
            )
            for sample in G.EXPECTED_TWIN["samples"]
        ],
    }


# ----------------------------------------------- K3.29 typing by ordered rules

@pytest.mark.parametrize("name,expected", sorted(DOCX_CASES.items()))
def test_k3_29_tables_are_typed_by_ordered_rules(docs, name, expected):
    tree, _ = docs[name]
    types = {t.facts["table_type"] for t in _tables(tree, "docx")}
    assert expected in types, f"{name}: {types}"
    for table in _tables(tree, "docx"):
        assert table.facts["table_type"] in DOCX_TYPES
        assert table.facts["signals"]["rule"].startswith("D")


def test_k3_29_positive_evidence_promotes(docs):
    tree, _ = docs["docx_two_tier"]
    table = _tables(tree, "docx")[0]
    assert table.facts["table_type"] == "data-form"
    assert table.facts["signals"]["rule"] == "D2-ruled-grid-of-two-dimensions"


def test_k3_29_absent_evidence_abstains_rather_than_demoting(docs):
    """Calling a data table 'layout' drops its content silently. Abstention does not."""
    tree, _ = docs["docx_nested"]
    parent = next(
        t for t in _tables(tree, "docx") if t.provenance.leaf.flow == "body"
    )
    assert parent.facts["table_type"] == "table_uncertain"
    assert parent.facts["signals"]["rule"] == "D5-no-positive-evidence"


def test_k3_29_a_marker_is_enough_on_its_own(docs):
    tree, _ = docs["docx_markers"]
    table = _tables(tree, "docx")[0]
    assert table.facts["signals"]["rule"] == "D1-fillable-markers"


# ------------------------------------------------------ K3.30 the baseline

def test_k3_30_typing_beats_every_table_is_data(docs):
    """Measured, not assumed: the baseline promotes the prose tables too."""
    tree, traces = docs["docx_layout_prose"]
    tables = _tables(tree, "docx")
    baseline = ["data-form"] * len(tables)
    ours = [t.facts["table_type"] for t in tables]
    assert ours != baseline
    assert ours == ["layout", "layout"]
    assert traces["grid_tables"]["by_type"]["layout"] == 2


# ------------------------------------- K3.31 the same core over both formats

def test_k3_31_one_table_written_three_times_analyses_identically(twins):
    """A spreadsheet, a document, and a slide. Equal, not similar."""
    analyses = {name: _analysis(twins[name][2]) for name in twins}
    first = next(iter(analyses.values()))
    for name, analysis in analyses.items():
        assert analysis == first, name + ": the same table must not depend on its format"
    assert len(analyses) == 3


def test_k3_31_and_it_matches_what_the_builders_were_told_to_draw(twins):
    for name in twins:
        analysis = _analysis(twins[name][2])
        assert analysis["core"] == G.EXPECTED_TWIN["core"]
        assert analysis["header_rows"] == G.EXPECTED_TWIN["header_rows"]
        assert analysis["label_cols"] == G.EXPECTED_TWIN["label_cols"]
        for sample, (ref, col_chain, row_chain) in zip(
            G.EXPECTED_TWIN["samples"], analysis["chains"]
        ):
            assert ref == sample["cell"]
            assert list(col_chain) == sample["col_chain"]
            assert list(row_chain) == sample["row_chain"]


def test_k3_31_the_span_and_the_vertical_merge_resolve_like_a_merged_range(twins):
    """The two formats spell a merge differently; the core must not notice."""
    for name in twins:
        cells, _ = grid_cells(twins[name][2])
        extents = sorted(c.extent.to_a1() for c in cells if c.merged)
        assert extents == ["A1:A2", "B1:C1"], name


def test_k3_31_the_self_reference_guard_holds_in_both_lanes(twins):
    for name in twins:
        block = twins[name][2]
        for ref in ("A1", "B1", "B2", "A3"):
            with pytest.raises(SelfReferenceError):
                anchors(block, ref)


def test_k3_31_a_label_tiling_reads_the_same_way_in_a_document(docs):
    tree, _ = docs["docx_label_tiling"]
    block = next(n for n in tree.walk() if n.facts.get("block_type") == "table")
    assert block.facts["header"]["label_cols"] == ["A", "B"]
    assert anchors(block, "C1")["row_chain"] == ["Phase amont", "cadrer"]
    assert anchors(block, "C3")["row_chain"] == ["Phase aval", "livrer"]


# --------------------------------------------- K3.32 abstention by typing

@pytest.mark.parametrize("name", ["docx_layout_prose", "docx_nested"])
def test_k3_32_a_table_that_is_not_data_is_not_analysed(docs, name):
    tree, _ = docs[name]
    for table in _tables(tree, "docx"):
        if table.facts["table_type"] == "data-form":
            continue
        assert table.facts["block_type"] == "excluded"
        assert "header" not in table.facts, "no band is read over a table that is not one"


def test_k3_32_the_reason_it_was_not_analysed_is_on_the_record(docs):
    tree, traces = docs["docx_layout_prose"]
    assert len(traces["grid_tables"]["abstained"]) == 2
    for entry in traces["grid_tables"]["abstained"]:
        assert entry["type"] in ("layout", "table_uncertain")
        assert entry["rule"].startswith("D")


# ------------------------------------------- K3.33 detail columns as labels

def test_k3_33_a_detail_column_is_label_eligible(docs):
    """The slot column ends the label zone; everything textual left of it is question."""
    tree, _ = docs["docx_label_tiling"]
    block = next(n for n in tree.walk() if n.facts.get("block_type") == "table")
    header = block.facts["header"]
    assert header["label_cols"] == ["A", "B"]
    assert header["signals"]["rules"] == ["L2-slot-column"]
    assert header["core"] == "C1:C4"


def test_k3_33_detail_cells_are_label_positions_not_answers(docs):
    tree, _ = docs["docx_label_tiling"]
    block = next(n for n in tree.walk() if n.facts.get("block_type") == "table")
    for ref in ("A1", "B1", "B3"):
        with pytest.raises(SelfReferenceError):
            anchors(block, ref)


def test_k3_33_the_detail_enriches_the_reconstructed_question(docs):
    tree, _ = docs["docx_label_tiling"]
    block = next(n for n in tree.walk() if n.facts.get("block_type") == "table")
    chain = anchors(block, "C2")["row_chain"]
    assert chain == ["Phase amont", "planifier"]
    assert len(chain) > 1, "the detail rung is what tells two sibling questions apart"


# ------------------------------------ K3.34 the mapping is declared, not implied

def test_k3_34_each_format_declares_its_mapping_in_order(twins):
    assert XLSX_MAPPING and DOCX_MAPPING
    for mapping in (XLSX_MAPPING, DOCX_MAPPING):
        assert all(rule[:2].rstrip().isalnum() for rule in mapping), "each rule is numbered"
    assert len(set(DOCX_MAPPING)) == len(DOCX_MAPPING)


def test_k3_34_the_mapping_is_recorded_in_the_trace(twins):
    from ragix_kernels.saqqara.analyzers.grid import PPTX_MAPPING

    for name, expected in (
        ("twin_grid_xlsx", list(XLSX_MAPPING)),
        ("twin_grid_docx", list(DOCX_MAPPING)),
        ("twin_grid_pptx", list(PPTX_MAPPING)),
    ):
        signals = twins[name][2].facts["header"]["signals"]
        assert signals["mapping"] == expected


def test_k3_34_no_format_name_reaches_the_rules(twins):
    """The core sees grid cells. If a format name appears in its signals, it leaked."""
    for name in twins:
        signals = dict(twins[name][2].facts["header"]["signals"])
        signals.pop("mapping", None)
        rendered = repr(signals).lower()
        for fmt in ("xlsx", "docx", "pptx"):
            assert fmt not in rendered, name + ": " + fmt + " reached the rules"


def test_k3_34_both_lanes_produce_the_same_neutral_vocabulary(twins):
    xlsx_cells, _ = grid_cells(twins["twin_grid_xlsx"][2])
    docx_cells, _ = grid_cells(twins["twin_grid_docx"][2])

    def shape(cells):
        return sorted(
            (c.row, c.col, c.text, c.dtype, c.bold, c.merged) for c in cells
        )

    assert shape(xlsx_cells) == shape(docx_cells)


# --------------------------------------------- K2.18 a table on a slide is a table

def test_k2_18_a_slide_table_is_read_at_all(twins):
    """A table shape has no text frame; a reader that walks text frames sees nothing."""
    tree = twins["twin_grid_pptx"][0]
    tables = [n for n in tree.walk() if n.kind == "table"]
    assert len(tables) == 1
    cells = [n for n in tree.walk() if n.kind == "cell"]
    # Twelve positions: one is covered from the left and is not stored, one is
    # covered from above and is stored as a continuation, as a document would.
    assert len(cells) == 11


def test_k2_18_its_cells_are_addressable_individually(twins):
    tree = twins["twin_grid_pptx"][0]
    addresses = {
        (n.provenance.leaf.slide, n.provenance.leaf.shape,
         n.provenance.leaf.row, n.provenance.leaf.col)
        for n in tree.walk() if n.kind == "cell"
    }
    assert len(addresses) == 11, "each cell has its own coordinate"
    assert all(row is not None and col is not None
               for _, _, row, col in addresses)


def test_k2_18_the_slide_table_reaches_the_shared_core(twins):
    """The third format, proving the claim rather than restating it."""
    cells, mapping = grid_cells(twins["twin_grid_pptx"][2])
    assert cells and mapping
    extents = sorted(c.extent.to_a1() for c in cells if c.merged)
    assert extents == ["A1:A2", "B1:C1"], "the same merges as its two twins"
