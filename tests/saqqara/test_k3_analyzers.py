"""
Gate K3 — the analyzers: segmentation, header bands, chains, islands.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K3.1-K3.28. K3.29-K3.34 are carried by test_k3f_cross_format.py and K3.35-K3.39 by test_k3g_services.py.K3.40-K3.53 (sections, typed outline) are carried by test_k3hi_sections_outline.py.

Ground truth is EXPECTED_GRID and EXPECTED_BLOCKS in the generators, written next to the builders
that produce the fixtures. Every geometry below is compared against what the builder was told to
draw, never against what the analyzer happened to find.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.analyzers import (  # noqa: E402
    ABSTENTION_REASONS,
    TYPING_REASONS,
    BLANK_RUNG,
    SelfReferenceError,
    TablesAnalyzer,
    anchors,
    pipeline,
)
from ragix_kernels.saqqara.analyzers.geometry import coord, parse_range  # noqa: E402
from ragix_kernels.saqqara.analyzers.grid import grid_cells  # noqa: E402
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402

GRID_FIXTURES = sorted(G.EXPECTED_GRID)


def _analyze(path):
    adapter = adapter_for(path)
    tree = build_tree(
        read_path(path), str(path), adapter.format, adapter.format, adapter.version
    ).tree
    return pipeline(tree)


def _blocks(tree):
    return [block for section in tree.root.children for block in section.children]


def _tables(tree):
    return [b for b in _blocks(tree) if b.facts.get("block_type") == "table"]


@pytest.fixture(scope="module")
def grids(tmp_path_factory):
    root = tmp_path_factory.mktemp("k3")
    out = {}
    for name in GRID_FIXTURES:
        tree, traces = _analyze(G.FIXTURES[name](root / f"{name}.xlsx"))
        out[name] = (tree, traces)
    return out


@pytest.fixture(scope="module")
def workbook(tmp_path_factory):
    return _analyze(G.FIXTURES["mixed_workbook"](tmp_path_factory.mktemp("k3w") / "m.xlsx"))


# ------------------------------------------ K3.1-K3.5 the segmentation cascade

def test_k3_1_block_boundaries_match_the_declared_ground_truth(workbook):
    tree, _ = workbook
    for section in tree.root.children:
        got = sorted((b.facts["block_type"], b.facts["block_range"]) for b in section.children)
        assert got == sorted(G.EXPECTED_BLOCKS[section.text]), section.text


def test_k3_1_a_declared_table_object_is_authoritative(workbook):
    tree, traces = workbook
    declared = [
        s for s in traces["tables"]["sheets"] if s.get("declared_objects")
    ]
    assert declared, "the fixture declares a table object, by construction"
    for sheet in declared:
        for ref in sheet["declared_objects"]:
            assert ref in sheet["ranges"], "a declared boundary survives the cascade intact"
    block = next(b for b in _blocks(tree) if b.facts["block_range"] == "F1:F3")
    assert block.facts["signals"]["rule"] == "T1-declared-table-object"


def test_k3_2_a_blank_spacer_row_does_not_split_a_ruled_box(workbook):
    """The questionnaire block is crossed by a fully blank ruled row."""
    tree, _ = workbook
    block = next(b for b in _blocks(tree) if b.facts["block_range"] == "B4:D8")
    rows = {c.provenance.leaf.row for c in block.children}
    assert 7 in rows, "the spacer row belongs to the box it sits inside"


def test_k3_3_a_blank_slot_stays_addressable(workbook):
    tree, _ = workbook
    block = next(b for b in _blocks(tree) if b.facts["block_range"] == "B4:D8")
    slots = [c for c in block.children if c.text is None]
    assert slots, "the answer columns are ruled and empty"
    for slot in slots:
        assert slot.provenance.leaf.cell, "an empty slot still has an address"


def test_k3_4_an_empty_ruled_box_is_dropped_with_its_reason(tmp_path):
    from openpyxl import Workbook
    from openpyxl.styles import Border, Side

    thin = Side(style="thin")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Un titre"
    for row in (4, 5):
        for col in ("C", "D"):
            ws[f"{col}{row}"].border = box          # ruled, and holding nothing
    path = tmp_path / "empty_box.xlsx"
    wb.save(path)

    _, traces = _analyze(path)
    drops = traces["tables"]["drops"]
    assert len(drops) == 1
    assert drops[0]["reason"] == "bordered-region-holds-no-value"
    assert drops[0]["range"] == "C4:D5"
    assert traces["tables"]["dropped"] == 1


def test_k3_5_the_cascade_beats_one_block_per_sheet(workbook):
    """Asserted, not assumed: the baseline fuses every multi-object sheet."""
    tree, _ = workbook
    for section in tree.root.children:
        expected = G.EXPECTED_BLOCKS[section.text]
        baseline = 1
        assert len(section.children) == len(expected)
        if len(expected) > 1:
            assert len(section.children) > baseline, section.text


# --------------------------------------------------- K3.6-K3.7 typing

def test_k3_6_types_match_the_declared_ground_truth(workbook):
    tree, _ = workbook
    for section in tree.root.children:
        expected = dict((r, t) for t, r in G.EXPECTED_BLOCKS[section.text])
        for block in section.children:
            assert block.facts["block_type"] == expected[block.facts["block_range"]]


def test_k3_6_a_blank_answer_column_does_not_make_a_form_a_list(workbook):
    """Shape is counted over addressable positions, not over filled ones."""
    tree, _ = workbook
    block = next(b for b in _blocks(tree) if b.facts["block_range"] == "B4:D8")
    assert block.facts["block_type"] == "table"
    signals = block.facts["signals"]
    assert signals["addressable_cells"] > signals["valued_cells"]


def test_k3_7_an_ambiguous_layout_abstains(tmp_path):
    tree, _ = _analyze(G.FIXTURES["ambiguous_layout"](tmp_path / "ambiguous.xlsx"))
    block = _blocks(tree)[0]
    assert block.facts["block_type"] == "uncertain"
    assert block.facts["block_uncertain"]["reason"] in TYPING_REASONS
    assert block.confidence < 1.0 and block.origin == "inferred"


# --------------------------------- K3.8-K3.10 anchoring, K3.11-K3.12 locators

@pytest.mark.parametrize("name", ["two_tier_header", "label_tiling"])
def test_k3_8_sample_chains_are_exact(grids, name):
    tree, _ = grids[name]
    block = _tables(tree)[0]
    for sample in G.EXPECTED_GRID[name]["samples"]:
        got = anchors(block, sample["cell"])
        assert got["col_chain"] == sample["col_chain"], sample["cell"]
        assert got["row_chain"] == sample["row_chain"], sample["cell"]


def test_k3_9_an_interior_blank_is_never_invented(grids):
    """A blank tile surfaces as blank; it does not borrow the tile above it."""
    tree, _ = grids["label_tiling"]
    chain = anchors(_tables(tree)[0], "D10")["row_chain"]
    assert chain == ["Production", BLANK_RUNG]
    assert "Relecture" not in chain


def test_k3_10_the_deepest_rung_is_exposed_and_is_never_the_reference(grids):
    tree, _ = grids["two_tier_header"]
    block = _tables(tree)[0]
    got = anchors(block, "B4")
    assert got["col_header"] == got["col_chain"][-1] == "Humains"
    assert got["col_header"] != "B4"


# --------------------------------------------- K3.72 a rung carries its address

def _cell_at(block, ref):
    """The grid cell an A1 reference names, by the top-left of its rectangle."""
    rect = parse_range(ref)
    for cell in grid_cells(block)[0]:
        if (cell.row, cell.col) == (rect.top, rect.left):
            return cell
    return None


@pytest.mark.parametrize("name", ["two_tier_header", "label_tiling"])
def test_k3_72_every_rung_names_the_cell_it_was_read_from(grids, name):
    """A citation that cannot be followed back to a position is not a reading.

    The address is checked by USING it: resolve each rung's reference and read
    the cell there. A reference that merely looks well formed would satisfy a
    test that only parsed it.
    """
    tree, _ = grids[name]
    block = _tables(tree)[0]
    for sample in G.EXPECTED_GRID[name]["samples"]:
        got = anchors(block, sample["cell"])
        for key, chain_key in (("col_rungs", "col_chain"), ("row_rungs", "row_chain")):
            rungs = got[key]
            assert [step["text"] for step in rungs] == got[chain_key], sample["cell"]
            for step in rungs:
                assert step["ref"], f"{sample['cell']}: a rung with no address"
                cell = _cell_at(block, step["ref"])
                assert cell is not None, f"{step['ref']} names no cell"
                expected = cell.text if cell.text is not None else BLANK_RUNG
                assert step["text"] == expected, step["ref"]


def _spans(rect):
    """More than one position. Every address is written `top:bottom`, so the
    separator says nothing about whether the tile is merged."""
    return rect.bottom > rect.top or rect.right > rect.left


def test_k3_72_a_merged_tile_reports_its_whole_extent(grids):
    """The rung is the tile, not a corner of it: a merge spanning two tiers
    contributes one rung, and its address is the range the merge covers."""
    tree, _ = grids["two_tier_header"]
    block = _tables(tree)[0]
    rungs = anchors(block, "B4")["col_rungs"]
    spanning = [step for step in rungs if _spans(parse_range(step["ref"]))]
    assert len(spanning) == 1, rungs        # one rung, not one per column covered
    rect = parse_range(spanning[0]["ref"])
    assert rect.right > rect.left           # the tier above covers both columns
    assert rect.contains(rect.top, coord("B1")[1])
    assert rect.contains(rect.top, coord("C1")[1])


def test_k3_72_a_blank_rung_still_carries_an_address(grids):
    """Where the label is missing is exactly the fact a reviewer needs, and it
    is unaddressable without a reference."""
    tree, _ = grids["label_tiling"]
    rungs = anchors(_tables(tree)[0], "D10")["row_rungs"]
    blanks = [step for step in rungs if step["text"] == BLANK_RUNG]
    assert blanks, [step["text"] for step in rungs]
    for step in blanks:
        assert step["ref"] and _cell_at(_tables(tree)[0], step["ref"]) is not None


def test_k3_72_an_abstention_invents_no_addresses(grids):
    """The fallback returns no rungs rather than rungs pointing nowhere."""
    tree, _ = grids["overlapping_merges"]
    got = anchors(_tables(tree)[0], "C3")
    assert got["uncertain"] is True
    assert got["col_rungs"] == [] and got["row_rungs"] == []


@pytest.mark.parametrize("name", ["two_tier_header", "label_tiling"])
def test_k3_72_the_result_is_json_native(grids, name):
    """Everything `anchors` returns is storable beside the answer it explains.
    A rung as an object would be the one value in the result that will not
    serialise, met by whoever stored a chain rather than by whoever wrote it."""
    tree, _ = grids[name]
    block = _tables(tree)[0]
    for sample in G.EXPECTED_GRID[name]["samples"]:
        json.dumps(anchors(block, sample["cell"]))


@pytest.mark.parametrize("name", GRID_FIXTURES)
def test_k3_11_every_block_carries_a_complete_locator(grids, name):
    tree, _ = grids[name]
    for block in _blocks(tree):
        locator = block.provenance.leaf
        assert locator.sheet and locator.sheet_index is not None
        assert locator.cell == block.facts["block_range"]
        assert locator.row and locator.col
        assert block.facts["block_type"]


@pytest.mark.parametrize("name", GRID_FIXTURES)
def test_k3_12_a_merged_cell_names_its_extent(grids, name):
    tree, _ = grids[name]
    merged = [
        c for b in _blocks(tree) for c in b.children
        if c.kind == "cell" and c.facts.get("merged")
    ]
    for cell in merged:
        assert cell.provenance.leaf.merged_range
        assert ":" in cell.provenance.leaf.merged_range


# ----------------------------------- K3.13-K3.20 the header band, or abstention

@pytest.mark.parametrize("name", GRID_FIXTURES)
def test_k3_13_the_split_matches_the_declared_ground_truth(grids, name):
    expected = G.EXPECTED_GRID[name]
    if expected.get("uncertain") or "core" not in expected:
        pytest.skip("this fixture pins an abstention, not a split")
    tree, _ = grids[name]
    header = _tables(tree)[0].facts["header"]
    assert header["uncertain"] is False
    assert header["core"] == expected["core"]
    assert header["header_rows"] == expected["header_rows"]
    assert header["label_cols"] == expected["label_cols"]
    assert header["title"] == (
        tuple(expected["title"]) if expected["title"] else None
    ) or header["title"] == expected["title"]
    assert header["section_rows"] == expected["section_rows"]


def test_k3_14_a_label_tiling_and_its_band_header_are_exact(grids):
    tree, _ = grids["label_tiling"]
    header = _tables(tree)[0].facts["header"]
    expected = G.EXPECTED_GRID["label_tiling"]
    assert tuple(header["band_header"]) == expected["band_header"]
    assert [tuple(t) for t in header["tiling"]] == expected["tiling"]


def test_k3_15_crossing_merges_abstain(grids):
    tree, _ = grids["overlapping_merges"]
    header = _tables(tree)[0].facts["header"]
    assert header["uncertain"] is True
    assert header["abstention"]["reason"] == "non-laminar-band-merges"
    assert "core" not in header, "no band is emitted over a layout that has none"


def test_k3_16_a_block_with_no_evidence_abstains(grids):
    tree, _ = grids["undecidable_block"]
    header = _tables(tree)[0].facts["header"]
    assert header["uncertain"] is True
    assert header["abstention"]["reason"] == "uniform-block"


@pytest.mark.parametrize("name", ["overlapping_merges", "undecidable_block"])
def test_k3_17_every_reason_is_in_the_frozen_vocabulary(grids, name):
    tree, _ = grids[name]
    header = _tables(tree)[0].facts["header"]
    assert header["abstention"]["reason"] in ABSTENTION_REASONS
    assert header["abstention"]["signals"], "an abstention without signals explains nothing"


def test_k3_18_the_trace_is_decomposed_not_scored(grids):
    tree, _ = grids["two_tier_header"]
    signals = _tables(tree)[0].facts["header"]["signals"]
    assert isinstance(signals["rules"], list) and signals["rules"]
    assert "merge_intervals" in signals and "dtypes" in signals
    assert not any(
        isinstance(value, float) for value in signals.values()
    ), "a float in a trace is a score standing in for the evidence"


@pytest.mark.parametrize("name", GRID_FIXTURES)
def test_k3_19_the_analysis_serialises(grids, name):
    tree, traces = grids[name]
    for block in _blocks(tree):
        json.dumps(block.facts, sort_keys=True, ensure_ascii=False)
    json.dumps(traces, sort_keys=True, ensure_ascii=False)


def test_k3_20_the_band_beats_the_header_equals_one_baseline(grids):
    """The legacy reading takes the block's first row as the header. Always.

    That is the honest baseline: it does not abstain, it does not notice a title,
    and it does not go two rows deep. Measured on every fixture, wins and ties
    both, because a comparison that reported only the wins would be advertising.
    """
    from ragix_kernels.saqqara.analyzers.geometry import parse_range

    wins, ties = [], []
    for name in GRID_FIXTURES:
        expected = G.EXPECTED_GRID[name]
        if expected.get("uncertain") or "header_rows" not in expected:
            continue
        tree, _ = grids[name]
        block = _tables(tree)[0]
        baseline = [parse_range(block.facts["block_range"]).top]
        (wins if expected["header_rows"] != baseline else ties).append(name)

    assert "two_tier_header" in wins, "h=1 sees one tier of a two-tier band"
    assert "headerless_list" in wins, "h=1 invents a header where there is none"
    assert "full_width_title" in wins, "h=1 reads the title row as the header"
    assert ties, "ties are recorded, not omitted"
    assert "label_tiling" in ties, "h=1 happens to be right here, and that is reported"
    assert "numeric_bold_header" in ties
    assert len(wins) + len(ties) == len(
        [n for n in GRID_FIXTURES if not G.EXPECTED_GRID[n].get("uncertain")
         and "header_rows" in G.EXPECTED_GRID[n]]
    ), "every comparable fixture is counted, in one column or the other"


# ---------------------------------- K3.21-K3.24 chains and the self-reference guard

def test_k3_21_a_column_chain_reads_broadest_first(grids):
    tree, _ = grids["two_tier_header"]
    chain = anchors(_tables(tree)[0], "B4")["col_chain"]
    assert chain == ["Moyens engages", "Humains"]


def test_k3_21_a_merge_across_tiers_is_one_rung(grids):
    """The label merge spans both header rows and must not appear twice."""
    tree, _ = grids["label_tiling"]
    chain = anchors(_tables(tree)[0], "E12")["row_chain"]
    assert chain == ["Cloture"], "one tile spanning two label columns is one rung"


def test_k3_22_an_empty_tile_is_an_addressable_rung(grids):
    tree, _ = grids["label_tiling"]
    assert anchors(_tables(tree)[0], "D10")["row_chain"][-1] == BLANK_RUNG


@pytest.mark.parametrize("name", ["two_tier_header", "label_tiling"])
def test_k3_23_header_and_label_positions_raise(grids, name):
    tree, _ = grids[name]
    block = _tables(tree)[0]
    for ref in G.EXPECTED_GRID[name]["guarded"]:
        with pytest.raises(SelfReferenceError):
            anchors(block, ref)


def test_k3_24_an_abstaining_block_flags_every_result(grids):
    tree, _ = grids["overlapping_merges"]
    got = anchors(_tables(tree)[0], "B4")
    assert got["uncertain"] is True
    assert got["how"] == "fallback-no-band"
    assert got["col_chain"] == [] and got["row_chain"] == []
    assert got["abstention"]["reason"] == "non-laminar-band-merges"


# ------------------------------------------------------- K3.25-K3.28 islands

def test_k3_25_two_panels_in_one_box_are_reported(grids):
    tree, _ = grids["two_islands"]
    islands = _tables(tree)[0].facts["islands"]
    assert islands["count"] == 2
    assert [r["subtitle"] for r in islands["regions"]] == ["Volet gauche", "Volet droit"]


def test_k3_25_a_panel_with_no_header_reports_an_empty_chain(grids):
    """Honest, not borrowed from the panel next door."""
    tree, _ = grids["two_islands"]
    for region in _tables(tree)[0].facts["islands"]["regions"]:
        assert region["col_chain"] == []


def test_k3_26_the_flag_is_raised_and_never_acted_on(grids):
    tree, traces = grids["two_islands"]
    islands = _tables(tree)[0].facts["islands"]
    assert islands["segmentation_feedback"] is True
    assert islands["resegmented"] is False
    assert traces["islands"]["feedback"][0]["action"] == "reported-never-resegmented"
    assert len(_tables(tree)) == 1, "the block keeps the shape its border gave it"


def test_k3_27_a_non_laminar_block_stays_uncertain(grids):
    tree, _ = grids["overlapping_merges"]
    block = _tables(tree)[0]
    assert block.facts["header"]["uncertain"] is True
    assert anchors(block, "B4")["uncertain"] is True


@pytest.mark.parametrize(
    "name", [n for n in GRID_FIXTURES if n != "two_islands"]
)
def test_k3_28_the_fallback_changes_nothing_outside_its_trigger(grids, name):
    tree, _ = grids[name]
    for block in _tables(tree):
        assert block.facts["islands"]["count"] == 1
        assert block.facts["islands"]["segmentation_feedback"] is False



# ============ K3.73 — the cap is declared, and the record says how deep the band was

def _deep_block(tmp_path):
    """The five-row bold band, as a block the header rules can be run on."""
    from ragix_kernels.saqqara.adapters.contract import adapter_for, read_path
    from ragix_kernels.saqqara.builder import build_tree

    path = G.FIXTURES["deep_header_band"](tmp_path / "deep.xlsx")
    adapter = adapter_for(path)
    tree = build_tree(read_path(path), str(path), adapter.format,
                      adapter.format, adapter.version).tree
    from ragix_kernels.saqqara.analyzers import TablesAnalyzer
    tree = TablesAnalyzer().run(tree).tree
    return next(n for n in tree.walk() if n.facts.get("block_type") == "table")


def test_k3_73_the_cap_is_load_bearing_not_decorative(tmp_path):
    """The same block abstains under one cap and is read under another.

    A declared parameter that changes nothing is a comment. This is the assertion
    that says the value reaches the rule.

    Falsified by: an analysis that does not change when the declared cap does.
    """
    from ragix_kernels.saqqara.analyzers.header_bands import analyze_block

    block = _deep_block(tmp_path)

    tight = analyze_block(block, max_header_rows=3)
    assert tight["uncertain"] is True
    assert tight["abstention"]["reason"] == "band-too-deep"

    loose = analyze_block(block, max_header_rows=5)
    assert loose["uncertain"] is False, "the cap was declared and the rule ignored it"


def test_k3_73_the_abstention_says_how_deep_the_band_was_and_what_the_cap_was(tmp_path):
    """A rule that abstains without saying how far over it went cannot be tuned.

    The loop used to return the moment it passed the cap, so the record could say
    only "deeper than the cap" — and the depths behind the default had to be
    re-derived from the cells by re-implementing this rule elsewhere. It counts the
    band to its end first now.

    Falsified by: an abstention whose signals carry no depth, or a depth that is
    merely the cap plus one.
    """
    from ragix_kernels.saqqara.analyzers.header_bands import analyze_block

    signals = analyze_block(_deep_block(tmp_path), max_header_rows=3)["abstention"]["signals"]

    assert signals["max_header_rows"] == 3
    assert signals["depth_found"] == 5, "the band is five rows deep and the record must say so"
    assert signals["depth_found"] > 3 + 1, "the depth is counted, not stopped at the cap"


def test_k3_73_an_analyzers_trace_says_what_it_ran_with(tmp_path):
    """Defaults included, never only what a manifest overrode.

    Falsified by: a trace without its options, or one showing only the overrides.
    """
    from ragix_kernels.saqqara.analyzers import HeaderBandsAnalyzer, IslandsAnalyzer

    # The trace of a real run, not the attribute on the instance: removing the
    # options from what an analyzer returns passed every assertion of the first
    # version of this test, which checked `.options` and never the trace — the
    # falsifier K3.73 names was not covered by the gate that claimed it.
    from ragix_kernels.saqqara.adapters.contract import adapter_for, read_path
    from ragix_kernels.saqqara.analyzers import PIPELINE
    from ragix_kernels.saqqara.builder import build_tree

    path = G.FIXTURES["deep_header_band"](tmp_path / "traced.xlsx")
    adapter = adapter_for(path)
    tree = build_tree(read_path(path), str(path), adapter.format,
                      adapter.format, adapter.version).tree
    for analyzer_class in PIPELINE:
        analyzer = analyzer_class()
        result = analyzer.run(tree)
        tree = result.tree
        assert "options" in result.trace, f"{analyzer.name} does not say what it ran with"
        assert result.trace["options"] == analyzer.options

    declared_run = HeaderBandsAnalyzer({"max_header_rows": 5}).run(tree)
    assert declared_run.trace["options"]["max_header_rows"] == 5, \
        "the trace does not carry the value the run declared"

    default = HeaderBandsAnalyzer()
    assert default.options == {"max_header_rows": 3, "max_label_cols": 3}

    declared = HeaderBandsAnalyzer({"max_header_rows": 5})
    assert declared.options == {"max_header_rows": 5, "max_label_cols": 3}, \
        "an override must not drop the defaults beside it"

    # An analyzer that declares no option still says so, rather than staying silent.
    assert IslandsAnalyzer().options == {}


def test_k3_73_an_option_a_analyzer_does_not_declare_is_refused(tmp_path):
    """A mistyped key that silently does nothing is the defect met three times today.

    Falsified by: an unknown option accepted and ignored.
    """
    from ragix_kernels.saqqara.analyzers import HeaderBandsAnalyzer

    with pytest.raises(ValueError, match="does not take"):
        HeaderBandsAnalyzer({"max_header_row": 5})       # singular: a plausible typo


def test_k3_73_the_default_is_unchanged(tmp_path):
    """Declared does not mean changed: a run stating nothing behaves as before.

    Falsified by: a default that moved with the mechanism.
    """
    from ragix_kernels.saqqara.analyzers.header_bands import (
        MAX_HEADER_ROWS, MAX_LABEL_COLS, HeaderBandsAnalyzer, analyze_block)

    assert MAX_HEADER_ROWS == 3 and MAX_LABEL_COLS == 3
    assert HeaderBandsAnalyzer.DEFAULTS == {"max_header_rows": 3, "max_label_cols": 3}

    block = _deep_block(tmp_path)
    assert analyze_block(block) == analyze_block(block, max_header_rows=MAX_HEADER_ROWS)
