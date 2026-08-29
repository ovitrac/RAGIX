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

from ragix_kernels.saqqara.assets import AssetStore  # noqa: E402
from ragix_kernels.saqqara.adapters import (  # noqa: E402
    GRID_CELL_FACTS,
    GRID_TABLE_FACTS,
    OpenVocabulary,
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
    return {
        name: G.FIXTURES[name](G.fixture_path(name, root))
        for name in (
            "numeric_bold_header", "two_tier_header", "mixed_workbook",
            "docx_two_tier", "docx_label_tiling", "docx_layout_prose", "docx_markers",
            "docx_header_stream", "docx_nested", "docx_twin_pair",
            "slide_deck", "markdown_document", "unsupported_format", "duplicate_pair",
            "empty_string_cells", "format_headings_docx",
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


def test_k2_4_a_fact_may_be_a_list_of_primitives(built):
    """Both of the kernel's non-scalar facts, so the wording is exercised, not assumed.

    The proposition says primitives *and lists of them*; a fixture holding only
    scalars would leave the wider half of the claim untested and free to be wrong.
    """
    sheets = [r for r in read_path(built["mixed_workbook"]) if r.kind == "sheet"]
    tables = [r.facts["list_objects"] for r in sheets]
    assert any(t for t in tables), "no sheet declares a table: the list case is untested"
    assert all(isinstance(t, list) and all(isinstance(x, str) for x in t) for t in tables)

    metadata = next(r for r in read_path(built["markdown_document"]) if r.kind == "metadata")
    assert metadata.facts["_unparsed"] == ["une ligne sans deux-points"]


# -------------------------------------------------- K2.5 declared versions

#: version and declared vocabularies, per reader, per record kind.
#:
#: One entry per kind, not one per reader: a flat set could only ever pin the
#: kind whose facts it happened to hold, which is how five readers came to
#: declare five vocabularies while emitting sixteen.
PINNED = {
    "xlsx": ("0.6.0", {
        "figure": ("asset", "source", "media_type", "width", "height",
                   "x", "y", "w", "h", "colorspace", "bits", "smask"),
        "sheet": ("hidden", "list_objects", "max_row", "max_column"),
        "cell": ("dtype", "bold", "number_format", "locked", "formula", "merged"),
        "border": ("left", "right", "top", "bottom"),
    }),
    "docx": ("0.6.0", {
        "figure": ("asset", "source", "media_type", "width", "height",
                   "x", "y", "w", "h", "colorspace", "bits", "smask"),
        "table": ("n_rows", "n_grid_cols", "ragged", "style"),
        "cell": ("span", "vmerge", "empty", "fillable", "marker", "bold", "shaded"),
        "paragraph": ("marker", "in_table", "style", "numbered", "outline_level",
                      "bold_frac", "size", "size_frac"),
        "marker": ("marker", "in_table", "style", "numbered", "outline_level",
                   "bold_frac", "size", "size_frac"),
    }),
    "pptx": ("0.5.0", {
        "figure": ("asset", "source", "media_type", "width", "height",
                   "x", "y", "w", "h", "colorspace", "bits", "smask"),
        "slide": ("shape_count",),
        "shape": ("shape_type", "is_title", "on_slide"),
        "notes": ("on_slide",),
        "table": ("n_rows", "n_grid_cols", "ragged", "style"),
        "cell": ("span", "vmerge", "empty", "fillable", "marker", "bold", "shaded"),
    }),
    "md": ("0.2.0", {
        "metadata": OpenVocabulary(reserved=("_unparsed",)),
        "heading": ("level",),
        "paragraph": (),
    }),
    "pdf": ("0.6.0", {
        "outline_entry": ("level",),
        "page": ("has_text", "image_count", "needs_ocr"),
        "text": ("x", "y", "font_size", "font"),
        "figure": ("asset", "source", "media_type", "width", "height",
                   "x", "y", "w", "h", "colorspace", "bits", "smask"),
        "drawing": ("x", "y", "w", "h", "ops", "stroke", "fill"),
    }),
}


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """Every record every reader produces over the whole fixture registry.

    Built rather than described: K2.20 compares declarations with emissions, and
    a description of the emissions would be one more declaration to keep true.
    """
    root = tmp_path_factory.mktemp("k2vocab")
    # With a store, so the object vocabularies are exercised too: a reader that
    # emits figures only when it has somewhere to put them would otherwise
    # declare a vocabulary this sweep never sees, which is precisely what K2.20
    # refuses.
    store = AssetStore(root / "_assets")
    out = {}
    for name, build in sorted(G.FIXTURES.items()):
        home = root / name
        home.mkdir(parents=True, exist_ok=True)
        build(G.fixture_path(name, home))
        for path in sorted(q for q in home.rglob("*") if q.is_file()):
            adapter = adapter_for(path)
            if adapter is None:                       # the refusal fixtures, on purpose
                continue
            for record in read_path(path, store=store):
                out.setdefault(adapter.format, {}).setdefault(record.kind, set()).update(
                    record.facts
                )
    return out


def test_k2_5_each_reader_declares_a_version_and_its_vocabularies():
    """Pinned here, so changing what a reader declares without bumping it fails."""
    seen = {a.format: a for a in registered_adapters().values()}
    assert set(seen) >= set(PINNED)
    for fmt, (version, vocabularies) in PINNED.items():
        assert seen[fmt].version == version, f"{fmt}: version moved without updating this pin"
        assert dict(seen[fmt].fact_sets) == vocabularies, (
            f"{fmt}: a declared vocabulary changed — bump the version"
        )


def test_k2_5_xlsx_and_docx_cell_vocabularies_match_their_modules():
    seen = {a.format: a for a in registered_adapters().values()}
    assert seen["xlsx"].fact_sets["cell"] == XLSX_CELL_FACTS
    assert seen["docx"].fact_sets["cell"] == DOCX_CELL_FACTS


# --------------------------------------------- K2.19 a vocabulary per record kind

def test_k2_19_every_emitted_kind_has_a_declared_vocabulary(emitted):
    seen = {a.format: a for a in registered_adapters().values()}
    undeclared = [
        f"{fmt}.{kind}"
        for fmt, kinds in emitted.items()
        for kind in kinds
        if kind not in seen[fmt].fact_sets
    ]
    assert not undeclared, f"emitted with no declared vocabulary: {sorted(undeclared)}"


def test_k2_19_no_fact_escapes_the_vocabulary_of_its_own_kind(emitted):
    seen = {a.format: a for a in registered_adapters().values()}
    escaped = []
    for fmt, kinds in emitted.items():
        for kind, facts in kinds.items():
            declared = seen[fmt].fact_sets[kind]
            if isinstance(declared, OpenVocabulary):
                continue                              # its names are the document's (K2.21)
            escaped += [f"{fmt}.{kind}.{f}" for f in sorted(facts) if f not in declared]
    assert not escaped, f"emitted outside the declared vocabulary: {sorted(escaped)}"


def test_k2_19_a_kind_is_not_described_by_another_kinds_facts(emitted):
    """The claim that makes per-kind worth the edit: a cell and a paragraph differ."""
    docx = emitted["docx"]
    assert docx["cell"] != docx["paragraph"]
    assert "span" in docx["cell"] and "span" not in docx["paragraph"]
    assert "style" in docx["paragraph"] and "style" not in docx["cell"]


# ------------------------------ K2.20 declaration against emission, both directions

def test_k2_20_every_declared_name_is_produced_by_something(emitted):
    """The direction the old flat pin could not check — and where `kind_hint` hid."""
    seen = {a.format: a for a in registered_adapters().values()}
    wished = []
    for fmt, adapter in seen.items():
        produced = emitted.get(fmt, {})
        for kind, declared in adapter.fact_sets.items():
            names = declared.reserved if isinstance(declared, OpenVocabulary) else declared
            wished += [
                f"{fmt}.{kind}.{name}"
                for name in names
                if name not in produced.get(kind, set())
            ]
    assert not wished, f"declared and emitted by nothing: {sorted(wished)}"


def test_k2_20_the_sweep_reaches_every_reader(emitted):
    """A direction-check over a corpus that missed a reader would prove nothing."""
    assert set(emitted) == {"xlsx", "docx", "pptx", "pdf", "md"}


# ------------------------------------------------- K2.21 declared-open vocabularies

def test_k2_21_front_matter_is_declared_open_not_left_undeclared():
    seen = {a.format: a for a in registered_adapters().values()}
    assert isinstance(seen["md"].fact_sets["metadata"], OpenVocabulary)
    assert seen["md"].fact_sets["metadata"].reserved == ("_unparsed",)


def test_k2_21_an_open_vocabulary_takes_its_names_from_the_document(built):
    records = read_path(built["markdown_document"])
    metadata = next(r for r in records if r.kind == "metadata")
    assert {"title", "language"} <= set(metadata.facts)
    assert metadata.facts["_unparsed"] == ["une ligne sans deux-points"]


def test_k2_21_every_other_vocabulary_is_closed():
    seen = {a.format: a for a in registered_adapters().values()}
    open_ones = [
        f"{fmt}.{kind}"
        for fmt, adapter in seen.items()
        for kind, declared in adapter.fact_sets.items()
        if isinstance(declared, OpenVocabulary)
    ]
    assert open_ones == ["md.metadata"]


# --------------------------------------------- K2.22 one grid vocabulary, two readers

def test_k2_22_the_grid_kinds_share_one_vocabulary():
    seen = {a.format: a for a in registered_adapters().values()}
    assert seen["docx"].fact_sets["cell"] == seen["pptx"].fact_sets["cell"] == GRID_CELL_FACTS
    assert seen["docx"].fact_sets["table"] == seen["pptx"].fact_sets["table"] == GRID_TABLE_FACTS


def test_k2_22_they_share_the_object_not_a_copy_of_it():
    """Two equal tuples may drift apart; one tuple cannot."""
    seen = {a.format: a for a in registered_adapters().values()}
    assert seen["docx"].fact_sets["cell"] is seen["pptx"].fact_sets["cell"]
    assert seen["docx"].fact_sets["table"] is seen["pptx"].fact_sets["table"]


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


# ------------------------------------ K2.24 the size on the page, not the operand

def test_k2_24_the_same_type_reads_the_same_however_it_is_scaled(tmp_path):
    """Four pages, one visual size pair, four ways of expressing it.

    A reader that reports the operand passes the first page and fails the rest,
    which is exactly the shape of the corpus finding this proposition came from.
    """
    path = G.FIXTURES["pdf_type_scales"](tmp_path / "scales.pdf")
    by_page = {}
    for record in read_path(path):
        if record.kind != "text":
            continue
        by_page.setdefault(record.locator.page, set()).add(
            round(record.facts["font_size"], 1)
        )

    assert len(by_page) == 4, "the fixture writes four pages"
    expected = {round(v, 1) for v in G.TYPE_SCALE_SIZES}
    for page in sorted(by_page):
        assert by_page[page] == expected, (
            f"page {page} reads {sorted(by_page[page])}, expected {sorted(expected)}"
        )


def test_k2_24_rotated_text_is_not_read_as_zero(tmp_path):
    """The reason the scale is a column length and not a single matrix cell."""
    path = G.FIXTURES["pdf_type_scales"](tmp_path / "scales.pdf")
    rotated = [
        r for r in read_path(path)
        if r.kind == "text" and r.locator.page == 4
    ]
    assert rotated, "the fourth page carries the rotated text"
    assert all(r.facts["font_size"] > 0 for r in rotated)


# ------------------------------------- K2.23 weight as a fraction, size as a ratio

def _paragraphs(records):
    return [r for r in records if r.kind in ("paragraph", "marker")]


def test_k2_23_weight_is_a_fraction_of_characters_not_a_flag(built):
    records = read_path(built["format_headings_docx"])
    paragraphs = _paragraphs(records)
    assert paragraphs
    for record in paragraphs:
        assert "bold" not in record.facts, "the boolean was replaced, not kept alongside"
        assert isinstance(record.facts["bold_frac"], float)

    fractions = sorted({r.facts["bold_frac"] for r in paragraphs})
    assert fractions[0] == 0.0 and fractions[-1] == 1.0


def test_k2_23_one_bold_word_in_twenty_is_not_a_bold_paragraph(built):
    """The case a boolean cannot express, and the reason this fact changed shape."""
    records = _paragraphs(read_path(built["format_headings_docx"]))
    partial = [r for r in records if 0.0 < r.facts["bold_frac"] < 1.0]
    assert len(partial) == 1, "the fixture holds exactly one partly bold paragraph"
    assert partial[0].facts["bold_frac"] < 0.2
    # a boolean over the same paragraph would have said True, indistinguishably
    # from the fully bold heading two paragraphs above it
    fully = [r for r in records if r.facts["bold_frac"] == 1.0]
    assert fully, "and the fixture holds fully bold lines to be distinguished from"


def test_k2_23_the_fraction_is_character_mass_not_a_run_count(built):
    """One short bold run among long plain ones: counted by runs it would dominate."""
    record = next(
        r for r in _paragraphs(read_path(built["format_headings_docx"]))
        if 0.0 < r.facts["bold_frac"] < 1.0
    )
    # one bold run out of three would be 0.33 by run count; by character mass it is far less
    assert record.facts["bold_frac"] < 0.33


def test_k2_23_size_is_absolute_and_against_the_documents_own_modal(built):
    records = _paragraphs(read_path(built["format_headings_docx"]))
    for record in records:
        assert record.facts["size"] == 11.0            # the fixture sets one size
        assert record.facts["size_frac"] == 1.0        # so every ratio is one


def test_k2_23_an_undeclared_size_is_not_guessed(built):
    """A run that inherits its size from a style leaves the fact unknown, not zero."""
    records = _paragraphs(read_path(built["docx_markers"]))
    assert records
    assert all(r.facts["size"] is None for r in records)
    assert all(r.facts["size_frac"] is None for r in records)


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
