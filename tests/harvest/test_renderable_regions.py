"""G1–G8 and list/figure extensions on independent source observations.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import base64
from dataclasses import replace
import hashlib
import io
import json
import pytest
from ragix_kernels.harvest.regions import (
    RegionIndex,
    RegionMember,
    PageGeometry,
    BoundaryPolicy,
    RegionWindow,
    RegionLimits,
    RegionRefused,
    FigureInput,
    image_payload,
    cell_member,
)
from ragix_kernels.harvest.table_context import Cell
from .test_region_boundaries import line, fixtures, score

PAGES = (PageGeometry(1, 400, 800), PageGeometry(2, 400, 800))


def png():
    from PIL import Image

    target = io.BytesIO()
    Image.new("RGB", (4, 3), (12, 34, 56)).save(target, format="PNG", compress_level=0)
    return target.getvalue()


def table():
    members = []
    for r, texts in enumerate((("Channel", "Travel [mm]"), ("Axis A", "23"), ("Axis B", "31"))):
        for c, text in enumerate(texts):
            cell = Cell(
                "synthetic",
                "table-a",
                f"cell-{r}-{c}",
                1,
                r,
                c,
                text,
                (20 + c * 100, 150 + r * 25, 120 + c * 100, 175 + r * 25),
                (f"cell-source-{r}-{c}",),
            )
            members.append(cell_member(cell, is_header=r == 0, is_row_label=r > 0 and c == 0))
    return tuple(members)


def test_g1_exact_text_including_non_normalized_unicode_and_spacing():
    text = "Cafe\u0301\u00a0  and  a line\nwith spacing."
    source = line("a", text, 30)
    payload = RegionIndex("synthetic", PAGES, (source,)).get("a").to_dict()
    assert payload["members"][0]["text"].encode() == text.encode()
    assert payload["members"][0]["source_spans"] == ("a-source",)


def test_g2_cellspan_returns_whole_table_and_the_scored_slice():
    cells = table()
    source = cells[3]
    cell = Cell(
        source.source_id,
        source.table_id,
        source.member_id,
        source.page,
        source.row,
        source.column,
        source.text,
        source.bbox,
        source.source_spans,
    )
    result = RegionIndex("synthetic", PAGES, (), tables=(cells,)).get(cell.span(0, 1)).to_dict()
    assert result["kind"] == "TABLE" and len(result["members"]) == 6
    assert sum(m["is_header"] for m in result["members"]) == 2
    assert sum(m["is_row_label"] for m in result["members"]) == 2
    assert result["anchor_member_id"] == "cell-1-1" and result["anchor"]["text"] == "2"
    assert all(
        m["text"] == next(c.text for c in cells if c.member_id == m["member_id"])
        for m in result["members"]
    )


def test_g2_table_line_anchor_keeps_original_line_identity():
    source = line("scored", "23", 178, x=125, right=145)
    result = RegionIndex("synthetic", PAGES, (source,), tables=(table(),)).get("scored").to_dict()
    assert result["kind"] == "TABLE" and result["anchor_member_id"] == "scored"
    alias = next(m for m in result["members"] if m["member_id"] == "scored")
    assert alias["text"] == "23" and "SOURCE_LINE_ALIAS" in alias["flags"]


def test_merged_cell_metadata_is_not_rebuilt_or_lost():
    c = Cell(
        "synthetic",
        "t",
        "merged",
        1,
        0,
        0,
        "Merged title",
        (10, 10, 210, 50),
        (),
        column_span=2,
        row_span=2,
    )
    payload = (
        RegionIndex("synthetic", PAGES, (), tables=((cell_member(c, is_header=True),),))
        .get(c.span())
        .to_dict()
    )
    assert payload["members"][0]["row_span"] == 2 and payload["members"][0]["column_span"] == 2


def test_g3_cross_page_paragraph_has_one_identity_and_page_specific_boxes():
    lines, _ = fixtures()[-1]
    index = RegionIndex("synthetic", PAGES, lines)
    a = index.get("a").to_dict()
    b = index.get("b").to_dict()
    assert a["region_id"] == b["region_id"] and len(a["members"]) == 2
    assert a["pages"] == (1, 2) and "CROSSES_PAGE" in a["flags"]
    assert [p for p, box in a["page_boxes"]] == [1, 2]
    assert b["anchor"]["page"] == 2 and b["neighbourhood"]["scope_page"] == 2


def test_g4_neighbourhood_count_and_truncation_are_explicit():
    lines = tuple(line(chr(97 + i), f"Paragraph {i}.", 30 + i * 60) for i in range(4))
    index = RegionIndex("synthetic", PAGES, lines)
    first = index.get("a").to_dict()
    middle = index.get("b").to_dict()
    last = index.get("d").to_dict()
    assert "TRUNCATED_AT_WINDOW" in first["flags"] and "TRUNCATED_AT_WINDOW" in last["flags"]
    assert "TRUNCATED_AT_WINDOW" not in middle["flags"]
    n = middle["neighbourhood"]
    assert len(n["before"]) == len(n["after"]) == 1
    assert (
        n["usage"] == "CONTEXT_ONLY" and n["window_rule"] == "1 region before, 1 after, same page"
    )
    assert "anchor_member_id" not in n["before"][0]
    assert "TRUNCATED_AT_WINDOW" not in index.get("a", window=RegionWindow(0, 0)).flags


def test_g5_no_observed_cells_means_no_table():
    result = RegionIndex("synthetic", PAGES, (line("a", "Grid-shaped drawing label", 30),)).get("a")
    assert result.region.kind == "PROSE"
    with pytest.raises(RegionRefused, match="OBSERVED_CELLS"):
        RegionIndex("synthetic", PAGES, (), tables=((line("a", "Grid label", 30),),))


def test_g6_contamination_is_flagged_without_cleaning():
    cells = list(table())
    cells[3] = replace(cells[3], text="2 ARCHIVED 3", flags=("LIFECYCLE_GLYPH_SUSPECTED",))
    result = RegionIndex("synthetic", PAGES, (), tables=(cells,)).get(cells[3].member_id).to_dict()
    assert "LIFECYCLE_GLYPH_SUSPECTED" in result["flags"]
    assert result["anchor"]["text"] == "2 ARCHIVED 3"


def test_g7_repeatability_and_region_identity_do_not_depend_on_window():
    lines, _ = fixtures()[0]
    index = RegionIndex("synthetic", PAGES, lines)
    assert index.get("a").to_json() == index.get("a").to_json()
    assert index.get("a").region_id == index.get("b", window=RegionWindow(0, 0)).region_id


@pytest.mark.parametrize(
    "text",
    [
        "<b>source</b>",
        "&lt;source&gt;",
        'style="display:none"',
        "color: red;",
        "<script>alert(1)</script>",
    ],
)
def test_g8_markup_in_source_is_refused_not_cleaned(text):
    source = line("a", text, 30)
    index = RegionIndex("synthetic", PAGES, (source,))
    with pytest.raises(RegionRefused, match="MARKUP_OR_STYLE"):
        index.get("a")
    assert source.text == text


def test_g8_unsafe_neighbour_is_not_silently_dropped():
    index = RegionIndex(
        "synthetic", PAGES, (line("a", "Plain", 30), line("b", "<b>source</b>", 90))
    )
    with pytest.raises(RegionRefused):
        index.get("a")
    assert index.get("a", window=RegionWindow(0, 0)).anchor.text == "Plain"


def test_list_region_preserves_nested_and_wrapped_members():
    lines = (
        line("a", "• Outer", 20),
        line("b", "• Nested", 33, x=35),
        line("c", "wrapped", 46, x=38),
        line("d", "• Another outer", 59),
    )
    result = RegionIndex("synthetic", PAGES, lines).get("b").to_dict()
    assert result["kind"] == "LIST"
    assert [m["text"] for m in result["members"]] == [m.text for m in lines]
    assert [m["list_depth"] for m in result["members"]] == [0, 1, 1, 0]


def test_figure_base64_bytes_mime_dimensions_and_checksum():
    raw = png()
    image = image_payload(raw, "image/png")
    source = line("label", "Diagram label", 30, x=30, right=100)
    figure = FigureInput("figure-a", "synthetic", 1, (20, 20, 200, 100), image)
    result = RegionIndex("synthetic", PAGES, (source,), figures=(figure,)).get("label").to_dict()
    assert result["kind"] == "FIGURE"
    raster = result["image"]
    assert raster["encoding"] == "base64" and raster["media_type"] == "image/png"
    assert (
        base64.b64decode(raster["data"]) == raw
        and raster["sha256"] == hashlib.sha256(raw).hexdigest()
    )
    assert (raster["width"], raster["height"], raster["byte_count"]) == (4, 3, len(raw))


def test_figure_caption_outside_box_is_explicit_and_missing_raster_is_flagged():
    source = line("caption", "Diagram caption", 120)
    f = FigureInput(
        "f",
        "synthetic",
        1,
        (20, 20, 200, 100),
        caption_ids=("caption",),
        image_reason="FIGURE_RASTER_UNAVAILABLE",
    )
    r = RegionIndex("synthetic", PAGES, (source,), figures=(f,)).get("caption").to_dict()
    assert r["kind"] == "FIGURE" and r["members"][0]["kind"] == "CAPTION"
    assert r["image"] is None and "FIGURE_IMAGE_UNAVAILABLE" in r["flags"]


def test_figure_payloads_cannot_forge_a_checked_image_or_use_svg():
    with pytest.raises(RegionRefused):
        image_payload(b"<svg/>", "image/svg+xml")
    image = image_payload(png(), "image/png")
    with pytest.raises(RegionRefused, match="PAYLOAD_MISMATCH"):
        f = FigureInput("f", "synthetic", 1, (20, 20, 200, 100), replace(image, width=5))
        RegionIndex("synthetic", PAGES, (line("a", "Label", 30),), figures=(f,)).get("a")


def test_figure_validation_cache_does_not_trust_checksum_alone():
    image = image_payload(png(), "image/png")
    first = FigureInput("f1", "synthetic", 1, (20, 20, 200, 80), image)
    second = FigureInput("f2", "synthetic", 1, (20, 100, 200, 150), replace(image, width=5))
    index = RegionIndex(
        "synthetic",
        PAGES,
        (line("a", "First", 30), line("b", "Second", 110)),
        figures=(first, second),
    )
    index.get("a", window=RegionWindow(0, 0))
    with pytest.raises(RegionRefused, match="PAYLOAD_MISMATCH"):
        index.get("b", window=RegionWindow(0, 0))


def test_limits_refuse_instead_of_truncating_or_dropping_anchor():
    with pytest.raises(RegionRefused, match="OUTPUT_LIMIT"):
        RegionIndex(
            "synthetic", PAGES, (line("a", "Text", 30),), limits=RegionLimits(max_output_bytes=50)
        ).get("a")
    with pytest.raises(RegionRefused, match="PIXEL_LIMIT"):
        image_payload(png(), "image/png", limits=RegionLimits(max_image_pixels=1))
    with pytest.raises(RegionRefused, match="MEMBER_LIMIT"):
        RegionIndex(
            "synthetic",
            PAGES,
            (line("a", "A", 30), line("b", "B", 80)),
            limits=RegionLimits(max_members=1),
        )


def test_foreign_and_stale_anchors_are_refused():
    c = Cell("synthetic", "t", "c", 1, 0, 0, "23", (20, 20, 60, 40))
    index = RegionIndex("synthetic", PAGES, (), tables=((cell_member(c),),))
    with pytest.raises(RegionRefused, match="FOREIGN"):
        index.get(replace(c.span(), source_id="other"))
    with pytest.raises(RegionRefused, match="FOREIGN"):
        index.get(replace(c.span(), raw="99"))
    with pytest.raises(RegionRefused, match="NOT_FOUND"):
        index.get("missing")


def test_conflicting_table_figure_membership_is_not_resolved_by_preference():
    source = line("a", "23", 178, x=125, right=145)
    f = FigureInput(
        "f", "synthetic", 1, (20, 150, 220, 225), image_reason="FIGURE_RASTER_UNAVAILABLE"
    )
    with pytest.raises(RegionRefused, match="AMBIGUOUS_ENCLOSING"):
        RegionIndex("synthetic", PAGES, (source,), tables=(table(),), figures=(f,)).get("a")


def test_figure_raster_box_is_separate_from_caption_extent():
    f = FigureInput(
        "f",
        "synthetic",
        1,
        (20, 20, 200, 100),
        image_payload(png(), "image/png"),
        caption_ids=("caption",),
    )
    r = (
        RegionIndex("synthetic", PAGES, (line("caption", "Caption", 120),), figures=(f,))
        .get("caption")
        .to_dict()
    )
    assert r["figure_bbox"] == (20, 20, 200, 100) and r["bbox"][3] == 130


def test_same_page_window_never_borrows_a_region_from_another_page():
    index = RegionIndex(
        "synthetic", PAGES, (line("a", "First.", 30), line("b", "Second.", 40, page=2))
    )
    result = index.get("a").to_dict()
    assert not result["neighbourhood"]["after"] and "TRUNCATED_AT_WINDOW" in result["flags"]
    assert len(index.get("a", window=RegionWindow(0, 1, False)).neighbourhood.after) == 1


def test_empty_header_is_not_invented():
    cells = tuple(replace(m, is_header=False) for m in table())
    result = RegionIndex("synthetic", PAGES, (), tables=(cells,)).get(cells[-1].member_id)
    assert "NO_COLUMN_HEADER" in result.flags


def test_semantic_attributes_are_not_part_of_the_member_contract():
    with pytest.raises(TypeError):
        RegionMember(
            "a", "synthetic", 1, "LINE", "source", (20, 20, 100, 30), ("s",), relevant=True
        )


@pytest.mark.parametrize(
    "text",
    ["Display: temperature and pressure.", "Position: upper panel.", "Color: blue.", "5 < T <= 12"],
)
def test_ordinary_source_labels_and_comparators_are_not_css(text):
    result = RegionIndex("synthetic", PAGES, (line("a", text, 30),)).get("a")
    assert result.anchor.text == text


@pytest.mark.parametrize("text", [".panel { display: none }", "display:none", "position:fixed"])
def test_css_rules_and_bare_style_values_are_refused(text):
    with pytest.raises(RegionRefused, match="MARKUP_OR_STYLE"):
        RegionIndex("synthetic", PAGES, (line("a", text, 30),)).get("a")
