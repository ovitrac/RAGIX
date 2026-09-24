"""X25: a table nested in another table's cell is that cell's content (K9.25).

A page drawn inside a frame is read as a table whose one cell holds the whole page, and a small ruled grid on that page
is read as a second table. The frame cell already carries the grid's text, so the grid's text was stated twice: once by
the frame cell and once by the grid's cells.

A table whose cell box lies inside another table's cell box (strictly: equal boxes nest neither way), and whose every
source span is carried by one single cell of that table, is now refused as TABLE_INSIDE_TABLE_CELL. The refusal is
counted with the table's cells; its text stays stated once, by the holding cell. A table whose text is spread over
several cells of the other, a duplicate of equal box, a grid carrying no text and a table reaching outside the other's
box are unchanged.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from ragix_kernels.harvest.regions import (
    INSIDE_TABLE_CELL,
    PageGeometry,
    RegionIndex,
    RegionMember,
    TableRegionRefusal,
    cell_member,
)
from ragix_kernels.harvest.table_context import Cell

PAGES = (PageGeometry(1, 400, 800),)


def grid(table_id, cells):
    """cells: (row, column, text, box, spans)."""
    return tuple(
        cell_member(Cell("synthetic", table_id, f"{table_id}-{r}-{c}", 1, r, c, text, box, tuple(spans)))
        for r, c, text, box, spans in cells
    )


def frame(spans=("title", "g-0-0", "g-0-1", "g-1-0", "g-1-1", "body")):
    """One cell holding the page, and a footer cell of its own."""
    return grid("frame", [(0, 0, "Page", (10, 10, 390, 760), spans), (1, 0, "Footer", (10, 760, 390, 790), ("foot",))])


def inner(table_id="grid", box=(100, 300, 300, 400), spans=None):
    x0, y0, x1, y1 = box
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    spans = spans or {(0, 0): ("g-0-0",), (0, 1): ("g-0-1",), (1, 0): ("g-1-0",), (1, 1): ("g-1-1",)}
    boxes = {(0, 0): (x0, y0, mx, my), (0, 1): (mx, y0, x1, my), (1, 0): (x0, my, mx, y1), (1, 1): (mx, my, x1, y1)}
    return grid(table_id, [(r, c, "x" if spans[(r, c)] else "", boxes[(r, c)], spans[(r, c)]) for r, c in boxes])


def line(ident, box, spans):
    return RegionMember(ident, "synthetic", 1, "LINE", ident, box, tuple(spans), section_id="s")


LINES = (
    line("title", (20, 20, 200, 30), ("title",)),
    line("grid_row_0", (105, 320, 295, 330), ("g-0-0", "g-0-1")),
    line("grid_row_1", (105, 370, 295, 380), ("g-1-0", "g-1-1")),
    line("body", (20, 500, 380, 510), ("body",)),
)


def tables_of(idx):
    return {m.table_id for r in idx.regions if r.kind == "TABLE" for m in r.members if m.kind == "CELL"}


def refused(idx):
    return {r.table_id: r for r in idx.refusals}


def test_x25_a_grid_whose_text_one_frame_cell_carries_is_refused_and_counted_with_its_cells():
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(frame(), inner()))
    assert tables_of(idx) == {"frame"}
    r = refused(idx)["grid"]
    assert r.code == INSIDE_TABLE_CELL == "TABLE_INSIDE_TABLE_CELL"
    assert r.pages == (1,) and set(r.member_ids) == {m.member_id for m in inner()}


def test_x25_no_span_is_stated_by_two_regions_and_the_grid_lines_are_the_frame_s_alone():
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(frame(), inner()))
    holders = {}
    for reg in idx.regions:
        for m in reg.members:
            if m.kind == "CELL" or "SOURCE_LINE_ALIAS" not in m.flags:
                for span in m.source_spans:
                    holders.setdefault(span, set()).add(reg.region_id)
    assert all(len(v) == 1 for v in holders.values())
    table = next(reg for reg in idx.regions if reg.kind == "TABLE")
    assert {m.member_id for m in table.members if "SOURCE_LINE_ALIAS" in m.flags} == {"title", "grid_row_0",
                                                                                        "grid_row_1", "body"}
    # one owner each: a line two tables claim is refused as an ambiguous anchor
    assert all(idx.get(ident).region_id == table.region_id for ident in ("grid_row_0", "grid_row_1"))


def test_x25_without_the_rule_both_tables_would_state_the_grid_text():
    """The fixture's own control: the frame cell does carry every grid span."""
    carried = set(frame()[0].source_spans)
    assert {span for m in inner() for span in m.source_spans} <= carried


def test_x25_text_spread_over_two_cells_of_the_other_table_is_not_nesting():
    split = grid("frame", [(0, 0, "Left", (10, 10, 200, 760), ("title", "g-0-0", "g-1-0", "body")),
                           (0, 1, "Right", (200, 10, 390, 760), ("g-0-1", "g-1-1"))])
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(split, inner()))
    assert tables_of(idx) == {"frame", "grid"} and not idx.refusals


def test_x25_equal_boxes_nest_neither_way():
    """Two one-cell tables of the same box and span: each cell carries the other's text, only strictness decides."""
    a, b = (grid(t, [(0, 0, "x", (100, 300, 300, 400), ("g-0-0",))]) for t in ("a", "b"))
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(a, b))
    assert tables_of(idx) == {"a", "b"} and not idx.refusals


def test_x25_a_grid_carrying_no_span_is_never_refused():
    empty = inner(spans={(0, 0): (), (0, 1): (), (1, 0): (), (1, 1): ()})
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(frame(), empty))
    assert tables_of(idx) == {"frame", "grid"} and not idx.refusals


def test_x25_a_grid_reaching_outside_the_frame_box_is_unchanged():
    wide = inner(box=(100, 300, 395, 400))
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(frame(), wide))
    assert tables_of(idx) == {"frame", "grid"} and not idx.refusals


def test_x25_every_level_of_a_nest_is_refused_and_the_outermost_table_kept():
    middle = inner("middle", box=(50, 250, 350, 450),
                   spans={(0, 0): ("g-0-0", "g-0-1", "g-1-0", "g-1-1"), (0, 1): (), (1, 0): (), (1, 1): ()})
    innermost = grid("innermost", [(0, 0, "x", (60, 260, 190, 340), ("g-0-0", "g-0-1"))])
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(frame(), middle, innermost))
    assert tables_of(idx) == {"frame"}
    assert set(refused(idx)) == {"middle", "innermost"}


def test_x25_refusals_supplied_by_the_caller_are_kept_first():
    given = TableRegionRefusal("synthetic", "table:9:9", (1,), "STRADDLING_OR_OUTSIDE_BANDS", ("c-9",))
    idx = RegionIndex("synthetic", PAGES, LINES, tables=(frame(), inner()), refusals=(given,))
    assert [r.table_id for r in idx.refusals] == ["table:9:9", "grid"]


def test_x25_a_table_continuing_on_a_page_the_other_does_not_reach_is_unchanged():
    pages = (PageGeometry(1, 400, 800), PageGeometry(2, 400, 800))
    two_pages = (*inner(), cell_member(Cell("synthetic", "grid", "grid-2-0", 2, 2, 0, "x", (100, 50, 300, 90),
                                             ("g-2-0",))))
    carrier = grid("frame", [(0, 0, "Page", (10, 10, 390, 760),
                              ("title", "g-0-0", "g-0-1", "g-1-0", "g-1-1", "g-2-0", "body"))])
    idx = RegionIndex("synthetic", pages, LINES, tables=(carrier, two_pages))
    assert tables_of(idx) == {"frame", "grid"} and not idx.refusals
