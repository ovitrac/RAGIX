"""X23: a table's source lines are claimed by their spans, not only by their boxes (K9.23).

The alias rule claimed a line for a table only when the line's box lay wholly inside the table's cell box. A source
line can lie across the box edge (glyph overshoot, a line running past the last cell) and still be nothing but cell
text: it then stayed a prose line while the same source spans were a cell's text, so one statement was stated twice.

A line is now the table's source line when it lies inside the cell box or when every one of its source spans is
carried by the table's cells. A line of which some spans, not all, are carried stays in its own region, flagged
LINE_PARTLY_IN_TABLE_CELLS: never dropped, and never stated twice without saying so. A line that only touches the box,
sharing no span with its cells, and a line outside the box are unchanged.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from ragix_kernels.harvest.regions import (
    PARTLY_IN_TABLE,
    PageGeometry,
    RegionIndex,
    RegionMember,
    cell_member,
)
from ragix_kernels.harvest.table_context import Cell

PAGES = (PageGeometry(1, 400, 800),)


def table():
    """Two rows of two cells, box (20, 150, 220, 200); each cell carries one source span."""
    members = []
    for r, texts in enumerate((("Channel", "Travel"), ("Axis A", "23"))):
        for c, text in enumerate(texts):
            cell = Cell("synthetic", "table-a", f"cell-{r}-{c}", 1, r, c, text,
                        (20 + c * 100, 150 + r * 25, 120 + c * 100, 175 + r * 25), (f"span-{r}-{c}",))
            members.append(cell_member(cell, is_header=r == 0, is_row_label=r > 0 and c == 0))
    return tuple(members)


def member(ident, text, box, spans):
    return RegionMember(ident, "synthetic", 1, "LINE", text, box, tuple(spans), section_id="s")


LINES = (
    member("inside", "Axis A", (30, 177, 110, 187), ("span-1-0",)),
    member("across_all", "Axis A 23", (30, 177, 260, 187), ("span-1-0", "span-1-1")),
    member("across_some", "Travel and more", (130, 152, 300, 162), ("span-0-1", "tail")),
    member("touching", "Note beside", (200, 190, 330, 199), ("touch",)),
    member("outside", "A later paragraph", (20, 400, 180, 410), ("far",)),
)


def index():
    return RegionIndex("synthetic", PAGES, LINES, tables=(table(),))


def owners(idx):
    out = {}
    for r in idx.regions:
        for m in r.members:
            out.setdefault(m.member_id, []).append(r)
    return out


def test_x23_a_line_inside_the_box_and_a_line_across_it_made_only_of_cell_text_are_the_table_s():
    table_region = next(r for r in index().regions if r.kind == "TABLE")
    aliases = {m.member_id for m in table_region.members if "SOURCE_LINE_ALIAS" in m.flags}
    assert aliases == {"inside", "across_all"}


def test_x23_no_cell_span_is_stated_by_another_region_except_the_flagged_partial_line():
    idx = index()
    cell_spans = {sp for r in idx.regions if r.kind == "TABLE" for m in r.members if m.kind == "CELL"
                  for sp in m.source_spans}
    for r in idx.regions:
        if r.kind == "TABLE":
            continue
        for m in r.members:
            if set(m.source_spans) & cell_spans:
                assert m.member_id == "across_some" and PARTLY_IN_TABLE in m.flags
                assert PARTLY_IN_TABLE in r.flags, "the region says it"


def test_x23_a_partly_carried_line_is_kept_in_its_own_region_never_dropped():
    own = owners(index())["across_some"]
    assert len(own) == 1 and own[0].kind != "TABLE"


def test_x23_a_line_touching_the_box_without_a_shared_span_and_a_line_outside_are_unchanged():
    got = owners(index())
    for ident in ("touching", "outside"):
        (r,) = got[ident]
        m = next(x for x in r.members if x.member_id == ident)
        assert r.kind != "TABLE" and PARTLY_IN_TABLE not in m.flags and "SOURCE_LINE_ALIAS" not in m.flags


def test_x23_every_input_line_is_in_exactly_one_region():
    got = owners(index())
    assert all(len(got[m.member_id]) == 1 for m in LINES)
