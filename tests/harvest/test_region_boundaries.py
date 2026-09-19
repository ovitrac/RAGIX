"""Independent planted paragraph/list boundaries and baseline comparison.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace
from ragix_kernels.harvest.region_types import RegionMember, PageGeometry, BoundaryPolicy
from ragix_kernels.harvest.region_boundaries import group_lines


def line(id, text, y, *, x=20, right=180, page=1, kind="LINE", section="section-a"):
    return RegionMember(
        id,
        "synthetic",
        page,
        kind,
        text,
        (x, y, right, y + 10),
        (id + "-source",),
        section_id=section,
    )


def fixtures():
    return (
        (
            (
                line("a", "First sentence", 30),
                line("b", "continues here.", 43),
                line("c", "New paragraph", 80),
                line("d", "continues too.", 93),
            ),
            ({"a", "b"}, {"c", "d"}),
        ),
        (
            (
                line("a", "Left first", 30, right=120),
                line("b", "Left second.", 43, right=120),
                line("c", "Right first", 30, x=220, right=320),
                line("d", "Right second.", 43, x=220, right=320),
            ),
            ({"a", "b"}, {"c", "d"}),
        ),
        (
            (
                line("a", "Indented opener", 30, x=30),
                line("b", "body continuation.", 43),
                line("c", "Separate paragraph.", 85),
            ),
            ({"a", "b"}, {"c"}),
        ),
        (
            (
                line("a", "Section title", 20, kind="HEADING"),
                line("b", "Paragraph begins", 33),
                line("c", "and continues.", 46),
            ),
            ({"a"}, {"b", "c"}),
        ),
        (
            (line("a", "A paragraph continues", 775), line("b", "on the next page.", 25, page=2)),
            ({"a", "b"},),
        ),
    )


def score(mode):
    results = []
    for lines, truth in fixtures():
        groups = group_lines(
            lines, (PageGeometry(1, 400, 800), PageGeometry(2, 400, 800)), BoundaryPolicy(mode=mode)
        )
        actual = {frozenset(m.member_id for m in g.members) for g in groups}
        results.append(actual == {frozenset(x) for x in truth})
    return sum(results), len(results)


def test_geometry_beats_line_and_section_baselines_on_planted_boundaries():
    assert score("geometry") == (5, 5)
    assert score("line")[0] < 5 and score("section")[0] < 5


def test_lists_keep_markers_and_wrapped_lines():
    lines = (
        line("a", "• First item", 20),
        line("b", "wrapped detail", 33, x=35),
        line("c", "• Second item", 46),
        line("d", "Ordinary paragraph.", 90),
    )
    groups = group_lines(lines, (PageGeometry(1, 400, 800),))
    assert [(g.kind, [m.member_id for m in g.members]) for g in groups] == [
        ("LIST", ["a", "b", "c"]),
        ("PROSE", ["d"]),
    ]
    assert [m.text for m in groups[0].members] == [m.text for m in lines[:3]]
    assert [m.list_item for m in groups[0].members] == [0, 0, 1]


def test_numbered_heading_is_not_automatically_a_list():
    groups = group_lines(
        (line("a", "2. A heading", 20, kind="HEADING"), line("b", "Plain body.", 33)),
        (PageGeometry(1, 400, 800),),
    )
    assert all(g.kind == "PROSE" for g in groups)


def test_same_page_table_barrier_stops_prose_join():
    lines = (line("a", "Before", 20), line("b", "After", 40))
    groups = group_lines(lines, (PageGeometry(1, 400, 800),), barriers=((1, (0, 31, 300, 39)),))
    assert len(groups) == 2


def test_unclear_page_boundary_is_flagged_and_declared_continuation_can_resolve_it():
    lines = (line("a", "Sentence ends.", 775), line("b", "Another sentence.", 25, page=2))
    pages = (PageGeometry(1, 400, 800), PageGeometry(2, 400, 800))
    groups = group_lines(lines, pages)
    assert len(groups) == 2 and all("PAGE_BOUNDARY_UNRESOLVED" in g.flags for g in groups)
    groups = group_lines(lines, pages, continuations=(("a", "b"),))
    assert len(groups) == 1 and "DECLARED_CONTINUATION" in groups[0].flags


def test_page_continuation_never_jumps_a_heading():
    lines = (
        line("a", "Unfinished line", 775),
        line("h", "New section", 10, page=2, kind="HEADING"),
        line("b", "New body", 30, page=2),
    )
    groups = group_lines(lines, (PageGeometry(1, 400, 800), PageGeometry(2, 400, 800)))
    assert not any({m.member_id for m in g.members} >= {"a", "b"} for g in groups)


def test_declared_geometry_sensitivity_on_planted_layouts():
    from itertools import product

    for gap, indent, edge in product((0.5, 1.0, 1.5), (1.0, 1.5, 2.0), (0.05, 0.12, 0.2)):
        for lines, truth in fixtures():
            groups = group_lines(
                lines,
                (PageGeometry(1, 400, 800), PageGeometry(2, 400, 800)),
                BoundaryPolicy(max_gap_ratio=gap, max_indent_ratio=indent, page_edge_fraction=edge),
            )
            assert {frozenset(m.member_id for m in g.members) for g in groups} == {
                frozenset(t) for t in truth
            }
