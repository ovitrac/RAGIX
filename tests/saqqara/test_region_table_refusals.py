"""One unsupported table must not erase independently observed regions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import pytest

from ragix_kernels.harvest.regions import RegionRefused
from ragix_kernels.saqqara.census import TableObservation, Evidence
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.saqqara.field_views import TextSpan
from ragix_kernels.saqqara.renderable_regions import regions_from_explorer, table_members
from ragix_kernels.saqqara.table_views import TableCell, _native_header_bands
from .test_header_unit_context import table_fixture


def defective_table(shape):
    def cell(ident, box, text="x"):
        return TableCell(ident, text, box, (ident + "-source",), geometry_kind="cell_box")

    header = (cell("bad-h0", (400, 90, 450, 100), "A"), cell("bad-h1", (450, 90, 500, 100), "B"))
    offset = {"gap": 0.01, "overlap": -0.01}.get(shape, 0)
    body = (cell("bad-v0", (400, 100, 450, 110)), cell("bad-v1", (450 + offset, 100, 500, 110)))
    if shape == "merged":
        body = (cell("bad-v0", (400, 100, 500, 110)),)
    elif shape == "empty-row":
        body = ()
    elif shape == "missing-geometry":
        body = (replace(body[0], flags=("MISSING_CELL_GEOMETRY",)), body[1])
    rows = (header, body)
    evidence = tuple(
        Evidence("synthetic", 1, c.cell_id, 0, len(c.text), c.text, c.bbox)
        for row in rows
        for c in row
    )
    return TableObservation("bad", 1, ("A", "B"), (("x", "x"),), evidence, cell_rows=rows)


def document_with_table(shape=None):
    document, good = table_fixture()
    page = replace(
        document.pages[0],
        spans=(
            TextSpan(
                "synthetic", "prose-a", 1, "First independent paragraph.", (30, 350, 220, 362)
            ),
            TextSpan(
                "synthetic", "prose-b", 1, "Second independent paragraph.", (30, 420, 240, 432)
            ),
        ),
        tables=(good,) + ((defective_table(shape),) if shape else ()),
    )
    return replace(document, pages=(page,))


def test_native_bands_control():
    table = defective_table("control")
    assert _native_header_bands(table.cell_rows[0], table.cell_rows[1:]) == ((400, 450), (450, 500))


@pytest.mark.parametrize(
    "shape,code",
    [
        ("gap", "STRADDLING_OR_OUTSIDE_BANDS"),
        ("overlap", "STRADDLING_OR_OUTSIDE_BANDS"),
        ("merged", "BAND_COUNT_VARIES"),
    ],
)
def test_native_band_failure_has_region_exception_type(shape, code):
    table = defective_table(shape)
    with pytest.raises(RegionRefused) as caught:
        _native_header_bands(table.cell_rows[0], table.cell_rows[1:])
    assert caught.value.code == code


@pytest.mark.parametrize(
    "shape,code",
    [
        ("gap", "STRADDLING_OR_OUTSIDE_BANDS"),
        ("overlap", "STRADDLING_OR_OUTSIDE_BANDS"),
        ("merged", "BAND_COUNT_VARIES"),
        ("empty-row", "TABLE_COLUMN_LAYOUT_UNAVAILABLE"),
        ("missing-geometry", "TABLE_CELL_GEOMETRY_UNAVAILABLE"),
    ],
)
def test_bad_table_refusal_preserves_other_regions(shape, code):
    baseline = regions_from_explorer(explore(document_with_table()))
    result = explore(document_with_table(shape))
    view = regions_from_explorer(result)
    assert [asdict(r) for r in view.regions] == [asdict(r) for r in baseline.regions]
    assert {r.kind for r in view.regions} == {"PROSE", "TABLE"}
    assert len(view.refusals) == 1
    refused = view.refusals[0]
    assert refused.source_id == "synthetic" and refused.table_id == "bad"
    assert refused.pages == (1,) and refused.code == code
    assert set(refused.member_ids) == {
        c.cell_id for row in defective_table(shape).cell_rows for c in row
    }
    report = view.refusal_report()
    assert report["count"] == 1 and report["tables"][0]["code"] == code
    with pytest.raises(RegionRefused, match="ANCHOR_NOT_FOUND"):
        view.get("bad-h0")
    # A direct table-only caller without a refusal sink keeps its strict API.
    with pytest.raises(RegionRefused):
        table_members(result)


def test_recovered_table_failure_is_also_local():
    from .fixtures_explorer_slice4 import realistic_table, TableGeometry

    document = realistic_table(TableGeometry(15, 100, 32, 60, 45)).document
    page = replace(
        document.pages[0],
        spans=document.pages[0].spans
        + (
            TextSpan(
                document.source_id, "independent", 1, "Independent paragraph.", (20, 740, 200, 752)
            ),
        ),
        tables=document.pages[0].tables + (defective_table("control"),),
    )
    result = explore(replace(document, pages=(page, *document.pages[1:])))
    recovered = max(result.census.table_analysis.tables, key=lambda t: len(t.rows))
    row = replace(
        recovered.rows[0],
        members=(("missing-physical-cell",), *recovered.rows[0].members[1:]),
        cell_flags=(),
    )
    broken = replace(recovered, rows=(row, *recovered.rows[1:]))
    analysis = replace(
        result.census.table_analysis,
        tables=tuple(
            broken if t.table_id == recovered.table_id else t
            for t in result.census.table_analysis.tables
        ),
    )
    index = regions_from_explorer(
        replace(result, census=replace(result.census, table_analysis=analysis))
    )
    assert len(index.refusals) == 1
    assert index.refusals[0].code == "TABLE_SOURCE_MEMBER_MISSING"
    assert index.refusals[0].table_id == recovered.table_id
    assert index.refusals[0].pages == recovered.pages
    assert index.get("bad-h0").region.kind == "TABLE"
    assert any(m.text == "Independent paragraph." for r in index.regions for m in r.members)
