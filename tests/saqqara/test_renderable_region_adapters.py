"""Region envelopes reuse observations; their renderer and semantics stay outside.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace
import hashlib
from pathlib import Path
import pytest
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.saqqara.renderable_regions import regions_from_explorer, figures_from_tree
from ragix_kernels.saqqara.assets import AssetStore
from ragix_kernels.saqqara.model import Node, Tree, Provenance, PdfLocator
from ragix_kernels.harvest.regions import RegionRefused, RegionIndex, PageGeometry, RegionWindow
from ragix_kernels.harvest.table_context import context_from_dict
from tests.harvest.test_renderable_regions import png
from tests.harvest.test_region_boundaries import line
from .test_header_unit_context import table_fixture


def tree(asset, *, kind="figure", rule=None, source="synthetic"):
    prov = Provenance(
        "synthetic.pdf", "pdf", (PdfLocator(page=1),), "synthetic", "1", source_sha256=source
    )
    facts = {"asset": asset, "x": 20, "y": 700, "w": 180, "h": 80}
    if rule:
        facts["rule"] = rule
    child = Node(
        kind,
        prov,
        facts=facts,
        origin="inferred" if kind == "vector_region" else "read",
        confidence=0.8 if kind == "vector_region" else 1,
    )
    return Tree(Node("document", prov, children=[child]))


def transform(page, box):
    x, bottom, right, top = box
    return x, 800 - top, right, 800 - bottom


def test_native_cell_context_anchor_returns_all_original_cells():
    doc, observed = table_fixture()
    result = explore(doc)
    ctx = context_from_dict(result.reading.cell_contexts[0])
    region = regions_from_explorer(result).get(ctx.value.span()).to_dict()
    assert region["kind"] == "TABLE"
    expected = {c.cell_id: c.text for row in observed.cell_rows for c in row}
    assert {m["member_id"]: m["text"] for m in region["members"]} == expected
    assert sum(m["is_header"] for m in region["members"]) == 3
    assert sum(m["is_row_label"] for m in region["members"]) == 4


def test_recovered_table_keeps_identifier_cells_and_repeated_headers():
    from .fixtures_explorer_slice4 import realistic_table, TableGeometry

    result = explore(realistic_table(TableGeometry(15, 100, 32, 60, 45)).document)
    index = regions_from_explorer(result)
    tables = [r for r in index.regions if r.kind == "TABLE"]
    assert len(tables) >= 1
    target = max(tables, key=lambda r: len(r.members))
    assert "CROSSES_PAGE" in target.flags
    ids = {m.member_id for m in target.members}
    recovered = max(result.census.table_analysis.tables, key=lambda t: len(t.rows))
    assert all(ident in ids for row in recovered.rows for group in row.members for ident in group)


def test_lattice_strokes_are_not_promoted_by_region_envelope():
    from ragix_kernels.saqqara.census import DocumentDigest, PageDigest
    from ragix_kernels.saqqara.field_views import TextSpan, VerticalRule, HorizontalRule

    span = TextSpan("synthetic", "label", 1, "A drawing label", (30, 60, 130, 72))
    doc = DocumentDigest(
        "synthetic",
        (
            PageDigest(
                1,
                400,
                800,
                (span,),
                rules=tuple(VerticalRule(x, 40, 160) for x in (20, 80, 140)),
                horizontal_rules=tuple(HorizontalRule(y, 20, 140) for y in (40, 100, 160)),
                drawing_count=6,
            ),
        ),
        "synthetic",
        "1",
    )
    result = explore(doc)
    index = regions_from_explorer(result)
    assert all(r.kind != "TABLE" for r in index.regions)
    assert index.get(result.report.presentation_lines[0]["line_id"]).region.kind == "PROSE"


def test_existing_asset_becomes_base64_without_changing_tree(tmp_path):
    store = AssetStore(tmp_path / "assets")
    asset = store.put(png(), "image/png", {"page": 1})
    source = tree(asset)
    before = source.to_json()
    figures = figures_from_tree(source, store, source_id="synthetic", to_region_box=transform)
    assert source.to_json() == before and len(figures) == 1
    assert figures[0].bbox == (20, 20, 200, 100) and figures[0].image.encoding == "base64"
    assert "source_path" not in figures[0].image.__dict__


def test_lattice_with_stored_raster_stays_figure_not_table(tmp_path, monkeypatch):
    from ragix_kernels.saqqara import model

    monkeypatch.setattr(
        model, "kind_registry", model.KindRegistry((*model.kind_registry.known(), "vector_region"))
    )
    store = AssetStore(tmp_path / "assets")
    asset = store.put(b'{"marks":[]}', "application/json", {})
    raster = store.put(png(), "image/png", {"derived_from": asset})
    figures = figures_from_tree(
        tree(asset, kind="vector_region", rule="table-as-image"),
        store,
        source_id="synthetic",
        to_region_box=transform,
    )
    assert figures[0].image.sha256 == raster and "LATTICE_AS_IMAGE" in figures[0].flags
    assert "FIGURE_STRUCTURE_INFERRED" in figures[0].flags
    index = RegionIndex(
        "synthetic",
        (PageGeometry(1, 400, 800),),
        (line("a", "Drawing label", 30),),
        figures=figures,
    )
    assert index.get("a").region.kind == "FIGURE"


def test_missing_and_ambiguous_rasters_are_explicit(tmp_path):
    store = AssetStore(tmp_path / "assets")
    asset = store.put(b"raw pixels", "image/x-raw", {})
    figures = figures_from_tree(tree(asset), store, source_id="synthetic", to_region_box=transform)
    assert figures[0].image is None and figures[0].image_reason == "FIGURE_RASTER_UNAVAILABLE"
    a = store.put(png(), "image/png", {"derived_from": asset})
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (5, 3), (1, 2, 3)).save(buf, format="PNG")
    store.put(buf.getvalue(), "image/png", {"derived_from": asset})
    figures = figures_from_tree(tree(asset), store, source_id="synthetic", to_region_box=transform)
    assert figures[0].image_reason == "FIGURE_RASTER_AMBIGUOUS"
    figures = figures_from_tree(
        tree(asset),
        store,
        source_id="synthetic",
        to_region_box=transform,
        raster_choices={asset: a},
    )
    assert figures[0].image.sha256 == a


def test_explicit_renderer_fallback_binds_to_original_file_and_records_parameters(tmp_path):
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"synthetic renderer input")
    sid = hashlib.sha256(path.read_bytes()).hexdigest()
    store = AssetStore(tmp_path / "assets")
    asset = store.put(b"raw pixels", "image/x-raw", {})

    class Renderer:
        name = "synthetic"
        version = "1"

        def render(self, p, page, box, dpi):
            assert p == path and page == 1 and box == (20, 700, 200, 780) and dpi == 96
            return png(), "image/png"

    figures = figures_from_tree(
        tree(asset, source=sid),
        store,
        source_id=sid,
        to_region_box=transform,
        renderer=Renderer(),
        source_path=path,
        dpi=96,
    )
    image = figures[0].image
    assert image.renderer == "synthetic" and image.dpi == 96 and image.source_asset == asset
    assert path.read_bytes() == b"synthetic renderer input"
    with pytest.raises(RegionRefused, match="SOURCE_HASH"):
        figures_from_tree(
            tree(asset, source="synthetic"),
            store,
            source_id="synthetic",
            to_region_box=transform,
            renderer=Renderer(),
            source_path=path,
        )


def test_corrupt_asset_is_a_failure_not_an_absence(tmp_path):
    store = AssetStore(tmp_path / "assets")
    asset = store.put(png(), "image/png", {})
    (store.root / asset).write_bytes(b"changed")
    figures = figures_from_tree(tree(asset), store, source_id="synthetic", to_region_box=transform)
    assert figures[0].image_reason == "FIGURE_ASSET_MISSING_OR_CHANGED"


def test_stale_report_line_is_refused():
    from ragix_kernels.saqqara.census import DocumentDigest, PageDigest
    from ragix_kernels.saqqara.field_views import TextSpan

    doc = DocumentDigest(
        "synthetic",
        (PageDigest(1, 400, 800, (TextSpan("synthetic", "s", 1, "Original", (20, 30, 100, 40)),)),),
        "synthetic",
        "1",
    )
    result = explore(doc)
    altered = replace(
        result.report,
        presentation_lines=tuple(
            {**p, "text": "Changed"} for p in result.report.presentation_lines
        ),
    )
    with pytest.raises(RegionRefused, match="STALE_PRESENTATION"):
        regions_from_explorer(replace(result, report=altered))


def test_missing_raw_asset_is_not_misreported_as_missing_raster(tmp_path):
    store = AssetStore(tmp_path / "assets")
    asset = store.put(b"raw pixels", "image/x-raw", {})
    (store.root / asset).unlink()
    figures = figures_from_tree(tree(asset), store, source_id="synthetic", to_region_box=transform)
    assert figures[0].image_reason == "FIGURE_ASSET_MISSING_OR_CHANGED"


def test_render_source_changes_are_refused(tmp_path):
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"original renderer input")
    sid = hashlib.sha256(path.read_bytes()).hexdigest()
    store = AssetStore(tmp_path / "assets")
    asset = store.put(b"raw", "image/x-raw", {})

    class Renderer:
        name = "synthetic"
        version = "1"

        def render(self, *args):
            path.write_bytes(b"changed renderer input")
            return png(), "image/png"

    with pytest.raises(RegionRefused, match="RENDER_SOURCE_CHANGED"):
        figures_from_tree(
            tree(asset, source=sid),
            store,
            source_id=sid,
            to_region_box=transform,
            renderer=Renderer(),
            source_path=path,
        )


def test_inferred_ordinary_vector_figure_retains_its_origin(tmp_path, monkeypatch):
    from ragix_kernels.saqqara import model

    monkeypatch.setattr(
        model, "kind_registry", model.KindRegistry((*model.kind_registry.known(), "vector_region"))
    )
    store = AssetStore(tmp_path / "assets")
    asset = store.put(b'{"marks":[]}', "application/json", {})
    store.put(png(), "image/png", {"derived_from": asset})
    figures = figures_from_tree(
        tree(asset, kind="vector_region", rule="region-render"),
        store,
        source_id="synthetic",
        to_region_box=transform,
    )
    assert "FIGURE_STRUCTURE_INFERRED" in figures[0].flags
    index = RegionIndex(
        "synthetic",
        (PageGeometry(1, 400, 800),),
        (line("a", "Drawing label", 30),),
        figures=figures,
    )
    assert "FIGURE_STRUCTURE_INFERRED" in index.get("a").flags
