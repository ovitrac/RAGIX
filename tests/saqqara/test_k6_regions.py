"""
Gate K6, vector regions — when a page of ink amounts to a figure, and at what cost.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-29

Carries SPEC.md K6.14, K6.15 and K6.16.

The refusals are as much the subject as the promotions: a rule line is the commonest mark in a
tender document and the first thing a naive area test promotes. And the licence claim is tested the
strong way — not "the kernel does not call the AGPL library" but "the process has not loaded it".
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
from ragix_kernels.saqqara.analyzers.vector_regions import (  # noqa: E402
    MIN_REGION_AREA,
    MIN_REGION_OPS,
    REGION_FACTS,
    REGION_REFUSALS,
    RENDER_CHANNEL,
    VectorRegionAnalyzer,
)
from ragix_kernels.saqqara.assets import AssetStore  # noqa: E402
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402
from ragix_kernels.saqqara.render import raster_key  # noqa: E402
from ragix_kernels.saqqara.render.guard import (  # noqa: E402
    AGPL_MODULES,
    EXEMPT,
    loaded_agpl_modules,
    scan_sources,
)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("k6region")
    out = {}
    for name in ("pdf_vector_region", "pdf_no_objects"):
        path = G.FIXTURES[name](G.fixture_path(name, root))
        adapter = adapter_for(path)
        store = AssetStore(root / f"{name}.store")
        records = read_path(path, store=store)
        tree = build_tree(records, str(path), adapter.format, adapter.format,
                          adapter.version).tree
        out[name] = (path, store, tree, VectorRegionAnalyzer(store=store).run(tree))
    return out


def _regions(tree):
    return [n for n in tree.walk() if n.kind == "vector_region"]


# ------------------------------------ K6.14 the minimums, and every refusal counted

def test_k6_14_a_drawing_worth_the_name_is_promoted(rendered):
    _, _, _, result = rendered["pdf_vector_region"]
    assert result.trace["promoted"] == 1
    region = _regions(result.tree)[0]
    assert region.facts["ops"] >= MIN_REGION_OPS
    assert (region.facts["x"], region.facts["y"]) == (100.0, 500.0)
    assert (region.facts["w"], region.facts["h"]) == (195.0, 150.0)


def test_k6_14_a_rule_line_is_refused_and_counted(rendered):
    """The commonest mark in a tender document, and never a figure."""
    _, _, _, result = rendered["pdf_vector_region"]
    assert result.trace["refused"].get("too-few-operators") == 1


def test_k6_14_ink_without_extent_is_refused_and_counted(rendered):
    _, _, _, result = rendered["pdf_vector_region"]
    assert result.trace["refused"].get("too-small") == 1


def test_k6_14_every_cluster_is_promoted_or_refused(rendered):
    """The invariant: nothing considered simply disappears."""
    _, _, _, result = rendered["pdf_vector_region"]
    assert result.trace["clusters"] == result.trace["promoted"] + sum(
        result.trace["refused"].values()
    )


def test_k6_14_refusal_reasons_come_from_the_closed_vocabulary(rendered):
    seen = set()
    for _, _, _, result in rendered.values():
        seen |= set(result.trace["refused"])
    assert seen <= set(REGION_REFUSALS)
    assert {"too-few-operators", "too-small"} <= seen, (
        "a declared reason nothing produces is a wish"
    )


def test_k6_14_a_page_of_prose_shows_that_it_was_looked_at(rendered):
    """The negative. Absence from the counts is not the same as having looked."""
    _, _, _, result = rendered["pdf_no_objects"]
    assert result.trace["pages_examined"] == 1
    assert result.trace["marks"] == 0
    assert result.trace["clusters"] == 0
    assert result.trace["promoted"] == 0
    assert _regions(result.tree) == []


def test_k6_14_the_minimums_are_declared_not_buried():
    assert MIN_REGION_OPS == 8 and MIN_REGION_AREA == 0.01


# --------------------------------- K6.15 identity is the source, never the raster

def test_k6_15_a_rendered_region_says_it_is_inferred(rendered):
    _, _, _, result = rendered["pdf_vector_region"]
    region = _regions(result.tree)[0]
    assert region.origin == "inferred"
    assert 0 < region.confidence < 1
    assert region.facts["channel"] == RENDER_CHANNEL
    assert region.facts["source"] == "render", "never confusable with an extracted image"
    assert set(REGION_FACTS) <= set(region.facts)


def test_k6_15_the_asset_is_the_source_and_the_store_can_produce_it(rendered):
    """A hash of something nobody kept would fail K6.6 the moment it was checked."""
    _, store, _, result = rendered["pdf_vector_region"]
    region = _regions(result.tree)[0]
    payload = store.read(region.facts["asset"])
    described = json.loads(payload.decode("utf-8"))
    assert described["box"] and described["marks"], described
    assert sum(m["ops"] for m in described["marks"]) == region.facts["ops"]


class _StubRenderer:
    """A different renderer, with different pixels. Nothing else about it matters."""

    name = "stub-renderer"
    version = "9.9.9"

    def render(self, path, page_number, box, dpi):
        return (b"not-a-real-raster-and-deliberately-so", "image/png")


def test_k6_15_the_identity_does_not_move_when_the_renderer_does(tmp_path):
    """The claim K4 rests on, tested by actually swapping the renderer."""
    path = G.FIXTURES["pdf_vector_region"](tmp_path / "region.pdf")
    adapter = adapter_for(path)

    assets, rasters = [], []
    for index, renderer in enumerate((None, _StubRenderer())):
        store = AssetStore(tmp_path / f"store{index}")
        records = read_path(path, store=store)
        tree = build_tree(records, str(path), adapter.format, adapter.format,
                          adapter.version).tree
        result = VectorRegionAnalyzer(store=store, renderer=renderer).run(tree)
        region = _regions(result.tree)[0]
        assets.append(region.facts["asset"])
        rasters.append({d for d in store.ids() if d != region.facts["asset"]})

    assert assets[0] == assets[1], (
        "the region's identity changed when only the renderer did"
    )
    assert rasters[0] != rasters[1], "and the pixels did change, or this proves nothing"


def test_k6_15_the_raster_key_carries_every_term_that_changes_pixels():
    keys = {
        raster_key("abc", "pypdfium2", "1.0", 150),
        raster_key("abc", "pymupdf", "1.0", 150),
        raster_key("abc", "pypdfium2", "2.0", 150),
        raster_key("abc", "pypdfium2", "1.0", 300),
        raster_key("def", "pypdfium2", "1.0", 150),
    }
    assert len(keys) == 5, "source, renderer, version and resolution each move the key"


def test_k6_15_the_raster_is_stored_with_its_conditions_not_on_the_node(rendered):
    _, store, _, result = rendered["pdf_vector_region"]
    region = _regions(result.tree)[0]
    manifest = store.manifest()
    rasters = [
        entry for digest, entry in manifest.items()
        if digest != region.facts["asset"]
    ]
    assert len(rasters) == 1, "one rendering of one region"
    reference = rasters[0]["references"][0]
    assert reference["derived_from"] == region.facts["asset"]
    for key in ("raster_key", "renderer", "renderer_version", "dpi"):
        assert reference.get(key) is not None, key
    assert rasters[0]["media_type"] == "image/png"
    assert not set(REGION_FACTS) & {"renderer", "dpi", "raster_key"}


def test_k6_15_no_raster_bytes_are_reachable_from_the_tree(rendered):
    _, _, _, result = rendered["pdf_vector_region"]
    for node in result.tree.walk():
        for value in node.facts.values():
            assert not isinstance(value, (bytes, bytearray))


# ------------------------------------------- K6.16 the AGPL renderer is not loaded

def test_k6_16_no_kernel_source_imports_the_agpl_library():
    violations = scan_sources()
    assert violations == [], violations


def test_k6_16_the_exemption_is_one_named_file():
    assert EXEMPT == ("render/mupdf.py",)
    exempt = Path(__file__).resolve().parents[2] / "ragix_kernels/saqqara" / EXEMPT[0]
    assert exempt.is_file(), "the exemption must name a file that exists"


def test_k6_16_the_scan_bites(tmp_path):
    """A guard nobody has watched refuse is not a guard."""
    root = tmp_path / "kernel"
    (root / "render").mkdir(parents=True)
    (root / "somewhere.py").write_text("import fitz\n", encoding="utf-8")
    (root / "render" / "mupdf.py").write_text("import pymupdf\n", encoding="utf-8")
    found = scan_sources(root)
    assert found == [("somewhere.py", 1, "fitz")], found


def test_k6_16_the_default_route_does_not_load_it(rendered):
    """The strong claim: not that we avoid calling it, but that it is not loaded."""
    assert loaded_agpl_modules() == [], (
        "an AGPL module is in sys.modules after reading and rendering a document "
        "by the default route"
    )


def test_k6_16_the_default_renderer_is_the_permissive_one(rendered):
    _, _, _, result = rendered["pdf_vector_region"]
    assert result.trace["renderer"].startswith("pypdfium2")
    assert not any(name in result.trace["renderer"] for name in AGPL_MODULES)


# ---------------------------------- K6.17 a region is a drawing, not a page

@pytest.fixture(scope="module")
def furniture(tmp_path_factory):
    root = tmp_path_factory.mktemp("k6furniture")
    path = G.FIXTURES["pdf_page_furniture"](G.fixture_path("pdf_page_furniture", root))
    adapter = adapter_for(path)
    store = AssetStore(root / "store")
    records = read_path(path, store=store)
    tree = build_tree(records, str(path), adapter.format, adapter.format,
                      adapter.version).tree
    return tree, VectorRegionAnalyzer(store=store, keep_raster=False).run(tree)


def test_k6_17_the_region_is_the_drawing_not_the_page(furniture):
    """The defect this proposition exists for, in one assertion.

    Before furniture was excluded, this page produced one region of 595 x 842 --
    the MediaBox -- around a drawing 120 points wide.
    """
    _, result = furniture
    regions = [n for n in _regions(result.tree)]
    assert len(regions) == 1, "one drawing on page one, nothing on page two"
    facts = regions[0].facts
    assert (facts["x"], facts["y"]) == (200.0, 300.0)
    assert (facts["w"], facts["h"]) == (120.0, 100.0)


def test_k6_17_furniture_is_excluded_and_counted(furniture):
    """Three marks removed: a background box, a full-width rule, a full-height one."""
    _, result = furniture
    assert result.trace["excluded"].get("page-furniture") == 3


def test_k6_17_a_chain_that_still_spans_the_page_is_refused(furniture):
    """The backstop, on a page carrying no furniture at all."""
    _, result = furniture
    assert result.trace["refused"].get("region-spans-page") == 1


def test_k6_17_every_mark_is_grouped_or_excluded(furniture):
    """Nothing considered simply disappears, one level below the cluster."""
    tree, result = furniture
    marks = sum(len(p.facts.get("drawings") or [])
                for p in tree.walk() if p.kind == "page")
    assert marks == result.trace["marks"]
    assert result.trace["marks"] == (result.trace["grouped"]
                                     + sum(result.trace["excluded"].values()))


def test_k6_17_the_page_size_is_read_not_assumed(furniture):
    """An extent minimum measured against a guessed page is measuring the guess."""
    tree, _ = furniture
    for page in [n for n in tree.walk() if n.kind == "page"]:
        assert page.facts["width"] == 595.0
        assert page.facts["height"] == 842.0


def test_k6_17_the_exclusion_vocabulary_is_closed_and_produced(furniture):
    from ragix_kernels.saqqara.analyzers.vector_regions import MARK_EXCLUSIONS

    _, result = furniture
    seen = set(result.trace["excluded"])
    assert seen <= set(MARK_EXCLUSIONS)
    assert seen == set(MARK_EXCLUSIONS), "a declared reason nothing produces is a wish"


def test_k6_17_the_existing_region_fixture_is_untouched(rendered):
    """Furniture exclusion must not eat the drawing it was written to protect."""
    _, _, _, result = rendered["pdf_vector_region"]
    assert result.trace["promoted"] == 1
    assert result.trace["excluded"] == {}
