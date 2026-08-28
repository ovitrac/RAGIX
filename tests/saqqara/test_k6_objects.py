"""
Gate K6 — objects: what a document shows rather than says.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-28

Carries SPEC.md K6.1-K6.6: the asset store, and images read one node per placement.

The distinction under test throughout is between what a file contains and what its parts have to do
with each other. Extraction is an observation and claims nothing; everything that interprets comes
later and says so.
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
from ragix_kernels.saqqara.assets import (  # noqa: E402
    AssetStore,
    MissingAsset,
    asset_id,
)
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every object fixture, read once, with a store beside it."""
    root = tmp_path_factory.mktemp("k6")
    out = {}
    for name in ("pdf_image_xobject", "pdf_image_twice", "pdf_inline_image"):
        path = G.FIXTURES[name](G.fixture_path(name, root))
        store = AssetStore(root / f"{name}.assets")
        out[name] = (path, store, read_path(path, store=store))
    return out


def _figures(records):
    return [r for r in records if r.kind == "figure"]


# --------------------------------------------- K6.1 an observation, one per placement

def test_k6_1_an_image_is_read_as_an_observation(built):
    _, _, records = built["pdf_image_xobject"]
    figures = _figures(records)
    assert len(figures) == 1
    facts = figures[0].facts
    assert facts["source"] == "xobject"
    assert facts["media_type"] and facts["asset"]
    assert facts["width"] == 4 and facts["height"] == 4


def test_k6_1_the_reader_declares_the_figure_vocabulary(built):
    _, _, records = built["pdf_image_xobject"]
    adapter = adapter_for(built["pdf_image_xobject"][0])
    declared = adapter.fact_sets["figure"]
    assert set(_figures(records)[0].facts) == set(declared)


def test_k6_1_a_figure_becomes_a_read_node(built):
    path, _, records = built["pdf_image_xobject"]
    adapter = adapter_for(path)
    tree = build_tree(records, str(path), adapter.format, adapter.format,
                      adapter.version).tree
    figures = [n for n in tree.walk() if n.kind == "figure"]
    assert len(figures) == 1
    assert figures[0].origin == "read" and figures[0].confidence == 1.0


# ------------------------------------------------------- K6.2 no bytes in the tree

def test_k6_2_no_image_bytes_are_reachable_from_the_tree(built):
    path, _, records = built["pdf_image_xobject"]
    adapter = adapter_for(path)
    tree = build_tree(records, str(path), adapter.format, adapter.format,
                      adapter.version).tree
    payload = json.dumps(tree.to_dict(), sort_keys=True, ensure_ascii=False)
    assert json.loads(payload) == tree.to_dict()
    for node in tree.walk():
        for value in node.facts.values():
            assert not isinstance(value, (bytes, bytearray)), "bytes in the tree"


def test_k6_2_the_asset_is_a_hash_and_nothing_more(built):
    _, store, records = built["pdf_image_xobject"]
    asset = _figures(records)[0].facts["asset"]
    assert isinstance(asset, str) and len(asset) == 64
    assert all(c in "0123456789abcdef" for c in asset)
    assert asset == asset_id(store.read(asset))


# ------------------------------------------ K6.3 two placements, two nodes, one asset

def test_k6_3_one_image_drawn_twice_is_two_nodes(built):
    _, _, records = built["pdf_image_twice"]
    figures = _figures(records)
    assert len(figures) == 2


def test_k6_3_the_two_nodes_share_one_asset(built):
    _, store, records = built["pdf_image_twice"]
    figures = _figures(records)
    assets = {f.facts["asset"] for f in figures}
    assert len(assets) == 1, "the file holds the bytes once"
    assert len(store.ids()) == 1, "and the store keeps them once"


def test_k6_3_the_manifest_records_both_positions(built):
    _, store, records = built["pdf_image_twice"]
    asset = _figures(records)[0].facts["asset"]
    entry = store.manifest()[asset]
    assert entry["bytes"] > 0 and entry["media_type"]
    assert len(entry["references"]) == 2, "one reference per placement"


def test_k6_3_the_two_nodes_differ_in_locator(built):
    _, _, records = built["pdf_image_twice"]
    a, b = _figures(records)
    assert a.locator.to_dict() != b.locator.to_dict()


# ------------------------------------------------- K6.4 the box is the matrix's box

def test_k6_4_the_box_comes_from_the_matrix_not_the_pixels(built):
    """The fixture places a 4x4 image at 100x50 points. Pixels are not geometry."""
    _, _, records = built["pdf_image_xobject"]
    facts = _figures(records)[0].facts
    assert (facts["x"], facts["y"]) == (72.0, 700.0)
    assert (facts["w"], facts["h"]) == (100.0, 50.0)
    assert (facts["width"], facts["height"]) == (4, 4)


def test_k6_4_two_placements_of_one_image_have_different_boxes(built):
    _, _, records = built["pdf_image_twice"]
    a, b = _figures(records)
    assert (a.facts["w"], a.facts["h"]) == (100.0, 50.0)
    assert (b.facts["w"], b.facts["h"]) == (60.0, 30.0)
    assert (a.facts["x"], a.facts["y"]) != (b.facts["x"], b.facts["y"])


# ------------------------------------------------------- K6.5 the inline image skip

def test_k6_5_an_inline_image_is_skipped_by_name_and_counted(built):
    path, store, records = built["pdf_inline_image"]
    adapter = adapter_for(path)
    assert adapter.skips["inline-image-not-extracted"] >= 1


def test_k6_5_the_object_image_is_still_read(built):
    """A fixture whose only image were inline could not tell a skip from a miss."""
    _, _, records = built["pdf_inline_image"]
    assert len(_figures(records)) == 1


def test_k6_5_source_declares_only_values_something_emits(built):
    from ragix_kernels.saqqara.adapters.pdf import FIGURE_SOURCES

    emitted = set()
    for _, _, records in built.values():
        emitted |= {f.facts["source"] for f in _figures(records)}
    assert set(FIGURE_SOURCES) == emitted, (
        "a declared source value nothing produces is the kind_hint defect"
    )


# ------------------------------------------------ K6.6 a missing asset is a failure

def test_k6_6_a_missing_asset_is_a_named_failure(tmp_path):
    store = AssetStore(tmp_path / "store")
    digest = store.put(b"some generated bytes", "image/png", reference={"page": 1})
    (store.root / digest).unlink()
    with pytest.raises(MissingAsset) as caught:
        store.read(digest)
    assert digest[:8] in str(caught.value)


def test_k6_6_the_store_verifies_what_it_returns(tmp_path):
    """A hash that does not describe its bytes is the same failure, later."""
    store = AssetStore(tmp_path / "store")
    digest = store.put(b"some generated bytes", "image/png", reference={"page": 1})
    (store.root / digest).write_bytes(b"different bytes entirely")
    with pytest.raises(MissingAsset):
        store.read(digest)


# ------------------------------------------------- K6.7 nothing is dropped in silence

def test_k6_7_an_undecodable_image_is_counted_not_dropped(tmp_path):
    """The defect the first corpus measurement of this layer exposed.

    A reader that drops what it cannot decode reports a document as holding fewer
    pictures than it does, and nothing anywhere records that the others were seen.
    """
    path = G.FIXTURES["pdf_image_unreadable"](tmp_path / "broken.pdf")
    store = AssetStore(tmp_path / "store")
    adapter = adapter_for(path)
    adapter.skips.clear()
    records = read_path(path, store=store)

    assert len(_figures(records)) == 1, "the readable image is still read"
    assert adapter.skips.get("xobject-empty") == 1, (
        "the stream that decodes to nothing is counted, not stored"
    )
    assert adapter.skips.get("xobject-unresolvable") == 1, (
        "and so is the resource naming an object that is not there"
    )
    assert store.ids() == [_figures(records)[0].facts["asset"]], (
        "a picture of zero length never becomes an asset"
    )


def test_k6_7_every_placement_is_read_or_counted(tmp_path):
    """The invariant: seen equals read plus declined, always."""
    path = G.FIXTURES["pdf_image_unreadable"](tmp_path / "broken.pdf")
    store = AssetStore(tmp_path / "store")
    adapter = adapter_for(path)
    adapter.skips.clear()
    records = read_path(path, store=store)
    assert len(_figures(records)) + sum(adapter.skips.values()) == 3


def test_k6_7_skip_reasons_come_from_the_closed_vocabulary(tmp_path):
    from ragix_kernels.saqqara.adapters.pdf import OBJECT_SKIPS

    seen = set()
    for name in ("pdf_image_unreadable", "pdf_inline_image"):
        path = G.FIXTURES[name](tmp_path / f"{name}.pdf")
        adapter = adapter_for(path)
        adapter.skips.clear()
        read_path(path, store=AssetStore(tmp_path / f"{name}.store"))
        seen |= set(adapter.skips)
    assert seen <= set(OBJECT_SKIPS)
    assert seen == set(OBJECT_SKIPS), "a declared reason nothing produces is a wish"
