"""
Gate K6 — objects: what a document shows rather than says.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-28

Carries SPEC.md K6.1-K6.8: the asset store, images read one node per placement, and the
office formats' pictures read with no renderer.

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


def test_k6_5_source_declares_only_values_something_emits(built, office):
    """Gathered from every reader that produces a figure, not asserted.

    An earlier version of this test added the office value by hand once the
    office readers existed. That turns the check into a statement of intent: it
    would have passed just as well if no reader emitted `part` at all.
    """
    from ragix_kernels.saqqara.adapters import FIGURE_SOURCES

    emitted = {
        figure.facts["source"]
        for source in (built, office)
        for _, _, records in source.values()
        for figure in _figures(records)
    }
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
    from ragix_kernels.saqqara.adapters.pdf import PLACEMENT_SKIPS

    placements = sum(n for reason, n in adapter.skips.items()
                     if reason in PLACEMENT_SKIPS)
    assert len(_figures(records)) + placements == 3


def test_k6_7_skip_reasons_come_from_the_closed_vocabulary(tmp_path):
    from ragix_kernels.saqqara.adapters.pdf import OBJECT_SKIPS

    seen = set()
    for name in ("pdf_image_unreadable", "pdf_inline_image", "pdf_form_pathologies"):
        path = G.FIXTURES[name](tmp_path / f"{name}.pdf")
        adapter = adapter_for(path)
        adapter.skips.clear()
        read_path(path, store=AssetStore(tmp_path / f"{name}.store"))
        seen |= set(adapter.skips)
    assert seen <= set(OBJECT_SKIPS)
    assert seen == set(OBJECT_SKIPS), "a declared reason nothing produces is a wish"


# ------------------------------------ K6.8 four readers, one vocabulary, no renderer

OFFICE = ("docx_embedded_image", "pptx_embedded_image", "xlsx_embedded_image")


@pytest.fixture(scope="module")
def office(tmp_path_factory):
    root = tmp_path_factory.mktemp("k6office")
    out = {}
    for name in OFFICE:
        path = G.FIXTURES[name](G.fixture_path(name, root))
        store = AssetStore(root / f"{name}.assets")
        out[name] = (path, store, read_path(path, store=store))
    return out


def test_k6_8_every_office_format_produces_a_figure(office):
    for name, (_, _, records) in office.items():
        assert len(_figures(records)) == 1, name


def test_k6_8_the_vocabulary_is_one_object_not_four_agreeing_copies(office):
    from ragix_kernels.saqqara.adapters import FIGURE_FACTS, registered_adapters

    readers = {a.format: a for a in registered_adapters().values()}
    for fmt in ("pdf", "docx", "pptx", "xlsx"):
        assert readers[fmt].fact_sets["figure"] is FIGURE_FACTS, fmt


def test_k6_8_every_reader_emits_the_whole_vocabulary(office, built):
    from ragix_kernels.saqqara.adapters import FIGURE_FACTS

    for source in (office, built):
        for name, (_, _, records) in source.items():
            for figure in _figures(records):
                assert set(figure.facts) == set(FIGURE_FACTS), name


def test_k6_8_addressing_lives_in_the_locator_not_the_facts(office):
    from ragix_kernels.saqqara.adapters import FIGURE_FACTS

    keys = {"docx_embedded_image": "relationship",
            "pptx_embedded_image": "shape_id",
            "xlsx_embedded_image": "anchor"}
    for name, key in keys.items():
        _, _, records = office[name]
        figure = _figures(records)[0]
        assert figure.locator.to_dict().get(key) is not None, name
        assert key not in FIGURE_FACTS


def test_k6_8_a_format_that_states_a_placement_reports_it(office):
    """The presentation format says where a picture landed; the others do not."""
    _, _, records = office["pptx_embedded_image"]
    facts = _figures(records)[0].facts
    assert (facts["x"], facts["y"], facts["w"], facts["h"]) == (72.0, 144.0, 96.0, 48.0)

    for name in ("docx_embedded_image", "xlsx_embedded_image"):
        _, _, other = office[name]
        silent = _figures(other)[0].facts
        assert silent["x"] is None and silent["w"] is None, (
            f"{name}: unknown is emitted as unknown, never guessed"
        )


def test_k6_8_the_same_picture_is_one_asset_across_formats(office):
    """Content addressing does not stop at the edge of a file format."""
    assets = {name: _figures(records)[0].facts["asset"]
              for name, (_, _, records) in office.items()}
    assert len(set(assets.values())) == 1, assets


def test_k6_8_no_renderer_is_imported_to_read_a_part():
    """The office path touches nothing that rasterises anything."""
    import sys

    assert "pypdfium2" not in sys.modules
    assert "fitz" not in sys.modules and "pymupdf" not in sys.modules


# The seven tests above assert on records. Records are the *reader's* contract; a
# layer that produces them and no nodes has produced nothing at all. These two
# assert on the tree, which is what K6.8 actually claims, and on the trace, which
# is where the kernel says what it refused to place.

FIGURE_NODES = {
    "docx_embedded_image": 1,
    "pptx_embedded_image": 1,
    "xlsx_embedded_image": 1,
    "pdf_image_xobject": 1,
    "pdf_image_twice": 2,
}


def _build(path, records):
    adapter = adapter_for(path)
    return build_tree(records, str(path), adapter.format, adapter.format,
                      adapter.version)


def test_k6_8_a_figure_record_becomes_a_figure_node(office, built):
    """A record every reader emits, and every builder plan places."""
    both = {**office, **built}
    for name, expected in FIGURE_NODES.items():
        path, _, records = both[name]
        result = _build(path, records)
        nodes = [n for n in result.tree.walk() if n.kind == "figure"]
        assert len(_figures(records)) == expected, f"{name}: records"
        assert len(nodes) == expected, f"{name}: records that never reached the tree"
        assert result.reconciles, f"{name}: {result.trace}"


def test_k6_8_no_figure_is_dropped_by_the_builder(office, built):
    """The drop the record-level tests could not see, and the kernel had counted."""
    both = {**office, **built}
    for name in FIGURE_NODES:
        path, _, records = both[name]
        dropped = [d for d in _build(path, records).trace["drops"]
                   if d.get("kind") == "figure"]
        assert dropped == [], f"{name}: {dropped}"


# ------------------------- K6.9 a picture that will not parse is not a lost document

DAMAGED = ("docx_image_unreadable", "pptx_image_unreadable")


@pytest.fixture(scope="module")
def damaged(tmp_path_factory):
    root = tmp_path_factory.mktemp("k6damaged")
    out = {}
    for name in DAMAGED:
        path = G.FIXTURES[name](G.fixture_path(name, root))
        adapter = adapter_for(path)
        adapter.skips.clear()
        store = AssetStore(root / f"{name}.assets")
        out[name] = (path, store, read_path(path, store=store), dict(adapter.skips))
    return out


def test_k6_9_one_unparseable_picture_does_not_cost_the_document(damaged):
    """The regression this proposition exists for.

    `part.image` is a property that parses; a default on `getattr` does not
    protect against an exception raised inside one. The reader let that escape,
    and one word-processing document out of 322 stopped being readable at all --
    every paragraph in it lost to one picture whose header would not parse.
    """
    _, _, records, _ = damaged["docx_image_unreadable"]
    assert [r for r in records if r.kind == "paragraph"], "the text is still read"


def test_k6_9_the_bytes_are_kept_and_only_the_size_is_unknown(damaged):
    """Both readers reach the bytes; neither can describe them."""
    for name in DAMAGED:
        _, store, records, _ = damaged[name]
        figures = _figures(records)
        assert len(figures) == 1, f"{name}: the reachable picture is emitted"
        facts = figures[0].facts
        assert facts["width"] is None and facts["height"] is None, (
            f"{name}: unknown, never guessed"
        )
        assert facts["asset"] in store.ids(), f"{name}: the bytes are kept even so"
        assert len(store.read(facts["asset"])) == 32


def test_k6_9_a_placement_survives_a_picture_it_cannot_describe(damaged):
    """What the format states is not what the header says, and does not fall with it."""
    _, _, records, _ = damaged["pptx_image_unreadable"]
    facts = _figures(records)[0].facts
    assert (facts["x"], facts["y"]) == (72.0, 144.0), "the slide still says where"


def test_k6_9_what_cannot_be_read_at_all_is_counted(damaged):
    """Two formats, two shapes, one vocabulary."""
    _, _, _, docx_skips = damaged["docx_image_unreadable"]
    assert docx_skips.get("image-part-empty") == 1, "a part present but empty"

    _, _, _, pptx_skips = damaged["pptx_image_unreadable"]
    assert pptx_skips.get("image-part-unreadable") == 1, "a part that is not there"


def test_k6_9_office_skip_reasons_come_from_one_shared_vocabulary(damaged):
    from ragix_kernels.saqqara.adapters import PART_SKIPS, registered_adapters

    readers = {a.format: a for a in registered_adapters().values()}
    for fmt in ("docx", "pptx", "xlsx"):
        assert readers[fmt].skip_reasons is PART_SKIPS, fmt

    seen = set()
    for _, _, _, skips in damaged.values():
        seen |= set(skips)
    assert seen == set(PART_SKIPS), "a declared reason nothing produces is a wish"


def test_k6_9_every_picture_is_read_or_counted(damaged):
    """The invariant of K6.7, carried into the office formats."""
    held = {"docx_image_unreadable": 2, "pptx_image_unreadable": 2}
    for name, total in held.items():
        _, _, records, skips = damaged[name]
        assert len(_figures(records)) + sum(skips.values()) == total, name


# --------------------------- K6.10 a picture is a placement wherever it is drawn

@pytest.fixture(scope="module")
def nested(tmp_path_factory):
    root = tmp_path_factory.mktemp("k6forms")
    out = {}
    for name in ("pdf_image_in_form", "pdf_form_pathologies"):
        path = G.FIXTURES[name](G.fixture_path(name, root))
        adapter = adapter_for(path)
        adapter.skips.clear()
        store = AssetStore(root / f"{name}.assets")
        out[name] = (path, store, read_path(path, store=store), dict(adapter.skips))
    return out


def test_k6_10_an_image_drawn_only_inside_a_form_is_still_a_placement(nested):
    """Four corpus documents stored images, drew them, and produced nothing."""
    from ragix_kernels.saqqara.adapters.pdf import PLACEMENT_SKIPS

    _, _, records, skips = nested["pdf_image_in_form"]
    assert len(_figures(records)) == 2, "one placement per drawing, at any depth"
    assert not set(skips) & set(PLACEMENT_SKIPS), "no placement was declined here"


def test_k6_10_the_box_composes_through_every_transformation(nested):
    """The page's matrix, the form's own /Matrix, and the one around the image."""
    _, _, records, _ = nested["pdf_image_in_form"]
    boxes = sorted((f.facts["x"], f.facts["y"], f.facts["w"], f.facts["h"])
                   for f in _figures(records))
    assert boxes == [(10.0, 20.0, 100.0, 50.0), (220.0, 430.0, 60.0, 30.0)], boxes


def test_k6_10_one_image_two_depths_is_one_asset(nested):
    _, store, records, _ = nested["pdf_image_in_form"]
    assert len({f.facts["asset"] for f in _figures(records)}) == 1
    assert len(store.ids()) == 1
    assert len(store.manifest()[_figures(records)[0].facts["asset"]]["references"]) == 2


def test_k6_10_a_descent_that_will_not_terminate_is_counted(nested):
    """A bound that stops in silence is the same lost placement in a new place."""
    from ragix_kernels.saqqara.adapters.pdf import MAX_FORM_DEPTH

    _, _, records, skips = nested["pdf_form_pathologies"]
    assert skips.get("form-cycle") == 1, "a form that invokes itself"
    assert skips.get("form-too-deep") == 1, "a chain deeper than the limit"
    assert _figures(records) == [], "the image below the limit is not reached"
    assert G.FORM_CHAIN > MAX_FORM_DEPTH, "the fixture must exceed the limit it tests"


def test_k6_10_the_limit_is_declared_not_buried(nested):
    from ragix_kernels.saqqara.adapters.pdf import MAX_FORM_DEPTH, OBJECT_SKIPS

    assert isinstance(MAX_FORM_DEPTH, int) and MAX_FORM_DEPTH > 0
    assert {"form-cycle", "form-too-deep"} <= set(OBJECT_SKIPS)


# ============== K6.20 — the page a Word node is on, read rather than rendered

def _docx_pages(tmp_path, fixture: str, name: str = "d.docx"):
    """The derived block of every body record a fixture emits."""
    path = G.FIXTURES[fixture](tmp_path / name)
    return [(o.text, o.facts.get("derived"))
            for o in read_path(path) if o.kind in ("paragraph", "marker")]


def test_k6_20_pages_are_read_from_the_documents_own_marks(tmp_path):
    """Word records where a page ended when it last rendered; a break is a fact.

    Falsified by: a body flow broken across pages whose records all claim one page,
    or pages that do not follow the marks in document order.
    """
    seen = _docx_pages(tmp_path, "docx_paged")
    pages = [block["page"] for _text, block in seen if block]

    assert pages == [1, 2, 3], f"the marks were not counted in document order: {pages}"
    assert all(block["source"] == "rendered-and-explicit-marks" for _t, block in seen if block)


def test_k6_20_a_break_in_a_paragraph_the_reader_skips_is_still_counted(tmp_path):
    """The trap: an explicit break lives in a paragraph of its own, with no text.

    The reader skips a paragraph carrying neither text nor a marker, so counting
    over emitted records instead of over body children loses exactly the breaks
    that matter — every record would claim page 1.

    Falsified by: a page map built from what the reader emitted.
    """
    from docx import Document
    from ragix_kernels.saqqara.adapters.docx import page_map

    path = G.FIXTURES["docx_paged"](tmp_path / "p.docx")
    document = Document(path)
    emitted = [o for o in read_path(path) if o.kind in ("paragraph", "marker")]
    body_children = list(document.element.body.iterchildren())

    assert len(emitted) < len(body_children), "the fixture no longer skips anything"
    mapping = page_map(document, declared=None)
    assert mapping["derived_pages"] == 3


def test_k6_20_no_page_information_yields_no_page_at_all(tmp_path):
    """Never page 1 by default: an unknown page is unknown.

    Falsified by: a document with no marks and no declared count whose records
    carry a page anyway.
    """
    from docx import Document
    from ragix_kernels.saqqara.adapters.docx import page_map

    path = G.FIXTURES["docx_two_tier"](tmp_path / "t.docx")
    mapping = page_map(Document(path), declared=None)

    assert mapping["pages"] == {} and mapping["source"] is None
    assert mapping["derived_pages"] == 0


def test_k6_20_a_single_declared_page_is_a_fact_and_names_its_source(tmp_path):
    """One page declared and no marks is not an absence of information.

    Falsified by: a one-page document whose records carry no page, or one whose
    page does not say where it came from.
    """
    from docx import Document
    from ragix_kernels.saqqara.adapters.docx import page_map

    path = G.FIXTURES["docx_markers"](tmp_path / "m.docx")
    mapping = page_map(Document(path), declared=1)

    assert mapping["source"] == "declared-single-page"
    assert mapping["pages"] and all(v == [1] for v in mapping["pages"].values())
    assert mapping["consistent"] is True


def test_k6_20_the_declared_count_is_recorded_and_checked_never_trusted(tmp_path):
    """python-docx declares one page whatever the content, because it never renders.

    So a disagreement between the declared count and the marks is a fact to report,
    not a number to clamp to.

    Falsified by: a mapping that silently follows the declared count, or one that
    hides the disagreement.
    """
    seen = _docx_pages(tmp_path, "docx_paged")
    blocks = [block for _t, block in seen if block]

    assert blocks and all(b["declared_pages"] == 1 for b in blocks)
    assert all(b["derived_pages"] == 3 for b in blocks)
    assert all(b["consistent"] is False for b in blocks), \
        "the disagreement between the marks and the declared count is not reported"


def test_k6_20_the_page_is_derived_provenance_and_stays_out_of_the_chain(tmp_path):
    """The chain is what the reader read; a page is what was worked out from it.

    A page in the chain would make the tree's own root depend on how the page was
    obtained — and under the renderer fallback, on which machine ran it.

    Falsified by: a page or its source appearing in a node's provenance chain.
    """
    from ragix_kernels.saqqara.adapters.contract import adapter_for
    from ragix_kernels.saqqara.builder import build_tree

    path = G.FIXTURES["docx_paged"](tmp_path / "c.docx")
    adapter = adapter_for(path)
    tree = build_tree(read_path(path), str(path), adapter.format,
                      adapter.format, adapter.version).tree

    with_page = [n for n in tree.walk() if (n.facts or {}).get("derived")]
    assert with_page, "no node carries the derived page"
    for node in tree.walk():
        # The chain's own keys, not a substring of the serialised record: the
        # first version of this assertion matched the word "page" inside the
        # temporary file's path and would have passed on anything.
        for link in node.provenance.to_dict()["chain"]:
            assert "page" not in link, f"a page reached the provenance chain: {link}"
            assert "derived" not in link
