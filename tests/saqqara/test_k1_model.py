"""
Gate K1 — the model: nodes, provenance, kinds, serialisation, signature.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K1.1-K1.8, one section per proposition, on `trees_per_format`:
trees built through the model API, one per format, each pinning its own locator
convention. No reader is involved — a claim about the model must not be able to
fail because of a file parser, nor pass because of one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

from generators import build_trees, trees_per_format  # noqa: E402

from ragix_kernels.saqqara.model import (  # noqa: E402
    CANONICAL_JSON,
    STANDARD_KINDS,
    DocumentLocator,
    KindError,
    LocatorError,
    MdLocator,
    Node,
    PdfLocator,
    Provenance,
    ProvenanceError,
    Tree,
    XlsxLocator,
    kind_registry,
)
from ragix_kernels.saqqara.views import (  # noqa: E402
    OUTLINE_CAP,
    signature_core,
    structure_signature,
)

FORMATS = ("docx", "md", "pdf", "pptx", "xlsx")


@pytest.fixture(scope="module")
def trees():
    return build_trees()


def _good_provenance() -> Provenance:
    return Provenance("f.md", "md", (MdLocator(line=1),), "test", "0.0.0")


# ---------------------------------------------------- K1.1 provenance required

def test_k1_1_node_without_provenance_is_refused():
    with pytest.raises(ProvenanceError):
        Node(kind="paragraph", provenance=None)


def test_k1_1_empty_locator_chain_is_refused():
    """A citation that points nowhere is not a citation."""
    with pytest.raises(ProvenanceError):
        Provenance("f.md", "md", (), "test", "0.0.0")


@pytest.mark.parametrize("field", ["source_path", "source_format", "kernel", "kernel_version"])
def test_k1_1_incomplete_provenance_is_refused(field):
    parts = {
        "source_path": "f.md",
        "source_format": "md",
        "chain": (MdLocator(line=1),),
        "kernel": "test",
        "kernel_version": "0.0.0",
    }
    parts[field] = ""
    with pytest.raises(ProvenanceError):
        Provenance(**parts)


def test_k1_1_chain_must_hold_locators():
    with pytest.raises(ProvenanceError):
        Provenance("f.md", "md", ({"page": 1},), "test", "0.0.0")


@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_1_every_node_of_every_fixture_is_cited(trees, fmt):
    for node in trees[fmt].walk():
        assert isinstance(node.provenance, Provenance)
        assert node.provenance.chain
        assert node.provenance.kernel and node.provenance.kernel_version
        assert node.provenance.leaf is node.provenance.chain[-1]


# ------------------------------------------------------- K1.2 stable JSON

@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_2_round_trip_is_byte_identical(trees, fmt):
    once = trees[fmt].to_json()
    twice = Tree.from_json(once).to_json()
    assert twice == once


@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_2_round_trip_preserves_locator_types(trees, fmt):
    rebuilt = Tree.from_json(trees[fmt].to_json())
    original = list(trees[fmt].walk())
    for before, after in zip(original, rebuilt.walk()):
        assert type(after.provenance.leaf) is type(before.provenance.leaf)
        assert after.provenance.leaf == before.provenance.leaf


def test_k1_2_fixture_file_round_trips(tmp_path):
    path = trees_per_format(tmp_path / "trees.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    again = {fmt: Tree.from_dict(d).to_dict() for fmt, d in payload.items()}
    assert json.dumps(again, **CANONICAL_JSON) == path.read_text(encoding="utf-8")


def test_k1_2_unknown_locator_format_is_refused():
    """Rebuilding must refuse a format it does not know, not approximate it."""
    payload = {
        "meta": {},
        "root": {
            "kind": "document",
            "origin": "read",
            "confidence": 1.0,
            "provenance": {
                "source_path": "f.zzz", "source_format": "zzz",
                "chain": [{"format": "zzz", "page": 1}],
                "kernel": "test", "kernel_version": "0.0.0",
            },
        },
    }
    with pytest.raises(LocatorError):
        Tree.from_dict(payload)


def test_k1_2_locators_order_within_a_format_and_refuse_across():
    assert PdfLocator(page=1) < PdfLocator(page=2)
    assert XlsxLocator(sheet="A", sheet_index=0, row=1, col=1) < XlsxLocator(
        sheet="A", sheet_index=0, row=2, col=1
    )
    with pytest.raises(LocatorError):
        _ = PdfLocator(page=1) < XlsxLocator(sheet="A", sheet_index=0)


# ------------------------------------------------------------ K1.3 kind registry

@pytest.fixture
def restorable_registry():
    """Register a kind, then put the process-wide registry back as it was.

    The registry is deliberately global — adapters register at import time — so a
    test that widens it must narrow it again, or it silently widens the vocabulary
    every later test runs under.
    """
    before = set(kind_registry._kinds)
    yield kind_registry
    kind_registry._kinds = before


def test_k1_3_unregistered_kind_is_refused():
    with pytest.raises(KindError):
        Node(kind="slide_note", provenance=_good_provenance())


@pytest.mark.parametrize("kind", STANDARD_KINDS)
def test_k1_3_standard_kinds_are_accepted(kind):
    assert Node(kind=kind, provenance=_good_provenance()).kind == kind


def test_k1_3_a_registered_kind_is_accepted_without_touching_the_core(restorable_registry):
    restorable_registry.register("cell_run")
    assert Node(kind="cell_run", provenance=_good_provenance()).kind == "cell_run"


def test_k1_3_registry_refuses_a_meaningless_kind(restorable_registry):
    for bad in ("", None, 3):
        with pytest.raises(KindError):
            restorable_registry.register(bad)


# --------------------------------------------------- K1.4 read versus inferred

def test_k1_4_inferred_node_may_not_claim_full_confidence():
    with pytest.raises(ValueError, match="inferred"):
        Node(kind="heading", provenance=_good_provenance(), origin="inferred", confidence=1.0)


def test_k1_4_read_node_may_not_carry_reduced_confidence():
    with pytest.raises(ValueError, match="full confidence"):
        Node(kind="heading", provenance=_good_provenance(), origin="read", confidence=0.7)


def test_k1_4_confidence_stays_within_bounds():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            Node(kind="heading", provenance=_good_provenance(), confidence=bad)


def test_k1_4_origin_vocabulary_is_closed():
    with pytest.raises(ValueError, match="origin"):
        Node(kind="heading", provenance=_good_provenance(), origin="guessed")


def test_k1_4_an_inferred_fixture_node_names_what_inferred_it(trees):
    inferred = [n for n in trees["docx"].walk() if n.origin == "inferred"]
    assert inferred, "the docx fixture must carry at least one inferred node"
    for node in inferred:
        assert node.confidence < 1.0
        assert node.provenance.kernel


def test_k1_4_origin_survives_serialisation(trees):
    rebuilt = Tree.from_json(trees["docx"].to_json())
    assert [n.origin for n in rebuilt.walk()] == [n.origin for n in trees["docx"].walk()]
    assert [n.confidence for n in rebuilt.walk()] == [n.confidence for n in trees["docx"].walk()]


# ------------------------------------------------------ K1.5 signature is stable

@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_5_signature_survives_serialisation(trees, fmt):
    assert signature_core(Tree.from_json(trees[fmt].to_json())) == signature_core(trees[fmt])


# ------------------------------------------------------- K1.6 it reconciles

@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_6_kind_totals_sum_to_the_node_count(trees, fmt):
    core = signature_core(trees[fmt])
    assert sum(core["by_kind"].values()) == core["nodes"]
    assert core["nodes"] == len(list(trees[fmt].walk()))


@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_6_heading_levels_sum_to_the_outline_total(trees, fmt):
    core = signature_core(trees[fmt])
    assert sum(core["headings_by_level"].values()) == core["outline_total"]
    assert core["outline_total"] == core["by_kind"].get("heading", 0)


def test_k1_6_the_cap_can_never_be_mistaken_for_the_count():
    """An outline longer than the cap still reports its true size."""
    prov = _good_provenance()
    root = Node(
        kind="document",
        provenance=prov,
        children=[
            Node(kind="heading", level=1, text=f"H{i}", provenance=prov)
            for i in range(OUTLINE_CAP + 7)
        ],
    )
    core = signature_core(Tree(root=root))
    assert core["outline_total"] == OUTLINE_CAP + 7
    assert len(core["outline"]) == OUTLINE_CAP
    assert core["outline_capped"] is True


# ------------------------------------------------- K1.7 invariant to the path

@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_7_core_signature_ignores_the_path(trees, fmt):
    moved = trees[fmt].replace_source_path("/somewhere/else/copy" + Path(fmt).suffix)
    assert signature_core(moved) == signature_core(trees[fmt])


@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_7_the_full_signature_still_tells_the_copies_apart(trees, fmt):
    moved = trees[fmt].replace_source_path("/somewhere/else/copy")
    before, after = structure_signature(trees[fmt]), structure_signature(moved)
    assert before["core"] == after["core"]
    assert before["source"]["path"] != after["source"]["path"]


# ------------------------------- K1.8 structured mass separated from flat mass

def test_k1_8_flat_and_structured_documents_do_not_score_alike(trees):
    ratio = {fmt: signature_core(trees[fmt])["flat_mass_ratio"] for fmt in FORMATS}
    assert ratio["md"] == 1.0, "a document with no heading is entirely flat"
    assert ratio["pptx"] == 0.0, "a document whose text all sits under headings is not flat"
    assert ratio["xlsx"] == 0.0
    assert 0.0 < ratio["pdf"] < 1.0, "a partly structured document sits between the two"
    assert ratio["md"] > ratio["pdf"] > ratio["docx"] > ratio["pptx"]


def test_k1_8_ratio_is_defined_for_a_document_without_text():
    empty = Tree(root=Node(kind="document", provenance=_good_provenance()))
    assert signature_core(empty)["flat_mass_ratio"] == 0.0


def test_k1_8_signature_is_json_serialisable(trees):
    for fmt in FORMATS:
        json.dumps(structure_signature(trees[fmt]), **CANONICAL_JSON)


# ----------------------------------------- K1.9 a locator is a unique address

# The one proposition of this gate exercised on read documents rather than built
# trees: it is a claim about the addresses a reader actually issues, and those do
# not exist until a reader has issued them.

READ_CASES = {
    "slide_deck": ".pptx",
    "mixed_workbook": ".xlsx",
    "docx_two_tier": ".docx",
    "markdown_document": ".md",
}


@pytest.fixture(scope="module")
def read_documents(tmp_path_factory):
    from generators import FIXTURES
    from ragix_kernels.saqqara.adapters import adapter_for, read_path

    root = tmp_path_factory.mktemp("k1_9")
    out = {}
    for name, suffix in READ_CASES.items():
        path = FIXTURES[name](root / f"{name}{suffix}")
        out[name] = (read_path(path), adapter_for(path), path)
    return out


@pytest.mark.parametrize("name", sorted(READ_CASES))
def test_k1_9_no_two_nodes_share_a_coordinate(read_documents, name):
    """Checked on the tree, not on the observation stream.

    Two observations may legitimately describe one position — a cell and the
    border around it — and the builder folds those into a single node. What must
    never happen is two NODES at one coordinate: that is a citation that cannot
    say which of two things it points at, which is the defect this proposition
    was written for after a slide and its speaker notes were addressed alike.
    """
    from ragix_kernels.saqqara.builder import build_tree

    records, adapter, path = read_documents[name]
    tree = build_tree(records, str(path), adapter.format, adapter.format, adapter.version).tree

    addresses = [
        tuple(sorted(node.provenance.leaf.to_dict().items())) for node in tree.walk()
    ]
    duplicates = {a for a in addresses if addresses.count(a) > 1}
    assert not duplicates, f"{len(duplicates)} coordinate(s) carried by two nodes: {duplicates}"


def test_k1_9_a_cell_and_its_border_are_one_position_and_one_node(read_documents):
    """The counter-example that keeps the proposition honest rather than merely strict."""
    records, _, _ = read_documents["mixed_workbook"]
    shared = [
        r for r in records
        if r.kind == "border"
        and any(
            o.kind == "cell" and o.locator.row == r.locator.row
            and o.locator.col == r.locator.col
            and o.locator.sheet_index == r.locator.sheet_index
            for o in records
        )
    ]
    assert shared, "the workbook fixture has bordered cells, by construction"


def test_k1_9_a_slide_and_its_notes_are_different_places(read_documents):
    records, _, _ = read_documents["slide_deck"]
    slides = [r for r in records if r.kind == "slide"]
    notes = [r for r in records if r.kind == "notes"]
    assert slides and len(notes) == len(slides)
    for slide, note in zip(slides, notes):
        assert note.locator.slide == slide.locator.slide
        assert note.locator != slide.locator
        assert note.locator.notes is True and slide.locator.notes is False


def test_k1_9_the_address_and_the_fact_are_both_kept(read_documents):
    """An address says where the text is; a fact says what it is. Not redundant."""
    for record in read_documents["slide_deck"][0]:
        if record.kind == "notes":
            assert record.locator.notes is True
            assert record.facts["on_slide"] is False


def test_k1_9_notes_sort_after_the_slides_shapes(read_documents):
    records = [r for r in read_documents["slide_deck"][0] if r.locator.slide == 1]
    ordered = sorted(records, key=lambda r: r.locator.key())
    assert [r.kind for r in ordered] == ["slide", "shape", "shape", "notes"]
