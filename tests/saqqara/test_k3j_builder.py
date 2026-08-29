"""
Gate K3.j — the builder: observations into a tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K3.54-K3.58.

The accounting tests are the important ones here. A builder that quietly lost one observation in a
thousand would be nearly impossible to notice downstream: the tree would simply be a little
thinner than the document, and every later measurement would be slightly wrong in the same
direction. So the three totals are reconciled against the number of observations read, and the
tree is counted independently of the trace that claims to describe it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.builder import (  # noqa: E402
    FORMAT_PLANS,
    BuildResult,
    build_tree,
)
from ragix_kernels.saqqara.model import DocumentLocator  # noqa: E402

CASES = {
    "mixed_workbook": ".xlsx",
    "docx_two_tier": ".docx",
    "pdf_outline": ".pdf",
    "slide_deck": ".pptx",
    "markdown_document": ".md",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every fixture read and built once, keyed by fixture name."""
    root = tmp_path_factory.mktemp("k3j")
    out = {}
    for name, suffix in CASES.items():
        path = G.FIXTURES[name](root / f"{name}{suffix}")
        adapter = adapter_for(path)
        observations = read_path(path)
        result = build_tree(
            observations, str(path), adapter.format, adapter.format, adapter.version
        )
        out[name] = (observations, result)
    return out


def _nodes(result: BuildResult) -> list:
    """Every node the tree actually holds, root excluded."""
    return [n for n in result.tree.walk() if n is not result.tree.root]


# ------------------------------------------------- K3.54 nothing lost or doubled

@pytest.mark.parametrize("name", sorted(CASES))
def test_k3_54_the_three_totals_reconcile(built, name):
    observations, result = built[name]
    trace = result.trace
    assert trace["observations"] == len(observations)
    assert trace["observations"] == trace["nodes"] + trace["attached"] + trace["dropped"]
    assert result.reconciles


@pytest.mark.parametrize("name", sorted(CASES))
def test_k3_54_the_tree_holds_exactly_what_the_trace_claims(built, name):
    """Counted from the tree, not from the trace that describes it."""
    _, result = built[name]
    assert len(_nodes(result)) == result.trace["nodes"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_k3_54_each_observation_is_consumed_exactly_once(built, name):
    """Counted as a multiset, not a set.

    Several observations may legitimately share a coordinate — two text blocks
    on one page, a slide and its notes. What must not happen is one observation
    becoming two nodes, and that is a question of counts, not of uniqueness.
    """
    observations, result = built[name]
    plan = FORMAT_PLANS[result.trace["format"]]
    node_kinds = set(plan.containers) | set(plan.node_kinds)

    expected = sorted(
        (o.kind, tuple(sorted(o.locator.to_dict().items())))
        for o in observations
        if o.kind in node_kinds
    )
    got = sorted(
        (_observation_kind(n, plan), tuple(sorted(n.provenance.leaf.to_dict().items())))
        for n in _nodes(result)
    )
    assert len(got) == len(expected)
    assert [locator for _, locator in got] == [locator for _, locator in expected]


def _observation_kind(node, plan) -> str:
    """The observation kind a node came from, read back through the plan."""
    for observation_kind, node_kind in {**plan.containers, **plan.node_kinds}.items():
        if node_kind == node.kind:
            return observation_kind
    return node.kind


def test_k3_54_an_unmapped_observation_is_dropped_with_its_reason(built):
    """Fail closed: an observation nobody planned for is reported, not ignored."""
    from ragix_kernels.saqqara.adapters.contract import Mastaba
    from ragix_kernels.saqqara.model import XlsxLocator

    observations, _ = built["mixed_workbook"]
    stranger = Mastaba(kind="constellation", locator=XlsxLocator(sheet="X", sheet_index=9))
    result = build_tree(
        [*observations, stranger], "x.xlsx", "xlsx", "xlsx", "0.1.0"
    )
    assert result.reconciles
    reasons = [d["reason"] for d in result.trace["drops"]]
    assert reasons.count("unmapped-observation") == 1


# --------------------------------------------- K3.55 provenance is derived

def test_k3_55_every_node_cites_a_coordinate_it_was_given(built):
    observations, result = built["mixed_workbook"]
    seen = {tuple(sorted(o.locator.to_dict().items())) for o in observations}
    for node in _nodes(result):
        assert tuple(sorted(node.provenance.leaf.to_dict().items())) in seen


def test_k3_55_the_root_cites_the_document_itself(built):
    _, result = built["mixed_workbook"]
    assert isinstance(result.tree.root.provenance.leaf, DocumentLocator)


@pytest.mark.parametrize("name", sorted(CASES))
def test_k3_55_nodes_name_the_reader_not_the_builder(built, name):
    """The builder placed the node; it did not see the document."""
    _, result = built[name]
    for node in result.tree.walk():
        assert node.provenance.kernel == result.trace["reader"]
        assert "builder" not in node.provenance.kernel


@pytest.mark.parametrize("name", sorted(CASES))
def test_k3_55_the_builder_records_itself_in_the_tree_metadata(built, name):
    _, result = built[name]
    assert result.tree.meta["builder"]["name"] == "saqqara.builder"
    assert result.tree.meta["builder"]["version"] == result.trace["builder_version"]


# ------------------------------------------------------- K3.56 borders

def test_k3_56_borders_are_attached_to_their_cell(built):
    observations, result = built["mixed_workbook"]
    borders = [o for o in observations if o.kind == "border"]
    assert borders, "the workbook fixture has bordered cells, by construction"

    with_border = [n for n in _nodes(result) if "border" in n.facts]
    assert len(with_border) == len(borders) == result.trace["attached"]
    for node in with_border:
        assert set(node.facts["border"]) == {"left", "right", "top", "bottom"}


def test_k3_56_a_border_never_becomes_a_node(built):
    _, result = built["mixed_workbook"]
    assert all(n.kind != "border" for n in _nodes(result))


def test_k3_56_a_border_matching_no_cell_is_a_counted_drop(built):
    """Remove the cell a border belongs to, and the border must be reported."""
    observations, _ = built["mixed_workbook"]
    bordered = {
        (o.locator.sheet_index, o.locator.row, o.locator.col)
        for o in observations
        if o.kind == "border"
    }
    orphaned = None
    kept = []
    for observation in observations:
        position = (
            getattr(observation.locator, "sheet_index", None),
            getattr(observation.locator, "row", None),
            getattr(observation.locator, "col", None),
        )
        if observation.kind == "cell" and position in bordered and orphaned is None:
            orphaned = position
            continue
        kept.append(observation)

    assert orphaned is not None, "the fixture must have at least one bordered cell"
    result = build_tree(kept, "x.xlsx", "xlsx", "xlsx", "0.1.0")

    drops = [d for d in result.trace["drops"] if d["reason"] == "border-matches-no-cell"]
    assert len(drops) == 1
    assert drops[0]["position"] == list(orphaned)
    assert result.reconciles, "a drop still has to reconcile"


# ----------------------------------------------------- K3.57 determinism

@pytest.mark.parametrize("name", ["mixed_workbook", "slide_deck"])
def test_k3_57_two_builds_are_byte_identical(built, name):
    observations, _ = built[name]
    fmt = _format_of(name)
    first = build_tree(observations, "p", fmt, "reader", "1.0").tree.to_json()
    second = build_tree(observations, "p", fmt, "reader", "1.0").tree.to_json()
    assert first == second


def _format_of(name: str) -> str:
    return {"mixed_workbook": "xlsx", "slide_deck": "pptx", "docx_two_tier": "docx",
            "pdf_outline": "pdf", "markdown_document": "md"}[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_k3_57_the_tree_survives_its_own_serialisation(built, name):
    from ragix_kernels.saqqara.model import Tree

    _, result = built[name]
    once = result.tree.to_json()
    assert Tree.from_json(once).to_json() == once


# ------------------------------------------------------- K3.58 one contract

def test_k3_58_every_format_is_built_through_one_entry_point(built):
    formats = {result.trace["format"] for _, result in built.values()}
    assert formats == {"xlsx", "docx", "pdf", "pptx", "md"}
    for _, result in built.values():
        assert result.tree.root.kind == "document"


def test_k3_58_what_differs_between_formats_is_declared_as_data():
    assert set(FORMAT_PLANS) == {"xlsx", "docx", "pdf", "pptx", "md"}
    for name, plan in FORMAT_PLANS.items():
        assert plan.format == name
        assert plan.containers or plan.node_kinds


def test_k3_58_an_undeclared_format_is_refused():
    with pytest.raises(ValueError, match="no build plan"):
        build_tree([], "x.zzz", "zzz", "reader", "1.0")
