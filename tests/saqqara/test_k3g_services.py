"""
Gate K3.g — the tree services: title, pages, lookup.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K3.35-K3.39.

These three belong to no analyzer, and each is a place where a plausible answer is worse than an
honest one: a title nobody can trace, a window number presented as a page, a page restriction that
silently does nothing under one format. The tests below are written against those failures rather
than against the happy path.

K3.39 runs the same services through the real readers. That is where locator drift shows up: a
service that works on a hand-built tree and fails on a parsed one is a service that has quietly
assumed a coordinate convention no reader actually produces.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402
from generators import build_trees  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402
from ragix_kernels.saqqara.model import Node, Provenance, Tree, MdLocator  # noqa: E402
from ragix_kernels.saqqara.services import (  # noqa: E402
    PAGE_WORD_WINDOW,
    TITLE_RUNGS,
    doc_title,
    lookup,
    page_nodes,
)

FORMATS = ("docx", "md", "pdf", "pptx", "xlsx")


@pytest.fixture(scope="module")
def trees():
    return build_trees()


@pytest.fixture(scope="module")
def read_trees(tmp_path_factory):
    """The same services, over trees produced by the real readers."""
    root = tmp_path_factory.mktemp("k3g")
    out = {}
    for name, suffix in (("mixed_workbook", ".xlsx"), ("docx_label_tiling", ".docx")):
        path = G.FIXTURES[name](root / f"{name}{suffix}")
        adapter = adapter_for(path)
        out[name] = build_tree(
            read_path(path), str(path), adapter.format, adapter.format, adapter.version
        ).tree
    return out


# ------------------------------------------------------- K3.35 title cascade

def test_k3_35_the_rungs_are_declared_and_walked_in_order(trees):
    assert TITLE_RUNGS == ("metadata", "heading", "first-text")
    result = doc_title(trees["md"])
    walked = [entry["rung"] for entry in result.trace["rungs"]]
    assert walked == list(TITLE_RUNGS[: len(walked)])


def test_k3_35_metadata_outranks_content(trees):
    """The pdf fixture declares a title AND carries a heading. The declared one wins."""
    result = doc_title(trees["pdf"])
    assert result.rung == "metadata"
    assert result.title == "Paged fixture"
    heading = next(n for n in trees["pdf"].walk() if n.kind == "heading")
    assert heading.text != result.title, "the fixture must make the two differ"


def test_k3_35_a_heading_answers_when_no_title_was_declared(trees):
    result = doc_title(trees["pptx"])
    assert result.rung == "heading"
    assert result.title == "Slide title"


def test_k3_35_the_last_rung_answers_when_nothing_else_does(trees):
    result = doc_title(trees["md"])
    assert result.rung == "first-text"
    assert result.title.startswith("Text with no heading")


def test_k3_35_an_empty_document_refuses_rather_than_inventing():
    empty = Tree(
        root=Node(
            kind="document",
            provenance=Provenance("e.md", "md", (MdLocator(line=1),), "test", "0"),
        )
    )
    result = doc_title(empty)
    assert result.title is None and result.rung is None
    assert result.trace["exhausted"] is True


# --------------------------------------------------- K3.36 provenance kept

@pytest.mark.parametrize("fmt", ["pptx", "md", "xlsx"])
def test_k3_36_a_title_read_from_a_node_returns_that_node(trees, fmt):
    result = doc_title(trees[fmt])
    assert result.node is not None
    assert result.node.text == result.title
    assert result.node.provenance.chain, "a title nobody can trace back is a claim"
    assert result.node.provenance.kernel


def test_k3_36_a_declared_title_returns_no_node_and_says_so(trees):
    """Metadata came from the file's header, not from a position in the body."""
    result = doc_title(trees["pdf"])
    assert result.node is None and result.rung == "metadata"


# ------------------------------------------------------ K3.37 page policy

@pytest.mark.parametrize(
    "fmt,policy,approximate",
    [
        ("pdf", "exact-page", False),
        ("pptx", "one-based-slide", False),
        ("xlsx", "sheet-index", False),
        ("docx", "word-window", True),
        ("md", "word-window", True),
    ],
)
def test_k3_37_each_format_declares_its_policy(trees, fmt, policy, approximate):
    page_map = page_nodes(trees[fmt])
    assert page_map.policy == policy
    assert page_map.approximate is approximate


def test_k3_37_pages_are_exact_where_the_format_has_them(trees):
    page_map = page_nodes(trees["pdf"])
    assert page_map.keys == [1, 2]
    for key, nodes in page_map.pages.items():
        for node in nodes:
            assert node.provenance.leaf.page == key


def test_k3_37_slides_are_counted_from_one(trees):
    assert min(page_nodes(trees["pptx"]).keys) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_k3_37_what_has_no_coordinate_is_counted_not_dropped(trees, fmt):
    page_map = page_nodes(trees[fmt])
    placed = sum(len(nodes) for nodes in page_map.pages.values())
    assert placed + page_map.skipped == len(list(trees[fmt].walk()))
    assert page_map.skipped >= 1, "the root cites the document, not a page"


def test_k3_37_a_window_is_labelled_approximate_because_it_is(trees):
    """A window is not a page. Presenting one as the other makes a citation unfollowable."""
    page_map = page_nodes(trees["docx"])
    assert page_map.approximate is True
    assert page_map.policy == "word-window"
    assert PAGE_WORD_WINDOW > 0


def test_k3_37_windows_advance_with_the_word_count():
    prov = Provenance("long.md", "md", (MdLocator(line=1),), "test", "0")
    root = Node(
        kind="document",
        provenance=prov,
        children=[
            Node(kind="paragraph", provenance=prov, text="mot " * PAGE_WORD_WINDOW)
            for _ in range(3)
        ],
    )
    page_map = page_nodes(Tree(root=root))
    assert len(page_map.keys) == 3, "three windows' worth of words makes three windows"


def test_k3_37_an_undeclared_format_is_refused():
    prov = Provenance("x.md", "md", (MdLocator(line=1),), "test", "0")
    tree = Tree(root=Node(kind="document", provenance=prov))
    object.__setattr__(tree.root.provenance, "source_format", "zzz")
    with pytest.raises(ValueError, match="no page policy"):
        page_nodes(tree)


# ---------------------------------------------------------- K3.38 lookup

def test_k3_38_a_literal_is_matched_literally(trees):
    hits, trace = lookup(trees["pdf"], "orphan")
    assert trace["pattern_kind"] == "literal"
    assert [h.page for h in hits] == [2]


def test_k3_38_a_literal_with_regex_characters_is_not_a_regex(trees):
    """`pag.` occurs nowhere; `pag` followed by any character occurs twice."""
    assert lookup(trees["pdf"], "pag.")[0] == [], "the dot is a dot, not any character"
    assert len(lookup(trees["pdf"], "pag.", regex=True)[0]) == 2


def test_k3_38_alternatives_are_accepted(trees):
    hits, trace = lookup(trees["pdf"], ["orphan", "Body text"])
    assert trace["pattern_kind"] == "alternation"
    assert sorted(h.page for h in hits) == [1, 2]


def test_k3_38_case_can_be_ignored(trees):
    assert lookup(trees["pdf"], "ORPHAN")[0] == []
    hits, trace = lookup(trees["pdf"], "ORPHAN", ignore_case=True)
    assert len(hits) == 1 and trace["ignore_case"] is True


def test_k3_38_a_regular_expression_is_accepted(trees):
    hits, trace = lookup(trees["pdf"], r"orphan\s+paragraph", regex=True)
    assert trace["pattern_kind"] == "regex" and len(hits) == 1


def test_k3_38_a_precompiled_pattern_is_accepted(trees):
    hits, trace = lookup(trees["pdf"], re.compile("ORPHAN", re.IGNORECASE))
    assert trace["pattern_kind"] == "precompiled" and len(hits) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_k3_38_the_page_restriction_bites_under_every_policy(trees, fmt):
    """A restriction that silently did nothing would be worse than none at all."""
    page_map = page_nodes(trees[fmt])
    everything, _ = lookup(trees[fmt], "")
    assert everything, fmt

    first = page_map.keys[0]
    restricted, trace = lookup(trees[fmt], "", pages=[first])
    assert all(h.page == first for h in restricted)
    assert trace["restricted_to"] == [first]
    assert trace["policy"] == page_map.policy

    impossible, trace = lookup(trees[fmt], "", pages=["no-such-page"])
    assert impossible == []
    assert trace["excluded_by_page"] > 0, "what the restriction removed is counted"


def test_k3_38_the_trace_names_the_policy_it_used(trees):
    _, trace = lookup(trees["docx"], "paragraph")
    assert trace["policy"] == "word-window"
    assert trace["approximate_pages"] is True


# ------------------------------------------- K3.39 through the real readers

@pytest.mark.parametrize("name", ["mixed_workbook", "docx_label_tiling"])
def test_k3_39_the_services_hold_on_parsed_trees(read_trees, name):
    tree = read_trees[name]
    result = doc_title(tree)
    assert result.rung in TITLE_RUNGS
    page_map = page_nodes(tree)
    placed = sum(len(nodes) for nodes in page_map.pages.values())
    assert placed + page_map.skipped == len(list(tree.walk()))


def test_k3_39_a_parsed_spreadsheet_groups_by_its_real_sheets(read_trees):
    page_map = page_nodes(read_trees["mixed_workbook"])
    assert page_map.policy == "sheet-index"
    assert page_map.keys == [0, 1, 2], "three sheets, indexed as the reader indexed them"


def test_k3_39_lookup_finds_text_the_reader_actually_produced(read_trees):
    hits, trace = lookup(read_trees["mixed_workbook"], "Consignes")
    assert len(hits) == 1
    assert hits[0].page == 0
    assert trace["policy"] == "sheet-index"


def test_k3_39_locator_drift_would_show_here(read_trees):
    """Every node a parsed tree produces must carry the coordinate the policy reads."""
    for node in read_trees["mixed_workbook"].walk():
        if node.kind == "document":
            continue
        assert getattr(node.provenance.leaf, "sheet_index", None) is not None
