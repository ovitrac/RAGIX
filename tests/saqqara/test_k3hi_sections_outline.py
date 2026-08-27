"""
Gate K3.h and K3.i — sections, and the typed outline.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K3.40-K3.53.

Both layers exist to refuse things, so most of what follows is written against the refusals: a
running header that looks exactly like a section, a printed contents page that would yield a
complete and entirely duplicated outline, a numbered list that is a list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara")) 

import generators as G  # noqa: E402
from generators import build_outline_trees, build_trees  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.analyzers import pipeline  # noqa: E402
from ragix_kernels.saqqara.analyzers.outline import (  # noqa: E402
    MIN_CHAIN,
    OUTLINE_CHANNEL,
    OUTLINE_DROPS,
    OutlineAnalyzer,
    parse_label,
    validate_walk,
)
from ragix_kernels.saqqara.analyzers.sections import (  # noqa: E402
    CHANNELS,
    DERIVED_CHANNELS,
    READER_CHANNELS,
    FURNITURE_PAGES,
    GAUNTLET,
    REJECTION_REASONS,
    SectionsAnalyzer,
    baseline_sections,
)
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402

SUFFIXES = (".pdf", ".docx", ".xlsx", ".pptx")


def _prepared(path):
    adapter = adapter_for(path)
    tree = build_tree(
        read_path(path), str(path), adapter.format, adapter.format, adapter.version
    ).tree
    return pipeline(tree)[0]


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    """The four documents of the multi-channel fixture, collected."""
    manifest = G.FIXTURES["sections_multi_channel"](
        tmp_path_factory.mktemp("k3h") / "sec.json"
    )
    out = {}
    for suffix in SUFFIXES:
        tree = _prepared(manifest.with_suffix(suffix))
        out[suffix] = (tree, SectionsAnalyzer().run(tree))
    return out


@pytest.fixture(scope="module")
def furniture(tmp_path_factory):
    tree = _prepared(G.FIXTURES["running_headers"](tmp_path_factory.mktemp("k3h2") / "rh.pdf"))
    return tree, SectionsAnalyzer().run(tree)


def _channels(multi):
    found = {}
    for _, result in multi.values():
        found.update(result.trace["channels"])
    return found


# ------------------------------------------------- K3.40 every record is routed

def test_k3_40_every_channel_is_in_the_closed_list(multi):
    for _, result in multi.values():
        for channel in result.trace["channels"]:
            assert channel in CHANNELS


def test_k3_40_all_eight_reader_channels_fire(multi):
    """Every reader channel, and only reader channels: no analyzer has run yet."""
    assert set(_channels(multi)) == set(READER_CHANNELS), "a channel that never fires is untested"
    assert not set(_channels(multi)) & set(DERIVED_CHANNELS)


def test_k3_40_the_trace_carries_the_route(multi):
    tree, result = multi[".pdf"]
    assert result.trace["gauntlet"] == list(GAUNTLET)
    assert result.trace["format"] == "pdf"


# ------------------------------------------------------ K3.41 blind documents

def test_k3_41_a_format_with_no_channel_is_reported_not_ignored():
    tree = build_trees()["md"]
    result = SectionsAnalyzer().run(tree)
    assert result.trace["blind"]["reason"] == "no-channel-for-format"
    assert result.trace["accepted"] == 0


def test_k3_41_a_document_with_no_candidate_says_so(tmp_path):
    tree = _prepared(G.FIXTURES["pdf_no_text_layer"](tmp_path / "scan.pdf"))
    result = SectionsAnalyzer().run(tree)
    assert result.trace["blind"]["reason"] == "no-candidate-found"


# -------------------------------------------------- K3.42 the split heading

def test_k3_42_a_number_alone_joins_the_line_beneath_it(multi):
    """`2.` then `Moyens engages` is one heading typeset across two lines."""
    names = _channels(multi)["pdf-numbered-heading"]
    assert "Moyens engages" in names


def test_k3_42_the_join_is_recorded(multi):
    _, result = multi[".pdf"]
    assert result.trace["accepted"] >= 2


def test_k3_42_a_number_with_nothing_beneath_it_does_not_join(multi):
    """A trailing bare number is a page number, not a label with a missing title."""
    _, result = multi[".pdf"]
    assert result.trace["rejected"]["orphan-number"] == G.EXPECTED_SECTIONS[
        "refused_orphan_number"
    ]


# ------------------------------------------ K3.43 printed contents refused

def test_k3_43_a_contents_page_is_refused_under_its_own_reason(multi):
    _, result = multi[".pdf"]
    assert result.trace["rejected"]["printed-contents"] == G.EXPECTED_SECTIONS[
        "refused_printed_contents"
    ]
    tests = {r["test"] for r in result.trace["rejections"] if r["reason"] == "printed-contents"}
    assert tests == {"G1-printed-contents"}


def test_k3_43_no_contents_line_survives_into_the_sections(multi):
    for names in _channels(multi).values():
        for name in names:
            assert "...." not in name, "a dot-leader line is a contents entry, not a section"


# --------------------------------------------------- K3.44 page furniture

def test_k3_44_a_title_on_three_or_more_pages_is_furniture(furniture):
    _, result = furniture
    assert result.trace["rejected"]["page-furniture"] == G.RUNNING_HEADER_PAGES
    entry = next(r for r in result.trace["rejections"] if r["reason"] == "page-furniture")
    assert len(entry["pages"]) >= FURNITURE_PAGES


def test_k3_44_the_real_sections_survive_it(furniture):
    _, result = furniture
    assert result.trace["channels"]["pdf-numbered-heading"] == G.RUNNING_HEADER_SECTIONS


def test_k3_44_furniture_never_enters_an_ancestry(furniture):
    tree, _ = furniture
    for node in tree.walk():
        assert G.RUNNING_HEADER not in (node.facts.get("section_ancestry") or [])


# ------------------------------------------------------- K3.45 the gauntlet

def test_k3_45_the_tests_run_in_a_declared_order(multi):
    _, result = multi[".pdf"]
    assert result.trace["gauntlet"] == ["G1-printed-contents", "G2-orphan-number",
                                        "G3-page-furniture"]


def test_k3_45_every_rejection_names_a_reason_from_the_closed_list(multi, furniture):
    for _, result in [*multi.values(), furniture]:
        for entry in result.trace["rejections"]:
            assert entry["reason"] in REJECTION_REASONS
            assert entry["test"] in GAUNTLET
        assert set(result.trace["rejected"]) == set(REJECTION_REASONS)


def test_k3_45_nothing_is_both_accepted_and_rejected(multi):
    for _, result in multi.values():
        rejected = {r["name"] for r in result.trace["rejections"]}
        accepted = {n for names in result.trace["channels"].values() for n in names}
        assert not (rejected & accepted)


def test_k3_45_the_counts_reconcile(multi, furniture):
    for _, result in [*multi.values(), furniture]:
        total = result.trace["accepted"] + sum(result.trace["rejected"].values())
        assert total == result.trace["candidates"]


# -------------------------------------------------- K3.46 one name per channel

@pytest.mark.parametrize("channel", sorted(G.EXPECTED_SECTIONS))
def test_k3_46_each_channel_finds_what_it_was_built_to_find(multi, channel):
    if channel.startswith("refused"):
        pytest.skip("counted as a refusal, asserted in K3.43 and K3.42")
    assert _channels(multi)[channel] == G.EXPECTED_SECTIONS[channel]


def test_k3_46_no_two_channels_report_under_one_name(multi):
    for _, result in multi.values():
        assert len(result.trace["channels"]) == len(set(result.trace["channels"]))


# ---------------------------------------------------------- K3.47 ancestry

def test_k3_47_ancestry_is_broadest_first(multi):
    tree, _ = multi[".docx"]
    chains = [
        node.facts["section_ancestry"] for node in tree.walk()
        if node.facts.get("section_ancestry")
    ]
    assert chains
    deep = max(chains, key=len)
    assert deep[0] == "Titre principal", "the outermost section comes first"


def test_k3_47_a_node_under_no_section_carries_an_empty_chain(multi):
    tree, _ = multi[".docx"]
    assert tree.root.facts["section_ancestry"] == []


def test_k3_47_every_node_carries_a_chain(multi):
    for suffix in SUFFIXES:
        tree, _ = multi[suffix]
        for node in tree.walk():
            assert isinstance(node.facts.get("section_ancestry"), list)


# ---------------------------------------------------------- K3.48 baseline

def test_k3_48_the_baseline_swallows_what_the_gauntlet_refuses(furniture):
    """Measured: the naive reading takes the running header on every page."""
    tree, result = furniture
    naive = baseline_sections(tree)
    assert len(naive) > result.trace["accepted"]
    assert sum(1 for name in naive if G.RUNNING_HEADER in name) == G.RUNNING_HEADER_PAGES


def test_k3_48_the_baseline_misses_what_no_style_records(multi):
    """And in the other direction: numbering held as a property is invisible to it."""
    tree, result = multi[".docx"]
    naive = baseline_sections(tree)
    assert result.trace["accepted"] > len(naive)
    assert "Portee du document" not in naive


def test_k3_48_the_comparison_is_reported_not_asserted(multi, furniture):
    for _, result in [*multi.values(), furniture]:
        baseline = result.trace["baseline"]
        assert "rule" in baseline and "found" in baseline and "ours" in baseline


# ============================================================ K3.i — outline

@pytest.fixture(scope="module")
def outlines():
    return {
        case: OutlineAnalyzer().run(tree)
        for case, tree in build_outline_trees().items()
    }


def test_k3_49_labels_parse_by_kind_and_value():
    assert parse_label("1.2.3 Titre")[:2] == ("num", (1, 2, 3))
    assert parse_label("2) Portee")[:2] == ("num", (2,))
    assert parse_label("B. Annexe")[:2] == ("let", (2,))
    assert parse_label("IV. Moyens")[:2] == ("rom", (4,))
    assert parse_label("I. Contexte")[:2] == ("rom", (1,)), "roman wins over letter"
    assert parse_label("du texte normal") is None


def test_k3_49_only_a_legal_step_promotes():
    walk = [(1,), (1, 1), (1, 2), (2,), (2, 1), (3,)]
    assert validate_walk(walk) == list(range(len(walk)))
    broken = [(1,), (1, 1), (7,), (1, 2), (2,)]
    assert 2 not in validate_walk(broken), "the illegal jump is skipped"


def test_k3_49_a_legal_walk_promotes_whole(outlines):
    trace = outlines["legal_walk"].trace
    assert trace["promoted"] == G.EXPECTED_OUTLINE["legal_walk"]["promoted"]
    assert trace["walks"] == 1 and trace["dropped"] == 0


def test_k3_50_an_enumeration_stays_paragraphs(outlines):
    trace = outlines["enumeration"].trace
    assert trace["promoted"] == 0
    assert {d["reason"] for d in trace["drops"]} == {"flat-uncorroborated"}


def test_k3_50_the_same_shape_promotes_when_style_corroborates(outlines):
    """One rule, both sides: numbering alone is not evidence, numbering plus style is."""
    trace = outlines["flat_corroborated"].trace
    assert trace["promoted"] == G.EXPECTED_OUTLINE["flat_corroborated"]["promoted"]
    assert trace["dropped"] == 0


def test_k3_51_a_chain_too_short_to_be_evidence_abstains(outlines):
    trace = outlines["short_chain"].trace
    assert trace["promoted"] == 0
    assert {d["reason"] for d in trace["drops"]} == {"chain-too-short"}
    assert MIN_CHAIN >= 3


def test_k3_52_an_annex_restart_opens_a_second_walk(outlines):
    trace = outlines["annex_restart"].trace
    assert trace["walks"] == 2
    assert trace["promoted"] == G.EXPECTED_OUTLINE["annex_restart"]["promoted"]
    assert trace["dropped"] == 0


def test_k3_52_a_stray_label_never_captures_the_walk(outlines):
    trace = outlines["stray_restart"].trace
    assert trace["walks"] == 1
    assert trace["promoted"] == G.EXPECTED_OUTLINE["stray_restart"]["promoted"]


def test_k3_53_every_drop_is_counted_under_a_closed_reason(outlines):
    for case, result in outlines.items():
        trace = result.trace
        assert trace["dropped"] == len(trace["drops"])
        for drop in trace["drops"]:
            assert drop["reason"] in OUTLINE_DROPS
            assert drop["label"] and drop["step"]


def test_k3_53_promoted_and_dropped_account_for_every_candidate(outlines):
    for case, result in outlines.items():
        trace = result.trace
        assert trace["promoted"] + trace["dropped"] == trace["candidates"], case


def test_k3_53_a_promoted_node_says_it_was_inferred(outlines):
    for node in outlines["legal_walk"].tree.walk():
        if node.kind == "heading":
            assert node.origin == "inferred" and node.confidence < 1.0
            assert node.facts["outline_step"]


# ------------------------------------- promotion adds; it never rewrites

def _snapshot(tree):
    """Each existing node's own state, independent of what may be added beneath it."""
    return [
        (id(n), n.kind, n.text, n.level, n.origin, n.confidence, repr(sorted(n.facts.items())))
        for n in tree.walk()
    ]


def test_outline_never_mutates_an_existing_node():
    """An observation must never be silently replaced by an inference."""
    tree = build_outline_trees()["legal_walk"]
    before = _snapshot(tree)
    OutlineAnalyzer().run(tree)
    after = {row[0]: row for row in _snapshot(tree)}
    for row in before:
        assert after[row[0]] == row, "an existing node changed under a promotion"


def test_outline_adds_nodes_that_name_it_as_their_producer():
    tree = build_outline_trees()["legal_walk"]
    OutlineAnalyzer().run(tree)
    promoted = [n for n in tree.walk() if n.facts.get("channel") == OUTLINE_CHANNEL]
    assert len(promoted) == G.EXPECTED_OUTLINE["legal_walk"]["promoted"]
    for node in promoted:
        assert node.provenance.kernel == "saqqara.outline"
        assert node.origin == "inferred" and node.confidence < 1.0
        assert node.facts["promoted_from"], "a promotion cites the position it came from"


def test_outline_leaves_the_paragraph_it_promoted_intact():
    tree = build_outline_trees()["legal_walk"]
    OutlineAnalyzer().run(tree)
    paragraphs = [n for n in tree.walk() if n.kind == "paragraph"]
    assert len(paragraphs) == len(G.OUTLINE_CASES["legal_walk"])
    for node in paragraphs:
        assert node.origin == "read" and node.confidence == 1.0
        assert node.text.strip()[0].isdigit(), "the label is still in the text a reader saw"


def test_a_second_sections_pass_routes_promotions_under_their_own_channel():
    """The caller asked for promotions in the ancestry; they arrive labelled as inferences."""
    tree = build_outline_trees()["legal_walk"]
    first = SectionsAnalyzer().run(tree)
    assert first.trace["blind"]["reason"] == "no-channel-for-format" or not first.trace["accepted"]

    OutlineAnalyzer().run(tree)
    second = SectionsAnalyzer().run(tree)
    assert second.trace["channels"].get(OUTLINE_CHANNEL)
    assert not set(second.trace["channels"]) & set(READER_CHANNELS)
