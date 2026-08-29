"""
Gate K3.k — the headings a document only ever shows.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K3.59-K3.67.

The interesting half of this analyzer is what it refuses, so most of what follows is written
against refusals: a line that is heading-sized and ends in a full stop, a line that is
heading-sized and far too long, a decorative size with a single line behind it, and a document
with no contrast at all, where the correct output is nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.analyzers import PIPELINE, pipeline  # noqa: E402
from ragix_kernels.saqqara.analyzers.format_headings import (  # noqa: E402
    CONFIDENCE,
    GAUNTLET_LIMITS,
    WEIGHT_CONFIDENCE,
    WEIGHT_LEVEL,
    FORMAT_ABSTENTIONS,
    FORMAT_CHANNEL,
    MAX_LEVELS,
    MIN_SUPPORT,
    SHAPE_RULES,
    TIER_DROPS,
    FormatHeadingsAnalyzer,
    assemble_lines,
    baseline_format_headings,
    baseline_weight_headings,
    bin_size,
    shape_refusal,
)
from ragix_kernels.saqqara.analyzers.sections import (  # noqa: E402
    DERIVED_CHANNELS,
    READER_CHANNELS,
)
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402


def _prepared(path):
    adapter = adapter_for(path)
    tree = build_tree(
        read_path(path), str(path), adapter.format, adapter.format, adapter.version
    ).tree
    return pipeline(tree)[0]


@pytest.fixture(scope="module")
def contrast(tmp_path_factory):
    path = G.FIXTURES["format_headings_pdf"](tmp_path_factory.mktemp("k3k") / "fh.pdf")
    tree = _prepared(path)
    return tree, FormatHeadingsAnalyzer().run(tree)


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    path = G.FIXTURES["no_format_contrast"](tmp_path_factory.mktemp("k3k0") / "flat.pdf")
    tree = _prepared(path)
    return tree, FormatHeadingsAnalyzer().run(tree)


def _promotions(tree):
    return [
        node
        for node in tree.walk()
        if node.facts.get("channel") == FORMAT_CHANNEL
    ]


# ------------------------------------------------- K3.59 lines, not placements

def test_k3_59_placements_are_assembled_into_lines(contrast):
    tree, result = contrast
    trace = result.trace
    assert trace["observations"] > trace["lines"], (
        "the fixture writes a line in two operations; if none collapsed, "
        "the assembly is not being exercised"
    )
    assert trace["collapsed"] == trace["observations"] - trace["lines"]


def test_k3_59_a_split_line_is_one_line_with_its_words_in_order(contrast):
    tree, _ = contrast
    lines = assemble_lines(tree)
    split = [line for line in lines if len(line.nodes) > 1]
    assert len(split) == 1, "exactly one line of the fixture is written in two operations"
    assert split[0].text == "Mesures techniques"


def test_k3_59_the_split_line_is_promoted_once_not_per_segment(contrast):
    tree, _ = contrast
    promoted = [n.text for n in _promotions(tree)]
    assert promoted.count("Mesures techniques") == 1
    assert "Mesures" not in promoted and "techniques" not in promoted


# --------------------------------------------------- K3.60 body by character mass

def test_k3_60_body_is_the_size_most_characters_are_set_in(contrast):
    _, result = contrast
    assert result.trace["body"] == 12.0


def test_k3_60_body_is_not_the_most_frequent_line(contrast):
    """The fixture has 20 body lines and 11 others; mass and count agree here.

    What must not happen is body being taken from a heading repeated often — so
    the claim is checked where it is decidable: the body size carries more
    characters than every other size put together.
    """
    tree, result = contrast
    lines = assemble_lines(tree)
    body_mass = sum(len(l.text) for l in lines if l.size == result.trace["body"])
    other_mass = sum(len(l.text) for l in lines if l.size != result.trace["body"])
    assert body_mass > other_mass


def test_k3_60_sizes_are_binned_before_they_are_compared():
    assert bin_size(15.02) == bin_size(14.98) == 15.0
    assert bin_size(15.3) == 15.5 and bin_size(15.2) == 15.0


# ------------------------------------------------------- K3.61 one visual tier

def test_k3_61_nearby_sizes_are_one_tier(contrast):
    _, result = contrast
    levels = result.trace["levels"]
    assert levels["2"] == [15.5, 15.0], "15.0 and 15.5 must be one tier, not two"
    assert set(levels) == {"1", "2"}


def test_k3_61_clustering_never_reaches_the_body(contrast):
    _, result = contrast
    for tier in result.trace["tiers"]:
        assert result.trace["body"] not in tier["sizes"]


# --------------------------------------- K3.62 title tier, ladder tiers, support

def test_k3_62_the_title_tier_is_level_one_whatever_its_support(contrast):
    _, result = contrast
    title = next(t for t in result.trace["tiers"] if t.get("level") == 1)
    assert title["sizes"] == [24.0] and title["support"] == 1


def test_k3_62_a_decorative_one_off_is_dropped_for_want_of_support(contrast):
    _, result = contrast
    refused = [t for t in result.trace["tiers"] if t.get("level") is None]
    assert [t["sizes"] for t in refused] == [[18.0]]
    assert refused[0]["reason"] == "unsupported-tier"
    assert refused[0]["support"] < MIN_SUPPORT


def test_k3_62_every_refusal_is_counted_with_a_reason_from_the_closed_list(contrast):
    _, result = contrast
    reasons = {d["reason"] for d in result.trace["drops"]}
    assert reasons <= set(TIER_DROPS) | set(SHAPE_RULES)
    assert result.trace["dropped"] == len(result.trace["drops"])


def test_k3_62_the_ladder_descends(contrast):
    _, result = contrast
    ranked = [t for t in result.trace["tiers"] if t.get("level")]
    tops = [t["sizes"][0] for t in ranked]
    assert tops == sorted(tops, reverse=True)
    assert max(t["level"] for t in ranked) <= MAX_LEVELS


# ------------------------------------------------- K3.63 the ordered shape gauntlet

@pytest.mark.parametrize(
    "text,rule",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("x" * 200, "too-long"),
        (" ".join(["mot"] * 20), "too-many-words"),
        ("Un titre qui se termine par un point.", "ends-mid-sentence"),
        ("Perimetre", None),
    ],
)
def test_k3_63_the_gauntlet_names_the_rule_that_refused(text, rule):
    assert shape_refusal(text) == rule


def test_k3_63_the_first_rule_to_refuse_is_the_one_recorded():
    """A line that breaks several rules is refused by the earliest, not the worst."""
    both = " ".join(["mot"] * 60) + "."          # too long, too many words, and punctuated
    assert shape_refusal(both) == "too-long"
    assert SHAPE_RULES.index("too-long") < SHAPE_RULES.index("too-many-words")


def test_k3_63_a_heading_sized_line_is_not_promoted_on_size_alone(contrast):
    tree, result = contrast
    promoted = {n.text for n in _promotions(tree)}
    refusals = {d["reason"] for d in result.trace["drops"] if d["reason"] in SHAPE_RULES}
    assert refusals == {"ends-mid-sentence", "too-long"}
    assert not any(t.endswith(".") for t in promoted)


# ------------------------------------------------------------ K3.64 abstention

def test_k3_64_a_document_without_contrast_promotes_nothing(flat):
    tree, result = flat
    assert _promotions(tree) == []
    assert result.trace["promoted"] == 0


def test_k3_64_the_abstention_is_an_object_with_a_closed_reason(flat):
    _, result = flat
    assert result.trace["abstained"]["reason"] in FORMAT_ABSTENTIONS
    assert result.trace["abstained"]["reason"] == "no-size-contrast"


def test_k3_64_abstaining_is_not_electing_the_body_as_a_tier(flat):
    _, result = flat
    assert result.trace["levels"] == {} and result.trace["tiers"] == []


# ----------------------------------------- K3.65 promotions add, never rewrite

def test_k3_65_promotions_are_inferred_nodes_with_their_own_provenance(contrast):
    tree, _ = contrast
    promotions = _promotions(tree)
    assert promotions, "the contrast fixture must promote something"
    for node in promotions:
        assert node.kind == "heading"
        assert node.origin == "inferred"
        assert node.confidence == CONFIDENCE < 1.0
        assert node.provenance.kernel == "saqqara.format_headings"
        assert node.facts["promoted_from"]


def _snapshot(tree):
    """Each existing node's own state, independent of what may be added beneath it.

    Keyed by identity, not by position: a promotion is inserted as a child, so a
    walk taken afterwards interleaves the new nodes among the old ones and a
    positional comparison would report every node after the first promotion as
    changed.
    """
    return [
        (id(n), n.kind, n.text, n.level, n.origin, n.confidence,
         repr(sorted(n.facts.items())))
        for n in tree.walk()
    ]


def test_k3_65_no_pre_existing_node_is_touched(tmp_path):
    """Snapshot every node's own state before and after; require it unchanged."""
    path = G.FIXTURES["format_headings_pdf"](tmp_path / "fh.pdf")
    tree = _prepared(path)

    before = _snapshot(tree)
    FormatHeadingsAnalyzer().run(tree)
    after = {row[0]: row for row in _snapshot(tree)}
    for row in before:
        assert after[row[0]] == row, "an existing node changed under a promotion"
    assert len(after) > len(before), "promotions are added, so the tree must grow"


def test_k3_65_the_promotion_cites_the_line_not_the_document(contrast):
    tree, _ = contrast
    for node in _promotions(tree):
        assert node.facts["promoted_from"].get("page") is not None


# ------------------------------------------------- K3.66 a derived channel, opt-in

def test_k3_66_the_channel_is_derived_not_one_a_reader_fills():
    assert FORMAT_CHANNEL in DERIVED_CHANNELS
    assert FORMAT_CHANNEL not in READER_CHANNELS


def test_k3_66_the_analyzer_is_opt_in_and_not_in_the_default_pipeline():
    assert FormatHeadingsAnalyzer not in PIPELINE


def test_k3_66_the_default_pipeline_promotes_nothing_by_format(tmp_path):
    path = G.FIXTURES["format_headings_pdf"](tmp_path / "fh.pdf")
    tree = _prepared(path)
    assert _promotions(tree) == []


# --------------------------------------------------- K3.67 against the baseline

def test_k3_67_measured_against_the_baseline_in_both_directions(contrast, flat):
    """The baseline is wrong in both directions, and both are reported."""
    tree, result = contrast
    base = {line.text for line in baseline_format_headings(tree)}
    ours = {node.text for node in _promotions(tree)}

    only_baseline = base - ours
    only_ours = ours - base

    # it promotes what shape refuses, and the decorative one-off nothing supports
    assert only_baseline, "the baseline must promote something we refuse"
    assert any(t.endswith(".") for t in only_baseline)
    assert any(len(t) > 120 for t in only_baseline)
    assert "Une ligne decorative isolee" in only_baseline

    # and it cannot see a line it split, because it never assembled one
    assert not only_ours or only_ours == {"Mesures techniques"}

    flat_tree, flat_result = flat
    assert baseline_format_headings(flat_tree) == []
    assert flat_result.trace["promoted"] == 0


def test_k3_67_every_comparable_fixture_lands_in_one_column_or_the_other(contrast, flat):
    """No fixture may quietly drop out of the comparison."""
    outcomes = {}
    for name, (tree, result) in (("contrast", contrast), ("flat", flat)):
        base = {line.text for line in baseline_format_headings(tree)}
        ours = {node.text for node in _promotions(tree)}
        outcomes[name] = ("differs" if base != ours else "ties", len(base), len(ours))
    assert outcomes["contrast"][0] == "differs"
    assert outcomes["flat"] == ("ties", 0, 0)
    assert set(outcomes) == {"contrast", "flat"}


# ============================================================ the weight signal

@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    path = G.FIXTURES["format_headings_docx"](tmp_path_factory.mktemp("k3kw") / "fh.docx")
    tree = _prepared(path)
    return tree, FormatHeadingsAnalyzer().run(tree)


@pytest.fixture(scope="module")
def unweighted(tmp_path_factory):
    path = G.FIXTURES["no_weight_contrast"](tmp_path_factory.mktemp("k3kw0") / "flat.docx")
    tree = _prepared(path)
    return tree, FormatHeadingsAnalyzer().run(tree)


# ------------------------------------------------- K3.68 two signals, one order

def test_k3_68_size_decides_where_a_document_ranks_its_headings(contrast):
    _, result = contrast
    assert result.trace["signal"] == "size"


def test_k3_68_weight_decides_only_where_size_found_no_contrast(weighted):
    tree, result = weighted
    assert result.trace["signal"] == "weight"
    assert result.trace["levels"] == {"2": []}, "weight carries no sizes behind its level"


def test_k3_68_weight_is_not_consulted_where_size_already_decided(contrast):
    """A document that ranked its headings must not be given a second answer."""
    tree, result = contrast
    assert result.trace["signal"] == "size"
    assert all(n.facts["signal"] == "size" for n in _promotions(tree))


def test_k3_68_the_order_is_declared_not_incidental():
    assert tuple(GAUNTLET_LIMITS) == ("size", "weight")


# ------------------------------------------------------ K3.69 weight is flat

def test_k3_69_a_weight_promotion_lands_at_one_declared_level(weighted):
    tree, _ = weighted
    promoted = _promotions(tree)
    assert promoted
    assert {n.level for n in promoted} == {WEIGHT_LEVEL}


def test_k3_69_weight_is_the_weaker_signal_and_says_so(weighted, contrast):
    tree, _ = weighted
    size_tree, _ = contrast
    assert {n.confidence for n in _promotions(tree)} == {WEIGHT_CONFIDENCE}
    assert WEIGHT_CONFIDENCE < CONFIDENCE
    assert all(n.confidence < 1.0 for n in _promotions(size_tree))


# ------------------------------------------- K3.70 dominance, not presence

def test_k3_70_a_partly_bold_paragraph_is_refused(weighted):
    tree, _ = weighted
    promoted = {n.text for n in _promotions(tree)}
    assert promoted == {
        "Perimetre de la prestation", "Gouvernance et pilotage", "Moyens techniques"
    }
    assert not any(t.startswith("Le prestataire applique") for t in promoted)


def test_k3_70_the_gauntlet_is_tighter_for_weight_than_for_size():
    """No corroboration behind a bold line, so it is held to a stricter shape."""
    assert GAUNTLET_LIMITS["weight"]["words"] < GAUNTLET_LIMITS["size"]["words"]
    assert GAUNTLET_LIMITS["weight"]["chars"] is None
    long_line = " ".join(["mot"] * 14)
    assert shape_refusal(long_line, "weight") == "too-many-words"
    assert shape_refusal(long_line, "size") is None


def test_k3_70_measured_against_the_boolean_baseline_in_both_directions(weighted):
    tree, result = weighted
    base = {line.text for line in baseline_weight_headings(tree)}
    ours = {node.text for node in _promotions(tree)}

    only_baseline = base - ours
    assert len(only_baseline) == 3, "the boolean promotes three lines the fraction refuses"
    assert any(t.startswith("Le prestataire applique") for t in only_baseline), (
        "the one-bold-word-in-twenty paragraph is the population this is about"
    )
    assert any(t.endswith(".") for t in only_baseline)
    assert not (ours - base), "everything we promote, the boolean also promotes"


def test_k3_70_a_document_with_no_weight_contrast_abstains(unweighted):
    tree, result = unweighted
    assert _promotions(tree) == []
    assert result.trace["abstained"]["reason"] == "no-weight-contrast"
    assert result.trace["abstained"]["signals"]["bold_lines"] == 0


def test_k3_70_every_bold_line_is_no_contrast_either(tmp_path):
    """The symmetric negative: a document set entirely in bold says nothing."""
    from docx import Document

    doc = Document()
    for text in ("Introduction", "Perimetre", "Gouvernance"):
        run = doc.add_paragraph().add_run(text)
        run.bold = True
    path = tmp_path / "allbold.docx"
    doc.save(path)

    tree = _prepared(path)
    result = FormatHeadingsAnalyzer().run(tree)
    assert _promotions(tree) == []
    assert result.trace["abstained"]["reason"] == "no-weight-contrast"


# ============================== K3.71 a declaration is not corroborated by a guess

@pytest.fixture(scope="module")
def declared(tmp_path_factory):
    path = G.FIXTURES["pdf_declared_outline"](tmp_path_factory.mktemp("k3k71") / "d.pdf")
    tree = _prepared(path)
    return tree, FormatHeadingsAnalyzer().run(tree)


def test_k3_71_a_declared_outline_suppresses_size_inference(declared):
    tree, result = declared
    assert [n for n in tree.walk() if n.kind == "heading" and n.origin == "read"], (
        "the fixture must declare an outline, or this proves nothing"
    )
    assert _promotions(tree) == []
    assert result.trace["promoted"] == 0


def test_k3_71_the_skip_is_named_and_counted(declared):
    """A decision not to act has to be as visible as a decision to act."""
    _, result = declared
    skip = result.trace["abstained"]
    assert skip["reason"] == "declared-outline"
    assert skip["reason"] in FORMAT_ABSTENTIONS
    assert skip["signals"]["declared_headings"] == 3
    assert skip["signals"]["would_have_promoted"] == 3, (
        "the count must be what it would have done, not zero"
    )


def test_k3_71_a_document_that_declares_nothing_is_untouched(contrast):
    """The suppression is about declarations, not about laid-out documents."""
    tree, result = contrast
    assert result.trace["abstained"] is None
    assert result.trace["promoted"] == 4
