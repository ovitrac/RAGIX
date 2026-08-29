"""
Gate K6, captions — which line, if any, describes which figure.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-28

Carries SPEC.md K6.11-K6.13.

Everything here is an inference and says so. The tests below are as interested in the figures that
get NO caption as in the ones that do: an abstention is an object with a reason, and the reasons are
not interchangeable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.saqqara.adapters import adapter_for, read_path  # noqa: E402
from ragix_kernels.saqqara.analyzers.caption_binding import (  # noqa: E402
    BINDING_FACTS,
    CAPTION_CHANNEL,
    CAPTION_RULES,
    CaptionBindingAnalyzer,
)
from ragix_kernels.saqqara.analyzers.contract import (  # noqa: E402
    CAPTION_ABSTENTIONS,
    Abstention,
)
from ragix_kernels.saqqara.assets import AssetStore  # noqa: E402
from ragix_kernels.saqqara.builder import build_tree  # noqa: E402


@pytest.fixture(scope="module")
def bound(tmp_path_factory):
    root = tmp_path_factory.mktemp("k6caption")
    out = {}
    for name in ("pdf_caption_below", "pdf_caption_ambiguous"):
        path = G.FIXTURES[name](G.fixture_path(name, root))
        adapter = adapter_for(path)
        records = read_path(path, store=AssetStore(root / f"{name}.store"))
        tree = build_tree(records, str(path), adapter.format, adapter.format,
                          adapter.version).tree
        result = CaptionBindingAnalyzer().run(tree)
        out[name] = result
    return out


def _figures(tree):
    return [n for n in tree.walk() if n.kind == "figure"]


def _captions(tree):
    return [n for n in tree.walk() if n.kind == "caption"]


def _abstention(figure) -> Abstention | None:
    return figure.facts.get("caption_abstention")


# ------------------------------------------------- K6.11 a binding is an inference

def test_k6_11_a_binding_is_inferred_never_observed(bound):
    tree = bound["pdf_caption_below"].tree
    captions = _captions(tree)
    assert captions, "the fixture binds something"
    for caption in captions:
        assert caption.origin == "inferred", "layout lies; saying so is the point"
        assert 0 < caption.confidence < 1
        assert caption.facts["channel"] == CAPTION_CHANNEL


def test_k6_11_the_binding_cites_what_it_was_read_from(bound):
    tree = bound["pdf_caption_below"].tree
    for caption in _captions(tree):
        assert set(BINDING_FACTS) <= set(caption.facts)
        assert caption.facts["caption_of"], "the figure it is about"
        assert caption.facts["captioned_by"], "the paragraph the words came from"


def test_k6_11_the_producer_is_this_analyzer_not_the_reader(bound):
    """An inference must not borrow a reader's authority."""
    tree = bound["pdf_caption_below"].tree
    for caption in _captions(tree):
        assert caption.provenance.kernel == "saqqara.caption_binding"


def test_k6_11_the_channel_is_a_derived_one(bound):
    from ragix_kernels.saqqara.analyzers.sections import DERIVED_CHANNELS

    assert CAPTION_CHANNEL in DERIVED_CHANNELS


def test_k6_11_the_figure_stays_an_observation(bound):
    """Binding a caption does not turn the picture into a guess."""
    for figure in _figures(bound["pdf_caption_below"].tree):
        assert figure.origin == "read" and figure.confidence == 1.0


# --------------------------------------------------- K6.12 ordered rules, set-level

def test_k6_12_each_rule_fires_and_names_itself(bound):
    """A declared rule nothing produces is a wish, as everywhere else here."""
    trace = bound["pdf_caption_below"].trace
    assert trace["by_rule"] == {
        "caption-below-overlapping": 1,
        "caption-below-offset": 1,
        "caption-above-overlapping": 1,
    }, trace


#: The confidences of the signed contract, written out. Comparing a binding with
#: `CAPTION_RULES` only proves the module agrees with itself: moving 0.8 to 0.85
#: left every test green, which is a control calibrated against the thing it was
#: meant to check. These numbers are the specification's, and changing one must
#: fail here before it can pass anywhere.
DECLARED_CONFIDENCE = {
    "caption-below-overlapping": 0.9,
    "caption-below-offset": 0.8,
    "caption-above-overlapping": 0.7,
}


def test_k6_12_the_confidence_comes_from_the_rule_that_fired(bound):
    assert dict(CAPTION_RULES) == DECLARED_CONFIDENCE, "the module drifted from the contract"
    for caption in _captions(bound["pdf_caption_below"].tree):
        assert caption.confidence == DECLARED_CONFIDENCE[caption.facts["rule"]]
        assert caption.facts["confidence"] == caption.confidence


def test_k6_12_the_rules_are_ordered_strongest_first(bound):
    """Below and overlapping beats below and offset beats above."""
    names = [name for name, _ in CAPTION_RULES]
    scores = [score for _, score in CAPTION_RULES]
    assert names == ["caption-below-overlapping", "caption-below-offset",
                     "caption-above-overlapping"]
    assert scores == sorted(scores, reverse=True), "order and confidence must agree"


def test_k6_12_one_line_captions_at_most_one_figure(bound):
    tree = bound["pdf_caption_ambiguous"].tree
    sources = [c.facts["captioned_by"] for c in _captions(tree)]
    assert len(sources) == len({str(s) for s in sources}), sources


# --------------------------------------------------------- K6.13 abstention objects

def test_k6_13_a_figure_with_nothing_near_it_says_so(bound):
    tree = bound["pdf_caption_below"].tree
    reasons = [_abstention(f).reason for f in _figures(tree) if _abstention(f)]
    assert reasons == ["no-candidate-within-gap"], reasons


def test_k6_13_symmetry_is_not_settled_by_rule_order(bound):
    """Equally close above and below: only the ORDER could choose, so it abstains."""
    result = bound["pdf_caption_ambiguous"]
    reasons = {_abstention(f).reason for f in _figures(result.tree) if _abstention(f)}
    assert "ambiguous-candidates" in reasons
    assert result.trace["abstained"].get("ambiguous-candidates") == 1


def test_k6_13_a_candidate_taken_by_a_nearer_figure_is_reported_as_that(bound):
    """The reason that is easy to leave out, and the one that hides a competition."""
    result = bound["pdf_caption_ambiguous"]
    assert result.trace["abstained"].get("candidate-already-bound") == 1
    assert result.trace["abstained"].get("no-candidate-within-gap") is None, (
        "a contested line is not the same as no line at all"
    )


def test_k6_13_every_reason_is_in_the_closed_vocabulary(bound):
    seen = set()
    for result in bound.values():
        seen |= set(result.trace["abstained"])
        for figure in _figures(result.tree):
            abstention = _abstention(figure)
            if abstention is not None:
                seen.add(abstention.reason)
    assert seen <= set(CAPTION_ABSTENTIONS)
    assert seen == set(CAPTION_ABSTENTIONS), "a declared reason nothing produces is a wish"


def test_k6_13_a_reason_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError):
        Abstention(reason="looked-about-right")


def test_k6_13_an_abstaining_figure_carries_no_caption(bound):
    for result in bound.values():
        for figure in _figures(result.tree):
            if _abstention(figure) is not None:
                assert not [c for c in figure.children if c.kind == "caption"]
