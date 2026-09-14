"""The form's numbers per grain.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A window and a roll-up are not the same object and do not owe the same prose: a summary floor of
30 content words at window grain and 80 at roll-up grain; the hollow rule at four markers a
sentence at both, with six content words outside them at window grain and eight at roll-up. Each
number is tested at its edge — one below refused, the number itself accepted.
"""
from __future__ import annotations

import json

import pytest

from ragix_kernels.harvest import form
from ragix_kernels.harvest.form import HarvestRefusal, marker, validate

IDS = [f"v{i}" for i in range(1, 10)]


def _payload(sentences):
    return json.dumps({"node_id": "n1", "summary": sentences,
                       "summary_map": [{"sentence": i + 1, "children": [f"v{i + 1}"] if marker("claim", f"v{i + 1}") in s else []}
                                       for i, s in enumerate(sentences)],
                       "values": [], "entities": [],
                       "interpreted": {"relevance": "informative", "act": "informative"}}, ensure_ascii=False)


def _check(sentences, grain):
    return validate(_payload(sentences), node_id="n1", allowed_children=["n1#leaf"], allowed_claims=IDS,
                    value_ids=IDS, **form.grain_rules(grain))


def _prose(n):
    """n content words, no digits, ending as a sentence."""
    return " ".join(["mot"] * (n - 1)) + " final."


def test_the_numbers_per_grain():
    assert form.grain_rules("window") == {"min_words": 30, "max_refs": 4, "min_sentence_words": 6}
    assert form.grain_rules("rollup") == {"min_words": 80, "max_refs": 4, "min_sentence_words": 8}


@pytest.mark.parametrize("grain, floor", [("window", 30), ("rollup", 80)])
def test_the_summary_floor_at_its_edge(grain, floor):
    with pytest.raises(HarvestRefusal):
        _check([_prose(floor - 1)], grain)
    assert _check([_prose(floor)], grain)


@pytest.mark.parametrize("grain, words", [("window", 6), ("rollup", 8)])
def test_the_hollow_rule_at_its_edge(grain, words):
    pad = _prose(90)                                      # keeps the summary above either floor
    cited = lambda n: " ".join(["mot"] * (n - 1)) + f" {marker('claim', 'v1')} final."   # noqa: E731
    with pytest.raises(HarvestRefusal) as caught:
        _check([cited(words - 1), pad], grain)
    assert caught.value.reason == "hollow sentence"
    assert _check([cited(words), pad], grain)


def test_an_unknown_grain_is_refused():
    with pytest.raises(HarvestRefusal):
        form.grain_rules("chapter")
