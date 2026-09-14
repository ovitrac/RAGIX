"""A sentence that is a string of markers is not a summary.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A pilot once opened its summary with one sentence carrying eighty-five consecutive markers — it
cited everything, committed to nothing, and passed the contract. The form rewarded it, which is the
form's defect and not the model's. Form 0.6 declares the rule: at most four markers in a sentence,
and at least eight content words outside them; re-declared since so that a true ladder of
citations passes. The sentences below are invented, with the same shapes.
"""
from __future__ import annotations

import json
import re

import pytest

from ragix_kernels.harvest.form import (MAX_REFS_PER_SENTENCE, MIN_SENTENCE_WORDS, HarvestRefusal,
                                        grain_rules, marker, validate)

PILOT_SENTENCE_1 = ("Le cahier des charges techniques décrit les opérations de maintenance préventive "
                    "pour divers appareils, incluant des visites régulières à "
                    + ", ".join(marker("claim", f"v{i}") for i in range(9, 94)) + " heures.")
GOOD = ("Les visites de maintenance préventive des pompes de relevage comprennent des opérations "
        "spécifiques telles que la vidange des cuves, le remplacement des clapets usés, et le "
        "contrôle de l'étanchéité du réseau de refoulement.")
IDS = [f"v{i}" for i in range(1, 94)]


def _payload(sentences: list[str]) -> str:
    return json.dumps({
        "node_id": "n1", "summary": sentences,
        "summary_map": [{"sentence": i + 1,
                         "children": re.findall(r"\{\{claim:([A-Za-z0-9_.:-]{1,64})\}\}", s) or []}
                        for i, s in enumerate(sentences)],
        "values": [], "entities": [],
        "interpreted": {"relevance": "informative", "act": "informative"}}, ensure_ascii=False)


def _validate(sentences: list[str]):
    return validate(_payload(sentences), node_id="n1", allowed_children=["n1#leaf"],
                    allowed_claims=IDS, value_ids=IDS, min_words=0, max_refs=MAX_REFS_PER_SENTENCE,
                    min_sentence_words=MIN_SENTENCE_WORDS)


def test_the_pilot_sentence_is_refused():
    with pytest.raises(HarvestRefusal) as caught:
        _validate([PILOT_SENTENCE_1, GOOD, GOOD])
    assert caught.value.reason == "hollow sentence"
    assert "1" in str(caught.value.detail)


def test_a_sentence_of_markers_alone_is_refused():
    with pytest.raises(HarvestRefusal) as caught:
        _validate([" ".join(marker("claim", f"v{i}") for i in range(1, 4)), GOOD, GOOD])
    assert caught.value.reason == "hollow sentence"


def test_a_legitimate_ladder_of_citations_is_admitted():
    """Counting markers alone refused a true sentence. A visit ladder cites every rung — twelve markers
    carried by fifteen content words — and it is prose that cites, not a list of citations. Hollow is:
    more than four markers AND more markers than content words, or fewer than six content words."""
    rungs = ", ".join(marker("claim", f"v{i}") for i in range(1, 13))
    ladder = ("Les visites de maintenance préventive des pompes immergées sont dues aux échéances "
              f"suivantes du calendrier : {rungs}.")
    content = len([w for w in re.split(r"\s+", re.sub(r"\{\{[^}]*\}\}", " ", ladder))
                   if any(c.isalpha() for c in w)])
    assert (12, content) == (12, 15), f"the fixture drifted: {content} content words"
    result = validate(_payload([ladder, GOOD, GOOD]), node_id="n1", allowed_children=["n1#leaf"],
                      allowed_claims=IDS, value_ids=IDS, min_words=0, **{
                          k: v for k, v in grain_rules("window").items() if k != "min_words"})
    assert result.summary.startswith("Les visites")


def test_the_pilot_sentence_is_still_refused_under_the_new_rule():
    """Eighty-five markers on a score of content words: more than four, and more markers than words."""
    with pytest.raises(HarvestRefusal) as caught:
        validate(_payload([PILOT_SENTENCE_1, GOOD, GOOD]), node_id="n1", allowed_children=["n1#leaf"],
                 allowed_claims=IDS, value_ids=IDS, min_words=0, **{
                     k: v for k, v in grain_rules("window").items() if k != "min_words"})
    assert caught.value.reason == "hollow sentence"


def test_prose_citing_within_the_rule_passes():
    cited = (f"Les pompes sont visitées à {marker('claim', 'v1')} et {marker('claim', 'v2')} selon le "
             "calendrier de maintenance préventive que le titulaire doit respecter.")
    result = _validate([cited, GOOD, GOOD])
    assert result.summary.startswith("Les pompes")
