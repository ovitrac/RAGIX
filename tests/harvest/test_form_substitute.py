"""`form.substitute` — the values are protected, not written by the model.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

One implementation, used by every job that protects a figure, so they cannot drift. The two bugs
of a reference implementation are the first two gates here: the unit alternation matched « h »
before « heures », so « 1 000 h » was replaced INSIDE « 1 000 heures » and left « eures » standing;
and a roll-up's floor of 80 content words was applied to window answers, where the rule is 30.
"""
from __future__ import annotations

import pytest

from ragix_kernels.harvest.form import HarvestRefusal, grain_rules, marker, substitute


# --------------------------------------------------------------------- the two bugs

def test_a_longer_unit_is_never_replaced_inside_a_shorter_one():
    """« 1 000 h » must not be substituted inside « 1 000 heures », leaving « eures » behind."""
    offered = [{"value_id": "v1", "raw": "1 000 heures", "normalized": "PT1000H"},
               {"value_id": "v2", "raw": "1 000 h", "normalized": "PT1000H"}]
    out, made = substitute(["Une visite toutes les 1 000 heures est prévue au contrat."], offered)
    assert "eures" not in out[0], out[0]
    assert out[0] == f"Une visite toutes les {marker('claim', 'v1')} est prévue au contrat."
    assert [m["value_id"] for m in made] == ["v1"]


def test_the_grain_is_the_callers_to_choose_and_the_window_floor_is_not_the_rollups():
    """substitute() applies no floor at all; the caller passes the grain's own numbers."""
    assert grain_rules("window")["min_words"] == 30
    assert grain_rules("rollup")["min_words"] == 80
    assert grain_rules("window")["min_sentence_words"] == 6
    assert grain_rules("rollup")["min_sentence_words"] == 8


# --------------------------------------------------------------------- the rule itself

def test_a_figure_with_no_offered_value_is_a_refusal_that_names_it():
    with pytest.raises(HarvestRefusal) as caught:
        substitute(["Le délai est de 42 jours pour cette prestation."],
                   [{"value_id": "v1", "raw": "6 mois", "normalized": "P6M"}])
    assert caught.value.reason == "value not offered" and "42 jours" in caught.value.detail


def test_equality_is_exact_and_a_shorter_figure_never_claims_a_longer_ones_id():
    """The looser rule would let « 50 heures » match « 1 000 heures » by containment."""
    offered = [{"value_id": "v1", "raw": "1 000 heures", "normalized": "PT1000H"}]
    with pytest.raises(HarvestRefusal):
        substitute(["Une intervention sous 50 heures est demandée."], offered)


def test_a_value_is_matched_through_the_grammars_normalisation():
    """« 48h » in the source and « 48 heures » in the sentence are one value."""
    offered = [{"value_id": "v1", "raw": "48h", "normalized": "PT48H"}]
    out, made = substitute(["Le titulaire intervient sous 48 heures sur site."], offered)
    assert made and made[0]["value_id"] == "v1" and marker("claim", "v1") in out[0]


def test_a_reference_or_a_bare_number_is_neither_cited_nor_refused():
    """What counts as a figure is the grammar's answer: an article number is not a claimable value."""
    out, made = substitute(["Les articles 10 et 11 du CCTP précisent la visite."], [])
    assert out == ["Les articles 10 et 11 du CCTP précisent la visite."] and made == []
