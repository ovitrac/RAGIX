"""form — the harvest form's refusals and the extractive floor. Synthetic payloads only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The two refusals the design rests on are the first two tests: a critical value written by the
model, and a summary sentence that maps to no child. Every name and sentence is invented.
"""

from __future__ import annotations

import json

import pytest

from ragix_kernels.harvest.form import MIN_PROSE_WORDS, HarvestRefusal, extractive_summary, validate

CHILDREN = ["c1", "c2"]
CLAIMS = ["abcdef1234567890", "0123456789abcdef"]


def payload(**over):
    base = {"node_id": "n1", "summary": "La date limite est {{claim:abcdef1234}}.",
            "summary_map": [{"sentence": 1, "children": ["c1"]}],
            "interpreted": {"relevance": "critical", "type": "condition", "links": []},
            "references": {"claims": ["abcdef1234567890"], "children": ["c1"]}}
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


def check(raw):
    return validate(raw, node_id="n1", allowed_children=CHILDREN, allowed_claims=CLAIMS)


def refused(raw, reason):
    with pytest.raises(HarvestRefusal) as exc:
        check(raw)
    assert exc.value.reason == reason, exc.value.reason


def test_a_valid_call_is_accepted_and_renders_from_the_claims():
    result = check(payload())
    assert result.render({"abcdef1234567890": "2031-10-14T10:00"}) == "La date limite est 2031-10-14T10:00."


def test_a_critical_value_written_by_the_model_is_refused():
    for literal in ("La date limite est 2031-10-14.", "Le montant est 250,00 €.",
                    "La remise est de 35 %.", "Le délai est de 10 jours.",
                    "La clôture est à 12 heures 00.", "Déposé le 14/10/2031."):
        refused(payload(summary=literal), "critical value written by the model")


def test_a_sentence_that_cites_a_value_must_name_it():
    """form 0.5: a sentence citing a value may not leave it unnamed."""
    refused(payload(summary_map=[{"sentence": 1, "children": []}]), "unsupported sentence")


def test_a_sentence_that_cites_nothing_is_attributed_to_the_object():
    """A general sentence with no value in it, naming no child, belongs to the object itself; form 0.4
    refused it and pushed a model toward a fabricated citation."""
    body = json.loads(v2())
    body["summary"] = ["Le titulaire fournit ses propres outils et ses équipements de protection."]
    body["summary_map"] = [{"sentence": 1, "children": []}]
    result = validate(json.dumps(body), node_id="n1", allowed_children=["c1", "n1#leaf"],
                      allowed_claims=CLAIMS, value_ids=["v1", "v2"], text=TEXT)
    assert result.summary_map[0]["children"] == ["n1#leaf"]


def test_every_sentence_must_be_mapped():
    refused(payload(summary="Première phrase. Seconde phrase."), "unsupported sentence")


def test_an_unknown_child_or_claim_is_refused():
    refused(payload(summary_map=[{"sentence": 1, "children": ["c9"]}]), "unknown child")
    refused(payload(summary="Voir {{claim:ffffffff}}."), "unknown claim reference")
    refused(payload(references={"claims": ["nope"], "children": []}), "unknown claim reference")


def test_malformed_json_and_missing_fields_are_refused():
    refused("{not json", "strict JSON")
    refused(json.dumps([1, 2]), "strict JSON")
    body = json.loads(payload())
    body.pop("interpreted")
    refused(json.dumps(body), "missing field")


def test_the_vocabularies_are_closed():
    refused(payload(interpreted={"relevance": "vital", "type": "condition", "links": []}),
            "relevance outside vocabulary")
    refused(payload(interpreted={"relevance": "critical", "type": "advice", "links": []}),
            "act outside vocabulary")
    refused(payload(interpreted={"relevance": "critical", "type": "condition",
                                 "links": [{"target": "RC 6", "kind": "explains"}]}),
            "link kind outside vocabulary")


def test_the_interpreted_layer_may_not_carry_a_critical_value():
    refused(payload(interpreted={"relevance": "critical", "type": "condition", "links": [],
                                 "note": "échéance 2031-10-14"}),
            "critical value written by the model")


def test_the_wrong_node_is_refused():
    refused(payload(node_id="n2"), "wrong node")


def test_the_extractive_floor_selects_and_maps_without_a_model():
    children = [{"id": "c1", "text": "La visite est obligatoire. Elle se tient en septembre."},
                {"id": "c2", "text": "Les plis arrivent avant la date limite. Ensuite rien."},
                {"id": "c3", "text": ""}]
    result = extractive_summary(children, budget_sentences=2)
    assert result.summary == "La visite est obligatoire. Les plis arrivent avant la date limite."
    assert [m["children"] for m in result.summary_map] == [["c1"], ["c2"]]
    assert result.form_version.endswith("+extractive")
    assert extractive_summary(children, 2).summary == result.summary      # deterministic
    assert len(extractive_summary(children, 1).summary_map) == 1          # the budget is respected


def test_the_floor_refuses_when_no_child_carries_a_sentence():
    with pytest.raises(HarvestRefusal):
        extractive_summary([{"id": "c1", "text": "   "}])


# ---------------------------------------------------------------- form v0.2

TEXT = "L'Office du Val Fictif exige une visite. La remise est de {x}."


def v2(**over):
    base = {"node_id": "n1", "summary": "La date limite est {{claim:abcdef1234}}.",
            "summary_map": [{"sentence": 1, "children": ["c1"]}],
            "interpreted": {"relevance": "critical", "type": "condition", "links": []},
            "references": {"claims": ["abcdef1234567890"], "children": ["c1"]},
            "values": [{"value_id": "v1", "relevance": "critical", "type": "condition", "links": []}],
            "entities": [{"span": "Office du Val Fictif", "kind": "organisation",
                          "relevance": "informative", "type": "informative"}]}
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


def check2(raw):
    return validate(raw, node_id="n1", allowed_children=CHILDREN, allowed_claims=CLAIMS,
                    value_ids=["v1", "v2"], text=TEXT)


def refused2(raw, reason):
    with pytest.raises(HarvestRefusal) as exc:
        check2(raw)
    assert exc.value.reason == reason, exc.value.reason


def test_the_model_classifies_the_grammar_s_values_and_never_rewrites_them():
    result = check2(v2())
    assert result.values[0]["value_id"] == "v1" and result.form_version == "harvest-form/0.9"
    for key in ("value", "normalized", "raw"):
        refused2(v2(values=[{"value_id": "v1", "relevance": "critical", "type": "condition",
                             key: "2031-10-14"}]),
                 "critical value written by the model")


def test_an_unknown_value_reference_is_refused():
    refused2(v2(values=[{"value_id": "v9", "relevance": "critical", "type": "condition"}]),
             "unknown value reference")


def test_an_entity_span_must_be_byte_exact_in_the_text():
    result = check2(v2())
    assert result.entities[0]["kind"] == "organisation"
    refused2(v2(entities=[{"span": "Office du Val Voisin", "kind": "organisation",
                           "relevance": "informative", "type": "informative"}]),
             "entity span not verbatim")
    # a PDF splits a word across a line break; the model cannot echo that, so the span is located loosely
    # and the SOURCE's own bytes are recorded
    broken = validate(json.dumps({**json.loads(v2()),
                                  "entities": [{"span": "Office du Val Fictif", "kind": "organisation",
                                                "relevance": "informative", "act": "informative"}]}),
                      node_id="n1", allowed_children=CHILDREN, allowed_claims=CLAIMS,
                      value_ids=["v1", "v2"], text="L'Office du Val\nFictif exige une visite.")
    assert broken.entities[0]["span"] == "Office du Val\nFictif"
    refused2(v2(entities=[{"span": "Office du Val Fictif", "kind": "company",
                           "relevance": "informative", "type": "informative"}]),
             "entity kind outside vocabulary")
    refused2(v2(entities=[{"kind": "person", "relevance": "informative", "type": "informative"}]),
             "entity span missing")


# ------------------------------------------- what an early bake-off taught (form 0.3)

def test_references_is_optional_because_the_prompt_never_asked_for_it():
    body = json.loads(v2())
    body.pop("references")
    assert check2(json.dumps(body)).references == {"claims": [], "children": []}


def test_a_summary_may_arrive_as_a_list_of_sentences():
    body = json.loads(v2())
    body["summary"] = ["Première phrase.", "Seconde phrase."]
    body["summary_map"] = [{"sentence": 1, "children": ["c1"]}, {"sentence": 2, "children": ["v1"]}]
    result = check2(json.dumps(body))
    assert result.summary == "Première phrase. Seconde phrase."
    body["summary_map"] = [{"sentence": 1, "children": ["c1"]}]
    refused2(json.dumps(body), "unsupported sentence")


def test_a_sentence_may_map_to_the_values_it_draws_on():
    body = json.loads(v2())
    body["summary_map"] = [{"sentence": 1, "children": ["v1"]}]
    assert check2(json.dumps(body)).summary_map[0]["children"] == ["v1"]
    body["summary_map"] = [{"sentence": 1, "children": ["v9"]}]
    refused2(json.dumps(body), "unknown child")


def test_an_empty_body_is_refused_as_strict_json():
    refused2("", "strict JSON")


def test_the_node_prose_role_must_write_prose_not_placeholders():
    """A summarizer that cannot summarise even when asked for a minimum is refused for that role."""
    hollow = json.loads(v2())
    hollow["summary"] = "{{claim:v1}} {{claim:v1}} {{claim:v1}}"
    hollow["summary_map"] = [{"sentence": 1, "children": ["v1"]}]
    with pytest.raises(HarvestRefusal) as exc:
        validate(json.dumps(hollow), node_id="n1", allowed_children=CHILDREN,
                 allowed_claims=CLAIMS + ["v1", "v2"], value_ids=["v1", "v2"], text=TEXT,
                 min_words=MIN_PROSE_WORDS)
    assert exc.value.reason == "summary below the minimum"
    # the same answer is accepted for a role that has no prose floor
    accepted = validate(json.dumps(hollow), node_id="n1", allowed_children=CHILDREN,
                        allowed_claims=CLAIMS + ["v1", "v2"], value_ids=["v1", "v2"], text=TEXT)
    assert accepted.summary.startswith("{{claim:")


def test_real_prose_clears_the_floor():
    body = json.loads(v2())
    body["summary"] = ("Le dossier décrit les moyens humains affectés aux missions du marché et "
                       "l'organisation retenue pour les interventions de maintenance préventive, "
                       "corrective et curative dans chaque site desservi par le présent marché "
                       "public, avec les astreintes et les rapports attendus, avant la date limite "
                       "fixée à {{claim:v1}}.")
    body["summary_map"] = [{"sentence": 1, "children": ["v1"]}]
    result = validate(json.dumps(body), node_id="n1", allowed_children=CHILDREN,
                      allowed_claims=CLAIMS + ["v1", "v2"], value_ids=["v1", "v2"], text=TEXT,
                      min_words=MIN_PROSE_WORDS)
    assert result.form_version == "harvest-form/0.9"


def test_an_ambiguous_claim_prefix_is_refused_not_resolved_by_luck():
    """Two claims sharing a prefix were resolved by dict order: the wrong value could enter the prose."""
    result = check2(v2(summary="La date limite est {{claim:abcdef12}}."))
    with pytest.raises(HarvestRefusal) as exc:
        result.render({"abcdef1234567890": "2031-10-14T10:00", "abcdef1299999999": "2031-10-07T10:00"})
    assert exc.value.reason == "ambiguous claim reference"
    assert result.render({"abcdef1234567890": "2031-10-14T10:00"}) == "La date limite est 2031-10-14T10:00."
