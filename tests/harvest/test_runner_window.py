"""runner, pass 2 at window grain — gated without a model.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The job calls a model; everything that can be wrong about it can be wrong offline. What is held here:
the prompt never shows a marker (the model must not write one — the pipeline substitutes), the
provenance map is DERIVED from the substitutions rather than taken from the answer, a fabricated
figure is refused instead of laundered, and `num_ctx` is sized from the window rather than habit.
Every answer below is written for the test.
"""
from __future__ import annotations

import inspect

import pytest

from ragix_kernels.harvest import runner as J
from ragix_kernels.harvest.form import PATTERNS, HarvestRefusal

PROSE = ("Les visites de maintenance préventive des appareils sont dues au terme prévu par le calendrier "
         "contractuel du marché, et le compte rendu détaillé est remis au responsable technique du site "
         "desservi après chaque passage du technicien chargé des équipements concernés.")


def test_the_prompt_shows_no_marker_because_the_model_must_not_write_one():
    """The substitution variant's whole point: the pipeline protects the values."""
    values = [{"value_id": "v1", "kind": "duration", "raw": "1 000 heures", "normalized": "PT1000H"}]
    prompt = J.render_window("Une visite est due toutes les 1 000 heures.", values)
    # no MARKER of any kind. The JSON example the form shows does contain braces, and must.
    for kind, pattern in PATTERNS.items():
        assert not pattern.search(prompt), f"the prompt shows a {kind} marker"
    assert "{{claim:" not in prompt and "{claim:" not in prompt
    assert "v1" in prompt and "1 000 heures" in prompt          # the values are shown, by id and raw
    assert "aucune accolade, aucun" in prompt                   # and the rule is scoped to the prose


def test_an_answer_naming_a_place_is_accepted():
    """An entity is {kind, span}: asking it for a relevance and an act refused every answer that named
    a place or an organisation."""
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"}]
    text = "Les visites de l'Office du Val Fictif sont dues à 500 heures selon le calendrier."
    body = {"summary": [PROSE],
            "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"}],
            "entities": [{"kind": "organisation", "span": "Office du Val Fictif"}],
            "interpreted": {"relevance": "critical", "act": "condition"}}
    verdict = J.judge(body, text, values, "n1")
    assert verdict["form"].entities and verdict["form"].entities[0]["kind"] == "organisation"
    assert verdict["form"].entities[0]["span"] == "Office du Val Fictif"


def test_a_missing_field_says_missing_rather_than_outside_vocabulary():
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"}]
    body = {"summary": [PROSE],
            "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"}],
            "entities": [], "interpreted": {"act": "condition"}}
    with pytest.raises(HarvestRefusal) as caught:
        J.judge(body, "Les visites sont dues à 500 heures.", values, "n1")
    assert caught.value.reason == "missing field" and caught.value.detail == "relevance"


def test_the_timeout_carries_the_queue_and_the_tail():
    """Three pilot calls cannot see a tail, so the queue's worst case gains half again."""
    assert J.TIMEOUT_QUEUE_FACTOR == 1.5
    assert J.timeout_for(36.2, 4) == 277.2
    assert J.timeout_for(36.2, 4) > 186.99 * 1.4
    assert J.timeout_for(8, 1) == 72.0              # and a single worker is still bounded sanely


def test_the_context_is_sized_from_the_window_and_capped():
    small, large = "a" * 2000, "a" * 40000
    assert J.num_ctx_for(small, 32768) < J.num_ctx_for(large, 32768)
    assert J.num_ctx_for(large, 8192) == 8192                   # never beyond what the model declares
    assert J.num_ctx_for(small, 32768) >= 4096                  # and never below the prompt itself
    assert J.num_ctx_for(small, 32768) & (J.num_ctx_for(small, 32768) - 1) == 0   # a power of two


def test_the_map_is_derived_from_the_substitutions_not_taken_from_the_answer():
    """A model that claims to cite v3 in sentence 1 cannot make it so."""
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"},
              {"value_id": "v2", "kind": "duration", "raw": "1 000 heures", "normalized": "PT1000H"}]
    text = "Les visites sont dues à 500 heures et à 1 000 heures selon le calendrier du titulaire."
    body = {"summary": ["Les visites de maintenance préventive des pompes de relevage sont dues "
                        "à 500 heures puis à 1 000 heures selon le calendrier que le titulaire doit "
                        "respecter sur chaque site desservi par le marché, et chaque passage donne "
                        "lieu à un compte rendu remis au responsable technique de la station de "
                        "pompage concernée."],
            "summary_map": [{"sentence": 1, "children": ["v3"]}],      # the model's claim, ignored
            "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"},
                       {"value_id": "v2", "relevance": "critical", "act": "condition"}],
            "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    verdict = J.judge(body, text, values, "n1")
    assert verdict["map"] == [{"sentence": 1, "children": ["v1", "v2"]}]
    assert len(verdict["substitutions"]) == 2


def test_a_figure_the_text_does_not_carry_is_refused_not_laundered():
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"}]
    body = {"summary": ["Le titulaire intervient sous 48 heures pour toute panne bloquante signalée "
                        "par l'exploitant du site."],
            "values": [], "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    with pytest.raises(HarvestRefusal) as caught:
        J.judge(body, "Les visites sont dues à 500 heures.", values, "n1")
    assert caught.value.reason == "value not offered" and "48 heures" in caught.value.detail


def test_the_k_rows_of_a_value_not_offered_are_dropped_rather_than_stored():
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"}]
    text = "Les visites sont dues à 500 heures selon le calendrier."
    body = {"summary": ["Les visites de maintenance préventive de ces appareils sont dues à 500 heures "
                        "selon le calendrier que le titulaire doit respecter sur chaque site, et le "
                        "compte rendu d'intervention est remis à l'exploitant après chaque passage "
                        "du technicien chargé de la maintenance des équipements concernés."],
            "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"},
                       {"value_id": "v9", "relevance": "critical", "act": "condition"}],
            "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    verdict = J.judge(body, text, values, "n1")
    assert [k["value_id"] for k in verdict["k_rows"]] == ["v1"]


@pytest.mark.parametrize("piece, expected", [
    ("12.CCTP_Lot _Pompes et relevage.pdf", "12"),
    ("00.CCTP_Clauses communes.pdf", "00"),
    ("06.CCTP_Éclairage de sécurité.pdf", "06"),
    ("CCAP 2031XY-4 signé.pdf", None),
    ("RC 2031XY-4 (v2)_signé.pdf", None),
    (None, None),
])
def test_the_piece_number_is_derived_from_the_cards_file_name(piece, expected):
    """The cards carry `piece` as a FILE NAME; matching it against "12" found nothing, and the same
    field feeds lot_specific, so every window would have scored against an empty target set."""
    assert J.piece_number(piece) == expected


def test_coverage_is_a_pieces_number_and_a_thin_piece_carries_no_threshold():
    records = [
        {"ok": True, "piece": "12", "lot_specific_cited": 3, "lot_specific_offered": 4},
        {"ok": True, "piece": "12", "lot_specific_cited": 2, "lot_specific_offered": 3},
        {"ok": True, "piece": "12", "lot_specific_cited": 0, "lot_specific_offered": 1},
        {"ok": True, "piece": "07", "lot_specific_cited": 1, "lot_specific_offered": 4},
        {"ok": True, "piece": "03", "lot_specific_cited": 0, "lot_specific_offered": 1},
        {"ok": False, "piece": "12", "lot_specific_cited": 9, "lot_specific_offered": 9},
    ]
    pieces = J.coverage_by_piece(records)
    assert pieces["12"]["coverage"] == "5/8" and pieces["12"]["gated"] and pieces["12"]["passes"]
    assert pieces["07"]["coverage"] == "1/4" and pieces["07"]["passes"] is False
    assert pieces["03"]["gated"] is False and pieces["03"]["passes"] is None
    assert pieces["12"]["windows"] == 3, "a refused window must not be counted as read"


def test_a_summary_of_objects_is_a_counted_refusal_not_a_crash():
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"}]
    body = {"summary": [{"texte": "une phrase"}, "et une vraie phrase"], "values": [],
            "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    with pytest.raises(HarvestRefusal) as caught:
        J.judge(body, "Les visites sont dues à 500 heures.", values, "n1")
    assert caught.value.reason == "summary is not prose" and "dict" in caught.value.detail


def test_a_sentence_that_repeats_a_phrase_is_refused():
    """Both halves true and sourced, and padding all the same."""
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"}]
    body = {"summary": ["Le titulaire intervient sur le site de la station principale et son "
                        "astreinte couvre les interventions urgentes sur le site de la station principale "
                        "pendant toute la durée du marché conclu par le syndicat."],
            "values": [], "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    with pytest.raises(HarvestRefusal) as caught:
        J.judge(body, "Les visites sont dues à 500 heures.", values, "n1")
    assert caught.value.reason == "repeated phrase"
    assert "site de la" in caught.value.detail or "station principale" in caught.value.detail


def test_the_same_value_cited_twice_in_one_sentence_is_refused():
    """The exact rule beside the heuristic: one value, two markers, one sentence."""
    values = [{"value_id": "v1", "kind": "duration", "raw": "7 jours", "normalized": "P7D"}]
    text = "Le service fonctionne 7 jours sur 7."
    body = {"summary": ["Le service de garde des installations fonctionne 7 jours sur une semaine de "
                        "7 jours et le titulaire assure la permanence téléphonique demandée par "
                        "l'exploitant pour toutes les pompes du réseau de distribution de la commune."],
            "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"}],
            "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    with pytest.raises(HarvestRefusal) as caught:
        J.judge(body, text, values, "n1")
    assert caught.value.reason == "repeated phrase" and "cites v1 twice" in caught.value.detail


def test_one_normalised_value_carries_one_classification():
    """« 24h » and « 24 heures » are two spans of one value: the first classification stands, the ids
    it covers are recorded, and a disagreement is kept rather than averaged away."""
    values = [{"value_id": "v1", "kind": "duration", "raw": "24h", "normalized": "PT24H"},
              {"value_id": "v2", "kind": "duration", "raw": "24 heures", "normalized": "PT24H"},
              {"value_id": "v3", "kind": "duration", "raw": "20 minutes", "normalized": "PT20M"}]
    text = "Le service est disponible 24h sur 24 heures avec une réponse en 20 minutes."
    body = {"summary": ["Le service d'assistance technique et téléphonique reste disponible en continu "
                        "pour les sites desservis par le marché, la réponse au signalement intervient "
                        "dans le délai contractuel prévu par cette pièce du dossier de consultation, et "
                        "le titulaire consigne chaque appel reçu dans son registre."],
            "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"},
                       {"value_id": "v2", "relevance": "important", "act": "condition"},
                       {"value_id": "v3", "relevance": "critical", "act": "condition"}],
            "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    verdict = J.judge(body, text, values, "n1")
    rows = {r["canonical"]: r for r in verdict["k_rows"]}
    assert sorted(rows) == ["PT20M", "PT24H"]
    assert rows["PT24H"]["relevance"] == "critical"
    assert rows["PT24H"]["covers"] == ["v1", "v2"]
    assert verdict["k_disagreements"] and \
        verdict["k_disagreements"][0]["dropped_as"] == ["important", "condition"]


def test_a_k_row_carrying_the_shared_word_is_flagged():
    """RELEVANCE and TYPES share `informative`, so a swapped one passes the vocabulary check invisibly;
    the row says when it carries the shared word."""
    assert J.AMBIGUOUS == {"informative"}
    values = [{"value_id": "v1", "kind": "duration", "raw": "500 heures", "normalized": "PT500H"},
              {"value_id": "v2", "kind": "duration", "raw": "20 minutes", "normalized": "PT20M"}]
    text = "Les visites sont dues à 500 heures et la réponse intervient en 20 minutes."
    body = {"summary": ["Les visites de maintenance préventive des pompes de relevage sont dues au "
                        "terme prévu par le calendrier contractuel, et la réponse au signalement "
                        "adressé par l'exploitant intervient dans le délai fixé par cette pièce du "
                        "dossier de consultation remis aux candidats."],
            "values": [{"value_id": "v1", "relevance": "informative", "act": "definition"},
                       {"value_id": "v2", "relevance": "critical", "act": "condition"}],
            "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}
    rows = {r["value_id"]: r for r in J.judge(body, text, values, "n1")["k_rows"]}
    assert rows["v1"]["ambiguous_vocabulary"] is True
    assert rows["v2"]["ambiguous_vocabulary"] is False


# ----------------------------------------------------------- the job's own discipline, on its source

def _job_source() -> str:
    return inspect.getsource(J.run_window)


def test_the_release_is_registered_for_every_exit_including_a_signal():
    """A `finally` does not run for SIGTERM, and a killed run left its model resident."""
    source = _job_source()
    assert "atexit.register(release)" in source
    assert "signal.signal(signal.SIGTERM" in source
    assert source.index("atexit.register(release)") < source.index("# ---- 1. the pilot")


def test_the_write_path_does_not_depend_on_the_channel_home():
    """The records must survive a broken channel: progress goes to a file beside them, and the print
    onto stdout is best-effort."""
    source = _job_source()
    write_path = source[source.index("def keep(record: dict)"):source.index("def run(cards_to_run")]
    assert "progress.write" in write_path
    assert "except (BrokenPipeError, OSError)" in write_path
    assert write_path.index("handle.write") < write_path.index("print(line")


def test_work_in_flight_is_bounded_so_a_failure_costs_what_is_in_flight():
    """All the cards were once submitted at once: when the consuming loop stopped, the pool executed
    every queued call with nobody reading the results."""
    source = _job_source()
    assert "len(in_flight) < workers * 2" in source
    # a repeated crash stops the run — but AFTER the record is complete
    assert "return 4" in source and "raise SystemExit(4)" not in source
    assert source.index('(out / "summary.json").write_text') < source.index("if sum(raised) >")
    assert source.index("store.conn.commit()") < source.index("if sum(raised) >")
