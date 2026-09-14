"""The bake-off runner, with the model replaced by a stub. No network, no corpus, no GPU.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

What is tested is the runner's own contract: that a well-formed answer is validated and rendered from
the grammars' values, that each kind of bad answer is recorded as a refusal with its reason rather
than repaired, and that the raw response is always kept.
"""

from __future__ import annotations

import json

from ragix_kernels.harvest import runner

TEXT = "La demande doit intervenir au plus tard 10 jours avant la date limite. Le syndicat organise la visite."


def answer(**over):
    body = {"node_id": "c1", "summary": "Le délai est de {{claim:v1}}.",
            "summary_map": [{"sentence": 1, "children": ["c1#leaf"]}],
            "values": [{"value_id": "v1", "relevance": "critical", "type": "condition", "links": []}],
            "entities": [], "interpreted": {"relevance": "critical", "type": "condition", "links": []},
            "references": {"claims": [], "children": ["c1#leaf"]}}
    body.update(over)
    return json.dumps(body, ensure_ascii=False)


def run_one(monkeypatch, raw):
    monkeypatch.setattr(runner, "call_ollama", lambda model, prompt, timeout=120.0, host=None: (raw, 1.5))
    return runner.harvest_object("c1", TEXT, "stub-model")


def test_a_good_answer_is_validated_and_rendered_from_the_grammars(monkeypatch):
    record = run_one(monkeypatch, answer())
    assert record["ok"] is True and record["refusal"] is None
    assert record["rendered_summary"] == "Le délai est de P10D."
    assert record["grammar_values"][0]["kind"] == "duration"
    assert record["raw_response"] == answer()
    assert record["latency_s"] == 1.5


def test_a_written_critical_value_is_a_recorded_refusal(monkeypatch):
    record = run_one(monkeypatch, answer(summary="Le délai est de 10 jours."))
    assert record["ok"] is False and record["refusal"] == "critical value written by the model"
    assert record["raw_response"], "the raw answer is kept even when refused"


def test_an_unmapped_sentence_and_a_bad_entity_span_are_refusals(monkeypatch):
    record = run_one(monkeypatch, answer(summary_map=[{"sentence": 1, "children": []}]))
    assert record["refusal"] == "unsupported sentence"
    record = run_one(monkeypatch, answer(entities=[{"span": "La clinique du Val Voisin", "kind": "organisation",
                                                    "relevance": "informative", "type": "informative"}]))
    assert record["refusal"] == "entity span not verbatim"


def test_malformed_json_is_a_refusal_not_a_crash(monkeypatch):
    record = run_one(monkeypatch, "{oops")
    assert record["ok"] is False and record["refusal"] == "strict JSON"


def test_a_transport_failure_is_recorded_without_a_latency(monkeypatch):
    def boom(model, prompt, timeout=120.0, host=None):
        raise TimeoutError("no server")
    monkeypatch.setattr(runner, "call_ollama", boom)
    record = runner.harvest_object("c1", TEXT, "stub-model")
    assert record["ok"] is False and record["refusal"] == "transport" and record["latency_s"] is None


def test_the_call_goes_to_the_declared_host_and_no_other(monkeypatch):
    """The server is configuration: the request is built from the host given, localhost by default."""
    seen = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": answer()}).encode()

    def urlopen(request, timeout=None):
        seen.append(request.full_url)
        return _Resp()

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    runner.harvest_object("c1", TEXT, "stub-model")
    runner.harvest_object("c1", TEXT, "stub-model", host="http://127.0.0.1:9999/")
    assert seen == ["http://127.0.0.1:11434/api/generate", "http://127.0.0.1:9999/api/generate"]
