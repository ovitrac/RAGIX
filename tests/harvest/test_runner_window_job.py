"""runner, pass 2 at window grain — the whole job against a fake server.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The pilot inside the job, the core behind it, the three outcomes (accepted, refused, unknown) and
the exit gate that reads only the two that measure something; `--resume` that makes no call twice,
`--only` that refuses an id naming no window, and a pilot that fails stopping the core. The store
is synthetic, the model is a function, and every sentence is invented.
"""
from __future__ import annotations

import json

import pytest

from ragix_kernels.harvest import runner as J

from .synthetic import Doc, Sheet, build_store

PIECE = "07.CCTP_Lot _Pompes.pdf"
LEAVES = []
for i in range(4):
    LEAVES += [f"Article {i + 1} Intervention sur le poste {i + 1}",
               "Le titulaire intervient sur les pompes de relevage dans un délai de 48 heures après le signalement.",
               "Il remet un compte rendu écrit à l'exploitant du réseau après chaque intervention."]
SUMMARY = ("Le titulaire intervient sur les pompes de relevage dans un délai de 48 heures après chaque "
           "signalement transmis par l'exploitant, remet un compte rendu écrit détaillé et tient à jour le "
           "registre des interventions réalisées sur les installations du réseau communal.")
GOOD = {"summary": [SUMMARY], "values": [{"value_id": "v1", "relevance": "critical", "act": "condition"}],
        "entities": [], "interpreted": {"relevance": "critical", "act": "condition"}}


@pytest.fixture
def job(tmp_path):
    text = "\n".join(LEAVES)
    spans, pos = [], 0
    for i in range(4):
        block = "\n".join(LEAVES[3 * i:3 * i + 3])
        spans.append((pos, pos + len(block)))
        pos += len(block) + 1
    assert text[spans[-1][0]:spans[-1][1]] == "\n".join(LEAVES[9:12])
    built = build_store(tmp_path / "saqqara.db", [Doc("lot", f"/corpus/{PIECE}", [Sheet(LEAVES, windows=spans)])])
    windows = built["lot"].windows[0]
    cards = tmp_path / "core_abstracts.jsonl"
    cards.write_text("".join(json.dumps({"node_id": w, "level": "window", "piece": PIECE}) + "\n"
                             for w in windows), encoding="utf-8")
    commitments = tmp_path / "commitments.json"
    commitments.write_text(json.dumps({"pieces": {"07": {"sentences": [{"matches": [{"text": "48 heures"}]}]}}}),
                           encoding="utf-8")
    return tmp_path, windows, ["--store", str(tmp_path / "saqqara.db"), "--cards", str(cards),
                               "--commitments", str(commitments), "--pilot-piece", "07"]


def _server(monkeypatch, answer=lambda body: GOOD, fail=()):
    calls = []

    def post(host, path, body, timeout):
        if "prompt" not in body:                      # the release: keep_alive 0
            return {}
        calls.append(body)
        if any(w[:12] in body["prompt"] for w in fail) or (fail and body["prompt"] in fail):
            raise TimeoutError("timed out")
        return {"response": json.dumps(answer(body), ensure_ascii=False), "prompt_eval_count": 300,
                "eval_count": 80, "done_reason": "stop"}

    monkeypatch.setattr(J, "post", post)
    return calls


def _read(out):
    records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    return records, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_the_pilot_then_the_core_and_the_exit_gate(job, monkeypatch):
    root, windows, args = job
    calls = _server(monkeypatch)
    assert J.run_window(args + ["--out", str(root / "run")]) == 0
    records, summary = _read(root / "run")
    pilot = json.loads((root / "run/pilot.json").read_text(encoding="utf-8"))
    assert pilot["passes"] and pilot["passing"] == 3
    assert sorted(r["node_id"] for r in records) == sorted(windows) and len(calls) == 4
    assert all(r["outcome"] == "accepted" and r["coverage"] == "1/1" for r in records)
    assert all(r["summary"][0].count("{{claim:") == 1 and "48 heures" not in r["summary"][0] for r in records)
    assert summary["exit"]["passes"] and summary["exit"]["contract_rate"] == 1.0
    assert summary["coverage_by_piece"]["07"]["coverage"] == "4/4"
    assert len((root / "run/progress.log").read_text(encoding="utf-8").splitlines()) == 4


def test_a_resumed_run_makes_no_call_twice(job, monkeypatch):
    root, windows, args = job
    _server(monkeypatch)
    J.run_window(args + ["--out", str(root / "run")])
    calls = _server(monkeypatch)
    assert J.run_window(args + ["--out", str(root / "run"), "--resume"]) == 0
    assert calls == []
    assert _read(root / "run")[1]["windows"] == 4


def test_only_names_windows_and_an_unknown_id_is_refused(job, monkeypatch):
    root, windows, args = job
    only = root / "only.txt"
    only.write_text("f" * 64 + "\n", encoding="utf-8")
    _server(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        J.run_window(args + ["--out", str(root / "run"), "--only", str(only)])
    assert caught.value.code == 2


def test_a_timeout_is_unknown_and_stays_out_of_the_denominator(job, monkeypatch):
    root, windows, args = job
    last = windows[3]
    texts = {}

    def answer(body):
        return GOOD

    calls = []

    def post(host, path, body, timeout):
        if "prompt" not in body:
            return {}
        calls.append(body)
        if len(calls) == 4:                              # the core's only window
            raise TimeoutError("timed out")
        return {"response": json.dumps(GOOD, ensure_ascii=False), "prompt_eval_count": 300,
                "eval_count": 80, "done_reason": "stop"}

    monkeypatch.setattr(J, "post", post)
    assert J.run_window(args + ["--out", str(root / "run")]) == 0
    records, summary = _read(root / "run")
    unknown = [r for r in records if r["outcome"] == "unknown"]
    assert [r["node_id"] for r in unknown] == [last] and unknown[0]["refusal"] == "timeout"
    assert summary["exit"]["judged"] == 3 and summary["exit"]["unknown"] == 1
    assert summary["exit"]["contract_rate"] == 1.0


def test_a_failing_pilot_stops_the_core(job, monkeypatch):
    root, windows, args = job
    calls = _server(monkeypatch, answer=lambda body: {**GOOD, "summary": ["Trop court, 12 jours."]})
    assert J.run_window(args + ["--out", str(root / "run")]) == 3
    assert len(calls) == 3, "the core ran although the pilot failed"
    assert not (root / "run/summary.json").exists()


def test_a_pilot_declared_by_nothing_is_refused(job, monkeypatch):
    root, windows, args = job
    _server(monkeypatch)
    bare = [a for a in args if a not in ("--pilot-piece", "07")]
    with pytest.raises(SystemExit) as caught:
        J.run_window(bare + ["--out", str(root / "run")])
    assert caught.value.code == 2
