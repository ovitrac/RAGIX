"""runner, the node roll-ups — the helpers, the plan, and a whole run against a fake server.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The job asks a writer (and optionally a challenger) for a document-grain summary built around the
piece's own values. What is held without a model: which values are the piece's own, what the plan
does with a document too long for the writer's context, and that a run writes every record the
moment it is made, pilot first, with the decision it took recorded beside the numbers it took it on.
The store is synthetic and every sentence is invented.
"""
from __future__ import annotations

import json
import re

import pytest

from ragix_kernels.harvest import runner as R

from .synthetic import Doc, Sheet, build_store, windows_of

COMMITMENTS = {"pieces": {
    "07": {"sentences": [{"matches": [{"text": "48 heures"}, {"text": "12 heures"}, {"text": "000 heures"}]}]},
    "08": {"sentences": [{"matches": [{"text": "12 heures"}, {"text": "3 jours"}]}]},
    "09": {"sentences": [{"matches": [{"text": "12 heures"}, {"text": "3 jours"}]}]},
}}


def test_the_own_values_of_a_piece_exclude_the_register_artefact():
    assert R.own_values("07", COMMITMENTS) == {"48heures", "12heures"}


def test_a_lot_specific_value_is_carried_by_at_most_two_pieces():
    """« 12 heures » is on every cover of the family: it measures the template, not the lot."""
    assert R.lot_specific("07", COMMITMENTS) == {"48heures"}
    assert R.lot_specific("08", COMMITMENTS) == {"3jours"}


def test_offered_reports_what_the_grammar_offers_and_is_never_the_denominator():
    values = [{"raw": "48 heures"}, {"raw": "5 jours"}]
    assert R.offered({"48heures", "3jours"}, values) == {"48heures"}


def test_the_letter_ratio_separates_prose_from_a_table():
    assert R.letter_ratio("Le titulaire assure la maintenance des pompes.") > R.SMOKE_LETTER_RATIO
    assert R.letter_ratio("| 12 | 34 | 56 | 78 |\n| 90 | 12 | 34 | 56 |") < R.SMOKE_LETTER_RATIO


def test_the_unguided_prompt_drops_the_piece_s_own_values_and_nothing_else():
    guided = R.rendered_prompt("n1", '  v1 : "48 heures" (duration)  PROPRE', "Un texte.", guided=True)
    plain = R.rendered_prompt("n1", '  v1 : "48 heures" (duration)', "Un texte.", guided=False)
    assert "PROPRE" in guided and "PROPRE" not in plain
    assert plain.splitlines()[-1] == guided.splitlines()[-1] == "Un texte."


# ------------------------------------------------------------------ the store and the plan

LOT = ["Article 3 Délais d'intervention",
       "Le titulaire intervient sur les pompes de relevage dans un délai de 48 heures après le signalement.",
       "Il remet un compte rendu écrit à l'exploitant du réseau après chaque intervention."]
LONG = [f"Article {i} Obligations générales du titulaire" if i % 4 == 0 else
        "Le titulaire respecte les consignes de sécurité du site et informe l'exploitant de toute anomalie "
        "constatée pendant ses interventions sur les installations du réseau." for i in range(1, 49)]


@pytest.fixture
def store(tmp_path):
    long_text = "\n".join(LONG)
    built = build_store(tmp_path / "saqqara.db", [
        Doc("lot", "/corpus/07.CCTP_Lot _Pompes.pdf", [Sheet(LOT)]),
        Doc("ccap", "/corpus/CCAP 2031XY-4.pdf", [Sheet(LONG, windows=windows_of(long_text, 1500))]),
    ])
    pieces = tmp_path / "pieces.json"
    pieces.write_text(json.dumps({built["lot"].doc_id[:8]: "07", built["ccap"].doc_id[:8]: "CCAP"}),
                      encoding="utf-8")
    commitments = tmp_path / "commitments.json"
    commitments.write_text(json.dumps(COMMITMENTS), encoding="utf-8")
    return tmp_path / "saqqara.db", pieces, commitments, built


def _args(store, out, *extra):
    db, pieces, commitments, _ = store
    return ["--store", str(db), "--pieces", str(pieces), "--commitments", str(commitments),
            "--out", str(out), "--pilot-piece", "07", "--ceiling", "3000", *extra]


def test_the_plan_reads_a_long_piece_through_its_parts(store, tmp_path, capsys):
    assert R.run_nodes(_args(store, tmp_path / "out", "--plan-only")) == 0
    plan = json.loads((tmp_path / "out/plan.json").read_text(encoding="utf-8"))
    by_piece = {d["piece"]: d for d in plan["documents"]}
    assert by_piece["07"]["whole"] is True and by_piece["CCAP"]["whole"] is False
    assert by_piece["CCAP"]["parts"] == len(store[3]["ccap"].windows[0]) > 1
    assert plan["calls"] == 1 + by_piece["CCAP"]["parts"] + 1


def test_a_piece_outside_the_declared_scope_is_not_planned(store, tmp_path):
    db, pieces, commitments, built = store
    pieces.write_text(json.dumps({built["lot"].doc_id[:8]: "07"}), encoding="utf-8")
    assert R.run_nodes(_args(store, tmp_path / "out", "--plan-only")) == 0
    plan = json.loads((tmp_path / "out/plan.json").read_text(encoding="utf-8"))
    assert [d["piece"] for d in plan["documents"]] == ["07"]


# ------------------------------------------------------------------ a whole run, fake server

SENTENCES = [
    "Le titulaire assure la maintenance préventive des pompes de relevage et intervient dans le délai "
    "contractuel{cite} après chaque signalement transmis par l'exploitant du réseau communal.",
    "Il remet un compte rendu écrit et détaillé après chaque passage sur les installations et tient à "
    "jour le registre des interventions réalisées sur le site.",
    "Les pièces de rechange sont fournies par le titulaire, qui garantit leur conformité aux normes en "
    "vigueur et assure la traçabilité complète des opérations effectuées pendant toute la durée du marché.",
    "Les consignes de sécurité du site sont respectées par chaque technicien, et toute anomalie constatée "
    "pendant une intervention est signalée sans attendre au responsable désigné par l'exploitant.",
    "Le personnel affecté aux prestations possède les habilitations requises, suit les formations prévues "
    "et dispose des équipements de protection adaptés aux risques identifiés sur chaque installation.",
]


def _answer(prompt: str) -> str:
    node = re.search(r"^node_id : (\S+)$", prompt, re.M).group(1)
    hit = re.search(r'^  (v\d+) : "48 heures"', prompt, re.M)
    cite = f" de {{{{claim:{hit.group(1)}}}}}" if hit else ""
    summary = [s.format(cite=cite if i == 0 else "") for i, s in enumerate(SENTENCES)]
    return json.dumps({"node_id": node, "summary": summary,
                       "summary_map": [{"sentence": i + 1, "children": [hit.group(1)] if hit and i == 0 else []}
                                       for i in range(len(summary))],
                       "values": [{"value_id": hit.group(1), "relevance": "critical", "act": "condition"}]
                       if hit else [],
                       "entities": [], "interpreted": {"relevance": "important", "act": "injunction"}},
                      ensure_ascii=False)


def test_a_whole_run_writes_every_record_as_it_goes(store, tmp_path, monkeypatch):
    calls = []

    def fake_call(prompt, num_ctx, timeout, model=R.WRITER, think=False, host=R.DEFAULT_HOST):
        calls.append((model, num_ctx, host))
        return {"response": _answer(prompt), "prompt_eval_count": 400, "eval_count": 200,
                "done": True, "done_reason": "stop"}, 0.1

    monkeypatch.setattr(R, "call_node", fake_call)
    monkeypatch.setattr(R, "capabilities", lambda model, host=R.DEFAULT_HOST: [])
    monkeypatch.setattr(R, "unload", lambda model, host=R.DEFAULT_HOST: True)
    out = tmp_path / "out"
    assert R.run_nodes(_args(store, out, "--coverage-gate", "1")) == 0

    records = [json.loads(line) for line in (out / "nodes.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert records[0]["stage"] == "pilot" and records[0]["ok"], records[0].get("refusal")
    assert records[0]["coverage"] == {"targets": 1, "covered": 1, "offered_by_the_grammar": 1, "uncitable": 0}
    assert summary["guided"] is True and summary["writer"] == R.WRITER and summary["challenger"] is None
    parts = [r for r in records if r.get("level") == "part"]
    assert len(parts) == len(store[3]["ccap"].windows[0])
    partial = next(r for r in records if r.get("level") == "document" and r.get("partial"))
    assert partial["piece"] == "CCAP" and partial["parts_total"] == len(parts)
    assert partial["parts_read"] == [r["node"] for r in parts] and partial["parts_not_read"] == []
    # the parts' summaries joined are themselves longer than the writer's context: the roll-up is
    # refused and says why, rather than being cut to fit
    assert partial["ok"] is False and partial["refusal"] == "does not fit"
    assert summary["documents"] == 2 and summary["valid"] == 1 and summary["refusals"] == {"does not fit": 1}
    # a record refused as « does not fit » was never sent: every other record is exactly one call
    assert len(calls) == sum(r.get("refusal") != "does not fit" for r in records)
    assert all(host == R.DEFAULT_HOST for _, _, host in calls)


def test_a_pilot_piece_absent_from_the_scope_stops_the_run(store, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "call_node", lambda *a, **k: pytest.fail("no call may be made"))
    with pytest.raises(SystemExit, match="piece 99"):
        R.run_nodes(_args(store, tmp_path / "out")[:-4] + ["--pilot-piece", "99", "--ceiling", "3000"])
