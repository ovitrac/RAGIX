"""Synthetic strict-schema and exhaustive-accounting falsifiers, no model calls.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

from copy import deepcopy
from dataclasses import replace
import json
import pytest

from ragix_kernels.harvest.bindings import BindingRefusal, record_batch, validate_bindings
from ragix_kernels.harvest.derived import DerivedStore
from ragix_kernels.harvest.quantitative import harvest, roots

TEXT = "Temperature 8 °C à 19 °C"


def test_unparsed_escape_hatch_is_not_a_number_producer():
    from ragix_kernels.harvest.bindings import validate_unparsed
    observation = {"node_id": "node", "start": 0, "end": 7, "exact_raw_span": "several",
                   "suspected_kind": "count", "reason": "not parsed"}
    accepted = validate_unparsed(observation, source_id="copy", node_id="node", text="several items")
    assert accepted.raw == "several" and not hasattr(accepted, "number")
    with pytest.raises(BindingRefusal, match="unparsed_schema"):
        validate_unparsed({**observation, "number": 4}, source_id="copy", node_id="node", text="several items")
    with pytest.raises(BindingRefusal, match="unparsed_source_mismatch"):
        validate_unparsed({**observation, "end": 8}, source_id="copy", node_id="node", text="several items")


def candidates(text=TEXT, **kwargs):
    return harvest(text, source_id="synthetic-copy", node_id="node", classification="CONTENT", **kwargs)


def packet(cs=None):
    cs = candidates() if cs is None else cs
    root, = roots(cs)
    return {"claims": [{"claim_id": "claim-a", "node_id": "node", "candidate_ids": [root.candidate_id],
                        "parameter": {"raw_name": "Temperature", "canonical_name": "temperature", "canonical_name_status": "proposed"},
                        "constraint": {"kind": "range", "composite_candidate_id": root.candidate_id},
                        "statement_role": "requirement", "comparison_role": "bound", "source_authority": "author",
                        "quoted_source_authority": "none", "applicability": "local"}], "abstentions": []}


def validate(p, cs=None, **kwargs):
    return validate_bindings(json.dumps(p), candidates() if cs is None else cs,
                             node_texts={"node": TEXT}, policy_id="synthetic-policy/1",
                             semantic_validator=lambda claim, refs: (), **kwargs)


def test_composite_accounts_members_and_multiple_claims():
    p = packet()
    p["claims"].append({**deepcopy(p["claims"][0]), "claim_id": "claim-b"})
    result = validate(p)
    assert len(result.accounting) == 3
    assert all(ids == ("claim-a", "claim-b") for ids in result.accounting.values())


def test_complete_abstention_is_valid():
    p = {"claims": [], "abstentions": [{"candidate_id": c.candidate_id, "reason": "unresolved"} for c in candidates()]}
    assert len(validate(p).abstentions) == 3


@pytest.mark.parametrize("mutation,rule", [
    (lambda p: p["claims"].clear(), "candidate_omitted"),
    (lambda p: p["claims"][0].update(review_state="ready"), "claim_schema"),
    (lambda p: p["claims"][0]["constraint"].update(lower=8), "constraint_schema"),
    (lambda p: p["claims"][0].update(node_id="other"), "candidate_scope"),
    (lambda p: p["claims"][0].update(candidate_ids=["stale"]), "unknown_candidate"),
    (lambda p: p["claims"][0]["parameter"].update(raw_name="8 °C"), "parameter_value_smuggling"),
    (lambda p: p["claims"][0]["parameter"].update(raw_name="8"), "parameter_value_smuggling"),
    (lambda p: p["claims"][0]["parameter"].update(canonical_name="19 °C"), "parameter_value_smuggling"),
    (lambda p: p["claims"][0]["parameter"].update(canonical_name_status="accepted"), "parameter_authority"),
    (lambda p: p["claims"].append(deepcopy(p["claims"][0])), "claim_identity"),
    (lambda p: p["abstentions"].append({"candidate_id": candidates()[0].candidate_id, "reason": "unknown"}), "binding_and_abstention"),
])
def test_refusals(mutation, rule):
    p = packet()
    mutation(p)
    with pytest.raises(BindingRefusal) as exc:
        validate(p)
    assert exc.value.rule == rule


def test_model_cannot_assemble_range():
    p = packet()
    p["claims"][0]["candidate_ids"] = [c.candidate_id for c in candidates() if not c.members]
    with pytest.raises(BindingRefusal, match="model_assembled_composite"):
        validate(p)


def test_scientific_name_allowed_and_uncertainty_preserved():
    cs = candidates(uncertainty=("CELL_BOUNDARY_UNRESOLVED",))
    p = packet(cs)
    p["claims"][0]["parameter"]["raw_name"] = "CO2"
    assert "CELL_BOUNDARY_UNRESOLVED" in validate(p, cs).review_reasons["claim-a"]


@pytest.mark.parametrize("raw", ['{"claims":[],"claims":[],"abstentions":[]}', '{"claims":NaN,"abstentions":[]}', '```json\n{}\n```'])
def test_strict_json(raw):
    with pytest.raises(BindingRefusal):
        validate_bindings(raw, (), node_texts={}, policy_id="test/1", semantic_validator=lambda c, r: ())


def test_source_tampering_and_cross_copy():
    cs = candidates()
    with pytest.raises(BindingRefusal, match="candidate_source_mismatch"):
        validate(packet(cs), [replace(cs[0], raw="changed"), *cs[1:]])
    changed = [replace(c, source_id="other") if not c.members else c for c in cs]
    with pytest.raises(BindingRefusal, match="composite_scope"):
        validate(packet(cs), changed)


def test_semantic_callback_is_mandatory_and_enforced():
    with pytest.raises(BindingRefusal, match="semantic_policy"):
        validate_bindings(json.dumps(packet()), candidates(), node_texts={"node": TEXT},
                          policy_id="test/1", semantic_validator=lambda c, r: ("role-conflict",))
    with pytest.raises(ValueError):
        validate_bindings("{}", (), node_texts={}, policy_id="test/1", semantic_validator=None)


def test_existing_store_append_replay_and_staleness(tmp_path):
    store = DerivedStore(str(tmp_path / "derived.sqlite"), "synthetic-root", "source-hash", "run")
    try:
        p = packet()
        raw = json.dumps(p)
        report = validate(p)
        first = record_batch(store, "batch", report, raw_output=raw, created_at="2026-01-01")
        assert first == record_batch(store, "batch", report, raw_output=raw, created_at="2026-01-02")
        assert store.conn.execute("SELECT count(*) FROM knowledge").fetchone()[0] == 1
        with pytest.raises(BindingRefusal, match="raw_output_mismatch"):
            record_batch(store, "batch", report, raw_output="{}", created_at="now")
        store.source_sha256 = "different"
        with pytest.raises(BindingRefusal, match="batch_identity_conflict"):
            record_batch(store, "batch", report, raw_output=raw, created_at="now")
    finally:
        store.conn.close()


def test_refusal_is_audited_without_becoming_accepted(tmp_path):
    from ragix_kernels.harvest.bindings import BindingFailure, validate_and_record
    store = DerivedStore(str(tmp_path / "derived.sqlite"), "synthetic", "source", "run")
    try:
        raw = '{"claims":[],"abstentions":[]}'
        result = validate_and_record(store, "refused", raw, candidates(), node_texts={"node": TEXT},
                                     policy_id="test/1", semantic_validator=lambda c, r: (), created_at="now")
        assert isinstance(result, BindingFailure) and result.rule == "candidate_omitted"
        assert result.outcome == "refused"
        payload = json.loads(store.conn.execute("SELECT k_json FROM knowledge").fetchone()[0])
        assert payload["raw_output"] == raw and payload["report"]["outcome"] == "refused"
        store.conn.commit()
    finally:
        store.conn.close()
