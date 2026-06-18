"""
Tests for the Sprint 1 sealed contracts (WP §13).

Verifies that the canonical contracts load and that every cross-check in the validator
both passes on the real contracts and fails on deliberately broken copies.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-17
"""

import copy

import pytest

from ragix_sealed.contracts import (
    ContractError,
    SealedContracts,
    load_contracts,
    validate,
)
from ragix_sealed.contracts.loader import EXPECTED_EXPORT_MODES


# --------------------------------------------------------------------------------------
# Happy path: the shipped contracts load and validate.
# --------------------------------------------------------------------------------------

def test_canonical_contracts_load():
    c = load_contracts()
    assert isinstance(c, SealedContracts)


def test_trust_invariant_present():
    c = load_contracts()
    inv = c.policy["trust_invariant"]
    assert "Local sealed-zone models are INTERNAL" in inv
    assert "Cooled kernels operate on placeholderized content only" in inv


def test_export_modes_exact():
    c = load_contracts()
    assert set(c.policy["export_modes"].keys()) == EXPECTED_EXPORT_MODES


def test_location_and_multimodal_classes():
    c = load_contracts()
    assert "LOCATION" in c.placeholder_schema["entity_classes"]
    assert c.placeholder_schema["multimodal_entity_classes"]


def test_both_flows_have_cooled_states():
    c = load_contracts()
    assert "COOLED_INDEXABLE" in c.state_machine["document_flow"]["cooled_states"]
    assert "DERIVED_COOLED_DESCRIPTOR" in c.state_machine["derivative_flow"]["cooled_states"]


def test_no_model_allows_reidentify_for_llm():
    c = load_contracts()
    for name, m in c.model_registry["models"].items():
        assert "reidentify_for_llm" not in (m.get("allowed_tasks") or []), name


def test_tool_matrix_disjoint():
    c = load_contracts()
    safe = set(c.tool_matrix["safe"])
    restricted = set(c.tool_matrix["restricted"])
    forbidden = set(c.tool_matrix["forbidden"])
    assert safe.isdisjoint(restricted)
    assert safe.isdisjoint(forbidden)
    assert restricted.isdisjoint(forbidden)


def test_human_traceability_manifest_may_hold_encrypted_filename():
    """The encrypted, human-only manifest legitimately carries original_filename."""
    c = load_contracts()
    enc = c.provenance["human_traceability_manifest"]["encrypted_fields"]
    assert "original_filename" in enc  # exempt because encrypted + human-only
    # ...but it must NOT be a public-facing artifact field.
    assert "original_filename" not in c.provenance["artifact_required_fields"]
    assert "original_filename" not in c.provenance["artifact_optional_fields"]


# --------------------------------------------------------------------------------------
# Negative path: each validator rule rejects a broken contract.
# --------------------------------------------------------------------------------------

def _broken(**overrides) -> SealedContracts:
    """Clone the real contracts, then mutate selected sections for negative tests."""
    c = load_contracts()
    data = {k: copy.deepcopy(getattr(c, k)) for k in c.__dataclass_fields__}
    data.update(overrides)
    return SealedContracts(**{k: data[k] for k in c.__dataclass_fields__})


def test_missing_trust_phrase_rejected():
    c = load_contracts()
    pol = copy.deepcopy(c.policy)
    pol["trust_invariant"] = "something else entirely"
    with pytest.raises(ContractError, match="trust_invariant"):
        validate(_broken(policy=pol))


def test_wrong_export_modes_rejected():
    c = load_contracts()
    pol = copy.deepcopy(c.policy)
    pol["export_modes"] = {"SANITIZED_LLM_SAFE": {}}
    with pytest.raises(ContractError, match="export_modes"):
        validate(_broken(policy=pol))


def test_missing_location_rejected():
    c = load_contracts()
    ps = copy.deepcopy(c.placeholder_schema)
    del ps["entity_classes"]["LOCATION"]
    with pytest.raises(ContractError, match="LOCATION"):
        validate(_broken(placeholder_schema=ps))


def test_transition_to_unknown_state_rejected():
    c = load_contracts()
    sm = copy.deepcopy(c.state_machine)
    sm["document_flow"]["states"]["RECEIVED"]["to"] = ["NOWHERE"]
    with pytest.raises(ContractError, match="unknown state"):
        validate(_broken(state_machine=sm))


def test_unreachable_state_rejected():
    c = load_contracts()
    sm = copy.deepcopy(c.state_machine)
    sm["document_flow"]["states"]["ORPHAN"] = {"to": []}
    with pytest.raises(ContractError, match="unreachable"):
        validate(_broken(state_machine=sm))


def test_terminal_with_outgoing_rejected():
    c = load_contracts()
    sm = copy.deepcopy(c.state_machine)
    sm["document_flow"]["states"]["REASONABLE"]["to"] = ["RECEIVED"]
    with pytest.raises(ContractError, match="terminal"):
        validate(_broken(state_machine=sm))


def test_model_allowing_forbidden_task_rejected():
    c = load_contracts()
    reg = copy.deepcopy(c.model_registry)
    reg["models"]["primary"]["allowed_tasks"].append("reidentify_for_llm")
    with pytest.raises(ContractError, match="forbidden task"):
        validate(_broken(model_registry=reg))


def test_tool_in_two_buckets_rejected():
    c = load_contracts()
    tm = copy.deepcopy(c.tool_matrix)
    tm["forbidden"].append(tm["safe"][0])  # same tool now safe AND forbidden
    with pytest.raises(ContractError, match="overlap"):
        validate(_broken(tool_matrix=tm))


def test_public_audit_field_named_raw_rejected():
    c = load_contracts()
    ae = copy.deepcopy(c.audit_event)
    ae["optional_fields"].append("raw_value")
    with pytest.raises(ContractError, match="forbidden_fields"):
        validate(_broken(audit_event=ae))


def test_registered_worker_missing_field_rejected():
    c = load_contracts()
    pol = copy.deepcopy(c.policy)
    pol["registered_workers"] = [{"worker_id": "w1"}]  # missing required fields
    with pytest.raises(ContractError, match="missing fields"):
        validate(_broken(policy=pol))
