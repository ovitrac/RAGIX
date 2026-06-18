"""
RAGIX-Sealed — contract loader and validator (WP §13, Sprint 1).

Loads the eight declarative YAML contracts in this directory and validates their
internal consistency. These contracts are "the contract everything else obeys"; this
module is the single gate that guarantees they stay coherent.

Cross-checks enforced by ``validate()`` (each raises ``ContractError`` with a precise
message):

- policy: the §1.2 trust invariant text is present (key phrases); the four export modes
  are exactly the §10.1 set; every registered worker satisfies the worker schema.
- placeholder schema: ``LOCATION`` and the multimodal classes are present; every format
  string is renderable.
- state machine: for each flow, the initial state exists, every transition target is a
  declared state, terminals have no outgoing edges, and every non-initial state is
  reachable from the initial state.
- model registry: no model lists a globally-forbidden task in ``allowed_tasks``; every
  model zone is INTERNAL/INTERNAL_WORKER; every payload state is recognised.
- tool matrix: safe / restricted / forbidden are pairwise disjoint.
- audit & provenance schemas: no declared field name collides with the schema's own
  ``forbidden_fields`` (no-raw discipline baked into the contract).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-17
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

CONTRACTS_DIR = Path(__file__).resolve().parent

# Canonical contract filenames.
FILES = {
    "policy": "policy.yaml",
    "placeholder_schema": "placeholder_schema.yaml",
    "state_machine": "ingestion_state_machine.yaml",
    "model_registry": "model_registry.yaml",
    "worker_schema": "worker_config.schema.yaml",
    "tool_matrix": "tool_matrix.yaml",
    "audit_event": "audit_event.schema.yaml",
    "provenance": "provenance.schema.yaml",
}

# §10.1 export modes — the validator pins this exact set.
EXPECTED_EXPORT_MODES = {
    "SANITIZED_LLM_SAFE",
    "HUMAN_AUTHORIZED",
    "AUDIT_ONLY",
    "ORCHESTRATOR_METRICS",
}

# Phrases that must appear verbatim-ish in the trust invariant (§1.2).
TRUST_INVARIANT_PHRASES = (
    "Local sealed-zone models are INTERNAL",
    "OUTSIDE the sealed zone",
    "sealed execution profile",
    "Cooled kernels operate on placeholderized content only",
)

VALID_MODEL_ZONES = {"INTERNAL", "INTERNAL_WORKER"}


class ContractError(Exception):
    """Raised when a contract is missing, unparseable, or internally inconsistent."""


@dataclass(frozen=True)
class SealedContracts:
    """All loaded contracts, keyed as in ``FILES``."""

    policy: Dict[str, Any]
    placeholder_schema: Dict[str, Any]
    state_machine: Dict[str, Any]
    model_registry: Dict[str, Any]
    worker_schema: Dict[str, Any]
    tool_matrix: Dict[str, Any]
    audit_event: Dict[str, Any]
    provenance: Dict[str, Any]


def load_contracts(contracts_dir: Path | str = CONTRACTS_DIR) -> SealedContracts:
    """Load and validate all contracts from ``contracts_dir``.

    Raises ``ContractError`` if any file is missing, unparseable, or inconsistent.
    """
    base = Path(contracts_dir)
    loaded: Dict[str, Any] = {}
    for key, fname in FILES.items():
        path = base / fname
        if not path.exists():
            raise ContractError(f"missing contract: {fname}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ContractError(f"unparseable contract {fname}: {exc}") from exc
        if not isinstance(data, dict):
            raise ContractError(f"contract {fname} must be a mapping at top level")
        loaded[key] = data

    contracts = SealedContracts(**loaded)
    validate(contracts)
    return contracts


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def validate(c: SealedContracts) -> None:
    """Run every cross-check. Raises ``ContractError`` on the first failure."""
    _validate_policy(c)
    _validate_placeholder_schema(c)
    _validate_flow(c.state_machine, "document_flow")
    _validate_flow(c.state_machine, "derivative_flow")
    _validate_model_registry(c)
    _validate_tool_matrix(c)
    # No-raw check applies to PUBLIC-FACING field lists only. Encrypted, human-only
    # fields (e.g. human_traceability_manifest.encrypted_fields) are deliberately exempt.
    _validate_no_raw_schema(
        c.audit_event, "audit_event.schema.yaml",
        public_field_keys=("required_fields", "optional_fields"),
    )
    _validate_no_raw_schema(
        c.provenance, "provenance.schema.yaml",
        public_field_keys=("artifact_required_fields", "artifact_optional_fields"),
    )


def _validate_policy(c: SealedContracts) -> None:
    p = c.policy
    invariant = p.get("trust_invariant", "")
    for phrase in TRUST_INVARIANT_PHRASES:
        if phrase not in invariant:
            raise ContractError(
                f"policy.yaml trust_invariant missing required phrase: {phrase!r}"
            )
    modes = set((p.get("export_modes") or {}).keys())
    if modes != EXPECTED_EXPORT_MODES:
        raise ContractError(
            f"policy.yaml export_modes {sorted(modes)} != expected {sorted(EXPECTED_EXPORT_MODES)}"
        )
    # Every registered worker must satisfy the worker schema's required fields.
    required = set(c.worker_schema.get("required_fields", []))
    for w in p.get("registered_workers") or []:
        missing = required - set(w.keys())
        if missing:
            raise ContractError(
                f"registered worker {w.get('worker_id', '?')} missing fields: {sorted(missing)}"
            )
        if w.get("zone") not in c.worker_schema.get("allowed_zone", []):
            raise ContractError(
                f"registered worker {w.get('worker_id', '?')} has invalid zone {w.get('zone')!r}"
            )


def _validate_placeholder_schema(c: SealedContracts) -> None:
    classes = c.placeholder_schema.get("entity_classes") or {}
    if "LOCATION" not in classes:
        raise ContractError("placeholder_schema.yaml must define LOCATION (required by §8bis)")
    mm = set(c.placeholder_schema.get("multimodal_entity_classes") or [])
    if not mm:
        raise ContractError("placeholder_schema.yaml must list multimodal_entity_classes")
    for etype, spec in classes.items():
        fmt = (spec or {}).get("format", "")
        if "{n" not in fmt:
            raise ContractError(f"entity class {etype} format must contain a counter '{{n}}': {fmt!r}")
        # The format must actually render.
        try:
            fmt.format(etype=etype, n=1)
        except Exception as exc:  # noqa: BLE001
            raise ContractError(f"entity class {etype} format unrenderable: {exc}") from exc


def _validate_flow(state_machine: Dict[str, Any], flow_name: str) -> None:
    flow = state_machine.get(flow_name)
    if not isinstance(flow, dict):
        raise ContractError(f"ingestion_state_machine.yaml missing {flow_name}")
    states = flow.get("states") or {}
    initial = flow.get("initial")
    terminal = set(flow.get("terminal") or [])
    cooled = set(flow.get("cooled_states") or [])

    if initial not in states:
        raise ContractError(f"{flow_name}: initial state {initial!r} not declared")
    for state, spec in states.items():
        for target in (spec or {}).get("to", []):
            if target not in states:
                raise ContractError(f"{flow_name}: {state} -> unknown state {target!r}")
    for t in terminal:
        if t not in states:
            raise ContractError(f"{flow_name}: terminal {t!r} not declared")
        if (states[t] or {}).get("to"):
            raise ContractError(f"{flow_name}: terminal {t!r} must have no outgoing transitions")
    for cs in cooled:
        if cs not in states:
            raise ContractError(f"{flow_name}: cooled state {cs!r} not declared")

    # Reachability: every non-initial state must be reachable from initial.
    reached = {initial}
    frontier = [initial]
    while frontier:
        cur = frontier.pop()
        for target in (states[cur] or {}).get("to", []):
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    unreachable = set(states) - reached
    if unreachable:
        raise ContractError(f"{flow_name}: unreachable states {sorted(unreachable)}")


def _validate_model_registry(c: SealedContracts) -> None:
    reg = c.model_registry
    forbidden = set(reg.get("globally_forbidden_tasks") or [])
    payload_states = set(reg.get("payload_states") or [])
    models = reg.get("models") or {}
    if not models:
        raise ContractError("model_registry.yaml defines no models")
    for name, m in models.items():
        allowed = set(m.get("allowed_tasks") or [])
        leaked = allowed & forbidden
        if leaked:
            raise ContractError(
                f"model {name}: allowed_tasks contains globally-forbidden task(s) {sorted(leaked)}"
            )
        if m.get("zone") not in VALID_MODEL_ZONES:
            raise ContractError(f"model {name}: invalid zone {m.get('zone')!r}")
        for ps in m.get("allowed_payloads") or []:
            if ps not in payload_states:
                raise ContractError(f"model {name}: unknown payload state {ps!r}")


def _validate_tool_matrix(c: SealedContracts) -> None:
    safe = set(c.tool_matrix.get("safe") or [])
    restricted = set(c.tool_matrix.get("restricted") or [])
    forbidden = set(c.tool_matrix.get("forbidden") or [])
    for a, b, an, bn in [
        (safe, restricted, "safe", "restricted"),
        (safe, forbidden, "safe", "forbidden"),
        (restricted, forbidden, "restricted", "forbidden"),
    ]:
        overlap = a & b
        if overlap:
            raise ContractError(f"tool_matrix.yaml: {an} and {bn} overlap on {sorted(overlap)}")


def _validate_no_raw_schema(
    schema: Dict[str, Any], fname: str, public_field_keys: tuple[str, ...]
) -> None:
    """No PUBLIC-FACING field name may be in the schema's own ``forbidden_fields``.

    Encrypted, human-only field lists (e.g. ``encrypted_fields``) are NOT public-facing
    and are deliberately exempt — that is exactly how the human traceability manifest
    holds an ``original_filename`` safely.
    """
    forbidden = set(schema.get("forbidden_fields") or [])
    if not forbidden:
        raise ContractError(f"{fname}: forbidden_fields must be non-empty (no-raw discipline)")
    declared: set[str] = set()
    for key in public_field_keys:
        declared |= set(schema.get(key) or [])
    leak = declared & forbidden
    if leak:
        raise ContractError(f"{fname}: public field name(s) in forbidden_fields: {sorted(leak)}")
