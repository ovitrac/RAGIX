"""
RAGIX-Sealed — policy-first model cascade (WP §8quater, Sprint 2bis).

Routes a task to eligible sealed-zone models in policy order. The authority is the
policy + leak scanner + the registry, never a model:

    RAGIX policy -> model eligibility -> model call -> output post-check -> release/block

A refusal/abstention routes to the next eligible model (if the task is allowed); an
unsafe output blocks with no automatic fallback. Models are injected as callables, so
this is fully testable without a live model server. The audit trail carries no raw
content (model id, status, decision only).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .contracts import ModelResponse, ModelStatus, normalize

# A model is any callable (task, payload_text) -> ModelResponse | dict.
ModelCallable = Callable[[str, str], Any]
# A prober answers whether a model accepts an abstract task class (no raw content sent).
Prober = Callable[[str, str], bool]
# A post-check returns True if model output is safe to release (e.g. leak scan PASS).
PostCheck = Callable[[str], bool]

VALID_ZONES = {"INTERNAL", "INTERNAL_WORKER"}


class RouterDecision(Enum):
    RELEASED = "RELEASED"
    BLOCKED = "BLOCKED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    NO_ELIGIBLE_MODEL = "NO_ELIGIBLE_MODEL"


@dataclass(frozen=True)
class RouterResult:
    decision: RouterDecision
    content: Optional[str] = None
    model_id: Optional[str] = None
    reason: Optional[str] = None
    audit: List[Dict[str, Any]] = field(default_factory=list)  # sanitized, no raw content


class ModelRouter:
    """Selects and runs sealed-zone models per the model_registry contract."""

    def __init__(
        self,
        contracts: Any,
        models: Dict[str, ModelCallable],
        post_check: Optional[PostCheck] = None,
        prober: Optional[Prober] = None,
    ) -> None:
        self._registry = contracts.model_registry
        self._models = models
        # Default post-check is conservative: with no leak scanner wired, do not release.
        self._post_check = post_check if post_check is not None else (lambda _content: False)
        self._prober = prober

    def _candidates(self, task: str, payload_state: str) -> List[str]:
        """Eligible model names in registry (cascade) order."""
        out: List[str] = []
        for name, m in (self._registry.get("models") or {}).items():
            if m.get("zone") not in VALID_ZONES:
                continue
            if task not in (m.get("allowed_tasks") or []):
                continue
            if payload_state not in (m.get("allowed_payloads") or []):
                continue
            out.append(name)
        return out

    def run(self, task: str, payload_state: str, payload_text: str) -> RouterResult:
        audit: List[Dict[str, Any]] = []

        # 1. Policy gate: globally-forbidden tasks never run.
        if task in set(self._registry.get("globally_forbidden_tasks") or []):
            return RouterResult(RouterDecision.BLOCKED, reason="task globally forbidden", audit=audit)

        candidates = self._candidates(task, payload_state)
        if not candidates:
            return RouterResult(
                RouterDecision.NO_ELIGIBLE_MODEL,
                reason=f"no model allows task={task!r} for payload={payload_state!r}",
                audit=audit,
            )

        # 2. Cascade.
        for name in candidates:
            if name not in self._models:
                audit.append({"model_id": name, "event": "NOT_WIRED"})
                continue
            if self._prober is not None and not self._prober(name, task):
                audit.append({"model_id": name, "event": "PROBE_REFUSED"})
                continue

            resp: ModelResponse = normalize(self._models[name](task, payload_text))
            audit.append({"model_id": name, "event": "ATTEMPT", "status": resp.status.value})

            if resp.status is ModelStatus.ANSWERED:
                content = resp.content or ""
                if not self._post_check(content):
                    audit.append({"model_id": name, "event": "POST_CHECK_FAILED", "decision": "block"})
                    return RouterResult(
                        RouterDecision.BLOCKED, model_id=name,
                        reason="output failed post-check / leak scan", audit=audit,
                    )
                audit.append({"model_id": name, "event": "RELEASED", "decision": "allow"})
                return RouterResult(RouterDecision.RELEASED, content=content, model_id=name, audit=audit)

            if resp.status is ModelStatus.UNSAFE_OUTPUT:
                audit.append({"model_id": name, "event": "UNSAFE", "decision": "block"})
                return RouterResult(
                    RouterDecision.BLOCKED, model_id=name,
                    reason="unsafe output; no automatic fallback", audit=audit,
                )
            # REFUSED / ABSTAINED / FORMAT_ERROR -> try next eligible model.

        return RouterResult(
            RouterDecision.HUMAN_REVIEW,
            reason="no eligible model completed the allowed task",
            audit=audit,
        )
