"""
RAGIX-Sealed — model response contract (WP §8quater, Sprint 2bis).

Every model call is normalized into a structured response, so a refusal is a routing
signal — never the security boundary. No free-text refusal parsing downstream.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ModelStatus(Enum):
    ANSWERED = "ANSWERED"
    REFUSED = "REFUSED"
    ABSTAINED = "ABSTAINED"
    FORMAT_ERROR = "FORMAT_ERROR"
    UNSAFE_OUTPUT = "UNSAFE_OUTPUT"


@dataclass(frozen=True)
class ModelResponse:
    status: ModelStatus
    content: Optional[str] = None
    refusal_reason: Optional[str] = None
    requires_human_review: bool = False
    confidence: Optional[float] = None


def normalize(obj: Any) -> ModelResponse:
    """Coerce a model's raw return (ModelResponse | dict) into a ModelResponse.

    An unparseable shape becomes FORMAT_ERROR rather than raising — the router decides.
    """
    if isinstance(obj, ModelResponse):
        return obj
    if isinstance(obj, dict):
        raw_status = str(obj.get("status", "")).upper()
        try:
            status = ModelStatus(raw_status)
        except ValueError:
            return ModelResponse(status=ModelStatus.FORMAT_ERROR, refusal_reason="unknown status")
        return ModelResponse(
            status=status,
            content=obj.get("content"),
            refusal_reason=obj.get("refusal_reason"),
            requires_human_review=bool(obj.get("requires_human_review", False)),
            confidence=obj.get("confidence"),
        )
    return ModelResponse(status=ModelStatus.FORMAT_ERROR, refusal_reason="unparseable response")
