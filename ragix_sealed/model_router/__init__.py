"""
RAGIX-Sealed model router (WP §8quater, Sprint 2bis).

Policy-first cascade over sealed-zone models. A model refusal is a routing signal, not
the security boundary. Models are injected, so the router is testable without a live
model server.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .contracts import ModelResponse, ModelStatus, normalize
from .router import ModelRouter, RouterDecision, RouterResult

__all__ = [
    "ModelResponse",
    "ModelStatus",
    "normalize",
    "ModelRouter",
    "RouterDecision",
    "RouterResult",
]
