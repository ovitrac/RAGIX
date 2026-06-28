"""ragix_chat — a small, mode-agnostic chat engine over a pluggable document backend.

CLEAR mode (plaintext, development/quality) and SEALED mode (confidential) share one engine
and one conversation loop; only the backend differs. The system prompt is customizable.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-28
"""
from .engine import ChatEngine, DEFAULT_SYSTEM
from .backends import ContextBackend, ClearBackend

__all__ = ["ChatEngine", "DEFAULT_SYSTEM", "ContextBackend", "ClearBackend"]
