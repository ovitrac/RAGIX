"""
RAGIX-Sealed — leak scanner v0 (WP §9, Sprint 2).

LLM-boundary egress filter. Runs INSIDE the sealed enclave (so it may legitimately see the
raw values that were placeholderized); its OUTPUT is sanitized — a verdict plus counts and
category names, never raw content.

v0 chain:
  1. known-entity exact scanner — any raw value still present verbatim ⇒ FAIL
  2. regex re-scan — any v0 detector still matches the placeholderized text ⇒ FAIL
  3. otherwise PASS

Deny-by-default: callers treat anything other than PASS as "do not release".

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .detect import detect_entities


@dataclass(frozen=True)
class LeakVerdict:
    """Sanitized leak-scan result. Carries NO raw content."""

    verdict: str  # "PASS" | "FAIL" | "UNCERTAIN"
    suspect_count: int = 0
    categories: List[str] = field(default_factory=list)  # detector types still matching

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "suspect_count": self.suspect_count,
            "categories": sorted(self.categories),
        }


def scan(
    placeholderized_text: str,
    raw_values: Iterable[str],
    placeholder_schema: Dict[str, Any],
) -> LeakVerdict:
    """Scan placeholderized text for residual raw values or detectable quasi-identifiers."""
    suspects = 0

    # 1. Known-entity exact scanner.
    for raw in raw_values:
        if raw and raw in placeholderized_text:
            suspects += 1
    if suspects:
        return LeakVerdict(verdict="FAIL", suspect_count=suspects, categories=["known_entity"])

    # 2. Regex re-scan — nothing the detector recognises should remain.
    residual = detect_entities(placeholderized_text, placeholder_schema)
    if residual:
        cats = sorted({d.entity_type for d in residual})
        return LeakVerdict(verdict="FAIL", suspect_count=len(residual), categories=cats)

    return LeakVerdict(verdict="PASS")
