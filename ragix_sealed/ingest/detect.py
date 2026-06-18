"""
RAGIX-Sealed — basic entity detection v0 (WP §8 "basic placeholderization", Sprint 2).

Deterministic, regex-based detection of high-confidence quasi-identifiers: EMAIL, PHONE,
DATE, AMOUNT, REFERENCE. This is NOT NER — names/orgs/locations require the NER layer
(Sprint 4+/2bis) and human review. v0 deliberately covers only patterns that can be
matched without a model, so it never invents entities.

Only entity types present in the active ``placeholder_schema`` are emitted.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

# Conservative, tight patterns. Precedence order matters for overlap resolution
# (more specific / less ambiguous first).
_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("REFERENCE", re.compile(r"\bn[°o]\s?\d{2,}[-/]\d{2,}\b", re.IGNORECASE)),
    ("AMOUNT", re.compile(
        r"(?:€|EUR|\$|USD)\s?\d[\d .,]*\d"
        r"|\b\d[\d .,]*\d\s?(?:€|EUR|euros?|\$|USD|dollars?)\b",
        re.IGNORECASE,
    )),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    # FR-style phone, kept tight to avoid swallowing dates/amounts.
    ("PHONE", re.compile(r"(?:\+33|0)\s?[1-9](?:[ .\-]?\d{2}){4}\b")),
]

_PRECEDENCE = [etype for etype, _ in _PATTERNS]


@dataclass(frozen=True)
class Detection:
    """A detected span. ``value`` is raw and stays INTERNAL (never crosses the boundary)."""

    entity_type: str
    start: int
    end: int
    value: str


def detect_entities(text: str, placeholder_schema: Dict[str, Any]) -> List[Detection]:
    """Detect high-confidence entities, returning non-overlapping spans sorted by start.

    Only entity types declared in ``placeholder_schema['entity_classes']`` are emitted.
    """
    classes = set((placeholder_schema or {}).get("entity_classes", {}).keys())
    found: List[Detection] = []
    for etype, pattern in _PATTERNS:
        if etype not in classes:
            continue
        for m in pattern.finditer(text):
            found.append(Detection(etype, m.start(), m.end(), m.group()))

    # Resolve overlaps: prefer earlier start, then longer span, then higher precedence.
    found.sort(key=lambda d: (d.start, -(d.end - d.start), _PRECEDENCE.index(d.entity_type)))
    accepted: List[Detection] = []
    last_end = -1
    for d in found:
        if d.start >= last_end:
            accepted.append(d)
            last_end = d.end
    accepted.sort(key=lambda d: d.start)
    return accepted
