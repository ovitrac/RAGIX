"""tender.agreement — inter-annotator / inter-channel agreement.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Cohen's kappa on a closed nominal vocabulary (gate GR3-2). Kept deliberately
small and dependency-free: agreement is reported, disagreements go to review,
they are NEVER averaged away.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Cohen's kappa. None when undefined (empty input or degenerate chance)."""
    if len(a) != len(b):
        raise ValueError("kappa on sequences of different length")
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return None                     # every label identical: chance = 1
    return (po - pe) / (1 - pe)


def agreement_table(a: Sequence[str], b: Sequence[str]) -> dict:
    """Raw agreement plus the disagreement pairs, for human review."""
    pairs = Counter((x, y) for x, y in zip(a, b) if x != y)
    return {"n": len(a),
            "observed_agreement": (sum(1 for x, y in zip(a, b) if x == y)
                                   / len(a)) if a else None,
            "kappa": cohen_kappa(a, b),
            "disagreements": [{"a": x, "b": y, "count": c}
                              for (x, y), c in pairs.most_common()]}
