"""
saqqara.analyzers.contract — what an analyzer is, and how it is allowed to fail.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K3 in SPEC.md.

An analyzer takes a tree and returns a tree plus a trace. Two rules shape everything below.

**Ordered hard rules, never a blended score.** Recognition applies declared rules in a declared
order, and the trace names every signal it looked at. A single number standing in for several
kinds of evidence is convenient and it hides exactly the disagreements worth seeing: a layout that
two rules read differently is interesting, and a weighted average of the two is not.

**Abstention is an object.** When the evidence does not decide, the analyzer says so, in a word
drawn from a closed list, and the abstention travels onward attached to the thing it is about. It
is never a default value, never a silent fallback to a simpler answer, and never absence. The
reason vocabulary is frozen precisely so that "I could not tell" cannot quietly become "there was
nothing there".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..model import Tree

__all__ = [
    "ABSTENTION_REASONS",
    "CAPTION_ABSTENTIONS",
    "Abstention",
    "Analyzer",
    "AnalyzerResult",
    "TYPING_REASONS",
]

#: Why a header band could not be read. Closed: a reason outside this list is a bug,
#: not a new case, and the gate refuses it.
ABSTENTION_REASONS = (
    "no-populated-region",
    "no-body-rows",
    "uniform-block",
    "band-too-deep",
    "non-laminar-band-merges",
    "no-column-evidence-within-cap",
)

#: Why a block could not be typed. Closed for the same reason.
TYPING_REASONS = (
    "single-column-with-header-evidence",
    "no-value-evidence",
)

#: Why a figure could not be given a caption. Closed for the same reason: a
#: reason outside this list is a bug, not a new case.
#:
#: `candidate-already-bound` is the one that is easy to leave out, and it is the
#: one that matters most: a figure whose only candidate went to a nearer figure
#: has NOT been examined and found wanting, and reporting it as though no line
#: was ever near would lose the competition that actually decided it.
CAPTION_ABSTENTIONS = (
    "ambiguous-candidates",
    "no-candidate-within-gap",
    "candidate-already-bound",
)


@dataclass(frozen=True)
class Abstention:
    """A decision not to decide, carried by the object it is about."""

    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reason not in ABSTENTION_REASONS + TYPING_REASONS + CAPTION_ABSTENTIONS:
            raise ValueError(
                f"abstention reason {self.reason!r} is outside the frozen vocabulary"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "signals": dict(self.signals)}


@dataclass
class AnalyzerResult:
    """A tree, and the decomposed account of how it was reached."""

    tree: Tree
    trace: dict[str, Any]


class Analyzer:
    """One recognition step. Composable, and testable on its own."""

    name: str = ""
    version: str = "0.0.0"

    def run(self, tree: Tree) -> AnalyzerResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Analyzer {self.name}@{self.version}>"
