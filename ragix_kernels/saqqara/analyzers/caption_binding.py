"""
saqqara.analyzers.caption_binding — which line, if any, captions which figure.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-28

Specified by K6.11, K6.12 and K6.13 in SPEC.md.

A figure and the line beneath it are two observations. That the second describes the first is a
**judgement about layout, and layout lies**: the same arrangement carries a caption on one page and
an unrelated sentence on the next. So a binding is never a reader's fact. It is a new node, marked
inferred, carrying the name of the rule that produced it and that rule's declared confidence, and
it cites the paragraph it was read from so the words keep their own provenance.

**Ordered hard rules, never a blended score.** Three rules in a fixed order, each naming itself.
The first that binds is the one recorded. No rule contributes a fraction of a shared number: a
layout two rules read differently is interesting, and an average of the two is not.

**Order breaks distance, not symmetry.** Where the two best candidates sit at the *same* distance on
opposite sides of a figure, the rule order is the only thing that could choose between them — and an
order is a preference, not evidence. The analyzer abstains instead, in a word from a closed list,
attached to the figure it is about.

**Assignment is set-level.** A line captions at most one figure. Candidates across the whole page
are ranked once and taken best-first, so the strongest evidence wins wherever it is; a figure whose
only candidate went to a nearer one says exactly that, and is not reported as though no line had
ever been near it.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..model import Node, Provenance
from .contract import CAPTION_ABSTENTIONS, Abstention, Analyzer, AnalyzerResult
from .format_headings import assemble_lines

__all__ = [
    "CAPTION_ABSTENTIONS",
    "CAPTION_CHANNEL",
    "CAPTION_GAP",
    "CAPTION_RULES",
    "BINDING_FACTS",
    "CaptionBindingAnalyzer",
]

#: Bindings travel under their own channel, beside the promotion channels: they
#: are inferences, and a channel is what keeps them from being read back as
#: observations by anything downstream.
CAPTION_CHANNEL = "caption-binding"

#: How far from a figure a line may sit and still caption it, in multiples of
#: that line's own height. Declared, not fitted: calibration is private to the
#: corpus that owns it and no corpus-fitted number is published as a gate.
CAPTION_GAP = 1.5

#: The rules, in the order they are tried, each with the confidence it confers.
#: A confidence is a property of the RULE, never of the individual binding, so a
#: number in a trace can always be traced back to the rule that chose it.
CAPTION_RULES: tuple[tuple[str, float], ...] = (
    ("caption-below-overlapping", 0.9),
    ("caption-below-offset", 0.8),
    ("caption-above-overlapping", 0.7),
)

#: What a binding records.
BINDING_FACTS = ("caption_of", "captioned_by", "distance", "rule", "confidence")

#: Distances closer together than this are the same distance. Page coordinates
#: are floats and an exact comparison would make a tie depend on arithmetic.
_TIE = 0.5


class _Candidate:
    """One (figure, line) pair a rule accepts, and why."""

    __slots__ = ("figure", "line", "rule_index", "distance", "below")

    def __init__(self, figure, line, rule_index: int, distance: float, below: bool) -> None:
        self.figure = figure
        self.line = line
        self.rule_index = rule_index
        self.distance = distance
        self.below = below

    @property
    def rule(self) -> str:
        return CAPTION_RULES[self.rule_index][0]

    @property
    def confidence(self) -> float:
        return CAPTION_RULES[self.rule_index][1]

    def key(self) -> tuple:
        return (self.rule_index, self.distance)


def _page_of(node: Node) -> Any:
    return getattr(node.provenance.leaf, "page", None)


def _box(figure: Node) -> tuple[float, float, float, float] | None:
    """A figure's placement as (left, bottom, right, top), or None if it has none."""
    facts = figure.facts
    try:
        left, bottom = float(facts["x"]), float(facts["y"])
        width, height = float(facts["w"]), float(facts["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return (left, bottom, left + width, bottom + height)


def _line_x(line) -> float | None:
    """Where a line starts.

    Not where it ends: a text record carries one origin and no width, so the
    extent of a line is not among the facts this kernel has. Overlap is
    therefore tested on the caption's ORIGIN falling inside the figure's
    horizontal span, which is narrower than the fraction-of-width test the
    contract's geometry section describes. Closing that gap is a change to what
    a reader emits, not to how this analyzer reasons, and it is named rather
    than quietly redefined.
    """
    xs = [float(n.facts["x"]) for n in line.nodes if n.facts.get("x") is not None]
    return min(xs) if xs else None


class CaptionBindingAnalyzer(Analyzer):
    """Bind captions to figures, or say why not.

    Opt-in, like the other inference steps: a caller that wants bindings asks for
    them. A default pipeline that bound captions everywhere would put this
    module's judgements into every tree, including the trees used to measure
    whether binding is any good.
    """

    name = "saqqara.caption_binding"
    version = "0.1.0"

    def run(self, tree) -> AnalyzerResult:
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "figures": 0,
            "candidates": 0,
            "bound": 0,
            "by_rule": {},
            "abstained": {},
        }

        figures = [n for n in tree.walk() if n.kind == "figure"]
        trace["figures"] = len(figures)
        if not figures:
            return AnalyzerResult(tree=tree, trace=trace)

        lines_by_page: dict[Any, list] = {}
        for line in assemble_lines(tree):
            lines_by_page.setdefault(line.page, []).append(line)

        candidates: list[_Candidate] = []
        for figure in figures:
            candidates.extend(self._candidates(figure, lines_by_page))
        trace["candidates"] = len(candidates)

        by_figure: dict[int, list[_Candidate]] = {}
        for candidate in candidates:
            by_figure.setdefault(id(candidate.figure), []).append(candidate)
        for group in by_figure.values():
            group.sort(key=_Candidate.key)

        # Symmetry is decided before anything is assigned: a figure whose two best
        # candidates are equally close on opposite sides is not a figure whose
        # caption a rule order should pick, and its lines stay free for others.
        ambiguous = set()
        for key, group in by_figure.items():
            if len(group) > 1 and self._symmetric(group[0], group[1]):
                ambiguous.add(key)
                self._abstain(group[0].figure, "ambiguous-candidates", trace,
                              {"distance": group[0].distance,
                               "rules": [group[0].rule, group[1].rule]})

        taken: set[int] = set()
        bound: set[int] = set()
        for candidate in sorted(candidates, key=_Candidate.key):
            if id(candidate.figure) in ambiguous or id(candidate.figure) in bound:
                continue
            if id(candidate.line) in taken:
                continue
            self._bind(candidate, trace)
            taken.add(id(candidate.line))
            bound.add(id(candidate.figure))

        for figure in figures:
            key = id(figure)
            if key in bound or key in ambiguous:
                continue
            group = by_figure.get(key, [])
            if not group:
                self._abstain(figure, "no-candidate-within-gap", trace,
                              {"candidates": 0})
            else:
                # Every line it could have taken went to a nearer figure. That is
                # a different statement from "nothing was near", and the closed
                # vocabulary exists so the two cannot be confused.
                self._abstain(figure, "candidate-already-bound", trace,
                              {"candidates": len(group),
                               "nearest": group[0].distance})

        return AnalyzerResult(tree=tree, trace=trace)

    # ------------------------------------------------------------- candidates

    def _candidates(self, figure, lines_by_page) -> Iterator[_Candidate]:
        box = _box(figure)
        if box is None:
            return
        left, bottom, right, top = box
        for line in lines_by_page.get(_page_of(figure), ()):
            height = float(line.size or 0.0)
            if height <= 0:
                continue
            x = _line_x(line)
            if x is None:
                continue
            y = float(line.nodes[0].facts["y"])
            overlapping = left <= x <= right
            reach = CAPTION_GAP * height

            if y <= bottom and bottom - y <= reach:
                index = 0 if overlapping else 1
                yield _Candidate(figure, line, index, bottom - y, True)
            elif y >= top and y - top <= reach and overlapping:
                # There is no above-offset rule. A line neither beneath a figure
                # nor over it is not a candidate at all, and inventing a fourth
                # rule to catch it would be adding a claim no evidence asked for.
                yield _Candidate(figure, line, 2, y - top, False)

    @staticmethod
    def _symmetric(first: _Candidate, second: _Candidate) -> bool:
        return abs(first.distance - second.distance) <= _TIE and first.below != second.below

    # ---------------------------------------------------------------- outcome

    def _bind(self, candidate: _Candidate, trace) -> None:
        figure = candidate.figure
        source = candidate.line.nodes[0]
        figure.children.append(
            Node(
                kind="caption",
                text=candidate.line.text,
                provenance=Provenance(
                    source_path=source.provenance.source_path,
                    source_format=source.provenance.source_format,
                    chain=source.provenance.chain,
                    kernel=self.name,
                    kernel_version=self.version,
                ),
                origin="inferred",
                confidence=candidate.confidence,
                facts={
                    "channel": CAPTION_CHANNEL,
                    "caption_of": figure.provenance.leaf.to_dict(),
                    "captioned_by": source.provenance.leaf.to_dict(),
                    "distance": round(candidate.distance, 2),
                    "rule": candidate.rule,
                    "confidence": candidate.confidence,
                },
            )
        )
        trace["bound"] += 1
        trace["by_rule"][candidate.rule] = trace["by_rule"].get(candidate.rule, 0) + 1

    @staticmethod
    def _abstain(figure: Node, reason: str, trace, signals: dict) -> None:
        figure.facts["caption_abstention"] = Abstention(reason=reason, signals=signals)
        trace["abstained"][reason] = trace["abstained"].get(reason, 0) + 1
