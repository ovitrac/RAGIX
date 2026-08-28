"""
saqqara.analyzers.outline — a numbered line is not a heading because it is numbered.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.49-K3.53 of SPEC.md.

Plenty of documents number their headings and style nothing. Reading those numbers is the only way
to recover their structure, and reading them naively is how a contract's list of obligations
becomes a five-level table of contents.

The rule that separates the two is not about the label. It is about the **walk**. A label promotes
when it takes a legal step from the one before it — a descent, a sibling, or a return to a level
already open — so that the labels together trace a shape a document could actually have. A list of
numbered clauses takes only sibling steps and never descends; it is a list, and it stays one.

Where a chain is flat, style can still corroborate it: if every one of those lines is also styled
as a heading, two independent signals agree and the chain promotes. One signal alone does not.

**Nothing here is rewritten.** A promotion adds a node; it never turns a paragraph into a heading
in place. The paragraph is what a reader observed and it stays exactly that, while the promotion
sits beneath it as a separate, inferred node citing the position it came from and naming this
analyzer as its producer. Two consequences follow, and both are the point: an observation is never
silently replaced by an inference, and a caller who disagrees with the promotion still has the
document as it was read.

For the same reason this analyzer runs **after** `sections`, never before. Run first, it would turn
every numbered line into a heading, and the eight reader-fed channels would then be reading this
module's conclusions back to themselves — the numbered-heading channel would go quiet and the
native-outline channel would fill with our own guesses. A caller who wants promotions inside the
section ancestry re-runs `sections` explicitly afterwards, and the trace records both passes.

Three ways to be wrong, all of them counted:

  `illegal-step`         a label that does not fit the walk. Skipped; the walk continues without it.
  `chain-too-short`      too few labels to be evidence of anything.
  `flat-uncorroborated`  a flat chain with nothing but its own numbering to recommend it.
"""

from __future__ import annotations

import re
from typing import Any

from ..model import Node, Provenance, kind_registry
from .contract import Analyzer, AnalyzerResult

__all__ = [
    "MIN_CHAIN",
    "OUTLINE_CHANNEL",
    "OUTLINE_DROPS",
    "OutlineAnalyzer",
    "parse_label",
    "validate_walk",
]

#: Promotions travel under their own channel. They are inferences, and keeping
#: them nameable is what stops them being mistaken for something a reader saw.
OUTLINE_CHANNEL = "outline-promotion"

#: Fewer promoted labels than this is not evidence of an outline.
MIN_CHAIN = 3

#: What counts as bold for corroboration. A fraction, not a flag: a line with
#: one emphasised word is not styled as a heading, and the boolean this rule
#: used to read could not say so.
BOLD_DOMINANT = 0.9

#: Why a label did not become a heading. Closed.
OUTLINE_DROPS = ("illegal-step", "chain-too-short", "flat-uncorroborated")

# Two shapes, and the difference matters. A multi-part label may drop its final
# separator — documents write both "1.2.3 Scope" and "1.2.3. Scope" — because the
# internal dots already mark it as a label. A single-part label may not: without a
# trailing separator, "Le contexte" would parse as label "Le" over title
# "contexte", and every prose line in the document would become a heading.
_LABEL = re.compile(
    r"^\s*("
    r"(?:[0-9A-Za-z]+(?:[.)][0-9A-Za-z]+)+[.)]?)"     # 1.2.3  or  1.2.3.
    r"|(?:[0-9A-Za-z]+[.)])"                          # 2.  2)  B.
    r")\s+(\S.*)$"
)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
          "viii": 8, "ix": 9, "x": 10}


def _component(token: str) -> tuple[str, int] | None:
    if token.isdigit():
        return "num", int(token)
    lowered = token.lower()
    if lowered in _ROMAN:
        # Roman wins over letter: "I." opens a part, it is not item nine.
        return "rom", _ROMAN[lowered]
    if len(token) == 1 and token.isalpha():
        return "let", ord(token.upper()) - ord("A") + 1
    return None


def parse_label(text: str) -> tuple[str, tuple[int, ...], str] | None:
    """`1.2 Scope` -> ('num', (1, 2), 'Scope'). The kind is the first component's."""
    match = _LABEL.match(text or "")
    if not match:
        return None
    raw, title = match.groups()
    parts = [p for p in re.split(r"[.)]", raw) if p]
    parsed = [_component(part) for part in parts]
    if not parsed or any(component is None for component in parsed):
        return None
    return parsed[0][0], tuple(value for _, value in parsed), title.strip()


def _legal(previous: tuple[int, ...] | None, current: tuple[int, ...]) -> bool:
    """A descent, a sibling, or a return to a level already open."""
    if previous is None:
        return current[-1] == 1 and len(current) == 1
    if len(current) == len(previous) + 1 and current[:-1] == previous:
        return current[-1] == 1                                   # descent
    if len(current) == len(previous) and current[:-1] == previous[:-1]:
        return current[-1] == previous[-1] + 1                    # sibling
    if len(current) < len(previous):
        depth = len(current)
        return (
            current[:-1] == previous[: depth - 1]
            and current[-1] == previous[depth - 1] + 1
        )                                                          # ascent
    return False


def validate_walk(steps: list[tuple[int, ...]]) -> list[int]:
    """Indices of the steps that fit. An illegal one is skipped, not fatal."""
    kept: list[int] = []
    previous: tuple[int, ...] | None = None
    for index, step in enumerate(steps):
        if _legal(previous, step):
            kept.append(index)
            previous = step
    return kept


class OutlineAnalyzer(Analyzer):
    """Promote typed labels to headings where the walk supports it."""

    name = "outline"
    version = "0.1.0"

    def run(self, tree) -> AnalyzerResult:
        candidates = []
        for node in tree.walk():
            if node.kind != "paragraph" or not node.text:
                continue
            parsed = parse_label(node.text)
            if parsed is not None:
                candidates.append((node, parsed))

        walks = self._split_walks(candidates)
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "candidates": len(candidates),
            "walks": 0,
            "promoted": 0,
            "dropped": 0,
            "drops": [],
        }

        for walk in walks:
            self._promote(walk, trace)

        return AnalyzerResult(tree=tree, trace=trace)

    @staticmethod
    def _split_walks(candidates):
        """A change of label kind opens a new walk: an annex restarts, it does not break."""
        walks: list[list] = []
        current: list = []
        kind = None
        for node, parsed in candidates:
            if kind is not None and parsed[0] != kind:
                walks.append(current)
                current = []
            kind = parsed[0]
            current.append((node, parsed))
        if current:
            walks.append(current)
        return walks

    def _promote(self, walk, trace) -> None:
        steps = [parsed[1] for _, parsed in walk]
        kept = validate_walk(steps)

        for index, (node, parsed) in enumerate(walk):
            if index not in kept:
                trace["dropped"] += 1
                trace["drops"].append(
                    {"reason": "illegal-step", "label": node.text, "step": list(parsed[1])}
                )

        surviving = [walk[i] for i in kept]
        if len(surviving) < MIN_CHAIN:
            self._reject(surviving, "chain-too-short", trace)
            return

        descends = any(len(parsed[1]) > 1 for _, parsed in surviving)
        corroborated = all(
            (node.facts.get("bold_frac") or 0.0) >= BOLD_DOMINANT
            for node, _ in surviving
        )
        if not descends and not corroborated:
            # A flat chain with nothing but its own numbering: a list, not an outline.
            self._reject(surviving, "flat-uncorroborated", trace)
            return

        trace["walks"] += 1
        for node, (_, step, title) in surviving:
            node.children.append(self._promotion(node, step, title, descends))
            trace["promoted"] += 1

    @staticmethod
    def _promotion(source: Node, step: tuple[int, ...], title: str, descends: bool) -> Node:
        """A new inferred heading, citing the paragraph it was promoted from.

        Its provenance names THIS analyzer as producer, not the reader: the words
        came from the document, the claim that they are a heading came from here,
        and a citation that blurred the two would let an inference borrow a
        reader's authority.
        """
        return Node(
            kind="heading",
            level=len(step),
            text=title,
            provenance=Provenance(
                source_path=source.provenance.source_path,
                source_format=source.provenance.source_format,
                chain=source.provenance.chain,
                kernel="saqqara.outline",
                kernel_version=OutlineAnalyzer.version,
            ),
            origin="inferred",
            confidence=0.9 if descends else 0.7,
            facts={
                "channel": OUTLINE_CHANNEL,
                "outline_step": list(step),
                "promoted_from": source.provenance.leaf.to_dict(),
            },
        )

    @staticmethod
    def _reject(walk, reason, trace) -> None:
        for node, parsed in walk:
            trace["dropped"] += 1
            trace["drops"].append(
                {"reason": reason, "label": node.text, "step": list(parsed[1])}
            )
