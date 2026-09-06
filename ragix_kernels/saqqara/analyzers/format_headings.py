"""
saqqara.analyzers.format_headings — the headings a document only ever shows.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.59-K3.67 of SPEC.md.

A laid-out document often declares nothing. No outline, no heading style, no numbering — none of
the things a structure can be *read* from. It still has headings, and a reader finds them without
effort, because they are set in larger type than the text around them. The contrast is already in
what the reader recorded; what was missing is the step that reads it.

The method is comparative throughout, and that is the whole of its defence against a document it
was not designed for. Nothing here knows what a heading looks like in the abstract. It knows only
that some sizes are larger than the size most of the text is set in, that sizes close together are
one tier to a reader rather than two, and that a tier with a single line behind it is decoration
rather than a level. A document set entirely in one size therefore yields nothing at all, which is
the correct answer and not a failure to produce one.

**Two signals, tried in a declared order.** A laid-out document shows its headings by size; a
word-processing document that styles nothing shows them by weight. Size is tried first, because it
carries depth — sizes rank, so a ladder of levels can be read off them — and weight does not: a
line is bold or it is not, which yields one flat level and no more. Weight is tried only where size
found no contrast, and the trace says which signal decided and why the other did not.

The two run the same ordered gauntlet with different declared limits, and the difference is not an
oversight. A size tier is corroborated by every other line sharing that size, so a long line at a
heading size is still probably a heading; weight has no such corroboration, so it is held to a
tighter word count and refuses on punctuation a size tier tolerates.

**The unit is a line.** A reader emits one observation per text-showing operation, and a line is
commonly several of them — a heading whose first two words are in one operation and the rest in
another is one heading, not two. So assembly comes first, it is a declared step with its own trace,
and every rule below applies to assembled lines.

**Nothing is rewritten**, exactly as in `outline`: a promotion adds an inferred node beneath the
line it came from, citing that line's own position and naming this analyzer as its producer. The
observation stays what the reader saw, and a caller who disagrees still has the document as read.

For the same reason as `outline`, this runs **after** `sections` and never before. Promoted first,
the section channels would read these inferences back as if a reader had seen them.

Ways to be wrong, all counted:

  `no-heading-shaped-line`  a tier whose lines are all refused by shape: nothing to support it.
  `unsupported-tier`        below the title factor, and too few shaped lines to earn a level.
  `beyond-the-ladder`       a tier that would fall past the last level the ladder admits.

and one way to decline outright:

  `no-size-contrast`        no size stands above the body, and no weight stands out either.
  `no-weight-contrast`      the document offers weight alone, and every line carries the same.
  `declared-outline`        the document states its own structure; inference has nothing to add.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..model import Node, Provenance
from .contract import Analyzer, AnalyzerResult

__all__ = [
    "CLUSTER_GAP",
    "CONFIDENCE",
    "FORMAT_ABSTENTIONS",
    "FORMAT_CHANNEL",
    "BOLD_DOMINANT",
    "GAUNTLET_LIMITS",
    "LINE_BAND",
    "MAX_LEVELS",
    "MIN_SUPPORT",
    "MIN_TIER_GAP",
    "SHAPE_RULES",
    "SIZE_BIN",
    "TIER_DROPS",
    "TITLE_FACTOR",
    "WEIGHT_CONFIDENCE",
    "WEIGHT_LEVEL",
    "FormatHeadingsAnalyzer",
    "baseline_format_headings",
    "baseline_weight_headings",
]

#: Promotions travel under their own channel, as the outline ones do: they are
#: inferences, and keeping them nameable is what stops them passing for something
#: a reader saw.
FORMAT_CHANNEL = "format-promotion"

# The declared defaults. They are defaults, not findings: a deployment that knows
# its documents can set its own, and the values below are what the analyzer does
# when nobody has. Each says what it is for, because a bare number in a threshold
# table is the fastest way to lose the reason it was chosen.

#: Sizes are compared in bins this wide. A document that sets one heading at 15.02
#: and the next at 14.98 has set them the same; only the bin makes that sayable.
SIZE_BIN = 0.5

#: Two observations this close vertically are on the same line.
LINE_BAND = 1.0

#: Bins closer than this are one visual tier — 15.0 and 15.5 are one heading size
#: to a reader, and counting them separately halves the evidence for both.
CLUSTER_GAP = 0.5

#: A size must stand at least this far above the body size to be a tier at all.
MIN_TIER_GAP = 1.0

#: At or above this multiple of the body size, a tier is the title level whatever
#: its support: a title page has exactly one line on it by design.
TITLE_FACTOR = 1.6

#: Below the title factor, a tier needs this many heading-shaped lines to earn a
#: level. It is what separates a heading size from one decorative line.
MIN_SUPPORT = 3

#: How deep the ladder goes. Past this, further tiers are refused and counted.
MAX_LEVELS = 4

#: A line is bold when this much of it is. The whole point of a fraction: one
#: emphasised word in twenty is 0.05, and a boolean would call it bold.
BOLD_DOMINANT = 0.9

#: The gauntlet's limits, per signal. Two entries rather than one shared set,
#: because the two signals do not carry the same weight of evidence — see the
#: module docstring. `chars` may be None: the weight branch caps words only.
GAUNTLET_LIMITS = {
    "size": {"words": 16, "chars": 120, "enders": (".", ";", ",", ":")},
    "weight": {"words": 12, "chars": None, "enders": (".", ";", ",")},
}

#: Every promotion here is an inference and says so. Weight is the weaker of the
#: two, and its confidence says that rather than leaving the caller to guess.
CONFIDENCE = 0.7
WEIGHT_CONFIDENCE = 0.6

#: A weight promotion is a flat signal: bold does not rank, so it cannot say how
#: deep a heading sits. It lands at one level and never invents a ladder.
WEIGHT_LEVEL = 2

#: The shape gauntlet, in the order it is applied. The first rule that refuses a
#: line is the one recorded against it — an ordered gauntlet whose reasons were
#: reported out of order would name the least interesting objection every time.
SHAPE_RULES = ("empty", "too-long", "too-many-words", "ends-mid-sentence")

#: Why a tier did not become a level. Closed.
TIER_DROPS = ("no-heading-shaped-line", "unsupported-tier", "beyond-the-ladder")

#: Why the analyzer declined to read a document at all. Closed.
FORMAT_ABSTENTIONS = ("no-size-contrast", "no-weight-contrast", "declared-outline")


def shape_refusal(text: str, signal: str = "size") -> str | None:
    """Run the gauntlet in order; return the first rule that refuses, or None."""
    limits = GAUNTLET_LIMITS[signal]
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return "empty"
    if limits["chars"] is not None and len(collapsed) > limits["chars"]:
        return "too-long"
    if len(collapsed.split()) > limits["words"]:
        return "too-many-words"
    if collapsed.endswith(limits["enders"]):
        return "ends-mid-sentence"
    return None


def bin_size(size: float) -> float:
    """Round to the declared bin. Sizes are measurements, and measurements wobble."""
    return round(round(float(size) / SIZE_BIN) * SIZE_BIN, 2)


class Line:
    """One assembled line: its text, the size and weight it is mostly set in."""

    __slots__ = ("text", "size", "weight", "nodes", "page")

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes
        self.page = _page_of(nodes[0])
        self.text = " ".join(" ".join((n.text or "").split()) for n in nodes).strip()
        # The size a line is *in* is the size most of its characters are in: a
        # heading with a small trailing footnote mark is still a heading.
        mass: Counter[float] = Counter()
        bold = total = 0
        for node in nodes:
            length = len(node.text or "")
            declared = node.facts.get("font_size")
            if declared is None:
                declared = node.facts.get("size")        # word-processing readers
            mass[bin_size(declared or 0.0)] += length
            total += length
            bold += length * float(node.facts.get("bold_frac") or 0.0)
        self.size = _modal(mass)
        self.weight = round(bold / total, 2) if total else 0.0


def _page_of(node: Node) -> Any:
    return getattr(node.provenance.leaf, "page", None)


def _modal(mass: Counter) -> float:
    """The heaviest key; ties go to the smaller size.

    Ties are rare and the tie-break still matters: resolved by insertion order,
    the body size of a document would depend on the order its pages were read,
    which is not a property of the document.
    """
    if not mass:
        return 0.0
    best = max(mass.values())
    return min(size for size, weight in mass.items() if weight == best)


def assemble_lines(tree) -> list[Line]:
    """Group text observations into lines: same page, same vertical band, left to right.

    A reader that already hands over paragraphs — a word-processing document,
    where a paragraph is a structural object and not a position on a page —
    needs no assembly, and gets none: each of its paragraphs is one line. The
    step exists for the readers that emit placements, and pretending otherwise
    would put a spurious count in the trace.
    """
    unplaced = [
        node
        for node in tree.walk()
        if node.kind in ("paragraph", "marker")
        and node.text
        and node.facts.get("y") is None
        and (node.facts.get("bold_frac") is not None or node.facts.get("size") is not None)
    ]
    if unplaced:
        return [Line([node]) for node in unplaced]

    placed = [
        node
        for node in tree.walk()
        if node.kind == "paragraph"
        and node.facts.get("font_size") is not None
        and node.facts.get("y") is not None
    ]
    placed.sort(key=lambda n: (str(_page_of(n)), -float(n.facts["y"]),
                               float(n.facts.get("x") or 0.0)))

    lines: list[Line] = []
    current: list[Node] = []
    for node in placed:
        if current and (
            _page_of(node) != _page_of(current[0])
            or abs(float(node.facts["y"]) - float(current[0].facts["y"])) > LINE_BAND
        ):
            lines.append(Line(current))
            current = []
        current.append(node)
    if current:
        lines.append(Line(current))
    return lines


def baseline_format_headings(tree) -> list[Line]:
    """The obvious rule, kept honest: every line larger than the most common size.

    No clustering, no support, no shape. It is what the method has to beat, and it
    has to be beaten in both directions — it promotes what shape would refuse, and
    it splits one visual tier into as many levels as the document has sizes.
    """
    lines = assemble_lines(tree)
    if not lines:
        return []
    mass: Counter[float] = Counter()
    for line in lines:
        mass[line.size] += len(line.text)
    body = _modal(mass)
    return [line for line in lines if line.size > body]


def baseline_weight_headings(tree) -> list[Line]:
    """The boolean rule the readers used to make possible: any bold run at all.

    It is the baseline the fraction has to beat, and the population it fails on is
    not exotic: a paragraph with one emphasised word reports bold, and this rule
    promotes it.
    """
    return [line for line in assemble_lines(tree) if line.weight > 0]


class FormatHeadingsAnalyzer(Analyzer):
    """Promote lines set larger than the body, where a tier and a shape support it."""

    name = "format_headings"
    version = "0.1.0"

    def run(self, tree) -> AnalyzerResult:
        lines = assemble_lines(tree)
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "observations": sum(len(line.nodes) for line in lines),
            "lines": len(lines),
            "collapsed": sum(len(line.nodes) - 1 for line in lines),
            "signal": None,
            "body": None,
            "tiers": [],
            "levels": {},
            "promoted": 0,
            "dropped": 0,
            "drops": [],
            "abstained": None,
        }
        if not lines:
            trace["abstained"] = {"reason": "no-size-contrast", "signals": {"lines": 0}}
            return AnalyzerResult(tree=tree, trace=self.traced(trace))

        declared = [
            node for node in tree.walk()
            if node.kind == "heading" and node.origin == "read"
        ]
        if declared:
            # The document has stated its own structure. An inference set beside a
            # declaration cannot corroborate it — it agrees redundantly, or it
            # disagrees and casts doubt on the better evidence. So the rules run,
            # and their result is counted rather than attached: a decision not to
            # act has to be as visible as a decision to act.
            probe: dict[str, Any] = {"tiers": [], "levels": {}, "promoted": 0,
                                     "dropped": 0, "drops": [], "body": None,
                                     "signal": None}
            self._by_size(lines, probe, attach=False)
            trace["body"] = probe["body"]
            trace["abstained"] = {
                "reason": "declared-outline",
                "signals": {
                    "declared_headings": len(declared),
                    "would_have_promoted": probe["promoted"],
                },
            }
            return AnalyzerResult(tree=tree, trace=self.traced(trace))

        if self._by_size(lines, trace):
            return AnalyzerResult(tree=tree, trace=self.traced(trace))
        if self._by_weight(lines, trace):
            return AnalyzerResult(tree=tree, trace=self.traced(trace))

        # Which signal the document OFFERS, not which one happened to be non-zero:
        # a word-processing document with no bold anywhere has no weight contrast,
        # and calling that a want of size contrast would name a signal its reader
        # never supplied.
        offers_weight = any(
            node.facts.get("bold_frac") is not None
            for line in lines for node in line.nodes
        )
        trace["abstained"] = {
            "reason": "no-weight-contrast" if offers_weight else "no-size-contrast",
            "signals": {
                "body": trace["body"],
                "lines": len(lines),
                "bold_lines": sum(1 for line in lines if line.weight >= BOLD_DOMINANT),
            },
        }
        return AnalyzerResult(tree=tree, trace=self.traced(trace))

    # ---------------------------------------------------------------- by size

    def _by_size(self, lines, trace, attach: bool = True) -> bool:
        """Size first: it ranks, so it can say how deep a heading sits."""
        mass: Counter[float] = Counter()
        shaped: Counter[float] = Counter()
        for line in lines:
            mass[line.size] += len(line.text)
            if shape_refusal(line.text, "size") is None:
                shaped[line.size] += 1
        body = _modal(mass)
        trace["body"] = body

        clusters = self._cluster(sorted(mass, reverse=True), body)
        if not clusters:
            return False

        level_of = self._levels(clusters, shaped, body, trace)
        if not level_of:
            return False

        trace["signal"] = "size"
        for line in lines:
            level = level_of.get(line.size)
            if level is None:
                continue
            self._consider(line, level, "size", CONFIDENCE, trace, attach)
        return True

    # -------------------------------------------------------------- by weight

    def _by_weight(self, lines, trace) -> bool:
        """Weight second, and flat: bold does not rank, so it yields one level.

        Only where size found nothing. A document that ranks its headings by size
        has already said how they nest; reading its bold runs afterwards would add
        a second, contradictory answer to a question already settled.
        """
        candidates = [line for line in lines if line.weight >= BOLD_DOMINANT]
        if not candidates or len(candidates) == len(lines):
            # every line bold is no contrast at all, exactly as no line bold is
            return False

        trace["signal"] = "weight"
        trace["tiers"].append(
            {"sizes": [], "support": len(candidates), "level": WEIGHT_LEVEL,
             "signal": "weight"}
        )
        trace["levels"] = {str(WEIGHT_LEVEL): []}
        for line in candidates:
            self._consider(line, WEIGHT_LEVEL, "weight", WEIGHT_CONFIDENCE, trace)
        return True

    # ----------------------------------------------------------------- shared

    def _consider(self, line, level, signal, confidence, trace, attach=True) -> None:
        """One line, one level, one gauntlet: promote it or count why not.

        `attach` False counts what would have been promoted without promoting it,
        which is how a named skip reports the size of what it declined to do.
        """
        refusal = shape_refusal(line.text, signal)
        if refusal is not None:
            trace["dropped"] += 1
            trace["drops"].append(
                {"reason": refusal, "signal": signal, "size": line.size,
                 "chars": len(line.text)}
            )
            return
        if attach:
            line.nodes[0].children.append(
                self._promotion(line, level, signal, confidence)
            )
        trace["promoted"] += 1

    @staticmethod
    def _cluster(sizes: list[float], body: float) -> list[list[float]]:
        """Single linkage over the sizes above the body, largest first.

        Clustering never reaches down to the body itself: chained far enough, a
        document with a size at every half-point would join its title to its prose
        and elect one enormous tier.
        """
        clusters: list[list[float]] = []
        for size in sizes:
            if size <= body + MIN_TIER_GAP:
                continue
            if clusters and clusters[-1][-1] - size <= CLUSTER_GAP:
                clusters[-1].append(size)
            else:
                clusters.append([size])
        return clusters

    @staticmethod
    def _levels(clusters, shaped, body, trace) -> dict[float, int]:
        """Title tier by size, ladder tiers by support, both counted."""
        level_of: dict[float, int] = {}
        rank = 2
        for cluster in clusters:
            support = sum(shaped[size] for size in cluster)
            top = cluster[0]
            entry = {"sizes": list(cluster), "support": support}
            if support == 0:
                reason = "no-heading-shaped-line"
            elif top >= body * TITLE_FACTOR:
                level_of.update({size: 1 for size in cluster})
                trace["tiers"].append({**entry, "level": 1})
                continue
            elif support < MIN_SUPPORT:
                reason = "unsupported-tier"
            elif rank > MAX_LEVELS:
                reason = "beyond-the-ladder"
            else:
                level_of.update({size: rank for size in cluster})
                trace["tiers"].append({**entry, "level": rank})
                rank += 1
                continue
            trace["tiers"].append({**entry, "level": None, "reason": reason})
            trace["dropped"] += 1
            trace["drops"].append({"reason": reason, **entry})

        trace["levels"] = {
            str(level): sorted((s for s, lv in level_of.items() if lv == level), reverse=True)
            for level in sorted(set(level_of.values()))
        }
        return level_of

    @staticmethod
    def _promotion(line: Line, level: int, signal: str, confidence: float) -> Node:
        """A new inferred heading, citing the line it was promoted from.

        The provenance names this analyzer, not the reader: the words are the
        document's, the claim that they are a heading is ours, and a citation
        blurring the two would let the inference borrow a reader's authority.
        """
        source = line.nodes[0]
        return Node(
            kind="heading",
            level=level,
            text=line.text,
            provenance=Provenance(
                source_path=source.provenance.source_path,
                source_format=source.provenance.source_format,
                chain=source.provenance.chain,
                kernel="saqqara.format_headings",
                kernel_version=FormatHeadingsAnalyzer.version,
            ),
            origin="inferred",
            confidence=confidence,
            facts={
                "channel": FORMAT_CHANNEL,
                "signal": signal,
                "font_size": line.size,
                "bold_frac": line.weight,
                "segments": len(line.nodes),
                "promoted_from": source.provenance.leaf.to_dict(),
            },
        )
