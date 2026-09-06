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
from typing import Any, Optional

from ..model import Tree

__all__ = [
    "ABSTENTION_REASONS",
    "CAPTION_ABSTENTIONS",
    "Abstention",
    "Analyzer",
    "AnalyzerResult",
    "ABSTENTION_KEYS",
    "ABSTENTION_SOURCES",
    "AbstentionSource",
    "TREE_ABSTENTION_SOURCES",
    "tree_abstention_records",
    "TYPING_REASONS",
    "abstention_records",
    "reports_abstention",
    "count_reported",
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
    """One recognition step. Composable, and testable on its own.

    **Options are declared, given at construction, and recorded.** `run(tree)` is
    unchanged: an analyzer is still a function of a tree, and what it was
    configured with is fixed before it runs. Before this, a rule like the header
    band's depth cap was a module constant with no way for a run to state it and
    no way for a reader to know what a run used — the number could be found only by
    reading the source of the version that produced the tree.

    `DEFAULTS` sits beside the analyzer that reads it, so the value and the code
    that uses it are read together. An option the analyzer does not declare is a
    **refusal, not a hint**: a mistyped key that silently does nothing is the
    defect this package has met three times in one day.
    """

    name: str = ""
    version: str = "0.0.0"

    #: Declared options and their defaults. Empty means the analyzer takes none.
    DEFAULTS: dict[str, Any] = {}

    def __init__(self, options: Optional[dict[str, Any]] = None) -> None:
        unknown = sorted(set(options or {}) - set(self.DEFAULTS))
        if unknown:
            raise ValueError(
                f"{self.name or type(self).__name__} does not take {unknown}; "
                f"it declares {sorted(self.DEFAULTS) or 'no options'}"
            )
        self.options = {**self.DEFAULTS, **(options or {})}

    def traced(self, trace: dict[str, Any]) -> dict[str, Any]:
        """The trace with the options this analyzer actually ran with.

        Defaults included, never only what a manifest overrode: a run that does not
        say what it ran with cannot be compared with another run.
        """
        return {**trace, "options": dict(self.options)}

    def run(self, tree: Tree) -> AnalyzerResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Analyzer {self.name}@{self.version}>"


# --------------------------------------------------------------- counting them

#: The shapes an analyzer may report a count in. Closed, and the reason it is a
#: list rather than a convention is measured: `saqqara_run.summarize` cast every
#: one of them with `int(...)`, which is right for two of the four and raises for
#: a third — silently, until a document actually abstains.
REPORT_SHAPES = "None, an int, a list of records, a histogram of reason -> count, or one record"


def count_reported(value: Any) -> int:
    """How many things a trace field reports, whatever shape its producer chose.

    An analyzer reports what did not decide in the shape that suits what it saw,
    and four shapes are in use across this package — all of them legitimate, none
    of them interchangeable with the others:

    ================  ==========================================  ==============
    shape             producer                                    counts as
    ================  ==========================================  ==============
    ``None``          `format_headings`, nothing to report        0
    ``int``           `header_bands`, a running tally             itself
    ``list``          `grid_tables`, one record per table         its length
    ``dict[str,int]`` `caption_binding`, reason -> how many       the sum
    ``{}``            `caption_binding` with nothing to report    0
    ``dict``          `format_headings`, one abstention record    1
    ================  ==========================================  ==============

    A histogram and a single record are both mappings, so they are told apart by
    what they hold rather than by who wrote them: every value an int means the
    mapping counts things; anything else means the mapping *is* the thing.

    Two of those producers are in `PIPELINE` and two are run apart from it, so a
    run's traces carry the int and the list today. The counter takes all four
    because which analyzers the pipeline runs is a decision that has changed
    before, and a counter that only handles the current membership fails on the
    edit that changes it rather than in the gate.

    **Anything else raises**, with the type named. A fifth shape is a decision
    about how abstention is reported, and it is registered here in the same edit
    as the producer that introduces it — the alternative is a count that quietly
    means something else, which is the failure this function exists to end rather
    than to move somewhere quieter.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        # bool is an int in Python and would count as 1. Nothing reports a count
        # as a flag today, and a flag is not a count, so refuse rather than agree.
        raise TypeError(f"a report is a count, not a flag: got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        if not value:
            # `caption_binding` starts its histogram empty and a document that
            # binds every caption leaves it that way. An empty mapping reports
            # nothing; counting it as one record would put an abstention into
            # every document that had none.
            return 0
        values = list(value.values())
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return sum(values)
        return 1
    raise TypeError(
        f"a trace reported a count as {type(value).__name__}; the declared shapes "
        f"are {REPORT_SHAPES}"
    )


# ------------------------------------------------------- the abstention register


@dataclass(frozen=True)
class AbstentionSource:
    """Where one analyzer keeps its abstentions, and how one of its records reads.

    Declared per analyzer, in one place. The alternative — a register that
    searches traces for likely key names — loses an analyzer silently the day it
    renames one, which is the failure mode this whole register exists to end.
    """

    #: trace key holding the records (or, for `caption_binding`, the histogram)
    records: str
    #: record key naming the reason; None where the shape carries reasons as keys
    reason: str | None = None
    #: record keys that address the thing abstained on; empty = document-level
    locator: tuple[str, ...] = ()
    #: trace key holding the count when the producer keeps it apart from its records
    tally: str | None = None
    #: record key holding the signals, where the producer nests them under one;
    #: None means everything the record holds beside reason and locator IS the signals
    signals: str | None = None


#: One entry per analyzer that reports an abstention. Measured 2026-09-05, not
#: assumed: `header_bands` keeps a tally in `abstained` and its records in
#: `abstentions`; `grid_tables` puts the records in `abstained` itself;
#: `format_headings` reports one record or None; `caption_binding` keeps a
#: histogram and no records, so its rows address nothing and say so.
#: The trace keys an analyzer says "I abstained" with. Two names, because two are
#: in use: a producer that keeps a tally writes `abstained` and its records under
#: `abstentions`, and one that keeps only records writes them under `abstained`.
#: A trace carrying neither is not abstaining — `builder` and `tables` report only
#: what they DROPPED, which is a different fact with its own count.
ABSTENTION_KEYS = ("abstained", "abstentions")


def reports_abstention(trace: dict[str, Any]) -> bool:
    """Whether this trace says anything about having abstained."""
    return any(key in trace for key in ABSTENTION_KEYS)


def _text_layer_abstentions(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Pages the reader could not read, from the facts the reader already wrote.

    The PDF adapter declares `has_text` and `needs_ocr` per page and counts a
    `no-text-layer` skip (K6.19) — but the count lives on the adapter, without the
    page it is about, and the register never saw it: its sources all read analyzer
    traces, and this producer is a reader. Five documents of the demo corpus are
    image-only, yield no chunk and no vector, and appeared in no register at all.

    Read from the tree rather than from the adapter's counter because the tree is
    what the run carries: the page node holds the locator and the signals, so the
    record can say **which** page, and how much ink was on it.
    """
    records: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        if node.get("kind") == "page":
            facts = node.get("facts") or {}
            if facts.get("has_text") is False:
                page = (node.get("provenance") or {}).get("chain") or [{}]
                records.append({
                    "analyzer": "saqqara.text_layer",
                    "locator": {"page": page[-1].get("page")},
                    "reason": "no-text-layer",
                    "signals": {"image_count": facts.get("image_count"),
                                "needs_ocr": facts.get("needs_ocr"),
                                "width": facts.get("width"),
                                "height": facts.get("height")},
                    "count": 1,
                })
        for child in node.get("children") or ():
            visit(child)

    visit(tree.get("root") or {})
    return records


#: Producers that leave an abstention in the **tree** rather than in a trace. The
#: register knows both kinds or it is not a register: a reader that declines a page
#: abstains exactly as an analyzer that declines a block does, and listing only one
#: of them makes the claim "everything is listed" false in silence.
TREE_ABSTENTION_SOURCES = {
    "saqqara.text_layer": _text_layer_abstentions,
}


def tree_abstention_records(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tree-borne abstention of one document, from every declared source."""
    records: list[dict[str, Any]] = []
    for source in TREE_ABSTENTION_SOURCES.values():
        records.extend(source(tree))
    return records


ABSTENTION_SOURCES = {
    "grid_tables": AbstentionSource(
        records="abstained", reason="rule", locator=("flow", "table_index")),
    "header_bands": AbstentionSource(
        records="abstentions", reason="reason", locator=("range",), tally="abstained",
        signals="signals"),
    "format_headings": AbstentionSource(
        records="abstained", reason="reason", signals="signals"),
    "saqqara.caption_binding": AbstentionSource(records="abstained"),
}


def abstention_records(analyzer: str, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """One register record per abstention this trace reports.

    Every record answers the three questions of the abstention contract: **who**
    abstained (the analyzer), **on what** (`locator`, `None` where the producer
    kept only a count — never invented), and **why** (`reason`, from the
    producer's own closed vocabulary), with the `signals` it saw.

    `count` is 1 for a record that stands for one abstention, and N for a
    histogram row that stands for N of them. Nothing in `PIPELINE` reports a
    histogram today, so a run's register has count 1 throughout and its length is
    its total; the field exists so that stays true if that changes.

    An analyzer with no entry in `ABSTENTION_SOURCES` raises. A register is a
    claim to list everything, and an analyzer this function does not know would
    make that claim false in silence.
    """
    if analyzer not in ABSTENTION_SOURCES:
        raise KeyError(
            f"analyzer {analyzer!r} reports an abstention and declares no source; "
            f"register it in ABSTENTION_SOURCES beside the producer that emits it"
        )
    source = ABSTENTION_SOURCES[analyzer]
    value = trace.get(source.records)

    def one(record: dict[str, Any], count: int = 1) -> dict[str, Any]:
        locator = {key: record[key] for key in source.locator if key in record}
        reason = record.get(source.reason) if source.reason else None
        if source.signals is not None:
            signals = dict(record.get(source.signals) or {})
        else:
            signals = {k: v for k, v in record.items()
                       if k != source.reason and k not in source.locator}
        return {"analyzer": analyzer, "locator": locator or None,
                "reason": reason, "signals": signals, "count": count}

    if value is None:
        return []
    if isinstance(value, list):
        records = [one(r) for r in value if isinstance(r, dict)]
    elif isinstance(value, dict) and not value:
        records = []
    elif isinstance(value, dict) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
        # a histogram: the reasons are the keys, and each row stands for its count
        records = [{"analyzer": analyzer, "locator": None, "reason": reason,
                    "signals": {}, "count": count} for reason, count in value.items()]
    elif isinstance(value, dict):
        records = [one(value)]
    else:
        raise TypeError(
            f"{analyzer} reported abstentions as {type(value).__name__}; the "
            f"declared shapes are {REPORT_SHAPES}"
        )

    if source.tally is not None:
        tally = count_reported(trace.get(source.tally))
        if tally != sum(r["count"] for r in records):
            raise ValueError(
                f"{analyzer} counts {tally} abstention(s) in {source.tally!r} and "
                f"keeps {len(records)} in {source.records!r}: the tally and the "
                f"records disagree, and a register cannot choose between them"
            )
    return records
