"""
saqqara.analyzers.vector_regions — when a page of ink amounts to a figure.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-29

Specified by K6.14 and K6.15 in SPEC.md.

The reader says where ink is and how much of it there is, and claims nothing. Deciding that some of
it is a *figure* is an inference, and a bad one is expensive in both directions: promote too
readily and every tree fills with rules, borders and page furniture; promote too reluctantly and the
schematics that carry a tender's actual content go missing.

So two declared minimums, applied in the open, **and every refusal counted with its reason**. A rule
line is the case this module exists for: the commonest mark in a tender document, never a figure,
and the thing a naive area test promotes first.

**What a region is, and what it looks like, are different questions.** Its identity is the marks
that draw it and the extent they cover, serialised and stored. Its raster is derived — from a
renderer, at a version, at a resolution — and lives under a key of its own. Change the renderer and
every pixel may change while the tree stays byte-identical, which is what K4 rests on.
"""

from __future__ import annotations

from typing import Any

from ..model import Node, Provenance
from ..render import RENDER_DPI, RenderFailed, default_renderer, raster_key, source_id
from .contract import Analyzer, AnalyzerResult

__all__ = [
    "MIN_REGION_AREA",
    "MIN_REGION_OPS",
    "REGION_MERGE_GAP",
    "REGION_FACTS",
    "REGION_REFUSALS",
    "RENDER_CHANNEL",
    "TABLE_RULE",
    "is_lattice",
    "VectorRegionAnalyzer",
]

#: Rendered regions travel under their own channel, like every other inference.
RENDER_CHANNEL = "object-render"

#: Below this many vector operators, a cluster is a mark rather than a drawing.
#: Eight, declared: a rule line is two, a box is five, and anything a reader
#: would call a diagram is far above it.
MIN_REGION_OPS = 8

#: And below this fraction of the page, it is decoration whatever its operators.
MIN_REGION_AREA = 0.01

#: How close two marks must be to belong to the same drawing, in points. About
#: one line of type: ink separated by more than a line of text is not one
#: picture. NOT in the signed contract's geometry table -- the contract declares
#: the two minimums but not the grouping they are applied to, and a grouping rule
#: is needed before either minimum means anything. Declared here, named in the
#: report, and open to the lead to move.
REGION_MERGE_GAP = 12.0

#: What a promoted region records.
REGION_FACTS = ("asset", "x", "y", "w", "h", "ops", "rule", "confidence")

#: Why a cluster of ink was refused. Closed: a reason outside this list is a bug.
REGION_REFUSALS = ("too-few-operators", "too-small", "render-failed",
                   "region-spans-page")

#: Why a mark never reached the grouping at all. Closed for the same reason, and
#: counted for a sharper one: a mark removed in silence is a mark nobody can
#: argue about, and this exclusion is the most opinionated judgement the module
#: makes.
MARK_EXCLUSIONS = ("page-furniture",)

#: A mark covering this fraction of the page in BOTH dimensions is background,
#: not drawing; a straight rule covering it in ONE is a border or a margin line.
#: Measured, not guessed: the regions that turned out to be whole pages were
#: chained by single-operator rectangles the size of the MediaBox.
FURNITURE_SPAN = 0.9

#: How thin a mark must be to count as a rule rather than a shape, in points.
FURNITURE_THIN = 2.0

#: And the backstop: a cluster still spanning this much of the page in both
#: dimensions is refused. It is deliberately close to the whole page, because a
#: genuine full-page schematic would be refused by it too -- that trade is
#: visible in the count rather than hidden in the threshold.
MAX_REGION_SPAN = 0.95

#: The rule that promotes, and the confidence it confers.
REGION_RULE = ("region-render", 0.8)

#: And the rule for a drawn lattice, at a lower confidence because what it is
#: carrying is a table nobody can read: no cells, no header, no values (K6.18).
TABLE_RULE = ("table-as-image", 0.5)

#: How many distinct rules in each direction make a lattice rather than a corner.
#: Two and two: one horizontal and one vertical line cross, and a crossing is not
#: a table.
TABLE_MIN_LINES = 2

#: And how much of a region must be table evidence -- a straight rule, or a
#: filled cell -- before it is a table. A drawing with a few straight edges is
#: still a drawing.
TABLE_RULE_FRACTION = 0.8


def is_furniture(mark: dict, width: float, height: float) -> bool:
    """Whether a mark is part of the page rather than part of a drawing.

    Three shapes, all measured on real documents before being written here: a
    rectangle the size of the page (a background fill or a clip), a straight rule
    across its full width (a header or footer line), and one down its full height
    (a margin line). Each is furniture; each, merged transitively, reaches every
    other mark on the page and turns a region into the MediaBox.
    """
    if width <= 0 or height <= 0:
        return False
    w, h = float(mark["w"]), float(mark["h"])
    if w >= FURNITURE_SPAN * width and h >= FURNITURE_SPAN * height:
        return True
    if w <= FURNITURE_THIN and h >= FURNITURE_SPAN * height:
        return True
    if h <= FURNITURE_THIN and w >= FURNITURE_SPAN * width:
        return True
    return False


def _merge(marks: list[dict], gap: float) -> list[list[dict]]:
    """Cluster marks whose boxes lie within `gap` of one another.

    Deliberately the simplest thing that can work: boxes are grown by half the
    gap and unioned transitively. A cleverer clustering would be a claim about
    layout that nothing here has measured.
    """
    boxes = [(float(m["x"]), float(m["y"]),
              float(m["x"]) + float(m["w"]), float(m["y"]) + float(m["h"])) for m in marks]
    parent = list(range(len(marks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            a, b = boxes[i], boxes[j]
            if (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                    and a[1] - gap <= b[3] and b[1] - gap <= a[3]):
                parent[find(i)] = find(j)

    groups: dict[int, list[dict]] = {}
    for index, mark in enumerate(marks):
        groups.setdefault(find(index), []).append(mark)
    return list(groups.values())


def is_lattice(members: list[dict]) -> bool:
    """Whether a cluster is a drawn table rather than a drawing.

    Ordered facts, not a score: the marks are almost all straight rules, and they
    run in both directions, at two or more distinct positions each. A column of
    parallel lines is a chart axis or a hatch; a lattice needs both directions.
    """
    if not members:
        return False
    horizontal: set[float] = set()
    vertical: set[float] = set()
    evidence = 0
    for mark in members:
        w, h = float(mark["w"]), float(mark["h"])
        if h <= FURNITURE_THIN and w > FURNITURE_THIN:
            evidence += 1
            horizontal.add(round(float(mark["y"]), 1))
        elif w <= FURNITURE_THIN and h > FURNITURE_THIN:
            evidence += 1
            vertical.add(round(float(mark["x"]), 1))
        elif mark.get("fill") and not mark.get("stroke"):
            # A filled rectangle in a grid is a CELL, and a cell is the strongest
            # evidence of a table there is. Counting it against the lattice --
            # which the first version of this rule did -- refuses exactly the
            # tables this corpus draws: rules down one side, coloured cells down
            # the other. Measured on a real document, then written.
            evidence += 1
            horizontal.add(round(float(mark["y"]), 1))
            vertical.add(round(float(mark["x"]), 1))
    if evidence < TABLE_RULE_FRACTION * len(members):
        return False
    return len(horizontal) >= TABLE_MIN_LINES and len(vertical) >= TABLE_MIN_LINES


def _extent(members: list[dict]) -> tuple[float, float, float, float]:
    left = min(float(m["x"]) for m in members)
    bottom = min(float(m["y"]) for m in members)
    right = max(float(m["x"]) + float(m["w"]) for m in members)
    top = max(float(m["y"]) + float(m["h"]) for m in members)
    return (left, bottom, right, top)


class VectorRegionAnalyzer(Analyzer):
    """Promote clusters of ink to figures, or refuse them by name.

    Opt-in: rendering costs, and a default pipeline that rasterised every page of
    every document would make the cheap paths expensive for a result most callers
    did not ask for.
    """

    name = "saqqara.vector_regions"
    version = "0.1.0"

    def __init__(self, store=None, renderer=None, dpi: int = RENDER_DPI,
                 keep_raster: bool = True) -> None:
        self.store = store
        self.renderer = renderer
        self.dpi = dpi
        #: Whether to persist the rendered bytes. A caller measuring the tree --
        #: counting regions, or collecting refusals -- still needs the render to
        #: RUN, because a renderer that fails is a refusal it must see; it does
        #: not need the pixels kept. That this is even an option is the point of
        #: K6.15: the raster is a derived artefact, so dropping it costs nothing
        #: that identity depends on.
        self.keep_raster = keep_raster

    def run(self, tree) -> AnalyzerResult:
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "pages_examined": 0,
            "marks": 0,
            "clusters": 0,
            "grouped": 0,
            "promoted": 0,
            "excluded": {},
            "refused": {},
            "rendered": 0,
            "renderer": None,
            "dpi": self.dpi,
        }
        pages = [n for n in tree.walk() if n.kind == "page"]
        # The negative matters: a page of prose must be visibly LOOKED at, not
        # merely absent from the counts (K6.14).
        trace["pages_examined"] = len(pages)

        renderer = None
        for page in pages:
            marks = page.facts.get("drawings") or []
            trace["marks"] += len(marks)
            if not marks:
                continue
            width, height = self._page_size(page)
            page_area = width * height
            # Furniture goes BEFORE grouping, not after: refusing it afterwards
            # is too late, because by then it has already merged everything it
            # touched into one cluster.
            drawing = []
            for mark in marks:
                if is_furniture(mark, width, height):
                    self._exclude(trace, "page-furniture")
                else:
                    drawing.append(mark)
            trace["grouped"] += len(drawing)
            if not drawing:
                continue
            for members in _merge(drawing, REGION_MERGE_GAP):
                trace["clusters"] += 1
                ops = sum(int(m["ops"]) for m in members)
                box = _extent(members)
                if ops < MIN_REGION_OPS:
                    self._refuse(trace, "too-few-operators")
                    continue
                span_w = (box[2] - box[0]) / width if width > 0 else 0.0
                span_h = (box[3] - box[1]) / height if height > 0 else 0.0
                if span_w >= MAX_REGION_SPAN and span_h >= MAX_REGION_SPAN:
                    # A chain furniture exclusion did not explain. Counted, so
                    # that "how often does the backstop fire" is a number rather
                    # than a guess.
                    self._refuse(trace, "region-spans-page")
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                if page_area <= 0 or area / page_area < MIN_REGION_AREA:
                    self._refuse(trace, "too-small")
                    continue
                if self.store is None:
                    # Nowhere to put what a region IS. Refusing here rather than
                    # emitting a node whose asset nothing can produce (K6.6).
                    self._refuse(trace, "render-failed")
                    continue
                if renderer is None:
                    renderer = self.renderer or default_renderer()
                    trace["renderer"] = f"{renderer.name} {renderer.version}"
                self._promote(page, members, box, ops, renderer, trace)

        return AnalyzerResult(tree=tree, trace=trace)

    # ------------------------------------------------------------------ steps

    @staticmethod
    def _page_size(page: Node) -> tuple[float, float]:
        """The page's own dimensions, or the declared default if it will not say.

        The default is a last resort and not a convenience: every fraction-of-the
        -page test in this module is a test against these two numbers, and this
        corpus holds pages of 720 x 405 beside pages of 595 x 842.
        """
        width, height = page.facts.get("width"), page.facts.get("height")
        if width is None or height is None:
            return (595.0, 842.0)
        return (float(width), float(height))

    @staticmethod
    def _exclude(trace, reason: str) -> None:
        if reason not in MARK_EXCLUSIONS:
            raise ValueError(f"exclusion reason {reason!r} is outside the frozen vocabulary")
        trace["excluded"][reason] = trace["excluded"].get(reason, 0) + 1

    @staticmethod
    def _refuse(trace, reason: str) -> None:
        if reason not in REGION_REFUSALS:
            raise ValueError(f"refusal reason {reason!r} is outside the frozen vocabulary")
        trace["refused"][reason] = trace["refused"].get(reason, 0) + 1

    def _promote(self, page, members, box, ops, renderer, trace) -> None:
        source = source_id(members, box)
        digest = self.store.put(
            source, "application/json",
            reference={"page": getattr(page.provenance.leaf, "page", None),
                       "marks": len(members), "ops": ops},
        )

        path = page.provenance.source_path
        page_number = getattr(page.provenance.leaf, "page", None) or 1
        try:
            raster, media = renderer.render(path, page_number, box, self.dpi)
        except RenderFailed:
            self._refuse(trace, "render-failed")
            return

        # The raster is stored, and its cache key is recorded WITH it rather than
        # on the node: the node says what the region is, the manifest says what
        # was made from it and under what conditions. Render parameters live where
        # the derived artefact lives.
        if self.keep_raster:
            self.store.put(
                raster, media,
                reference={"derived_from": digest,
                           "raster_key": raster_key(digest, renderer.name,
                                                    renderer.version, self.dpi),
                           "renderer": renderer.name,
                           "renderer_version": renderer.version,
                           "dpi": self.dpi},
            )
        trace["rendered"] += 1

        # Ordered rules: a lattice first, because a drawn table is the specific
        # case and a drawing is the general one.
        rule, confidence = TABLE_RULE if is_lattice(members) else REGION_RULE
        left, bottom, right, top = box
        page.children.append(
            Node(
                kind="vector_region",
                provenance=Provenance(
                    source_path=page.provenance.source_path,
                    source_format=page.provenance.source_format,
                    chain=page.provenance.chain,
                    kernel=self.name,
                    kernel_version=self.version,
                ),
                origin="inferred",
                confidence=confidence,
                facts={
                    "channel": RENDER_CHANNEL,
                    "source": "render",
                    "asset": digest,
                    "x": round(left, 2), "y": round(bottom, 2),
                    "w": round(right - left, 2), "h": round(top - bottom, 2),
                    "ops": ops,
                    "rule": rule,
                    "confidence": confidence,
                },
            )
        )
        trace["promoted"] += 1
