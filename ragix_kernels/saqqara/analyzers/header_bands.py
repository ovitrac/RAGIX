"""
saqqara.analyzers.header_bands — where the headers stop and the content starts.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.13-K3.24 of SPEC.md.

Every cell in a table means something only in the company of the cells that head its column and
label its row. Finding those is this module's whole job, and getting it wrong is expensive in a
particular way: a header band read one row too shallow silently turns a header into data, and
every answer read out of that table afterwards is attached to the wrong question.

So the rules are hard, ordered, and each one names itself in the trace:

  R1  **laminar or abstain.** A header band is a nesting of spans: each tier refines the one
      above, so any two merges are disjoint or nested. Two that cross describe no tree, and there
      is no chain to read from them. The block abstains and says so.
  R2  **uniform blocks abstain.** Same type everywhere, nothing bold, nothing merged: there is no
      evidence here, and a rule that produced an answer anyway would be reporting its own
      preferences.
  R3  **title**: a merge spanning the full width on the block's first row.
  R4  **band depth by style contrast**: a row whose every populated cell is bold, with a
      non-bold row below it, is a header row. The contrast is the evidence — bold alone is not,
      or a bold-everywhere table would be all header.
  R5  **section rows**: a full-width merge inside the body divides the data; it is not a title
      and it is not a header.
  R6  **label columns**: first from the band header, which declares the width of the label zone
      by spanning it; failing that from a type contrast between a text column and the values to
      its right. Failing both, none are claimed.
  R7  **core**: what is left.

R6 is where the temptation lies. In a table whose every column is text, nothing structurally
distinguishes a label column from a content column, and the honest output is no label columns at
all. A rule that picked the leftmost anyway would be right often enough to look correct and would
mis-attribute every table where it was wrong.
"""

from __future__ import annotations

from typing import Any

from ..model import Node
from .contract import Abstention, Analyzer, AnalyzerResult
from .geometry import Rect, a1, is_laminar, parse_range, rng
from .grid import GridCell, grid_cells

__all__ = ["MAX_HEADER_ROWS", "MAX_LABEL_COLS", "HeaderBandsAnalyzer", "analyze_block"]

#: A band deeper than this is not a band. Reported, not silently truncated.
#:
#: The **default**, not the value: a run declares its own through the analyzer's
#: options and the trace records what it used (K3.73). It stays at 3 on the
#: evidence rather than by inertia — measured over 811 abstaining blocks of a real
#: corpus, 742 hold exactly one dense row under sparse context rows and only 4
#: hold two or more, so a larger cap would admit context rows as header tiers and
#: give every column in those blocks a confident, wrong ancestry.
MAX_HEADER_ROWS = 3

#: Likewise for the label zone.
MAX_LABEL_COLS = 3


def analyze_block(block: Node, *, max_header_rows: int = MAX_HEADER_ROWS,
                  max_label_cols: int = MAX_LABEL_COLS) -> dict[str, Any]:
    """Split one block into title, band, labels, sections and core — or abstain.

    The block's own format is read once, here, to choose a mapping; below this
    line nothing knows which format it came from (K3.34).
    """
    cells_list, mapping = grid_cells(block)
    return analyze_grid(
        parse_range(block.facts["block_range"]), cells_list, mapping=mapping,
        max_header_rows=max_header_rows, max_label_cols=max_label_cols,
    )


def analyze_grid(
    rect: Rect, cells_list: list[GridCell], mapping: tuple[str, ...] = (),
    *, max_header_rows: int = MAX_HEADER_ROWS, max_label_cols: int = MAX_LABEL_COLS,
) -> dict[str, Any]:
    """The rules. Format-neutral: they see grid cells and nothing else.

    The two limits are keyword-only with the module defaults, so every existing
    caller — the tests call this directly — is unaffected, and a run that declares
    its own reaches them through the analyzer's options (K3.73).
    """
    cells = {(c.row, c.col): c for c in cells_list}
    merges = sorted(
        (c.extent for c in cells_list if c.merged), key=lambda r: (r.top, r.left)
    )
    signals: dict[str, Any] = {"rules": [], "mapping": list(mapping)}

    valued = {p: c for p, c in cells.items() if c.text is not None}
    if not valued:
        return _abstain("no-populated-region", signals)

    # R1 — laminar or abstain
    intervals = [(m.left, m.right) for m in merges]
    signals["merge_intervals"] = intervals
    if not is_laminar(intervals):
        signals["rules"].append("R1-laminarity")
        return _abstain("non-laminar-band-merges", signals)

    # R2 — uniform block
    dtypes = {c.dtype for c in valued.values()}
    any_bold = any(c.bold for c in valued.values())
    any_slot = any(c.slot and c.text is None for c in cells.values())
    signals["dtypes"] = sorted(d for d in dtypes if d)
    signals["any_bold"] = any_bold
    signals["any_slot"] = any_slot
    # A slot is evidence. A block whose every value looks alike still has a shape
    # if part of it is addressable and waiting to be filled: that is a form, and
    # abstaining on it would throw away the one structure it does have.
    if len(dtypes) == 1 and not any_bold and not merges and not any_slot:
        signals["rules"].append("R2-uniform")
        return _abstain("uniform-block", signals)

    # R3 — title: a full-width merge on the block's first row
    top = rect.top
    title = None
    for merge in merges:
        if merge.top == top and merge.left == rect.left and merge.right == rect.right:
            if rect.bottom - merge.bottom >= 2:
                title = (merge.to_a1(), valued[(merge.top, merge.left)].text)
                top = merge.bottom + 1
                signals["rules"].append("R3-title")
            break

    # R4 — band depth by style contrast
    header_rows: list[int] = []
    row = top
    while row <= rect.bottom:
        populated = [c for (r, _), c in valued.items() if r == row]
        if not populated or not all(c.bold for c in populated):
            break
        below = [c for (r, _), c in valued.items() if r > row]
        if not below or all(c.bold for c in below):
            break                      # bold everywhere is not a contrast
        header_rows.append(row)
        row += 1
    if len(header_rows) > max_header_rows:
        # The band is counted to its end BEFORE abstaining, and the depth is
        # recorded. Returning at the cap left the record saying only "deeper than
        # the cap", so the one number needed to judge the cap had to be re-derived
        # from the cells by re-implementing this rule elsewhere — which is how the
        # depths behind this default were obtained at all. A rule that abstains
        # without saying how far over it went cannot be tuned from its own output.
        signals["rules"].append("R4-band-depth")
        signals["depth_found"] = len(header_rows)
        signals["max_header_rows"] = max_header_rows
        return _abstain("band-too-deep", signals)
    if header_rows:
        signals["rules"].append("R4-style-contrast")
    signals["header_rows"] = list(header_rows)

    body_top = header_rows[-1] + 1 if header_rows else top
    if body_top > rect.bottom:
        return _abstain("no-body-rows", signals)

    # R5 — section rows: a full-width merge inside the body
    section_rows = sorted(
        m.top for m in merges
        if m.left == rect.left and m.right == rect.right and m.top >= body_top
    )
    if section_rows:
        signals["rules"].append("R5-section-rows")

    # R6 — label columns
    label_cols, label_rule = _label_columns(rect, valued, merges, header_rows, body_top,
                                            section_rows, cells, max_label_cols=max_label_cols)
    signals["rules"].append(label_rule)
    signals["label_cols"] = [a1(1, c)[:-1] for c in label_cols]

    core_left = (max(label_cols) + 1) if label_cols else rect.left
    if core_left > rect.right:
        return _abstain("no-column-evidence-within-cap", signals)

    # R7 — the core is what is left
    core = rng(body_top, core_left, rect.bottom, rect.right)

    band_header = None
    if header_rows and label_cols:
        for merge in merges:
            if merge.top == header_rows[0] and merge.left == rect.left and merge.width > 1:
                anchor = valued.get((merge.top, merge.left))
                band_header = (merge.to_a1(), anchor.text if anchor else None)
                break

    tiling = [
        (m.to_a1(), valued[(m.top, m.left)].text if (m.top, m.left) in valued else None)
        for m in merges
        if label_cols and m.left in label_cols and m.top >= body_top
    ]

    return {
        "uncertain": False,
        "core": core,
        "header_rows": list(header_rows),
        "label_cols": [a1(1, c)[:-1] for c in label_cols],
        "title": title,
        "section_rows": section_rows,
        "band_header": band_header,
        "tiling": tiling,
        "signals": signals,
    }


def _label_columns(rect, valued, merges, header_rows, body_top, section_rows, cells=None,
                   *, max_label_cols: int = MAX_LABEL_COLS):
    """L1 the band header declares the zone; L2 the slot column; L3 a type
    contrast; L4 none.

    L2 is what a form looks like. Where a whole column is addressable and empty,
    that column is where the answers go, and everything to its left of it that
    carries text is the question. It matters because a blank form has no values
    at all to contrast types against: L3 would find nothing and report no labels,
    which is the one reading guaranteed to be wrong on the documents this package
    exists to read.
    """
    body = {
        (r, c): cell for (r, c), cell in valued.items()
        if r >= body_top and r not in section_rows
    }

    # L1 — a merge on the first header row, starting at the left edge, spans the zone
    if header_rows:
        for merge in merges:
            if merge.top == header_rows[0] and merge.left == rect.left:
                width = min(merge.width, max_label_cols)
                if merge.right < rect.right:
                    return list(range(rect.left, rect.left + width)), "L1-band-header-span"

    # L2 — a column that is entirely addressable slots ends the label zone
    if cells:
        body_rows = [r for r in rect.rows if r >= body_top and r not in section_rows]
        for col in rect.cols:
            column = [cells.get((r, col)) for r in body_rows]
            present = [c for c in column if c is not None]
            if not present or not all(c.slot and c.text is None for c in present):
                continue
            left = list(range(rect.left, col))
            if not left or len(left) > max_label_cols:
                break
            texted = all(
                any(
                    (cells.get((r, c)) is not None and cells[(r, c)].text is not None)
                    for r in body_rows
                )
                for c in left
            )
            if texted:
                return left, "L2-slot-column"
            break

    # L3 — a text column with non-text values somewhere to its right
    labels: list[int] = []
    for col in range(rect.left, min(rect.left + max_label_cols, rect.right + 1)):
        column = [cell for (r, c), cell in body.items() if c == col]
        if not column or not all(cell.dtype == "s" for cell in column):
            break
        right = [
            cell for (r, c), cell in body.items()
            if c > col and cell.dtype not in (None, "s")
        ]
        if not right:
            break
        labels.append(col)
    if labels:
        return labels, "L3-type-contrast"

    return [], "L4-no-label-evidence"


def _abstain(reason: str, signals: dict[str, Any]) -> dict[str, Any]:
    return {
        "uncertain": True,
        "abstention": Abstention(reason=reason, signals=signals).to_dict(),
        "signals": signals,
    }


class HeaderBandsAnalyzer(Analyzer):
    """Split every table block into its header band, labels, sections and core."""

    name = "header_bands"
    version = "0.2.0"

    #: Declared beside the code that reads them, so the number and its use are read
    #: together. The values are the module defaults; a run may state its own.
    DEFAULTS = {"max_header_rows": MAX_HEADER_ROWS, "max_label_cols": MAX_LABEL_COLS}

    def run(self, tree) -> AnalyzerResult:
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "blocks": 0,
            "abstained": 0,
            "abstentions": [],
        }

        # Walked, not iterated over one level: a spreadsheet block hangs under a
        # sheet, a word-processing table hangs under the document, and the rules
        # are the same for both.
        for block in tree.walk():
            if block.facts.get("block_type") == "table":
                analysis = analyze_block(
                    block,
                    max_header_rows=self.options["max_header_rows"],
                    max_label_cols=self.options["max_label_cols"],
                )
                block.facts["header"] = analysis
                trace["blocks"] += 1
                if analysis["uncertain"]:
                    trace["abstained"] += 1
                    trace["abstentions"].append(
                        {
                            "range": block.facts["block_range"],
                            "reason": analysis["abstention"]["reason"],
                            # The signals the rules read, carried rather than left
                            # on the node: a reason names a rule, and a register
                            # record that gives the rule without what it saw tells
                            # a reader which test failed and nothing about why.
                            "signals": analysis["abstention"]["signals"],
                        }
                    )

        return AnalyzerResult(tree=tree, trace=self.traced(trace))
