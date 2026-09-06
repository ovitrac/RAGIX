"""
saqqara.analyzers.islands — when one box holds two tables.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.25-K3.28 of SPEC.md.

Somebody drew a box around two separate panels. The border says "one block"; the content says
"two". Both are evidence, and they disagree.

This analyzer reports the disagreement and does not resolve it. That restraint is the whole
design. Re-segmenting would mean overruling a border somebody drew deliberately, on the strength
of a gap that might be a column of blanks in a single wide table — and it would do so silently,
leaving nothing for a reviewer to look at. So the finding is recorded on the block, a flag is
raised for whoever segments next, and the block keeps the shape the border gave it.

Where a block does hold several islands, the column chain is reported as **empty** rather than
borrowed from the neighbouring island. A two-column label-and-value panel has no column header,
and inventing one from the panel next door would attach every answer to a heading from a
different table.
"""

from __future__ import annotations

from typing import Any

from ..model import Node
from .contract import Analyzer, AnalyzerResult
from .geometry import Rect, parse_range

__all__ = ["IslandsAnalyzer", "find_islands"]


def find_islands(block: Node) -> list[Rect]:
    """Connected regions of populated cells, eight-way, inside one block.

    A merged region holds its text at the anchor and nowhere else, so its other
    positions look empty. They are not: the value covers all of them. Projecting
    the extent before looking for gaps is what stops a tiled label column from
    reading as a handful of disconnected islands — the gaps would be an artefact
    of where the text is stored, not of how the document is laid out.
    """
    valued: set[tuple[int, int]] = set()
    for cell in block.children:
        if cell.kind != "cell" or cell.text is None:
            continue
        locator = cell.provenance.leaf
        valued.add((locator.row, locator.col))
        if locator.merged_range:
            valued.update(parse_range(locator.merged_range).positions())

    seen: set[tuple[int, int]] = set()
    found: list[Rect] = []
    for start in sorted(valued):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        member = []
        while stack:
            row, col = stack.pop()
            member.append((row, col))
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbour = (row + d_row, col + d_col)
                    if neighbour in valued and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
        rows = [r for r, _ in member]
        cols = [c for _, c in member]
        found.append(Rect(min(rows), min(cols), max(rows), max(cols)))

    return sorted(found, key=lambda r: (r.top, r.left))


class IslandsAnalyzer(Analyzer):
    """Report blocks that hold several disconnected panels. Never re-segment them."""

    name = "islands"
    version = "0.1.0"

    def run(self, tree) -> AnalyzerResult:
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "blocks": 0,
            "multi_island": 0,
            "feedback": [],
        }

        for section in tree.root.children:
            for block in section.children:
                if block.facts.get("block_type") not in ("table", "uncertain"):
                    continue
                trace["blocks"] += 1
                islands = find_islands(block)
                cells = {
                    (c.provenance.leaf.row, c.provenance.leaf.col): c
                    for c in block.children
                    if c.kind == "cell"
                }

                block.facts["islands"] = {
                    "count": len(islands),
                    "regions": [
                        {
                            "range": island.to_a1(),
                            # The top-left populated cell names the panel, where it
                            # names anything. It is a subtitle, not a column header.
                            "subtitle": (
                                cells[(island.top, island.left)].text
                                if (island.top, island.left) in cells else None
                            ),
                            # Honest, not borrowed: a panel with no header row of
                            # its own has no column chain, and the neighbouring
                            # panel's header belongs to the neighbouring panel.
                            "col_chain": [],
                        }
                        for island in islands
                    ],
                    # Recorded for whoever segments next. Never acted on here:
                    # the border is evidence somebody drew on purpose.
                    "segmentation_feedback": len(islands) > 1,
                    "resegmented": False,
                }

                if len(islands) > 1:
                    trace["multi_island"] += 1
                    trace["feedback"].append(
                        {
                            "range": block.facts["block_range"],
                            "islands": [i.to_a1() for i in islands],
                            "action": "reported-never-resegmented",
                        }
                    )

        return AnalyzerResult(tree=tree, trace=self.traced(trace))
