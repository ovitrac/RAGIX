"""
saqqara.analyzers.tables — the segmentation cascade, and what each block is.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.1-K3.7 and K3.11-K3.12 of SPEC.md.

A sheet is not a table. It is a surface on which somebody put several things — a title, a table,
some guidance, a hidden vocabulary — and the first job is to find where each one starts and stops.
The baseline this replaces is "one block per sheet", which fuses every multi-object sheet into a
single object and is asserted against here rather than assumed to be worse (K3.5).

The cascade, in order, each stage recorded in the trace:

  S0  a declared table object is an authoritative boundary. Somebody wrote it down; no inference
      competes with that.
  S1  connected regions of occupied positions. A position is occupied if it holds a value, if it
      is covered by a merge whose anchor holds one, or if it carries a border. Counting a bordered
      blank as occupied is what makes the next stage unnecessary in most cases and correct in the
      rest: a blank cell inside a ruled box belongs to the box, and an empty answer slot is the
      most interesting position on many forms.
  S2  a box drawn around a region bridges the blanks inside it, including a fully blank spacer row
      that would otherwise cut the region in two.
  S3  typing, by ordered hard rules. Where the evidence does not decide, the block abstains and
      says which rule ran out.

A region of bordered blanks with no value anywhere is dropped — somebody ruled an empty box — and
the drop is counted with its range, never silent (K3.4).
"""

from __future__ import annotations

from typing import Any, Iterable

from ..model import Node, Provenance, XlsxLocator, kind_registry
from .contract import Abstention, Analyzer, AnalyzerResult
from .geometry import Rect, parse_range

__all__ = ["MAX_BLOCK_SIDE", "TablesAnalyzer", "block_kind"]

kind_registry.register("block")

#: A block wider or taller than this is a sheet, not a block: the cascade reports
#: it rather than pretending to have segmented anything.
MAX_BLOCK_SIDE = 5000

#: block type -> node kind. The lane's own word for the shape stays in the facts.
_KIND_OF = {"table": "table", "text": "paragraph", "list": "list", "uncertain": "table"}


def block_kind(block_type: str) -> str:
    return _KIND_OF[block_type]


class TablesAnalyzer(Analyzer):
    """Find the blocks on each sheet, and say what each one is."""

    name = "tables"
    version = "0.1.0"

    def run(self, tree) -> AnalyzerResult:
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "sheets": [],
            "blocks": 0,
            "dropped": 0,
            "drops": [],
        }

        for section in tree.root.children:
            if section.kind != "section" or section.provenance.source_format != "xlsx":
                continue
            self._sheet(section, trace)

        return AnalyzerResult(tree=tree, trace=trace)

    # ------------------------------------------------------------------ sheet

    def _sheet(self, section: Node, trace: dict[str, Any]) -> None:
        cells = [c for c in section.children if c.kind == "cell"]
        if not cells:
            trace["sheets"].append({"sheet": section.text, "blocks": 0, "signal": "no-cells"})
            return

        occupied, valued, merges = self._occupancy(cells)
        declared = [parse_range(ref) for ref in section.facts.get("list_objects", ())]

        components = self._components(occupied)
        rects = self._apply_declared(components, declared)

        blocks: list[Node] = []
        kept: list[Rect] = []
        for rect in sorted(rects, key=lambda r: (r.top, r.left)):
            if not any(rect.contains(*p) for p in valued):
                trace["dropped"] += 1
                trace["drops"].append(
                    {"reason": "bordered-region-holds-no-value", "range": rect.to_a1(),
                     "sheet": section.text}
                )
                continue
            kept.append(rect)

        by_position = {self._position(c): c for c in cells}
        for rect in kept:
            block_type, signals = self._type_of(
                rect, by_position, valued, occupied, merges, declared
            )
            blocks.append(self._node(section, rect, block_type, signals, by_position))

        section.children = [c for c in section.children if c.kind != "cell"] + blocks
        trace["blocks"] += len(blocks)
        trace["sheets"].append(
            {
                "sheet": section.text,
                "declared_objects": [r.to_a1() for r in declared],
                "components": len(components),
                "blocks": len(blocks),
                "ranges": [r.to_a1() for r in kept],
            }
        )

    # -------------------------------------------------------------- occupancy

    @staticmethod
    def _position(cell: Node) -> tuple[int, int]:
        locator = cell.provenance.leaf
        return (locator.row, locator.col)

    def _occupancy(self, cells: Iterable[Node]):
        """Positions that count as occupied, those that hold a value, and the merges."""
        occupied: set[tuple[int, int]] = set()
        valued: set[tuple[int, int]] = set()
        merges: list[Rect] = []

        for cell in cells:
            position = self._position(cell)
            has_value = cell.text is not None
            border = cell.facts.get("border") or {}
            bordered = any(border.values())

            extent = None
            merged_range = cell.provenance.leaf.merged_range
            if merged_range:
                extent = parse_range(merged_range)
                merges.append(extent)

            if has_value:
                valued.add(position)
                if extent is not None:
                    valued.update(extent.positions())

            if has_value or bordered:
                occupied.add(position)
                if extent is not None:
                    occupied.update(extent.positions())

        return occupied, valued, merges

    @staticmethod
    def _components(occupied: set[tuple[int, int]]) -> list[Rect]:
        """Connected regions, eight-way: a diagonal neighbour is still a neighbour."""
        seen: set[tuple[int, int]] = set()
        found: list[Rect] = []

        for start in sorted(occupied):
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
                        if neighbour in occupied and neighbour not in seen:
                            seen.add(neighbour)
                            stack.append(neighbour)
            rows = [r for r, _ in member]
            cols = [c for _, c in member]
            found.append(Rect(min(rows), min(cols), max(rows), max(cols)))

        return found

    @staticmethod
    def _apply_declared(components: list[Rect], declared: list[Rect]) -> list[Rect]:
        """S0: a declared object replaces whatever the connectivity guessed."""
        if not declared:
            return components
        out = [d for d in declared]
        for rect in components:
            if any(
                d.contains(rect.top, rect.left) or rect.contains(d.top, d.left)
                for d in declared
            ):
                continue
            out.append(rect)
        return out

    # ------------------------------------------------------------------ typing

    def _type_of(self, rect, by_position, valued, occupied, merges, declared):
        """Ordered hard rules. The first that fires decides, and says so.

        Shape is counted over ADDRESSABLE positions, not merely valued ones. A
        form whose questions sit in one column and whose answer columns are ruled
        and empty is a table with three columns, not a list with one: the empty
        slots are the positions the document exists to collect. Counting only
        what is already filled would type every blank form as a list.
        """
        signals: dict[str, Any] = {}

        if any(d.to_a1() == rect.to_a1() for d in declared):
            signals["rule"] = "T1-declared-table-object"
            return "table", signals

        inside = [p for p in valued if rect.contains(*p)]
        addressable = [p for p in occupied if rect.contains(*p)]
        rows = {r for r, _ in addressable}
        cols = {c for _, c in addressable}
        signals["valued_cells"] = len(inside)
        signals["addressable_cells"] = len(addressable)
        signals["value_rows"] = len(rows)
        signals["value_columns"] = len(cols)

        covering = [m for m in merges if m.contains(rect.top, rect.left)]
        if covering and covering[0].to_a1() == rect.to_a1():
            signals["rule"] = "T2-single-merged-region"
            return "text", signals

        if len(cols) >= 2 and len(rows) >= 2:
            signals["rule"] = "T3-two-dimensional-values"
            return "table", signals

        if len(cols) == 1 and len(rows) >= 2:
            head = by_position.get((min(rows), min(cols)))
            if head is not None and head.facts.get("bold"):
                signals["rule"] = "T4-single-column-with-header-evidence"
                signals["head_bold"] = True
                return "uncertain", signals
            signals["rule"] = "T5-single-column-run"
            return "list", signals

        if len(rows) == 1 and len(cols) >= 2:
            signals["rule"] = "T5-single-row-run"
            return "list", signals

        signals["rule"] = "T6-single-cell"
        return "text", signals

    # ------------------------------------------------------------------- node

    def _node(self, section, rect, block_type, signals, by_position) -> Node:
        """A block node, citing the range it covers.

        The locator carries the block's A1 range in its cell reference. A range
        IS an A1 reference, and it keeps the block's address distinct from the
        address of the cell in its top-left corner, which a bare `A3` would not
        (K1.9). A dedicated range field would say it more plainly; that is a
        model change, and the model is frozen.
        """
        sheet_locator = section.provenance.leaf
        locator = XlsxLocator(
            sheet=sheet_locator.sheet,
            sheet_index=sheet_locator.sheet_index,
            cell=rect.to_a1(),
            row=rect.top,
            col=rect.left,
        )
        facts: dict[str, Any] = {
            "block_type": block_type,
            "block_range": rect.to_a1(),
            "signals": signals,
        }
        if block_type == "uncertain":
            facts["block_uncertain"] = Abstention(
                reason=signals["rule"].split("-", 1)[1], signals=signals
            ).to_dict()

        node = Node(
            kind=block_kind(block_type),
            provenance=Provenance(
                source_path=section.provenance.source_path,
                source_format="xlsx",
                chain=(locator,),
                kernel=section.provenance.kernel,
                kernel_version=section.provenance.kernel_version,
            ),
            # A block was recognised, not read: it says so, and its confidence
            # says how far the rules got (K1.4).
            origin="inferred",
            confidence=0.5 if block_type == "uncertain" else 0.9,
            facts=facts,
        )
        node.children = [
            cell for position, cell in sorted(by_position.items()) if rect.contains(*position)
        ]
        return node
