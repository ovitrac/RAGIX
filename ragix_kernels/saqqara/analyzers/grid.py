"""
saqqara.analyzers.grid — the format-neutral vocabulary the recognition rules see.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.34 of SPEC.md, and is the seam K3.29-K3.33 depend on.

A spreadsheet and a word-processing table are the same object seen through two file formats. One
records a data type and a merged range; the other records a column span, a vertical merge and text
that happens to look like a number. Underneath, both are a grid of positions where some carry
values, some carry labels, and some are waiting to be filled in.

`GridCell` is that underneath. Everything in `header_bands` and `chains` operates on it, and
neither ever learns which format it came from — which is what makes "the same rules, both lanes" a
fact rather than an intention.

The mapping from raw facts to this vocabulary is where the formats differ, and it is **declared**:
each rule below is written down, applied in order, and recorded in the trace. That matters more
than it looks. A mapping decision taken silently inside a rule is a format leaking into the core,
and the next format will leak differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..model import Node
from .geometry import Rect, parse_range, rng

__all__ = [
    "DOCX_MAPPING",
    "PPTX_MAPPING",
    "XLSX_MAPPING",
    "GridCell",
    "grid_cells",
    "grid_extent",
]

#: The spreadsheet lane records these facts directly; the mapping is a reading.
XLSX_MAPPING = (
    "X1 data type, boldness and number format are read as the file recorded them",
    "X2 a merged range is read as the file recorded it",
    "X3 a blank that is ruled or anchors a merge is an addressable slot",
)

#: The word-processing lane has no data types and no ranges; both are derived,
#: under rules written here rather than decided inside a recognition rule.
DOCX_MAPPING = (
    "M1 every populated cell is text",
    "M2 text that parses as a plain number or percentage counts as numeric",
    "M3 an empty cell that is not a merge continuation is an addressable slot",
    "M4 a cell carrying a fillable marker is a slot, with or without text",
    "M5 a column span and a vertical merge resolve to one merged extent",
)

#: A slide table is described in the same words as a word-processing one, so the
#: mapping is the same rules under a different name — which is the claim, not a
#: coincidence: a third format reaches the core without the core changing.
PPTX_MAPPING = (
    "M1 every populated cell is text",
    "M2 text that parses as a plain number or percentage counts as numeric",
    "M3 an empty cell that is not a merge continuation is an addressable slot",
    "M5 a column span and a row span resolve to one merged extent",
)

_NUMERIC = re.compile(r"^[-+]?[\d  ]*[.,]?\d+\s*%?$")


@dataclass(frozen=True)
class GridCell:
    """One position on a grid, in the vocabulary the rules are written against."""

    row: int
    col: int
    text: str | None = None
    dtype: str | None = None
    bold: bool = False
    merged: str | None = None
    slot: bool = False
    node: Node | None = None

    @property
    def extent(self) -> Rect:
        return parse_range(self.merged) if self.merged else Rect(self.row, self.col,
                                                                 self.row, self.col)


def grid_extent(cells: list[GridCell]) -> Rect:
    """The rectangle a set of grid cells covers, merges included."""
    tops, lefts, bottoms, rights = [], [], [], []
    for cell in cells:
        extent = cell.extent
        tops.append(extent.top)
        lefts.append(extent.left)
        bottoms.append(extent.bottom)
        rights.append(extent.right)
    return Rect(min(tops), min(lefts), max(bottoms), max(rights))


def grid_cells(block: Node) -> tuple[list[GridCell], tuple[str, ...]]:
    """Map one block's cells into the neutral vocabulary, and say how."""
    fmt = block.provenance.source_format
    if fmt == "xlsx":
        return _from_xlsx(block), XLSX_MAPPING
    if fmt == "docx":
        return _from_docx(block), DOCX_MAPPING
    if fmt == "pptx":
        return _from_docx(block), PPTX_MAPPING
    raise ValueError(f"no grid mapping declares the format {fmt!r}")


# ------------------------------------------------------------------- xlsx

def _from_xlsx(block: Node) -> list[GridCell]:
    out = []
    for cell in block.children:
        if cell.kind != "cell":
            continue
        locator = cell.provenance.leaf
        border = cell.facts.get("border") or {}
        out.append(
            GridCell(
                row=locator.row,
                col=locator.col,
                text=cell.text,
                dtype=cell.facts.get("dtype"),
                bold=bool(cell.facts.get("bold")),
                merged=locator.merged_range,
                slot=cell.text is None and (any(border.values()) or bool(locator.merged_range)),
                node=cell,
            )
        )
    return out


# ------------------------------------------------------------------- docx

def _from_docx(block: Node) -> list[GridCell]:
    """Apply M1-M5, in order, to the resolved-grid facts of a table.

    Shared with presentations: both readers describe a grid in the same words, so
    one mapping serves both. If a third grid-bearing format ever needs different
    rules, it declares them; it does not fork these.
    """
    cells = [c for c in block.children if c.kind == "cell"]
    by_position = {(c.provenance.leaf.row, c.provenance.leaf.col): c for c in cells}

    out = []
    for cell in cells:
        locator = cell.provenance.leaf
        raw_row, raw_col = locator.row, locator.col
        # The word-processing grid counts from zero and the neutral vocabulary
        # counts from one, because A1 notation does. Converting here, once, is
        # what lets a rule compare a spreadsheet position with a document one.
        row, col = raw_row + 1, raw_col + 1

        if cell.facts.get("vmerge") == "continue":
            # The value lives at the restart. A continuation carries nothing, and
            # counting it would turn one merged label into several blank ones.
            continue

        text = cell.text
        dtype = None
        if text is not None and text.strip():
            dtype = "s"                                            # M1
            if _NUMERIC.match(text.strip()):
                dtype = "n"                                        # M2

        merged = _docx_extent(cell, by_position, raw_row, raw_col)  # M5
        marker = cell.facts.get("marker")
        slot = bool(cell.facts.get("empty")) or marker is not None  # M3, M4

        out.append(
            GridCell(
                row=row,
                col=col,
                text=text,
                dtype=dtype,
                bold=bool(cell.facts.get("bold")),
                merged=merged,
                slot=slot,
                node=cell,
            )
        )
    return out


def _docx_extent(cell: Node, by_position: dict, row: int, col: int) -> str | None:
    """A column span and a vertical merge, resolved into one rectangle (M5)."""
    span = int(cell.facts.get("span") or 1)
    bottom = row
    if cell.facts.get("vmerge") == "restart":
        while True:
            below = by_position.get((bottom + 1, col))
            if below is None or below.facts.get("vmerge") != "continue":
                break
            bottom += 1

    right = col + span - 1
    if bottom == row and right == col:
        return None
    return rng(row + 1, col + 1, bottom + 1, right + 1)
