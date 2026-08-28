"""
saqqara.adapters.xlsx — the spreadsheet reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K2.1-K2.5 of SPEC.md.

Six facts per cell, and exactly six: the data type the file recorded, whether the text is bold,
the number format, the lock flag, the formula if there is one, and whether the cell anchors a
merge. They are read, never judged. A bold numeric cell records both facts and nothing more —
whether that makes it a header is a question this module is not entitled to answer, and answering
it here would destroy the evidence an analyzer needs to answer it well.

Borders are emitted as their own records rather than as cell facts. They are an observation about
a cell, but a different one, and folding them into the cell fact set would make that set something
other than what K2.1 declares. Keeping them separate costs one record kind and keeps the
declaration exact.

A cell holding only whitespace is blank. It looks blank on screen, and treating its empty string
as content let it count as a value in segmentation and typing — a difference in a table's shape
that nothing visible explains. The data type still records that the file held a string, so the
distinction between an empty string and an absent value survives the normalisation.

An empty cell still gets a record when it carries a border or anchors a merge. Both are
addressable positions — a ruled blank is very often the most interesting position on a sheet, and
an empty merged tile is a label somebody left out, which is a fact about the document rather than
an absence of one. A reader that emitted nothing for either would make them unreachable to every
later stage, and the omission would be invisible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..model import XlsxLocator
from .contract import Adapter, Mastaba, register_adapter

#: The declared vocabularies, one per record kind this reader emits (K2.19).
#: Changing any of them means changing `version` in the same edit (K2.5).
SHEET_FACTS = ("hidden", "list_objects", "max_row", "max_column")
CELL_FACTS = ("dtype", "bold", "number_format", "locked", "formula", "merged")

#: Border edges, emitted as their own record kind — and its vocabulary.
BORDER_EDGES = ("left", "right", "top", "bottom")


class XlsxAdapter(Adapter):
    """Read a workbook into sheet, cell and border observations."""

    format = "xlsx"
    version = "0.4.0"          # 0.4.0: a vocabulary declared per record kind
    extensions = (".xlsx", ".xlsm")
    fact_sets = {
        "sheet": SHEET_FACTS,
        "cell": CELL_FACTS,
        "border": BORDER_EDGES,
    }

    def read(self, path: Path) -> Iterator[Mastaba]:
        from openpyxl import load_workbook

        workbook = load_workbook(path, data_only=False)
        try:
            for index, sheet in enumerate(workbook.worksheets):
                yield from self._sheet(sheet, index)
        finally:
            workbook.close()

    # ------------------------------------------------------------------ sheet

    def _sheet(self, sheet, index: int) -> Iterator[Mastaba]:
        anchors, continuations = self._merges(sheet)

        yield Mastaba(
            kind="sheet",
            locator=XlsxLocator(sheet=sheet.title, sheet_index=index),
            text=sheet.title,
            facts={
                "hidden": sheet.sheet_state != "visible",
                "max_row": sheet.max_row,
                "max_column": sheet.max_column,
                "list_objects": sorted(str(t.ref) for t in sheet.tables.values()),
            },
        )

        for row in sheet.iter_rows():
            for cell in row:
                position = (cell.row, cell.column)
                if position in continuations:
                    # The value and its facts live at the anchor. Emitting them
                    # again here would turn one observation into several, and a
                    # later stage counting evidence would count a merge twice.
                    continue

                bordered = self._borders(cell)
                anchors_merge = position in anchors
                if cell.value is None and not any(bordered.values()) and not anchors_merge:
                    # Nothing here: no value, no rule, no merge. Emitting a record
                    # for every empty cell of a sheet would drown the real ones.
                    continue

                yield Mastaba(
                    kind="cell",
                    locator=XlsxLocator(
                        sheet=sheet.title,
                        sheet_index=index,
                        cell=cell.coordinate,
                        row=cell.row,
                        col=cell.column,
                        merged_range=anchors.get(position),
                    ),
                    text=self._text(cell.value),
                    facts=self._cell_facts(cell, merged=position in anchors),
                )

                if any(bordered.values()):
                    yield Mastaba(
                        kind="border",
                        locator=XlsxLocator(
                            sheet=sheet.title,
                            sheet_index=index,
                            cell=cell.coordinate,
                            row=cell.row,
                            col=cell.column,
                        ),
                        facts=bordered,
                    )

    # ------------------------------------------------------------------ facts

    @staticmethod
    def _text(value):
        """A cell holding nothing but whitespace holds nothing.

        A spreadsheet cell whose value is the empty string looks blank to anyone
        reading the document and, before this, read as content to everything
        downstream — it counted as a value in segmentation and typing, which is
        the sort of error that changes a table's shape without changing anything
        anyone can see.

        Nothing is lost by normalising it here: `dtype` still records that the
        file held a string, so an empty string (`dtype` "s", no text) and an
        absent value (no dtype, no text) stay distinguishable.
        """
        if value is None:
            return None
        text = str(value)
        return None if not text.strip() else text

    @staticmethod
    def _cell_facts(cell, merged: bool) -> dict:
        """Exactly CELL_FACTS, every key always present, every value a primitive."""
        return {
            "dtype": cell.data_type,
            "bold": bool(cell.font.bold) if cell.font is not None else False,
            "number_format": cell.number_format,
            "locked": bool(cell.protection.locked) if cell.protection is not None else False,
            "formula": str(cell.value) if cell.data_type == "f" else None,
            "merged": merged,
        }

    @staticmethod
    def _borders(cell) -> dict:
        border = cell.border
        if border is None:
            return {edge: False for edge in BORDER_EDGES}
        return {
            edge: bool(getattr(border, edge) is not None and getattr(border, edge).style)
            for edge in BORDER_EDGES
        }

    @staticmethod
    def _merges(sheet) -> tuple[dict, set]:
        """Anchor position -> range string, and the positions that continue it."""
        anchors: dict[tuple[int, int], str] = {}
        continuations: set[tuple[int, int]] = set()
        for merged in sheet.merged_cells.ranges:
            anchors[(merged.min_row, merged.min_col)] = str(merged)
            for row in range(merged.min_row, merged.max_row + 1):
                for col in range(merged.min_col, merged.max_col + 1):
                    if (row, col) != (merged.min_row, merged.min_col):
                        continuations.add((row, col))
        return anchors, continuations


register_adapter(XlsxAdapter())
