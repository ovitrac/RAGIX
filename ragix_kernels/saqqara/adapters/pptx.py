"""
saqqara.adapters.pptx — the presentation reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K2.15 of SPEC.md.

Slides are numbered the way a reader counts them, from one. A zero-based slide number is correct
in the file and wrong in a citation: nobody looking for slide 7 wants slide 8.

**A table on a slide is a table.** It arrives as a shape with no text frame, so a reader that
walks text frames sees nothing at all — not an empty table, but no table. Its cells are emitted
here in the same vocabulary a word-processing or spreadsheet grid is emitted in, which means the
header and label rules read a slide table without learning that slides exist.

Speaker notes carry their own address (`notes=True`) as well as their own fact
(`on_slide=False`). The two are not redundant: the address says where the text is, the fact says
what it is. Speaker notes are kept apart from what is on the slide. They are frequently the most substantive
text in a deck, and they are also the text the audience never saw — merging the two produces a
document that claims to show what it merely mentioned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..model import PptxLocator
from .contract import (
    GRID_CELL_FACTS,
    GRID_TABLE_FACTS,
    Adapter,
    Mastaba,
    register_adapter,
)

#: A table on a slide is a grid, and it is described in the same words a grid is
#: described in everywhere else — so the same recognition rules can read it. The
#: words themselves come from the contract, not from a copy kept here (K2.22).
CELL_FACTS = GRID_CELL_FACTS
TABLE_FACTS = GRID_TABLE_FACTS

#: The vocabularies of what only a deck has (K2.19).
SLIDE_FACTS = ("shape_count",)
SHAPE_FACTS = ("shape_type", "is_title", "on_slide")
NOTES_FACTS = ("on_slide",)


class PptxAdapter(Adapter):
    """Read a deck into slide, shape and notes observations."""

    format = "pptx"
    version = "0.3.0"          # 0.3.0: a vocabulary declared per record kind
    extensions = (".pptx",)
    fact_sets = {
        "slide": SLIDE_FACTS,
        "shape": SHAPE_FACTS,
        "notes": NOTES_FACTS,
        "table": TABLE_FACTS,
        "cell": CELL_FACTS,
    }

    def read(self, path: Path) -> Iterator[Mastaba]:
        from pptx import Presentation

        deck = Presentation(path)
        for number, slide in enumerate(deck.slides, start=1):
            title = slide.shapes.title
            # Identity is not reliable here: the library hands back a fresh
            # proxy object on each access, so `shape is title` is False even
            # for the title itself. The shape id is stable; compare that.
            title_id = None if title is None else title.shape_id
            yield Mastaba(
                kind="slide",
                locator=PptxLocator(slide=number),
                text=title.text if title is not None else None,
                facts={"shape_count": len(slide.shapes)},
            )

            for index, shape in enumerate(slide.shapes):
                if getattr(shape, "has_table", False):
                    yield from self._table(shape, number, index)
                    continue
                if not shape.has_text_frame:
                    continue
                text = shape.text_frame.text
                if not text.strip():
                    continue
                yield Mastaba(
                    kind="shape",
                    locator=PptxLocator(slide=number, shape=index),
                    text=text,
                    facts={
                        "is_title": shape.shape_id == title_id,
                        "shape_type": str(shape.shape_type),
                        "on_slide": True,
                    },
                )

            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text
                if notes.strip():
                    yield Mastaba(
                        kind="notes",
                        locator=PptxLocator(slide=number, notes=True),
                        text=notes,
                        facts={"on_slide": False},
                    )


    # ------------------------------------------------------------------ table

    def _table(self, shape, slide: int, index: int):
        """A slide table, as a resolved grid in the shared vocabulary."""
        table = shape.table
        rows = len(table.rows)
        cols = len(table.columns)

        yield Mastaba(
            kind="table",
            locator=PptxLocator(slide=slide, shape=index),
            facts={"n_rows": rows, "n_grid_cols": cols, "style": None, "ragged": False},
        )

        # Where each covered position's merge began, so a position covered from
        # ABOVE can be told from one covered from the LEFT.
        origin_of = {}
        for row in range(rows):
            for col in range(cols):
                cell = table.cell(row, col)
                if not cell.is_merge_origin:
                    continue
                for r in range(row, row + cell.span_height):
                    for c in range(col, col + cell.span_width):
                        origin_of[(r, c)] = (row, col)

        for row in range(rows):
            for col in range(cols):
                cell = table.cell(row, col)
                if cell.is_spanned:
                    origin = origin_of.get((row, col))
                    if origin is None or origin[1] != col:
                        # Covered from the left: the span already says so, and a
                        # word-processing file does not store this position either.
                        continue
                    # Covered from above. A word-processing reader emits this as a
                    # continuation, so this one does too — the shared mapping walks
                    # continuations to find how far a merge reaches, and a format
                    # that stayed silent here would lose the extent.
                    yield Mastaba(
                        kind="cell",
                        locator=PptxLocator(slide=slide, shape=index, row=row, col=col),
                        facts={"span": 1, "vmerge": "continue", "empty": True,
                               "fillable": False, "marker": None, "bold": False,
                               "shaded": False},
                    )
                    continue
                text = cell.text.strip()
                yield Mastaba(
                    kind="cell",
                    locator=PptxLocator(slide=slide, shape=index, row=row, col=col),
                    text=text or None,
                    facts={
                        "span": cell.span_width,
                        "vmerge": "restart" if cell.span_height > 1 else None,
                        "empty": not text,
                        "fillable": False,
                        "marker": None,
                        "bold": self._bold(cell),
                        "shaded": self._shaded(cell),
                    },
                )

    @staticmethod
    def _bold(cell) -> bool:
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                if run.font.bold:
                    return True
        return False

    @staticmethod
    def _shaded(cell) -> bool:
        try:
            return str(cell.fill.type) not in ("None", "MSO_FILL_TYPE.BACKGROUND", "-1")
        except Exception:
            return False


register_adapter(PptxAdapter())
