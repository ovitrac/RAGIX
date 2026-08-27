"""
saqqara.adapters.docx — the word-processing reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K2.6-K2.10 of SPEC.md.

Three things make this format harder to read than it looks, and all three are the reason this
module exists rather than a call to a library's `.text`.

**A table is not the grid it appears to be.** Horizontal merges remove cells from a row and
vertical merges leave behind continuation cells that hold nothing. Reading rows naively gives
ragged widths and phantom blanks. This reader resolves the grid: every row reports the same number
of columns, each cell reports its span and its merge state, and a continuation is labelled as one
so that later stages can exclude it. That last point has a direct consequence — counting blank
cells without excluding continuations overstates the number of answer slots on every merged form.

**Tables live in more than one place.** A table in the page header and a table nested inside a
cell are invisible to a walk over the body, and both routinely carry the reference information a
form depends on. Each stream is read, and each is indexed within itself: two tables numbered zero
are not a collision if one is in the body and the other in the header. An index that ran across
streams would point at the wrong table the moment a header table appeared.

**Numbering is not always in the text.** A document may number its headings as a paragraph
property: the reader of the file sees "Scope", the reader of the screen sees "1.2 Scope". Reading
only text misses the structure entirely, so the property is recorded as a fact.

**A blank is not always a blank.** A cell may hold a form field or a content control — a marker
that says "an answer goes here" without any border to show it. A marker inside a cell and a marker
in a body paragraph outside every table mean different things, and a reader that reports both as
"a marker" has thrown away the distinction that tells a slot from a signature line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..model import DocxLocator
from .contract import Adapter, Mastaba, register_adapter

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _q(tag: str) -> str:
    return f"{{{_W}}}{tag}"


#: Keywords that mark a fillable position in the legacy field syntax.
FIELD_MARKERS = {"FORMTEXT": "form-text", "FORMCHECKBOX": "form-checkbox"}

CELL_FACTS = ("span", "vmerge", "empty", "fillable", "marker", "bold", "shaded")


class DocxAdapter(Adapter):
    """Read a document into table, cell, paragraph and marker observations."""

    format = "docx"
    version = "0.2.0"          # 0.2.0: body paragraphs, with style and numbering
    extensions = (".docx",)
    fact_set = CELL_FACTS

    def read(self, path: Path) -> Iterator[Mastaba]:
        from docx import Document

        document = Document(path)

        for index, table in enumerate(document.tables):
            yield from self._table(table, flow="body", index=index)

        for position, paragraph in enumerate(document.paragraphs):
            marker = self._field_marker(paragraph._p)
            text = paragraph.text.strip()
            if not marker and not text:
                continue
            # One record per paragraph, whatever it holds. Emitting a marker
            # record AND a paragraph record for the same position would give two
            # observations one address, and an address that names two things is
            # not an address.
            yield Mastaba(
                kind="marker" if marker else "paragraph",
                locator=DocxLocator(flow="body", paragraph=position),
                text=text or None,
                facts={
                    "marker": marker,
                    "in_table": False,
                    "style": self._style(paragraph),
                    "numbered": self._numbered(paragraph),
                    "outline_level": self._outline_level(paragraph),
                    "bold": self._bold(paragraph._p),
                },
            )

        for number, section in enumerate(document.sections):
            flow = f"header:{number}"
            for index, table in enumerate(section.header.tables):
                yield from self._table(table, flow=flow, index=index)

    # ------------------------------------------------------------------ table

    def _table(self, table, flow: str, index: int) -> Iterator[Mastaba]:
        grid = self._resolve(table)
        widths = {len(row) for row in grid}

        yield Mastaba(
            kind="table",
            locator=DocxLocator(flow=flow, table_index=index),
            facts={
                "n_rows": len(grid),
                "n_grid_cols": max(widths) if widths else 0,
                "style": table.style.name if table.style is not None else None,
                "ragged": len(widths) > 1,
            },
        )

        for row_index, row in enumerate(grid):
            for col_index, entry in enumerate(row):
                if entry is None:
                    continue
                tc, span, vmerge = entry
                text = self._cell_text(tc)
                marker = self._field_marker(tc) or self._content_control(tc)
                yield Mastaba(
                    kind="cell",
                    locator=DocxLocator(
                        flow=flow, table_index=index, row=row_index, col=col_index
                    ),
                    text=text or None,
                    facts={
                        "span": span,
                        "vmerge": vmerge,
                        "empty": not text.strip(),
                        "fillable": marker is not None,
                        "marker": marker,
                        "bold": self._bold(tc),
                        "shaded": self._shaded(tc),
                    },
                )

                for nested_index, nested in enumerate(tc.findall(_q("tbl"))):
                    yield from self._nested(
                        nested,
                        flow=f"nested:{flow}:{index}:{row_index}:{col_index}",
                        index=nested_index,
                    )

    def _nested(self, tbl, flow: str, index: int) -> Iterator[Mastaba]:
        """A nested table, read through the raw element rather than the library."""
        grid = self._resolve_element(tbl)
        yield Mastaba(
            kind="table",
            locator=DocxLocator(flow=flow, table_index=index),
            facts={
                "n_rows": len(grid),
                "n_grid_cols": max((len(r) for r in grid), default=0),
                "style": None,
                "ragged": len({len(r) for r in grid}) > 1,
            },
        )
        for row_index, row in enumerate(grid):
            for col_index, entry in enumerate(row):
                if entry is None:
                    continue
                tc, span, vmerge = entry
                text = self._cell_text(tc)
                yield Mastaba(
                    kind="cell",
                    locator=DocxLocator(
                        flow=flow, table_index=index, row=row_index, col=col_index
                    ),
                    text=text or None,
                    facts={
                        "span": span,
                        "vmerge": vmerge,
                        "empty": not text.strip(),
                        "fillable": False,
                        "marker": None,
                        "bold": self._bold(tc),
                        "shaded": self._shaded(tc),
                    },
                )

    # ------------------------------------------------------- grid resolution

    def _resolve(self, table):
        return self._resolve_element(table._tbl)

    @staticmethod
    def _resolve_element(tbl):
        """Expand each row to full grid width, marking spans and merge state.

        A cell spanning three columns occupies one entry and three positions;
        the two it covers stay empty rather than being invented as cells.
        """
        grid = []
        for tr in tbl.findall(_q("tr")):
            row: list = []
            for tc in tr.findall(_q("tc")):
                properties = tc.find(_q("tcPr"))
                span = 1
                vmerge = None
                if properties is not None:
                    grid_span = properties.find(_q("gridSpan"))
                    if grid_span is not None:
                        span = int(grid_span.get(_q("val"), "1"))
                    merge = properties.find(_q("vMerge"))
                    if merge is not None:
                        vmerge = merge.get(_q("val"), "continue")
                row.append((tc, span, vmerge))
                row.extend([None] * (span - 1))
            grid.append(row)
        return grid

    # ------------------------------------------------------------------ facts

    @staticmethod
    def _cell_text(tc) -> str:
        """Text of this cell only.

        Direct paragraphs, never the paragraphs of a table nested inside it:
        letting nested content bleed upward makes the parent cell claim words
        that belong to a different table (K2.7).
        """
        parts = []
        for paragraph in tc.findall(_q("p")):
            parts.append("".join(node.text or "" for node in paragraph.iter(_q("t"))))
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _style(paragraph) -> str | None:
        try:
            return paragraph.style.name
        except Exception:
            return None

    @staticmethod
    def _numbered(paragraph) -> bool:
        """Numbering applied as a property rather than typed into the text.

        A document numbered this way shows "1.2 Scope" on screen and stores
        "Scope": a reader that only looks at text sees no number at all.
        """
        properties = paragraph._p.find(_q("pPr"))
        return properties is not None and properties.find(_q("numPr")) is not None

    @staticmethod
    def _outline_level(paragraph) -> int | None:
        properties = paragraph._p.find(_q("pPr"))
        if properties is None:
            return None
        level = properties.find(_q("outlineLvl"))
        if level is not None:
            return int(level.get(_q("val"), "0"))
        numbering = properties.find(_q("numPr"))
        if numbering is not None:
            ilvl = numbering.find(_q("ilvl"))
            if ilvl is not None:
                return int(ilvl.get(_q("val"), "0"))
        return None

    @staticmethod
    def _field_marker(element) -> str | None:
        for instruction in element.iter(_q("instrText")):
            text = (instruction.text or "").strip().upper()
            for keyword, name in FIELD_MARKERS.items():
                if keyword in text:
                    return name
        return None

    @staticmethod
    def _content_control(element) -> str | None:
        return "content-control" if element.find(_q("sdt")) is not None else None

    @staticmethod
    def _bold(tc) -> bool:
        for run_properties in tc.iter(_q("rPr")):
            bold = run_properties.find(_q("b"))
            if bold is not None and bold.get(_q("val"), "1") not in ("0", "false"):
                return True
        return False

    @staticmethod
    def _shaded(tc) -> bool:
        properties = tc.find(_q("tcPr"))
        if properties is None:
            return False
        shading = properties.find(_q("shd"))
        if shading is None:
            return False
        fill = shading.get(_q("fill"))
        return bool(fill and fill.lower() not in ("auto", "ffffff"))


register_adapter(DocxAdapter())
