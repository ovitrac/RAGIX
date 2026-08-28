"""
saqqara.adapters.pdf — the laid-out-document reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K2.13 and K2.14 of SPEC.md.

Two things separate this format from the others.

**Some documents declare their own outline, and most do not.** Where one is declared it is the
best structural evidence a document will ever give: an author wrote it down. This reader reports
it as read, at the depth it was written, and does not attempt to improve on it. Where none is
declared, this reader says nothing about structure — inferring headings from font sizes is real
work and it belongs to an analyzer that can be measured against a baseline, not to a reader that
would smuggle its guesses in as observations.

**A page can carry no text at all.** A scan has ink and no characters. `extract_text` returns an
empty string, and a reader that passes that along as content has told the truth about the bytes
and a lie about the document: downstream, an empty page is indistinguishable from a page that was
read and found blank. So a page with no text layer is reported with `needs_ocr` set and its image
count attached — a declared pending state, never a silent emptiness (K2.14).

Positions: each text observation records the point the text matrix placed it at, and the locator
carries the page. It does not carry a bounding box. A true box needs glyph metrics this reader
does not have, and a box invented from a font size would be a measurement nobody made — the kind
of plausible number that survives precisely because it looks like the others.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

from ..model import PdfLocator
from .contract import Adapter, Mastaba, register_adapter

#: The declared vocabularies, one per record kind this reader emits (K2.19).
TEXT_FACTS = ("x", "y", "font_size", "font")


def _y_scale(matrix) -> float:
    """How much a matrix stretches the vertical direction (K2.24).

    A pdf matrix (a, b, c, d, e, f) sends a point (x, y) to
    (a*x + c*y, b*x + d*y), so the unit vertical vector (0, 1) lands on (c, d)
    and the height scale is the LENGTH of that image, hypot(c, d).

    Two other readings are available and both are wrong here. Taking `d` alone
    is the same number for upright type and zero for type turned a quarter turn,
    which would report rotated headings as sizeless. Taking the square root of
    the determinant averages the horizontal and vertical scales, so type stretched
    in one direction only would read at a size it is set in neither.

    A matrix that cannot be read is treated as neutral rather than as a reason to
    fail: the size is then the operand alone, which is what this reader did for
    every document before this fact was corrected.
    """
    try:
        return math.hypot(float(matrix[2]), float(matrix[3]))
    except (TypeError, ValueError, IndexError):
        return 1.0
PAGE_FACTS = ("has_text", "image_count", "needs_ocr")
OUTLINE_FACTS = ("level",)


class PdfAdapter(Adapter):
    """Read a laid-out document into outline, page and text observations."""

    format = "pdf"
    # 0.3.0 reports the size type is set at rather than the size it asked for:
    # the operand scaled by the text matrix and the page transformation (K2.24).
    version = "0.3.0"
    extensions = (".pdf",)
    fact_sets = {
        "outline_entry": OUTLINE_FACTS,
        "page": PAGE_FACTS,
        "text": TEXT_FACTS,
    }

    def read(self, path: Path) -> Iterator[Mastaba]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = {id(page.indirect_reference): n for n, page in enumerate(reader.pages, start=1)}

        yield from self._outline(reader, pages)

        for number, page in enumerate(reader.pages, start=1):
            yield from self._page(page, number)

    # ---------------------------------------------------------------- outline

    def _outline(self, reader, pages) -> Iterator[Mastaba]:
        try:
            declared = reader.outline
        except Exception:                      # a malformed outline is not a readable one
            return

        for item, depth in self._flatten(declared):
            title = getattr(item, "title", None)
            if not title:
                continue
            yield Mastaba(
                kind="outline_entry",
                locator=PdfLocator(page=self._page_of(reader, item)),
                text=str(title),
                facts={"level": depth},
            )

    @staticmethod
    def _flatten(items, depth: int = 1):
        """Walk the outline tree, keeping the depth each entry was written at."""
        for item in items:
            if isinstance(item, list):
                yield from PdfAdapter._flatten(item, depth + 1)
            else:
                yield item, depth

    @staticmethod
    def _page_of(reader, item) -> int:
        try:
            return reader.get_destination_page_number(item) + 1
        except Exception:
            return 0

    # ------------------------------------------------------------------- page

    def _page(self, page, number: int) -> Iterator[Mastaba]:
        placements: list[tuple[str, float, float, float, str]] = []

        def visitor(text, cm, tm, font_dict, font_size):
            if text and text.strip():
                name = ""
                if isinstance(font_dict, dict):
                    name = str(font_dict.get("/BaseFont", ""))
                # The operand is what the type asked for; the two matrices are what
                # it got. Only their product is the size on the page (K2.24).
                size = float(font_size or 0) * _y_scale(tm) * _y_scale(cm)
                placements.append((text.strip(), float(tm[4]), float(tm[5]),
                                   size, name))

        try:
            page.extract_text(visitor_text=visitor)
        except Exception:
            placements = []

        try:
            image_count = len(page.images)
        except Exception:
            image_count = 0

        has_text = bool(placements)
        yield Mastaba(
            kind="page",
            locator=PdfLocator(page=number),
            facts={
                "has_text": has_text,
                "image_count": image_count,
                # Declared, not inferred later: a page with ink and no characters
                # is pending, and saying so here is what keeps it from passing for
                # a page that was read and found empty.
                "needs_ocr": not has_text,
            },
        )

        for text, x, y, size, font in placements:
            yield Mastaba(
                kind="text",
                locator=PdfLocator(page=number),
                text=text,
                facts={"x": round(x, 2), "y": round(y, 2),
                       "font_size": round(size, 2), "font": font},
            )


register_adapter(PdfAdapter())
