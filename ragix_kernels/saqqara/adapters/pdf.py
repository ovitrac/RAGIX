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

from pathlib import Path
from typing import Iterator

from ..model import PdfLocator
from .contract import Adapter, Mastaba, register_adapter

#: The declared vocabularies, one per record kind this reader emits (K2.19).
TEXT_FACTS = ("x", "y", "font_size", "font")
PAGE_FACTS = ("has_text", "image_count", "needs_ocr")
OUTLINE_FACTS = ("level",)


class PdfAdapter(Adapter):
    """Read a laid-out document into outline, page and text observations."""

    format = "pdf"
    version = "0.2.0"          # 0.2.0: a vocabulary declared per record kind
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
                placements.append((text.strip(), float(tm[4]), float(tm[5]),
                                   float(font_size or 0), name))

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
                facts={"x": round(x, 2), "y": round(y, 2), "font_size": size, "font": font},
            )


register_adapter(PdfAdapter())
