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
from .contract import (
    FIGURE_FACTS,
    FIGURE_SOURCES,
    Adapter,
    Mastaba,
    register_adapter,
)

#: The declared vocabularies, one per record kind this reader emits (K2.19).
TEXT_FACTS = ("x", "y", "font_size", "font")

#: Why an object was not read. Closed, and every entry is produced by something:
#: an image carried in the content stream, a resource naming an object that is not
#: there, and a stream that decodes to nothing at all. The last is the quiet one —
#: it raises nothing, and a reader that trusted it would store a picture of zero
#: length under a perfectly valid hash.
OBJECT_SKIPS = ("inline-image-not-extracted", "xobject-unresolvable", "xobject-empty")


def _concat(a, b) -> tuple[float, ...]:
    """a then b, in pdf order: the new matrix multiplies into the current one."""
    return (
        a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5],
    )


def _unit_square(m) -> tuple[float, float, float, float]:
    """The box an image occupies: the unit square under `m`.

    Returned as (left, bottom, right, top) in page space, where y increases
    upward. The caller reorders it for the locator, whose bbox is declared
    left/top/right/bottom.
    """
    corners = [(m[4], m[5]),
               (m[0] + m[4], m[1] + m[5]),
               (m[2] + m[4], m[3] + m[5]),
               (m[0] + m[2] + m[4], m[1] + m[3] + m[5])]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _media_type(obj) -> str:
    filters = obj.get("/Filter")
    names = {str(f) for f in (filters if isinstance(filters, list) else [filters]) if f}
    if "/DCTDecode" in names:
        return "image/jpeg"
    if "/JPXDecode" in names:
        return "image/jp2"
    return "image/x-raw"


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _name_or_none(value):
    return str(value) if value is not None and not isinstance(value, list) else None


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
    # 0.4.0 reads the images a document holds, one record per placement (K6.1).
    version = "0.4.0"
    extensions = (".pdf",)
    fact_sets = {
        "outline_entry": OUTLINE_FACTS,
        "page": PAGE_FACTS,
        "text": TEXT_FACTS,
        "figure": FIGURE_FACTS,
    }
    skip_reasons = OBJECT_SKIPS

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

    # --------------------------------------------------------------- placements

    def _boxes(self, page) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Every `Do` of an image, with the box the transformation gave it.

        The content stream is walked keeping the graphics state: `q` and `Q` push
        and pop, `cm` concatenates, and a `Do` names a resource. An image occupies
        the UNIT SQUARE under the current transformation, so its box is that
        matrix applied to (0,0) and (1,1) — not the stored image's pixel counts,
        which say how finely it was sampled and nothing about where it landed.
        """
        from pypdf.generic import ContentStream, NameObject

        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        images, unresolved = set(), set()
        for name in list(xobjects.keys()):
            try:
                if xobjects[name].get("/Subtype") == "/Image":
                    images.add(str(name))
            except Exception:
                # A resource naming an object that is not there. It is still a
                # placement the document asks for, so it is counted where it is
                # first noticed rather than dropped here and missed downstream.
                unresolved.add(str(name))
        if not images and not unresolved:
            return []

        content = ContentStream(page.get_contents(), page.pdf)
        ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        stack: list[tuple[float, ...]] = []
        found: list[tuple[str, tuple[float, float, float, float]]] = []

        for operands, operator in content.operations:
            if operator == b"q":
                stack.append(ctm)
            elif operator == b"Q":
                ctm = stack.pop() if stack else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            elif operator == b"cm" and len(operands) == 6:
                ctm = _concat(tuple(float(v) for v in operands), ctm)
            elif operator == b"Do" and operands:
                name = str(operands[0])
                if name in images:
                    found.append((name, _unit_square(ctm)))
                elif name in unresolved:
                    self._skip("xobject-unresolvable")
        return found

    @staticmethod
    def _inline_count(page) -> int:
        """Images carried in the content stream rather than as objects (K6.5)."""
        from pypdf.generic import ContentStream

        try:
            content = ContentStream(page.get_contents(), page.pdf)
        except Exception:
            return 0
        return sum(1 for _, operator in content.operations
                   if operator == b"INLINE IMAGE")

    def _figures(self, page, number: int) -> Iterator[Mastaba]:
        if self.store is None:
            return
        for _ in range(self._inline_count(page)):
            self._skip("inline-image-not-extracted")

        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        for name, box in self._boxes(page):
            try:
                obj = xobjects[name]
                payload = obj.get_data()
            except Exception:
                # A resource naming an object that is not there. Counted: a
                # placement the reader could not honour is a fact about the
                # document, and dropping it silently is how a shortfall becomes
                # invisible to the measurement meant to find it (K6.7).
                self._skip("xobject-unresolvable")
                continue
            if not payload:
                self._skip("xobject-empty")
                continue
            media = _media_type(obj)
            left, bottom, right, top = box          # pdf space: y increases upward
            digest = self.store.put(payload, media,
                                    reference={"page": number, "xobject": name,
                                               "bbox": [left, top, right, bottom]})
            yield Mastaba(
                kind="figure",
                locator=PdfLocator(page=number, bbox=(left, top, right, bottom),
                                   xobject=name),
                facts={
                    "asset": digest,
                    "source": "xobject",
                    "media_type": media,
                    "width": _int_or_none(obj.get("/Width")),
                    "height": _int_or_none(obj.get("/Height")),
                    "x": round(left, 2),
                    "y": round(bottom, 2),
                    "w": round(right - left, 2),
                    "h": round(top - bottom, 2),
                    "colorspace": _name_or_none(obj.get("/ColorSpace")),
                    "bits": _int_or_none(obj.get("/BitsPerComponent")),
                    "smask": obj.get("/SMask") is not None,
                },
            )

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

        yield from self._figures(page, number)

        for text, x, y, size, font in placements:
            yield Mastaba(
                kind="text",
                locator=PdfLocator(page=number),
                text=text,
                facts={"x": round(x, 2), "y": round(y, 2),
                       "font_size": round(size, 2), "font": font},
            )


register_adapter(PdfAdapter())
