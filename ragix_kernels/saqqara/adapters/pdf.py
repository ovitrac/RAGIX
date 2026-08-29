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
OBJECT_SKIPS = ("inline-image-not-extracted", "xobject-unresolvable", "xobject-empty",
                "form-cycle", "form-too-deep")

#: What a drawing observation records: how many vector operators, over what
#: extent, and whether the path was stroked or filled. NOT a picture -- saying a
#: page has ink here claims nothing about whether the ink means anything.
DRAWING_FACTS = ("x", "y", "w", "h", "ops", "stroke", "fill")

#: Path construction, by operand shape. `re` is a rectangle and contributes four
#: corners; the curve operators contribute every control point, because a curve
#: stays inside the hull of its controls and an extent that ignored them would
#: be too small rather than merely imprecise.
_PATH_OPS: dict[bytes, str] = {
    b"m": "point", b"l": "point", b"re": "rect",
    b"c": "points", b"v": "points", b"y": "points",
}

#: Painting ends a path. `n` ends it having painted nothing -- a clip, usually --
#: and is counted like the rest: the operators were there and the extent is real.
_PAINT_OPS: dict[bytes, tuple[bool, bool]] = {
    b"S": (True, False), b"s": (True, False),
    b"f": (False, True), b"F": (False, True), b"f*": (False, True),
    b"B": (True, True), b"B*": (True, True),
    b"b": (True, True), b"b*": (True, True),
    b"n": (False, False),
}

#: How far the reader follows a Form XObject into another. Six, because the
#: deepest nesting measured over the reference corpus was three, and a limit
#: set at the deepest thing yet seen is a limit that will be hit by the next
#: document. It is a declared bound and not a guess about geometry: a form
#: below it is a COUNTED skip, so the limit can be raised on evidence rather
#: than on the absence of any.
MAX_FORM_DEPTH = 6


#: The transformation that changes nothing, and the starting state of every stream.
_IDENTITY: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _deref(obj):
    """Follow an indirect reference, or return what was already direct."""
    from pypdf.generic import IndirectObject

    if isinstance(obj, IndirectObject):
        obj = obj.get_object()
    return obj if obj is not None else {}


def _form_matrix(form) -> tuple[float, ...]:
    """A form's own `/Matrix`, which maps form space into the space that invoked it.

    Absent, it is the identity. Malformed, it is also the identity: a matrix that
    cannot be read is not a reason to lose every picture the form draws, and the
    placement is still recorded where the rest of the transformation puts it.
    """
    try:
        values = _deref(form.get("/Matrix"))
        if values and len(values) == 6:
            return tuple(float(v) for v in values)
    except Exception:
        pass
    return _IDENTITY


def _concat(a, b) -> tuple[float, ...]:
    """a then b, in pdf order: the new matrix multiplies into the current one."""
    return (
        a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5],
    )


def _apply(m, x: float, y: float) -> tuple[float, float]:
    """One point under a transformation."""
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _path_points(operator: bytes, operands) -> list[tuple[float, float]]:
    """The points a construction operator contributes, in user space."""
    try:
        values = [float(v) for v in operands]
    except (TypeError, ValueError):
        return []
    shape = _PATH_OPS.get(operator)
    if shape == "rect" and len(values) == 4:
        x, y, w, h = values
        return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    if shape in ("point", "points"):
        return [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
    return []


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
    version = "0.6.0"
    extensions = (".pdf",)
    fact_sets = {
        "outline_entry": OUTLINE_FACTS,
        "page": PAGE_FACTS,
        "text": TEXT_FACTS,
        "figure": FIGURE_FACTS,
        "drawing": DRAWING_FACTS,
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

    def _boxes(self, page) -> list[tuple[str, object, tuple[float, float, float, float]]]:
        """Every image placed on this page, with the box the transformations gave it.

        A page description may put a picture on the page without ever naming it in
        the page's own resources: a Form XObject is a content stream invoked by
        name, and what it draws is on the page exactly as if it had been drawn
        directly. So the walk descends, and the transformation composes across the
        nesting -- the form's own `/Matrix` included, which maps form space into
        the space that invoked it.

        Four corpus documents stored images, drew them inside forms, and produced
        nothing at all from a reader that walked the page stream alone: a `Do`
        naming a form matched neither the image branch nor the unresolved one, so
        nothing emitted it and nothing counted it either (K6.10).
        """
        return self._content(page)[0]

    def _content(self, page) -> tuple[list, list[dict]]:
        """Both harvests of one walk: image placements, and painted paths.

        Two walks would parse every content stream on every page twice, and the
        descent into forms is the expensive half.
        """
        found: list[tuple[str, object, tuple[float, float, float, float]]] = []
        drawings: list[dict] = []
        self._descend(page.get_contents(), page.get("/Resources"), page.pdf,
                      _IDENTITY, 0, (), found, drawings)
        return found, drawings

    def _descend(self, source, resources, pdf, ctm, depth, seen, found, drawings) -> None:
        """Walk one content stream, following the forms it invokes.

        `seen` is the chain of forms currently open, by object number: a form that
        invokes one already in that chain would never terminate. `depth` bounds
        the descent for the chains that terminate only eventually. Both bounds
        COUNT when they bite -- a descent that stops in silence loses exactly the
        placements it was written to find.
        """
        from pypdf.generic import ContentStream

        xobjects = _deref(_deref(resources).get("/XObject") if resources else None)
        catalogue: dict[str, tuple[str, object, object]] = {}
        for name in list(xobjects or {}):
            # raw_get, because subscripting resolves the reference and the object
            # number is what tells a cycle from a form merely invoked twice.
            try:
                ref = xobjects.raw_get(name)
            except Exception:
                ref = xobjects[name]
            try:
                obj = _deref(ref)
                subtype = obj.get("/Subtype")
            except Exception:
                obj, subtype = None, None
            # A reference that resolves to nothing has no subtype, and a resource
            # the reader cannot classify must not fall through every branch in
            # silence -- that is the defect this proposition exists to close.
            catalogue[str(name)] = (
                str(subtype) if subtype is not None else "unresolvable", obj, ref)
        # No early return on an empty catalogue: a page whose only content is ink
        # has no XObject resources at all, and skipping it would make the vector
        # lane blind to exactly the pages it exists for.

        try:
            content = ContentStream(source, pdf)
        except Exception:
            # A stream the reader cannot open is a resource it cannot honour, and
            # every placement inside it is one it will not see. Counted as such.
            self._skip("xobject-unresolvable")
            return

        stack: list[tuple[float, ...]] = []
        points: list[tuple[float, float]] = []
        ops = 0
        for operands, operator in content.operations:
            if operator in _PATH_OPS:
                # Construction happens in the space in force NOW: a `cm` after the
                # path is built moves later marks, not this one.
                points.extend(_apply(ctm, x, y)
                              for x, y in _path_points(operator, operands))
                ops += 1
                continue
            if operator in _PAINT_OPS:
                if ops and points:
                    stroke, fill = _PAINT_OPS[operator]
                    xs = [pt[0] for pt in points]
                    ys = [pt[1] for pt in points]
                    drawings.append({
                        "x": min(xs), "y": min(ys),
                        "w": max(xs) - min(xs), "h": max(ys) - min(ys),
                        "ops": ops, "stroke": stroke, "fill": fill,
                    })
                points, ops = [], 0
                continue
            if operator == b"q":
                stack.append(ctm)
            elif operator == b"Q":
                ctm = stack.pop() if stack else _IDENTITY
            elif operator == b"cm" and len(operands) == 6:
                ctm = _concat(tuple(float(v) for v in operands), ctm)
            elif operator == b"Do" and operands:
                entry = catalogue.get(str(operands[0]))
                if entry is None:
                    continue
                subtype, obj, ref = entry
                if subtype == "/Image":
                    found.append((str(operands[0]), obj, _unit_square(ctm)))
                elif subtype == "/Form":
                    self._enter(obj, ref, pdf, ctm, depth, seen, found, drawings)
                elif subtype == "unresolvable":
                    self._skip("xobject-unresolvable")

    def _enter(self, form, ref, pdf, ctm, depth, seen, found, drawings) -> None:
        """Follow one form, unless following it would not terminate."""
        key = getattr(ref, "idnum", None)
        if key is not None and key in seen:
            self._skip("form-cycle")
            return
        if depth + 1 > MAX_FORM_DEPTH:
            self._skip("form-too-deep")
            return
        chain = seen + (key,) if key is not None else seen
        self._descend(form, form.get("/Resources"), pdf,
                      _concat(_form_matrix(form), ctm), depth + 1, chain,
                      found, drawings)

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

    def _figures(self, page, number: int, found) -> Iterator[Mastaba]:
        if self.store is None:
            return
        for _ in range(self._inline_count(page)):
            self._skip("inline-image-not-extracted")

        for name, obj, box in found:
            try:
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

        found, drawings = self._content(page)
        yield from self._figures(page, number, found)

        # Ink, as read: how many operators over what extent. Whether any of it
        # amounts to a figure is a judgement, and it is made elsewhere (K6.14).
        for mark in drawings:
            yield Mastaba(
                kind="drawing",
                locator=PdfLocator(page=number),
                facts={
                    "x": round(mark["x"], 2), "y": round(mark["y"], 2),
                    "w": round(mark["w"], 2), "h": round(mark["h"], 2),
                    "ops": mark["ops"],
                    "stroke": mark["stroke"], "fill": mark["fill"],
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
