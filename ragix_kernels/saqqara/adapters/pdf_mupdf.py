"""
saqqara.adapters.pdf_mupdf — the OPT-IN text reader. AGPL-3.0, one of the two files allowed to reach it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

LICENCE WARNING
---------------
This module uses **pymupdf**, which is **AGPL-3.0** or a paid commercial licence from its vendor.
Importing it subjects the running system to AGPL obligations that are **incompatible with
distributing this package under MIT**.

Nothing imports this module unless a caller sets the pdf reader option `text_reader: pymupdf`. It is
not installed by the base requirements, but IS pulled in by `all` through `translate`. It is not
required by the default Saqqara reader path; the
default text reader is `pypdf` (BSD-3-Clause). The `saqqara-mupdf` extra installs it, for a caller who
holds a commercial licence or is not distributing. The guard of K6.16 (`render/guard.py`) names this
file and `render/mupdf.py` as the only two allowed to import it, and proves at run time that no
default route loaded it.

What it reads: **the text of each page, one observation per span** — a run of characters in one font
and size on one line, which the library assembles from glyph positions rather than from the order of
the text-showing operations. The outline, the pages, the pictures and the ink are still read by
`pypdf` whichever text reader is chosen, so choosing this one changes the text records and nothing
else. Two readers may disagree about the same page, which is why a tree read this way says so in its
provenance: the reader's version carries `+pymupdf-<library version>`.

Positions: `x` and `y` are the span's origin in the page's own space, in points with y upward, and
`width` is the extent of the span's box along the line — measured by the library from the glyphs it
placed, never estimated here.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["MuPdfTextReader"]


class MuPdfTextReader:
    """Read a page's text with the AGPL library, on purpose."""

    name = "pymupdf"

    def __init__(self) -> None:
        try:
            import pymupdf  # noqa: F401  # AGPL-3.0 -- see the module docstring
        except ImportError as exc:
            raise ImportError(
                "pymupdf is not installed. It is the `saqqara-mupdf` extra, it is "
                "AGPL-3.0, and installing it changes what you may distribute. The "
                "default text reader is pypdf and needs nothing."
            ) from exc
        self.version = str(getattr(pymupdf, "__version__", "?"))
        self._document = None

    def open(self, path) -> None:
        import pymupdf

        self._document = pymupdf.open(str(Path(path)))

    def close(self) -> None:
        if self._document is not None:
            self._document.close()
            self._document = None

    def page_geometry(self, number: int, *, source_id: str):
        """Opt-in glyph observations and drawn vertical rules for derived field views.

        Leaves the existing placement contract unchanged. Coordinates are unrotated
        top-left page points; source identity is supplied by the caller's manifest.
        Reader failures propagate rather than becoming an empty page.
        """
        from ..field_views import TextSpan, VerticalRule, stable_id

        if self._document is None:
            raise RuntimeError("reader is not open")
        if not 1 <= number <= len(self._document):
            raise ValueError("page number outside document")
        page = self._document[number - 1]
        spans, rules = [], []
        for block in page.get_text("rawdict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    chars = span.get("chars", [])
                    text = "".join(c["c"] for c in chars)
                    if not text.strip():
                        continue
                    boxes = tuple(tuple(c["bbox"]) for c in chars for _ in c["c"])
                    ident = stable_id("mupdf-glyphs/1.0", self.version, source_id, number, len(spans))
                    spans.append(TextSpan(source_id, ident, number, text, tuple(span["bbox"]), boxes,
                                          tuple(span["origin"]), tuple(line["dir"]), span["size"]))
        for path in page.get_drawings():
            for item in path["items"]:
                segments = []
                if item[0] == "l":
                    segments.append((item[1], item[2]))
                elif item[0] == "re":
                    rect = item[1]
                    segments.extend(((rect.top_left, rect.bottom_left), (rect.top_right, rect.bottom_right)))
                for a, b in segments:
                    if abs(a.x - b.x) <= 0.01 and abs(a.y - b.y) > 1:
                        rules.append(VerticalRule((a.x + b.x) / 2, min(a.y, b.y), max(a.y, b.y)))
        return {"spans": spans, "vertical_rules": rules, "width": page.cropbox.width,
                "height": page.cropbox.height, "rotation": page.rotation,
                "coordinate_system": "unrotated-top-left-pt", "extractor": self.name,
                "extractor_version": self.version, "producer": "mupdf-glyphs/1.0",
                "transformation_matrix": tuple(page.transformation_matrix)}

    def placements(self, number: int, geometry: list | None = None) -> list[tuple]:
        """The spans of one page, as the pdf reader holds its text placements.

        Returns (text, x, y, font size, font, width) per span carrying anything but
        white space, in the order the library found them. When `geometry` is a list
        it receives, per placement, what the line join needs: the origin, whether the
        line is upright, and whether the raw span began or ended with white space.
        """
        import pymupdf

        page = self._document[number - 1]
        # The library's space has y downward from the top of the page; the inverse
        # of the page's transformation takes a point back to the file's own space.
        to_pdf = ~page.transformation_matrix
        flags = getattr(pymupdf, "TEXTFLAGS_TEXT", 0)
        found: list[tuple] = []
        for block in page.get_text("dict", flags=flags).get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                direction = tuple(float(v) for v in line.get("dir", (1.0, 0.0)))
                upright = abs(direction[0] - 1.0) < 1e-6 and abs(direction[1]) < 1e-6
                for span in line.get("spans", []):
                    raw = span.get("text") or ""
                    if not raw.strip():
                        continue
                    origin = pymupdf.Point(*span["origin"]) * to_pdf
                    left, _top, right, _bottom = span["bbox"]
                    found.append((raw.strip(), float(origin.x), float(origin.y),
                                  float(span.get("size") or 0), str(span.get("font", "")),
                                  float(right - left)))
                    if geometry is not None:
                        geometry.append((float(origin.x), float(origin.y), upright,
                                         raw[:1].isspace(), raw[-1:].isspace()))
        return found
