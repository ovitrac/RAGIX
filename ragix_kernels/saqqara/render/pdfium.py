"""
saqqara.render.pdfium — the default renderer, Apache-2.0.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-29

Chosen as the default because of its licence, not despite it: it imposes nothing on a deployment of
this MIT package, so the kernel's default path can never surprise anyone about what they may ship.
"""

from __future__ import annotations

import io
from pathlib import Path

from . import RenderFailed

__all__ = ["PdfiumRenderer"]


class PdfiumRenderer:
    """Rasterise a rectangle of a page."""

    name = "pypdfium2"

    def __init__(self) -> None:
        try:
            import pypdfium2  # noqa: F401
        except ImportError as exc:                     # pragma: no cover - env
            raise RenderFailed(
                "pypdfium2 is not installed: it is declared by the `saqqara` extra"
            ) from exc
        self.version = self._version()

    @staticmethod
    def _version() -> str:
        """Renderer and library version together: both change the pixels."""
        import pypdfium2

        info = getattr(pypdfium2, "version", None)
        return "%s+pdfium%s" % (getattr(info, "PYPDFIUM_INFO", "?"),
                                getattr(info, "PDFIUM_INFO", "?"))

    def render(self, path, page_number: int, box, dpi: int) -> tuple[bytes, str]:
        import pypdfium2

        left, bottom, right, top = (float(v) for v in box)
        document = pypdfium2.PdfDocument(str(Path(path)))
        try:
            page = document[page_number - 1]
            width, height = page.get_size()
            # `crop` is how much to cut from each edge, not a rectangle. Clamped
            # at zero: a region may legitimately touch or overrun a page edge,
            # and a negative margin would be a silent failure rather than a
            # picture of the part that is actually there.
            crop = (max(0.0, left), max(0.0, bottom),
                    max(0.0, width - right), max(0.0, height - top))
            if crop[0] + crop[2] >= width or crop[1] + crop[3] >= height:
                raise RenderFailed("region leaves no area to render")
            bitmap = page.render(scale=dpi / 72.0, crop=crop)
            buffer = io.BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
        except RenderFailed:
            raise
        except Exception as exc:
            raise RenderFailed(f"{self.name} could not render page {page_number}") from exc
        finally:
            document.close()
        return buffer.getvalue(), "image/png"
