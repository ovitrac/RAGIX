"""
saqqara.render.mupdf — the OPT-IN renderer. AGPL-3.0, and the only file allowed to reach it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-29

LICENCE WARNING
---------------
This module uses **pymupdf**, which is **AGPL-3.0** or a paid commercial licence from its vendor.
Importing it subjects the running system to AGPL obligations that are **incompatible with
distributing this package under MIT**.

Nothing in this package imports this module. It is not installed by default, not pulled in by the
`all` extra, and not required by any test; the default renderer is `pypdfium2` (Apache-2.0). This
file exists so that a caller who has *deliberately* installed the `saqqara-mupdf` extra — because
they hold a commercial licence, or are not distributing — can construct the renderer explicitly.

K6.16 exists because a licence obligation that depends on which code path happened to run is not an
obligation anyone can audit. The guard beside this file enforces that this is the single exception,
and a test asserts the module is absent from `sys.modules` after every default route.
"""

from __future__ import annotations

from pathlib import Path

from . import RenderFailed

__all__ = ["MuPdfRenderer"]


class MuPdfRenderer:
    """Rasterise a rectangle of a page with the AGPL library, on purpose."""

    name = "pymupdf"

    def __init__(self) -> None:
        try:
            import pymupdf  # noqa: F401  # AGPL-3.0 -- see the module docstring
        except ImportError as exc:                     # pragma: no cover - opt-in
            raise RenderFailed(
                "pymupdf is not installed. It is the `saqqara-mupdf` extra, it is "
                "AGPL-3.0, and installing it changes what you may distribute. The "
                "default renderer is pypdfium2 and needs nothing."
            ) from exc
        self.version = self._version()

    @staticmethod
    def _version() -> str:                             # pragma: no cover - opt-in
        import pymupdf

        return str(getattr(pymupdf, "__version__", "?"))

    def render(self, path, page_number: int, box, dpi: int) -> tuple[bytes, str]:  # pragma: no cover - opt-in
        import pymupdf

        left, bottom, right, top = (float(v) for v in box)
        try:
            document = pymupdf.open(str(Path(path)))
            page = document[page_number - 1]
            height = page.rect.height
            # pdf space has y upward; this library's rectangles have y downward.
            clip = pymupdf.Rect(left, height - top, right, height - bottom)
            pixmap = page.get_pixmap(dpi=dpi, clip=clip)
            payload = pixmap.tobytes("png")
            document.close()
        except Exception as exc:
            raise RenderFailed(f"{self.name} could not render page {page_number}") from exc
        return payload, "image/png"
