"""
saqqara.render — turning a region of a page into pixels, and keeping pixels out of identity.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-29

Specified by K6.15 and K6.16 in SPEC.md.

A renderer is a **port**, and the pipeline never names a vendor. That is not decoration here: the
two renderers this package can use are under different licences, and one of them changes what a
deployment of this MIT package may legally be. A caller chooses; nothing chooses for them.

**A raster is never identity.** What a region *is* is the operators that draw it and the extent they
cover. What it *looks like* depends on a renderer, its version and a resolution, all of which change
without the document changing at all. So the raster is a derived artefact under a key of its own,
and two renderers may disagree about every pixel while the tree stays byte-identical (K4).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

__all__ = [
    "RENDER_DPI",
    "Renderer",
    "RenderFailed",
    "default_renderer",
    "raster_key",
    "source_id",
]

#: Rasterisation resolution. Part of the raster's cache key and of nothing else.
RENDER_DPI = 150


class RenderFailed(RuntimeError):
    """A region could not be rasterised. Counted by the caller, never swallowed."""


class Renderer(Protocol):
    """What the kernel needs of a renderer, and nothing more."""

    name: str
    version: str

    def render(self, path, page_number: int, box, dpi: int) -> tuple[bytes, str]:
        """Rasterise `box` (left, bottom, right, top, in points) of one page."""


def source_id(members: list[dict[str, Any]], box) -> bytes:
    """The canonical description of what a region IS: its marks and their extent.

    Hashed for identity and **stored verbatim**, so a node's asset is a thing the
    store can actually produce rather than a hash of something nobody kept. This
    is the description a raster is derived FROM, and it moves only when the
    drawing moves.
    """
    payload = {
        "box": [round(float(v), 3) for v in box],
        "marks": sorted(
            [
                {
                    "x": round(float(m["x"]), 3), "y": round(float(m["y"]), 3),
                    "w": round(float(m["w"]), 3), "h": round(float(m["h"]), 3),
                    "ops": int(m["ops"]),
                    "stroke": bool(m.get("stroke")), "fill": bool(m.get("fill")),
                }
                for m in members
            ],
            key=lambda m: (m["x"], m["y"], m["w"], m["h"], m["ops"]),
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def raster_key(source_digest: str, renderer: str, version: str, dpi: int) -> str:
    """Where a rendering of a source lives: source, renderer, version, resolution.

    Every term is here because changing it changes the pixels. None of them is
    part of what the region is, which is why this key is not the node's identity.
    """
    material = f"{source_digest}|{renderer}|{version}|{dpi}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def default_renderer() -> Renderer:
    """The renderer the kernel uses unless a caller supplies another.

    Always the permissively licensed one. The alternative exists, is never
    reached from here, and is only ever constructed by a caller who has installed
    an extra whose licence warning they had to read to find (K6.16).
    """
    from .pdfium import PdfiumRenderer

    return PdfiumRenderer()
