"""
Kernel: pres_layout_assign
Stage: 2 (Structuring)

Assigns MARP layout directives to each slide based on its SlideType.
Reads ThemeConfig for color integration. Produces an updated SlideDeck
with SlideLayout populated on every slide.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import (
    AssetCatalog,
    Slide,
    SlideDeck,
    SlideLayout,
    SlideType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout rules: SlideType → default SlideLayout
# ---------------------------------------------------------------------------

def _default_layout(slide_type: SlideType, colors: Dict[str, str]) -> SlideLayout:
    """
    Return a default SlideLayout for a given SlideType.

    Args:
        slide_type: The type of slide.
        colors: Theme color dict with keys: primary, secondary, accent, background, text.

    Returns:
        SlideLayout with appropriate directives.
    """
    bg = colors.get("primary", "#0066cc")

    if slide_type == SlideType.TITLE:
        return SlideLayout(
            css_class="lead",
            paginate=False,
            background_color=bg,
        )

    if slide_type == SlideType.SECTION:
        return SlideLayout(
            css_class="lead",
            paginate=False,
            background_color=bg,
        )

    if slide_type == SlideType.IMAGE_TEXT:
        return SlideLayout(
            bg_position="left",
            bg_size="40%",
        )

    if slide_type == SlideType.IMAGE_FULL:
        return SlideLayout(
            css_class="invert",
            bg_size="contain",
        )

    if slide_type == SlideType.QUOTE:
        return SlideLayout(css_class="quote")

    if slide_type == SlideType.SUMMARY:
        return SlideLayout(
            css_class="lead",
            paginate=False,
        )

    if slide_type == SlideType.BLANK:
        return SlideLayout(paginate=False)

    # CONTENT, TABLE, CODE, EQUATION, TWO_COLUMN → default
    return SlideLayout()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresLayoutAssignKernel(Kernel):
    """Assign MARP layout directives to each slide."""

    name = "pres_layout_assign"
    version = "1.0.0"
    category = "presenter"
    stage = 2
    description = "Slide type → MARP layout directives (class, bg, paginate)"

    requires: List[str] = ["pres_slide_plan", "pres_asset_catalog"]
    provides: List[str] = ["slide_deck_with_layout"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load slide deck
        plan_path = input.dependencies["pres_slide_plan"]
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        deck = SlideDeck.from_dict(plan_data["data"])

        # Load asset catalog (for image dimensions)
        catalog_path = input.dependencies["pres_asset_catalog"]
        catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = AssetCatalog.from_dict({
            "assets": catalog_data["data"].get("assets", []),
            "by_type": catalog_data["data"].get("by_type", {}),
            "by_file": catalog_data["data"].get("by_file", {}),
        })

        # Theme colors from config
        theme_cfg = input.config.get("theme", {})
        colors = theme_cfg.get("colors", {
            "primary": "#0066cc",
            "secondary": "#2d3436",
            "accent": "#e17055",
            "background": "#ffffff",
            "text": "#2d3436",
        })

        # Apply layout to each slide
        layouts_assigned = 0
        inline_images = 0
        table_scaled = 0
        for slide in deck.slides:
            layout = _default_layout(slide.type, colors)

            # --- Aspect-ratio-aware image layout (v1.2) ---
            # All images use inline rendering to avoid MARP bg cropping.
            # Portrait → side panel (.figure), Landscape → full-width top (.figure-landscape).
            if slide.type in (SlideType.IMAGE_TEXT, SlideType.IMAGE_FULL):
                asset = None
                if slide.content.image:
                    if slide.content.image.asset_id:
                        asset = catalog.get(slide.content.image.asset_id)
                    if asset is None and slide.content.image.path:
                        asset = catalog.get_by_path(slide.content.image.path)

                # Default: inline rendering for all images with known dimensions
                if asset and asset.dimensions:
                    w, h = asset.dimensions
                    if w > 0 and h > 0:
                        layout.inline_image = True
                        layout.bg_size = ""
                        layout.bg_position = ""
                        aspect = w / h

                        if slide.type == SlideType.IMAGE_FULL:
                            layout.image_class = "figure-full"
                        elif aspect < 0.9:
                            # Portrait/tall → side panel
                            layout.image_class = "figure"
                        else:
                            # Landscape → full-width top, text below
                            layout.image_class = "figure-landscape"
                        inline_images += 1

            # --- Table density classes ---
            if slide.type == SlideType.TABLE and slide.content.table:
                tbl = slide.content.table
                n_rows = len(tbl.rows)
                n_cols = len(tbl.headers) if tbl.headers else (
                    len(tbl.rows[0]) if tbl.rows else 0
                )
                cells = n_rows * n_cols
                if cells > 80 or n_rows > 12:
                    layout.table_class = "table-tiny"
                    table_scaled += 1
                elif cells > 40 or n_rows > 8:
                    layout.table_class = "table-small"
                    table_scaled += 1

            slide.layout = layout
            layouts_assigned += 1

        result = deck.to_dict()
        result["statistics"] = {
            "total_slides": len(deck.slides),
            "layouts_assigned": layouts_assigned,
            "inline_images": inline_images,
            "table_scaled": table_scaled,
        }
        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        total = stats.get("total_slides", 0)
        return f"Layout assignment: {total} slides with MARP directives"
