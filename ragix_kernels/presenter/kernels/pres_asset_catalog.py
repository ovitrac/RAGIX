"""
Kernel: pres_asset_catalog
Stage: 1 (Collection)

Unified inventory of presentable assets: images, equations, tables,
code blocks, and diagrams. Combines file-system assets (from folder scan)
with embedded assets (from content extraction).

Image dimensions: SVG via regex on root element, raster via PIL (soft dep).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import (
    Asset,
    AssetCatalog,
    AssetType,
    FileEntry,
    FileType,
    SemanticUnit,
    UnitType,
)

logger = logging.getLogger(__name__)

# Soft dependency on Pillow
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# SVG dimension patterns
_SVG_WIDTH_RE = re.compile(r'<svg[^>]*\bwidth\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_SVG_HEIGHT_RE = re.compile(r'<svg[^>]*\bheight\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_SVG_VIEWBOX_RE = re.compile(
    r'<svg[^>]*\bviewBox\s*=\s*["\']'
    r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*["\']',
    re.IGNORECASE,
)

# UnitType → AssetType mapping for embedded assets
_UNIT_TO_ASSET: Dict[UnitType, AssetType] = {
    UnitType.EQUATION_BLOCK: AssetType.EQUATION,
    UnitType.TABLE: AssetType.TABLE,
    UnitType.CODE_BLOCK: AssetType.CODE,
    UnitType.MERMAID: AssetType.DIAGRAM,
    UnitType.IMAGE_REF: AssetType.IMAGE,
}

# Extension → format mapping
_EXT_TO_FORMAT = {
    ".svg": "svg", ".png": "png", ".jpg": "jpg", ".jpeg": "jpg",
    ".gif": "gif", ".webp": "webp", ".bmp": "bmp", ".pdf": "pdf",
}


def _get_image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    """Get (width, height) for an image file. Returns None if unavailable."""
    ext = path.suffix.lower()

    # SVG: parse from XML
    if ext == ".svg":
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:4096]
            w_m = _SVG_WIDTH_RE.search(content)
            h_m = _SVG_HEIGHT_RE.search(content)
            if w_m and h_m:
                w = _parse_svg_dim(w_m.group(1))
                h = _parse_svg_dim(h_m.group(1))
                if w and h:
                    return (w, h)
            # Fallback to viewBox
            vb_m = _SVG_VIEWBOX_RE.search(content)
            if vb_m:
                w = int(float(vb_m.group(3)))
                h = int(float(vb_m.group(4)))
                if w > 0 and h > 0:
                    return (w, h)
        except Exception:
            pass
        return None

    # Raster: use PIL
    if HAS_PIL and ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        try:
            with PILImage.open(path) as img:
                return img.size
        except Exception:
            pass

    return None


def _parse_svg_dim(val: str) -> Optional[int]:
    """Parse an SVG dimension string (e.g. '100', '100px', '10em') to int pixels."""
    val = val.strip().lower()
    # Strip common units
    for suffix in ("px", "pt", "em", "rem", "%", "mm", "cm", "in"):
        if val.endswith(suffix):
            val = val[:-len(suffix)].strip()
            break
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


class PresAssetCatalogKernel(Kernel):
    """Unified asset inventory from file-system and embedded content."""

    name = "pres_asset_catalog"
    version = "1.0.0"
    category = "presenter"
    stage = 1
    description = "Image/equation/table/code/diagram inventory"

    requires: List[str] = ["pres_folder_scan", "pres_content_extract"]
    provides: List[str] = ["asset_catalog"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load dependencies
        scan_path = input.dependencies["pres_folder_scan"]
        scan_data = json.loads(scan_path.read_text(encoding="utf-8"))
        folder_root = Path(scan_data["data"]["root"])
        files = [FileEntry.from_dict(f) for f in scan_data["data"]["files"]]

        extract_path = input.dependencies["pres_content_extract"]
        extract_data = json.loads(extract_path.read_text(encoding="utf-8"))
        units = [SemanticUnit.from_dict(u) for u in extract_data["data"].get("units", [])]

        assets: List[Asset] = []
        seen_paths: Dict[str, str] = {}  # rel_path → asset_id (dedup)
        asset_counter = 0

        def _next_id() -> str:
            nonlocal asset_counter
            asset_counter += 1
            return f"asset-{asset_counter:03d}"

        # --- 1. File-system assets ---
        for fe in files:
            if fe.file_type != FileType.ASSET:
                continue

            full_path = folder_root / fe.path
            ext = fe.extension.lower()
            fmt = _EXT_TO_FORMAT.get(ext)
            dims = _get_image_dimensions(full_path) if full_path.exists() else None

            aid = _next_id()
            seen_paths[fe.path] = aid

            assets.append(Asset(
                id=aid,
                type=AssetType.IMAGE,
                source_file=fe.path,
                source_lines=(0, 0),
                content="",
                path=fe.path,
                format=fmt,
                dimensions=dims,
            ))

        # --- 2. Embedded assets from content units ---
        for u in units:
            if u.type not in _UNIT_TO_ASSET:
                continue

            asset_type = _UNIT_TO_ASSET[u.type]

            # IMAGE_REF: resolve against file-system, deduplicate
            if u.type == UnitType.IMAGE_REF:
                img_path = u.metadata.get("path", "")
                if img_path in seen_paths:
                    # Already cataloged as file-system asset — link back
                    u.metadata["asset_id"] = seen_paths[img_path]
                    continue

                aid = _next_id()
                seen_paths[img_path] = aid
                u.metadata["asset_id"] = aid

                assets.append(Asset(
                    id=aid,
                    type=AssetType.IMAGE,
                    source_file=u.source_file,
                    source_lines=u.source_lines,
                    content=u.content,
                    path=img_path,
                    alt_text=u.metadata.get("alt", ""),
                ))
                continue

            # Other embedded assets (equation, table, code, mermaid)
            aid = _next_id()
            meta: Dict[str, Any] = {}
            fmt = None

            if u.type == UnitType.MERMAID:
                fmt = "mermaid"
                meta["diagram_type"] = u.metadata.get("diagram_type", "")
            elif u.type == UnitType.CODE_BLOCK:
                meta["language"] = u.metadata.get("language", "")
            elif u.type == UnitType.TABLE:
                meta["rows"] = u.metadata.get("rows", 0)
                meta["cols"] = u.metadata.get("cols", 0)

            assets.append(Asset(
                id=aid,
                type=asset_type,
                source_file=u.source_file,
                source_lines=u.source_lines,
                content=u.content,
                format=fmt,
            ))

        # --- 3. Build index maps ---
        by_type: Dict[str, List[str]] = {}
        by_file: Dict[str, List[str]] = {}
        for a in assets:
            by_type.setdefault(a.type.value, []).append(a.id)
            by_file.setdefault(a.source_file, []).append(a.id)

        catalog = AssetCatalog(assets=assets, by_type=by_type, by_file=by_file)

        stats = {
            "total_assets": len(assets),
            "file_system_assets": sum(1 for a in assets if a.source_lines == (0, 0)),
            "embedded_assets": sum(1 for a in assets if a.source_lines != (0, 0)),
            "by_type_count": {k: len(v) for k, v in by_type.items()},
        }

        logger.info(
            f"[pres_asset_catalog] {stats['total_assets']} assets "
            f"({stats['file_system_assets']} file-system, "
            f"{stats['embedded_assets']} embedded)"
        )

        result = catalog.to_dict()
        result["statistics"] = stats
        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        by_type = stats.get("by_type_count", {})
        parts = [f"{v} {k}" for k, v in sorted(by_type.items())]
        return (
            f"Asset catalog: {stats.get('total_assets', 0)} assets "
            f"({', '.join(parts) if parts else 'none'})"
        )
