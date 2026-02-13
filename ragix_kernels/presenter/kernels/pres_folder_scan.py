"""
Kernel: pres_folder_scan
Stage: 1 (Collection)

Recursive folder discovery, file classification by extension, SHA-256 hashing,
line counting, and front-matter detection for Markdown files.

Produces a flat file inventory used by downstream kernels
(pres_content_extract, pres_asset_catalog).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import FileEntry, FileType
from ragix_kernels.presenter.md_parse_utils import detect_front_matter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → FileType mapping
# ---------------------------------------------------------------------------

_DOCUMENT_EXTS = {".md", ".txt", ".rst", ".markdown", ".mdown"}
_ASSET_EXTS = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".webp", ".bmp", ".ico"}
_DATA_EXTS = {".json", ".yaml", ".yml", ".csv", ".tsv", ".xml"}
_CONFIG_EXTS = {".toml", ".ini", ".cfg", ".conf", ".env"}


def _classify_extension(ext: str) -> FileType:
    """Map a lowercase file extension to a FileType enum value."""
    ext = ext.lower()
    if ext in _DOCUMENT_EXTS:
        return FileType.DOCUMENT
    if ext in _ASSET_EXTS:
        return FileType.ASSET
    if ext in _DATA_EXTS:
        return FileType.DATA
    if ext in _CONFIG_EXTS:
        return FileType.CONFIG
    return FileType.UNKNOWN


def _matches_any(rel_path: str, patterns: List[str]) -> bool:
    """Check if *rel_path* matches any of the glob/fnmatch *patterns*.

    Handles ``**/X`` patterns by also matching root-level files against
    the suffix after ``**/``.  Falls back to fnmatch for simple patterns.
    """
    name = Path(rel_path).name
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if fnmatch.fnmatch(name, pat):
            return True
        # Handle **/ prefix: also match the part after **/
        if pat.startswith("**/"):
            suffix = pat[3:]  # e.g. "*.md" from "**/*.md"
            if fnmatch.fnmatch(rel_path, suffix):
                return True
            if fnmatch.fnmatch(name, suffix):
                return True
    return False


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file using chunked reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _count_lines(path: Path) -> int:
    """Count lines in a text file. Returns 0 if the file cannot be read as text."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresFolderScanKernel(Kernel):
    """Recursive folder scan with classification and hashing."""

    name = "pres_folder_scan"
    version = "1.0.0"
    category = "presenter"
    stage = 1
    description = "Recursive folder walk, file classification, SHA-256 hashing"

    requires: List[str] = []
    provides: List[str] = ["file_inventory"]

    def validate_input(self, input: KernelInput) -> List[str]:
        errors = super().validate_input(input)
        folder_path = input.config.get("folder_path", "")
        if not folder_path:
            errors.append("Missing required config: folder_path")
        elif not Path(folder_path).is_dir():
            errors.append(f"folder_path is not a directory: {folder_path}")
        return errors

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        folder = Path(input.config["folder_path"]).resolve()
        scan_cfg = input.config.get("folder_scan", {})

        include_patterns = scan_cfg.get("include_patterns", [
            "**/*.md", "**/*.txt", "**/*.rst",
        ])
        asset_patterns = scan_cfg.get("asset_patterns", [
            "**/*.svg", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.pdf",
        ])
        exclude_patterns = scan_cfg.get("exclude_patterns", [
            "**/node_modules/**", "**/.git/**", "**/.*", "**/__pycache__/**",
        ])
        max_depth = scan_cfg.get("max_depth", 10)
        follow_symlinks = scan_cfg.get("follow_symlinks", False)

        files: List[FileEntry] = []
        stats: Dict[str, int] = {
            "total_files": 0,
            "documents": 0,
            "assets": 0,
            "data": 0,
            "config": 0,
            "unknown": 0,
            "excluded": 0,
        }

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue

            # Respect follow_symlinks
            if path.is_symlink() and not follow_symlinks:
                continue

            rel = path.relative_to(folder)
            rel_str = str(rel)

            # Depth check
            if len(rel.parts) > max_depth:
                continue

            # Exclusion
            if _matches_any(rel_str, exclude_patterns):
                stats["excluded"] += 1
                continue

            # Classify
            ext = path.suffix.lower()
            file_type = _classify_extension(ext)

            # Accept: documents match include_patterns, assets match asset_patterns,
            # data/config match by extension, unknown is skipped unless explicitly included
            accepted = False
            if file_type == FileType.DOCUMENT:
                accepted = _matches_any(rel_str, include_patterns)
            elif file_type == FileType.ASSET:
                accepted = _matches_any(rel_str, asset_patterns)
            elif file_type in (FileType.DATA, FileType.CONFIG):
                accepted = True
            else:
                # UNKNOWN — accept only if it matches include or asset patterns
                accepted = _matches_any(rel_str, include_patterns) or _matches_any(rel_str, asset_patterns)

            if not accepted:
                continue

            # Hash
            file_hash = _hash_file(path)

            # Line count (documents only)
            line_count = _count_lines(path) if file_type == FileType.DOCUMENT else 0

            # Front-matter detection (Markdown only)
            front_matter: Optional[Dict[str, Any]] = None
            if ext in (".md", ".markdown", ".mdown"):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    fm = detect_front_matter(text.splitlines())
                    if fm is not None:
                        front_matter = fm[2]
                except Exception:
                    pass

            entry = FileEntry(
                path=rel_str,
                file_type=file_type,
                size_bytes=path.stat().st_size,
                file_hash=file_hash,
                line_count=line_count,
                extension=ext,
                front_matter=front_matter,
            )
            files.append(entry)
            stats["total_files"] += 1
            _type_key = {
                FileType.DOCUMENT: "documents",
                FileType.ASSET: "assets",
                FileType.DATA: "data",
                FileType.CONFIG: "config",
                FileType.UNKNOWN: "unknown",
            }
            stats[_type_key.get(file_type, "unknown")] += 1

        logger.info(
            f"[pres_folder_scan] {folder}: {stats['total_files']} files "
            f"({stats['documents']} docs, {stats['assets']} assets, "
            f"{stats['excluded']} excluded)"
        )

        return {
            "root": str(folder),
            "files": [f.to_dict() for f in files],
            "statistics": stats,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        s = data.get("statistics", {})
        return (
            f"Folder scan: {s.get('total_files', 0)} files "
            f"({s.get('documents', 0)} documents, {s.get('assets', 0)} assets) "
            f"in {data.get('root', '?')}"
        )
