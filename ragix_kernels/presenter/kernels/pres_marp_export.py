"""
Kernel: pres_marp_export
Stage: 3 (Rendering)

Bundle the MARP Markdown and referenced assets into an output folder:
    {workspace}/output/presentation.md
    {workspace}/output/assets/...
    {workspace}/output/metadata.json

Optionally invoke marp-cli for PDF/HTML export.

Copies (or symlinks) referenced images, updates paths in Markdown.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# Path to the bundled theme
_THEMES_DIR = Path(__file__).resolve().parent.parent / "themes"


# ---------------------------------------------------------------------------
# marp-cli invocation
# ---------------------------------------------------------------------------

def _resolve_theme_css(
    theme_name: Optional[str],
    custom_css_path: Optional[str],
) -> Optional[Path]:
    """Resolve a theme CSS file path.

    Priority:
        1. Explicit custom_css_path (absolute or relative)
        2. Bundled theme by name (e.g. "koas-professional")
        3. None → let marp-cli use its built-in theme
    """
    if custom_css_path:
        p = Path(custom_css_path)
        if p.exists():
            return p.resolve()
        logger.warning(f"[pres_marp_export] Custom CSS not found: {custom_css_path}")

    if theme_name and theme_name not in ("default", "gaia", "uncover"):
        candidate = _THEMES_DIR / f"{theme_name}.css"
        if candidate.exists():
            return candidate.resolve()
        logger.warning(
            f"[pres_marp_export] Bundled theme '{theme_name}' not found at {candidate}"
        )

    return None


def run_marp_cli(
    presentation_md: Path,
    output_format: str = "pdf",
    theme_css: Optional[Path] = None,
    allow_local_files: bool = True,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Invoke marp-cli as subprocess.

    Args:
        presentation_md: Path to the MARP Markdown file.
        output_format: "pdf" or "html".
        theme_css: Optional path to a custom CSS theme file.
        allow_local_files: Allow local file access for images.
        timeout: Subprocess timeout in seconds.

    Returns:
        Dict with keys: success, output_file, stderr, exit_code.
    """
    if output_format not in ("pdf", "html"):
        return {
            "success": False,
            "output_file": "",
            "stderr": f"Unsupported format: {output_format!r}. Use 'pdf' or 'html'.",
            "exit_code": -1,
        }

    cmd = ["npx", "@marp-team/marp-cli", str(presentation_md)]
    cmd.append(f"--{output_format}")

    if allow_local_files:
        cmd.append("--allow-local-files")

    if theme_css and theme_css.exists():
        cmd.extend(["--theme", str(theme_css)])

    output_ext = f".{output_format}"
    expected_output = presentation_md.with_suffix(output_ext)

    logger.info(f"[marp-cli] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(presentation_md.parent),
        )

        success = result.returncode == 0 and expected_output.exists()

        return {
            "success": success,
            "output_file": str(expected_output) if expected_output.exists() else "",
            "stderr": result.stderr.strip(),
            "stdout": result.stdout.strip(),
            "exit_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output_file": "",
            "stderr": f"marp-cli timed out after {timeout}s",
            "exit_code": -2,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "output_file": "",
            "stderr": "npx not found. Install Node.js and run: npm install @marp-team/marp-cli",
            "exit_code": -3,
        }


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresMarpExportKernel(Kernel):
    """Bundle MARP Markdown + assets into output folder, optionally export to PDF/HTML."""

    name = "pres_marp_export"
    version = "1.1.0"
    category = "presenter"
    stage = 3
    description = "Copy MARP Markdown + assets to output/ folder, optional marp-cli export"

    requires: List[str] = ["pres_marp_render", "pres_folder_scan"]
    provides: List[str] = ["presentation_bundle"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load render output
        render_path = input.dependencies["pres_marp_render"]
        render_data = json.loads(render_path.read_text(encoding="utf-8"))
        marp_md = render_data["data"]["marp_markdown"]
        asset_refs = render_data["data"].get("asset_refs", [])

        # Load folder scan for root path
        scan_path = input.dependencies["pres_folder_scan"]
        scan_data = json.loads(scan_path.read_text(encoding="utf-8"))
        folder_root = Path(scan_data["data"]["root"])

        # Config
        export_cfg = input.config.get("export", {})
        symlink_assets = export_cfg.get("symlink_assets", False)
        export_format = export_cfg.get("format", "md")

        theme_cfg = input.config.get("theme", {})
        theme_name = theme_cfg.get("name", "default")
        custom_css_path = theme_cfg.get("custom_css_path")

        # Create output directory
        output_dir = input.workspace / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Copy assets and build path mapping
        assets_copied = 0
        total_bytes = 0
        path_map: Dict[str, str] = {}  # original_path -> new_relative_path

        for ref in asset_refs:
            src = folder_root / ref
            if not src.exists():
                logger.warning(f"[pres_marp_export] Asset not found: {src}")
                continue

            # Flatten to assets/ directory
            dest_name = Path(ref).name
            # Handle duplicates by prepending parent dir
            if (assets_dir / dest_name).exists():
                parent_name = Path(ref).parent.name
                dest_name = f"{parent_name}_{dest_name}" if parent_name else dest_name
            dest = assets_dir / dest_name

            try:
                if symlink_assets:
                    dest.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dest)
                assets_copied += 1
                total_bytes += dest.stat().st_size
                path_map[ref] = f"assets/{dest_name}"
            except Exception as e:
                logger.warning(f"[pres_marp_export] Failed to copy {src}: {e}")

        # Update image paths in Markdown
        updated_md = marp_md
        for old_path, new_path in path_map.items():
            # Replace in image references: ![...](old_path) -> ![...](new_path)
            updated_md = updated_md.replace(f"]({old_path})", f"]({new_path})")
            # Also replace in background image directives
            updated_md = updated_md.replace(f"url('{old_path}')", f"url('{new_path}')")

        # Write presentation.md
        pres_file = output_dir / "presentation.md"
        pres_file.write_text(updated_md, encoding="utf-8")
        md_bytes = len(updated_md.encode("utf-8"))
        total_bytes += md_bytes

        # Write metadata.json
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "generated_by": "koas-presenter",
            "presentation_file": "presentation.md",
            "assets_copied": assets_copied,
            "total_bytes": total_bytes,
            "asset_refs": asset_refs,
            "path_map": path_map,
        }
        # Add deck metadata if available from config
        for key in ("title", "author", "subtitle", "organization", "date", "lang"):
            val = input.config.get(key, "")
            if val:
                metadata[key] = val

        meta_file = output_dir / "metadata.json"
        meta_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            f"[pres_marp_export] Output: {pres_file} "
            f"({md_bytes} bytes, {assets_copied} assets)"
        )

        result: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "presentation_file": str(pres_file),
            "assets_copied": assets_copied,
            "total_bytes": total_bytes,
            "metadata_file": str(meta_file),
        }

        # Optional marp-cli export
        if export_format in ("pdf", "html", "both"):
            theme_css = _resolve_theme_css(theme_name, custom_css_path)
            formats = ["pdf", "html"] if export_format == "both" else [export_format]

            for fmt in formats:
                marp_result = run_marp_cli(
                    presentation_md=pres_file,
                    output_format=fmt,
                    theme_css=theme_css,
                )
                key = f"{fmt}_file"
                result[key] = marp_result.get("output_file", "")
                result[f"marp_{fmt}_exit_code"] = marp_result.get("exit_code", -1)

                if marp_result["success"]:
                    logger.info(f"[pres_marp_export] {fmt.upper()}: {marp_result['output_file']}")
                else:
                    logger.warning(
                        f"[pres_marp_export] marp-cli {fmt} failed: "
                        f"{marp_result.get('stderr', 'unknown error')}"
                    )

        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        assets = data.get("assets_copied", 0)
        total = data.get("total_bytes", 0)
        parts = [
            f"Export: presentation.md + {assets} assets ({total:,} bytes)",
        ]
        if data.get("pdf_file"):
            parts.append(f"PDF: {Path(data['pdf_file']).name}")
        if data.get("html_file"):
            parts.append(f"HTML: {Path(data['html_file']).name}")
        parts.append(f"-> {data.get('output_dir', '?')}")
        return " | ".join(parts)
