"""
Kernel: md_protected_regions
Stage: 1 (Collection)

Detect and mark immutable spans: code fences, inline code, YAML front matter,
HTML blocks, tables, math blocks, link reference definitions.

Policy: conservative — when in doubt, mark as protected.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.md_parser import detect_protected_regions

import logging

logger = logging.getLogger(__name__)


class MdProtectedRegionsKernel(Kernel):
    """Locate protected (immutable) spans in Markdown."""

    name = "md_protected_regions"
    version = "1.0.0"
    category = "reviewer"
    stage = 1
    description = "Code fences, inline code, YAML, tables, math"

    requires: List[str] = ["md_inventory"]
    provides: List[str] = ["protected_spans"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load snapshot
        inv_path = input.dependencies.get("md_inventory")
        if not inv_path or not inv_path.exists():
            raise RuntimeError("Missing dependency: md_inventory")

        inv_data = json.loads(inv_path.read_text())["data"]
        snapshot_path = Path(inv_data["snapshot_path"])
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        style_cfg = reviewer_cfg.get("style", {})
        protect_tables = style_cfg.get("protect_tables", True)
        protect_math = style_cfg.get("protect_math", True)

        # Detect protected regions
        spans = detect_protected_regions(lines, protect_tables, protect_math)

        # Compute coverage
        protected_lines = set()
        for span in spans:
            for line_num in range(span.line_start, span.line_end + 1):
                protected_lines.add(line_num)

        total_lines = len(lines)
        coverage_pct = round(100 * len(protected_lines) / max(1, total_lines), 1)

        # Count by kind
        by_kind: Dict[str, int] = {}
        for span in spans:
            k = span.kind.value
            by_kind[k] = by_kind.get(k, 0) + 1

        # Save output
        stage_dir = input.workspace / "stage1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "protected_spans.json").write_text(
            json.dumps({
                "spans": [s.to_dict() for s in spans],
                "total_spans": len(spans),
                "protected_lines": len(protected_lines),
                "coverage_pct": coverage_pct,
                "by_kind": by_kind,
            }, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[md_protected_regions] {len(spans)} spans, "
            f"{len(protected_lines)}/{total_lines} lines protected ({coverage_pct}%)"
        )

        return {
            "protected_spans": [s.to_dict() for s in spans],
            "total_spans": len(spans),
            "protected_lines": len(protected_lines),
            "total_lines": total_lines,
            "coverage_pct": coverage_pct,
            "by_kind": by_kind,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        by_kind = data.get("by_kind", {})
        parts = [f"{v} {k}" for k, v in sorted(by_kind.items())]
        return (
            f"Protected regions: {data['total_spans']} spans "
            f"({data['coverage_pct']}% of lines). "
            f"Types: {', '.join(parts) if parts else 'none'}"
        )
