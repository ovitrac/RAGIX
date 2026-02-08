"""
Kernel: md_structure
Stage: 1 (Collection)

Heading tree, section anchors, numbering pattern detection.
Parses Markdown via md_parser to extract heading tree with stable IDs.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.md_parser import extract_headings, build_heading_tree

import logging

logger = logging.getLogger(__name__)


class MdStructureKernel(Kernel):
    """Extract heading tree, section anchors, and numbering patterns."""

    name = "md_structure"
    version = "1.0.0"
    category = "reviewer"
    stage = 1
    description = "Heading tree, anchors, numbering patterns"

    requires: List[str] = ["md_inventory"]
    provides: List[str] = ["heading_tree", "section_index", "anchor_map"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load document from inventory snapshot
        inv_path = input.dependencies.get("md_inventory")
        if not inv_path or not inv_path.exists():
            raise RuntimeError("Missing dependency: md_inventory")

        inv_data = json.loads(inv_path.read_text())["data"]
        snapshot_path = Path(inv_data["snapshot_path"])
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        total_lines = len(lines)

        # Extract headings
        raw_headings = extract_headings(lines)

        # Build tree
        tree = build_heading_tree(raw_headings, total_lines)

        # Build flat index and anchor map
        section_index = []
        anchor_map = {}

        def _flatten(nodes: list, depth: int = 0):
            for node in nodes:
                entry = {
                    "id": node.id,
                    "level": node.level,
                    "title": node.title,
                    "anchor": node.anchor,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "numbering": node.numbering,
                    "depth": depth,
                    "child_count": len(node.children),
                }
                section_index.append(entry)
                anchor_map[node.anchor] = node.id
                _flatten(node.children, depth + 1)

        _flatten(tree)

        # Detect numbering pattern
        has_explicit_numbering = any(h["numbering"] for h in section_index)
        max_depth = max((h["depth"] for h in section_index), default=0)

        # Save outputs
        stage_dir = input.workspace / "stage1"
        stage_dir.mkdir(parents=True, exist_ok=True)

        outline = {
            "heading_tree": [n.to_dict() for n in tree],
            "total_headings": len(section_index),
            "max_depth": max_depth,
            "has_explicit_numbering": has_explicit_numbering,
        }
        (stage_dir / "outline.json").write_text(
            json.dumps(outline, indent=2), encoding="utf-8"
        )
        (stage_dir / "anchors.json").write_text(
            json.dumps(anchor_map, indent=2), encoding="utf-8"
        )

        logger.info(
            f"[md_structure] {len(section_index)} headings, "
            f"max depth {max_depth}, "
            f"explicit numbering: {has_explicit_numbering}"
        )

        return {
            "heading_tree": [n.to_dict() for n in tree],
            "section_index": section_index,
            "anchor_map": anchor_map,
            "total_headings": len(section_index),
            "max_depth": max_depth,
            "has_explicit_numbering": has_explicit_numbering,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        return (
            f"Structure: {data['total_headings']} headings, "
            f"max depth {data['max_depth']}, "
            f"explicit numbering: {data['has_explicit_numbering']}"
        )
