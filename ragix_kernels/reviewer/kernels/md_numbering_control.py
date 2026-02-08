"""
Kernel: md_numbering_control
Stage: 2 (Analysis)

Validate heading numbering, figure/table/equation numbering sequences,
and cross-reference existence. Generate deterministic renumber proposals.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput

import logging

logger = logging.getLogger(__name__)

# Patterns for numbered items
FIGURE_NUM_RE = re.compile(r"(?i)\bFigure\s+(\d+)\b")
TABLE_NUM_RE = re.compile(r"(?i)\bTable\s+(\d+)\b")
EQUATION_NUM_RE = re.compile(r"(?i)\bEquation\s+(\d+)\b")


def _check_heading_levels(section_index: List[Dict]) -> List[Dict[str, Any]]:
    """Check that heading levels don't skip (e.g. H2 -> H4)."""
    findings = []
    prev_level = 0
    for section in section_index:
        level = section["level"]
        if prev_level > 0 and level > prev_level + 1:
            findings.append({
                "type": "heading_level_skip",
                "line": section["line_start"],
                "heading": section["title"],
                "expected_max_level": prev_level + 1,
                "actual_level": level,
                "severity": "attention",
            })
        prev_level = level
    return findings


def _check_numbered_items(
    lines: List[str], pattern: re.Pattern, item_type: str
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """
    Check numbering sequence for figures/tables/equations.

    Returns: (findings, number_map) where number_map is {number: line}.
    """
    findings = []
    occurrences: Dict[int, List[int]] = {}  # number -> list of lines

    for i, line in enumerate(lines):
        for m in pattern.finditer(line):
            num = int(m.group(1))
            occurrences.setdefault(num, []).append(i + 1)

    if not occurrences:
        return findings, {}

    # Check for duplicates
    for num, line_list in occurrences.items():
        if len(line_list) > 2:  # Allow 2 (definition + reference)
            findings.append({
                "type": f"{item_type}_duplicate",
                "number": num,
                "lines": line_list,
                "severity": "attention",
            })

    # Check for gaps
    if occurrences:
        max_num = max(occurrences.keys())
        for expected in range(1, max_num + 1):
            if expected not in occurrences:
                findings.append({
                    "type": f"{item_type}_gap",
                    "missing_number": expected,
                    "severity": "minor",
                })

    number_map = {num: lines[0] for num, lines in occurrences.items()}
    return findings, number_map


def _check_cross_references(
    lines: List[str],
    figure_map: Dict[int, int],
    table_map: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Check that referenced figures/tables actually exist."""
    findings = []
    ref_patterns = [
        (re.compile(r"(?i)\bsee\s+Figure\s+(\d+)"), "figure", figure_map),
        (re.compile(r"(?i)\bsee\s+Table\s+(\d+)"), "table", table_map),
        (re.compile(r"(?i)\bFigure\s+(\d+)"), "figure", figure_map),
        (re.compile(r"(?i)\bTable\s+(\d+)"), "table", table_map),
    ]

    for i, line in enumerate(lines):
        for pattern, item_type, num_map in ref_patterns:
            for m in pattern.finditer(line):
                num = int(m.group(1))
                if num not in num_map:
                    findings.append({
                        "type": f"{item_type}_ref_broken",
                        "line": i + 1,
                        "reference": m.group(),
                        "missing_number": num,
                        "severity": "attention",
                    })

    # Deduplicate (same reference on same line)
    seen = set()
    unique = []
    for f in findings:
        key = (f["type"], f.get("line", 0), f.get("missing_number", 0))
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


class MdNumberingControlKernel(Kernel):
    """Heading/figure/table numbering validation and fix proposals."""

    name = "md_numbering_control"
    version = "1.0.0"
    category = "reviewer"
    stage = 2
    description = "Heading/figure/table numbering validation"

    requires: List[str] = ["md_structure"]
    provides: List[str] = ["numbering_findings", "renumber_ops"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load structure
        struct_path = input.dependencies.get("md_structure")
        if not struct_path or not struct_path.exists():
            raise RuntimeError("Missing dependency: md_structure")

        struct_data = json.loads(struct_path.read_text())["data"]
        section_index = struct_data["section_index"]

        # Load document
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        all_findings: List[Dict[str, Any]] = []

        # 1. Heading level continuity
        heading_findings = _check_heading_levels(section_index)
        all_findings.extend(heading_findings)

        # 2. Figure numbering
        figure_findings, figure_map = _check_numbered_items(
            lines, FIGURE_NUM_RE, "figure"
        )
        all_findings.extend(figure_findings)

        # 3. Table numbering
        table_findings, table_map = _check_numbered_items(
            lines, TABLE_NUM_RE, "table"
        )
        all_findings.extend(table_findings)

        # 4. Equation numbering
        eq_findings, _ = _check_numbered_items(
            lines, EQUATION_NUM_RE, "equation"
        )
        all_findings.extend(eq_findings)

        # 5. Cross-reference existence
        xref_findings = _check_cross_references(lines, figure_map, table_map)
        all_findings.extend(xref_findings)

        # Summary
        by_type = Counter(f["type"] for f in all_findings)

        # Save
        stage_dir = input.workspace / "stage2"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "numbering_findings.json").write_text(
            json.dumps({
                "findings": all_findings,
                "total_findings": len(all_findings),
                "by_type": dict(by_type),
            }, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[md_numbering_control] {len(all_findings)} findings: "
            f"{dict(by_type)}"
        )

        return {
            "findings": all_findings,
            "total_findings": len(all_findings),
            "by_type": dict(by_type),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        by_type = data.get("by_type", {})
        parts = [f"{v} {k}" for k, v in sorted(by_type.items())]
        return (
            f"Numbering control: {data['total_findings']} findings. "
            f"{', '.join(parts) if parts else 'All correct.'}"
        )
