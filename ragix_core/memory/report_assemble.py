"""
Report Assembly — Structured Summary Output

Assembles per-domain sections into a complete summary report with:
- Executive summary (top findings across domains)
- Per-domain structured sections
- Coverage table
- Citation appendix

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_core.memory.citation_verify import CitationReport
from ragix_core.memory.types import MemoryItem

logger = logging.getLogger(__name__)


@dataclass
class DomainSection:
    """A structured section for one domain."""
    domain: str
    title: str
    content: str  # markdown content with [MID: ...] citations
    item_count: int = 0
    rule_ids: List[str] = field(default_factory=list)


@dataclass
class SummaryReport:
    """Complete assembled summary report."""
    title: str = ""
    scope: str = ""
    created_at: str = ""
    model: str = ""
    sections: List[DomainSection] = field(default_factory=list)
    executive_summary: str = ""
    coverage_table: str = ""
    citation_report: Optional[CitationReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def assemble_report(
    sections: List[DomainSection],
    scope: str,
    model: str = "",
    budgeted_meta: Optional[Dict[str, Any]] = None,
    citation_report: Optional[CitationReport] = None,
    title: Optional[str] = None,
) -> SummaryReport:
    """
    Assemble sections into a complete summary report.

    Args:
        sections: Per-domain content sections
        scope: Memory scope label
        model: LLM model used for generation
        budgeted_meta: Metadata from budgeted recall (domain stats)
        citation_report: Citation verification results
        title: Report title (auto-generated if None)

    Returns:
        SummaryReport with all components
    """
    report = SummaryReport(
        title=title or f"KOAS Summary: {scope}",
        scope=scope,
        created_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        sections=sections,
    )

    # Build executive summary
    report.executive_summary = _build_executive_summary(sections)

    # Build coverage table
    report.coverage_table = _build_coverage_table(
        sections, budgeted_meta or {}
    )

    # Attach citation report
    report.citation_report = citation_report

    # Metadata
    report.metadata = {
        "scope": scope,
        "model": model,
        "sections_count": len(sections),
        "total_domains": len(set(s.domain for s in sections)),
        "created_at": report.created_at,
    }

    return report


def render_markdown(report: SummaryReport) -> str:
    """
    Render a SummaryReport as markdown.

    Returns:
        Complete markdown string
    """
    parts = []

    # Title
    parts.append(f"# {report.title}\n")
    parts.append(f"**Scope**: {report.scope}  ")
    parts.append(f"**Date**: {report.created_at[:10]}  ")
    if report.model:
        parts.append(f"**Model**: {report.model}  ")
    parts.append("")

    # Executive summary
    parts.append("## Executive Summary\n")
    parts.append(report.executive_summary)
    parts.append("")

    # Coverage table
    parts.append("## Domain Coverage\n")
    parts.append(report.coverage_table)
    parts.append("")

    # Per-domain sections
    for section in report.sections:
        parts.append(f"## {section.title}\n")
        parts.append(section.content)
        parts.append("")

    # Citation statistics
    if report.citation_report:
        cr = report.citation_report
        parts.append("## Citation Statistics\n")
        parts.append(f"| Metric | Value |")
        parts.append(f"|--------|------:|")
        parts.append(f"| Bullets with citation | {cr.bullets_with_citation}/{cr.total_bullets} |")
        rate = (
            f"{cr.bullets_with_citation / cr.total_bullets:.1%}"
            if cr.total_bullets > 0 else "N/A"
        )
        parts.append(f"| Citation rate | {rate} |")
        parts.append(f"| Valid MIDs | {cr.valid_mids} |")
        parts.append(f"| Invalid MIDs | {cr.invalid_mids} |")
        parts.append(f"| Unique MIDs referenced | {cr.unique_mids} |")
        parts.append("")

        if cr.domain_coverage:
            parts.append("### MIDs per Domain\n")
            parts.append("| Domain | MIDs |")
            parts.append("|--------|-----:|")
            for domain, count in sorted(
                cr.domain_coverage.items(), key=lambda kv: kv[1], reverse=True
            ):
                parts.append(f"| {domain} | {count} |")
            parts.append("")

    return "\n".join(parts)


def render_json(report: SummaryReport) -> Dict[str, Any]:
    """Render a SummaryReport as a JSON-serializable dict."""
    return {
        "title": report.title,
        "scope": report.scope,
        "created_at": report.created_at,
        "model": report.model,
        "executive_summary": report.executive_summary,
        "sections": [
            {
                "domain": s.domain,
                "title": s.title,
                "content": s.content,
                "item_count": s.item_count,
                "rule_ids": s.rule_ids,
            }
            for s in report.sections
        ],
        "coverage": report.coverage_table,
        "citation": (
            report.citation_report.to_dict()
            if report.citation_report else None
        ),
        "metadata": report.metadata,
    }


def save_report(
    report: SummaryReport,
    output_dir: Path,
    citation_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Save report artifacts to a directory.

    Returns:
        Dict of artifact name -> file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    # Markdown report
    md_path = output_dir / "summary.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    artifacts["summary_md"] = md_path

    # JSON report
    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps(render_json(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts["summary_json"] = json_path

    # Citation map
    if citation_map:
        cm_path = output_dir / "citation_map.json"
        cm_path.write_text(
            json.dumps(citation_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["citation_map"] = cm_path

    logger.info(f"Report saved: {list(artifacts.keys())} -> {output_dir}")
    return artifacts


def _build_executive_summary(sections: List[DomainSection]) -> str:
    """Extract top findings from each section for the executive summary."""
    if not sections:
        return "*No sections generated.*"

    parts = []
    parts.append(f"This report covers **{len(sections)} technology domains**:\n")
    for s in sections:
        # Extract first bullet from each section as the key finding
        lines = s.content.strip().split("\n")
        first_bullet = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                first_bullet = stripped[2:].strip()
                break
        if first_bullet:
            parts.append(f"- **{s.domain}**: {first_bullet[:120]}")
        else:
            parts.append(f"- **{s.domain}**: {s.item_count} items analyzed")
    return "\n".join(parts)


def _build_coverage_table(
    sections: List[DomainSection],
    budgeted_meta: Dict[str, Any],
) -> str:
    """Build a coverage summary table."""
    per_domain = budgeted_meta.get("per_domain", {})

    lines = [
        "| Domain | Items Selected | Items Available | Rule IDs |",
        "|--------|:--------------:|:---------------:|:--------:|",
    ]
    for s in sorted(sections, key=lambda x: x.domain):
        dm = per_domain.get(s.domain, {})
        selected = dm.get("selected", s.item_count)
        available = dm.get("available", "?")
        rule_ids = ", ".join(s.rule_ids) if s.rule_ids else "-"
        lines.append(f"| {s.domain} | {selected} | {available} | {rule_ids} |")

    total_selected = budgeted_meta.get("total_items", sum(s.item_count for s in sections))
    total_tokens = budgeted_meta.get("total_tokens", "?")
    lines.append(f"\n**Total**: {total_selected} items, ~{total_tokens} tokens injected")

    return "\n".join(lines)
