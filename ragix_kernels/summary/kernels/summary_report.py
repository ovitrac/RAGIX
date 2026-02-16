"""
summary_report — Stage 3: Report Assembly

Assembles per-domain sections, executive summary, coverage table,
and citation appendix into a final deliverable.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput


class SummaryReportKernel(Kernel):
    name = "summary_report"
    version = "1.0.0"
    category = "summary"
    stage = 3
    description = "Assemble final summary report"
    requires = ["summary_generate", "summary_verify"]
    provides = ["summary_md", "summary_json", "citation_map"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Assemble per-domain sections, executive summary, and citation appendix."""
        from ragix_core.memory.citation_verify import CitationReport
        from ragix_core.memory.report_assemble import (
            DomainSection,
            assemble_report,
            render_markdown,
            render_json,
            save_report,
        )

        cfg = input.config
        scope = cfg.get("scope", "project")
        model = cfg.get("model", "")

        # Load generated sections
        gen_file = input.dependencies.get("summary_generate")
        if gen_file and gen_file.exists():
            gen_data = json.loads(gen_file.read_text())["data"]
        else:
            raise RuntimeError("Missing summary_generate dependency")

        # Load verification results
        verify_file = input.dependencies.get("summary_verify")
        verify_data = {}
        citation_map = {}
        if verify_file and verify_file.exists():
            verify_data = json.loads(verify_file.read_text())["data"]
            citation_map = verify_data.get("citation_map", {})

        # Build DomainSection objects
        sections = []
        for s in gen_data.get("sections", []):
            sections.append(DomainSection(
                domain=s["domain"],
                title=s["title"],
                content=s["content"],
                item_count=s.get("item_count", 0),
            ))

        # Load budgeted recall metadata
        recall_file = input.dependencies.get("summary_budgeted_recall")
        budgeted_meta = {}
        if recall_file and recall_file.exists():
            recall_data = json.loads(recall_file.read_text())["data"]
            budgeted_meta = recall_data.get("meta", {})

        # Assemble
        report = assemble_report(
            sections=sections,
            scope=scope,
            model=model or gen_data.get("model", ""),
            budgeted_meta=budgeted_meta,
        )

        # Save artifacts
        output_dir = input.workspace / "stage3"
        artifacts = save_report(report, output_dir, citation_map)

        return {
            "artifacts": {k: str(v) for k, v in artifacts.items()},
            "sections_count": len(sections),
            "domains": [s.domain for s in sections],
            "summary_preview": render_markdown(report)[:500],
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of report assembly results."""
        return (
            f"Report assembled: {data.get('sections_count', 0)} sections, "
            f"domains={data.get('domains', [])}"
        )
