"""
Kernel: Security Report Section Generator
Stage: 3 (Reporting)
Category: security

Generates the security section for audit reports.
Produces structured markdown with executive summary, findings, and recommendations.

Dependencies:
- risk_network: Aggregated risk scores (required)
- All Stage 1 and Stage 2 outputs (for details)

Input:
    output_format: "markdown" or "json" (default: markdown)
    include_executive_summary: Include summary section (default: true)
    include_technical_details: Include full technical details (default: true)
    include_remediation_plan: Include remediation roadmap (default: true)
    language: "en" or "fr" (default: en)
    max_findings: Maximum findings to include in detail (default: 50)

Output:
    report_section: Generated report content
    metadata: Report metadata (date, scope, etc.)

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 3 -k section_security

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Report templates
EXECUTIVE_SUMMARY_TEMPLATE = """## Executive Summary

### Security Assessment Overview

| Metric | Value |
|--------|-------|
| Hosts Analyzed | {hosts_analyzed} |
| Total Findings | {total_findings} |
| Critical Issues | {critical_count} |
| High-Risk Issues | {high_count} |
| Average Risk Score | {avg_score}/10 |

### Risk Distribution

{risk_chart}

### Key Findings

{key_findings}

### Recommended Actions

{recommended_actions}
"""

FINDINGS_SECTION_TEMPLATE = """## Detailed Findings

### Critical and High-Severity Issues

{critical_high_findings}

### Medium-Severity Issues

{medium_findings}

### Informational Findings

{info_findings}
"""

REMEDIATION_TEMPLATE = """## Remediation Roadmap

### Immediate Actions (Within 24-48 hours)

{immediate_actions}

### Short-Term Actions (Within 1 week)

{short_term_actions}

### Medium-Term Actions (Within 1 month)

{medium_term_actions}

### Long-Term Improvements

{long_term_actions}
"""

HOST_DETAIL_TEMPLATE = """### {host}

**Risk Score:** {risk_score}/10 ({risk_level})

**Services:** {services_count} | **Vulnerabilities:** {vuln_count} | **SSL Issues:** {ssl_count}

#### Issues

{issues_list}

"""


class SectionSecurityKernel(Kernel):
    """
    Security report section generator kernel.

    Configuration options:
        output_format: "markdown" or "json"
        include_executive_summary: bool
        include_technical_details: bool
        include_remediation_plan: bool
        language: "en" or "fr"
        max_findings: int

    Example manifest:
        section_security:
          enabled: true
          options:
            output_format: markdown
            include_executive_summary: true
            include_remediation_plan: true
    """

    name = "section_security"
    version = "1.0.0"
    category = "security"
    stage = 3
    description = "Security report section generation"

    requires = ["risk_network"]
    provides = ["security_report_section"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate security report section."""

        # Load risk network data
        risk_data = self._load_risk_data(input)

        if not risk_data:
            return {
                "report_section": "# Security Assessment\n\n*No security data available.*",
                "metadata": {"error": "No risk data found"},
            }

        # Configuration
        output_format = input.config.get("output_format", "markdown")
        include_exec_summary = input.config.get("include_executive_summary", True)
        include_tech_details = input.config.get("include_technical_details", True)
        include_remediation = input.config.get("include_remediation_plan", True)
        language = input.config.get("language", "en")
        max_findings = input.config.get("max_findings", 50)

        logger.info(f"[section_security] Generating {output_format} report")

        # Load additional stage data for enrichment
        stage_data = self._load_all_stage_data(input)

        # Generate report sections
        sections = []
        sections.append(self._generate_header(risk_data, stage_data))

        if include_exec_summary:
            sections.append(self._generate_executive_summary(risk_data, stage_data))

        if include_tech_details:
            sections.append(self._generate_findings_section(risk_data, stage_data, max_findings))
            sections.append(self._generate_host_details(risk_data, max_findings))

        if include_remediation:
            sections.append(self._generate_remediation_section(risk_data))

        sections.append(self._generate_appendix(stage_data))

        # Combine sections
        if output_format == "json":
            report_content = {
                "sections": sections,
                "generated_at": datetime.now().isoformat(),
            }
        else:
            report_content = "\n\n---\n\n".join(sections)

        # Save to file
        output_path = input.workspace / "stage3" / "security_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "markdown":
            output_path.write_text(report_content)
        else:
            output_path = output_path.with_suffix(".json")
            output_path.write_text(json.dumps(report_content, indent=2))

        logger.info(f"[section_security] Report saved to {output_path}")

        return {
            "report_section": report_content if output_format == "markdown" else json.dumps(report_content, indent=2),
            "report_path": str(output_path),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "hosts_analyzed": risk_data.get("statistics", {}).get("hosts_analyzed", 0),
                "total_findings": risk_data.get("statistics", {}).get("total_issues", 0),
                "output_format": output_format,
            },
        }

    def _load_risk_data(self, input: KernelInput) -> Optional[Dict]:
        """Load risk_network output."""
        risk_file = input.workspace / "stage2" / "risk_network.json"
        if risk_file.exists():
            with open(risk_file) as f:
                return json.load(f).get("data", {})
        return None

    def _load_all_stage_data(self, input: KernelInput) -> Dict[str, Any]:
        """Load all available stage data."""
        data = {}

        stage_files = [
            ("net_discover", "stage1"),
            ("port_scan", "stage1"),
            ("dns_enum", "stage1"),
            ("ssl_analysis", "stage2"),
            ("vuln_assess", "stage2"),
        ]

        for kernel_name, stage_dir in stage_files:
            file_path = input.workspace / stage_dir / f"{kernel_name}.json"
            if file_path.exists():
                with open(file_path) as f:
                    data[kernel_name] = json.load(f).get("data", {})

        return data

    def _generate_header(self, risk_data: Dict, stage_data: Dict) -> str:
        """Generate report header."""
        now = datetime.now()
        hosts_count = risk_data.get("statistics", {}).get("hosts_analyzed", 0)

        return f"""# Network Security Assessment Report

**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}
**Scope:** {hosts_count} host(s) analyzed
**Assessment Type:** Automated Security Scan

---

**Disclaimer:** This automated security assessment provides an initial overview of
potential security issues. Manual verification is recommended for all critical findings.
False positives may occur. This report does not guarantee completeness of security coverage.
"""

    def _generate_executive_summary(self, risk_data: Dict, stage_data: Dict) -> str:
        """Generate executive summary section."""
        stats = risk_data.get("statistics", {})
        prioritized = risk_data.get("prioritized_hosts", [])
        remediations = risk_data.get("remediation_priorities", [])

        # Risk distribution chart (ASCII)
        risk_levels = {
            "CRITICAL": stats.get("critical_hosts", 0),
            "HIGH": stats.get("high_risk_hosts", 0),
            "MEDIUM": stats.get("medium_risk_hosts", 0),
            "LOW": stats.get("low_risk_hosts", 0),
        }
        total = sum(risk_levels.values()) or 1

        risk_chart = "```\n"
        for level, count in risk_levels.items():
            bar_len = int((count / total) * 40)
            bar = "#" * bar_len
            risk_chart += f"{level:10} [{bar:<40}] {count}\n"
        risk_chart += "```"

        # Key findings
        key_findings = ""
        critical_issues = [r for r in remediations if r.get("severity") in ["critical", "high"]][:5]
        if critical_issues:
            for i, issue in enumerate(critical_issues, 1):
                key_findings += f"{i}. **{issue.get('issue', 'Unknown')}** ({issue.get('host')})\n"
        else:
            key_findings = "*No critical or high-severity issues identified.*"

        # Recommended actions
        recommended = ""
        for i, rem in enumerate(remediations[:5], 1):
            recommended += f"{i}. {rem.get('recommendation', 'Review issue')[:100]}\n"
        if not recommended:
            recommended = "*No immediate actions required.*"

        # Count findings by type from all stage data
        vuln_count = len(stage_data.get("vuln_assess", {}).get("vulnerabilities", []))
        ssl_count = len(stage_data.get("ssl_analysis", {}).get("vulnerabilities", []))

        return EXECUTIVE_SUMMARY_TEMPLATE.format(
            hosts_analyzed=stats.get("hosts_analyzed", 0),
            total_findings=stats.get("total_issues", 0),
            critical_count=stats.get("critical_hosts", 0),
            high_count=stats.get("high_risk_hosts", 0),
            avg_score=stats.get("average_risk_score", 0),
            risk_chart=risk_chart,
            key_findings=key_findings,
            recommended_actions=recommended,
        )

    def _generate_findings_section(
        self,
        risk_data: Dict,
        stage_data: Dict,
        max_findings: int
    ) -> str:
        """Generate detailed findings section."""

        # Collect all findings from vuln_assess
        vulns = stage_data.get("vuln_assess", {}).get("vulnerabilities", [])

        # Sort by severity
        def severity_key(v):
            scores = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
            return -scores.get(v.get("severity", "info"), 0)

        vulns.sort(key=severity_key)

        # Critical and High
        critical_high = [v for v in vulns if v.get("severity") in ["critical", "high"]]
        critical_high_text = ""
        for v in critical_high[:max_findings // 2]:
            cve_text = f" ({v['cve']})" if v.get("cve") else ""
            critical_high_text += f"- **{v.get('name', 'Unknown')}**{cve_text} - {v.get('host', 'Unknown')}\n"
            critical_high_text += f"  - Severity: {v.get('severity', 'unknown').upper()}\n"
            if v.get("description"):
                critical_high_text += f"  - {v['description'][:200]}\n"

        if not critical_high_text:
            critical_high_text = "*No critical or high-severity vulnerabilities found.*"

        # Medium
        medium = [v for v in vulns if v.get("severity") == "medium"]
        medium_text = ""
        for v in medium[:max_findings // 3]:
            medium_text += f"- {v.get('name', 'Unknown')} - {v.get('host', 'Unknown')}\n"

        if not medium_text:
            medium_text = "*No medium-severity findings.*"

        # SSL findings
        ssl_vulns = stage_data.get("ssl_analysis", {}).get("vulnerabilities", [])
        info_text = ""
        for v in ssl_vulns[:10]:
            info_text += f"- {v.get('type', 'SSL Issue')}: {v.get('detail', '')} ({v.get('host')})\n"

        if not info_text:
            info_text = "*No additional informational findings.*"

        return FINDINGS_SECTION_TEMPLATE.format(
            critical_high_findings=critical_high_text,
            medium_findings=medium_text,
            info_findings=info_text,
        )

    def _generate_host_details(self, risk_data: Dict, max_hosts: int = 20) -> str:
        """Generate per-host detail section."""

        risk_matrix = risk_data.get("risk_matrix", [])

        content = "## Host Risk Details\n\n"

        for host_data in risk_matrix[:max_hosts]:
            issues_list = ""
            for issue in host_data.get("top_issues", [])[:10]:
                severity_badge = f"[{issue.get('severity', 'info').upper()}]"
                issues_list += f"- {severity_badge} {issue.get('name', 'Unknown')}\n"

            if not issues_list:
                issues_list = "*No significant issues identified.*"

            content += HOST_DETAIL_TEMPLATE.format(
                host=host_data.get("host", "Unknown"),
                risk_score=host_data.get("total_score", 0),
                risk_level=host_data.get("risk_level", "UNKNOWN"),
                services_count=host_data.get("services_count", 0),
                vuln_count=host_data.get("vulnerabilities_count", 0),
                ssl_count=host_data.get("ssl_issues_count", 0),
                issues_list=issues_list,
            )

        return content

    def _generate_remediation_section(self, risk_data: Dict) -> str:
        """Generate remediation roadmap."""

        remediations = risk_data.get("remediation_priorities", [])

        # Categorize by urgency
        immediate = []    # Critical severity
        short_term = []   # High severity
        medium_term = []  # Medium severity
        long_term = []    # Low/info severity

        for rem in remediations:
            severity = rem.get("severity", "info")
            entry = f"- **{rem.get('host')}**: {rem.get('recommendation', 'Review issue')}\n"

            if severity == "critical":
                immediate.append(entry)
            elif severity == "high":
                short_term.append(entry)
            elif severity == "medium":
                medium_term.append(entry)
            else:
                long_term.append(entry)

        return REMEDIATION_TEMPLATE.format(
            immediate_actions="".join(immediate[:10]) or "*No immediate actions required.*",
            short_term_actions="".join(short_term[:10]) or "*No short-term actions required.*",
            medium_term_actions="".join(medium_term[:10]) or "*No medium-term actions required.*",
            long_term_actions="".join(long_term[:10]) or "*Continue regular security monitoring.*",
        )

    def _generate_appendix(self, stage_data: Dict) -> str:
        """Generate appendix with scan details."""

        content = "## Appendix: Scan Details\n\n"

        # Port scan summary
        if "port_scan" in stage_data:
            ps = stage_data["port_scan"]
            ps_stats = ps.get("statistics", {})
            content += f"### Port Scan\n\n"
            content += f"- Hosts scanned: {ps_stats.get('hosts_scanned', 0)}\n"
            content += f"- Open ports found: {ps_stats.get('total_open_ports', 0)}\n"
            content += f"- Scan time: {ps_stats.get('scan_time_sec', 0):.1f}s\n\n"

        # DNS enum summary
        if "dns_enum" in stage_data:
            dns = stage_data["dns_enum"]
            dns_stats = dns.get("statistics", {})
            content += f"### DNS Enumeration\n\n"
            content += f"- Domains scanned: {dns_stats.get('domains_scanned', 0)}\n"
            content += f"- Records found: {dns_stats.get('total_records', 0)}\n"
            content += f"- Misconfigurations: {dns_stats.get('misconfigurations_found', 0)}\n\n"

        # SSL analysis summary
        if "ssl_analysis" in stage_data:
            ssl = stage_data["ssl_analysis"]
            ssl_stats = ssl.get("statistics", {})
            content += f"### SSL/TLS Analysis\n\n"
            content += f"- Targets scanned: {ssl_stats.get('targets_scanned', 0)}\n"
            content += f"- Valid certificates: {ssl_stats.get('certificates_valid', 0)}\n"
            content += f"- Expiring soon: {ssl_stats.get('certificates_expiring_soon', 0)}\n"
            content += f"- Weak ciphers: {ssl_stats.get('weak_ciphers_found', 0)}\n\n"

        # Vuln assess summary
        if "vuln_assess" in stage_data:
            vuln = stage_data["vuln_assess"]
            vuln_stats = vuln.get("statistics", {})
            content += f"### Vulnerability Assessment\n\n"
            content += f"- Targets scanned: {vuln_stats.get('targets_scanned', 0)}\n"
            content += f"- Total findings: {vuln_stats.get('total_findings', 0)}\n"
            content += f"- Unique CVEs: {vuln_stats.get('unique_cves', 0)}\n\n"

        content += """---

*Report generated by RAGIX KOAS Security Audit System*
"""

        return content

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        metadata = data.get("metadata", {})

        return (
            f"Security Report: {metadata.get('hosts_analyzed', 0)} hosts, "
            f"{metadata.get('total_findings', 0)} findings. "
            f"Format: {metadata.get('output_format', 'markdown')}. "
            f"Generated: {metadata.get('generated_at', 'unknown')}."
        )
