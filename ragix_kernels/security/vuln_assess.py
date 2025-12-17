"""
Kernel: Vulnerability Assessment
Stage: 2 (Analysis)
Category: security

Automated vulnerability scanning using nuclei and CVE database correlation.
Scans discovered services for known vulnerabilities and misconfigurations.

Wraps:
- nuclei (fast vulnerability scanner)
- CVE database lookups
- Custom detection rules

Dependencies:
- port_scan: Services to scan
- ssl_analysis: TLS vulnerabilities (optional)

Input:
    targets: URLs/hosts to scan (or use port_scan output)
    templates: Nuclei template categories (default: ["cves", "misconfigurations"])
    severity: Minimum severity to report (default: "low")
    rate_limit: Requests per second (default: 100)
    template_path: Custom template directory (optional)

Output:
    vulnerabilities: Detected vulnerabilities with CVE references
    by_severity: Vulnerabilities grouped by severity
    by_host: Vulnerabilities grouped by target
    cve_details: CVE information for found vulnerabilities

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 2 -k vuln_assess

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Default nuclei template categories
DEFAULT_TEMPLATE_CATEGORIES = [
    "cves",
    "misconfiguration",
    "exposed-panels",
    "default-logins",
    "takeovers",
    "vulnerabilities",
]

# Severity mapping for consistent reporting
SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
}


class VulnAssessKernel(Kernel):
    """
    Vulnerability assessment kernel using nuclei.

    Configuration options:
        targets: Direct target list (URLs or host:port)
        templates: Template categories to use
        severity: Minimum severity filter ("info", "low", "medium", "high", "critical")
        rate_limit: Max requests per second (default: 100)
        timeout: Scan timeout in seconds (default: 600)
        template_path: Path to nuclei templates (default: ~/nuclei-templates)
        exclude_templates: Templates to exclude

    Example manifest:
        vuln_assess:
          enabled: true
          options:
            templates: ["cves", "misconfiguration"]
            severity: "medium"
            rate_limit: 50
    """

    name = "vuln_assess"
    version = "1.0.0"
    category = "security"
    stage = 2
    description = "Vulnerability assessment with nuclei"

    requires = ["port_scan"]  # Optional dependency
    provides = ["vulnerabilities", "cve_mappings", "security_findings"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Scan targets for vulnerabilities."""

        # Get targets
        targets = self._get_targets(input)
        if not targets:
            return {
                "vulnerabilities": [],
                "by_severity": {},
                "by_host": {},
                "statistics": {"error": "No targets to scan"},
            }

        # Configuration
        templates = input.config.get("templates", DEFAULT_TEMPLATE_CATEGORIES)
        severity_filter = input.config.get("severity", "low")
        rate_limit = input.config.get("rate_limit", 100)
        timeout = input.config.get("timeout", 600)
        template_path = input.config.get(
            "template_path",
            os.path.expanduser("~/nuclei-templates")
        )
        exclude_templates = input.config.get("exclude_templates", [])

        logger.info(f"[vuln_assess] Scanning {len(targets)} target(s)")

        # Check if nuclei is available
        if not shutil.which("nuclei"):
            logger.error("[vuln_assess] nuclei not found in PATH")
            return {
                "vulnerabilities": [],
                "by_severity": {},
                "by_host": {},
                "statistics": {"error": "nuclei not installed"},
            }

        # Run nuclei scan
        vulnerabilities = self._run_nuclei_scan(
            targets=targets,
            templates=templates,
            severity_filter=severity_filter,
            rate_limit=rate_limit,
            timeout=timeout,
            template_path=template_path,
            exclude_templates=exclude_templates,
            workspace=input.workspace,
        )

        # Group results
        by_severity = self._group_by_severity(vulnerabilities)
        by_host = self._group_by_host(vulnerabilities)

        # Calculate statistics
        statistics = {
            "targets_scanned": len(targets),
            "total_findings": len(vulnerabilities),
            "critical": len(by_severity.get("critical", [])),
            "high": len(by_severity.get("high", [])),
            "medium": len(by_severity.get("medium", [])),
            "low": len(by_severity.get("low", [])),
            "info": len(by_severity.get("info", [])),
            "unique_cves": len(set(
                v.get("cve", "")
                for v in vulnerabilities
                if v.get("cve")
            )),
        }

        return {
            "vulnerabilities": vulnerabilities,
            "by_severity": by_severity,
            "by_host": by_host,
            "targets_scanned": targets,
            "statistics": statistics,
        }

    def _get_targets(self, input: KernelInput) -> List[str]:
        """Get targets from port_scan output or config."""

        targets = []

        # Check port_scan output
        port_scan_path = input.dependencies.get("port_scan")
        if port_scan_path and port_scan_path.exists():
            with open(port_scan_path) as f:
                data = json.load(f).get("data", {})
                services = data.get("services", [])

                for svc in services:
                    host = svc.get("host", "")
                    port = svc.get("port", 0)
                    service_name = svc.get("service", "").lower()

                    # Build appropriate URL
                    if port == 443 or "https" in service_name or "ssl" in service_name:
                        targets.append(f"https://{host}:{port}")
                    elif port == 80 or "http" in service_name:
                        targets.append(f"http://{host}:{port}")
                    else:
                        # Generic host:port
                        targets.append(f"{host}:{port}")

                return list(set(targets))

        # Check stage1 output
        stage1_port_scan = input.workspace / "stage1" / "port_scan.json"
        if stage1_port_scan.exists():
            with open(stage1_port_scan) as f:
                data = json.load(f).get("data", {})
                services = data.get("services", [])

                for svc in services:
                    host = svc.get("host", "")
                    port = svc.get("port", 0)
                    if port in [80, 8080]:
                        targets.append(f"http://{host}:{port}")
                    elif port in [443, 8443]:
                        targets.append(f"https://{host}:{port}")
                    else:
                        targets.append(f"{host}:{port}")

                return list(set(targets))

        # Use config targets
        return input.config.get("targets", [])

    def _run_nuclei_scan(
        self,
        targets: List[str],
        templates: List[str],
        severity_filter: str,
        rate_limit: int,
        timeout: int,
        template_path: str,
        exclude_templates: List[str],
        workspace: Path,
    ) -> List[Dict[str, Any]]:
        """Run nuclei scan and parse results."""

        vulnerabilities = []

        # Create targets file
        targets_file = workspace / "stage2" / "nuclei_targets.txt"
        targets_file.parent.mkdir(parents=True, exist_ok=True)
        targets_file.write_text("\n".join(targets))

        # Output file for JSON results
        output_file = workspace / "stage2" / "nuclei_output.json"

        # Build nuclei command
        cmd = [
            "nuclei",
            "-l", str(targets_file),
            "-json-export", str(output_file),
            "-rate-limit", str(rate_limit),
            "-severity", severity_filter,
            "-silent",
        ]

        # Add template path if exists
        if Path(template_path).exists():
            cmd.extend(["-t", template_path])

        # Add template categories
        for tmpl in templates:
            # Check if it's a category in the template path
            tmpl_dir = Path(template_path) / tmpl
            if tmpl_dir.exists():
                cmd.extend(["-t", str(tmpl_dir)])

        # Add exclusions
        for excl in exclude_templates:
            cmd.extend(["-exclude-templates", excl])

        logger.info(f"[vuln_assess] Running: {' '.join(cmd[:10])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace),
            )

            # Parse JSON output
            if output_file.exists():
                with open(output_file) as f:
                    for line in f:
                        try:
                            finding = json.loads(line.strip())
                            vuln = self._parse_nuclei_finding(finding)
                            if vuln:
                                vulnerabilities.append(vuln)
                        except json.JSONDecodeError:
                            continue

            # Also parse stdout for any findings
            for line in result.stdout.split("\n"):
                if line.strip():
                    try:
                        finding = json.loads(line)
                        vuln = self._parse_nuclei_finding(finding)
                        if vuln:
                            vulnerabilities.append(vuln)
                    except json.JSONDecodeError:
                        # Parse text output format
                        vuln = self._parse_nuclei_text(line)
                        if vuln:
                            vulnerabilities.append(vuln)

            if result.returncode != 0:
                logger.warning(f"[vuln_assess] nuclei returned {result.returncode}")
                if result.stderr:
                    logger.debug(f"[vuln_assess] stderr: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            logger.error(f"[vuln_assess] nuclei timeout after {timeout}s")
        except Exception as e:
            logger.error(f"[vuln_assess] nuclei error: {e}")

        # Deduplicate
        seen = set()
        unique_vulns = []
        for v in vulnerabilities:
            key = f"{v.get('host')}:{v.get('template_id')}:{v.get('matched_at', '')}"
            if key not in seen:
                seen.add(key)
                unique_vulns.append(v)

        return unique_vulns

    def _parse_nuclei_finding(self, finding: Dict) -> Optional[Dict[str, Any]]:
        """Parse a nuclei JSON finding."""
        try:
            info = finding.get("info", {})

            vuln = {
                "template_id": finding.get("template-id", finding.get("templateID", "")),
                "name": info.get("name", ""),
                "severity": info.get("severity", "info").lower(),
                "host": finding.get("host", ""),
                "matched_at": finding.get("matched-at", finding.get("matchedAt", "")),
                "type": finding.get("type", ""),
                "description": info.get("description", ""),
                "reference": info.get("reference", []),
                "tags": info.get("tags", []),
                "cve": None,
                "cvss": None,
                "cwe": None,
            }

            # Extract CVE
            classification = info.get("classification", {})
            if classification:
                cve_id = classification.get("cve-id", [])
                if cve_id:
                    vuln["cve"] = cve_id[0] if isinstance(cve_id, list) else cve_id
                vuln["cvss"] = classification.get("cvss-score")
                cwe_id = classification.get("cwe-id", [])
                if cwe_id:
                    vuln["cwe"] = cwe_id[0] if isinstance(cwe_id, list) else cwe_id

            # Check template ID for CVE pattern
            if not vuln["cve"]:
                cve_match = re.search(r"CVE-\d{4}-\d+", vuln["template_id"], re.I)
                if cve_match:
                    vuln["cve"] = cve_match.group(0).upper()

            # Extract matcher name
            vuln["matcher_name"] = finding.get("matcher-name", finding.get("matcherName", ""))

            # Extracted data
            extracted = finding.get("extracted-results", [])
            if extracted:
                vuln["extracted"] = extracted

            return vuln

        except Exception as e:
            logger.debug(f"[vuln_assess] Failed to parse finding: {e}")
            return None

    def _parse_nuclei_text(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse nuclei text output line."""
        # Format: [template-id] [type] [severity] target
        match = re.match(r"\[([^\]]+)\]\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+(.+)", line)
        if match:
            return {
                "template_id": match.group(1),
                "type": match.group(2),
                "severity": match.group(3).lower(),
                "host": match.group(4),
                "name": match.group(1),
                "description": "",
                "reference": [],
                "tags": [],
                "cve": None,
            }
        return None

    def _group_by_severity(
        self,
        vulnerabilities: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Group vulnerabilities by severity."""
        groups = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": [],
        }

        for v in vulnerabilities:
            sev = v.get("severity", "info").lower()
            if sev in groups:
                groups[sev].append(v)
            else:
                groups["info"].append(v)

        return groups

    def _group_by_host(
        self,
        vulnerabilities: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Group vulnerabilities by target host."""
        groups = {}

        for v in vulnerabilities:
            host = v.get("host", "unknown")
            if host not in groups:
                groups[host] = []
            groups[host].append(v)

        # Sort each host's vulns by severity
        for host in groups:
            groups[host].sort(
                key=lambda x: -SEVERITY_ORDER.get(x.get("severity", "info"), 0)
            )

        return groups

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        vulns = data.get("vulnerabilities", [])

        # Top vulnerabilities
        top_vulns = sorted(
            vulns,
            key=lambda x: -SEVERITY_ORDER.get(x.get("severity", "info"), 0)
        )[:5]
        top_names = [f"{v.get('name', 'Unknown')} ({v.get('severity', 'info')})" for v in top_vulns]

        return (
            f"Vulnerability Assessment: {stats.get('targets_scanned', 0)} targets, "
            f"{stats.get('total_findings', 0)} findings. "
            f"Critical: {stats.get('critical', 0)}, High: {stats.get('high', 0)}, "
            f"Medium: {stats.get('medium', 0)}, Low: {stats.get('low', 0)}. "
            f"Unique CVEs: {stats.get('unique_cves', 0)}. "
            f"Top issues: {'; '.join(top_names[:3]) if top_names else 'None'}."
        )
