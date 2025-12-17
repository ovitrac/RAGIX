"""
Kernel: Web Application Scanner
Stage: 2 (Analysis)
Category: security

Web application vulnerability scanning using nikto and OWASP ZAP.
Identifies common web vulnerabilities, misconfigurations, and security issues.

Wraps:
- nikto (CGI scanner, misconfigurations, outdated software)
- OWASP ZAP (active/passive scanning, OWASP Top 10)

Dependencies:
- port_scan: HTTP/HTTPS services to scan (optional)

Input:
    targets: URLs to scan (or use port_scan output)
    scanners: ["nikto", "zap"] - which scanners to use
    scan_mode: "quick", "standard", "full" (default: standard)
    zap_policy: ZAP scan policy name (default: "Default Policy")
    timeout: Scan timeout per target (default: 600)

Output:
    findings: Web vulnerabilities and misconfigurations
    by_scanner: Findings grouped by scanner
    by_target: Findings grouped by target
    statistics: Scan statistics

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 2 -k web_scan

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Nikto tuning options
NIKTO_TUNING = {
    "quick": "1",      # Interesting File/Seen in logs
    "standard": "123", # File Upload, Interesting, Misconfiguration
    "full": "0",       # All tests
}

# ZAP scan strength
ZAP_STRENGTH = {
    "quick": "LOW",
    "standard": "MEDIUM",
    "full": "HIGH",
}

# Severity mapping for nikto
NIKTO_SEVERITY_MAP = {
    "OSVDB": "medium",
    "CVE": "high",
    "error": "low",
    "warning": "low",
    "info": "info",
}


class WebScanKernel(Kernel):
    """
    Web application vulnerability scanning kernel.

    Configuration options:
        targets: Direct target list (URLs)
        scanners: List of scanners to use ["nikto", "zap"]
        scan_mode: "quick", "standard", "full"
        zap_api_key: ZAP API key (if using daemon mode)
        zap_policy: ZAP scan policy name
        timeout: Per-target timeout in seconds
        nikto_plugins: Additional nikto plugins to enable
        exclude_patterns: URL patterns to exclude

    Example manifest:
        web_scan:
          enabled: true
          options:
            scanners: ["nikto", "zap"]
            scan_mode: "standard"
            timeout: 600
    """

    name = "web_scan"
    version = "1.0.0"
    category = "security"
    stage = 2
    description = "Web application vulnerability scanning"

    requires = ["port_scan"]  # Optional dependency
    provides = ["web_vulnerabilities", "web_misconfigurations"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Scan web applications for vulnerabilities."""

        # Get targets
        targets = self._get_targets(input)
        if not targets:
            return {
                "findings": [],
                "by_scanner": {},
                "by_target": {},
                "statistics": {"error": "No web targets to scan"},
            }

        # Configuration
        scanners = input.config.get("scanners", ["nikto"])
        scan_mode = input.config.get("scan_mode", "standard")
        timeout = input.config.get("timeout", 600)
        exclude_patterns = input.config.get("exclude_patterns", [])

        logger.info(f"[web_scan] Scanning {len(targets)} target(s) with {scanners}")

        all_findings = []
        by_scanner = {"nikto": [], "zap": []}
        by_target = {}
        statistics = {
            "targets_scanned": 0,
            "total_findings": 0,
            "scanners_used": [],
            "scan_time_sec": 0,
        }

        start_time = time.time()

        for target in targets:
            logger.info(f"[web_scan] Scanning {target}")
            target_findings = []

            # Run nikto
            if "nikto" in scanners and shutil.which("nikto"):
                nikto_findings = self._run_nikto(
                    target, scan_mode, timeout, input.workspace
                )
                target_findings.extend(nikto_findings)
                by_scanner["nikto"].extend(nikto_findings)
                if "nikto" not in statistics["scanners_used"]:
                    statistics["scanners_used"].append("nikto")

            # Run ZAP
            if "zap" in scanners and self._has_zap():
                zap_findings = self._run_zap(
                    target, scan_mode, timeout,
                    input.config.get("zap_api_key"),
                    input.config.get("zap_policy", "Default Policy"),
                    input.workspace
                )
                target_findings.extend(zap_findings)
                by_scanner["zap"].extend(zap_findings)
                if "zap" not in statistics["scanners_used"]:
                    statistics["scanners_used"].append("zap")

            all_findings.extend(target_findings)
            by_target[target] = target_findings
            statistics["targets_scanned"] += 1

        statistics["total_findings"] = len(all_findings)
        statistics["scan_time_sec"] = round(time.time() - start_time, 2)

        # Sort findings by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        all_findings.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 4))

        return {
            "findings": all_findings,
            "by_scanner": by_scanner,
            "by_target": by_target,
            "targets_scanned": targets,
            "statistics": statistics,
        }

    def _get_targets(self, input: KernelInput) -> List[str]:
        """Get web targets from port_scan or config."""

        targets = []

        # Check port_scan output
        port_scan_file = input.workspace / "stage1" / "port_scan.json"
        if port_scan_file.exists():
            with open(port_scan_file) as f:
                data = json.load(f).get("data", {})
                services = data.get("services", [])

                for svc in services:
                    host = svc.get("host", "")
                    port = svc.get("port", 0)
                    service_name = svc.get("service", "").lower()

                    # Build URL for web services
                    if port == 443 or "https" in service_name or "ssl" in service_name:
                        targets.append(f"https://{host}:{port}" if port != 443 else f"https://{host}")
                    elif port == 80 or "http" in service_name:
                        targets.append(f"http://{host}:{port}" if port != 80 else f"http://{host}")
                    elif port in [8080, 8000, 8888, 9000, 9090]:
                        targets.append(f"http://{host}:{port}")
                    elif port in [8443, 9443]:
                        targets.append(f"https://{host}:{port}")

        if targets:
            return list(set(targets))

        # Use config targets
        return input.config.get("targets", [])

    def _has_zap(self) -> bool:
        """Check if ZAP is available."""
        return (
            shutil.which("zap") is not None or
            shutil.which("zaproxy") is not None or
            shutil.which("zap.sh") is not None or
            Path("/opt/zaproxy/zap.sh").exists()
        )

    def _get_zap_cmd(self) -> str:
        """Get ZAP command path."""
        for cmd in ["zap", "zaproxy", "zap.sh", "/opt/zaproxy/zap.sh"]:
            if shutil.which(cmd) or Path(cmd).exists():
                return cmd
        return "zap"

    def _run_nikto(
        self,
        target: str,
        scan_mode: str,
        timeout: int,
        workspace: Path
    ) -> List[Dict[str, Any]]:
        """Run nikto scan."""
        findings = []

        if not shutil.which("nikto"):
            logger.warning("[web_scan] nikto not found")
            return findings

        # Output file
        output_dir = workspace / "stage2" / "web_scan"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize target for filename
        safe_target = re.sub(r'[^\w\-.]', '_', target)
        output_file = output_dir / f"nikto_{safe_target}.json"

        # Build command
        tuning = NIKTO_TUNING.get(scan_mode, "123")
        cmd = [
            "nikto",
            "-h", target,
            "-Format", "json",
            "-output", str(output_file),
            "-Tuning", tuning,
            "-timeout", "10",
            "-nointeractive",
        ]

        logger.info(f"[web_scan] Running nikto on {target}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse JSON output
            if output_file.exists():
                try:
                    with open(output_file) as f:
                        nikto_data = json.load(f)

                    # nikto JSON structure varies, handle both formats
                    if isinstance(nikto_data, dict):
                        vulns = nikto_data.get("vulnerabilities", [])
                        if not vulns and "host" in nikto_data:
                            vulns = nikto_data.get("host", {}).get("items", [])
                    elif isinstance(nikto_data, list):
                        vulns = nikto_data
                    else:
                        vulns = []

                    for vuln in vulns:
                        finding = self._parse_nikto_finding(vuln, target)
                        if finding:
                            findings.append(finding)

                except json.JSONDecodeError:
                    # Try parsing text output
                    findings.extend(self._parse_nikto_text(result.stdout, target))

            # Also parse stdout for additional findings
            findings.extend(self._parse_nikto_text(result.stdout, target))

        except subprocess.TimeoutExpired:
            logger.warning(f"[web_scan] nikto timeout on {target}")
        except Exception as e:
            logger.error(f"[web_scan] nikto error: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for f in findings:
            key = f"{f.get('target')}:{f.get('id')}:{f.get('description', '')[:50]}"
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return unique

    def _parse_nikto_finding(self, vuln: Dict, target: str) -> Optional[Dict[str, Any]]:
        """Parse a nikto finding from JSON."""
        try:
            # Determine severity
            osvdb = vuln.get("OSVDB", vuln.get("osvdb", ""))
            msg = vuln.get("msg", vuln.get("message", vuln.get("description", "")))

            severity = "info"
            if "CVE" in str(osvdb) or "CVE" in msg:
                severity = "high"
            elif osvdb and osvdb != "0":
                severity = "medium"
            elif any(kw in msg.lower() for kw in ["vulnerable", "exploit", "injection"]):
                severity = "high"
            elif any(kw in msg.lower() for kw in ["outdated", "version", "disclosure"]):
                severity = "medium"

            return {
                "scanner": "nikto",
                "target": target,
                "id": vuln.get("id", osvdb or "nikto-finding"),
                "osvdb": osvdb,
                "severity": severity,
                "description": msg,
                "uri": vuln.get("uri", vuln.get("url", "")),
                "method": vuln.get("method", "GET"),
            }
        except Exception:
            return None

    def _parse_nikto_text(self, output: str, target: str) -> List[Dict[str, Any]]:
        """Parse nikto text output for findings."""
        findings = []

        for line in output.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Match lines like "+ OSVDB-3092: /admin/: This might be interesting..."
            match = re.match(r"\+ (OSVDB-\d+|CVE-[\d-]+)?:?\s*(.+)", line)
            if match:
                osvdb = match.group(1) or ""
                msg = match.group(2)

                severity = "info"
                if "CVE" in osvdb:
                    severity = "high"
                elif "OSVDB" in osvdb:
                    severity = "medium"

                findings.append({
                    "scanner": "nikto",
                    "target": target,
                    "id": osvdb or f"nikto-{len(findings)}",
                    "osvdb": osvdb,
                    "severity": severity,
                    "description": msg,
                    "uri": "",
                    "method": "GET",
                })

        return findings

    def _run_zap(
        self,
        target: str,
        scan_mode: str,
        timeout: int,
        api_key: Optional[str],
        policy: str,
        workspace: Path
    ) -> List[Dict[str, Any]]:
        """Run OWASP ZAP scan."""
        findings = []

        zap_cmd = self._get_zap_cmd()
        if not shutil.which(zap_cmd) and not Path(zap_cmd).exists():
            logger.warning("[web_scan] ZAP not found")
            return findings

        # Output directory
        output_dir = workspace / "stage2" / "web_scan"
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_target = re.sub(r'[^\w\-.]', '_', target)
        output_file = output_dir / f"zap_{safe_target}.json"

        # Use ZAP in command-line mode (baseline scan)
        # Full active scan would require daemon mode
        cmd = [
            zap_cmd,
            "-cmd",
            "-quickurl", target,
            "-quickout", str(output_file),
            "-quickprogress",
        ]

        # Add API key if provided
        if api_key:
            cmd.extend(["-config", f"api.key={api_key}"])

        logger.info(f"[web_scan] Running ZAP baseline scan on {target}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse JSON output
            if output_file.exists():
                try:
                    with open(output_file) as f:
                        zap_data = json.load(f)

                    # ZAP JSON report structure
                    sites = zap_data.get("site", [])
                    if isinstance(sites, dict):
                        sites = [sites]

                    for site in sites:
                        alerts = site.get("alerts", [])
                        for alert in alerts:
                            finding = self._parse_zap_alert(alert, target)
                            if finding:
                                findings.append(finding)

                except json.JSONDecodeError:
                    logger.warning("[web_scan] Could not parse ZAP JSON output")

        except subprocess.TimeoutExpired:
            logger.warning(f"[web_scan] ZAP timeout on {target}")
        except Exception as e:
            logger.error(f"[web_scan] ZAP error: {e}")

        return findings

    def _parse_zap_alert(self, alert: Dict, target: str) -> Optional[Dict[str, Any]]:
        """Parse a ZAP alert."""
        try:
            # ZAP risk levels: 0=Info, 1=Low, 2=Medium, 3=High
            risk_map = {0: "info", 1: "low", 2: "medium", 3: "high"}
            risk_level = alert.get("riskcode", 0)
            if isinstance(risk_level, str):
                risk_level = int(risk_level) if risk_level.isdigit() else 0

            return {
                "scanner": "zap",
                "target": target,
                "id": alert.get("pluginid", alert.get("alertRef", "")),
                "severity": risk_map.get(risk_level, "info"),
                "confidence": alert.get("confidence", ""),
                "name": alert.get("name", alert.get("alert", "")),
                "description": alert.get("desc", alert.get("description", "")),
                "solution": alert.get("solution", ""),
                "reference": alert.get("reference", ""),
                "uri": alert.get("uri", ""),
                "method": alert.get("method", "GET"),
                "cweid": alert.get("cweid", ""),
                "wascid": alert.get("wascid", ""),
            }
        except Exception:
            return None

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        findings = data.get("findings", [])

        # Severity breakdown
        severity_counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
        for f in findings:
            sev = f.get("severity", "info")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return (
            f"Web Scan: {stats.get('targets_scanned', 0)} targets, "
            f"{stats.get('total_findings', 0)} findings. "
            f"Scanners: {', '.join(stats.get('scanners_used', []))}. "
            f"High: {severity_counts['high']}, Medium: {severity_counts['medium']}, "
            f"Low: {severity_counts['low']}. "
            f"Scan time: {stats.get('scan_time_sec', 0)}s."
        )
