"""
Kernel: Network Risk Scoring
Stage: 2 (Analysis)
Category: security

Aggregates findings from discovery and analysis stages into a unified
risk score per host/service. Produces prioritized remediation lists.

Dependencies:
- net_discover: Host inventory
- port_scan: Service inventory
- dns_enum: DNS findings (optional)
- ssl_analysis: TLS findings (optional)
- vuln_assess: Vulnerability findings (optional)

Input:
    weights: Risk factor weights (vulnerabilities, ssl, exposure, dns)
    thresholds: Risk level thresholds (critical, high, medium)

Output:
    risk_matrix: Risk scores per host
    prioritized_hosts: Hosts sorted by risk
    remediation_priorities: Prioritized action items
    summary_statistics: Overall security posture metrics

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 2 -k risk_network

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Default risk weights
DEFAULT_WEIGHTS = {
    "vulnerabilities": 0.40,   # CVEs and security findings
    "ssl_issues": 0.20,        # TLS/SSL problems
    "exposure": 0.25,          # Service exposure (open ports, dangerous services)
    "dns_issues": 0.15,        # DNS misconfigurations
}

# Severity to score mapping
SEVERITY_SCORES = {
    "critical": 10.0,
    "high": 7.5,
    "medium": 5.0,
    "low": 2.5,
    "info": 1.0,
}

# Risk level thresholds
DEFAULT_THRESHOLDS = {
    "critical": 8.0,
    "high": 6.0,
    "medium": 4.0,
    "low": 2.0,
}

# Service exposure risk scores
SERVICE_EXPOSURE_SCORES = {
    # High-risk services
    "telnet": 9.0,
    "ftp": 7.0,
    "rsh": 9.0,
    "rlogin": 9.0,
    "rexec": 9.0,
    "tftp": 6.0,
    "finger": 5.0,
    "echo": 3.0,
    "discard": 2.0,
    "daytime": 2.0,
    "chargen": 3.0,
    # Medium-risk services
    "vnc": 6.0,
    "rdp": 5.0,
    "smb": 5.0,
    "netbios": 5.0,
    "snmp": 5.0,
    "nfs": 5.0,
    # Database services (exposure is concerning)
    "mysql": 6.0,
    "postgres": 5.0,
    "mssql": 6.0,
    "oracle": 6.0,
    "mongodb": 6.0,
    "redis": 7.0,
    "memcached": 6.0,
    "elasticsearch": 5.0,
    # Lower risk but still notable
    "ssh": 2.0,
    "http": 1.5,
    "https": 1.0,
    "smtp": 2.0,
    "pop3": 2.0,
    "imap": 2.0,
}

# Port-based exposure scores (when service unknown)
PORT_EXPOSURE_SCORES = {
    21: 7.0,    # FTP
    23: 9.0,    # Telnet
    25: 3.0,    # SMTP
    53: 2.0,    # DNS
    69: 6.0,    # TFTP
    79: 5.0,    # Finger
    111: 5.0,   # RPC
    135: 5.0,   # MSRPC
    139: 5.0,   # NetBIOS
    445: 5.0,   # SMB
    512: 8.0,   # rexec
    513: 8.0,   # rlogin
    514: 8.0,   # rsh
    1433: 6.0,  # MSSQL
    1521: 6.0,  # Oracle
    2049: 5.0,  # NFS
    3306: 6.0,  # MySQL
    3389: 5.0,  # RDP
    5432: 5.0,  # PostgreSQL
    5900: 6.0,  # VNC
    6379: 7.0,  # Redis
    27017: 6.0, # MongoDB
}


class RiskNetworkKernel(Kernel):
    """
    Network security risk aggregation kernel.

    Configuration options:
        weights: Risk factor weights (dict)
        thresholds: Risk level thresholds (dict)
        include_remediation: Generate remediation priorities (default: true)

    Example manifest:
        risk_network:
          enabled: true
          options:
            weights:
              vulnerabilities: 0.40
              ssl_issues: 0.20
              exposure: 0.25
              dns_issues: 0.15
    """

    name = "risk_network"
    version = "1.0.0"
    category = "security"
    stage = 2
    description = "Network risk scoring and aggregation"

    requires = ["net_discover", "port_scan"]
    provides = ["network_risk_matrix", "remediation_priorities"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Aggregate security findings into risk scores."""

        # Load all available data
        data = self._load_stage_data(input)

        if not data.get("hosts"):
            return {
                "risk_matrix": [],
                "prioritized_hosts": [],
                "remediation_priorities": [],
                "statistics": {"error": "No host data available"},
            }

        # Get configuration
        weights = input.config.get("weights", DEFAULT_WEIGHTS)
        thresholds = input.config.get("thresholds", DEFAULT_THRESHOLDS)

        logger.info(f"[risk_network] Calculating risk for {len(data['hosts'])} host(s)")

        # Calculate per-host risk
        risk_matrix = []
        for host_ip in data["hosts"]:
            host_risk = self._calculate_host_risk(host_ip, data, weights)
            host_risk["risk_level"] = self._classify_risk(host_risk["total_score"], thresholds)
            risk_matrix.append(host_risk)

        # Sort by risk score (highest first)
        risk_matrix.sort(key=lambda x: -x["total_score"])

        # Generate prioritized host list
        prioritized_hosts = [
            {
                "host": r["host"],
                "risk_score": round(r["total_score"], 2),
                "risk_level": r["risk_level"],
                "top_issues": r["top_issues"][:3],
            }
            for r in risk_matrix
        ]

        # Generate remediation priorities
        remediation_priorities = self._generate_remediation(risk_matrix, thresholds)

        # Statistics
        statistics = {
            "hosts_analyzed": len(risk_matrix),
            "critical_hosts": len([r for r in risk_matrix if r["risk_level"] == "CRITICAL"]),
            "high_risk_hosts": len([r for r in risk_matrix if r["risk_level"] == "HIGH"]),
            "medium_risk_hosts": len([r for r in risk_matrix if r["risk_level"] == "MEDIUM"]),
            "low_risk_hosts": len([r for r in risk_matrix if r["risk_level"] == "LOW"]),
            "average_risk_score": round(
                sum(r["total_score"] for r in risk_matrix) / len(risk_matrix)
                if risk_matrix else 0, 2
            ),
            "max_risk_score": round(max(r["total_score"] for r in risk_matrix) if risk_matrix else 0, 2),
            "total_issues": sum(len(r.get("all_issues", [])) for r in risk_matrix),
        }

        return {
            "risk_matrix": risk_matrix,
            "prioritized_hosts": prioritized_hosts,
            "remediation_priorities": remediation_priorities,
            "weights_used": weights,
            "thresholds_used": thresholds,
            "statistics": statistics,
        }

    def _load_stage_data(self, input: KernelInput) -> Dict[str, Any]:
        """Load data from Stage 1 and Stage 2 outputs."""
        data = {
            "hosts": set(),
            "services": defaultdict(list),
            "vulnerabilities": defaultdict(list),
            "ssl_issues": defaultdict(list),
            "dns_issues": [],
        }

        stage1_dir = input.workspace / "stage1"
        stage2_dir = input.workspace / "stage2"

        # Load net_discover
        net_discover_file = stage1_dir / "net_discover.json"
        if net_discover_file.exists():
            with open(net_discover_file) as f:
                nd_data = json.load(f).get("data", {})
                for host in nd_data.get("hosts", []):
                    if host.get("status") == "up":
                        data["hosts"].add(host["ip"])

        # Load port_scan
        port_scan_file = stage1_dir / "port_scan.json"
        if port_scan_file.exists():
            with open(port_scan_file) as f:
                ps_data = json.load(f).get("data", {})
                for svc in ps_data.get("services", []):
                    host = svc.get("host", "")
                    if host:
                        data["hosts"].add(host)
                        data["services"][host].append(svc)

        # Load dns_enum
        dns_enum_file = stage1_dir / "dns_enum.json"
        if dns_enum_file.exists():
            with open(dns_enum_file) as f:
                dns_data = json.load(f).get("data", {})
                data["dns_issues"] = dns_data.get("misconfigurations", [])

        # Load ssl_analysis
        ssl_analysis_file = stage2_dir / "ssl_analysis.json"
        if ssl_analysis_file.exists():
            with open(ssl_analysis_file) as f:
                ssl_data = json.load(f).get("data", {})
                for vuln in ssl_data.get("vulnerabilities", []):
                    host = vuln.get("host", "")
                    if host:
                        data["ssl_issues"][host].append(vuln)

        # Load vuln_assess
        vuln_assess_file = stage2_dir / "vuln_assess.json"
        if vuln_assess_file.exists():
            with open(vuln_assess_file) as f:
                vuln_data = json.load(f).get("data", {})
                for vuln in vuln_data.get("vulnerabilities", []):
                    # Extract host from URL or host field
                    host = vuln.get("host", "")
                    if "://" in host:
                        # Parse URL
                        import re
                        match = re.search(r"://([^:/]+)", host)
                        if match:
                            host = match.group(1)
                    if host:
                        data["vulnerabilities"][host].append(vuln)

        return data

    def _calculate_host_risk(
        self,
        host_ip: str,
        data: Dict,
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate risk score for a single host."""

        scores = {
            "vulnerabilities": 0.0,
            "ssl_issues": 0.0,
            "exposure": 0.0,
            "dns_issues": 0.0,
        }
        all_issues = []

        # 1. Vulnerability score
        host_vulns = data["vulnerabilities"].get(host_ip, [])
        for vuln in host_vulns:
            severity = vuln.get("severity", "info").lower()
            score = SEVERITY_SCORES.get(severity, 1.0)
            scores["vulnerabilities"] += score

            all_issues.append({
                "type": "vulnerability",
                "severity": severity,
                "name": vuln.get("name", vuln.get("template_id", "Unknown")),
                "cve": vuln.get("cve"),
                "score": score,
            })

        # Normalize (cap at 10)
        if host_vulns:
            scores["vulnerabilities"] = min(10.0, scores["vulnerabilities"])

        # 2. SSL/TLS issues score
        host_ssl = data["ssl_issues"].get(host_ip, [])
        for ssl_issue in host_ssl:
            severity = ssl_issue.get("severity", "medium").lower()
            score = SEVERITY_SCORES.get(severity, 2.5)
            scores["ssl_issues"] += score

            all_issues.append({
                "type": "ssl",
                "severity": severity,
                "name": ssl_issue.get("type", "SSL Issue"),
                "detail": ssl_issue.get("detail", ""),
                "score": score,
            })

        if host_ssl:
            scores["ssl_issues"] = min(10.0, scores["ssl_issues"])

        # 3. Exposure score (based on services)
        host_services = data["services"].get(host_ip, [])
        for svc in host_services:
            service_name = svc.get("service", "").lower()
            port = svc.get("port", 0)

            # Get exposure score
            if service_name in SERVICE_EXPOSURE_SCORES:
                exp_score = SERVICE_EXPOSURE_SCORES[service_name]
            elif port in PORT_EXPOSURE_SCORES:
                exp_score = PORT_EXPOSURE_SCORES[port]
            else:
                exp_score = 1.0  # Default exposure

            scores["exposure"] += exp_score

            if exp_score >= 5.0:
                all_issues.append({
                    "type": "exposure",
                    "severity": "high" if exp_score >= 7.0 else "medium",
                    "name": f"Exposed service: {service_name or 'unknown'} on port {port}",
                    "score": exp_score,
                })

        if host_services:
            scores["exposure"] = min(10.0, scores["exposure"])

        # 4. DNS issues (global, proportionally attributed)
        dns_issues = data.get("dns_issues", [])
        if dns_issues and data["hosts"]:
            dns_score = sum(
                SEVERITY_SCORES.get(d.get("severity", "low"), 2.5)
                for d in dns_issues
            ) / len(data["hosts"])
            scores["dns_issues"] = min(10.0, dns_score)

        # Calculate weighted total
        total_score = sum(
            scores[factor] * weights.get(factor, 0.25)
            for factor in scores
        )

        # Sort issues by score
        all_issues.sort(key=lambda x: -x.get("score", 0))

        return {
            "host": host_ip,
            "total_score": round(total_score, 2),
            "component_scores": {k: round(v, 2) for k, v in scores.items()},
            "services_count": len(host_services),
            "vulnerabilities_count": len(host_vulns),
            "ssl_issues_count": len(host_ssl),
            "top_issues": all_issues[:5],
            "all_issues": all_issues,
        }

    def _classify_risk(self, score: float, thresholds: Dict[str, float]) -> str:
        """Classify risk score into level."""
        if score >= thresholds.get("critical", 8.0):
            return "CRITICAL"
        elif score >= thresholds.get("high", 6.0):
            return "HIGH"
        elif score >= thresholds.get("medium", 4.0):
            return "MEDIUM"
        elif score >= thresholds.get("low", 2.0):
            return "LOW"
        else:
            return "INFO"

    def _generate_remediation(
        self,
        risk_matrix: List[Dict],
        thresholds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized remediation list."""

        remediations = []
        priority = 1

        # Collect all issues with host context
        all_issues_with_host = []
        for host_data in risk_matrix:
            for issue in host_data.get("all_issues", []):
                all_issues_with_host.append({
                    **issue,
                    "host": host_data["host"],
                    "host_risk": host_data["total_score"],
                })

        # Sort by severity and host risk
        all_issues_with_host.sort(
            key=lambda x: (
                -SEVERITY_SCORES.get(x.get("severity", "info"), 0),
                -x.get("host_risk", 0),
            )
        )

        # Generate remediations
        seen = set()
        for issue in all_issues_with_host:
            # Deduplicate by issue name
            issue_key = f"{issue.get('name')}:{issue.get('type')}"
            if issue_key in seen:
                continue
            seen.add(issue_key)

            remediation = {
                "priority": priority,
                "host": issue["host"],
                "issue_type": issue.get("type", "unknown"),
                "severity": issue.get("severity", "info"),
                "issue": issue.get("name", "Unknown issue"),
                "recommendation": self._get_recommendation(issue),
            }

            if issue.get("cve"):
                remediation["cve"] = issue["cve"]

            remediations.append(remediation)
            priority += 1

            if priority > 50:  # Limit to top 50 remediations
                break

        return remediations

    def _get_recommendation(self, issue: Dict) -> str:
        """Generate recommendation based on issue type."""

        issue_type = issue.get("type", "")
        name = issue.get("name", "").lower()
        severity = issue.get("severity", "")

        if issue_type == "vulnerability":
            if issue.get("cve"):
                return f"Apply security patch for {issue['cve']}. Check vendor advisories for updates."
            return "Review and patch the vulnerable component. Check for security updates."

        elif issue_type == "ssl":
            if "deprecated" in name or "protocol" in name:
                return "Disable deprecated TLS protocols (SSL 2.0, SSL 3.0, TLS 1.0, TLS 1.1). Enable TLS 1.2+ only."
            elif "cipher" in name:
                return "Update cipher suite configuration to use strong ciphers only. Disable weak algorithms (RC4, 3DES, etc.)."
            elif "heartbleed" in name:
                return "Urgent: Patch OpenSSL to fix Heartbleed (CVE-2014-0160). Replace compromised certificates."
            elif "certificate" in name or "expir" in name:
                return "Renew SSL certificate before expiration. Verify certificate chain validity."
            return "Review and harden TLS configuration per industry best practices."

        elif issue_type == "exposure":
            if any(s in name for s in ["telnet", "ftp", "rsh", "rlogin"]):
                return "Critical: Disable insecure service and replace with secure alternatives (SSH, SFTP)."
            elif any(s in name for s in ["mysql", "postgres", "mongo", "redis"]):
                return "Restrict database access to localhost or trusted networks. Enable authentication."
            elif any(s in name for s in ["vnc", "rdp"]):
                return "Restrict remote access to VPN or trusted IPs. Enable strong authentication and encryption."
            return "Review service necessity and restrict access with firewall rules."

        elif issue_type == "dns":
            if "zone transfer" in name:
                return "Critical: Disable zone transfers for non-authorized hosts."
            elif "spf" in name or "dmarc" in name:
                return "Configure SPF and DMARC records to prevent email spoofing."
            elif "dnssec" in name:
                return "Consider enabling DNSSEC for domain validation."
            return "Review and harden DNS configuration."

        return f"Review and remediate: {issue.get('name', 'Unknown issue')}"

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        prioritized = data.get("prioritized_hosts", [])

        # Top at-risk hosts
        top_hosts = [
            f"{h['host']}({h['risk_score']:.1f})"
            for h in prioritized[:3]
        ]

        return (
            f"Network Risk Assessment: {stats.get('hosts_analyzed', 0)} hosts. "
            f"Critical: {stats.get('critical_hosts', 0)}, "
            f"High: {stats.get('high_risk_hosts', 0)}, "
            f"Medium: {stats.get('medium_risk_hosts', 0)}, "
            f"Low: {stats.get('low_risk_hosts', 0)}. "
            f"Avg score: {stats.get('average_risk_score', 0):.1f}/10. "
            f"Total issues: {stats.get('total_issues', 0)}. "
            f"Top at-risk: {', '.join(top_hosts) if top_hosts else 'None'}."
        )
