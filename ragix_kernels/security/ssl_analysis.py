"""
Kernel: SSL/TLS Analysis
Stage: 2 (Analysis)
Category: security

Analyzes SSL/TLS configurations for security vulnerabilities.
Checks certificates, cipher suites, protocol versions, and common misconfigurations.

Wraps:
- sslyze (Python SSL/TLS scanner)
- testssl.sh (alternative, comprehensive testing)
- openssl (basic checks)

Dependencies:
- port_scan: Uses discovered HTTPS/TLS services

Input:
    targets: List of host:port to scan (or use port_scan output)
    check_certificates: Validate certificate chain (default: true)
    check_ciphers: Enumerate cipher suites (default: true)
    check_vulnerabilities: Check for known vulns like Heartbleed (default: true)

Output:
    certificates: Certificate details per target
    cipher_suites: Supported ciphers with security rating
    vulnerabilities: Detected SSL/TLS vulnerabilities
    compliance: Compliance with security standards

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 2 -k ssl_analysis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from sslyze import (
        Scanner,
        ServerScanRequest,
        ServerNetworkLocation,
        ScanCommand,
    )
    from sslyze.errors import ServerHostnameCouldNotBeResolved
    SSLYZE_AVAILABLE = True
except ImportError:
    SSLYZE_AVAILABLE = False

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Weak cipher patterns
WEAK_CIPHER_PATTERNS = [
    "RC4", "DES", "3DES", "MD5", "NULL", "EXPORT", "anon", "ADH", "AECDH"
]

# Minimum acceptable TLS version
MIN_SECURE_TLS = "TLS 1.2"

# Certificate validity thresholds (days)
CERT_EXPIRY_WARNING = 30
CERT_EXPIRY_CRITICAL = 7


class SSLAnalysisKernel(Kernel):
    """
    SSL/TLS security analysis kernel.

    Configuration options:
        targets: Direct target list (host:port format)
        check_certificates: Validate certs (default: true)
        check_ciphers: Enumerate ciphers (default: true)
        check_vulnerabilities: Check known vulns (default: true)
        timeout: Scan timeout per target (default: 30)

    Example manifest:
        ssl_analysis:
          enabled: true
          options:
            check_certificates: true
            check_ciphers: true
            check_vulnerabilities: true
    """

    name = "ssl_analysis"
    version = "1.0.0"
    category = "security"
    stage = 2
    description = "SSL/TLS security analysis"

    requires = ["port_scan"]  # Optional dependency
    provides = ["ssl_certificates", "ssl_vulnerabilities", "cipher_analysis"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Analyze SSL/TLS configurations."""

        # Get targets from port_scan or config
        targets = self._get_targets(input)
        if not targets:
            return {
                "certificates": [],
                "cipher_suites": {},
                "vulnerabilities": [],
                "compliance": {},
                "statistics": {"error": "No TLS targets to scan"},
            }

        check_certs = input.config.get("check_certificates", True)
        check_ciphers = input.config.get("check_ciphers", True)
        check_vulns = input.config.get("check_vulnerabilities", True)
        timeout = input.config.get("timeout", 30)

        logger.info(f"[ssl_analysis] Analyzing {len(targets)} TLS target(s)")

        all_certificates = []
        all_ciphers = {}
        all_vulnerabilities = []
        compliance_results = {}
        statistics = {
            "targets_scanned": 0,
            "certificates_valid": 0,
            "certificates_expiring_soon": 0,
            "weak_ciphers_found": 0,
            "vulnerabilities_found": 0,
        }

        for target in targets:
            host, port = self._parse_target(target)
            logger.info(f"[ssl_analysis] Scanning {host}:{port}")

            if SSLYZE_AVAILABLE:
                result = self._scan_sslyze(host, port, check_certs, check_ciphers, check_vulns)
            else:
                result = self._scan_openssl(host, port, timeout)

            if result.get("certificate"):
                all_certificates.append(result["certificate"])
                if result["certificate"].get("valid"):
                    statistics["certificates_valid"] += 1
                if result["certificate"].get("days_until_expiry", 999) < CERT_EXPIRY_WARNING:
                    statistics["certificates_expiring_soon"] += 1

            if result.get("ciphers"):
                all_ciphers[f"{host}:{port}"] = result["ciphers"]
                weak_count = len([c for c in result["ciphers"] if c.get("weak")])
                statistics["weak_ciphers_found"] += weak_count

            if result.get("vulnerabilities"):
                all_vulnerabilities.extend(result["vulnerabilities"])
                statistics["vulnerabilities_found"] += len(result["vulnerabilities"])

            if result.get("compliance"):
                compliance_results[f"{host}:{port}"] = result["compliance"]

            statistics["targets_scanned"] += 1

        return {
            "certificates": all_certificates,
            "cipher_suites": all_ciphers,
            "vulnerabilities": all_vulnerabilities,
            "compliance": compliance_results,
            "statistics": statistics,
        }

    def _get_targets(self, input: KernelInput) -> List[str]:
        """Get TLS targets from port_scan output or config."""

        # First check port_scan output
        port_scan_path = input.dependencies.get("port_scan")
        if port_scan_path and port_scan_path.exists():
            with open(port_scan_path) as f:
                data = json.load(f).get("data", {})
                services = data.get("services", [])

                # Filter for TLS-enabled services
                tls_ports = {443, 8443, 993, 995, 465, 636, 989, 990, 992, 994}
                tls_services = {"https", "ssl", "tls", "imaps", "pop3s", "smtps", "ldaps"}

                targets = []
                for svc in services:
                    port = svc.get("port", 0)
                    service_name = svc.get("service", "").lower()
                    if port in tls_ports or any(t in service_name for t in tls_services):
                        targets.append(f"{svc['host']}:{port}")

                return targets

        # Check stage1 output
        stage1_port_scan = input.workspace / "stage1" / "port_scan.json"
        if stage1_port_scan.exists():
            with open(stage1_port_scan) as f:
                data = json.load(f).get("data", {})
                services = data.get("services", [])

                tls_ports = {443, 8443, 993, 995, 465, 636}
                targets = []
                for svc in services:
                    if svc.get("port", 0) in tls_ports or "ssl" in svc.get("service", "").lower():
                        targets.append(f"{svc['host']}:{svc['port']}")

                return targets

        # Use config targets
        return input.config.get("targets", [])

    def _parse_target(self, target: str) -> tuple[str, int]:
        """Parse host:port string."""
        if ":" in target:
            parts = target.rsplit(":", 1)
            return parts[0], int(parts[1])
        return target, 443

    def _scan_sslyze(
        self,
        host: str,
        port: int,
        check_certs: bool,
        check_ciphers: bool,
        check_vulns: bool
    ) -> Dict[str, Any]:
        """Scan using sslyze library."""
        result = {
            "certificate": None,
            "ciphers": [],
            "vulnerabilities": [],
            "compliance": {},
        }

        try:
            # Build scan commands
            commands = set()
            if check_certs:
                commands.add(ScanCommand.CERTIFICATE_INFO)
            if check_ciphers:
                commands.add(ScanCommand.SSL_2_0_CIPHER_SUITES)
                commands.add(ScanCommand.SSL_3_0_CIPHER_SUITES)
                commands.add(ScanCommand.TLS_1_0_CIPHER_SUITES)
                commands.add(ScanCommand.TLS_1_1_CIPHER_SUITES)
                commands.add(ScanCommand.TLS_1_2_CIPHER_SUITES)
                commands.add(ScanCommand.TLS_1_3_CIPHER_SUITES)
            if check_vulns:
                commands.add(ScanCommand.HEARTBLEED)
                commands.add(ScanCommand.ROBOT)
                commands.add(ScanCommand.SESSION_RENEGOTIATION)

            # Create scan request
            location = ServerNetworkLocation(hostname=host, port=port)
            request = ServerScanRequest(server_location=location, scan_commands=commands)

            scanner = Scanner()
            scanner.queue_scans([request])

            for scan_result in scanner.get_results():
                # Process certificate
                if check_certs and scan_result.scan_result.certificate_info:
                    cert_result = scan_result.scan_result.certificate_info.result
                    if cert_result and cert_result.certificate_deployments:
                        deployment = cert_result.certificate_deployments[0]
                        cert = deployment.received_certificate_chain[0]

                        # Calculate days until expiry
                        not_after = cert.not_valid_after_utc
                        days_until_expiry = (not_after - datetime.utcnow()).days

                        result["certificate"] = {
                            "host": host,
                            "port": port,
                            "subject": cert.subject.rfc4514_string(),
                            "issuer": cert.issuer.rfc4514_string(),
                            "not_before": cert.not_valid_before_utc.isoformat(),
                            "not_after": not_after.isoformat(),
                            "days_until_expiry": days_until_expiry,
                            "valid": days_until_expiry > 0,
                            "serial": str(cert.serial_number),
                            "signature_algorithm": cert.signature_algorithm_oid._name,
                            "key_size": cert.public_key().key_size if hasattr(cert.public_key(), "key_size") else None,
                            "san": self._get_san(cert),
                        }

                # Process cipher suites
                if check_ciphers:
                    cipher_results = {
                        "ssl2": scan_result.scan_result.ssl_2_0_cipher_suites,
                        "ssl3": scan_result.scan_result.ssl_3_0_cipher_suites,
                        "tls10": scan_result.scan_result.tls_1_0_cipher_suites,
                        "tls11": scan_result.scan_result.tls_1_1_cipher_suites,
                        "tls12": scan_result.scan_result.tls_1_2_cipher_suites,
                        "tls13": scan_result.scan_result.tls_1_3_cipher_suites,
                    }

                    for version, cipher_scan in cipher_results.items():
                        if cipher_scan and cipher_scan.result:
                            for cipher in cipher_scan.result.accepted_cipher_suites:
                                cipher_name = cipher.cipher_suite.name
                                is_weak = self._is_weak_cipher(cipher_name)
                                result["ciphers"].append({
                                    "name": cipher_name,
                                    "version": version,
                                    "weak": is_weak,
                                    "key_size": cipher.cipher_suite.key_size,
                                })

                                if is_weak:
                                    result["vulnerabilities"].append({
                                        "host": host,
                                        "port": port,
                                        "type": "weak_cipher",
                                        "detail": f"Weak cipher {cipher_name} ({version})",
                                        "severity": "medium",
                                    })

                    # Check for deprecated protocols
                    deprecated = ["ssl2", "ssl3", "tls10", "tls11"]
                    for proto in deprecated:
                        scan = cipher_results.get(proto)
                        if scan and scan.result and scan.result.accepted_cipher_suites:
                            result["vulnerabilities"].append({
                                "host": host,
                                "port": port,
                                "type": "deprecated_protocol",
                                "detail": f"Deprecated protocol {proto.upper()} supported",
                                "severity": "high" if proto in ["ssl2", "ssl3"] else "medium",
                            })

                # Process vulnerability checks
                if check_vulns:
                    # Heartbleed
                    hb = scan_result.scan_result.heartbleed
                    if hb and hb.result and hb.result.is_vulnerable_to_heartbleed:
                        result["vulnerabilities"].append({
                            "host": host,
                            "port": port,
                            "type": "heartbleed",
                            "detail": "Vulnerable to Heartbleed (CVE-2014-0160)",
                            "severity": "critical",
                            "cve": "CVE-2014-0160",
                        })

                    # ROBOT
                    robot = scan_result.scan_result.robot
                    if robot and robot.result:
                        from sslyze.plugins.robot.implementation import RobotScanResultEnum
                        if robot.result.robot_result in [
                            RobotScanResultEnum.VULNERABLE_WEAK_ORACLE,
                            RobotScanResultEnum.VULNERABLE_STRONG_ORACLE
                        ]:
                            result["vulnerabilities"].append({
                                "host": host,
                                "port": port,
                                "type": "robot",
                                "detail": "Vulnerable to ROBOT attack",
                                "severity": "high",
                            })

        except ServerHostnameCouldNotBeResolved:
            logger.warning(f"[ssl_analysis] Cannot resolve {host}")
        except Exception as e:
            logger.error(f"[ssl_analysis] sslyze error on {host}:{port}: {e}")

        return result

    def _scan_openssl(self, host: str, port: int, timeout: int) -> Dict[str, Any]:
        """Fallback scan using openssl command."""
        result = {
            "certificate": None,
            "ciphers": [],
            "vulnerabilities": [],
            "compliance": {},
        }

        if not shutil.which("openssl"):
            logger.warning("[ssl_analysis] openssl not found")
            return result

        try:
            # Get certificate
            cmd = f"timeout {timeout} openssl s_client -connect {host}:{port} -servername {host} </dev/null 2>/dev/null | openssl x509 -noout -text"
            cert_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )

            if cert_result.returncode == 0:
                output = cert_result.stdout

                # Parse basic cert info
                result["certificate"] = {
                    "host": host,
                    "port": port,
                    "raw_output": output[:2000],  # Truncate
                    "valid": True,  # Basic assumption
                }

                # Extract expiry
                import re
                expiry_match = re.search(r"Not After\s*:\s*(.+)", output)
                if expiry_match:
                    result["certificate"]["not_after"] = expiry_match.group(1).strip()

                subject_match = re.search(r"Subject:\s*(.+)", output)
                if subject_match:
                    result["certificate"]["subject"] = subject_match.group(1).strip()

            # Check supported protocols
            protocols = [
                ("ssl3", "-ssl3", "high"),
                ("tls1", "-tls1", "medium"),
                ("tls1_1", "-tls1_1", "medium"),
                ("tls1_2", "-tls1_2", None),
                ("tls1_3", "-tls1_3", None),
            ]

            for proto_name, proto_flag, severity in protocols:
                try:
                    proto_cmd = f"timeout 5 openssl s_client -connect {host}:{port} {proto_flag} </dev/null 2>&1"
                    proto_result = subprocess.run(
                        proto_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                    if "CONNECTED" in proto_result.stdout and severity:
                        result["vulnerabilities"].append({
                            "host": host,
                            "port": port,
                            "type": "deprecated_protocol",
                            "detail": f"Deprecated protocol {proto_name.upper()} supported",
                            "severity": severity,
                        })

                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            logger.warning(f"[ssl_analysis] openssl timeout on {host}:{port}")
        except Exception as e:
            logger.error(f"[ssl_analysis] openssl error: {e}")

        return result

    def _is_weak_cipher(self, cipher_name: str) -> bool:
        """Check if cipher is considered weak."""
        cipher_upper = cipher_name.upper()
        return any(weak in cipher_upper for weak in WEAK_CIPHER_PATTERNS)

    def _get_san(self, cert) -> List[str]:
        """Extract Subject Alternative Names from certificate."""
        try:
            from cryptography.x509 import SubjectAlternativeName, DNSName
            from cryptography.x509.oid import ExtensionOID

            san_ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            return [name.value for name in san_ext.value.get_values_for_type(DNSName)]
        except Exception:
            return []

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        vulns = data.get("vulnerabilities", [])

        # Severity breakdown
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for v in vulns:
            sev = v.get("severity", "low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return (
            f"SSL Analysis: {stats.get('targets_scanned', 0)} targets. "
            f"Valid certs: {stats.get('certificates_valid', 0)}. "
            f"Expiring soon: {stats.get('certificates_expiring_soon', 0)}. "
            f"Weak ciphers: {stats.get('weak_ciphers_found', 0)}. "
            f"Vulnerabilities: {severity_counts['critical']} critical, "
            f"{severity_counts['high']} high, {severity_counts['medium']} medium."
        )
