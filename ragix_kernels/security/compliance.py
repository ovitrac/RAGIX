"""
Kernel: Compliance Assessment
Stage: 2 (Analysis)
Category: security

Evaluates security findings against compliance frameworks.
Primary focus on ANSSI (French National Cybersecurity Agency) recommendations,
with support for NIST CSF and CIS Controls.

Frameworks:
- ANSSI: Agence Nationale de la Sécurité des Systèmes d'Information
  - Guide d'hygiène informatique (42 measures)
  - Recommandations de sécurité relatives à TLS
  - Recommandations de sécurité relatives aux réseaux
- NIST CSF: Cybersecurity Framework (Identify, Protect, Detect, Respond, Recover)
- CIS Controls: Center for Internet Security Critical Security Controls v8

Dependencies:
- port_scan: Service inventory
- ssl_analysis: TLS configuration
- dns_enum: DNS configuration
- vuln_assess: Vulnerability findings (optional)
- risk_network: Risk scores (optional)

Input:
    frameworks: ["anssi", "nist", "cis"] - frameworks to evaluate
    anssi_level: "essential", "standard", "reinforced" (default: standard)
    include_recommendations: Generate remediation recommendations (default: true)

Output:
    compliance_scores: Score per framework
    findings: Non-compliant items by framework
    recommendations: Remediation recommendations
    summary: Executive compliance summary

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 2 -k compliance

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# =============================================================================
# ANSSI - Guide d'hygiène informatique (42 measures)
# Reference: https://www.ssi.gouv.fr/guide/guide-dhygiene-informatique/
# =============================================================================

ANSSI_HYGIENE_RULES = {
    # Section 1: Sensibiliser et former
    "ANSSI-01": {
        "id": "ANSSI-01",
        "title": "Former les équipes à la cybersécurité",
        "category": "awareness",
        "level": "essential",
        "checks": [],  # Manual check
    },
    # Section 2: Connaître le système d'information
    "ANSSI-02": {
        "id": "ANSSI-02",
        "title": "Établir une cartographie du SI",
        "category": "inventory",
        "level": "essential",
        "checks": ["has_network_inventory"],
    },
    "ANSSI-03": {
        "id": "ANSSI-03",
        "title": "Identifier les données et traitements les plus sensibles",
        "category": "inventory",
        "level": "essential",
        "checks": [],  # Manual check
    },
    # Section 3: Authentifier et contrôler les accès
    "ANSSI-04": {
        "id": "ANSSI-04",
        "title": "Mettre en œuvre une politique de mots de passe robuste",
        "category": "authentication",
        "level": "essential",
        "checks": ["no_default_credentials"],
    },
    "ANSSI-05": {
        "id": "ANSSI-05",
        "title": "Protéger les mots de passe stockés",
        "category": "authentication",
        "level": "standard",
        "checks": [],
    },
    # Section 4: Sécuriser les postes
    "ANSSI-06": {
        "id": "ANSSI-06",
        "title": "Mettre à jour régulièrement les logiciels",
        "category": "patching",
        "level": "essential",
        "checks": ["no_outdated_software", "no_critical_cves"],
    },
    # Section 5: Sécuriser le réseau
    "ANSSI-07": {
        "id": "ANSSI-07",
        "title": "Limiter l'exposition des services sur Internet",
        "category": "network",
        "level": "essential",
        "checks": ["no_dangerous_services", "limited_exposure"],
    },
    "ANSSI-08": {
        "id": "ANSSI-08",
        "title": "Protéger les interconnexions réseau",
        "category": "network",
        "level": "standard",
        "checks": ["firewall_present"],
    },
    "ANSSI-09": {
        "id": "ANSSI-09",
        "title": "Segmenter le réseau",
        "category": "network",
        "level": "standard",
        "checks": [],  # Requires network topology
    },
    # Section 6: Sécuriser les communications
    "ANSSI-10": {
        "id": "ANSSI-10",
        "title": "Utiliser des protocoles sécurisés (TLS)",
        "category": "crypto",
        "level": "essential",
        "checks": ["tls_1_2_minimum", "no_weak_ciphers", "valid_certificates"],
    },
    "ANSSI-11": {
        "id": "ANSSI-11",
        "title": "Chiffrer les données sensibles",
        "category": "crypto",
        "level": "standard",
        "checks": [],
    },
    # Section 7: Sécuriser l'administration
    "ANSSI-12": {
        "id": "ANSSI-12",
        "title": "Sécuriser l'accès d'administration",
        "category": "admin",
        "level": "essential",
        "checks": ["no_telnet", "ssh_secure"],
    },
    # Section 8: Gérer les incidents
    "ANSSI-13": {
        "id": "ANSSI-13",
        "title": "Journaliser les événements",
        "category": "logging",
        "level": "essential",
        "checks": [],  # Manual check
    },
    # DNS Security
    "ANSSI-DNS-01": {
        "id": "ANSSI-DNS-01",
        "title": "Protéger les transferts de zone DNS",
        "category": "dns",
        "level": "essential",
        "checks": ["no_zone_transfer"],
    },
    "ANSSI-DNS-02": {
        "id": "ANSSI-DNS-02",
        "title": "Configurer SPF et DMARC",
        "category": "dns",
        "level": "standard",
        "checks": ["has_spf", "has_dmarc"],
    },
    "ANSSI-DNS-03": {
        "id": "ANSSI-DNS-03",
        "title": "Activer DNSSEC",
        "category": "dns",
        "level": "reinforced",
        "checks": ["has_dnssec"],
    },
}

# ANSSI TLS Recommendations
# Reference: https://www.ssi.gouv.fr/guide/recommandations-de-securite-relatives-a-tls/
ANSSI_TLS_RULES = {
    "ANSSI-TLS-01": {
        "id": "ANSSI-TLS-01",
        "title": "Utiliser TLS 1.2 au minimum",
        "level": "essential",
        "checks": ["tls_1_2_minimum"],
    },
    "ANSSI-TLS-02": {
        "id": "ANSSI-TLS-02",
        "title": "Privilégier TLS 1.3",
        "level": "reinforced",
        "checks": ["tls_1_3_preferred"],
    },
    "ANSSI-TLS-03": {
        "id": "ANSSI-TLS-03",
        "title": "Désactiver SSL 2.0, SSL 3.0, TLS 1.0, TLS 1.1",
        "level": "essential",
        "checks": ["no_deprecated_protocols"],
    },
    "ANSSI-TLS-04": {
        "id": "ANSSI-TLS-04",
        "title": "Utiliser des suites cryptographiques robustes",
        "level": "essential",
        "checks": ["no_weak_ciphers"],
    },
    "ANSSI-TLS-05": {
        "id": "ANSSI-TLS-05",
        "title": "Utiliser ECDHE pour l'échange de clés",
        "level": "standard",
        "checks": ["ecdhe_preferred"],
    },
    "ANSSI-TLS-06": {
        "id": "ANSSI-TLS-06",
        "title": "Certificats valides et correctement configurés",
        "level": "essential",
        "checks": ["valid_certificates", "cert_not_expiring"],
    },
}

# =============================================================================
# NIST Cybersecurity Framework
# Reference: https://www.nist.gov/cyberframework
# =============================================================================

NIST_CSF_CONTROLS = {
    # Identify (ID)
    "NIST-ID.AM-1": {
        "id": "NIST-ID.AM-1",
        "function": "Identify",
        "category": "Asset Management",
        "title": "Physical devices and systems are inventoried",
        "checks": ["has_network_inventory"],
    },
    "NIST-ID.AM-2": {
        "id": "NIST-ID.AM-2",
        "function": "Identify",
        "category": "Asset Management",
        "title": "Software platforms and applications are inventoried",
        "checks": ["has_service_inventory"],
    },
    # Protect (PR)
    "NIST-PR.AC-1": {
        "id": "NIST-PR.AC-1",
        "function": "Protect",
        "category": "Access Control",
        "title": "Identities and credentials are managed",
        "checks": ["no_default_credentials"],
    },
    "NIST-PR.AC-5": {
        "id": "NIST-PR.AC-5",
        "function": "Protect",
        "category": "Access Control",
        "title": "Network integrity is protected",
        "checks": ["limited_exposure", "no_dangerous_services"],
    },
    "NIST-PR.DS-2": {
        "id": "NIST-PR.DS-2",
        "function": "Protect",
        "category": "Data Security",
        "title": "Data-in-transit is protected",
        "checks": ["tls_1_2_minimum", "no_weak_ciphers"],
    },
    "NIST-PR.IP-12": {
        "id": "NIST-PR.IP-12",
        "function": "Protect",
        "category": "Protective Technology",
        "title": "Vulnerability management plan is implemented",
        "checks": ["no_critical_cves", "no_high_cves"],
    },
    # Detect (DE)
    "NIST-DE.CM-1": {
        "id": "NIST-DE.CM-1",
        "function": "Detect",
        "category": "Continuous Monitoring",
        "title": "Network is monitored for events",
        "checks": [],  # Manual check
    },
    "NIST-DE.CM-8": {
        "id": "NIST-DE.CM-8",
        "function": "Detect",
        "category": "Continuous Monitoring",
        "title": "Vulnerability scans are performed",
        "checks": ["vulnerability_scan_performed"],
    },
}

# =============================================================================
# CIS Controls v8
# Reference: https://www.cisecurity.org/controls
# =============================================================================

CIS_CONTROLS = {
    # CIS Control 1: Inventory of Enterprise Assets
    "CIS-1.1": {
        "id": "CIS-1.1",
        "control": 1,
        "title": "Establish and Maintain Detailed Enterprise Asset Inventory",
        "ig": 1,  # Implementation Group
        "checks": ["has_network_inventory"],
    },
    # CIS Control 2: Inventory of Software Assets
    "CIS-2.1": {
        "id": "CIS-2.1",
        "control": 2,
        "title": "Establish and Maintain a Software Inventory",
        "ig": 1,
        "checks": ["has_service_inventory"],
    },
    # CIS Control 3: Data Protection
    "CIS-3.10": {
        "id": "CIS-3.10",
        "control": 3,
        "title": "Encrypt Sensitive Data in Transit",
        "ig": 1,
        "checks": ["tls_1_2_minimum", "no_weak_ciphers"],
    },
    # CIS Control 4: Secure Configuration
    "CIS-4.1": {
        "id": "CIS-4.1",
        "control": 4,
        "title": "Establish Secure Configuration Process",
        "ig": 1,
        "checks": ["no_default_credentials", "no_dangerous_services"],
    },
    "CIS-4.8": {
        "id": "CIS-4.8",
        "control": 4,
        "title": "Uninstall or Disable Unnecessary Services",
        "ig": 2,
        "checks": ["no_unnecessary_services"],
    },
    # CIS Control 7: Continuous Vulnerability Management
    "CIS-7.1": {
        "id": "CIS-7.1",
        "control": 7,
        "title": "Establish Vulnerability Management Process",
        "ig": 1,
        "checks": ["vulnerability_scan_performed"],
    },
    "CIS-7.4": {
        "id": "CIS-7.4",
        "control": 7,
        "title": "Perform Automated Application Patch Management",
        "ig": 1,
        "checks": ["no_critical_cves", "no_high_cves"],
    },
    # CIS Control 12: Network Infrastructure Management
    "CIS-12.1": {
        "id": "CIS-12.1",
        "control": 12,
        "title": "Ensure Network Infrastructure is Up-to-Date",
        "ig": 1,
        "checks": ["no_outdated_software"],
    },
    # CIS Control 13: Network Monitoring and Defense
    "CIS-13.6": {
        "id": "CIS-13.6",
        "control": 13,
        "title": "Collect DNS Traffic Logs",
        "ig": 2,
        "checks": [],  # Manual check
    },
}

# Dangerous services that should not be exposed
DANGEROUS_SERVICES = ["telnet", "ftp", "rsh", "rlogin", "rexec", "tftp", "finger"]
DANGEROUS_PORTS = [21, 23, 69, 79, 512, 513, 514]

# Weak TLS versions
WEAK_TLS = ["ssl2", "ssl3", "tls10", "tls11", "ssl_2_0", "ssl_3_0", "tls_1_0", "tls_1_1"]


class ComplianceKernel(Kernel):
    """
    Compliance assessment kernel.

    Configuration options:
        frameworks: List of frameworks ["anssi", "nist", "cis"]
        anssi_level: "essential", "standard", "reinforced"
        include_recommendations: bool

    Example manifest:
        compliance:
          enabled: true
          options:
            frameworks: ["anssi", "nist", "cis"]
            anssi_level: "standard"
    """

    name = "compliance"
    version = "1.0.0"
    category = "security"
    stage = 2
    description = "Compliance assessment (ANSSI, NIST, CIS)"

    requires = ["port_scan"]
    provides = ["compliance_scores", "compliance_findings"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Evaluate compliance against selected frameworks."""

        # Load all available data
        data = self._load_stage_data(input)

        # Configuration
        frameworks = input.config.get("frameworks", ["anssi", "nist", "cis"])
        anssi_level = input.config.get("anssi_level", "standard")
        include_recommendations = input.config.get("include_recommendations", True)

        logger.info(f"[compliance] Evaluating against {frameworks}")

        # Build check results
        check_results = self._run_checks(data)

        # Evaluate each framework
        compliance_results = {}
        all_findings = []

        if "anssi" in frameworks:
            anssi_result = self._evaluate_anssi(check_results, anssi_level)
            compliance_results["anssi"] = anssi_result
            all_findings.extend(anssi_result["findings"])

        if "nist" in frameworks:
            nist_result = self._evaluate_nist(check_results)
            compliance_results["nist"] = nist_result
            all_findings.extend(nist_result["findings"])

        if "cis" in frameworks:
            cis_result = self._evaluate_cis(check_results)
            compliance_results["cis"] = cis_result
            all_findings.extend(cis_result["findings"])

        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(all_findings)

        # Calculate overall score
        total_controls = sum(r.get("total_controls", 0) for r in compliance_results.values())
        compliant_controls = sum(r.get("compliant", 0) for r in compliance_results.values())
        overall_score = (compliant_controls / total_controls * 100) if total_controls > 0 else 0

        return {
            "compliance_scores": compliance_results,
            "findings": all_findings,
            "recommendations": recommendations,
            "check_results": check_results,
            "statistics": {
                "frameworks_evaluated": len(compliance_results),
                "total_controls": total_controls,
                "compliant_controls": compliant_controls,
                "non_compliant_controls": total_controls - compliant_controls,
                "overall_score": round(overall_score, 1),
            },
        }

    def _load_stage_data(self, input: KernelInput) -> Dict[str, Any]:
        """Load all stage data for compliance checks."""
        data = {
            "port_scan": None,
            "ssl_analysis": None,
            "dns_enum": None,
            "vuln_assess": None,
            "risk_network": None,
        }

        stage_files = [
            ("port_scan", "stage1"),
            ("dns_enum", "stage1"),
            ("ssl_analysis", "stage2"),
            ("vuln_assess", "stage2"),
            ("risk_network", "stage2"),
        ]

        for kernel_name, stage_dir in stage_files:
            file_path = input.workspace / stage_dir / f"{kernel_name}.json"
            if file_path.exists():
                with open(file_path) as f:
                    data[kernel_name] = json.load(f).get("data", {})

        return data

    def _run_checks(self, data: Dict) -> Dict[str, bool]:
        """Run all compliance checks and return results."""
        results = {}

        # Network inventory checks
        port_scan = data.get("port_scan") or {}
        services = port_scan.get("services", [])
        hosts = set(s.get("host") for s in services)

        results["has_network_inventory"] = len(hosts) > 0
        results["has_service_inventory"] = len(services) > 0

        # Dangerous services check
        dangerous_found = False
        for svc in services:
            svc_name = svc.get("service", "").lower()
            port = svc.get("port", 0)
            if svc_name in DANGEROUS_SERVICES or port in DANGEROUS_PORTS:
                dangerous_found = True
                break

        results["no_dangerous_services"] = not dangerous_found
        results["no_telnet"] = not any(
            s.get("service", "").lower() == "telnet" or s.get("port") == 23
            for s in services
        )

        # Limited exposure (less than 20 open ports per host average)
        if hosts:
            avg_ports = len(services) / len(hosts)
            results["limited_exposure"] = avg_ports < 20
        else:
            results["limited_exposure"] = True

        # SSH secure (if SSH present, no weak versions)
        ssh_services = [s for s in services if s.get("service", "").lower() == "ssh"]
        ssh_secure = True
        for ssh in ssh_services:
            version = ssh.get("version", "").lower()
            # Check for old SSH versions
            if "openssh" in version:
                # Extract version number
                import re
                match = re.search(r"openssh[_\s]*(\d+\.?\d*)", version)
                if match:
                    ver = float(match.group(1))
                    if ver < 7.0:
                        ssh_secure = False
        results["ssh_secure"] = ssh_secure

        # TLS checks
        ssl_analysis = data.get("ssl_analysis") or {}
        ssl_vulns = ssl_analysis.get("vulnerabilities", [])
        ciphers = ssl_analysis.get("cipher_suites", {})

        # Check for deprecated protocols
        deprecated_protocol_found = any(
            "deprecated_protocol" in v.get("type", "")
            for v in ssl_vulns
        )
        results["no_deprecated_protocols"] = not deprecated_protocol_found
        results["tls_1_2_minimum"] = not deprecated_protocol_found

        # Check for TLS 1.3
        tls13_found = False
        for host_ciphers in ciphers.values():
            for cipher in host_ciphers:
                if "tls13" in cipher.get("version", "").lower():
                    tls13_found = True
                    break
        results["tls_1_3_preferred"] = tls13_found

        # Weak ciphers check
        weak_cipher_found = any(
            cipher.get("weak", False)
            for host_ciphers in ciphers.values()
            for cipher in host_ciphers
        )
        results["no_weak_ciphers"] = not weak_cipher_found

        # ECDHE preference
        ecdhe_found = False
        for host_ciphers in ciphers.values():
            for cipher in host_ciphers:
                if "ECDHE" in cipher.get("name", ""):
                    ecdhe_found = True
                    break
        results["ecdhe_preferred"] = ecdhe_found

        # Certificate checks
        certificates = ssl_analysis.get("certificates", [])
        valid_certs = all(c.get("valid", False) for c in certificates) if certificates else True
        expiring_soon = any(c.get("days_until_expiry", 999) < 30 for c in certificates)

        results["valid_certificates"] = valid_certs
        results["cert_not_expiring"] = not expiring_soon

        # DNS checks
        dns_enum = data.get("dns_enum") or {}
        dns_misconfigs = dns_enum.get("misconfigurations", [])

        zone_transfer = any(
            "zone_transfer" in m.get("type", "")
            for m in dns_misconfigs
        )
        results["no_zone_transfer"] = not zone_transfer

        # SPF/DMARC checks from DNS records
        records = dns_enum.get("records", {})
        has_spf = False
        has_dmarc = False
        for domain_records in records.values():
            txt_records = domain_records.get("TXT", [])
            for txt in txt_records:
                txt_value = txt.get("value", "").lower()
                if "v=spf1" in txt_value:
                    has_spf = True
                if "v=dmarc1" in txt_value:
                    has_dmarc = True

        results["has_spf"] = has_spf
        results["has_dmarc"] = has_dmarc
        results["has_dnssec"] = False  # Would need to check DNSKEY/DS records

        # Vulnerability checks
        vuln_assess = data.get("vuln_assess") or {}
        vulns = vuln_assess.get("vulnerabilities", [])

        results["vulnerability_scan_performed"] = data.get("vuln_assess") is not None

        critical_cves = [v for v in vulns if v.get("severity") == "critical"]
        high_cves = [v for v in vulns if v.get("severity") == "high"]

        results["no_critical_cves"] = len(critical_cves) == 0
        results["no_high_cves"] = len(high_cves) == 0

        # Outdated software (from nuclei findings)
        outdated_found = any(
            "outdated" in v.get("name", "").lower() or
            "version" in v.get("name", "").lower()
            for v in vulns
        )
        results["no_outdated_software"] = not outdated_found

        # Default credentials
        default_creds = any(
            "default" in v.get("name", "").lower() and
            ("login" in v.get("name", "").lower() or "credential" in v.get("name", "").lower())
            for v in vulns
        )
        results["no_default_credentials"] = not default_creds

        # Unnecessary services (placeholder)
        results["no_unnecessary_services"] = True  # Would need baseline comparison
        results["firewall_present"] = True  # Would need specific check

        return results

    def _evaluate_anssi(
        self,
        check_results: Dict[str, bool],
        level: str
    ) -> Dict[str, Any]:
        """Evaluate ANSSI compliance."""

        level_order = {"essential": 0, "standard": 1, "reinforced": 2}
        target_level = level_order.get(level, 1)

        all_rules = {**ANSSI_HYGIENE_RULES, **ANSSI_TLS_RULES}

        compliant = 0
        non_compliant = 0
        findings = []

        for rule_id, rule in all_rules.items():
            rule_level = level_order.get(rule.get("level", "essential"), 0)

            # Skip rules above target level
            if rule_level > target_level:
                continue

            # Check all required checks
            checks = rule.get("checks", [])
            if not checks:
                # Manual check - assume compliant for now
                compliant += 1
                continue

            is_compliant = all(check_results.get(check, False) for check in checks)

            if is_compliant:
                compliant += 1
            else:
                non_compliant += 1
                failed_checks = [c for c in checks if not check_results.get(c, False)]
                findings.append({
                    "framework": "anssi",
                    "rule_id": rule_id,
                    "title": rule.get("title", ""),
                    "level": rule.get("level", ""),
                    "category": rule.get("category", ""),
                    "status": "non_compliant",
                    "failed_checks": failed_checks,
                })

        total = compliant + non_compliant
        score = (compliant / total * 100) if total > 0 else 0

        return {
            "framework": "ANSSI",
            "level": level,
            "total_controls": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "score": round(score, 1),
            "findings": findings,
        }

    def _evaluate_nist(self, check_results: Dict[str, bool]) -> Dict[str, Any]:
        """Evaluate NIST CSF compliance."""

        compliant = 0
        non_compliant = 0
        findings = []

        for control_id, control in NIST_CSF_CONTROLS.items():
            checks = control.get("checks", [])

            if not checks:
                compliant += 1
                continue

            is_compliant = all(check_results.get(check, False) for check in checks)

            if is_compliant:
                compliant += 1
            else:
                non_compliant += 1
                failed_checks = [c for c in checks if not check_results.get(c, False)]
                findings.append({
                    "framework": "nist",
                    "rule_id": control_id,
                    "title": control.get("title", ""),
                    "function": control.get("function", ""),
                    "category": control.get("category", ""),
                    "status": "non_compliant",
                    "failed_checks": failed_checks,
                })

        total = compliant + non_compliant
        score = (compliant / total * 100) if total > 0 else 0

        return {
            "framework": "NIST CSF",
            "total_controls": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "score": round(score, 1),
            "findings": findings,
        }

    def _evaluate_cis(self, check_results: Dict[str, bool]) -> Dict[str, Any]:
        """Evaluate CIS Controls compliance."""

        compliant = 0
        non_compliant = 0
        findings = []

        for control_id, control in CIS_CONTROLS.items():
            checks = control.get("checks", [])

            if not checks:
                compliant += 1
                continue

            is_compliant = all(check_results.get(check, False) for check in checks)

            if is_compliant:
                compliant += 1
            else:
                non_compliant += 1
                failed_checks = [c for c in checks if not check_results.get(c, False)]
                findings.append({
                    "framework": "cis",
                    "rule_id": control_id,
                    "title": control.get("title", ""),
                    "control": control.get("control", 0),
                    "ig": control.get("ig", 1),
                    "status": "non_compliant",
                    "failed_checks": failed_checks,
                })

        total = compliant + non_compliant
        score = (compliant / total * 100) if total > 0 else 0

        return {
            "framework": "CIS Controls v8",
            "total_controls": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "score": round(score, 1),
            "findings": findings,
        }

    def _generate_recommendations(
        self,
        findings: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate remediation recommendations from findings."""

        recommendations = []
        seen_checks = set()

        # Check-specific recommendations
        check_recommendations = {
            "no_dangerous_services": {
                "priority": "high",
                "action": "Désactiver les services dangereux (telnet, FTP, rsh) et les remplacer par des alternatives sécurisées (SSH, SFTP)",
                "action_en": "Disable dangerous services (telnet, FTP, rsh) and replace with secure alternatives (SSH, SFTP)",
            },
            "no_telnet": {
                "priority": "critical",
                "action": "Désactiver immédiatement telnet et utiliser SSH",
                "action_en": "Immediately disable telnet and use SSH",
            },
            "tls_1_2_minimum": {
                "priority": "high",
                "action": "Configurer TLS 1.2 minimum sur tous les services. Désactiver SSL 2.0, SSL 3.0, TLS 1.0 et TLS 1.1",
                "action_en": "Configure TLS 1.2 minimum on all services. Disable SSL 2.0, SSL 3.0, TLS 1.0 and TLS 1.1",
            },
            "no_weak_ciphers": {
                "priority": "high",
                "action": "Désactiver les suites cryptographiques faibles (RC4, 3DES, NULL, EXPORT, MD5)",
                "action_en": "Disable weak cipher suites (RC4, 3DES, NULL, EXPORT, MD5)",
            },
            "no_deprecated_protocols": {
                "priority": "high",
                "action": "Désactiver les protocoles obsolètes (SSL 2.0, SSL 3.0, TLS 1.0, TLS 1.1)",
                "action_en": "Disable deprecated protocols (SSL 2.0, SSL 3.0, TLS 1.0, TLS 1.1)",
            },
            "valid_certificates": {
                "priority": "medium",
                "action": "Renouveler les certificats expirés ou invalides",
                "action_en": "Renew expired or invalid certificates",
            },
            "cert_not_expiring": {
                "priority": "medium",
                "action": "Planifier le renouvellement des certificats avant expiration",
                "action_en": "Schedule certificate renewal before expiration",
            },
            "no_zone_transfer": {
                "priority": "high",
                "action": "Restreindre les transferts de zone DNS aux serveurs autorisés uniquement",
                "action_en": "Restrict DNS zone transfers to authorized servers only",
            },
            "has_spf": {
                "priority": "medium",
                "action": "Configurer un enregistrement SPF pour prévenir l'usurpation d'email",
                "action_en": "Configure SPF record to prevent email spoofing",
            },
            "has_dmarc": {
                "priority": "medium",
                "action": "Configurer un enregistrement DMARC pour renforcer la protection email",
                "action_en": "Configure DMARC record to strengthen email protection",
            },
            "no_critical_cves": {
                "priority": "critical",
                "action": "Appliquer immédiatement les correctifs de sécurité pour les CVE critiques",
                "action_en": "Immediately apply security patches for critical CVEs",
            },
            "no_high_cves": {
                "priority": "high",
                "action": "Planifier l'application des correctifs pour les CVE de sévérité haute",
                "action_en": "Schedule patching for high severity CVEs",
            },
            "no_default_credentials": {
                "priority": "critical",
                "action": "Modifier immédiatement tous les identifiants par défaut",
                "action_en": "Immediately change all default credentials",
            },
            "limited_exposure": {
                "priority": "medium",
                "action": "Réduire la surface d'attaque en fermant les ports inutiles",
                "action_en": "Reduce attack surface by closing unnecessary ports",
            },
        }

        for finding in findings:
            for check in finding.get("failed_checks", []):
                if check in seen_checks:
                    continue
                seen_checks.add(check)

                if check in check_recommendations:
                    rec = check_recommendations[check]
                    recommendations.append({
                        "check": check,
                        "priority": rec["priority"],
                        "frameworks": [finding.get("framework", "")],
                        "action_fr": rec["action"],
                        "action": rec["action_en"],
                    })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

        return recommendations

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        scores = data.get("compliance_scores", {})

        framework_scores = []
        for fw, result in scores.items():
            framework_scores.append(f"{result.get('framework', fw)}: {result.get('score', 0)}%")

        return (
            f"Compliance Assessment: {stats.get('frameworks_evaluated', 0)} frameworks. "
            f"Overall: {stats.get('overall_score', 0)}%. "
            f"Controls: {stats.get('compliant_controls', 0)}/{stats.get('total_controls', 0)} compliant. "
            f"Scores: {', '.join(framework_scores)}."
        )
