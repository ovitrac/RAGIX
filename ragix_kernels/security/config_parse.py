"""
Kernel: Configuration Parser
Stage: 1 (Discovery)
Category: security

Parses and analyzes network device configurations (firewalls, routers, switches).
Extracts security-relevant rules, identifies misconfigurations, and maps network topology.

Supported formats:
- iptables / nftables (Linux)
- Cisco IOS / IOS-XE
- Cisco ASA
- pfSense / OPNsense (XML)
- Fortinet FortiGate
- Generic rule-based configs

Input:
    config_files: List of configuration file paths
    config_type: "auto", "iptables", "cisco_ios", "cisco_asa", "pfsense", "fortigate"
    analyze_rules: Analyze firewall rules (default: true)
    check_best_practices: Check against best practices (default: true)

Output:
    devices: Parsed device configurations
    rules: Firewall/ACL rules extracted
    findings: Security issues and misconfigurations
    network_map: Network topology inferred from configs

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 1 -k config_parse

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Dangerous ports/services that should not be exposed
DANGEROUS_PORTS = {
    21: ("ftp", "high"),
    23: ("telnet", "critical"),
    69: ("tftp", "high"),
    111: ("rpcbind", "medium"),
    135: ("msrpc", "medium"),
    139: ("netbios", "medium"),
    445: ("smb", "high"),
    512: ("rexec", "critical"),
    513: ("rlogin", "critical"),
    514: ("rsh", "critical"),
    1433: ("mssql", "medium"),
    1521: ("oracle", "medium"),
    3306: ("mysql", "medium"),
    3389: ("rdp", "medium"),
    5432: ("postgres", "medium"),
    5900: ("vnc", "medium"),
    6379: ("redis", "high"),
    27017: ("mongodb", "medium"),
}

# Best practice rules
BEST_PRACTICES = {
    "explicit_deny": "Default policy should be DENY/DROP",
    "no_any_any": "Avoid 'any to any' rules",
    "no_permit_all": "Avoid permitting all traffic",
    "log_denied": "Log denied traffic",
    "no_telnet": "Block telnet (port 23)",
    "no_rsh": "Block remote shell services (512-514)",
    "limit_icmp": "Limit ICMP to necessary types",
    "no_source_routing": "Disable IP source routing",
}


class ConfigParseKernel(Kernel):
    """
    Network configuration parser and analyzer kernel.

    Configuration options:
        config_files: List of configuration file paths
        config_dir: Directory containing config files
        config_type: "auto" or specific type
        analyze_rules: Analyze firewall rules
        check_best_practices: Check against best practices
        extract_topology: Extract network topology

    Example manifest:
        config_parse:
          enabled: true
          options:
            config_dir: "data/configs"
            config_type: "auto"
            check_best_practices: true
    """

    name = "config_parse"
    version = "1.0.0"
    category = "security"
    stage = 1
    description = "Network configuration parser and analyzer"

    requires = []
    provides = ["firewall_rules", "network_topology", "config_findings"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Parse and analyze network configurations."""

        # Get configuration files
        config_files = self._get_config_files(input)
        if not config_files:
            return {
                "devices": [],
                "rules": [],
                "findings": [],
                "network_map": {},
                "statistics": {"error": "No configuration files found"},
            }

        config_type = input.config.get("config_type", "auto")
        analyze_rules = input.config.get("analyze_rules", True)
        check_bp = input.config.get("check_best_practices", True)

        logger.info(f"[config_parse] Processing {len(config_files)} config file(s)")

        all_devices = []
        all_rules = []
        all_findings = []
        network_map = {
            "interfaces": [],
            "networks": [],
            "routes": [],
            "nat_rules": [],
        }

        for config_file in config_files:
            logger.info(f"[config_parse] Parsing {config_file}")

            # Detect config type
            detected_type = self._detect_config_type(config_file) if config_type == "auto" else config_type

            # Parse configuration
            device, rules, topology = self._parse_config(config_file, detected_type)

            if device:
                all_devices.append(device)
            all_rules.extend(rules)

            # Merge topology
            network_map["interfaces"].extend(topology.get("interfaces", []))
            network_map["networks"].extend(topology.get("networks", []))
            network_map["routes"].extend(topology.get("routes", []))
            network_map["nat_rules"].extend(topology.get("nat_rules", []))

            # Analyze rules
            if analyze_rules:
                findings = self._analyze_rules(rules, config_file)
                all_findings.extend(findings)

            # Check best practices
            if check_bp:
                bp_findings = self._check_best_practices(rules, device, config_file)
                all_findings.extend(bp_findings)

        # Sort findings by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        all_findings.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 4))

        # Statistics
        statistics = {
            "devices_parsed": len(all_devices),
            "total_rules": len(all_rules),
            "permit_rules": len([r for r in all_rules if r.get("action") == "permit"]),
            "deny_rules": len([r for r in all_rules if r.get("action") == "deny"]),
            "findings_count": len(all_findings),
            "critical_findings": len([f for f in all_findings if f.get("severity") == "critical"]),
            "high_findings": len([f for f in all_findings if f.get("severity") == "high"]),
        }

        return {
            "devices": all_devices,
            "rules": all_rules,
            "findings": all_findings,
            "network_map": network_map,
            "statistics": statistics,
        }

    def _get_config_files(self, input: KernelInput) -> List[Path]:
        """Get configuration files from input."""
        files = []

        # Check explicit file list
        config_files = input.config.get("config_files", [])
        for cf in config_files:
            path = Path(cf) if Path(cf).is_absolute() else input.workspace / cf
            if path.exists():
                files.append(path)

        # Check config directory
        config_dir = input.config.get("config_dir")
        if config_dir:
            dir_path = Path(config_dir) if Path(config_dir).is_absolute() else input.workspace / config_dir
            if dir_path.exists() and dir_path.is_dir():
                # Find config files
                for pattern in ["*.conf", "*.cfg", "*.txt", "*.xml", "*.rules"]:
                    files.extend(dir_path.glob(pattern))

        # Check default locations
        default_dirs = [
            input.workspace / "data" / "configs",
            input.workspace / "data" / "firewall",
        ]
        for dd in default_dirs:
            if dd.exists():
                for pattern in ["*.conf", "*.cfg", "*.txt", "*.xml", "*.rules"]:
                    files.extend(dd.glob(pattern))

        return list(set(files))

    def _detect_config_type(self, config_file: Path) -> str:
        """Auto-detect configuration type."""
        content = config_file.read_text(errors="ignore")[:5000]

        # XML detection
        if content.strip().startswith("<?xml") or "<pfsense>" in content:
            return "pfsense"

        # Cisco IOS patterns
        if "hostname" in content and ("interface " in content or "ip route" in content):
            if "access-list " in content or "ip access-list" in content:
                return "cisco_ios"

        # Cisco ASA patterns
        if "ASA Version" in content or ("access-group" in content and "nameif" in content):
            return "cisco_asa"

        # FortiGate patterns
        if "config firewall policy" in content or "set srcintf" in content:
            return "fortigate"

        # iptables patterns
        if "*filter" in content or "-A INPUT" in content or "-A FORWARD" in content:
            return "iptables"

        # nftables patterns
        if "table" in content and "chain" in content and "{" in content:
            return "nftables"

        return "generic"

    def _parse_config(
        self,
        config_file: Path,
        config_type: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse configuration file based on type."""

        content = config_file.read_text(errors="ignore")

        parsers = {
            "iptables": self._parse_iptables,
            "nftables": self._parse_nftables,
            "cisco_ios": self._parse_cisco_ios,
            "cisco_asa": self._parse_cisco_asa,
            "pfsense": self._parse_pfsense,
            "fortigate": self._parse_fortigate,
            "generic": self._parse_generic,
        }

        parser = parsers.get(config_type, self._parse_generic)
        return parser(content, str(config_file))

    def _parse_iptables(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse iptables configuration."""
        device = {
            "name": Path(filename).stem,
            "type": "linux_firewall",
            "config_type": "iptables",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        # Parse chains and policies
        current_table = "filter"
        for line in content.split("\n"):
            line = line.strip()

            # Table marker
            if line.startswith("*"):
                current_table = line[1:]
                continue

            # Policy
            if line.startswith(":"):
                parts = line.split()
                if len(parts) >= 2:
                    chain = parts[0][1:]
                    policy = parts[1]
                    rules.append({
                        "type": "policy",
                        "table": current_table,
                        "chain": chain,
                        "action": "deny" if policy == "DROP" else "permit",
                        "raw": line,
                    })

            # Rule
            if line.startswith("-A"):
                rule = self._parse_iptables_rule(line, current_table)
                if rule:
                    rules.append(rule)

        return device, rules, topology

    def _parse_iptables_rule(self, line: str, table: str) -> Optional[Dict]:
        """Parse a single iptables rule."""
        try:
            parts = line.split()
            rule = {
                "type": "rule",
                "table": table,
                "chain": parts[1] if len(parts) > 1 else "",
                "source": "any",
                "destination": "any",
                "protocol": "any",
                "port": None,
                "action": "permit",
                "raw": line,
            }

            i = 2
            while i < len(parts):
                opt = parts[i]

                if opt == "-s" and i + 1 < len(parts):
                    rule["source"] = parts[i + 1]
                    i += 2
                elif opt == "-d" and i + 1 < len(parts):
                    rule["destination"] = parts[i + 1]
                    i += 2
                elif opt == "-p" and i + 1 < len(parts):
                    rule["protocol"] = parts[i + 1]
                    i += 2
                elif opt == "--dport" and i + 1 < len(parts):
                    rule["port"] = parts[i + 1]
                    i += 2
                elif opt == "-j" and i + 1 < len(parts):
                    action = parts[i + 1]
                    rule["action"] = "deny" if action in ["DROP", "REJECT"] else "permit"
                    i += 2
                else:
                    i += 1

            return rule
        except Exception:
            return None

    def _parse_nftables(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse nftables configuration."""
        device = {
            "name": Path(filename).stem,
            "type": "linux_firewall",
            "config_type": "nftables",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        # Simplified nftables parsing
        current_table = ""
        current_chain = ""

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("table"):
                match = re.match(r"table\s+(\w+)\s+(\w+)", line)
                if match:
                    current_table = match.group(2)

            elif line.startswith("chain"):
                match = re.match(r"chain\s+(\w+)", line)
                if match:
                    current_chain = match.group(1)

            elif any(line.startswith(kw) for kw in ["accept", "drop", "reject"]):
                action = "permit" if line.startswith("accept") else "deny"
                rules.append({
                    "type": "rule",
                    "table": current_table,
                    "chain": current_chain,
                    "action": action,
                    "raw": line,
                    "source": "any",
                    "destination": "any",
                    "protocol": "any",
                })

        return device, rules, topology

    def _parse_cisco_ios(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse Cisco IOS configuration."""
        device = {
            "name": "",
            "type": "cisco_router",
            "config_type": "cisco_ios",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        # Extract hostname
        hostname_match = re.search(r"hostname\s+(\S+)", content)
        if hostname_match:
            device["name"] = hostname_match.group(1)

        # Parse interfaces
        interface_blocks = re.findall(
            r"interface\s+(\S+)(.*?)(?=interface\s+|\Z)",
            content,
            re.DOTALL
        )
        for iface_name, iface_config in interface_blocks:
            iface = {"name": iface_name, "ip": None, "description": ""}

            ip_match = re.search(r"ip address\s+(\S+)\s+(\S+)", iface_config)
            if ip_match:
                iface["ip"] = f"{ip_match.group(1)}/{ip_match.group(2)}"

            desc_match = re.search(r"description\s+(.+)", iface_config)
            if desc_match:
                iface["description"] = desc_match.group(1).strip()

            topology["interfaces"].append(iface)

        # Parse access lists
        acl_patterns = [
            r"access-list\s+(\d+)\s+(permit|deny)\s+(.+)",
            r"ip access-list\s+\w+\s+(\S+)(.*?)(?=ip access-list|\Z)",
        ]

        # Standard/extended ACLs
        for match in re.finditer(r"access-list\s+(\d+)\s+(permit|deny)\s+(.+)", content):
            acl_num = match.group(1)
            action = match.group(2)
            rest = match.group(3)

            rules.append({
                "type": "acl",
                "acl_name": acl_num,
                "action": action,
                "raw": match.group(0),
                "source": "any",
                "destination": "any",
                "protocol": "any",
            })

        # Parse routes
        for match in re.finditer(r"ip route\s+(\S+)\s+(\S+)\s+(\S+)", content):
            topology["routes"].append({
                "network": match.group(1),
                "mask": match.group(2),
                "next_hop": match.group(3),
            })

        return device, rules, topology

    def _parse_cisco_asa(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse Cisco ASA configuration."""
        device = {
            "name": "",
            "type": "cisco_asa",
            "config_type": "cisco_asa",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        # Extract hostname
        hostname_match = re.search(r"hostname\s+(\S+)", content)
        if hostname_match:
            device["name"] = hostname_match.group(1)

        # Parse access-group (applied ACLs)
        for match in re.finditer(r"access-list\s+(\S+)\s+\w+\s+(permit|deny)\s+(.+)", content):
            acl_name = match.group(1)
            action = match.group(2)
            rule_spec = match.group(3)

            rule = {
                "type": "acl",
                "acl_name": acl_name,
                "action": action,
                "raw": match.group(0),
                "source": "any",
                "destination": "any",
                "protocol": "any",
            }

            # Parse protocol and ports
            parts = rule_spec.split()
            if parts:
                rule["protocol"] = parts[0]

            rules.append(rule)

        # Parse NAT rules
        for match in re.finditer(r"nat\s+\((\S+),(\S+)\)\s+(.+)", content):
            topology["nat_rules"].append({
                "source_if": match.group(1),
                "dest_if": match.group(2),
                "rule": match.group(3),
            })

        return device, rules, topology

    def _parse_pfsense(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse pfSense XML configuration."""
        device = {
            "name": "",
            "type": "pfsense",
            "config_type": "pfsense",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        try:
            root = ET.fromstring(content)

            # Hostname
            hostname = root.find(".//hostname")
            if hostname is not None:
                device["name"] = hostname.text

            # Interfaces
            for iface in root.findall(".//interfaces/*"):
                iface_data = {
                    "name": iface.tag,
                    "ip": "",
                    "description": "",
                }
                ipaddr = iface.find("ipaddr")
                if ipaddr is not None:
                    iface_data["ip"] = ipaddr.text
                descr = iface.find("descr")
                if descr is not None:
                    iface_data["description"] = descr.text or ""

                topology["interfaces"].append(iface_data)

            # Firewall rules
            for rule in root.findall(".//filter/rule"):
                rule_data = {
                    "type": "rule",
                    "action": "permit",
                    "source": "any",
                    "destination": "any",
                    "protocol": "any",
                    "port": None,
                    "raw": "",
                }

                rule_type = rule.find("type")
                if rule_type is not None:
                    rule_data["action"] = "permit" if rule_type.text == "pass" else "deny"

                proto = rule.find("protocol")
                if proto is not None:
                    rule_data["protocol"] = proto.text

                src = rule.find("source/any")
                if src is None:
                    src_addr = rule.find("source/address")
                    if src_addr is not None:
                        rule_data["source"] = src_addr.text

                dst = rule.find("destination/any")
                if dst is None:
                    dst_addr = rule.find("destination/address")
                    if dst_addr is not None:
                        rule_data["destination"] = dst_addr.text

                dst_port = rule.find("destination/port")
                if dst_port is not None:
                    rule_data["port"] = dst_port.text

                rules.append(rule_data)

        except ET.ParseError as e:
            logger.error(f"[config_parse] XML parse error: {e}")

        return device, rules, topology

    def _parse_fortigate(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Parse FortiGate configuration."""
        device = {
            "name": "",
            "type": "fortigate",
            "config_type": "fortigate",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        # Extract hostname
        hostname_match = re.search(r'set hostname "([^"]+)"', content)
        if hostname_match:
            device["name"] = hostname_match.group(1)

        # Parse firewall policies
        policy_blocks = re.findall(
            r"edit\s+(\d+)(.*?)next",
            content,
            re.DOTALL
        )

        for policy_id, policy_content in policy_blocks:
            rule = {
                "type": "policy",
                "id": policy_id,
                "action": "permit",
                "source": "any",
                "destination": "any",
                "protocol": "any",
                "raw": policy_content[:200],
            }

            # Action
            action_match = re.search(r"set action\s+(\w+)", policy_content)
            if action_match:
                rule["action"] = "permit" if action_match.group(1) == "accept" else "deny"

            # Source/Destination
            srcaddr_match = re.search(r'set srcaddr\s+"([^"]+)"', policy_content)
            if srcaddr_match:
                rule["source"] = srcaddr_match.group(1)

            dstaddr_match = re.search(r'set dstaddr\s+"([^"]+)"', policy_content)
            if dstaddr_match:
                rule["destination"] = dstaddr_match.group(1)

            rules.append(rule)

        return device, rules, topology

    def _parse_generic(
        self,
        content: str,
        filename: str
    ) -> Tuple[Optional[Dict], List[Dict], Dict]:
        """Generic configuration parser."""
        device = {
            "name": Path(filename).stem,
            "type": "generic",
            "config_type": "generic",
            "filename": filename,
        }

        rules = []
        topology = {"interfaces": [], "networks": [], "routes": [], "nat_rules": []}

        # Look for common patterns
        for line in content.split("\n"):
            line = line.strip()

            # Skip comments
            if line.startswith("#") or line.startswith("!") or not line:
                continue

            # Generic permit/deny pattern
            if re.search(r"\b(permit|deny|allow|drop|reject|accept)\b", line, re.I):
                action = "deny" if re.search(r"\b(deny|drop|reject)\b", line, re.I) else "permit"
                rules.append({
                    "type": "rule",
                    "action": action,
                    "raw": line,
                    "source": "any",
                    "destination": "any",
                    "protocol": "any",
                })

        return device, rules, topology

    def _analyze_rules(
        self,
        rules: List[Dict],
        config_file: Path
    ) -> List[Dict[str, Any]]:
        """Analyze rules for security issues."""
        findings = []

        for rule in rules:
            # Check for any-to-any permits
            if (rule.get("action") == "permit" and
                rule.get("source") == "any" and
                rule.get("destination") == "any" and
                rule.get("type") == "rule"):

                findings.append({
                    "type": "any_to_any",
                    "severity": "high",
                    "config_file": str(config_file),
                    "rule": rule.get("raw", ""),
                    "description": "Rule permits any-to-any traffic",
                    "recommendation": "Restrict source and destination to specific networks",
                })

            # Check for dangerous port exposure
            port = rule.get("port")
            if port and rule.get("action") == "permit":
                try:
                    port_num = int(port.split("-")[0].split(":")[0])
                    if port_num in DANGEROUS_PORTS:
                        service, severity = DANGEROUS_PORTS[port_num]
                        findings.append({
                            "type": "dangerous_port",
                            "severity": severity,
                            "config_file": str(config_file),
                            "rule": rule.get("raw", ""),
                            "port": port_num,
                            "service": service,
                            "description": f"Rule permits access to dangerous service: {service} (port {port_num})",
                            "recommendation": f"Block or restrict access to {service}",
                        })
                except (ValueError, AttributeError):
                    pass

        return findings

    def _check_best_practices(
        self,
        rules: List[Dict],
        device: Optional[Dict],
        config_file: Path
    ) -> List[Dict[str, Any]]:
        """Check configuration against best practices."""
        findings = []

        # Check for default deny policy
        policies = [r for r in rules if r.get("type") == "policy"]
        has_default_deny = any(
            r.get("action") == "deny" and r.get("chain") in ["INPUT", "FORWARD"]
            for r in policies
        )

        if not has_default_deny and policies:
            findings.append({
                "type": "best_practice",
                "check": "explicit_deny",
                "severity": "medium",
                "config_file": str(config_file),
                "description": "No explicit default deny policy found",
                "recommendation": "Set default policy to DROP/DENY for INPUT and FORWARD chains",
            })

        # Check for telnet permit
        for rule in rules:
            port = rule.get("port")
            if port and rule.get("action") == "permit":
                if port in ["23", "telnet"]:
                    findings.append({
                        "type": "best_practice",
                        "check": "no_telnet",
                        "severity": "critical",
                        "config_file": str(config_file),
                        "rule": rule.get("raw", ""),
                        "description": "Telnet access is permitted",
                        "recommendation": "Disable telnet and use SSH instead",
                    })

        return findings

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        findings = data.get("findings", [])

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            sev = f.get("severity", "low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return (
            f"Config Parse: {stats.get('devices_parsed', 0)} devices, "
            f"{stats.get('total_rules', 0)} rules "
            f"({stats.get('permit_rules', 0)} permit, {stats.get('deny_rules', 0)} deny). "
            f"Findings: {severity_counts['critical']} critical, "
            f"{severity_counts['high']} high, {severity_counts['medium']} medium."
        )
