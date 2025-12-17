"""
Kernel: Port Scanner
Stage: 1 (Discovery)
Category: security

Performs port scanning and service detection on discovered hosts.
Uses nmap for comprehensive service fingerprinting.

Wraps:
- nmap -sV (service version detection)
- nmap -sS (SYN scan, requires root)
- nmap -sT (TCP connect scan)
- nmap -sU (UDP scan)
- nmap --script=banner

Dependencies:
- net_discover: List of hosts to scan (optional, can use direct targets)

Input:
    ports: Port specification ("1-1024", "22,80,443", "top100")
    scan_type: "syn", "connect", "udp" (default: connect)
    service_detection: true/false (default: true)
    scripts: List of nmap scripts to run

Output:
    services: List of discovered services with port, protocol, version
    by_host: Services grouped by host
    statistics: Scan statistics

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 1 -k port_scan

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Common port presets
PORT_PRESETS = {
    "top20": "21,22,23,25,53,80,110,111,135,139,143,443,445,993,995,1723,3306,3389,5900,8080",
    "top100": "7,9,13,21-23,25-26,37,53,79-81,88,106,110-111,113,119,135,139,143-144,179,199,389,427,443-445,465,513-515,543-544,548,554,587,631,646,873,990,993,995,1025-1029,1110,1433,1720,1723,1755,1900,2000-2001,2049,2121,2717,3000,3128,3306,3389,3986,4899,5000,5009,5051,5060,5101,5190,5357,5432,5631,5666,5800,5900,6000-6001,6646,7070,8000,8008-8009,8080-8081,8443,8888,9100,9999-10000,32768,49152-49157",
    "web": "80,443,8000,8080,8443,8888,9000,9090,9443",
    "database": "1433,1521,3306,5432,6379,9042,27017",
    "common": "21,22,23,25,53,80,110,143,443,445,3306,3389,5432,8080",
}


class PortScanKernel(Kernel):
    """
    Port scanning and service detection kernel.

    Configuration options:
        targets: Direct target list (optional if net_discover ran)
        ports: Port specification (number, range, preset name)
        scan_type: "syn", "connect", "udp" (default: connect)
        service_detection: Enable version detection (default: true)
        scripts: Nmap scripts to run (e.g., ["banner", "http-title"])
        timeout: Scan timeout per host in seconds

    Example manifest:
        port_scan:
          enabled: true
          options:
            ports: "top100"
            scan_type: "connect"
            service_detection: true
    """

    name = "port_scan"
    version = "1.0.0"
    category = "security"
    stage = 1
    description = "Port scanning and service detection"

    requires = []  # net_discover is optional
    provides = ["services", "open_ports", "service_versions"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Scan ports and detect services."""

        # Get targets from net_discover or config
        targets = self._get_targets(input)
        if not targets:
            return {
                "services": [],
                "by_host": {},
                "statistics": {"error": "No targets to scan"},
            }

        # Get scan configuration
        ports = self._resolve_ports(input.config.get("ports", "top100"))
        scan_type = input.config.get("scan_type", "connect")
        service_detection = input.config.get("service_detection", True)
        scripts = input.config.get("scripts", [])
        timeout = input.config.get("timeout", 300)

        logger.info(f"[port_scan] Scanning {len(targets)} host(s), ports: {ports[:50]}...")

        all_services = []
        by_host = {}
        scan_stats = {
            "hosts_scanned": 0,
            "hosts_with_open_ports": 0,
            "total_open_ports": 0,
            "scan_time_sec": 0,
        }

        # Build nmap arguments
        nmap_args = self._build_nmap_args(scan_type, service_detection, scripts)

        for target in targets:
            host_ip = target if isinstance(target, str) else target.get("ip", target)
            logger.info(f"[port_scan] Scanning {host_ip}")

            services, stats = self._scan_host(host_ip, ports, nmap_args, timeout)

            if services:
                all_services.extend(services)
                by_host[host_ip] = services
                scan_stats["hosts_with_open_ports"] += 1
                scan_stats["total_open_ports"] += len(services)

            scan_stats["hosts_scanned"] += 1
            scan_stats["scan_time_sec"] += stats.get("elapsed", 0)

        # Sort services by risk (high ports, known vulnerable services first)
        all_services.sort(key=lambda x: (
            -self._service_risk_score(x),
            x.get("host", ""),
            x.get("port", 0),
        ))

        return {
            "services": all_services,
            "by_host": by_host,
            "targets_scanned": targets,
            "ports_specification": ports,
            "statistics": scan_stats,
        }

    def _get_targets(self, input: KernelInput) -> List[str]:
        """Get targets from net_discover output or config."""

        # First check for net_discover output
        net_discover_path = input.dependencies.get("net_discover")
        if net_discover_path and net_discover_path.exists():
            with open(net_discover_path) as f:
                data = json.load(f).get("data", {})
                hosts = data.get("hosts", [])
                return [h["ip"] for h in hosts if h.get("status") == "up"]

        # Check for existing stage1 output
        stage1_discover = input.workspace / "stage1" / "net_discover.json"
        if stage1_discover.exists():
            with open(stage1_discover) as f:
                data = json.load(f).get("data", {})
                hosts = data.get("hosts", [])
                return [h["ip"] for h in hosts if h.get("status") == "up"]

        # Use targets from config
        return input.config.get("targets", [])

    def _resolve_ports(self, ports_spec: str) -> str:
        """Resolve port specification (preset name or direct spec)."""
        if ports_spec.lower() in PORT_PRESETS:
            return PORT_PRESETS[ports_spec.lower()]
        return ports_spec

    def _build_nmap_args(
        self,
        scan_type: str,
        service_detection: bool,
        scripts: List[str]
    ) -> str:
        """Build nmap argument string."""
        args = []

        # Scan type
        if scan_type == "syn":
            args.append("-sS")  # Requires root
        elif scan_type == "udp":
            args.append("-sU")
        else:
            args.append("-sT")  # TCP connect (no root needed)

        # Service detection
        if service_detection:
            args.append("-sV")

        # Scripts
        if scripts:
            args.append(f"--script={','.join(scripts)}")

        # Timing (T4 = aggressive but not insane)
        args.append("-T4")

        return " ".join(args)

    def _scan_host(
        self,
        host: str,
        ports: str,
        nmap_args: str,
        timeout: int
    ) -> tuple[List[Dict], Dict]:
        """Scan a single host."""
        services = []
        stats = {"elapsed": 0}

        if NMAP_AVAILABLE:
            try:
                nm = nmap.PortScanner()
                nm.scan(hosts=host, ports=ports, arguments=nmap_args, timeout=timeout)

                if host in nm.all_hosts():
                    for proto in nm[host].all_protocols():
                        for port in nm[host][proto]:
                            port_info = nm[host][proto][port]
                            if port_info["state"] == "open":
                                services.append({
                                    "host": host,
                                    "port": port,
                                    "protocol": proto,
                                    "state": port_info["state"],
                                    "service": port_info.get("name", "unknown"),
                                    "version": port_info.get("version", ""),
                                    "product": port_info.get("product", ""),
                                    "extrainfo": port_info.get("extrainfo", ""),
                                    "cpe": port_info.get("cpe", ""),
                                })

                stats["elapsed"] = float(nm.scanstats().get("elapsed", 0))

            except Exception as e:
                logger.error(f"[port_scan] nmap error on {host}: {e}")
                services, stats = self._scan_subprocess(host, ports, nmap_args, timeout)
        else:
            services, stats = self._scan_subprocess(host, ports, nmap_args, timeout)

        return services, stats

    def _scan_subprocess(
        self,
        host: str,
        ports: str,
        nmap_args: str,
        timeout: int
    ) -> tuple[List[Dict], Dict]:
        """Fallback to subprocess nmap."""
        services = []
        stats = {"elapsed": 0}

        if not shutil.which("nmap"):
            logger.error("[port_scan] nmap not found")
            return services, stats

        cmd = f"nmap {nmap_args} -p {ports} {host}"

        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse output
            current_host = host
            for line in result.stdout.split("\n"):
                # Match port line: "22/tcp   open  ssh     OpenSSH 8.9p1"
                match = re.match(
                    r"(\d+)/(tcp|udp)\s+(\w+)\s+(\S+)\s*(.*)?",
                    line.strip()
                )
                if match:
                    port = int(match.group(1))
                    proto = match.group(2)
                    state = match.group(3)
                    service = match.group(4)
                    version = match.group(5) or ""

                    if state == "open":
                        services.append({
                            "host": current_host,
                            "port": port,
                            "protocol": proto,
                            "state": state,
                            "service": service,
                            "version": version.strip(),
                            "product": "",
                            "extrainfo": "",
                            "cpe": "",
                        })

        except subprocess.TimeoutExpired:
            logger.error(f"[port_scan] nmap timeout on {host}")
        except Exception as e:
            logger.error(f"[port_scan] subprocess error: {e}")

        return services, stats

    def _service_risk_score(self, service: Dict) -> int:
        """Calculate risk score for service sorting."""
        score = 0
        svc_name = service.get("service", "").lower()
        port = service.get("port", 0)

        # High-risk services
        high_risk = ["telnet", "ftp", "rsh", "rlogin", "vnc", "rdp", "smb", "netbios"]
        medium_risk = ["ssh", "mysql", "postgres", "mssql", "oracle", "redis", "mongodb"]

        if svc_name in high_risk:
            score += 100
        elif svc_name in medium_risk:
            score += 50

        # High-risk ports
        if port in [21, 23, 135, 139, 445, 3389, 5900]:
            score += 50

        # Unencrypted versions of services
        if port == 80:  # HTTP vs HTTPS
            score += 20
        if port == 21:  # FTP
            score += 30

        return score

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        services = data.get("services", [])
        stats = data.get("statistics", {})
        by_host = data.get("by_host", {})

        # Count by service type
        service_counts = {}
        for svc in services:
            name = svc.get("service", "unknown")
            service_counts[name] = service_counts.get(name, 0) + 1

        # Top services
        top_services = sorted(service_counts.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{name}({count})" for name, count in top_services)

        return (
            f"Port Scan: {stats.get('hosts_scanned', 0)} hosts, "
            f"{stats.get('total_open_ports', 0)} open ports. "
            f"Hosts with services: {stats.get('hosts_with_open_ports', 0)}. "
            f"Top services: {top_str}. "
            f"Scan time: {stats.get('scan_time_sec', 0):.1f}s."
        )
