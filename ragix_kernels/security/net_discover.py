"""
Kernel: Network Discovery
Stage: 1 (Discovery)
Category: security

Discovers hosts on target networks using nmap ping sweep and ARP scanning.
Provides the foundation for subsequent port scanning and vulnerability assessment.

Wraps:
- nmap -sn (ping sweep)
- nmap -sL (list scan for DNS resolution)
- arp-scan (local LAN discovery, if available)

Input (manifest or data/targets.yaml):
    targets:
      - "192.168.1.0/24"
      - "10.0.0.1-10"
    exclude:
      - "192.168.1.1"
    methods: ["ping", "arp"]

Output:
    hosts: List of discovered hosts with IP, MAC, hostname
    statistics: Scan statistics (hosts up, scan time)

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 1 -k net_discover

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
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

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class NetDiscoverKernel(Kernel):
    """
    Network discovery kernel using nmap and arp-scan.

    Configuration options:
        targets: List of target networks/hosts (CIDR, ranges, single IPs)
        exclude: List of hosts to exclude from scanning
        methods: Discovery methods ["ping", "arp", "list"]
        timeout: Scan timeout in seconds (default: 300)

    Example manifest:
        net_discover:
          enabled: true
          options:
            targets:
              - "192.168.1.0/24"
            methods: ["ping"]
            timeout: 120
    """

    name = "net_discover"
    version = "1.0.0"
    category = "security"
    stage = 1
    description = "Network host discovery"

    requires = []
    provides = ["hosts", "network_map"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Discover hosts on target networks."""

        # Load targets from config or file
        targets_config = self._load_targets(input)
        targets = targets_config.get("targets", [])
        exclude = targets_config.get("exclude", [])
        methods = targets_config.get("methods", ["ping"])
        timeout = input.config.get("timeout", 300)

        if not targets:
            logger.warning("[net_discover] No targets specified")
            return {
                "hosts": [],
                "statistics": {"error": "No targets specified"},
            }

        logger.info(f"[net_discover] Scanning {len(targets)} target(s): {targets}")

        all_hosts = []
        scan_stats = {
            "targets_scanned": len(targets),
            "methods_used": methods,
            "hosts_up": 0,
            "hosts_down": 0,
            "scan_time_sec": 0,
        }

        # Run discovery for each target
        for target in targets:
            if "ping" in methods:
                hosts, stats = self._nmap_ping_sweep(target, exclude, timeout)
                all_hosts.extend(hosts)
                scan_stats["scan_time_sec"] += stats.get("elapsed", 0)

            if "arp" in methods and self._has_arp_scan():
                arp_hosts = self._arp_scan(target)
                # Merge with existing hosts (avoid duplicates)
                for h in arp_hosts:
                    if not any(existing["ip"] == h["ip"] for existing in all_hosts):
                        all_hosts.append(h)

            if "list" in methods:
                list_hosts = self._nmap_list_scan(target)
                for h in list_hosts:
                    if not any(existing["ip"] == h["ip"] for existing in all_hosts):
                        all_hosts.append(h)

        # Deduplicate and sort
        seen_ips = set()
        unique_hosts = []
        for host in all_hosts:
            if host["ip"] not in seen_ips:
                seen_ips.add(host["ip"])
                unique_hosts.append(host)

        unique_hosts.sort(key=lambda x: self._ip_sort_key(x["ip"]))

        scan_stats["hosts_up"] = len(unique_hosts)

        return {
            "hosts": unique_hosts,
            "targets": targets,
            "exclude": exclude,
            "statistics": scan_stats,
        }

    def _load_targets(self, input: KernelInput) -> Dict[str, Any]:
        """Load targets from config or file."""

        # Check for targets file
        for filename in ["targets.yaml", "targets.yml", "targets.json"]:
            filepath = input.workspace / "data" / filename
            if filepath.exists():
                content = filepath.read_text()
                if filename.endswith(".json"):
                    return json.loads(content)
                elif YAML_AVAILABLE:
                    return yaml.safe_load(content) or {}

        # Use inline config
        return {
            "targets": input.config.get("targets", []),
            "exclude": input.config.get("exclude", []),
            "methods": input.config.get("methods", ["ping"]),
        }

    def _nmap_ping_sweep(
        self,
        target: str,
        exclude: List[str],
        timeout: int
    ) -> tuple[List[Dict], Dict]:
        """Run nmap ping sweep (-sn)."""
        hosts = []
        stats = {"elapsed": 0}

        if NMAP_AVAILABLE:
            try:
                nm = nmap.PortScanner()
                exclude_str = ",".join(exclude) if exclude else None

                logger.info(f"[net_discover] nmap -sn {target}")
                nm.scan(hosts=target, arguments=f"-sn --exclude {exclude_str}" if exclude_str else "-sn")

                for host in nm.all_hosts():
                    host_info = {
                        "ip": host,
                        "status": nm[host].state(),
                        "hostname": nm[host].hostname() or None,
                        "mac": None,
                        "vendor": None,
                        "discovery_method": "nmap_ping",
                    }

                    # Get MAC if available
                    if "mac" in nm[host]["addresses"]:
                        host_info["mac"] = nm[host]["addresses"]["mac"]
                    if "vendor" in nm[host]:
                        vendors = list(nm[host]["vendor"].values())
                        host_info["vendor"] = vendors[0] if vendors else None

                    hosts.append(host_info)

                stats["elapsed"] = float(nm.scanstats().get("elapsed", 0))

            except Exception as e:
                logger.error(f"[net_discover] nmap error: {e}")
                # Fallback to subprocess
                hosts, stats = self._nmap_subprocess(target, exclude, "-sn")

        else:
            hosts, stats = self._nmap_subprocess(target, exclude, "-sn")

        return hosts, stats

    def _nmap_subprocess(
        self,
        target: str,
        exclude: List[str],
        arguments: str
    ) -> tuple[List[Dict], Dict]:
        """Run nmap via subprocess (fallback)."""
        hosts = []
        stats = {"elapsed": 0}

        if not shutil.which("nmap"):
            logger.error("[net_discover] nmap not found")
            return hosts, stats

        cmd = ["nmap", arguments, target]
        if exclude:
            cmd.extend(["--exclude", ",".join(exclude)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Parse output
            for line in result.stdout.split("\n"):
                # Match "Nmap scan report for hostname (ip)" or "Nmap scan report for ip"
                match = re.search(r"Nmap scan report for (?:(\S+) \()?(\d+\.\d+\.\d+\.\d+)\)?", line)
                if match:
                    hostname = match.group(1)
                    ip = match.group(2)
                    hosts.append({
                        "ip": ip,
                        "hostname": hostname,
                        "status": "up",
                        "mac": None,
                        "vendor": None,
                        "discovery_method": "nmap_subprocess",
                    })

                # Match MAC address
                mac_match = re.search(r"MAC Address: ([0-9A-F:]+)(?: \((.+)\))?", line, re.I)
                if mac_match and hosts:
                    hosts[-1]["mac"] = mac_match.group(1)
                    hosts[-1]["vendor"] = mac_match.group(2)

        except subprocess.TimeoutExpired:
            logger.error("[net_discover] nmap timeout")
        except Exception as e:
            logger.error(f"[net_discover] nmap subprocess error: {e}")

        return hosts, stats

    def _nmap_list_scan(self, target: str) -> List[Dict]:
        """Run nmap list scan (-sL) for DNS resolution without probing."""
        hosts = []

        if not shutil.which("nmap"):
            return hosts

        try:
            result = subprocess.run(
                ["nmap", "-sL", target],
                capture_output=True,
                text=True,
                timeout=60,
            )

            for line in result.stdout.split("\n"):
                match = re.search(r"Nmap scan report for (?:(\S+) \()?(\d+\.\d+\.\d+\.\d+)\)?", line)
                if match:
                    hostname = match.group(1)
                    ip = match.group(2)
                    hosts.append({
                        "ip": ip,
                        "hostname": hostname,
                        "status": "unknown",
                        "mac": None,
                        "vendor": None,
                        "discovery_method": "nmap_list",
                    })

        except Exception as e:
            logger.error(f"[net_discover] nmap list scan error: {e}")

        return hosts

    def _has_arp_scan(self) -> bool:
        """Check if arp-scan is available."""
        return shutil.which("arp-scan") is not None

    def _arp_scan(self, target: str) -> List[Dict]:
        """Run arp-scan for local network discovery."""
        hosts = []

        if not self._has_arp_scan():
            return hosts

        try:
            # arp-scan requires root for raw sockets
            result = subprocess.run(
                ["arp-scan", "--localnet"] if "/" in target else ["arp-scan", target],
                capture_output=True,
                text=True,
                timeout=60,
            )

            for line in result.stdout.split("\n"):
                # Match "192.168.1.1    aa:bb:cc:dd:ee:ff    Vendor Name"
                match = re.match(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f:]+)\s+(.+)?", line, re.I)
                if match:
                    hosts.append({
                        "ip": match.group(1),
                        "mac": match.group(2),
                        "vendor": match.group(3).strip() if match.group(3) else None,
                        "hostname": None,
                        "status": "up",
                        "discovery_method": "arp_scan",
                    })

        except subprocess.TimeoutExpired:
            logger.warning("[net_discover] arp-scan timeout")
        except Exception as e:
            logger.warning(f"[net_discover] arp-scan error: {e}")

        return hosts

    def _ip_sort_key(self, ip: str) -> tuple:
        """Convert IP to sortable tuple."""
        try:
            return tuple(int(x) for x in ip.split("."))
        except (ValueError, AttributeError):
            return (0, 0, 0, 0)

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        hosts = data.get("hosts", [])
        stats = data.get("statistics", {})
        targets = data.get("targets", [])

        hosts_with_hostname = len([h for h in hosts if h.get("hostname")])
        hosts_with_mac = len([h for h in hosts if h.get("mac")])

        return (
            f"Discovery: {len(hosts)} hosts found on {len(targets)} target(s). "
            f"Methods: {', '.join(stats.get('methods_used', []))}. "
            f"Hostnames resolved: {hosts_with_hostname}. "
            f"MACs captured: {hosts_with_mac}. "
            f"Scan time: {stats.get('scan_time_sec', 0):.1f}s."
        )
