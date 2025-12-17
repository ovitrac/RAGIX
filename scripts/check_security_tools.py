#!/usr/bin/env python3
"""
RAGIX Security Tools Checker

Checks for required security tools and provides installation recommendations.
Supports multiple package managers: apt, dnf, yum, pacman, brew.

Usage:
    python scripts/check_security_tools.py
    python scripts/check_security_tools.py --install-python
    python scripts/check_security_tools.py --json

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-16
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ToolCategory(Enum):
    """Tool categories for security auditing."""
    DISCOVERY = "Network Discovery"
    PORT_SCAN = "Port Scanning"
    DNS = "DNS Analysis"
    SSL_TLS = "SSL/TLS Analysis"
    VULN_SCAN = "Vulnerability Scanning"
    WEB = "Web Tools"
    PACKET = "Packet Analysis"
    PYTHON = "Python Packages"


class InstallMethod(Enum):
    """Installation methods."""
    APT = "apt"
    DNF = "dnf"
    YUM = "yum"
    PACMAN = "pacman"
    BREW = "brew"
    PIP = "pip"
    GO = "go"
    BINARY = "binary"
    MANUAL = "manual"


@dataclass
class Tool:
    """Security tool definition."""
    name: str
    category: ToolCategory
    command: str  # Command to check availability
    description: str
    required: bool = True  # True = essential, False = optional
    version_cmd: Optional[str] = None  # Command to get version
    install_commands: Dict[str, str] = field(default_factory=dict)
    binary_url: Optional[str] = None
    notes: str = ""


# Tool definitions with multi-platform install commands
TOOLS = [
    # === DISCOVERY ===
    Tool(
        name="nmap",
        category=ToolCategory.DISCOVERY,
        command="nmap",
        description="Network mapper and port scanner",
        required=True,
        version_cmd="nmap --version | head -1",
        install_commands={
            "apt": "sudo apt install -y nmap",
            "dnf": "sudo dnf install -y nmap",
            "yum": "sudo yum install -y nmap",
            "pacman": "sudo pacman -S nmap",
            "brew": "brew install nmap",
        },
    ),
    Tool(
        name="masscan",
        category=ToolCategory.DISCOVERY,
        command="masscan",
        description="Fast port scanner (optional, nmap suffices)",
        required=False,
        version_cmd="masscan --version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y masscan",
            "dnf": "sudo dnf install -y masscan",
            "yum": "sudo yum install -y masscan",
            "pacman": "sudo pacman -S masscan",
            "brew": "brew install masscan",
        },
    ),
    Tool(
        name="arp-scan",
        category=ToolCategory.DISCOVERY,
        command="arp-scan",
        description="ARP scanner for LAN discovery",
        required=False,
        version_cmd="arp-scan --version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y arp-scan",
            "dnf": "sudo dnf install -y arp-scan",
            "yum": "sudo yum install -y arp-scan",
            "pacman": "sudo pacman -S arp-scan",
            "brew": "brew install arp-scan",
        },
    ),

    # === DNS ===
    Tool(
        name="dig",
        category=ToolCategory.DNS,
        command="dig",
        description="DNS lookup utility",
        required=True,
        version_cmd="dig -v 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y dnsutils",
            "dnf": "sudo dnf install -y bind-utils",
            "yum": "sudo yum install -y bind-utils",
            "pacman": "sudo pacman -S bind",
            "brew": "brew install bind",
        },
    ),
    Tool(
        name="dnsrecon",
        category=ToolCategory.DNS,
        command="dnsrecon",
        description="DNS enumeration tool",
        required=False,
        version_cmd="dnsrecon --version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y dnsrecon",
            "dnf": "sudo dnf install -y dnsrecon",
            "pip": "pip install dnsrecon",
        },
        notes="May require pip install on some systems",
    ),

    # === SSL/TLS ===
    Tool(
        name="openssl",
        category=ToolCategory.SSL_TLS,
        command="openssl",
        description="SSL/TLS toolkit",
        required=True,
        version_cmd="openssl version",
        install_commands={
            "apt": "sudo apt install -y openssl",
            "dnf": "sudo dnf install -y openssl",
            "yum": "sudo yum install -y openssl",
            "pacman": "sudo pacman -S openssl",
            "brew": "brew install openssl",
        },
    ),
    Tool(
        name="testssl",
        category=ToolCategory.SSL_TLS,
        command="testssl",  # apt installs as 'testssl', GitHub as 'testssl.sh'
        description="TLS/SSL testing tool",
        required=True,
        version_cmd="testssl --version 2>&1 | head -1 || testssl.sh --version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y testssl.sh",
            "dnf": "sudo dnf install -y testssl",
            "manual": "git clone https://github.com/drwetter/testssl.sh.git",
        },
        notes="apt installs as 'testssl', GitHub clone as 'testssl.sh'",
    ),

    # === VULNERABILITY SCANNING ===
    Tool(
        name="nuclei",
        category=ToolCategory.VULN_SCAN,
        command="nuclei",
        description="Template-based vulnerability scanner",
        required=True,
        version_cmd="nuclei --version 2>&1 | grep -oE 'v[0-9]+\\.[0-9]+\\.[0-9]+'",
        install_commands={
            "apt": "sudo apt install -y nuclei",  # Ubuntu 24.04+
            "go": "go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest",
            "binary": "Download from https://github.com/projectdiscovery/nuclei/releases",
        },
        binary_url="https://github.com/projectdiscovery/nuclei/releases",
        notes="Requires Go 1.21+ for go install, or download binary",
    ),
    Tool(
        name="nikto",
        category=ToolCategory.VULN_SCAN,
        command="nikto",
        description="Web server vulnerability scanner",
        required=False,
        version_cmd="nikto -Version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y nikto",
            "dnf": "sudo dnf install -y nikto",
            "yum": "sudo yum install -y nikto",
            "pacman": "sudo pacman -S nikto",
            "brew": "brew install nikto",
        },
    ),

    # === WEB TOOLS ===
    Tool(
        name="curl",
        category=ToolCategory.WEB,
        command="curl",
        description="HTTP client",
        required=True,
        version_cmd="curl --version | head -1",
        install_commands={
            "apt": "sudo apt install -y curl",
            "dnf": "sudo dnf install -y curl",
            "yum": "sudo yum install -y curl",
            "pacman": "sudo pacman -S curl",
            "brew": "brew install curl",
        },
    ),
    Tool(
        name="httpx",
        category=ToolCategory.WEB,
        command="httpx",
        description="HTTP probing tool",
        required=False,
        version_cmd="httpx --version 2>&1",
        install_commands={
            "go": "go install github.com/projectdiscovery/httpx/cmd/httpx@latest",
            "binary": "Download from https://github.com/projectdiscovery/httpx/releases",
        },
        binary_url="https://github.com/projectdiscovery/httpx/releases",
        notes="Optional, useful for HTTP probing at scale",
    ),

    # === PACKET ANALYSIS ===
    Tool(
        name="tcpdump",
        category=ToolCategory.PACKET,
        command="tcpdump",
        description="Packet analyzer",
        required=False,
        version_cmd="tcpdump --version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y tcpdump",
            "dnf": "sudo dnf install -y tcpdump",
            "yum": "sudo yum install -y tcpdump",
            "pacman": "sudo pacman -S tcpdump",
            "brew": "brew install tcpdump",
        },
    ),
    Tool(
        name="tshark",
        category=ToolCategory.PACKET,
        command="tshark",
        description="Terminal Wireshark",
        required=False,
        version_cmd="tshark --version 2>&1 | head -1",
        install_commands={
            "apt": "sudo apt install -y tshark",
            "dnf": "sudo dnf install -y wireshark-cli",
            "yum": "sudo yum install -y wireshark",
            "pacman": "sudo pacman -S wireshark-cli",
            "brew": "brew install wireshark",
        },
    ),
]

# Python packages for security kernels
PYTHON_PACKAGES = [
    {
        "name": "python-nmap",
        "import_name": "nmap",
        "description": "Python interface to nmap",
        "required": True,
    },
    {
        "name": "dnspython",
        "import_name": "dns",
        "description": "DNS toolkit for Python",
        "required": True,
    },
    {
        "name": "sslyze",
        "import_name": "sslyze",
        "description": "SSL/TLS scanner library",
        "required": True,
    },
    {
        "name": "cryptography",
        "import_name": "cryptography",
        "description": "Cryptographic recipes",
        "required": True,
    },
]


def detect_package_manager() -> Optional[str]:
    """Detect the system package manager."""
    managers = [
        ("apt", "apt --version"),
        ("dnf", "dnf --version"),
        ("yum", "yum --version"),
        ("pacman", "pacman --version"),
        ("brew", "brew --version"),
    ]

    for name, cmd in managers:
        if shutil.which(name):
            return name

    return None


def check_tool(tool: Tool) -> Tuple[bool, Optional[str]]:
    """Check if a tool is installed and get its version."""
    if not shutil.which(tool.command):
        return False, None

    if tool.version_cmd:
        try:
            result = subprocess.run(
                tool.version_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip() or result.stderr.strip()
            return True, version[:60] if version else "installed"
        except Exception:
            return True, "installed"

    return True, "installed"


def check_python_package(pkg: Dict) -> Tuple[bool, Optional[str]]:
    """Check if a Python package is installed."""
    try:
        module = __import__(pkg["import_name"])
        version = getattr(module, "__version__", None)
        # Handle case where __version__ is a module (e.g., sslyze)
        if version is None or not isinstance(version, str):
            # Try to get version from pkg_resources or importlib.metadata
            try:
                from importlib.metadata import version as get_version
                version = get_version(pkg["name"])
            except Exception:
                version = "installed"
        return True, str(version)
    except ImportError:
        return False, None


def get_install_command(tool: Tool, pkg_manager: str) -> Optional[str]:
    """Get installation command for a tool."""
    if pkg_manager in tool.install_commands:
        return tool.install_commands[pkg_manager]

    # Fallbacks
    if "binary" in tool.install_commands:
        return tool.install_commands["binary"]
    if "manual" in tool.install_commands:
        return tool.install_commands["manual"]

    return None


def print_status(name: str, installed: bool, version: str = None, required: bool = True):
    """Print tool status with colors."""
    status_icon = "✓" if installed else ("✗" if required else "○")
    color = "\033[32m" if installed else ("\033[31m" if required else "\033[33m")
    reset = "\033[0m"

    version_str = f" ({version})" if version and installed else ""
    req_str = "" if required else " [optional]"

    print(f"  {color}{status_icon}{reset} {name}{version_str}{req_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Check RAGIX security tools availability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_security_tools.py           # Check all tools
  python scripts/check_security_tools.py --json    # JSON output
  python scripts/check_security_tools.py --install-python  # Install Python packages
        """,
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--install-python", action="store_true", help="Install missing Python packages")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Detect package manager
    pkg_manager = detect_package_manager()

    # Check all tools
    results = {
        "system": {
            "platform": platform.system(),
            "release": platform.release(),
            "package_manager": pkg_manager,
            "python": platform.python_version(),
        },
        "tools": {},
        "python_packages": {},
        "missing_required": [],
        "missing_optional": [],
        "install_commands": [],
    }

    # Group tools by category
    by_category: Dict[ToolCategory, List[Tuple[Tool, bool, str]]] = {}

    for tool in TOOLS:
        installed, version = check_tool(tool)
        results["tools"][tool.name] = {
            "installed": installed,
            "version": version,
            "required": tool.required,
            "category": tool.category.value,
        }

        if tool.category not in by_category:
            by_category[tool.category] = []
        by_category[tool.category].append((tool, installed, version))

        if not installed:
            if tool.required:
                results["missing_required"].append(tool.name)
            else:
                results["missing_optional"].append(tool.name)

            cmd = get_install_command(tool, pkg_manager)
            if cmd:
                results["install_commands"].append({
                    "tool": tool.name,
                    "command": cmd,
                    "required": tool.required,
                })

    # Check Python packages
    for pkg in PYTHON_PACKAGES:
        installed, version = check_python_package(pkg)
        results["python_packages"][pkg["name"]] = {
            "installed": installed,
            "version": version,
            "required": pkg["required"],
        }

        if not installed and pkg["required"]:
            results["missing_required"].append(f"pip:{pkg['name']}")
            results["install_commands"].append({
                "tool": pkg["name"],
                "command": f"pip install {pkg['name']}",
                "required": pkg["required"],
            })

    # JSON output
    if args.json:
        print(json.dumps(results, indent=2))
        return 0 if not results["missing_required"] else 1

    # Human-readable output
    print("=" * 60)
    print("RAGIX Security Tools Check")
    print("=" * 60)
    print()
    print(f"Platform: {results['system']['platform']} {results['system']['release']}")
    print(f"Package Manager: {pkg_manager or 'unknown'}")
    print(f"Python: {results['system']['python']}")
    print()

    # Print by category
    for category in ToolCategory:
        if category == ToolCategory.PYTHON:
            continue

        if category in by_category:
            print(f"\n[{category.value}]")
            for tool, installed, version in by_category[category]:
                print_status(tool.name, installed, version, tool.required)

    # Python packages
    print(f"\n[{ToolCategory.PYTHON.value}]")
    for pkg in PYTHON_PACKAGES:
        installed, version = check_python_package(pkg)
        print_status(pkg["name"], installed, version, pkg["required"])

    # Summary
    print()
    print("=" * 60)

    missing_req = len(results["missing_required"])
    missing_opt = len(results["missing_optional"])

    if missing_req == 0:
        print("\033[32m✓ All required tools are installed!\033[0m")
    else:
        print(f"\033[31m✗ Missing {missing_req} required tool(s)\033[0m")

    if missing_opt > 0:
        print(f"\033[33m○ {missing_opt} optional tool(s) not installed\033[0m")

    # Installation recommendations
    if results["install_commands"]:
        print()
        print("Installation Commands:")
        print("-" * 40)

        # Group by type
        system_cmds = [c for c in results["install_commands"] if not c["command"].startswith("pip")]
        pip_cmds = [c for c in results["install_commands"] if c["command"].startswith("pip")]

        if system_cmds:
            required_cmds = [c["command"] for c in system_cmds if c["required"]]
            optional_cmds = [c["command"] for c in system_cmds if not c["required"]]

            if required_cmds:
                print("\n# Required (system packages):")
                for cmd in required_cmds:
                    print(f"  {cmd}")

            if optional_cmds and args.verbose:
                print("\n# Optional (system packages):")
                for cmd in optional_cmds:
                    print(f"  {cmd}")

        if pip_cmds:
            print("\n# Python packages:")
            pkg_names = [c["tool"] for c in pip_cmds]
            print(f"  pip install {' '.join(pkg_names)}")

    # Install Python packages if requested
    if args.install_python:
        missing_pip = [
            pkg["name"] for pkg in PYTHON_PACKAGES
            if not check_python_package(pkg)[0]
        ]
        if missing_pip:
            print()
            print(f"Installing Python packages: {', '.join(missing_pip)}")
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_pip)
        else:
            print("\nAll Python packages already installed.")

    print()

    # Nuclei templates check
    nuclei_templates = Path.home() / "nuclei-templates"
    if nuclei_templates.exists():
        template_count = len(list(nuclei_templates.rglob("*.yaml")))
        print(f"Nuclei Templates: {nuclei_templates} ({template_count} templates)")
    else:
        print("Nuclei Templates: Not found (run 'nuclei -ut' to download)")

    return 0 if not results["missing_required"] else 1


if __name__ == "__main__":
    sys.exit(main())
