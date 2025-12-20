"""
KOAS MCP Helpers
================

Helper utilities for simplified KOAS tool interfaces.
Designed for reliable usage by local LLMs (Ollama models).

Features:
- Auto-workspace creation in /tmp/koas_{uuid}
- Output simplification for LLM consumption
- Summary generation (<300 chars)
- Structured error handling

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-19
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# AUTO-WORKSPACE MANAGEMENT
# =============================================================================

def create_auto_workspace(
    prefix: str = "koas",
    category: str = "security",
) -> Path:
    """
    Create a temporary KOAS workspace.

    Parameters
    ----------
    prefix : str
        Workspace directory prefix.
    category : str
        Category for the workspace (security, audit).

    Returns
    -------
    Path
        Path to the created workspace directory.
    """
    workspace_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"{prefix}_{category}_{timestamp}_{workspace_id}"

    workspace = Path(tempfile.gettempdir()) / workspace_name
    workspace.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (workspace / "data").mkdir(exist_ok=True)
    (workspace / "stage1").mkdir(exist_ok=True)
    (workspace / "stage2").mkdir(exist_ok=True)
    (workspace / "stage3").mkdir(exist_ok=True)

    logger.info(f"Created auto-workspace: {workspace}")
    return workspace


def get_or_create_workspace(workspace: Optional[str] = None, category: str = "security") -> Path:
    """
    Get existing workspace or create a new one.

    Parameters
    ----------
    workspace : str, optional
        Path to existing workspace. If None, creates a new one.
    category : str
        Category for new workspace (security, audit).

    Returns
    -------
    Path
        Workspace path.
    """
    # Invalid workspace values that should trigger auto-creation
    INVALID_WORKSPACE_VALUES = {"", "auto-created", "auto", "none", "null", "undefined"}

    if workspace and workspace.lower() not in INVALID_WORKSPACE_VALUES:
        ws_path = Path(workspace)
        # Only use if it looks like a valid path (absolute or starts with /tmp)
        if ws_path.is_absolute() or str(ws_path).startswith("/tmp") or ws_path.exists():
            if ws_path.exists():
                return ws_path
            # Create if doesn't exist
            ws_path.mkdir(parents=True, exist_ok=True)
            return ws_path

    return create_auto_workspace(category=category)


# =============================================================================
# OUTPUT SIMPLIFICATION FOR LLMs
# =============================================================================

def simplify_output(
    result: Dict[str, Any],
    max_summary_chars: int = 300,
    max_items: int = 10,
    include_details_file: bool = True,
) -> Dict[str, Any]:
    """
    Simplify kernel output for LLM consumption.

    Transforms complex kernel outputs into LLM-friendly format:
    - Short summary (<300 chars)
    - Top N items only
    - Full details saved to file

    Parameters
    ----------
    result : dict
        Raw kernel output.
    max_summary_chars : int
        Maximum length for summary.
    max_items : int
        Maximum number of items in lists.
    include_details_file : bool
        Save full results to JSON file.

    Returns
    -------
    dict
        Simplified output with summary.
    """
    simplified = {
        "summary": "",
        "status": result.get("status", "unknown"),
        "key_findings": [],
        "action_items": [],
    }

    # Extract counts
    counts = {}
    for key in ["hosts", "ports", "services", "vulnerabilities", "findings", "issues"]:
        if key in result and isinstance(result[key], list):
            counts[key] = len(result[key])
            # Keep only top N items
            simplified[key] = result[key][:max_items]
            if len(result[key]) > max_items:
                simplified[f"{key}_truncated"] = True
                simplified[f"{key}_total"] = len(result[key])

    # Extract scores
    for key in ["risk_score", "compliance_score", "security_score"]:
        if key in result:
            simplified[key] = result[key]

    # Copy errors
    if "error" in result:
        simplified["error"] = result["error"]
        simplified["status"] = "error"

    # Generate summary
    summary_parts = []

    if "error" in result:
        summary_parts.append(f"Error: {result['error'][:100]}")
    else:
        if counts:
            count_str = ", ".join(f"{v} {k}" for k, v in counts.items())
            summary_parts.append(f"Found: {count_str}.")

        if "risk_score" in result:
            summary_parts.append(f"Risk score: {result['risk_score']}/10.")

        if "compliance_score" in result:
            summary_parts.append(f"Compliance: {result['compliance_score']}%.")

    simplified["summary"] = " ".join(summary_parts)[:max_summary_chars]

    return simplified


def generate_summary(
    result: Dict[str, Any],
    context: str = "",
    max_chars: int = 300,
) -> str:
    """
    Generate a concise summary from kernel results.

    Parameters
    ----------
    result : dict
        Kernel output.
    context : str
        Additional context for the summary.
    max_chars : int
        Maximum summary length.

    Returns
    -------
    str
        Concise summary.
    """
    parts = []

    if context:
        parts.append(context)

    # Error case
    if "error" in result:
        return f"Error: {result['error']}"[:max_chars]

    # Count items
    counts = []
    for key in ["hosts", "ports", "services", "vulnerabilities", "findings", "issues"]:
        if key in result and isinstance(result[key], list):
            count = len(result[key])
            if count > 0:
                counts.append(f"{count} {key}")

    if counts:
        parts.append(f"Found {', '.join(counts)}.")

    # Scores
    if "risk_score" in result:
        score = result["risk_score"]
        level = "CRITICAL" if score >= 8 else "HIGH" if score >= 6 else "MEDIUM" if score >= 4 else "LOW"
        parts.append(f"Risk: {level} ({score}/10).")

    if "compliance_score" in result:
        score = result["compliance_score"]
        parts.append(f"Compliance: {score}%.")

    # Critical findings
    if "critical" in result and result["critical"]:
        parts.append(f"Critical issues: {len(result['critical'])}!")

    return " ".join(parts)[:max_chars]


def extract_action_items(result: Dict[str, Any], max_items: int = 5) -> List[Dict[str, str]]:
    """
    Extract actionable items from kernel results.

    Parameters
    ----------
    result : dict
        Kernel output.
    max_items : int
        Maximum number of action items.

    Returns
    -------
    list
        List of action items with priority and description.
    """
    actions = []

    # From recommendations
    if "recommendations" in result:
        for rec in result["recommendations"][:max_items]:
            if isinstance(rec, dict):
                actions.append({
                    "priority": rec.get("priority", "medium"),
                    "action": rec.get("recommendation", rec.get("action", str(rec))),
                })
            else:
                actions.append({"priority": "medium", "action": str(rec)})

    # From critical findings
    if "critical" in result:
        for finding in result["critical"][:max_items]:
            if isinstance(finding, dict):
                actions.append({
                    "priority": "critical",
                    "action": f"Fix: {finding.get('title', finding.get('id', 'unknown'))}",
                })

    # From high-severity vulnerabilities
    if "vulnerabilities" in result:
        high_vulns = [v for v in result["vulnerabilities"]
                      if v.get("severity", "").lower() in ("critical", "high")]
        for vuln in high_vulns[:max_items]:
            actions.append({
                "priority": "high",
                "action": f"Patch: {vuln.get('cve', vuln.get('id', 'vulnerability'))}",
            })

    return actions[:max_items]


# =============================================================================
# RESULT PERSISTENCE
# =============================================================================

def save_results(
    workspace: Path,
    stage: int,
    kernel_name: str,
    results: Dict[str, Any],
) -> Path:
    """
    Save kernel results to workspace.

    Parameters
    ----------
    workspace : Path
        Workspace directory.
    stage : int
        Stage number (1, 2, 3).
    kernel_name : str
        Name of the kernel.
    results : dict
        Results to save.

    Returns
    -------
    Path
        Path to saved results file.
    """
    stage_dir = workspace / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    output_file = stage_dir / f"{kernel_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved results to {output_file}")
    return output_file


def load_stage_results(workspace: Path, stage: int) -> Dict[str, Dict[str, Any]]:
    """
    Load all results from a stage.

    Parameters
    ----------
    workspace : Path
        Workspace directory.
    stage : int
        Stage number.

    Returns
    -------
    dict
        Dictionary of kernel_name -> results.
    """
    results = {}
    stage_dir = workspace / f"stage{stage}"

    if stage_dir.exists():
        for json_file in stage_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    results[json_file.stem] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    return results


def load_stage_dependency_paths(workspace: Path, stage: int) -> Dict[str, Path]:
    """
    Get paths to stage result files for KernelInput.dependencies.

    Parameters
    ----------
    workspace : Path
        Workspace directory.
    stage : int
        Stage number.

    Returns
    -------
    dict
        Dictionary of kernel_name -> Path to result file.
    """
    paths = {}
    stage_dir = workspace / f"stage{stage}"

    if stage_dir.exists():
        for json_file in stage_dir.glob("*.json"):
            paths[json_file.stem] = json_file

    return paths


# =============================================================================
# TARGET RESOLUTION
# =============================================================================

def resolve_target(
    target: str,
    workspace: Optional[Path] = None,
) -> List[str]:
    """
    Resolve target specification to list of targets.

    Handles special keywords:
    - "discovered": Use hosts from previous net_discover
    - "all": Use all known targets

    Parameters
    ----------
    target : str
        Target specification (IP, CIDR, hostname, or keyword).
    workspace : Path, optional
        Workspace to look for discovered hosts.

    Returns
    -------
    list
        List of resolved targets.
    """
    # Handle special keywords
    if target.lower() == "discovered" and workspace:
        stage1_results = load_stage_results(workspace, 1)
        if "net_discover" in stage1_results:
            hosts = stage1_results["net_discover"].get("hosts", [])
            return [h.get("ip", h) if isinstance(h, dict) else h for h in hosts]
        return []

    if target.lower() == "all" and workspace:
        # Collect all known targets
        targets = set()
        for stage in [1, 2]:
            results = load_stage_results(workspace, stage)
            for kernel_results in results.values():
                if "hosts" in kernel_results:
                    for h in kernel_results["hosts"]:
                        ip = h.get("ip", h) if isinstance(h, dict) else h
                        if ip:
                            targets.add(ip)
        return list(targets) if targets else [target]

    # Single target or CIDR
    return [target]


# =============================================================================
# PORT PRESETS
# =============================================================================

PORT_PRESETS = {
    "common": "21,22,23,25,53,80,110,111,135,139,143,443,445,993,995,1723,3306,3389,5432,5900,8080,8443",
    "web": "80,443,8000,8080,8443,8888,9000,9090",
    "database": "1433,1521,3306,5432,5984,6379,27017,28017",
    "mail": "25,110,143,465,587,993,995",
    "admin": "22,23,3389,5900,5901",
    "top100": ",".join(map(str, [
        7, 9, 13, 21, 22, 23, 25, 26, 37, 53, 79, 80, 81, 88, 106, 110, 111, 113, 119, 135,
        139, 143, 144, 179, 199, 389, 427, 443, 444, 445, 465, 513, 514, 515, 543, 544, 548,
        554, 587, 631, 646, 873, 990, 993, 995, 1025, 1026, 1027, 1028, 1029, 1110, 1433, 1720,
        1723, 1755, 1900, 2000, 2001, 2049, 2121, 2717, 3000, 3128, 3306, 3389, 3986, 4899,
        5000, 5009, 5051, 5060, 5101, 5190, 5357, 5432, 5631, 5666, 5800, 5900, 6000, 6001,
        6646, 7070, 8000, 8008, 8009, 8080, 8081, 8443, 8888, 9100, 9999, 10000, 32768, 49152
    ])),
    "full": "1-65535",
}


def resolve_ports(ports: str) -> str:
    """
    Resolve port specification to nmap-compatible format.

    Parameters
    ----------
    ports : str
        Port specification (preset name or port list).

    Returns
    -------
    str
        Nmap-compatible port specification.
    """
    return PORT_PRESETS.get(ports.lower(), ports)


# =============================================================================
# COMPLIANCE FRAMEWORK PRESETS
# =============================================================================

COMPLIANCE_PRESETS = {
    "anssi": {
        "name": "ANSSI Guide d'hygiène informatique",
        "levels": ["essential", "standard", "reinforced"],
        "default_level": "standard",
    },
    "nist": {
        "name": "NIST Cybersecurity Framework",
        "functions": ["identify", "protect", "detect", "respond", "recover"],
    },
    "cis": {
        "name": "CIS Controls v8",
        "implementation_groups": ["IG1", "IG2", "IG3"],
    },
}


# =============================================================================
# ERROR HANDLING
# =============================================================================

def wrap_kernel_error(error: Exception, kernel_name: str) -> Dict[str, Any]:
    """
    Wrap kernel exceptions in standardized error format.

    Parameters
    ----------
    error : Exception
        The exception that occurred.
    kernel_name : str
        Name of the kernel that failed.

    Returns
    -------
    dict
        Standardized error response.
    """
    return {
        "status": "error",
        "error": str(error),
        "kernel": kernel_name,
        "summary": f"Kernel {kernel_name} failed: {str(error)[:100]}",
        "action_items": [
            {"priority": "high", "action": f"Investigate {kernel_name} failure"}
        ],
    }


def validate_target(target: str) -> tuple[bool, str]:
    """
    Validate target specification.

    Parameters
    ----------
    target : str
        Target to validate.

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    import re

    # Empty check
    if not target or not target.strip():
        return False, "Target cannot be empty"

    target = target.strip()

    # Keywords
    if target.lower() in ("discovered", "all"):
        return True, ""

    # IPv4
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$"
    if re.match(ipv4_pattern, target):
        return True, ""

    # IPv4 range
    range_pattern = r"^(\d{1,3}\.){3}\d{1,3}-\d{1,3}$"
    if re.match(range_pattern, target):
        return True, ""

    # Hostname
    hostname_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$"
    if re.match(hostname_pattern, target):
        return True, ""

    return False, f"Invalid target format: {target}"


# =============================================================================
# MANIFEST GENERATION FOR SECURITY SCANS
# =============================================================================

def generate_security_manifest(
    workspace: Path,
    targets: List[str],
    frameworks: List[str] = None,
    options: Dict[str, Any] = None,
) -> Path:
    """
    Generate a manifest.yaml for security scanning.

    Parameters
    ----------
    workspace : Path
        Workspace directory.
    targets : list
        List of target specifications.
    frameworks : list, optional
        Compliance frameworks to use.
    options : dict, optional
        Additional options.

    Returns
    -------
    Path
        Path to generated manifest.
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for manifest generation")

    manifest = {
        "name": "Security Assessment",
        "description": "Automated security scan via MCP",
        "created": datetime.now().isoformat(),
        "config": {
            "log_level": "info",
            "dry_run": options.get("dry_run", False) if options else False,
        },
        "targets": targets,
        "stages": {
            1: {
                "net_discover": {
                    "enabled": True,
                    "options": {
                        "targets": targets,
                        "methods": ["ping"],
                        "timeout": options.get("timeout", 120) if options else 120,
                    },
                },
                "port_scan": {
                    "enabled": True,
                    "options": {
                        "targets": "discovered",
                        "ports": options.get("ports", "common") if options else "common",
                    },
                },
            },
            2: {
                "ssl_analysis": {"enabled": True},
                "vuln_assess": {"enabled": True},
                "compliance": {
                    "enabled": bool(frameworks),
                    "options": {
                        "frameworks": frameworks or ["anssi"],
                    },
                },
                "risk_network": {"enabled": True},
            },
            3: {
                "section_security": {
                    "enabled": True,
                    "options": {
                        "format": "markdown",
                        "language": options.get("language", "en") if options else "en",
                    },
                },
            },
        },
    }

    manifest_path = workspace / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    return manifest_path
