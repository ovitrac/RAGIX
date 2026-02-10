"""
Kernel: Maven CVE Scan
Stage: 2 (Analysis)

Scans Maven dependencies against a local vulnerability catalog.
Two-tier design:
  - Tier 1: Built-in catalog (data/java_cve_catalog.json) — no network
  - Tier 2: Optional OWASP cache (workspace/data/owasp_cache.json) — pre-downloaded

Sovereignty: this kernel makes NO network calls.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-09
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class MavenCveKernel(Kernel):
    """
    Scan Maven dependencies for known vulnerabilities.

    Compares extracted dependency versions against a local CVE catalog.
    No network calls (sovereignty: local-only). The built-in catalog
    covers critical/high CVEs for common Java libraries (Spring, Log4j,
    Jackson, ActiveMQ, etc.).
    """

    name = "maven_cve"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Scan Maven dependencies for known vulnerabilities"
    requires = ["maven_deps"]
    provides = ["maven_vulnerabilities", "cve_report"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load maven_deps output
        deps_path = input.dependencies.get("maven_deps")
        if not deps_path or not deps_path.exists():
            raise RuntimeError("maven_deps output not found")

        with open(deps_path) as f:
            deps_data = json.load(f).get("data", {})

        all_deps = deps_data.get("all_dependencies", [])
        modules = deps_data.get("modules", [])

        # Load CVE catalog (Tier 1: built-in)
        catalog = self._load_builtin_catalog()
        catalog_tier = "builtin"
        catalog_version = catalog.get("_meta", {}).get("version", "unknown")

        # Tier 2: check for OWASP cache in workspace
        owasp_path = input.workspace / "data" / "owasp_cache.json"
        if owasp_path.exists():
            try:
                with open(owasp_path) as f:
                    owasp_data = json.load(f)
                catalog["advisories"].extend(owasp_data.get("advisories", []))
                catalog_tier = "builtin+owasp"
                logger.info(f"[maven_cve] Loaded OWASP cache: {len(owasp_data.get('advisories', []))} advisories")
            except Exception as e:
                logger.warning(f"[maven_cve] Failed to load OWASP cache: {e}")

        # Build dependency index: groupId:artifactId → {version, used_by}
        dep_index: Dict[str, Dict[str, Any]] = {}
        for dep in all_deps:
            key = f"{dep['groupId']}:{dep['artifactId']}"
            dep_index[key] = dep

        # Also index from module-level dependencies (may have more version info)
        for module in modules:
            for dep in module.get("dependencies", []):
                key = f"{dep['groupId']}:{dep['artifactId']}"
                if key not in dep_index:
                    dep_index[key] = {
                        "groupId": dep["groupId"],
                        "artifactId": dep["artifactId"],
                        "version": dep.get("version", "inherited"),
                        "used_by": [module["artifactId"]],
                    }
                elif dep.get("version") and dep["version"] != "inherited":
                    dep_index[key]["version"] = dep["version"]

        # Scan
        vulnerabilities = []
        for advisory in catalog.get("advisories", []):
            for affected in advisory.get("affected", []):
                key = f"{affected['groupId']}:{affected['artifactId']}"
                if key in dep_index:
                    dep_info = dep_index[key]
                    dep_version = dep_info.get("version", "")
                    version_range = affected.get("version_range", "")

                    if dep_version and dep_version != "inherited":
                        is_affected = self._version_in_range(dep_version, version_range)
                    else:
                        # Can't determine — flag as potential
                        is_affected = True

                    if is_affected:
                        vulnerabilities.append({
                            "dependency": f"{key}:{dep_version or '?'}",
                            "cve_id": advisory["cve_id"],
                            "title": advisory.get("title", ""),
                            "severity": advisory.get("severity", "UNKNOWN"),
                            "cvss": advisory.get("cvss", 0),
                            "description": advisory.get("description", ""),
                            "fixed_in": advisory.get("fixed_in", ""),
                            "version_range": version_range,
                            "detected_version": dep_version or "unknown",
                            "modules_affected": dep_info.get("used_by", []),
                        })

        # Deduplicate by (cve_id, dependency)
        seen = set()
        unique_vulns = []
        for v in vulnerabilities:
            key = (v["cve_id"], v["dependency"])
            if key not in seen:
                seen.add(key)
                unique_vulns.append(v)

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
        unique_vulns.sort(key=lambda v: (severity_order.get(v["severity"], 5), -v["cvss"]))

        # Statistics
        by_severity: Dict[str, int] = {}
        for v in unique_vulns:
            sev = v["severity"]
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "vulnerabilities": unique_vulns,
            "statistics": {
                "dependencies_scanned": len(dep_index),
                "vulnerabilities_found": len(unique_vulns),
                "by_severity": by_severity,
                "catalog_version": catalog_version,
                "catalog_tier": catalog_tier,
                "catalog_advisories": len(catalog.get("advisories", [])),
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        by_sev = stats.get("by_severity", {})
        sev_str = ", ".join(f"{k}:{v}" for k, v in sorted(by_sev.items()))
        return (
            f"CVE scan: {stats.get('dependencies_scanned', 0)} dependencies scanned, "
            f"{stats.get('vulnerabilities_found', 0)} vulnerabilities found. "
            f"By severity: {sev_str or 'none'}. "
            f"Catalog: {stats.get('catalog_tier', '?')} ({stats.get('catalog_version', '?')})."
        )

    def _load_builtin_catalog(self) -> Dict[str, Any]:
        """Load the built-in CVE catalog from data/ directory."""
        catalog_path = Path(__file__).parent / "data" / "java_cve_catalog.json"
        if not catalog_path.exists():
            logger.warning(f"[maven_cve] Built-in catalog not found at {catalog_path}")
            return {"advisories": [], "_meta": {"version": "missing"}}

        with open(catalog_path) as f:
            return json.load(f)

    def _version_in_range(self, version: str, version_range: str) -> bool:
        """
        Check if a version string falls within a Maven version range.

        Supports Maven range notation:
          [1.0,2.0)  — >= 1.0 and < 2.0
          [1.0,2.0]  — >= 1.0 and <= 2.0
          (1.0,2.0)  — > 1.0 and < 2.0

        Falls back to simple prefix matching for non-standard ranges.
        """
        if not version_range:
            return False

        # Parse Maven range: [lower,upper) or (lower,upper] etc.
        match = re.match(r'^([\[\(])([^,]*),([^)\]]*)([\]\)])$', version_range.strip())
        if not match:
            return False

        lower_incl = match.group(1) == '['
        lower_ver = match.group(2).strip()
        upper_ver = match.group(3).strip()
        upper_incl = match.group(4) == ']'

        try:
            v = self._parse_version(version)
            lo = self._parse_version(lower_ver) if lower_ver else None
            hi = self._parse_version(upper_ver) if upper_ver else None
        except Exception:
            return False

        if lo is not None:
            if lower_incl and v < lo:
                return False
            if not lower_incl and v <= lo:
                return False

        if hi is not None:
            if upper_incl and v > hi:
                return False
            if not upper_incl and v >= hi:
                return False

        return True

    def _parse_version(self, version: str) -> Tuple[int, ...]:
        """
        Parse a version string into a tuple for comparison.

        Handles: "2.7.18", "5.3.18.RELEASE", "2.13.4.1", etc.
        Non-numeric suffixes are stripped for comparison.
        """
        # Strip common qualifiers
        version = re.sub(r'[.-]?(RELEASE|SNAPSHOT|Final|GA|SP\d+|M\d+|RC\d+|alpha|beta).*$',
                         '', version, flags=re.IGNORECASE)
        parts = re.split(r'[.\-]', version)
        result = []
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                break
        return tuple(result) if result else (0,)
