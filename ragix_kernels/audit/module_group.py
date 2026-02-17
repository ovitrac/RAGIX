"""
Kernel: Module Grouper
Stage: 1 (Collection)

Groups files into functional modules based on path patterns and naming conventions.
Aggregates metrics per module for downstream analysis.

This kernel bridges the gap between file-level metrics and module-level analysis,
enabling volumetry attribution and risk assessment at the module level.

Dependencies:
    - ast_scan: File list with metrics

Output:
    - modules: Module definitions with aggregated metrics
    - file_mapping: File-to-module mapping
    - unassigned: Files that couldn't be assigned to a module

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-16
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Default patterns for common Java project structures
# Note: Using (?i) for case-insensitive matching
DEFAULT_PATTERNS = [
    # ACME-ERP-style: module-name-master/module-name-submodule
    r"(?i).*/([a-z]+-[a-z]+-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+)-master/.*",  # e.g., iow-iok-sk10-11-12-master
    r"(?i).*/([a-z]+-[a-z]+-[a-z]+-[a-z0-9]+)-master/.*",  # e.g., iow-iok-sg01-02-master
    r"(?i).*/([a-z]+-[a-z]+-[a-z0-9]+)-master/.*",      # e.g., acme-msg-hub-master
    r"(?i).*/([a-z]+-[a-z]+-[a-z]+)-master/.*",         # e.g., acme-support-commons-master
    # Maven/Gradle multi-module patterns (fallback)
    r"(?i).*/([a-z]+-[a-z]+-[a-z0-9]+)/src/.*",         # e.g., acme-msg-hub/src
    r"(?i).*/([a-z]+-[a-z]+)/src/.*",                   # e.g., acme-support/src
    # Package-based patterns (last resort)
    r"(?i).*/com/([a-z]+/[a-z]+)/.*",                   # e.g., com/acme/ech
]


class ModuleGrouperKernel(Kernel):
    """
    Group files into functional modules.

    This kernel analyzes file paths to group them into logical modules,
    aggregating metrics (LOC, classes, methods) at the module level.

    Configuration options:
        patterns: List of regex patterns to extract module names
        mapping: Explicit module name mappings (display names)
        min_files: Minimum files to create a module (default: 1)
        fallback_module: Name for unmatched files (default: "_other")

    Example manifest config:
        module_group:
          enabled: true
          options:
            patterns:
              - regex: ".*/([a-z]+-[a-z]+-[a-z0-9]+)/.*"
                group: 1
            mapping:
              acme-msg-hub: "Exchange MSG-HUB"
              iow-iok-sk01: "SC01 Processing"
    """

    name = "module_group"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "File grouping into functional modules"

    requires = ["ast_scan"]
    provides = ["modules", "module_metrics", "file_mapping"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Group files into modules and aggregate metrics."""

        # Load ast_scan data
        ast_scan_path = input.dependencies.get("ast_scan")
        if not ast_scan_path or not ast_scan_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")

        with open(ast_scan_path) as f:
            ast_data = json.load(f).get("data", {})

        files = ast_data.get("files", [])
        logger.info(f"[module_group] Processing {len(files)} files")

        # Get configuration
        patterns = self._get_patterns(input.config)
        mapping = input.config.get("mapping", {})
        min_files = input.config.get("min_files", 1)
        fallback_module = input.config.get("fallback_module", "_other")

        # Group files by module
        module_files: Dict[str, List[Dict[str, Any]]] = {}
        file_mapping: Dict[str, str] = {}
        unassigned: List[str] = []

        for file_entry in files:
            file_path = file_entry.get("path", "")
            module_name = self._extract_module(file_path, patterns)

            if module_name:
                # Normalize module name (remove suffixes like -master, -infra)
                module_name = self._normalize_module_name(module_name)

                if module_name not in module_files:
                    module_files[module_name] = []
                module_files[module_name].append(file_entry)
                file_mapping[file_path] = module_name
            else:
                unassigned.append(file_path)

        # Handle unassigned files
        if unassigned and fallback_module:
            module_files[fallback_module] = [
                {"path": p, "symbols": 0, "classes": 0, "methods": 0, "functions": 0}
                for p in unassigned
            ]

        # Filter modules by min_files
        filtered_modules = {
            name: files_list
            for name, files_list in module_files.items()
            if len(files_list) >= min_files
        }

        # Aggregate metrics per module
        modules = {}
        for module_name, files_list in filtered_modules.items():
            display_name = mapping.get(module_name, module_name)

            # Aggregate metrics
            total_loc = sum(self._estimate_loc(f) for f in files_list)
            total_classes = sum(f.get("classes", 0) for f in files_list)
            total_methods = sum(f.get("methods", 0) for f in files_list)
            total_functions = sum(f.get("functions", 0) for f in files_list)
            total_symbols = sum(f.get("symbols", 0) for f in files_list)

            # Calculate complexity indicators
            methods_per_class = total_methods / total_classes if total_classes > 0 else 0
            loc_per_class = total_loc / total_classes if total_classes > 0 else 0

            modules[module_name] = {
                "display_name": display_name,
                "files": len(files_list),
                "loc": total_loc,
                "classes": total_classes,
                "methods": total_methods,
                "functions": total_functions,
                "symbols": total_symbols,
                "methods_per_class": round(methods_per_class, 1),
                "loc_per_class": round(loc_per_class, 1),
                "file_list": [f.get("path", "") for f in files_list],
            }

        # Statistics
        statistics = {
            "total_modules": len(modules),
            "total_files": len(files),
            "assigned_files": len(file_mapping),
            "unassigned_files": len(unassigned),
            "total_loc": sum(m.get("loc", 0) for m in modules.values()),
            "total_classes": sum(m.get("classes", 0) for m in modules.values()),
            "total_methods": sum(m.get("methods", 0) for m in modules.values()),
            "largest_module": max(modules.items(), key=lambda x: x[1].get("loc", 0))[0] if modules else None,
        }

        # Ranking by LOC
        ranking = sorted(
            [{"module": k, "loc": v.get("loc", 0), "files": v.get("files", 0)}
             for k, v in modules.items()],
            key=lambda x: -x["loc"]
        )

        return {
            "modules": modules,
            "file_mapping": file_mapping,
            "unassigned": unassigned,
            "ranking": ranking,
            "statistics": statistics,
        }

    def _get_patterns(self, config: Dict[str, Any]) -> List[Tuple[re.Pattern, int]]:
        """Get regex patterns from config or use defaults."""
        patterns = []

        # User-defined patterns
        user_patterns = config.get("patterns", [])
        for p in user_patterns:
            if isinstance(p, str):
                patterns.append((re.compile(p), 1))
            elif isinstance(p, dict):
                regex = p.get("regex", "")
                group = p.get("group", 1)
                if regex:
                    patterns.append((re.compile(regex), group))

        # Add defaults if no user patterns
        if not patterns:
            for regex in DEFAULT_PATTERNS:
                patterns.append((re.compile(regex), 1))

        return patterns

    def _extract_module(
        self,
        file_path: str,
        patterns: List[Tuple[re.Pattern, int]]
    ) -> Optional[str]:
        """Extract module name from file path using patterns."""
        for pattern, group_idx in patterns:
            match = pattern.match(file_path)
            if match:
                try:
                    return match.group(group_idx)
                except IndexError:
                    continue
        return None

    def _normalize_module_name(self, name: str) -> str:
        """Normalize module name by removing common suffixes."""
        # Remove common suffixes like -master, -infra, -api, -impl, -core
        suffixes_to_remove = ["-master", "-main", "-infra", "-api", "-impl", "-core", "-common"]
        result = name

        for suffix in suffixes_to_remove:
            if result.endswith(suffix):
                result = result[:-len(suffix)]

        return result

    def _estimate_loc(self, file_entry: Dict[str, Any]) -> int:
        """
        Estimate LOC from file entry.

        Uses symbols count as a proxy if LOC not directly available.
        Typical Java file: ~30 lines per symbol (methods, fields, etc.)
        """
        # If LOC is directly available
        if "loc" in file_entry:
            return file_entry["loc"]

        # Estimate from symbols (rough heuristic)
        symbols = file_entry.get("symbols", 0)
        classes = file_entry.get("classes", 0)
        methods = file_entry.get("methods", 0)

        # Heuristic: ~50 lines per class, ~10 lines per method
        estimated = (classes * 50) + (methods * 10)

        # Minimum 20 lines per file with any content
        if symbols > 0 and estimated == 0:
            estimated = 20

        return estimated

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        ranking = data.get("ranking", [])

        # Top 3 modules
        top_modules = []
        for r in ranking[:3]:
            top_modules.append(f"{r['module']}({r['loc']:,} LOC)")
        top_str = ", ".join(top_modules) if top_modules else "none"

        return (
            f"Modules: {stats.get('total_modules', 0)} modules from {stats.get('total_files', 0)} files. "
            f"Total: {stats.get('total_loc', 0):,} LOC, {stats.get('total_classes', 0)} classes, "
            f"{stats.get('total_methods', 0)} methods. "
            f"Unassigned: {stats.get('unassigned_files', 0)}. "
            f"Top: {top_str}."
        )
