"""
Kernel: Risk Matrix
Stage: 2 (Analysis)

Combines code metrics with operational volumetry to compute weighted risk scores.
This is the core kernel for volumetry-aware audit analysis.

Risk Formula:
    Risk = (LOC_norm × w_loc) + (Complexity_norm × w_complexity) + (Volumetry_norm × w_volumetry)

Default weights: LOC=0.25, Complexity=0.25, Volumetry=0.50

Risk Levels:
    CRITICAL: Risk >= 7.0
    HIGH:     Risk >= 5.0
    MEDIUM:   Risk >= 3.0
    LOW:      Risk < 3.0

Dependencies:
    - module_group: Module definitions with LOC and complexity metrics
    - volumetry: Operational volume data per module

Output:
    - matrix: Full risk matrix with all factors
    - ranking: Modules ranked by risk score
    - critical_path: Identified critical processing path
    - summary: Risk distribution statistics

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-16
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Default normalization and weighting parameters
DEFAULT_CONFIG = {
    # Normalization maxima (value → score 10)
    "normalization": {
        "loc_max": 15000,          # 15K LOC → score 10
        "complexity_max": 15,      # 15 methods/class → score 10
        "volumetry_max": 4_000_000,  # 4M messages/day → score 10
    },
    # Risk factor weights (must sum to 1.0)
    "weights": {
        "loc": 0.25,
        "complexity": 0.25,
        "volumetry": 0.50,
    },
    # Risk level thresholds
    "thresholds": {
        "critical": 7.0,
        "high": 5.0,
        "medium": 3.0,
    },
}


class RiskMatrixKernel(Kernel):
    """
    Compute volumetry-weighted risk matrix for modules.

    This kernel combines:
    - Code size (LOC) from module_group
    - Code complexity (methods/class) from module_group
    - Operational volume from volumetry

    The result is a risk score per module that reflects both
    code quality concerns AND operational criticality.

    Configuration options:
        normalization: Custom normalization maxima
        weights: Custom factor weights (loc, complexity, volumetry)
        thresholds: Custom risk level thresholds

    Example manifest config:
        risk_matrix:
          enabled: true
          options:
            weights:
              loc: 0.25
              complexity: 0.25
              volumetry: 0.50
            normalization:
              volumetry_max: 4000000
    """

    name = "risk_matrix"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Volumetry-weighted risk assessment"

    requires = ["module_group", "volumetry"]
    provides = ["risk_matrix", "risk_ranking", "critical_path"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Compute risk matrix combining code metrics and volumetry."""

        # Load dependencies
        module_group_path = input.dependencies.get("module_group")
        volumetry_path = input.dependencies.get("volumetry")

        if not module_group_path or not module_group_path.exists():
            raise RuntimeError("Missing required dependency: module_group")
        if not volumetry_path or not volumetry_path.exists():
            raise RuntimeError("Missing required dependency: volumetry")

        with open(module_group_path) as f:
            module_data = json.load(f).get("data", {})

        with open(volumetry_path) as f:
            volumetry_data = json.load(f).get("data", {})

        modules = module_data.get("modules", {})
        module_volumetry = volumetry_data.get("module_volumetry", {})
        flows = volumetry_data.get("flows", {})

        logger.info(f"[risk_matrix] Computing risk for {len(modules)} modules")

        # Get configuration
        config = self._merge_config(input.config)
        normalization = config["normalization"]
        weights = config["weights"]
        thresholds = config["thresholds"]

        # Compute risk for each module
        matrix = []
        for module_name, module_info in modules.items():
            risk_entry = self._compute_module_risk(
                module_name,
                module_info,
                module_volumetry,
                flows,
                normalization,
                weights,
                thresholds,
            )
            matrix.append(risk_entry)

        # Sort by risk score (descending)
        matrix.sort(key=lambda x: -x["risk_score"])

        # Generate ranking
        ranking = [
            {
                "module": m["module"],
                "risk_score": m["risk_score"],
                "risk_level": m["risk_level"],
                "primary_factor": m["primary_factor"],
            }
            for m in matrix
        ]

        # Identify critical path
        critical_path = self._identify_critical_path(matrix, module_volumetry)

        # Compute statistics
        statistics = self._compute_statistics(matrix, thresholds)

        return {
            "matrix": matrix,
            "ranking": ranking,
            "critical_path": critical_path,
            "configuration": {
                "normalization": normalization,
                "weights": weights,
                "thresholds": thresholds,
            },
            "statistics": statistics,
        }

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults."""
        config = {
            "normalization": {**DEFAULT_CONFIG["normalization"]},
            "weights": {**DEFAULT_CONFIG["weights"]},
            "thresholds": {**DEFAULT_CONFIG["thresholds"]},
        }

        if "normalization" in user_config:
            config["normalization"].update(user_config["normalization"])
        if "weights" in user_config:
            config["weights"].update(user_config["weights"])
        if "thresholds" in user_config:
            config["thresholds"].update(user_config["thresholds"])

        return config

    def _compute_module_risk(
        self,
        module_name: str,
        module_info: Dict[str, Any],
        module_volumetry: Dict[str, str],
        flows: Dict[str, Dict[str, Any]],
        normalization: Dict[str, float],
        weights: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute risk score for a single module."""

        # Extract metrics
        loc = module_info.get("loc", 0)
        methods_per_class = module_info.get("methods_per_class", 0)
        classes = module_info.get("classes", 0)
        methods = module_info.get("methods", 0)
        files = module_info.get("files", 0)

        # Get volumetry for this module
        vol_info = module_volumetry.get(module_name, {})
        volume_day = vol_info.get("volume_day", 0)
        incident_count = vol_info.get("incident_count", 0)

        # If no direct mapping, check if module name is part of any volumetry module
        if not vol_info:
            for vol_module, vol_data in module_volumetry.items():
                # Check for partial match (e.g., iow-ech-sias matches iow-ech-sias-infra)
                if module_name.startswith(vol_module) or vol_module.startswith(module_name):
                    volume_day = vol_data.get("volume_day", 0)
                    incident_count = vol_data.get("incident_count", 0)
                    break

        # Normalize scores (0-10 scale)
        loc_norm = min(10.0, (loc / normalization["loc_max"]) * 10)
        complexity_norm = min(10.0, (methods_per_class / normalization["complexity_max"]) * 10)
        volumetry_norm = min(10.0, (volume_day / normalization["volumetry_max"]) * 10)

        # Compute weighted risk score
        risk_score = (
            loc_norm * weights["loc"] +
            complexity_norm * weights["complexity"] +
            volumetry_norm * weights["volumetry"]
        )

        # Boost risk for modules with incidents
        if incident_count > 0:
            risk_score = min(10.0, risk_score + (incident_count * 0.5))

        # Determine risk level
        risk_level = self._determine_risk_level(risk_score, thresholds)

        # Identify primary risk factor
        factors = {
            "loc": loc_norm * weights["loc"],
            "complexity": complexity_norm * weights["complexity"],
            "volumetry": volumetry_norm * weights["volumetry"],
        }
        primary_factor = max(factors.items(), key=lambda x: x[1])[0]

        return {
            "module": module_name,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level.value,
            "primary_factor": primary_factor,
            # Raw metrics
            "loc": loc,
            "classes": classes,
            "methods": methods,
            "files": files,
            "methods_per_class": round(methods_per_class, 1),
            "volume_day": volume_day,
            "incident_count": incident_count,
            # Normalized scores
            "loc_norm": round(loc_norm, 2),
            "complexity_norm": round(complexity_norm, 2),
            "volumetry_norm": round(volumetry_norm, 2),
            # Weighted contributions
            "loc_contribution": round(factors["loc"], 2),
            "complexity_contribution": round(factors["complexity"], 2),
            "volumetry_contribution": round(factors["volumetry"], 2),
        }

    def _determine_risk_level(
        self,
        score: float,
        thresholds: Dict[str, float]
    ) -> RiskLevel:
        """Determine risk level from score."""
        if score >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif score >= thresholds["high"]:
            return RiskLevel.HIGH
        elif score >= thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _identify_critical_path(
        self,
        matrix: List[Dict[str, Any]],
        module_volumetry: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Identify the critical processing path."""

        # Entry points (high volumetry + HIGH/CRITICAL risk)
        entry_points = []
        processors = []
        libraries = []

        for entry in matrix:
            module = entry["module"]
            vol_info = module_volumetry.get(module, {})
            role = vol_info.get("role", "unknown")

            if entry["risk_level"] in ("CRITICAL", "HIGH"):
                if role == "entry_point":
                    entry_points.append(module)
                elif role == "processor":
                    processors.append(module)
                elif role == "library":
                    libraries.append(module)
                elif entry["volumetry_norm"] > 5:
                    # High volume even without explicit role
                    processors.append(module)

        # Build path: entry → processors → libraries (in order of risk)
        path = entry_points + processors + libraries

        return {
            "path": path,
            "entry_points": entry_points,
            "processors": processors,
            "libraries": libraries,
            "length": len(path),
        }

    def _compute_statistics(
        self,
        matrix: List[Dict[str, Any]],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute risk distribution statistics."""

        if not matrix:
            return {
                "total_modules": 0,
                "by_level": {},
                "avg_risk": 0,
                "max_risk": 0,
            }

        # Count by level
        by_level = {level.value: 0 for level in RiskLevel}
        for entry in matrix:
            by_level[entry["risk_level"]] += 1

        # Compute averages
        avg_risk = sum(e["risk_score"] for e in matrix) / len(matrix)
        max_risk = max(e["risk_score"] for e in matrix)
        min_risk = min(e["risk_score"] for e in matrix)

        # High-risk summary
        high_risk_modules = [e["module"] for e in matrix if e["risk_level"] in ("CRITICAL", "HIGH")]

        return {
            "total_modules": len(matrix),
            "by_level": by_level,
            "avg_risk": round(avg_risk, 2),
            "max_risk": round(max_risk, 2),
            "min_risk": round(min_risk, 2),
            "high_risk_count": len(high_risk_modules),
            "high_risk_modules": high_risk_modules[:10],  # Top 10
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        by_level = stats.get("by_level", {})
        critical_path = data.get("critical_path", {})
        ranking = data.get("ranking", [])

        # Top 3 risk modules
        top_risk = []
        for r in ranking[:3]:
            top_risk.append(f"{r['module']}({r['risk_score']:.1f})")
        top_str = ", ".join(top_risk) if top_risk else "none"

        # Critical path summary
        path_len = critical_path.get("length", 0)
        path_modules = critical_path.get("path", [])[:3]
        path_str = " → ".join(path_modules) if path_modules else "N/A"

        return (
            f"Risk Matrix: {stats.get('total_modules', 0)} modules. "
            f"CRITICAL: {by_level.get('CRITICAL', 0)}, "
            f"HIGH: {by_level.get('HIGH', 0)}, "
            f"MEDIUM: {by_level.get('MEDIUM', 0)}, "
            f"LOW: {by_level.get('LOW', 0)}. "
            f"Avg risk: {stats.get('avg_risk', 0):.1f}. "
            f"Top: {top_str}. "
            f"Critical path ({path_len}): {path_str}."
        )
