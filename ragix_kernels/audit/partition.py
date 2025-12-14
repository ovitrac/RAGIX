"""
Kernel: Codebase Partition
Stage: 1 (Data Collection)
Wraps: ragix_audit.partitioner.CodebasePartitioner

Partitions a codebase into logical applications:
- Identifies application fingerprints (SIAS, TICC, etc.)
- Classifies classes using graph propagation algorithm
- Detects shared/common code
- Identifies dead code candidates

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class PartitionKernel(Kernel):
    """
    Partition codebase into logical applications.

    This kernel uses the graph propagation algorithm to classify
    classes into application groups (SIAS, TICC, GRDF, etc.) or
    identify them as shared code, unknown, or dead code.

    Configuration options:
        preset: Predefined configuration (e.g., "sias_ticc", "grdf")
        applications: List of application fingerprints
        propagation_iterations: Max propagation iterations (default: 5)
        confidence_threshold: Minimum confidence for classification (default: 0.7)
        forward_weight: Weight for outgoing deps (default: 0.7)
        reverse_weight: Weight for incoming deps (default: 0.3)
        package_cohesion_bonus: Bonus for same-package neighbors (default: 0.2)

    Dependencies:
        ast_scan: Provides symbol data
        dependency: Provides graph data

    Output:
        assignments: Classification for each class
        summary: Count by label
        partitions: Grouped classes by label
        statistics: Partition statistics
    """

    name = "partition"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Partition codebase into logical applications"

    requires = ["ast_scan", "dependency"]
    provides = ["partitions", "classifications"]

    # Preset configurations
    PRESETS = {
        "sias_ticc": {
            "applications": [
                {"name": "SIAS", "patterns": ["*sias*", "*Sias*", "SIAS*"]},
                {"name": "TICC", "patterns": ["*ticc*", "*Ticc*", "TICC*"]},
            ]
        },
        "grdf": {
            "applications": [
                {"name": "GRDF", "patterns": ["*grdf*", "*Grdf*", "GRDF*"]},
            ]
        },
        "generic": {
            "applications": []
        }
    }

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Run codebase partitioning."""

        # Import here to avoid circular imports
        from ragix_audit.partitioner import (
            CodebasePartitioner,
            PartitionConfig,
            ApplicationFingerprint,
        )

        # Get configuration
        preset = input.config.get("preset", "generic")
        preset_config = self.PRESETS.get(preset, self.PRESETS["generic"])

        # Build partition config
        applications = input.config.get("applications", preset_config.get("applications", []))
        app_fingerprints = []
        for app in applications:
            app_fingerprints.append(ApplicationFingerprint(
                name=app.get("name", "APP"),
                patterns=app.get("patterns", []),
                entry_points=app.get("entry_points", []),
            ))

        config = PartitionConfig(
            applications=app_fingerprints,
            propagation_iterations=input.config.get("propagation_iterations", 5),
            confidence_threshold=input.config.get("confidence_threshold", 0.7),
            forward_weight=input.config.get("forward_weight", 0.7),
            reverse_weight=input.config.get("reverse_weight", 0.3),
            package_cohesion_bonus=input.config.get("package_cohesion_bonus", 0.2),
        )

        # Load dependency data
        dependency_path = input.dependencies.get("dependency")
        if not dependency_path or not dependency_path.exists():
            raise RuntimeError("Missing required dependency: dependency")

        with open(dependency_path) as f:
            dep_data = json.load(f)

        graph_data = dep_data.get("data", {}).get("graph", {})
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        logger.info(f"[partition] Partitioning {len(nodes)} nodes with {len(edges)} edges")

        # Create partitioner and run
        partitioner = CodebasePartitioner(config)
        partitioner.load_from_graph({"nodes": nodes, "links": edges})
        result = partitioner.partition()

        # Build output
        assignments_data = {}
        for fqn, assignment in result.assignments.items():
            # Get primary evidence method if available
            primary_method = "unknown"
            primary_reason = ""
            if assignment.evidence:
                primary_method = assignment.evidence[0].method
                primary_reason = assignment.evidence[0].details

            assignments_data[fqn] = {
                "label": assignment.label,
                "confidence": assignment.confidence,
                "method": primary_method,
                "reason": primary_reason,
                "votes": assignment.votes,
            }

        # Group by label
        partitions = {}
        for fqn, assignment in result.assignments.items():
            label = assignment.label
            if label not in partitions:
                partitions[label] = []

            # Get primary method from evidence
            method = "unknown"
            if assignment.evidence:
                method = assignment.evidence[0].method

            partitions[label].append({
                "fqn": fqn,
                "confidence": assignment.confidence,
                "method": method,
            })

        # Sort partitions by size
        partition_summary = {
            label: len(classes)
            for label, classes in partitions.items()
        }

        # Statistics
        statistics = {
            "total_classes": len(assignments_data),
            "total_partitions": len(partitions),
            "by_label": partition_summary,
            "by_method": self._count_by_method(assignments_data),
            "dead_code_count": partition_summary.get("DEAD_CODE", 0),
            "unknown_count": partition_summary.get("UNKNOWN", 0),
            "shared_count": partition_summary.get("SHARED", 0),
        }

        return {
            "assignments": assignments_data,
            "partitions": partitions,
            "summary": partition_summary,
            "statistics": statistics,
            "config": {
                "preset": preset,
                "applications": [a.name for a in app_fingerprints],
                "propagation_iterations": config.propagation_iterations,
                "confidence_threshold": config.confidence_threshold,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        summary = data.get("summary", {})
        config = data.get("config", {})

        # Build partition list
        parts = []
        for label, count in sorted(summary.items(), key=lambda x: -x[1]):
            if label not in ("UNKNOWN", "DEAD_CODE", "SHARED"):
                parts.append(f"{label}:{count}")

        apps_str = ", ".join(parts[:5]) if parts else "none detected"
        if len(parts) > 5:
            apps_str += f" +{len(parts)-5} more"

        return (
            f"Partitioned {stats.get('total_classes', 0)} classes into "
            f"{stats.get('total_partitions', 0)} groups. "
            f"Applications: {apps_str}. "
            f"Unknown: {stats.get('unknown_count', 0)}, "
            f"Dead code: {stats.get('dead_code_count', 0)}, "
            f"Shared: {stats.get('shared_count', 0)}."
        )

    def _count_by_method(self, assignments: Dict[str, Dict]) -> Dict[str, int]:
        """Count assignments by classification method."""
        counts: Dict[str, int] = {}
        for data in assignments.values():
            method = data.get("method", "unknown")
            counts[method] = counts.get(method, 0) + 1
        return counts
