"""
Kernel: Entropy Analysis
Stage: 2 (Analysis)
Wraps: ragix_audit.entropy.EntropyComputer

Computes information-theoretic metrics for codebase analysis:
- Structural Entropy: How evenly is code distributed across components?
- Complexity Entropy: How evenly is complexity distributed across files?
- Coupling Entropy: How evenly are dependencies distributed?
- Inequality metrics: Gini coefficient, concentration ratios

High entropy = uniform distribution (well-balanced)
Low entropy = concentrated distribution (potential risk)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Set
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class EntropyKernel(Kernel):
    """
    Compute entropy metrics using EntropyComputer.

    This kernel wraps ragix_audit.entropy.EntropyComputer to compute
    information-theoretic metrics measuring code distribution uniformity.

    Configuration options:
        include_inequality: Include Gini/concentration metrics (default: true)

    Dependencies:
        metrics: File-level metrics (LOC, complexity)
        dependency: Coupling information
        partition: Component assignments (optional, enhances analysis)

    Output:
        entropy_metrics: Structural, complexity, and coupling entropy
        inequality_metrics: Gini coefficient, CR-4, CR-8, Herfindahl
        component_analysis: Per-component breakdown
        interpretations: Human-readable explanations
    """

    name = "entropy"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Information-theoretic distribution analysis"

    requires = ["metrics", "dependency"]
    provides = ["entropy_metrics", "inequality_metrics", "distribution_health"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Compute entropy metrics using EntropyComputer."""

        # Import existing tools (REUSE!)
        from ragix_audit.entropy import (
            EntropyComputer,
            compute_inequality_metrics,
            gini_coefficient,
            concentration_ratio,
        )

        # Load dependencies
        metrics_path = input.dependencies.get("metrics")
        dep_path = input.dependencies.get("dependency")
        partition_path = input.dependencies.get("partition")

        if not metrics_path or not metrics_path.exists():
            raise RuntimeError("Missing required dependency: metrics")
        if not dep_path or not dep_path.exists():
            raise RuntimeError("Missing required dependency: dependency")

        with open(metrics_path) as f:
            metrics_data = json.load(f).get("data", {})
        with open(dep_path) as f:
            dep_data = json.load(f).get("data", {})

        partition_data = {}
        if partition_path and partition_path.exists():
            with open(partition_path) as f:
                partition_data = json.load(f).get("data", {})

        # Get configuration
        include_inequality = input.config.get("include_inequality", True)

        logger.info("[entropy] Computing information-theoretic metrics")

        # Build input data for EntropyComputer
        files = metrics_data.get("files", [])
        hotspots = metrics_data.get("hotspots", [])
        graph_nodes = dep_data.get("graph", {}).get("nodes", [])

        # 1. Component sizes (LOC per component)
        component_sizes: Dict[str, int] = {}
        component_file_counts: Dict[str, int] = {}

        if partition_data:
            # Use partition data for component breakdown
            partitions = partition_data.get("partitions", {})
            for label, items in partitions.items():
                total_loc = 0
                file_count = 0
                for item in items:
                    loc = item.get("loc", 0)
                    if loc > 0:
                        total_loc += loc
                        file_count += 1
                if total_loc > 0:
                    component_sizes[label] = total_loc
                    component_file_counts[label] = file_count
        else:
            # Fall back to package-level grouping
            package_sizes: Dict[str, int] = {}
            package_counts: Dict[str, int] = {}
            for f in files:
                path = f.get("path", "")
                loc = f.get("loc", 0)
                # Extract package from path
                parts = path.split("/")
                pkg = parts[-2] if len(parts) >= 2 else "root"
                package_sizes[pkg] = package_sizes.get(pkg, 0) + loc
                package_counts[pkg] = package_counts.get(pkg, 0) + 1
            component_sizes = package_sizes
            component_file_counts = package_counts

        # 2. File complexities
        file_complexities: Dict[str, float] = {}
        for f in files:
            path = f.get("path", "")
            # Try to get complexity from file data
            complexity = f.get("complexity", 0) or f.get("avg_complexity", 0)
            if complexity > 0:
                file_complexities[path] = float(complexity)

        # If no per-file complexity, aggregate from hotspots
        if not file_complexities and hotspots:
            for h in hotspots:
                name = h.get("name", "")
                cc = h.get("complexity", 0)
                if cc > 0:
                    # Group by file prefix
                    file_key = name.rsplit(".", 1)[0] if "." in name else name
                    file_complexities[file_key] = file_complexities.get(file_key, 0) + cc

        # 3. Node degrees (coupling)
        node_degrees: Dict[str, int] = {}
        for node in graph_nodes:
            node_id = node.get("id", "")
            ca = node.get("afferent_coupling", 0)
            ce = node.get("efferent_coupling", 0)
            total_degree = ca + ce
            if total_degree > 0:
                node_degrees[node_id] = total_degree

        # Use existing EntropyComputer (REUSE!)
        entropy_computer = EntropyComputer()
        entropy_metrics = entropy_computer.compute_all(
            component_sizes=component_sizes if component_sizes else None,
            component_file_counts=component_file_counts if component_file_counts else None,
            file_complexities=file_complexities if file_complexities else None,
            node_degrees=node_degrees if node_degrees else None,
        )

        # Compute inequality metrics if requested (REUSE!)
        inequality_data = {}
        if include_inequality:
            # LOC distribution inequality
            if component_sizes:
                loc_values = list(component_sizes.values())
                loc_inequality = compute_inequality_metrics([float(v) for v in loc_values])
                inequality_data["loc_distribution"] = loc_inequality.to_dict()

            # Complexity distribution inequality
            if file_complexities:
                cc_values = list(file_complexities.values())
                cc_inequality = compute_inequality_metrics(cc_values)
                inequality_data["complexity_distribution"] = cc_inequality.to_dict()

            # Coupling distribution inequality
            if node_degrees:
                deg_values = list(node_degrees.values())
                deg_inequality = compute_inequality_metrics([float(v) for v in deg_values])
                inequality_data["coupling_distribution"] = deg_inequality.to_dict()

        # Build component analysis
        component_analysis = []
        for comp_id, loc in component_sizes.items():
            total_loc = sum(component_sizes.values())
            share = loc / total_loc if total_loc > 0 else 0
            component_analysis.append({
                "component": comp_id,
                "loc": loc,
                "share": round(share, 4),
                "file_count": component_file_counts.get(comp_id, 0),
            })
        component_analysis.sort(key=lambda x: -x["loc"])

        # Health score (0-100 based on entropy normalization)
        health_scores = {}
        if entropy_metrics.structural_normalized > 0:
            health_scores["structural"] = round(entropy_metrics.structural_normalized * 100, 1)
        if entropy_metrics.complexity_normalized > 0:
            health_scores["complexity"] = round(entropy_metrics.complexity_normalized * 100, 1)
        if entropy_metrics.coupling_normalized > 0:
            health_scores["coupling"] = round(entropy_metrics.coupling_normalized * 100, 1)

        # Overall health (average of normalized entropies)
        if health_scores:
            health_scores["overall"] = round(sum(health_scores.values()) / len(health_scores), 1)

        # Statistics
        statistics = {
            "components_analyzed": len(component_sizes),
            "files_with_complexity": len(file_complexities),
            "nodes_with_coupling": len(node_degrees),
            "max_possible_entropy": round(entropy_metrics.max_possible_entropy, 3),
        }

        return {
            "entropy_metrics": entropy_metrics.to_dict(),
            "inequality_metrics": inequality_data,
            "component_analysis": component_analysis[:30],  # Limit output
            "health_scores": health_scores,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        entropy = data.get("entropy_metrics", {})
        health = data.get("health_scores", {})
        inequality = data.get("inequality_metrics", {})
        stats = data.get("statistics", {})

        normalized = entropy.get("normalized", {})

        # Get Gini for LOC distribution
        loc_gini = inequality.get("loc_distribution", {}).get("gini", 0)

        # Determine overall assessment
        overall_health = health.get("overall", 0)
        if overall_health >= 70:
            assessment = "well-balanced"
        elif overall_health >= 50:
            assessment = "moderately concentrated"
        else:
            assessment = "highly concentrated (risk)"

        return (
            f"Entropy: {stats.get('components_analyzed', 0)} components. "
            f"Structural H={entropy.get('structural_entropy', 0):.2f} "
            f"({normalized.get('structural', 0)*100:.0f}% norm). "
            f"Complexity H={entropy.get('complexity_entropy', 0):.2f} "
            f"({normalized.get('complexity', 0)*100:.0f}% norm). "
            f"Coupling H={entropy.get('coupling_entropy', 0):.2f}. "
            f"Gini={loc_gini:.2f}. "
            f"Health: {overall_health:.0f}% ({assessment})."
        )
