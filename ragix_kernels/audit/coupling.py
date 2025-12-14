"""
Kernel: Coupling Analysis (Martin Metrics)
Stage: 2 (Analysis)
Wraps: ragix_audit.coupling.CouplingComputer

Computes Robert Martin's package coupling metrics:
- Instability (I): Ce / (Ca + Ce)
- Abstractness (A): Abstract classes / Total classes
- Distance from Main Sequence (D): |A + I - 1|
- Stable Dependencies Principle (SDP) violations
- Propagation factors for impact analysis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Set
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class CouplingKernel(Kernel):
    """
    Compute Martin coupling metrics using CouplingComputer.

    This kernel wraps ragix_audit.coupling.CouplingComputer to compute
    package-level coupling metrics following Robert Martin's design principles.

    Configuration options:
        min_coupling: Minimum coupling to include package (default: 3)

    Dependencies:
        dependency: Provides graph data with coupling info

    Output:
        packages: Per-package metrics (Ca, Ce, I, A, D, zone)
        sdp_violations: Stable Dependencies Principle violations
        propagation: Propagation factor analysis
        statistics: Summary statistics
    """

    name = "coupling"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Compute Martin coupling metrics"

    requires = ["dependency"]
    provides = ["coupling_metrics", "sdp_violations", "propagation"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Compute Martin coupling metrics using CouplingComputer."""

        # Import existing tool (REUSE!)
        from ragix_audit.coupling import (
            CouplingComputer,
            compute_all_propagation_factors,
        )

        # Load dependency data
        dep_path = input.dependencies.get("dependency")
        if not dep_path or not dep_path.exists():
            raise RuntimeError("Missing required dependency: dependency")

        with open(dep_path) as f:
            dep_data = json.load(f).get("data", {})

        # Extract graph data
        graph_nodes = dep_data.get("graph", {}).get("nodes", [])
        graph_edges = dep_data.get("graph", {}).get("edges", [])

        logger.info(f"[coupling] Computing metrics for {len(graph_nodes)} nodes")

        # Build package-level dependency graph for CouplingComputer
        # Format: {source_package: {target_packages}}
        package_deps: Dict[str, Set[str]] = {}
        package_classes: Dict[str, Dict[str, Any]] = {}

        for node in graph_nodes:
            node_id = node.get("id", "")
            # Extract package from qualified name
            parts = node_id.rsplit(".", 1)
            pkg = parts[0] if len(parts) == 2 else "(default)"

            if pkg not in package_classes:
                package_classes[pkg] = {"total": 0, "abstract": 0, "interfaces": 0}

            # Count classes (heuristic based on node type)
            if node.get("type") == "class":
                package_classes[pkg]["total"] += 1
                # Detect abstract/interface from naming patterns
                if any(p in node_id for p in ["Abstract", "Interface", "Base"]):
                    package_classes[pkg]["abstract"] += 1

        # Build package dependency edges
        for edge in graph_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")

            source_pkg = source.rsplit(".", 1)[0] if "." in source else "(default)"
            target_pkg = target.rsplit(".", 1)[0] if "." in target else "(default)"

            if source_pkg not in package_deps:
                package_deps[source_pkg] = set()
            package_deps[source_pkg].add(target_pkg)

        # Use existing CouplingComputer (REUSE!)
        coupling_computer = CouplingComputer()
        analysis = coupling_computer.compute_from_graph(
            dependencies=package_deps,
            package_classes=package_classes,
        )

        # Convert to JSON-serializable format
        packages_data = []
        for pkg_name, pkg_coupling in analysis.packages.items():
            packages_data.append({
                "package": pkg_coupling.package,
                "afferent_coupling": pkg_coupling.ca,
                "efferent_coupling": pkg_coupling.ce,
                "internal": pkg_coupling.internal,
                "total_classes": pkg_coupling.total_classes,
                "abstract_classes": pkg_coupling.abstract_classes,
                "instability": round(pkg_coupling.instability, 3),
                "abstractness": round(pkg_coupling.abstractness, 3),
                "distance": round(pkg_coupling.distance, 3),
                "zone": pkg_coupling.zone.value if hasattr(pkg_coupling.zone, 'value') else str(pkg_coupling.zone),
                "dependents": pkg_coupling.dependents[:10],
                "dependencies": pkg_coupling.dependencies[:10],
            })

        # SDP violations
        sdp_data = []
        for v in analysis.sdp_violations:
            sdp_data.append({
                "source_package": v.source_package,
                "source_instability": round(v.source_instability, 3),
                "target_package": v.target_package,
                "target_instability": round(v.target_instability, 3),
                "delta": round(v.delta, 3),
                "severity": v.severity,
            })

        # Compute propagation factors using existing tool (REUSE!)
        # Build node degree map for propagation
        node_degrees: Dict[str, int] = {}
        for node in graph_nodes:
            node_id = node.get("id", "")
            ca = node.get("afferent_coupling", 0)
            ce = node.get("efferent_coupling", 0)
            node_degrees[node_id] = ca + ce

        propagation_data = {}
        if node_degrees:
            try:
                prop_analysis = compute_all_propagation_factors(
                    {k: v for k, v in package_deps.items()},
                    node_degrees
                )
                propagation_data = {
                    "critical_nodes": prop_analysis.critical_nodes[:20],
                    "high_impact_nodes": prop_analysis.high_impact_nodes[:20],
                    "average_pf": round(prop_analysis.average_pf, 3),
                    "max_pf": round(prop_analysis.max_pf, 3),
                }
            except Exception as e:
                logger.warning(f"[coupling] Propagation analysis failed: {e}")
                propagation_data = {"error": str(e)}

        # Statistics - build zone distribution from individual counts
        zone_distribution = {}
        if analysis.packages_in_pain > 0:
            zone_distribution["pain"] = analysis.packages_in_pain
        if analysis.packages_useless > 0:
            zone_distribution["useless"] = analysis.packages_useless
        if analysis.packages_on_sequence > 0:
            zone_distribution["main_sequence"] = analysis.packages_on_sequence
        if analysis.packages_balanced > 0:
            zone_distribution["balanced"] = analysis.packages_balanced

        statistics = {
            "total_packages": len(packages_data),
            "avg_instability": round(analysis.avg_instability, 3),
            "avg_abstractness": round(analysis.avg_abstractness, 3),
            "avg_distance": round(analysis.avg_distance, 3),
            "sdp_violations": len(sdp_data),
            "zone_distribution": zone_distribution,
        }

        return {
            "packages": sorted(packages_data, key=lambda x: -x["distance"]),
            "sdp_violations": sdp_data,
            "propagation": propagation_data,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        zones = stats.get("zone_distribution", {})
        violations = data.get("sdp_violations", [])

        zone_str = ", ".join(f"{z}:{c}" for z, c in zones.items() if c > 0)

        return (
            f"Coupling: {stats.get('total_packages', 0)} packages. "
            f"Avg I={stats.get('avg_instability', 0):.2f}, "
            f"A={stats.get('avg_abstractness', 0):.2f}, "
            f"D={stats.get('avg_distance', 0):.2f}. "
            f"Zones: {zone_str}. "
            f"SDP violations: {len(violations)}."
        )
