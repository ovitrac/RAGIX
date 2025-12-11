"""
Coupling Module — Martin's Instability/Abstractness Metrics

Implements Robert C. Martin's package coupling metrics:
- Ca (Afferent Coupling): Incoming dependencies
- Ce (Efferent Coupling): Outgoing dependencies
- I (Instability): Ce / (Ca + Ce)
- A (Abstractness): #interfaces / #classes
- D (Distance from Main Sequence): |A + I - 1|

Reference: Martin, R.C. (2002). "Agile Software Development, Principles, Patterns, and Practices"

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-11
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Zones in the A-I plane."""
    MAIN_SEQUENCE = "main_sequence"  # D ≈ 0, balanced design
    ZONE_OF_PAIN = "zone_of_pain"     # A ≈ 0, I ≈ 0, rigid and hard to extend
    ZONE_OF_USELESSNESS = "zone_of_uselessness"  # A ≈ 1, I ≈ 1, unused abstractions
    BALANCED = "balanced"  # Acceptable distance from main sequence


@dataclass
class PackageCoupling:
    """
    Coupling metrics for a single package.
    """
    package: str

    # Raw coupling counts
    ca: int = 0              # Afferent coupling (incoming dependencies)
    ce: int = 0              # Efferent coupling (outgoing dependencies)
    internal: int = 0        # Internal dependencies (within package)

    # Class counts for abstractness
    total_classes: int = 0
    abstract_classes: int = 0  # Abstract classes + interfaces
    interfaces: int = 0
    concrete_classes: int = 0

    # Computed metrics
    instability: float = 0.0      # I = Ce / (Ca + Ce)
    abstractness: float = 0.0     # A = abstract_classes / total_classes
    distance: float = 0.0         # D = |A + I - 1|

    # Zone classification
    zone: ZoneType = ZoneType.BALANCED
    zone_label: str = ""

    # Detailed dependency lists (optional)
    dependents: List[str] = field(default_factory=list)   # Packages that depend on this
    dependencies: List[str] = field(default_factory=list)  # Packages this depends on

    def compute_metrics(self):
        """Compute derived metrics from raw counts."""
        # Instability
        total_coupling = self.ca + self.ce
        if total_coupling > 0:
            self.instability = self.ce / total_coupling
        else:
            self.instability = 0.0  # No dependencies = maximally stable

        # Abstractness
        if self.total_classes > 0:
            self.abstractness = self.abstract_classes / self.total_classes
        else:
            self.abstractness = 0.0

        # Distance from main sequence
        self.distance = abs(self.abstractness + self.instability - 1)

        # Zone classification
        self._classify_zone()

    def _classify_zone(self):
        """Classify package into A-I plane zone."""
        # Zone of Pain: low A, low I (concrete and stable = rigid)
        if self.abstractness < 0.2 and self.instability < 0.2:
            self.zone = ZoneType.ZONE_OF_PAIN
            self.zone_label = "Zone of Pain (rigid, hard to extend)"

        # Zone of Uselessness: high A, high I (abstract and unstable)
        elif self.abstractness > 0.8 and self.instability > 0.8:
            self.zone = ZoneType.ZONE_OF_USELESSNESS
            self.zone_label = "Zone of Uselessness (unused abstractions)"

        # Main Sequence: D ≈ 0
        elif self.distance < 0.1:
            self.zone = ZoneType.MAIN_SEQUENCE
            self.zone_label = "Main Sequence (optimal balance)"

        # Balanced: acceptable distance
        else:
            self.zone = ZoneType.BALANCED
            self.zone_label = f"Balanced (D={self.distance:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "package": self.package,
            "ca": self.ca,
            "ce": self.ce,
            "internal": self.internal,
            "total_coupling": self.ca + self.ce,
            "total_classes": self.total_classes,
            "abstract_classes": self.abstract_classes,
            "interfaces": self.interfaces,
            "concrete_classes": self.concrete_classes,
            "instability": round(self.instability, 4),
            "abstractness": round(self.abstractness, 4),
            "distance": round(self.distance, 4),
            "zone": self.zone.value,
            "zone_label": self.zone_label,
        }


@dataclass
class SDPViolation:
    """
    Stable Dependencies Principle violation.

    A package should only depend on packages that are more stable than itself.
    Violation: Package with I=0.2 depends on package with I=0.8
    """
    source_package: str
    source_instability: float
    target_package: str
    target_instability: float
    delta: float  # How much less stable the target is
    severity: str  # "low", "medium", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_package,
            "source_I": round(self.source_instability, 3),
            "target": self.target_package,
            "target_I": round(self.target_instability, 3),
            "delta": round(self.delta, 3),
            "severity": self.severity,
        }


@dataclass
class CouplingAnalysis:
    """
    Complete coupling analysis for a codebase.
    """
    packages: Dict[str, PackageCoupling] = field(default_factory=dict)

    # Summary statistics
    total_packages: int = 0
    avg_instability: float = 0.0
    avg_abstractness: float = 0.0
    avg_distance: float = 0.0

    # Zone distribution
    packages_in_pain: int = 0
    packages_useless: int = 0
    packages_on_sequence: int = 0
    packages_balanced: int = 0

    # SDP violations
    sdp_violations: List[SDPViolation] = field(default_factory=list)

    # Top offenders
    highest_instability: List[str] = field(default_factory=list)
    highest_distance: List[str] = field(default_factory=list)
    most_coupled: List[str] = field(default_factory=list)  # Highest Ca + Ce

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_packages": self.total_packages,
            "averages": {
                "instability": round(self.avg_instability, 4),
                "abstractness": round(self.avg_abstractness, 4),
                "distance": round(self.avg_distance, 4),
            },
            "zone_distribution": {
                "pain": self.packages_in_pain,
                "useless": self.packages_useless,
                "main_sequence": self.packages_on_sequence,
                "balanced": self.packages_balanced,
            },
            "sdp_violations_count": len(self.sdp_violations),
            "sdp_violations": [v.to_dict() for v in self.sdp_violations[:20]],
            "top_offenders": {
                "highest_instability": self.highest_instability[:10],
                "highest_distance": self.highest_distance[:10],
                "most_coupled": self.most_coupled[:10],
            },
            "packages": {k: v.to_dict() for k, v in self.packages.items()},
        }


class CouplingComputer:
    """
    Computes coupling metrics from dependency graph.
    """

    def __init__(self):
        self.analysis: Optional[CouplingAnalysis] = None

    def compute_from_graph(
        self,
        dependencies: Dict[str, Set[str]],  # {source: {targets}}
        package_classes: Optional[Dict[str, Dict[str, Any]]] = None,  # {package: {class_info}}
    ) -> CouplingAnalysis:
        """
        Compute coupling metrics from dependency graph.

        Args:
            dependencies: Dictionary of {source_package: {target_packages}}
            package_classes: Optional class information per package
                {package: {"total": N, "abstract": M, "interfaces": K}}

        Returns:
            CouplingAnalysis with all metrics computed
        """
        analysis = CouplingAnalysis()

        # Collect all packages
        all_packages: Set[str] = set()
        for source, targets in dependencies.items():
            all_packages.add(source)
            all_packages.update(targets)

        # Initialize package metrics
        for pkg in all_packages:
            analysis.packages[pkg] = PackageCoupling(package=pkg)

        # Compute Ca and Ce
        for source, targets in dependencies.items():
            if source not in analysis.packages:
                continue

            for target in targets:
                if target == source:
                    # Internal dependency
                    analysis.packages[source].internal += 1
                else:
                    # External dependency
                    analysis.packages[source].ce += 1
                    analysis.packages[source].dependencies.append(target)

                    if target in analysis.packages:
                        analysis.packages[target].ca += 1
                        analysis.packages[target].dependents.append(source)

        # Add class information if available
        if package_classes:
            for pkg, class_info in package_classes.items():
                if pkg in analysis.packages:
                    p = analysis.packages[pkg]
                    p.total_classes = class_info.get("total", 0)
                    p.abstract_classes = class_info.get("abstract", 0)
                    p.interfaces = class_info.get("interfaces", 0)
                    p.concrete_classes = p.total_classes - p.abstract_classes

        # Compute derived metrics for each package
        for pkg in analysis.packages.values():
            pkg.compute_metrics()

        # Compute summary statistics
        self._compute_summary(analysis)

        # Detect SDP violations
        self._detect_sdp_violations(analysis)

        # Find top offenders
        self._find_top_offenders(analysis)

        self.analysis = analysis
        return analysis

    def _compute_summary(self, analysis: CouplingAnalysis):
        """Compute summary statistics."""
        packages = list(analysis.packages.values())
        analysis.total_packages = len(packages)

        if not packages:
            return

        # Averages
        analysis.avg_instability = sum(p.instability for p in packages) / len(packages)
        analysis.avg_abstractness = sum(p.abstractness for p in packages) / len(packages)
        analysis.avg_distance = sum(p.distance for p in packages) / len(packages)

        # Zone distribution
        for pkg in packages:
            if pkg.zone == ZoneType.ZONE_OF_PAIN:
                analysis.packages_in_pain += 1
            elif pkg.zone == ZoneType.ZONE_OF_USELESSNESS:
                analysis.packages_useless += 1
            elif pkg.zone == ZoneType.MAIN_SEQUENCE:
                analysis.packages_on_sequence += 1
            else:
                analysis.packages_balanced += 1

    def _detect_sdp_violations(self, analysis: CouplingAnalysis):
        """
        Detect Stable Dependencies Principle violations.

        SDP: Depend in the direction of stability.
        Violation: A stable package depending on an unstable one.
        """
        for pkg in analysis.packages.values():
            for dep in pkg.dependencies:
                if dep not in analysis.packages:
                    continue

                target = analysis.packages[dep]

                # Violation if source is more stable than target
                # (lower I depends on higher I)
                if pkg.instability < target.instability:
                    delta = target.instability - pkg.instability

                    # Severity based on delta
                    if delta > 0.5:
                        severity = "high"
                    elif delta > 0.25:
                        severity = "medium"
                    else:
                        severity = "low"

                    violation = SDPViolation(
                        source_package=pkg.package,
                        source_instability=pkg.instability,
                        target_package=dep,
                        target_instability=target.instability,
                        delta=delta,
                        severity=severity,
                    )
                    analysis.sdp_violations.append(violation)

        # Sort by severity and delta
        analysis.sdp_violations.sort(
            key=lambda v: (
                {"high": 0, "medium": 1, "low": 2}[v.severity],
                -v.delta
            )
        )

    def _find_top_offenders(self, analysis: CouplingAnalysis):
        """Find packages with worst metrics."""
        packages = list(analysis.packages.values())

        # Highest instability (most unstable)
        by_instability = sorted(packages, key=lambda p: p.instability, reverse=True)
        analysis.highest_instability = [p.package for p in by_instability[:10]]

        # Highest distance from main sequence
        by_distance = sorted(packages, key=lambda p: p.distance, reverse=True)
        analysis.highest_distance = [p.package for p in by_distance[:10]]

        # Most coupled (highest Ca + Ce)
        by_coupling = sorted(packages, key=lambda p: p.ca + p.ce, reverse=True)
        analysis.most_coupled = [p.package for p in by_coupling[:10]]


def compute_propagation_factor(
    dependencies: Dict[str, Set[str]],
    node: str
) -> float:
    """
    Compute Propagation Factor: what fraction of the graph is affected by changes to this node?

    PF(n) = |reachable_downstream(n)| / |V|

    Args:
        dependencies: {source: {targets}} where targets depend on source
        node: Node to compute PF for

    Returns:
        Propagation factor (0-1)
    """
    if node not in dependencies and not any(node in targets for targets in dependencies.values()):
        return 0.0

    # Build reverse graph (who depends on whom)
    reverse_deps: Dict[str, Set[str]] = {}
    all_nodes: Set[str] = set()

    for source, targets in dependencies.items():
        all_nodes.add(source)
        for target in targets:
            all_nodes.add(target)
            if source not in reverse_deps:
                reverse_deps[source] = set()
            reverse_deps[source].add(target)

    if not all_nodes:
        return 0.0

    # BFS to find all downstream nodes
    from collections import deque

    reachable: Set[str] = set()
    queue = deque([node])

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)

        # Find nodes that depend on current
        if current in reverse_deps:
            for dependent in reverse_deps[current]:
                if dependent not in reachable:
                    queue.append(dependent)

    # Exclude the node itself
    reachable.discard(node)

    return len(reachable) / len(all_nodes)


@dataclass
class PropagationAnalysis:
    """
    Propagation impact analysis for all nodes.
    """
    node_pf: Dict[str, float] = field(default_factory=dict)

    # Summary
    avg_pf: float = 0.0
    max_pf: float = 0.0
    max_pf_node: str = ""

    # Risk categories
    critical_nodes: List[str] = field(default_factory=list)   # PF > 0.3
    high_impact_nodes: List[str] = field(default_factory=list)  # 0.1 < PF <= 0.3
    normal_nodes: List[str] = field(default_factory=list)      # PF <= 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_pf": round(self.avg_pf, 4),
            "max_pf": round(self.max_pf, 4),
            "max_pf_node": self.max_pf_node,
            "critical_count": len(self.critical_nodes),
            "high_impact_count": len(self.high_impact_nodes),
            "normal_count": len(self.normal_nodes),
            "critical_nodes": self.critical_nodes[:20],
            "high_impact_nodes": self.high_impact_nodes[:20],
        }


def compute_all_propagation_factors(
    dependencies: Dict[str, Set[str]]
) -> PropagationAnalysis:
    """
    Compute propagation factors for all nodes.

    Args:
        dependencies: {source: {targets}}

    Returns:
        PropagationAnalysis with all PF values
    """
    analysis = PropagationAnalysis()

    # Get all nodes
    all_nodes: Set[str] = set()
    for source, targets in dependencies.items():
        all_nodes.add(source)
        all_nodes.update(targets)

    if not all_nodes:
        return analysis

    # Compute PF for each node
    for node in all_nodes:
        pf = compute_propagation_factor(dependencies, node)
        analysis.node_pf[node] = pf

        # Categorize
        if pf > 0.3:
            analysis.critical_nodes.append(node)
        elif pf > 0.1:
            analysis.high_impact_nodes.append(node)
        else:
            analysis.normal_nodes.append(node)

    # Summary stats
    if analysis.node_pf:
        pf_values = list(analysis.node_pf.values())
        analysis.avg_pf = sum(pf_values) / len(pf_values)
        analysis.max_pf = max(pf_values)
        analysis.max_pf_node = max(analysis.node_pf, key=analysis.node_pf.get)

    # Sort by PF descending
    analysis.critical_nodes.sort(key=lambda n: analysis.node_pf.get(n, 0), reverse=True)
    analysis.high_impact_nodes.sort(key=lambda n: analysis.node_pf.get(n, 0), reverse=True)

    return analysis
