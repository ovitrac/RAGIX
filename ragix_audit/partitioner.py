"""
Codebase Partitioner — Classify classes into logical applications

Partitions a Java codebase into:
- APP_A, APP_B, ... (configurable application identifiers)
- SHARED (used by multiple applications)
- UNKNOWN (not yet classified)
- DEAD_CODE (no incoming dependencies, not entry points)

Uses:
1. Fingerprint matching (package patterns, class name patterns, annotations)
2. Graph propagation (neighbor majority voting)
3. Evidence chains for traceability

Inspired by SIAS audit methodology (codebase_partition.py).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-11
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class AssignmentLabel(str, Enum):
    """Labels for class assignment."""
    UNKNOWN = "UNKNOWN"
    SHARED = "SHARED"
    DEAD_CODE = "DEAD_CODE"
    # Application labels are dynamic (APP_A, APP_B, etc.)


@dataclass
class Evidence:
    """
    A piece of evidence contributing to a class assignment.

    Attributes:
        method: Source of evidence (e.g., "fingerprint:package", "graph:neighbor_majority")
        weight: Contribution weight in [0, 1]
        details: Human-readable explanation
    """
    method: str
    weight: float
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "weight": round(self.weight, 3),
            "details": self.details
        }


@dataclass
class ClassAssignment:
    """
    Final assignment for a class.

    Attributes:
        label: Assignment label (APP_A, APP_B, SHARED, UNKNOWN, DEAD_CODE)
        confidence: Aggregated confidence in [0, 1]
        evidence: List of evidence objects
        votes: Raw vote counts per label (for debugging)
    """
    label: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    votes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "votes": {k: round(v, 3) for k, v in self.votes.items()}
        }


@dataclass
class ApplicationFingerprint:
    """
    Fingerprint patterns for detecting an application.

    Attributes:
        app_id: Application identifier (e.g., "SIAS", "TICC", "APP_A")
        package_patterns: Package name patterns (glob-style or regex)
        class_patterns: Class name patterns (glob-style)
        annotation_patterns: Annotation patterns to match
        keyword_patterns: Keywords in class/method names
        entry_point_patterns: Entry point class patterns (controllers, main, etc.)
    """
    app_id: str
    package_patterns: List[str] = field(default_factory=list)
    class_patterns: List[str] = field(default_factory=list)
    annotation_patterns: List[str] = field(default_factory=list)
    keyword_patterns: List[str] = field(default_factory=list)
    entry_point_patterns: List[str] = field(default_factory=list)
    color: str = "#3498db"  # Default blue

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_id": self.app_id,
            "package_patterns": self.package_patterns,
            "class_patterns": self.class_patterns,
            "annotation_patterns": self.annotation_patterns,
            "keyword_patterns": self.keyword_patterns,
            "entry_point_patterns": self.entry_point_patterns,
            "color": self.color
        }


@dataclass
class PartitionConfig:
    """
    Configuration for codebase partitioning.
    """
    applications: List[ApplicationFingerprint] = field(default_factory=list)
    shared_patterns: List[str] = field(default_factory=lambda: [
        "*common*", "*util*", "*shared*", "*core*", "*base*",
        "*dto*", "*entity*", "*model*", "*mapper*", "*config*"
    ])
    dead_code_threshold: float = 0.0  # Min incoming deps to not be dead
    propagation_iterations: int = 5
    confidence_threshold: float = 0.6

    @classmethod
    def default_two_apps(cls, app_a: str = "APP_A", app_b: str = "APP_B") -> "PartitionConfig":
        """Create default config for two-application separation."""
        return cls(
            applications=[
                ApplicationFingerprint(
                    app_id=app_a,
                    package_patterns=[],
                    class_patterns=[],
                    color="#3498db"  # Blue
                ),
                ApplicationFingerprint(
                    app_id=app_b,
                    package_patterns=[],
                    class_patterns=[],
                    color="#e74c3c"  # Red
                )
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applications": [a.to_dict() for a in self.applications],
            "shared_patterns": self.shared_patterns,
            "dead_code_threshold": self.dead_code_threshold,
            "propagation_iterations": self.propagation_iterations,
            "confidence_threshold": self.confidence_threshold
        }


@dataclass
class ClassInfo:
    """
    Information about a class for partitioning.
    """
    fqn: str  # Fully qualified name
    package: str
    class_name: str
    file_path: str
    annotations: List[str] = field(default_factory=list)
    is_interface: bool = False
    is_abstract: bool = False
    is_enum: bool = False
    methods: List[str] = field(default_factory=list)
    loc: int = 0


@dataclass
class PartitionResult:
    """
    Result of codebase partitioning.
    """
    assignments: Dict[str, ClassAssignment] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)
    config: Optional[PartitionConfig] = None

    # For visualization
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": {k: v.to_dict() for k, v in self.assignments.items()},
            "summary": self.summary,
            "config": self.config.to_dict() if self.config else None,
            "nodes": self.nodes,
            "edges": self.edges
        }

    def get_classes_by_label(self, label: str) -> List[str]:
        """Get all classes with a specific label."""
        return [fqn for fqn, a in self.assignments.items() if a.label == label]


# =============================================================================
# Partitioner Engine
# =============================================================================

class CodebasePartitioner:
    """
    Partitions a codebase into logical applications.
    """

    def __init__(self, config: Optional[PartitionConfig] = None):
        self.config = config or PartitionConfig()
        self.classes: Dict[str, ClassInfo] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)  # from -> {to}
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)  # to -> {from}
        self.assignments: Dict[str, ClassAssignment] = {}

    def add_class(self, class_info: ClassInfo):
        """Add a class to the partitioner."""
        self.classes[class_info.fqn] = class_info

    def add_dependency(self, from_fqn: str, to_fqn: str):
        """Add a dependency edge."""
        if from_fqn != to_fqn:
            self.dependencies[from_fqn].add(to_fqn)
            self.reverse_deps[to_fqn].add(from_fqn)

    def load_from_graph(self, graph: Dict[str, Any]):
        """
        Load classes and dependencies from a D3-format graph.

        Args:
            graph: Dict with "nodes" and "links" keys
        """
        nodes = graph.get("nodes", [])
        links = graph.get("links", [])

        # Build node index
        node_by_id = {}
        for node in nodes:
            node_id = node.get("id", "")
            if not node_id:
                continue

            # Parse FQN
            parts = node_id.rsplit(".", 1)
            package = parts[0] if len(parts) > 1 else ""
            class_name = parts[-1]

            class_info = ClassInfo(
                fqn=node_id,
                package=package,
                class_name=class_name,
                file_path=node.get("file", ""),
                is_interface=node.get("type") == "interface",
                is_abstract="abstract" in node.get("modifiers", []),
                is_enum=node.get("type") == "enum",
                loc=node.get("loc", 0)
            )
            self.add_class(class_info)
            node_by_id[node_id] = class_info

        # Add dependencies
        for link in links:
            source = link.get("source")
            target = link.get("target")

            # Handle index or string references
            if isinstance(source, int) and source < len(nodes):
                source = nodes[source].get("id", "")
            if isinstance(target, int) and target < len(nodes):
                target = nodes[target].get("id", "")

            if source and target:
                self.add_dependency(str(source), str(target))

    def _match_pattern(self, value: str, pattern: str) -> bool:
        """Match a value against a glob-style pattern."""
        # Convert glob to regex-friendly pattern
        pattern_lower = pattern.lower()
        value_lower = value.lower()

        # Direct glob match
        if fnmatch(value_lower, pattern_lower):
            return True

        # Also try as substring
        if pattern_lower.replace("*", "") in value_lower:
            return True

        return False

    def _fingerprint_match(self, class_info: ClassInfo) -> List[Tuple[str, Evidence]]:
        """
        Match a class against all application fingerprints.

        Returns:
            List of (app_id, Evidence) tuples
        """
        matches = []

        for app in self.config.applications:
            # Check package patterns
            for pattern in app.package_patterns:
                if self._match_pattern(class_info.package, pattern):
                    matches.append((app.app_id, Evidence(
                        method="fingerprint:package",
                        weight=0.8,
                        details=f"Package '{class_info.package}' matches pattern '{pattern}'"
                    )))
                    break

            # Check class name patterns
            for pattern in app.class_patterns:
                if self._match_pattern(class_info.class_name, pattern):
                    matches.append((app.app_id, Evidence(
                        method="fingerprint:class_name",
                        weight=0.7,
                        details=f"Class '{class_info.class_name}' matches pattern '{pattern}'"
                    )))
                    break

            # Check keyword patterns
            for keyword in app.keyword_patterns:
                if keyword.lower() in class_info.fqn.lower():
                    matches.append((app.app_id, Evidence(
                        method="fingerprint:keyword",
                        weight=0.6,
                        details=f"FQN contains keyword '{keyword}'"
                    )))
                    break

            # Check annotation patterns
            for pattern in app.annotation_patterns:
                for annotation in class_info.annotations:
                    if self._match_pattern(annotation, pattern):
                        matches.append((app.app_id, Evidence(
                            method="fingerprint:annotation",
                            weight=0.9,
                            details=f"Annotation '{annotation}' matches pattern '{pattern}'"
                        )))
                        break

        # Check shared patterns
        for pattern in self.config.shared_patterns:
            if self._match_pattern(class_info.package, pattern) or \
               self._match_pattern(class_info.class_name, pattern):
                matches.append(("SHARED", Evidence(
                    method="fingerprint:shared_pattern",
                    weight=0.5,
                    details=f"Matches shared pattern '{pattern}'"
                )))
                break

        return matches

    def _detect_dead_code(self) -> Set[str]:
        """
        Detect dead code candidates.

        Dead code = classes with no incoming dependencies and not entry points.
        """
        dead = set()

        for fqn, class_info in self.classes.items():
            # Has incoming dependencies?
            incoming = len(self.reverse_deps.get(fqn, set()))
            if incoming > self.config.dead_code_threshold:
                continue

            # Is it an entry point? (controllers, main classes, etc.)
            is_entry = False
            for app in self.config.applications:
                for pattern in app.entry_point_patterns:
                    if self._match_pattern(class_info.class_name, pattern):
                        is_entry = True
                        break

            # Common entry point patterns
            entry_patterns = ["*Controller", "*Main", "*Application", "*Servlet", "*Filter"]
            for pattern in entry_patterns:
                if self._match_pattern(class_info.class_name, pattern):
                    is_entry = True
                    break

            if not is_entry:
                dead.add(fqn)

        return dead

    def _propagate_labels(self):
        """
        Propagate labels using neighbor majority voting.

        Classes inherit the majority label of their neighbors.
        """
        for iteration in range(self.config.propagation_iterations):
            changes = 0

            for fqn in self.classes:
                assignment = self.assignments.get(fqn)
                if not assignment or assignment.label not in ("UNKNOWN",):
                    continue

                # Collect neighbor labels
                neighbors = set()
                neighbors.update(self.dependencies.get(fqn, set()))
                neighbors.update(self.reverse_deps.get(fqn, set()))

                if not neighbors:
                    continue

                # Count votes
                votes = defaultdict(float)
                for neighbor in neighbors:
                    neighbor_assignment = self.assignments.get(neighbor)
                    if neighbor_assignment and neighbor_assignment.label not in ("UNKNOWN", "DEAD_CODE"):
                        votes[neighbor_assignment.label] += neighbor_assignment.confidence

                if not votes:
                    continue

                # Find winner
                winner = max(votes.keys(), key=lambda k: votes[k])
                total_votes = sum(votes.values())
                confidence = votes[winner] / total_votes if total_votes > 0 else 0

                if confidence >= self.config.confidence_threshold:
                    assignment.label = winner
                    assignment.confidence = confidence
                    assignment.votes = dict(votes)
                    assignment.evidence.append(Evidence(
                        method="graph:neighbor_majority",
                        weight=confidence,
                        details=f"Iteration {iteration+1}: {len(neighbors)} neighbors voted {winner} ({confidence:.1%})"
                    ))
                    changes += 1

            logger.debug(f"Propagation iteration {iteration+1}: {changes} changes")
            if changes == 0:
                break

    def _detect_shared(self):
        """
        Detect SHARED classes - used by multiple applications.
        """
        app_ids = {a.app_id for a in self.config.applications}

        for fqn, assignment in self.assignments.items():
            if assignment.label in app_ids:
                # Check if this class is used by multiple apps
                dependents = self.reverse_deps.get(fqn, set())
                dependent_apps = set()

                for dep in dependents:
                    dep_assignment = self.assignments.get(dep)
                    if dep_assignment and dep_assignment.label in app_ids:
                        dependent_apps.add(dep_assignment.label)

                if len(dependent_apps) > 1:
                    assignment.label = "SHARED"
                    assignment.evidence.append(Evidence(
                        method="analysis:multi_app_dependency",
                        weight=0.9,
                        details=f"Used by multiple apps: {', '.join(dependent_apps)}"
                    ))

    def partition(self) -> PartitionResult:
        """
        Run the full partitioning algorithm.

        Returns:
            PartitionResult with assignments and visualization data
        """
        # Initialize all as UNKNOWN
        for fqn in self.classes:
            self.assignments[fqn] = ClassAssignment(
                label="UNKNOWN",
                confidence=0.0
            )

        # Step 1: Detect dead code
        dead_code = self._detect_dead_code()
        for fqn in dead_code:
            self.assignments[fqn] = ClassAssignment(
                label="DEAD_CODE",
                confidence=0.8,
                evidence=[Evidence(
                    method="analysis:dead_code",
                    weight=0.8,
                    details="No incoming dependencies and not an entry point"
                )]
            )

        # Step 2: Fingerprint matching
        for fqn, class_info in self.classes.items():
            if self.assignments[fqn].label == "DEAD_CODE":
                continue

            matches = self._fingerprint_match(class_info)
            if matches:
                # Aggregate votes
                votes = defaultdict(float)
                evidence = []
                for app_id, ev in matches:
                    votes[app_id] += ev.weight
                    evidence.append(ev)

                # Winner takes all
                winner = max(votes.keys(), key=lambda k: votes[k])
                total = sum(votes.values())
                confidence = votes[winner] / total if total > 0 else 0

                self.assignments[fqn] = ClassAssignment(
                    label=winner,
                    confidence=confidence,
                    evidence=evidence,
                    votes=dict(votes)
                )

        # Step 3: Propagate labels through graph
        self._propagate_labels()

        # Step 4: Detect shared classes
        self._detect_shared()

        # Build result
        result = PartitionResult(
            assignments=self.assignments,
            config=self.config
        )

        # Compute summary
        summary = defaultdict(int)
        for assignment in self.assignments.values():
            summary[assignment.label] += 1
        result.summary = dict(summary)

        # Build visualization data
        result.nodes = self._build_vis_nodes()
        result.edges = self._build_vis_edges()

        return result

    def _build_vis_nodes(self) -> List[Dict[str, Any]]:
        """Build visualization nodes."""
        # Color map
        colors = {
            "UNKNOWN": "#95a5a6",    # Gray
            "SHARED": "#9b59b6",     # Purple
            "DEAD_CODE": "#2c3e50",  # Dark
        }
        for app in self.config.applications:
            colors[app.app_id] = app.color

        nodes = []
        for fqn, class_info in self.classes.items():
            assignment = self.assignments.get(fqn)
            label = assignment.label if assignment else "UNKNOWN"

            nodes.append({
                "id": fqn,
                "label": class_info.class_name,
                "package": class_info.package,
                "file": class_info.file_path,
                "loc": class_info.loc,
                "partition": label,
                "confidence": assignment.confidence if assignment else 0,
                "color": colors.get(label, "#95a5a6"),
                "size": max(5, min(30, class_info.loc / 20))  # Size by LOC
            })

        return nodes

    def _build_vis_edges(self) -> List[Dict[str, Any]]:
        """Build visualization edges."""
        edges = []
        for source, targets in self.dependencies.items():
            source_assignment = self.assignments.get(source)
            source_label = source_assignment.label if source_assignment else "UNKNOWN"

            for target in targets:
                target_assignment = self.assignments.get(target)
                target_label = target_assignment.label if target_assignment else "UNKNOWN"

                # Cross-partition edges are important
                is_cross = source_label != target_label and \
                           source_label not in ("UNKNOWN", "DEAD_CODE") and \
                           target_label not in ("UNKNOWN", "DEAD_CODE")

                edges.append({
                    "source": source,
                    "target": target,
                    "cross_partition": is_cross,
                    "source_partition": source_label,
                    "target_partition": target_label
                })

        return edges


# =============================================================================
# Convenience Functions
# =============================================================================

def partition_from_graph(
    graph: Dict[str, Any],
    config: Optional[PartitionConfig] = None
) -> PartitionResult:
    """
    Partition a codebase from a D3-format dependency graph.

    Args:
        graph: D3 graph with "nodes" and "links"
        config: Optional partition configuration

    Returns:
        PartitionResult
    """
    partitioner = CodebasePartitioner(config)
    partitioner.load_from_graph(graph)
    return partitioner.partition()


def create_sias_ticc_config() -> PartitionConfig:
    """
    Create configuration for SIAS/TICC separation (based on GRDF audit).
    """
    return PartitionConfig(
        applications=[
            ApplicationFingerprint(
                app_id="SIAS",
                package_patterns=["*sias*", "*supervision*", "*accompagnement*", "*pre*"],
                class_patterns=["*Sias*", "*Supervision*", "*Pre*"],
                keyword_patterns=["sias", "supervision", "accompagnement", "pre"],
                entry_point_patterns=["*SiasController*", "*PreController*"],
                color="#3498db"  # Blue
            ),
            ApplicationFingerprint(
                app_id="TICC",
                package_patterns=["*ticc*", "*telecommande*", "*controle*", "*hab*"],
                class_patterns=["*Ticc*", "*TeleCommande*", "*Hab*"],
                keyword_patterns=["ticc", "telecommande", "controle", "hab"],
                entry_point_patterns=["*TiccController*", "*HabController*"],
                color="#e74c3c"  # Red
            )
        ],
        shared_patterns=[
            "*common*", "*util*", "*shared*", "*core*", "*base*",
            "*transverse*", "*dto*", "*entity*", "*model*", "*mapper*",
            "*config*", "*exception*", "*constant*"
        ],
        propagation_iterations=10,
        confidence_threshold=0.5
    )
