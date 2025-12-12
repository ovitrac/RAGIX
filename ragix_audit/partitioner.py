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
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

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

    Graph Propagation Algorithm Parameters:
    ----------------------------------------
    The algorithm uses multi-phase directional propagation to classify classes:

    Phase 1 (Core): High-confidence fingerprint matches only
    Phase 2 (Near): Propagate from cores using weighted neighbor voting
    Phase 3 (Far):  Fill remaining UNKNOWN with lower threshold

    Directional Weighting:
    - forward_weight: Weight for outgoing deps (what I import) - stronger signal
    - reverse_weight: Weight for incoming deps (who imports me) - weaker signal

    Package Cohesion:
    - package_cohesion_bonus: Extra weight for same-package neighbors

    Confidence Decay:
    - confidence_decay: Reduction per propagation hop (prevents over-propagation)

    Multi-Phase Thresholds:
    - phase1_threshold: Core identification (fingerprints only)
    - phase2_threshold: Near propagation
    - phase3_threshold: Far propagation (fill remaining)
    """
    applications: List[ApplicationFingerprint] = field(default_factory=list)
    shared_patterns: List[str] = field(default_factory=lambda: [
        "*common*", "*util*", "*shared*", "*core*", "*base*",
        "*dto*", "*entity*", "*model*", "*mapper*", "*config*"
    ])
    dead_code_threshold: float = 0.0  # Min incoming deps to not be dead
    propagation_iterations: int = 10  # Max iterations per phase

    # Legacy parameter (kept for compatibility, use phase thresholds instead)
    confidence_threshold: float = 0.6

    # === Graph Propagation Algorithm Parameters ===

    # Directional weighting: forward deps (imports) vs reverse deps (callers)
    forward_weight: float = 0.7   # Weight for classes I depend on (strong signal)
    reverse_weight: float = 0.3   # Weight for classes that depend on me (weaker)

    # Package cohesion: same-package neighbors get bonus weight
    package_cohesion_bonus: float = 0.2  # Added to neighbor weight if same package

    # Confidence decay per propagation hop
    confidence_decay: float = 0.05  # Subtracted from confidence each iteration

    # Multi-phase propagation thresholds
    phase1_threshold: float = 0.8  # Core identification (high confidence)
    phase2_threshold: float = 0.6  # Near propagation (medium confidence)
    phase3_threshold: float = 0.4  # Far propagation (low confidence, fill gaps)

    # Convergence settings
    convergence_threshold: float = 0.01  # Stop when < 1% nodes change
    min_votes_required: int = 2  # Minimum neighbor votes to assign label

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
            "confidence_threshold": self.confidence_threshold,
            # Graph Propagation Algorithm parameters
            "forward_weight": self.forward_weight,
            "reverse_weight": self.reverse_weight,
            "package_cohesion_bonus": self.package_cohesion_bonus,
            "confidence_decay": self.confidence_decay,
            "phase1_threshold": self.phase1_threshold,
            "phase2_threshold": self.phase2_threshold,
            "phase3_threshold": self.phase3_threshold,
            "convergence_threshold": self.convergence_threshold,
            "min_votes_required": self.min_votes_required,
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
                annotations=node.get("annotations", []),
                is_interface=node.get("type") == "interface",
                is_abstract="abstract" in node.get("modifiers", []),
                is_enum=node.get("type") == "enum",
                methods=node.get("methods", []),
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

        Dead code = classes that are COMPLETELY ISOLATED:
        - No incoming dependencies (nobody calls them)
        - No outgoing dependencies (they don't call anything)
        - Not an entry point pattern

        A class that has outgoing deps but no incoming is likely still used
        via reflection (Spring DI, JPA, etc.) - those are NOT dead code.
        """
        dead = set()

        # Entry point patterns - these are NEVER dead code
        entry_class_patterns = [
            # Web/REST
            "*Controller", "*RestController", "*Resource", "*Endpoint",
            "*Servlet", "*Filter", "*Handler",
            # Application entry
            "*Main", "*Application", "*Bootstrap", "*Launcher", "*Runner",
            # Test classes
            "*Test", "*Tests", "*TestCase", "*Spec", "*IT",
        ]

        for fqn, class_info in self.classes.items():
            incoming = len(self.reverse_deps.get(fqn, set()))
            outgoing = len(self.dependencies.get(fqn, set()))

            # Has any connections? -> Not dead
            if incoming > self.config.dead_code_threshold or outgoing > 0:
                continue

            # Completely isolated - check if it's an entry point
            is_entry = False

            # 1. Check application-specific entry points
            for app in self.config.applications:
                for pattern in app.entry_point_patterns:
                    if self._match_pattern(class_info.class_name, pattern):
                        is_entry = True
                        break
                if is_entry:
                    break

            # 2. Check common entry point patterns
            if not is_entry:
                for pattern in entry_class_patterns:
                    if self._match_pattern(class_info.class_name, pattern):
                        is_entry = True
                        break

            # 3. Check annotations for entry points
            if not is_entry and hasattr(class_info, 'annotations') and class_info.annotations:
                entry_annotations = [
                    "*Controller", "*RestController", "*SpringBootApplication",
                    "*Test", "*RunWith", "*SpringBootTest",
                ]
                for ann in class_info.annotations:
                    for pattern in entry_annotations:
                        if self._match_pattern(ann, pattern):
                            is_entry = True
                            break
                    if is_entry:
                        break

            if not is_entry:
                dead.add(fqn)

        return dead

    def _propagate_labels(self):
        """
        Multi-phase directional graph propagation algorithm.

        This algorithm propagates partition labels through the dependency graph
        using weighted voting with the following features:

        1. DIRECTIONAL WEIGHTING:
           - Forward deps (what I import): weighted by config.forward_weight (default 0.7)
           - Reverse deps (who imports me): weighted by config.reverse_weight (default 0.3)
           Rationale: "What code I use" is a stronger signal of my partition than
           "who uses my code" (which could be shared utilities).

        2. PACKAGE COHESION:
           - Neighbors in the same Java package get a bonus weight
           - Controlled by config.package_cohesion_bonus (default 0.2)
           Rationale: Classes in the same package are architecturally related.

        3. MULTI-PHASE THRESHOLDS:
           - Phase 1: config.phase1_threshold (0.8) - only high-confidence assignments
           - Phase 2: config.phase2_threshold (0.6) - medium confidence propagation
           - Phase 3: config.phase3_threshold (0.4) - fill remaining gaps
           Rationale: Propagate from confident cores outward, reducing false positives.

        4. CONFIDENCE DECAY:
           - Each propagation iteration reduces max achievable confidence
           - Controlled by config.confidence_decay (default 0.05)
           Rationale: Distant nodes should have lower confidence than direct matches.

        5. CONVERGENCE DETECTION:
           - Stops when < convergence_threshold (1%) of nodes change
           - Prevents unnecessary iterations on stable graphs
        """
        total_nodes = len(self.classes)
        if total_nodes == 0:
            return

        # Multi-phase propagation with decreasing thresholds
        phases = [
            ("core", self.config.phase1_threshold),
            ("near", self.config.phase2_threshold),
            ("far", self.config.phase3_threshold),
        ]

        for phase_name, threshold in phases:
            self._run_propagation_phase(phase_name, threshold, total_nodes)

    def _run_propagation_phase(self, phase_name: str, threshold: float, total_nodes: int):
        """
        Run a single propagation phase with the given threshold.

        Args:
            phase_name: Name of the phase for logging/evidence
            threshold: Minimum confidence required to assign a label
            total_nodes: Total number of nodes (for convergence calculation)
        """
        for iteration in range(self.config.propagation_iterations):
            changes = 0

            # Apply confidence decay based on iteration
            iteration_decay = iteration * self.config.confidence_decay
            effective_max_confidence = max(0.5, 1.0 - iteration_decay)

            for fqn in self.classes:
                assignment = self.assignments.get(fqn)
                if not assignment or assignment.label != "UNKNOWN":
                    continue

                class_info = self.classes.get(fqn)
                if not class_info:
                    continue

                # Compute weighted votes from neighbors
                votes = self._compute_weighted_votes(fqn, class_info)

                if not votes:
                    continue

                # Check minimum votes requirement
                total_vote_count = sum(1 for v in votes.values() if v > 0)
                if total_vote_count < self.config.min_votes_required:
                    continue

                # Find winner
                winner = max(votes.keys(), key=lambda k: votes[k])
                total_weight = sum(votes.values())
                raw_confidence = votes[winner] / total_weight if total_weight > 0 else 0

                # Apply confidence decay
                confidence = min(raw_confidence, effective_max_confidence)

                if confidence >= threshold:
                    assignment.label = winner
                    assignment.confidence = confidence
                    assignment.votes = {k: round(v, 3) for k, v in votes.items()}
                    assignment.evidence.append(Evidence(
                        method=f"graph:propagation_{phase_name}",
                        weight=confidence,
                        details=f"Phase {phase_name}, iter {iteration+1}: "
                                f"{total_vote_count} voters → {winner} ({confidence:.1%})"
                    ))
                    changes += 1

            # Convergence check
            change_ratio = changes / total_nodes if total_nodes > 0 else 0
            logger.debug(f"Propagation {phase_name} iter {iteration+1}: "
                        f"{changes} changes ({change_ratio:.1%})")

            if changes == 0 or change_ratio < self.config.convergence_threshold:
                break

    def _compute_weighted_votes(self, fqn: str, class_info: ClassInfo) -> Dict[str, float]:
        """
        Compute weighted votes from neighbors using directional weighting and package cohesion.

        The voting weight for each neighbor is calculated as:
            weight = base_direction_weight × neighbor_confidence × (1 + package_bonus)

        Where:
            - base_direction_weight: forward_weight for imports, reverse_weight for callers
            - neighbor_confidence: the neighbor's classification confidence
            - package_bonus: package_cohesion_bonus if same package, else 0

        Args:
            fqn: Fully qualified name of the class to classify
            class_info: ClassInfo for the class

        Returns:
            Dictionary of {label: weighted_vote_total}
        """
        votes: Dict[str, float] = defaultdict(float)

        # Forward dependencies (classes I import) - stronger signal
        forward_deps = self.dependencies.get(fqn, set())
        for dep_fqn in forward_deps:
            self._add_neighbor_vote(
                votes, dep_fqn, class_info.package,
                self.config.forward_weight, "forward"
            )

        # Reverse dependencies (classes that import me) - weaker signal
        reverse_deps = self.reverse_deps.get(fqn, set())
        for dep_fqn in reverse_deps:
            self._add_neighbor_vote(
                votes, dep_fqn, class_info.package,
                self.config.reverse_weight, "reverse"
            )

        return dict(votes)

    def _add_neighbor_vote(
        self,
        votes: Dict[str, float],
        neighbor_fqn: str,
        my_package: str,
        direction_weight: float,
        direction: str
    ):
        """
        Add a neighbor's vote with appropriate weighting.

        Args:
            votes: Vote accumulator dictionary
            neighbor_fqn: FQN of the voting neighbor
            my_package: Package of the class being classified
            direction_weight: Base weight for this direction (forward/reverse)
            direction: "forward" or "reverse" (for debugging)
        """
        neighbor_assignment = self.assignments.get(neighbor_fqn)
        if not neighbor_assignment:
            return
        if neighbor_assignment.label in ("UNKNOWN", "DEAD_CODE"):
            return

        neighbor_info = self.classes.get(neighbor_fqn)
        neighbor_package = neighbor_info.package if neighbor_info else ""

        # Calculate vote weight
        weight = direction_weight * neighbor_assignment.confidence

        # Apply package cohesion bonus if same package
        if my_package and neighbor_package and my_package == neighbor_package:
            weight *= (1.0 + self.config.package_cohesion_bonus)

        votes[neighbor_assignment.label] += weight

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
                    weight=0.9,
                    details="Completely isolated: no callers and no callees"
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

        # Compute layout using MDS
        layout = self.compute_layout(width=800, height=600)

        # Build visualization data with precomputed positions
        result.nodes = self._build_vis_nodes(layout)
        result.edges = self._build_vis_edges()

        return result

    def _build_vis_nodes(self, layout: Dict[str, Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """Build visualization nodes with optional precomputed positions."""
        # Color map
        colors = {
            "UNKNOWN": "#95a5a6",    # Gray
            "SHARED": "#9b59b6",     # Purple
            "DEAD_CODE": "#e67e22",  # Orange (visible)
        }
        for app in self.config.applications:
            colors[app.app_id] = app.color

        layout = layout or {}
        nodes = []
        for fqn, class_info in self.classes.items():
            assignment = self.assignments.get(fqn)
            label = assignment.label if assignment else "UNKNOWN"
            pos = layout.get(fqn, (400, 300))  # Default center if no position

            nodes.append({
                "id": fqn,
                "label": class_info.class_name,
                "package": class_info.package,
                "file": class_info.file_path,
                "loc": class_info.loc,
                "partition": label,
                "confidence": assignment.confidence if assignment else 0,
                "color": colors.get(label, "#95a5a6"),
                "size": max(5, min(30, class_info.loc / 20)),  # Size by LOC
                "x": pos[0],  # Precomputed MDS position
                "y": pos[1],
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

    def compute_layout(self, width: int = 800, height: int = 600) -> Dict[str, Tuple[float, float]]:
        """
        Compute 2D layout using partition-based clustering.

        For large graphs (>500 nodes), uses a fast partition-based layout:
        - Groups nodes by partition
        - Arranges partitions in a grid or circle
        - Spreads nodes within each partition

        For smaller graphs, uses MDS for more accurate positioning.

        Args:
            width: Target width for layout
            height: Target height for layout

        Returns:
            Dictionary mapping node ID to (x, y) position
        """
        node_ids = list(self.classes.keys())
        n = len(node_ids)

        if n == 0:
            return {}

        if n == 1:
            return {node_ids[0]: (width / 2, height / 2)}

        # For large graphs, use fast partition-based layout
        if n > 500:
            return self._partition_based_layout(node_ids, width, height)

        # For smaller graphs, use MDS
        return self._mds_layout(node_ids, width, height)

    def _partition_based_layout(self, node_ids: List[str], width: int, height: int) -> Dict[str, Tuple[float, float]]:
        """
        Fast layout for large graphs using partition clustering.

        Arranges partitions in a circular layout, with nodes spread within each partition area.
        """
        # Group nodes by partition
        partition_groups: Dict[str, List[str]] = defaultdict(list)
        for nid in node_ids:
            assign = self.assignments.get(nid)
            label = assign.label if assign else "UNKNOWN"
            partition_groups[label].append(nid)

        # Sort partitions by size (largest first, but DEAD_CODE and UNKNOWN last)
        def partition_sort_key(label):
            if label == "DEAD_CODE":
                return (2, 0)
            if label == "UNKNOWN":
                return (1, 0)
            return (0, -len(partition_groups[label]))

        sorted_partitions = sorted(partition_groups.keys(), key=partition_sort_key)

        # Calculate partition centers in a circular arrangement
        n_partitions = len(sorted_partitions)
        center_x, center_y = width / 2, height / 2
        radius = min(width, height) * 0.35

        partition_centers = {}
        for i, label in enumerate(sorted_partitions):
            if n_partitions == 1:
                partition_centers[label] = (center_x, center_y)
            else:
                angle = 2 * math.pi * i / n_partitions - math.pi / 2
                px = center_x + radius * math.cos(angle)
                py = center_y + radius * math.sin(angle)
                partition_centers[label] = (px, py)

        # Spread nodes within each partition area
        layout = {}
        padding = 30

        for label, nodes in partition_groups.items():
            cx, cy = partition_centers[label]
            n_nodes = len(nodes)

            if n_nodes == 1:
                layout[nodes[0]] = (cx, cy)
                continue

            # Calculate spread radius based on node count
            # Larger partitions get more space
            spread = min(radius * 0.8, max(30, math.sqrt(n_nodes) * 8))

            # Arrange nodes in a spiral pattern within the partition
            for j, nid in enumerate(nodes):
                if n_nodes <= 20:
                    # Small partition: circular arrangement
                    angle = 2 * math.pi * j / n_nodes
                    r = spread * 0.6
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)
                else:
                    # Large partition: spiral arrangement
                    angle = j * 0.3  # Golden angle approximation
                    r = spread * math.sqrt(j / n_nodes)
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)

                # Clamp to canvas bounds
                x = max(padding, min(width - padding, x))
                y = max(padding, min(height - padding, y))
                layout[nid] = (x, y)

        return layout

    def _mds_layout(self, node_ids: List[str], width: int, height: int) -> Dict[str, Tuple[float, float]]:
        """
        MDS-based layout for smaller graphs.
        """
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.float32)
        for source, targets in self.dependencies.items():
            if source in id_to_idx:
                i = id_to_idx[source]
                for target in targets:
                    if target in id_to_idx:
                        j = id_to_idx[target]
                        adj[i, j] = 1
                        adj[j, i] = 1

        # Compute distances
        dist = np.full((n, n), float('inf'), dtype=np.float32)
        np.fill_diagonal(dist, 0)
        dist = np.where(adj > 0, 1, dist)

        # Floyd-Warshall for small graphs only
        for k in range(n):
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        # Handle infinite distances with partition info
        max_finite = np.max(dist[np.isfinite(dist)]) if np.any(np.isfinite(dist)) else 1

        for i, nid_i in enumerate(node_ids):
            assign_i = self.assignments.get(nid_i)
            label_i = assign_i.label if assign_i else "UNKNOWN"
            for j in range(i + 1, n):
                nid_j = node_ids[j]
                assign_j = self.assignments.get(nid_j)
                label_j = assign_j.label if assign_j else "UNKNOWN"

                if np.isinf(dist[i, j]):
                    if label_i == label_j and label_i not in ("UNKNOWN", "DEAD_CODE"):
                        dist[i, j] = dist[j, i] = max_finite + 1
                    else:
                        dist[i, j] = dist[j, i] = max_finite + 3

        # Classical MDS
        positions = self._classical_mds(dist, width, height)

        layout = {}
        for i, nid in enumerate(node_ids):
            layout[nid] = (float(positions[i, 0]), float(positions[i, 1]))

        return layout

    def _classical_mds(self, dist: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Classical (metric) MDS using eigendecomposition.

        Args:
            dist: Distance matrix (n x n)
            width: Target width
            height: Target height

        Returns:
            Positions array (n x 2)
        """
        n = dist.shape[0]

        if n <= 2:
            # Trivial cases
            if n == 1:
                return np.array([[width / 2, height / 2]])
            else:
                return np.array([[width * 0.3, height / 2], [width * 0.7, height / 2]])

        # Double centering
        D_sq = dist ** 2
        row_mean = np.mean(D_sq, axis=1, keepdims=True)
        col_mean = np.mean(D_sq, axis=0, keepdims=True)
        total_mean = np.mean(D_sq)
        B = -0.5 * (D_sq - row_mean - col_mean + total_mean)

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(B)
        except np.linalg.LinAlgError:
            # Fallback to random layout
            logger.warning("MDS eigendecomposition failed, using random layout")
            return np.column_stack([
                np.random.uniform(50, width - 50, n),
                np.random.uniform(50, height - 50, n)
            ])

        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top 2 positive eigenvalues
        pos_mask = eigenvalues > 1e-10
        if np.sum(pos_mask) < 2:
            # Not enough positive eigenvalues, use what we have
            logger.warning("MDS: fewer than 2 positive eigenvalues")
            eigenvalues = np.abs(eigenvalues[:2]) + 1e-10
            eigenvectors = eigenvectors[:, :2]
        else:
            eigenvalues = eigenvalues[:2]
            eigenvectors = eigenvectors[:, :2]

        # Compute positions
        positions = eigenvectors * np.sqrt(eigenvalues)

        # Scale to fit canvas with padding
        padding = 50
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        positions[:, 0] = padding + (positions[:, 0] - x_min) / x_range * (width - 2 * padding)
        positions[:, 1] = padding + (positions[:, 1] - y_min) / y_range * (height - 2 * padding)

        return positions


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
