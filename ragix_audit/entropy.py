"""
Entropy Module — Information-Theoretic Metrics for Code Analysis

Provides entropy-based metrics to measure uniformity of distributions:
- Structural Entropy: How evenly is code distributed across components?
- Complexity Entropy: How evenly is complexity distributed across files?
- Coupling Entropy: How evenly are dependencies distributed?

High entropy = uniform distribution (well-balanced)
Low entropy = concentrated distribution (potential risk)

Reference: Shannon, C.E. (1948). "A Mathematical Theory of Communication"

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-11
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntropyMetrics:
    """
    Complete entropy analysis for a codebase.
    """
    # Core entropy values (in bits)
    structural_entropy: float = 0.0      # Based on LOC distribution
    structural_entropy_files: float = 0.0  # Based on file count distribution
    complexity_entropy: float = 0.0      # Based on CC distribution
    coupling_entropy: float = 0.0        # Based on dependency distribution

    # Reference values for interpretation
    max_possible_entropy: float = 0.0    # log2(n) where n = number of components
    component_count: int = 0

    # Normalized entropy (0-1 scale)
    structural_normalized: float = 0.0
    complexity_normalized: float = 0.0
    coupling_normalized: float = 0.0

    # Interpretation
    structural_interpretation: str = ""
    complexity_interpretation: str = ""
    coupling_interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "structural_entropy": round(self.structural_entropy, 4),
            "structural_entropy_files": round(self.structural_entropy_files, 4),
            "complexity_entropy": round(self.complexity_entropy, 4),
            "coupling_entropy": round(self.coupling_entropy, 4),
            "max_possible_entropy": round(self.max_possible_entropy, 4),
            "component_count": self.component_count,
            "normalized": {
                "structural": round(self.structural_normalized, 4),
                "complexity": round(self.complexity_normalized, 4),
                "coupling": round(self.coupling_normalized, 4),
            },
            "interpretations": {
                "structural": self.structural_interpretation,
                "complexity": self.complexity_interpretation,
                "coupling": self.coupling_interpretation,
            },
        }


def shannon_entropy(distribution: Dict[str, float]) -> float:
    """
    Compute Shannon entropy of a distribution.

    H = -Σ pᵢ log₂(pᵢ)

    Args:
        distribution: Dictionary of {item: value} where values are counts/sizes

    Returns:
        Entropy in bits (0 = all mass in one item, log₂(n) = uniform)
    """
    if not distribution:
        return 0.0

    total = sum(distribution.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for value in distribution.values():
        if value > 0:
            p = value / total
            entropy -= p * math.log2(p)

    return entropy


def normalized_entropy(distribution: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute normalized entropy (0-1 scale).

    Args:
        distribution: Dictionary of {item: value}

    Returns:
        Tuple of (entropy, normalized_entropy)
    """
    if not distribution or len(distribution) <= 1:
        return 0.0, 0.0

    entropy = shannon_entropy(distribution)
    max_entropy = math.log2(len(distribution))

    if max_entropy > 0:
        normalized = entropy / max_entropy
    else:
        normalized = 0.0

    return entropy, normalized


def interpret_entropy(
    entropy: float,
    max_entropy: float,
    metric_name: str = "distribution"
) -> str:
    """
    Provide human-readable interpretation of entropy value.

    Args:
        entropy: Computed entropy in bits
        max_entropy: Maximum possible entropy (log₂(n))
        metric_name: Name for the interpretation message

    Returns:
        Interpretation string
    """
    if max_entropy <= 0:
        return f"Insufficient data for {metric_name} analysis"

    ratio = entropy / max_entropy

    if ratio < 0.5:
        return f"Highly concentrated {metric_name} (H={entropy:.2f}, {ratio*100:.0f}% of max) — potential single point of failure"
    elif ratio < 0.7:
        return f"Moderately concentrated {metric_name} (H={entropy:.2f}, {ratio*100:.0f}% of max)"
    elif ratio < 0.85:
        return f"Moderately distributed {metric_name} (H={entropy:.2f}, {ratio*100:.0f}% of max)"
    else:
        return f"Well-distributed {metric_name} (H={entropy:.2f}, {ratio*100:.0f}% of max)"


class EntropyComputer:
    """
    Computes entropy metrics for codebase analysis.
    """

    def __init__(self):
        self.metrics: Optional[EntropyMetrics] = None

    def compute_structural_entropy(
        self,
        component_sizes: Dict[str, int]
    ) -> Tuple[float, float, str]:
        """
        Compute structural entropy based on LOC distribution across components.

        Args:
            component_sizes: {component_id: LOC}

        Returns:
            Tuple of (entropy, normalized_entropy, interpretation)
        """
        if not component_sizes:
            return 0.0, 0.0, "No components to analyze"

        entropy, normalized = normalized_entropy(
            {k: float(v) for k, v in component_sizes.items()}
        )
        max_entropy = math.log2(len(component_sizes)) if len(component_sizes) > 1 else 0

        interpretation = interpret_entropy(entropy, max_entropy, "code distribution")

        return entropy, normalized, interpretation

    def compute_complexity_entropy(
        self,
        file_complexities: Dict[str, float]
    ) -> Tuple[float, float, str]:
        """
        Compute complexity entropy based on CC distribution across files.

        Args:
            file_complexities: {file_path: total_cc or avg_cc}

        Returns:
            Tuple of (entropy, normalized_entropy, interpretation)
        """
        if not file_complexities:
            return 0.0, 0.0, "No complexity data to analyze"

        # Filter out zero complexities
        non_zero = {k: v for k, v in file_complexities.items() if v > 0}
        if not non_zero:
            return 0.0, 0.0, "All files have zero complexity"

        entropy, normalized = normalized_entropy(non_zero)
        max_entropy = math.log2(len(non_zero)) if len(non_zero) > 1 else 0

        interpretation = interpret_entropy(entropy, max_entropy, "complexity distribution")

        return entropy, normalized, interpretation

    def compute_coupling_entropy(
        self,
        node_degrees: Dict[str, int]
    ) -> Tuple[float, float, str]:
        """
        Compute coupling entropy based on dependency degree distribution.

        Args:
            node_degrees: {node_id: in_degree + out_degree}

        Returns:
            Tuple of (entropy, normalized_entropy, interpretation)
        """
        if not node_degrees:
            return 0.0, 0.0, "No dependency data to analyze"

        # Filter out isolated nodes
        non_zero = {k: float(v) for k, v in node_degrees.items() if v > 0}
        if not non_zero:
            return 0.0, 0.0, "No dependencies found"

        entropy, normalized = normalized_entropy(non_zero)
        max_entropy = math.log2(len(non_zero)) if len(non_zero) > 1 else 0

        interpretation = interpret_entropy(entropy, max_entropy, "dependency distribution")

        return entropy, normalized, interpretation

    def compute_all(
        self,
        component_sizes: Optional[Dict[str, int]] = None,
        component_file_counts: Optional[Dict[str, int]] = None,
        file_complexities: Optional[Dict[str, float]] = None,
        node_degrees: Optional[Dict[str, int]] = None,
    ) -> EntropyMetrics:
        """
        Compute all entropy metrics.

        Args:
            component_sizes: {component_id: LOC}
            component_file_counts: {component_id: file_count}
            file_complexities: {file_path: avg_cc}
            node_degrees: {node_id: total_degree}

        Returns:
            EntropyMetrics with all values computed
        """
        metrics = EntropyMetrics()

        # Component count for max entropy calculation
        if component_sizes:
            metrics.component_count = len(component_sizes)
            metrics.max_possible_entropy = math.log2(metrics.component_count) if metrics.component_count > 1 else 0

        # Structural entropy (LOC-based)
        if component_sizes:
            h, norm, interp = self.compute_structural_entropy(component_sizes)
            metrics.structural_entropy = h
            metrics.structural_normalized = norm
            metrics.structural_interpretation = interp

        # Structural entropy (file count-based)
        if component_file_counts:
            h_files, _, _ = self.compute_structural_entropy(component_file_counts)
            metrics.structural_entropy_files = h_files

        # Complexity entropy
        if file_complexities:
            h, norm, interp = self.compute_complexity_entropy(file_complexities)
            metrics.complexity_entropy = h
            metrics.complexity_normalized = norm
            metrics.complexity_interpretation = interp

        # Coupling entropy
        if node_degrees:
            h, norm, interp = self.compute_coupling_entropy(node_degrees)
            metrics.coupling_entropy = h
            metrics.coupling_normalized = norm
            metrics.coupling_interpretation = interp

        self.metrics = metrics
        return metrics


def gini_coefficient(values: List[float]) -> float:
    """
    Compute Gini coefficient as an alternative inequality measure.

    G = 0: Perfect equality
    G = 1: Maximum inequality

    Complements entropy for understanding concentration.

    Args:
        values: List of positive values

    Returns:
        Gini coefficient (0-1)
    """
    if not values or len(values) < 2:
        return 0.0

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)

    if total <= 0:
        return 0.0

    # Compute using the relative mean absolute difference formula
    cumsum = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += (2 * (i + 1) - n - 1) * v

    gini = cumsum / (n * total)
    return max(0.0, min(1.0, gini))


def concentration_ratio(values: List[float], top_k: int = 4) -> float:
    """
    Compute concentration ratio (CR-k): share of top k items.

    CR-4: Share of top 4 components (common in economics)

    Args:
        values: List of values (e.g., LOC per component)
        top_k: Number of top items to consider

    Returns:
        Concentration ratio (0-1)
    """
    if not values:
        return 0.0

    sorted_vals = sorted(values, reverse=True)
    total = sum(sorted_vals)

    if total <= 0:
        return 0.0

    top_sum = sum(sorted_vals[:top_k])
    return top_sum / total


@dataclass
class InequalityMetrics:
    """
    Additional inequality metrics complementing entropy.
    """
    gini: float = 0.0
    cr4: float = 0.0   # Concentration ratio (top 4)
    cr8: float = 0.0   # Concentration ratio (top 8)
    herfindahl: float = 0.0  # Herfindahl-Hirschman Index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gini": round(self.gini, 4),
            "cr4": round(self.cr4, 4),
            "cr8": round(self.cr8, 4),
            "herfindahl": round(self.herfindahl, 4),
        }


def compute_inequality_metrics(values: List[float]) -> InequalityMetrics:
    """
    Compute comprehensive inequality metrics.

    Args:
        values: List of values (e.g., LOC per component)

    Returns:
        InequalityMetrics
    """
    metrics = InequalityMetrics()

    if not values:
        return metrics

    metrics.gini = gini_coefficient(values)
    metrics.cr4 = concentration_ratio(values, 4)
    metrics.cr8 = concentration_ratio(values, 8)

    # Herfindahl-Hirschman Index: sum of squared market shares
    total = sum(values)
    if total > 0:
        shares = [v / total for v in values]
        metrics.herfindahl = sum(s ** 2 for s in shares)

    return metrics
