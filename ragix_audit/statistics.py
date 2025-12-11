"""
Statistics Module — Descriptive Statistics for Code Metrics

Provides comprehensive distributional analysis including:
- Central tendency (mean, median)
- Dispersion (std, IQR, CV)
- Shape (skewness, kurtosis)
- Quantiles (Q1, Q3, percentiles)
- Histogram data for visualization

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-11
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """
    Complete descriptive statistics for a distribution.

    All statistics are computed from raw values using standard formulas.
    For small samples (n < 3), some statistics may be undefined (NaN).
    """
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    min: float = 0.0
    max: float = 0.0
    q1: float = 0.0          # 25th percentile
    median: float = 0.0      # 50th percentile
    q3: float = 0.0          # 75th percentile
    iqr: float = 0.0         # Interquartile range (Q3 - Q1)
    cv: float = 0.0          # Coefficient of variation (std / mean)
    skewness: float = 0.0    # Fisher-Pearson skewness
    kurtosis: float = 0.0    # Excess kurtosis (Fisher's definition)
    sum: float = 0.0

    # Percentiles for box plot
    p5: float = 0.0          # 5th percentile (whisker low)
    p95: float = 0.0         # 95th percentile (whisker high)

    # Outliers (values beyond 1.5 * IQR)
    outliers: List[float] = field(default_factory=list)
    outlier_count: int = 0

    @classmethod
    def from_values(cls, values: List[float], name: str = "") -> "DistributionStats":
        """
        Compute all statistics from raw values.

        Args:
            values: List of numeric values
            name: Optional name for logging

        Returns:
            DistributionStats with all metrics computed
        """
        if not values:
            return cls()

        n = len(values)
        sorted_vals = sorted(values)

        # Basic stats
        total = sum(values)
        mean = total / n

        # Variance and std (sample variance with n-1 denominator)
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            variance = 0.0
            std = 0.0

        # Quantiles using linear interpolation
        def percentile(sorted_data: List[float], p: float) -> float:
            """Compute percentile using linear interpolation."""
            if not sorted_data:
                return 0.0
            if len(sorted_data) == 1:
                return sorted_data[0]

            k = (len(sorted_data) - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)

            if f == c:
                return sorted_data[int(k)]

            d0 = sorted_data[int(f)] * (c - k)
            d1 = sorted_data[int(c)] * (k - f)
            return d0 + d1

        q1 = percentile(sorted_vals, 25)
        median = percentile(sorted_vals, 50)
        q3 = percentile(sorted_vals, 75)
        p5 = percentile(sorted_vals, 5)
        p95 = percentile(sorted_vals, 95)

        iqr = q3 - q1

        # Coefficient of variation
        cv = std / mean if mean != 0 else 0.0

        # Skewness (Fisher-Pearson)
        # γ₁ = (1/n) Σ((xᵢ - μ)/σ)³
        if n >= 3 and std > 0:
            m3 = sum((x - mean) ** 3 for x in values) / n
            skewness = m3 / (std ** 3)
        else:
            skewness = 0.0

        # Kurtosis (excess kurtosis, Fisher's definition)
        # γ₂ = (1/n) Σ((xᵢ - μ)/σ)⁴ - 3
        if n >= 4 and std > 0:
            m4 = sum((x - mean) ** 4 for x in values) / n
            kurtosis = (m4 / (std ** 4)) - 3
        else:
            kurtosis = 0.0

        # Outliers (beyond 1.5 * IQR from Q1/Q3)
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [x for x in sorted_vals if x < lower_fence or x > upper_fence]

        return cls(
            count=n,
            mean=mean,
            std=std,
            variance=variance,
            min=sorted_vals[0],
            max=sorted_vals[-1],
            q1=q1,
            median=median,
            q3=q3,
            iqr=iqr,
            cv=cv,
            skewness=skewness,
            kurtosis=kurtosis,
            sum=total,
            p5=p5,
            p95=p95,
            outliers=outliers,
            outlier_count=len(outliers),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "variance": round(self.variance, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "q1": round(self.q1, 4),
            "median": round(self.median, 4),
            "q3": round(self.q3, 4),
            "iqr": round(self.iqr, 4),
            "cv": round(self.cv, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "sum": round(self.sum, 4),
            "p5": round(self.p5, 4),
            "p95": round(self.p95, 4),
            "outlier_count": self.outlier_count,
        }

    def to_box_plot_data(self) -> Dict[str, Any]:
        """
        Export data for D3.js box plot visualization.

        Returns:
            Dictionary with whisker_low, q1, median, q3, whisker_high, outliers
        """
        return {
            "whisker_low": round(self.p5, 2),
            "q1": round(self.q1, 2),
            "median": round(self.median, 2),
            "q3": round(self.q3, 2),
            "whisker_high": round(self.p95, 2),
            "outliers": [round(x, 2) for x in self.outliers[:20]],  # Limit outliers
            "count": self.count,
            "mean": round(self.mean, 2),
        }

    def interpret(self) -> Dict[str, str]:
        """
        Provide human-readable interpretation of statistics.

        Returns:
            Dictionary with interpretations for key metrics
        """
        interpretations = {}

        # Skewness interpretation
        if abs(self.skewness) < 0.5:
            interpretations["skewness"] = "Symmetric distribution"
        elif self.skewness > 0:
            interpretations["skewness"] = f"Right-skewed (long tail of high values, γ₁={self.skewness:.2f})"
        else:
            interpretations["skewness"] = f"Left-skewed (long tail of low values, γ₁={self.skewness:.2f})"

        # Kurtosis interpretation
        if abs(self.kurtosis) < 1:
            interpretations["kurtosis"] = "Mesokurtic (normal-like tails)"
        elif self.kurtosis > 0:
            interpretations["kurtosis"] = f"Leptokurtic (heavy tails, more outliers, γ₂={self.kurtosis:.2f})"
        else:
            interpretations["kurtosis"] = f"Platykurtic (light tails, fewer outliers, γ₂={self.kurtosis:.2f})"

        # CV interpretation
        if self.cv < 0.2:
            interpretations["cv"] = "Low variability (CV < 20%)"
        elif self.cv < 0.5:
            interpretations["cv"] = "Moderate variability (20% ≤ CV < 50%)"
        else:
            interpretations["cv"] = f"High variability (CV = {self.cv*100:.1f}%)"

        # Outliers interpretation
        if self.count > 0:
            outlier_pct = (self.outlier_count / self.count) * 100
            if outlier_pct > 10:
                interpretations["outliers"] = f"Many outliers ({outlier_pct:.1f}%) - investigate extreme values"
            elif outlier_pct > 5:
                interpretations["outliers"] = f"Some outliers ({outlier_pct:.1f}%)"
            else:
                interpretations["outliers"] = "Few outliers (< 5%)"

        return interpretations


@dataclass
class ComponentStats:
    """
    Statistics for a single component (SK/SC/SG).
    """
    component_id: str
    component_type: str  # "service", "screen", "general"
    file_count: int = 0

    # Per-metric distributions
    loc_stats: Optional[DistributionStats] = None
    complexity_stats: Optional[DistributionStats] = None
    method_count_stats: Optional[DistributionStats] = None

    # Aggregates
    total_loc: int = 0
    total_methods: int = 0
    total_classes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "file_count": self.file_count,
            "total_loc": self.total_loc,
            "total_methods": self.total_methods,
            "total_classes": self.total_classes,
            "loc_stats": self.loc_stats.to_dict() if self.loc_stats else None,
            "complexity_stats": self.complexity_stats.to_dict() if self.complexity_stats else None,
            "method_count_stats": self.method_count_stats.to_dict() if self.method_count_stats else None,
        }


@dataclass
class CodebaseStats:
    """
    Global statistics for entire codebase.
    """
    # File-level distributions
    file_size_stats: Optional[DistributionStats] = None      # LOC per file
    file_complexity_stats: Optional[DistributionStats] = None # Avg CC per file

    # Method-level distributions
    method_complexity_stats: Optional[DistributionStats] = None  # CC per method
    method_size_stats: Optional[DistributionStats] = None        # LOC per method

    # Class-level distributions
    class_size_stats: Optional[DistributionStats] = None         # Methods per class
    class_loc_stats: Optional[DistributionStats] = None          # LOC per class

    # Package-level distributions
    package_size_stats: Optional[DistributionStats] = None       # Files per package

    # Per-component stats
    component_stats: Dict[str, ComponentStats] = field(default_factory=dict)

    # Totals
    total_files: int = 0
    total_loc: int = 0
    total_classes: int = 0
    total_methods: int = 0
    total_packages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "totals": {
                "files": self.total_files,
                "loc": self.total_loc,
                "classes": self.total_classes,
                "methods": self.total_methods,
                "packages": self.total_packages,
            },
            "file_size_stats": self.file_size_stats.to_dict() if self.file_size_stats else None,
            "file_complexity_stats": self.file_complexity_stats.to_dict() if self.file_complexity_stats else None,
            "method_complexity_stats": self.method_complexity_stats.to_dict() if self.method_complexity_stats else None,
            "method_size_stats": self.method_size_stats.to_dict() if self.method_size_stats else None,
            "class_size_stats": self.class_size_stats.to_dict() if self.class_size_stats else None,
            "class_loc_stats": self.class_loc_stats.to_dict() if self.class_loc_stats else None,
            "package_size_stats": self.package_size_stats.to_dict() if self.package_size_stats else None,
            "component_stats": {
                k: v.to_dict() for k, v in self.component_stats.items()
            },
        }


class StatisticsComputer:
    """
    Computes comprehensive statistics from AST metrics.
    """

    def __init__(self):
        self.codebase_stats: Optional[CodebaseStats] = None

    def compute_from_metrics(
        self,
        file_metrics: List[Any],  # List of FileMetrics from code_metrics.py
        component_mapping: Optional[Dict[str, str]] = None  # file_path -> component_id
    ) -> CodebaseStats:
        """
        Compute codebase statistics from file metrics.

        Args:
            file_metrics: List of FileMetrics objects
            component_mapping: Optional mapping of file paths to component IDs

        Returns:
            CodebaseStats with all distributions computed
        """
        stats = CodebaseStats()

        # Collect raw values
        file_locs: List[float] = []
        file_complexities: List[float] = []
        method_complexities: List[float] = []
        method_sizes: List[float] = []
        class_sizes: List[float] = []  # Methods per class
        class_locs: List[float] = []
        package_files: Dict[str, int] = {}

        # Per-component collections
        component_data: Dict[str, Dict[str, List[float]]] = {}

        for fm in file_metrics:
            # File-level
            file_locs.append(float(fm.loc))

            # Track package
            package = self._extract_package(fm.path)
            package_files[package] = package_files.get(package, 0) + 1

            # Component tracking
            comp_id = None
            if component_mapping:
                comp_id = component_mapping.get(fm.path)

            if comp_id:
                if comp_id not in component_data:
                    component_data[comp_id] = {
                        "locs": [],
                        "complexities": [],
                        "method_counts": [],
                    }
                component_data[comp_id]["locs"].append(float(fm.loc))

            # Process classes
            total_file_cc = 0
            total_file_methods = 0

            for cm in getattr(fm, 'class_metrics', []):
                class_method_count = len(getattr(cm, 'method_metrics', []))
                class_sizes.append(float(class_method_count))
                class_locs.append(float(getattr(cm, 'loc', 0)))
                stats.total_classes += 1

                # Process methods in class
                for mm in getattr(cm, 'method_metrics', []):
                    cc = getattr(mm, 'cyclomatic_complexity', 1)
                    method_complexities.append(float(cc))
                    method_sizes.append(float(getattr(mm, 'loc', 0)))
                    total_file_cc += cc
                    total_file_methods += 1
                    stats.total_methods += 1

            # Process standalone functions
            for func in getattr(fm, 'function_metrics', []):
                cc = getattr(func, 'cyclomatic_complexity', 1)
                method_complexities.append(float(cc))
                method_sizes.append(float(getattr(func, 'loc', 0)))
                total_file_cc += cc
                total_file_methods += 1
                stats.total_methods += 1

            # Average complexity for file
            if total_file_methods > 0:
                file_complexities.append(total_file_cc / total_file_methods)
                if comp_id:
                    component_data[comp_id]["complexities"].append(total_file_cc / total_file_methods)
                    component_data[comp_id]["method_counts"].append(float(total_file_methods))

        # Compute distributions
        stats.total_files = len(file_metrics)
        stats.total_loc = int(sum(file_locs))
        stats.total_packages = len(package_files)

        if file_locs:
            stats.file_size_stats = DistributionStats.from_values(file_locs)
        if file_complexities:
            stats.file_complexity_stats = DistributionStats.from_values(file_complexities)
        if method_complexities:
            stats.method_complexity_stats = DistributionStats.from_values(method_complexities)
        if method_sizes:
            stats.method_size_stats = DistributionStats.from_values(method_sizes)
        if class_sizes:
            stats.class_size_stats = DistributionStats.from_values(class_sizes)
        if class_locs:
            stats.class_loc_stats = DistributionStats.from_values(class_locs)
        if package_files:
            stats.package_size_stats = DistributionStats.from_values(
                [float(c) for c in package_files.values()]
            )

        # Compute per-component stats
        for comp_id, data in component_data.items():
            comp_stats = ComponentStats(
                component_id=comp_id,
                component_type=self._infer_component_type(comp_id),
                file_count=len(data["locs"]),
                total_loc=int(sum(data["locs"])),
                total_methods=int(sum(data["method_counts"])) if data["method_counts"] else 0,
            )

            if data["locs"]:
                comp_stats.loc_stats = DistributionStats.from_values(data["locs"])
            if data["complexities"]:
                comp_stats.complexity_stats = DistributionStats.from_values(data["complexities"])
            if data["method_counts"]:
                comp_stats.method_count_stats = DistributionStats.from_values(data["method_counts"])

            stats.component_stats[comp_id] = comp_stats

        self.codebase_stats = stats
        return stats

    def _extract_package(self, file_path: str) -> str:
        """Extract package name from file path."""
        # Simple heuristic: use parent directory
        from pathlib import Path
        p = Path(file_path)
        return p.parent.name if p.parent.name else "root"

    def _infer_component_type(self, comp_id: str) -> str:
        """Infer component type from ID."""
        comp_id_upper = comp_id.upper()
        if comp_id_upper.startswith("SK"):
            return "service"
        elif comp_id_upper.startswith("SC"):
            return "screen"
        elif comp_id_upper.startswith("SG"):
            return "general"
        return "unknown"


def compute_histogram(
    values: List[float],
    bins: int = 10,
    bin_edges: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute histogram data for D3.js visualization.

    Args:
        values: Raw data values
        bins: Number of bins (ignored if bin_edges provided)
        bin_edges: Explicit bin edges (e.g., [0, 5, 10, 20, 50])

    Returns:
        Dictionary with bin_edges, counts, percentages
    """
    if not values:
        return {"bin_edges": [], "counts": [], "percentages": []}

    sorted_vals = sorted(values)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # Determine bin edges
    if bin_edges is None:
        # Create uniform bins
        step = (max_val - min_val) / bins if max_val > min_val else 1
        bin_edges = [min_val + i * step for i in range(bins + 1)]

    # Count values in each bin
    counts = [0] * (len(bin_edges) - 1)
    for v in values:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                counts[i] += 1
                break
            elif i == len(bin_edges) - 2 and v == bin_edges[i + 1]:
                # Include max value in last bin
                counts[i] += 1
                break

    total = len(values)
    percentages = [round(c / total * 100, 1) for c in counts]

    return {
        "bin_edges": [round(e, 2) for e in bin_edges],
        "counts": counts,
        "percentages": percentages,
        "total": total,
    }


# Convenience function for complexity histogram with standard bins
def complexity_histogram(complexities: List[float]) -> Dict[str, Any]:
    """
    Compute complexity histogram with standard CC bins.

    Bins: [1-5] Simple, [6-10] Moderate, [11-20] Complex, [21+] Very Complex
    """
    return compute_histogram(
        complexities,
        bin_edges=[1, 6, 11, 21, float('inf')]
    )
