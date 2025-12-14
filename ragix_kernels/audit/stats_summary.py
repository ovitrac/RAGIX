"""
Kernel: Statistics Summary
Stage: 2 (Analysis)
Wraps: ragix_audit.statistics.DistributionStats, StatisticsComputer

Aggregates statistics from Stage 1 kernels with proper statistical analysis:
- Descriptive statistics (mean, std, quartiles, skewness, kurtosis)
- Distribution analysis with outlier detection
- Quality grading based on thresholds
- Per-component breakdowns

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class StatsSummaryKernel(Kernel):
    """
    Aggregate statistics from Stage 1 data using StatisticsComputer.

    This kernel wraps ragix_audit.statistics.DistributionStats to compute
    comprehensive distributional statistics including skewness, kurtosis,
    outlier detection, and box plot data.

    Configuration options:
        thresholds: Quality thresholds for grading
        include_interpretations: Include human-readable interpretations (default: true)

    Dependencies:
        ast_scan: Symbol and file data
        metrics: Code metrics

    Output:
        overview: Project-level summary
        quality: Quality grades
        distributions: Full statistical distributions (using DistributionStats)
        box_plot_data: Data formatted for D3.js visualization
        recommendations: Initial recommendations
    """

    name = "stats_summary"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Aggregate statistics with distributional analysis"

    requires = ["ast_scan", "metrics"]
    provides = ["overview", "quality_grades", "distributions", "box_plot_data"]

    # Default quality thresholds
    DEFAULT_THRESHOLDS = {
        "complexity_avg": {"good": 5, "moderate": 10, "high": 20},
        "maintainability_index": {"good": 80, "moderate": 60, "poor": 40},
        "test_coverage": {"good": 80, "moderate": 60, "poor": 40},
        "doc_coverage": {"good": 70, "moderate": 50, "poor": 30},
        "methods_per_class": {"good": 10, "moderate": 20, "high": 30},
        "loc_per_file": {"good": 300, "moderate": 500, "high": 1000},
    }

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Aggregate statistics using DistributionStats."""

        # Import existing tools (REUSE!)
        from ragix_audit.statistics import (
            DistributionStats,
            compute_histogram,
            complexity_histogram,
        )

        # Load dependencies
        ast_scan_path = input.dependencies.get("ast_scan")
        metrics_path = input.dependencies.get("metrics")

        if not ast_scan_path or not ast_scan_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")
        if not metrics_path or not metrics_path.exists():
            raise RuntimeError("Missing required dependency: metrics")

        with open(ast_scan_path) as f:
            ast_data = json.load(f).get("data", {})
        with open(metrics_path) as f:
            metrics_data = json.load(f).get("data", {})

        # Get configuration
        thresholds = input.config.get("thresholds", self.DEFAULT_THRESHOLDS)
        include_interpretations = input.config.get("include_interpretations", True)

        logger.info("[stats_summary] Computing distributional statistics")

        # Collect raw values for distribution analysis
        files = metrics_data.get("files", [])
        symbols = ast_data.get("symbols", [])

        # LOC per file
        file_locs = [float(f.get("loc", 0)) for f in files if f.get("loc", 0) > 0]

        # Complexity per method (from hotspots or compute from symbols)
        hotspots = metrics_data.get("hotspots", [])
        method_complexities = [float(h.get("complexity", 1)) for h in hotspots]

        # If no hotspots, try to extract from symbols list
        if not method_complexities:
            for sym in symbols:
                if sym.get("type") in ("method", "function"):
                    cc = sym.get("complexity", 1)
                    if cc > 0:
                        method_complexities.append(float(cc))

        # Methods per class
        class_method_counts: Dict[str, int] = {}
        for sym in symbols:
            if sym.get("type") == "class":
                qname = sym.get("qualified_name", "")
                if qname:
                    class_method_counts[qname] = 0
            elif sym.get("type") == "method":
                qname = sym.get("qualified_name", "")
                if "." in qname:
                    class_name = qname.rsplit(".", 1)[0]
                    if class_name in class_method_counts:
                        class_method_counts[class_name] += 1

        methods_per_class = [float(c) for c in class_method_counts.values()] if class_method_counts else []

        # Compute distributions using DistributionStats (REUSE!)
        distributions = {}
        box_plot_data = {}
        interpretations = {}

        if file_locs:
            loc_stats = DistributionStats.from_values(file_locs, "LOC per file")
            distributions["loc_per_file"] = loc_stats.to_dict()
            box_plot_data["loc_per_file"] = loc_stats.to_box_plot_data()
            if include_interpretations:
                interpretations["loc_per_file"] = loc_stats.interpret()

        if method_complexities:
            cc_stats = DistributionStats.from_values(method_complexities, "Complexity")
            distributions["complexity"] = cc_stats.to_dict()
            box_plot_data["complexity"] = cc_stats.to_box_plot_data()
            if include_interpretations:
                interpretations["complexity"] = cc_stats.interpret()
            # Also add complexity histogram with standard bins
            distributions["complexity_histogram"] = complexity_histogram(method_complexities)

        if methods_per_class:
            mpc_stats = DistributionStats.from_values(methods_per_class, "Methods per class")
            distributions["methods_per_class"] = mpc_stats.to_dict()
            box_plot_data["methods_per_class"] = mpc_stats.to_box_plot_data()
            if include_interpretations:
                interpretations["methods_per_class"] = mpc_stats.interpret()

        # Build overview
        overview = self._build_overview(ast_data, metrics_data, distributions)

        # Compute quality grades
        quality = self._compute_quality_grades(overview, thresholds)

        # Generate recommendations
        recommendations = self._generate_recommendations(overview, quality, interpretations)

        return {
            "overview": overview,
            "quality": quality,
            "distributions": distributions,
            "box_plot_data": box_plot_data,
            "interpretations": interpretations if include_interpretations else {},
            "recommendations": recommendations,
            "thresholds": thresholds,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        overview = data.get("overview", {})
        quality = data.get("quality", {})
        distributions = data.get("distributions", {})
        recommendations = data.get("recommendations", [])

        overall_grade = quality.get("overall_grade", "N/A")

        # Get distribution insights
        loc_dist = distributions.get("loc_per_file", {})
        cc_dist = distributions.get("complexity", {})

        skew_info = ""
        if cc_dist.get("skewness", 0) > 0.5:
            skew_info = " (right-skewed)"

        return (
            f"Project: {overview.get('total_files', 0)} files, "
            f"{overview.get('total_classes', 0)} classes, "
            f"{overview.get('total_loc', 0):,} LOC. "
            f"Quality: {overall_grade}. "
            f"MI={overview.get('maintainability_index', 0):.0f}, "
            f"CC: μ={cc_dist.get('mean', 0):.1f}, σ={cc_dist.get('std', 0):.1f}{skew_info}. "
            f"LOC/file: μ={loc_dist.get('mean', 0):.0f}, med={loc_dist.get('median', 0):.0f}. "
            f"Recommendations: {len(recommendations)}."
        )

    def _build_overview(
        self,
        ast_data: Dict[str, Any],
        metrics_data: Dict[str, Any],
        distributions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build project overview from Stage 1 data."""
        ast_stats = ast_data.get("statistics", {})
        metrics_summary = metrics_data.get("summary", {})

        # Extract from ast_scan
        by_type = ast_stats.get("by_type", {})
        total_files = ast_stats.get("total_files", 0)
        total_symbols = ast_stats.get("total_symbols", 0)

        # Extract from metrics
        lines = metrics_summary.get("lines", {})
        complexity = metrics_summary.get("complexity", {})
        debt = metrics_data.get("technical_debt", {})

        # Get distribution-based values if available
        loc_dist = distributions.get("loc_per_file", {})
        cc_dist = distributions.get("complexity", {})

        return {
            "total_files": total_files,
            "total_symbols": total_symbols,
            "total_classes": by_type.get("class", 0),
            "total_methods": by_type.get("method", 0),
            "total_functions": by_type.get("function", 0),
            "total_loc": lines.get("total", 0),
            "code_loc": lines.get("code", 0),
            "comment_loc": lines.get("comments", 0),
            "total_complexity": complexity.get("total", 0),
            "avg_complexity": cc_dist.get("mean", complexity.get("avg_per_method", 0)),
            "median_complexity": cc_dist.get("median", 0),
            "maintainability_index": metrics_summary.get("maintainability_index", 0),
            "technical_debt_hours": debt.get("hours", 0),
            "technical_debt_days": debt.get("days", 0),
            # Distribution-based metrics
            "avg_loc_per_file": loc_dist.get("mean", 0),
            "median_loc_per_file": loc_dist.get("median", 0),
            "complexity_outliers": cc_dist.get("outlier_count", 0),
        }

    def _compute_quality_grades(
        self,
        overview: Dict[str, Any],
        thresholds: Dict[str, Dict[str, int]]
    ) -> Dict[str, Any]:
        """Compute quality grades based on thresholds."""
        grades = {}

        # Maintainability Index grade
        mi = overview.get("maintainability_index", 0)
        mi_thresh = thresholds.get("maintainability_index", {})
        if mi >= mi_thresh.get("good", 80):
            grades["maintainability"] = "A"
        elif mi >= mi_thresh.get("moderate", 60):
            grades["maintainability"] = "B"
        elif mi >= mi_thresh.get("poor", 40):
            grades["maintainability"] = "C"
        else:
            grades["maintainability"] = "D"

        # Complexity grade (use median for robustness to outliers)
        avg_cc = overview.get("median_complexity", 0) or overview.get("avg_complexity", 0)
        cc_thresh = thresholds.get("complexity_avg", {})
        if avg_cc <= cc_thresh.get("good", 5):
            grades["complexity"] = "A"
        elif avg_cc <= cc_thresh.get("moderate", 10):
            grades["complexity"] = "B"
        elif avg_cc <= cc_thresh.get("high", 20):
            grades["complexity"] = "C"
        else:
            grades["complexity"] = "D"

        # Size grade (methods per class)
        total_classes = overview.get("total_classes", 1) or 1
        total_methods = overview.get("total_methods", 0)
        methods_per_class = total_methods / total_classes
        mpc_thresh = thresholds.get("methods_per_class", {})
        if methods_per_class <= mpc_thresh.get("good", 10):
            grades["size"] = "A"
        elif methods_per_class <= mpc_thresh.get("moderate", 20):
            grades["size"] = "B"
        elif methods_per_class <= mpc_thresh.get("high", 30):
            grades["size"] = "C"
        else:
            grades["size"] = "D"

        # Overall grade (weighted average)
        grade_values = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        weights = {"maintainability": 0.4, "complexity": 0.35, "size": 0.25}
        weighted_sum = sum(
            grade_values.get(grades.get(k, "C"), 2) * w
            for k, w in weights.items()
        )
        if weighted_sum >= 3.5:
            grades["overall_grade"] = "A"
        elif weighted_sum >= 2.5:
            grades["overall_grade"] = "B"
        elif weighted_sum >= 1.5:
            grades["overall_grade"] = "C"
        else:
            grades["overall_grade"] = "D"

        grades["methods_per_class"] = round(methods_per_class, 1)

        return grades

    def _generate_recommendations(
        self,
        overview: Dict[str, Any],
        quality: Dict[str, Any],
        interpretations: Dict[str, Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis and interpretations."""
        recommendations = []

        # Maintainability recommendations
        if quality.get("maintainability") in ("C", "D"):
            recommendations.append({
                "category": "maintainability",
                "severity": "high" if quality["maintainability"] == "D" else "medium",
                "title": "Improve Maintainability Index",
                "description": f"MI score is {overview.get('maintainability_index', 0):.0f}. "
                              "Consider reducing complexity and improving documentation.",
            })

        # Complexity recommendations (enhanced with distribution insights)
        if quality.get("complexity") in ("C", "D"):
            cc_interp = interpretations.get("complexity", {})
            skew_note = cc_interp.get("skewness", "")
            outlier_note = cc_interp.get("outliers", "")

            desc = f"Average complexity is {overview.get('avg_complexity', 0):.1f}. "
            if "right-skewed" in skew_note.lower():
                desc += "Distribution is right-skewed (few very complex methods). "
            if overview.get("complexity_outliers", 0) > 0:
                desc += f"{overview.get('complexity_outliers')} complexity outliers detected. "
            desc += "Refactor complex methods into smaller units."

            recommendations.append({
                "category": "complexity",
                "severity": "high" if quality["complexity"] == "D" else "medium",
                "title": "Reduce Code Complexity",
                "description": desc,
            })

        # Size recommendations
        if quality.get("size") in ("C", "D"):
            recommendations.append({
                "category": "architecture",
                "severity": "medium",
                "title": "Review Class Sizes",
                "description": f"Average {quality.get('methods_per_class', 0):.0f} methods per class. "
                              "Consider splitting large classes.",
            })

        # Technical debt recommendations
        debt_days = overview.get("technical_debt_days", 0)
        if debt_days > 5:
            recommendations.append({
                "category": "debt",
                "severity": "high" if debt_days > 20 else "medium",
                "title": "Address Technical Debt",
                "description": f"Estimated {debt_days:.1f} person-days of technical debt. "
                              "Prioritize refactoring high-complexity areas.",
            })

        # Outlier-based recommendations
        loc_interp = interpretations.get("loc_per_file", {})
        if "Many outliers" in loc_interp.get("outliers", ""):
            recommendations.append({
                "category": "architecture",
                "severity": "medium",
                "title": "Large File Outliers Detected",
                "description": "Several files are significantly larger than average. "
                              "Review these files for potential decomposition.",
            })

        return recommendations
