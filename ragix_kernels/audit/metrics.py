"""
Kernel: Code Metrics
Stage: 1 (Data Collection)
Wraps: ragix_core.code_metrics.calculate_metrics_from_graph

Calculates professional code metrics:
- Lines of Code (LOC, SLOC, comment lines)
- Cyclomatic Complexity (CC)
- Maintainability Index (MI)
- Technical Debt estimation
- Method/class size distributions

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class MetricsKernel(Kernel):
    """
    Calculate professional code metrics.

    This kernel computes various code quality metrics from
    the dependency graph built by ast_scan.

    Configuration options:
        complexity_threshold: Flag methods with CC > threshold (default: 10)
        loc_threshold: Flag files with LOC > threshold (default: 300)
        debt_rate_hours_per_issue: Hours per tech debt issue (default: 0.5)

    Dependencies:
        ast_scan: Provides symbol and dependency data

    Output:
        summary: Project-level metrics summary
        files: Per-file metrics
        methods: Per-method complexity
        hotspots: High-complexity methods
        technical_debt: Debt estimation
    """

    name = "metrics"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Calculate professional code metrics"

    requires = ["ast_scan"]
    provides = ["metrics", "complexity", "debt"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Calculate code metrics from AST data."""

        # Import here to avoid circular imports
        from ragix_core.dependencies import DependencyGraph
        from ragix_core.code_metrics import calculate_metrics_from_graph

        # Get configuration
        complexity_threshold = input.config.get("complexity_threshold", 10)
        loc_threshold = input.config.get("loc_threshold", 300)
        debt_rate = input.config.get("debt_rate_hours_per_issue", 0.5)

        # Load ast_scan output
        ast_scan_path = input.dependencies.get("ast_scan")
        if not ast_scan_path or not ast_scan_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")

        with open(ast_scan_path) as f:
            ast_data = json.load(f)

        # Reconstruct graph from cached data
        symbols = ast_data.get("data", {}).get("symbols", [])
        dependencies = ast_data.get("data", {}).get("dependencies", [])
        graph = DependencyGraph.from_cached_data(symbols, dependencies)

        logger.info(f"[metrics] Computing metrics for {len(symbols)} symbols")

        # Calculate metrics using existing RAGIX tool
        metrics = calculate_metrics_from_graph(graph)
        summary = metrics.summary()

        # Get hotspots (high complexity methods)
        hotspots = metrics.get_hotspots(50)
        hotspots_data = [
            {"name": name, "complexity": cc}
            for name, cc in hotspots
            if cc > complexity_threshold
        ]

        # Per-file metrics (this also computes actual LOC)
        self._total_loc = 0  # Initialize before calling
        file_metrics = self._compute_file_metrics(ast_data, loc_threshold)

        # Update summary with actual LOC (more accurate than estimate)
        if self._total_loc > 0:
            summary["lines"] = summary.get("lines", {})
            summary["lines"]["total"] = self._total_loc
            summary["lines"]["code"] = self._total_loc  # Conservative estimate

        # Technical debt estimation
        debt = self._estimate_debt(summary, hotspots_data, debt_rate)

        return {
            "summary": summary,
            "files": file_metrics,
            "hotspots": hotspots_data,
            "technical_debt": debt,
            "thresholds": {
                "complexity": complexity_threshold,
                "loc": loc_threshold,
                "debt_rate": debt_rate,
            }
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        summary = data.get("summary", {})
        debt = data.get("technical_debt", {})
        hotspots = data.get("hotspots", [])

        lines = summary.get("lines", {})
        complexity = summary.get("complexity", {})
        mi = summary.get("maintainability_index", 0)

        mi_rating = (
            "Excellent" if mi >= 80 else
            "Good" if mi >= 60 else
            "Moderate" if mi >= 40 else
            "Poor"
        )

        return (
            f"Code metrics: {lines.get('total', 0):,} LOC, "
            f"CC avg={complexity.get('avg_per_method', 0):.1f}. "
            f"Maintainability Index: {mi:.1f} ({mi_rating}). "
            f"Technical debt: {debt.get('hours', 0):.1f}h ({debt.get('days', 0):.1f} person-days). "
            f"High-complexity hotspots: {len(hotspots)}."
        )

    def _compute_file_metrics(
        self,
        ast_data: Dict[str, Any],
        loc_threshold: int
    ) -> List[Dict[str, Any]]:
        """Compute per-file metrics."""
        files = ast_data.get("data", {}).get("files", [])
        symbols = ast_data.get("data", {}).get("symbols", [])

        # Group symbols by file
        file_symbols: Dict[str, List] = {}
        for sym in symbols:
            file_path = sym.get("file", "")
            if file_path:
                if file_path not in file_symbols:
                    file_symbols[file_path] = []
                file_symbols[file_path].append(sym)

        result = []
        total_loc = 0
        for file_info in files:
            path = file_info.get("path", "")
            syms = file_symbols.get(path, [])

            # Try to get LOC from multiple sources:
            # 1. Actual file line count (most accurate)
            # 2. Symbol end_line estimates (fallback)
            loc = 0

            # Method 1: Count actual file lines
            if path:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        loc = sum(1 for _ in f)
                except (OSError, IOError):
                    pass

            # Method 2: Fallback to symbol end_line estimates
            if loc == 0 and syms:
                max_line = 0
                for sym in syms:
                    # Check both end_line and line
                    end_line = sym.get("end_line") or sym.get("line") or 0
                    if end_line and end_line > max_line:
                        max_line = end_line
                loc = max_line

            total_loc += loc
            result.append({
                "path": path,
                "loc": loc,
                "symbols": len(syms),
                "classes": file_info.get("classes", 0),
                "methods": file_info.get("methods", 0),
                "functions": file_info.get("functions", 0),
                "exceeds_loc_threshold": loc > loc_threshold,
            })

        # Store total for summary
        self._total_loc = total_loc

        return result

    def _estimate_debt(
        self,
        summary: Dict[str, Any],
        hotspots: List[Dict[str, Any]],
        debt_rate: float
    ) -> Dict[str, Any]:
        """Estimate technical debt based on metrics."""

        # Debt factors:
        # 1. High complexity methods (need refactoring)
        complexity_issues = len(hotspots)

        # 2. Maintainability issues (from MI score)
        mi = summary.get("maintainability_index", 100)
        mi_issues = 0
        if mi < 40:
            mi_issues = 10  # Poor maintainability
        elif mi < 60:
            mi_issues = 5   # Moderate issues
        elif mi < 80:
            mi_issues = 2   # Minor issues

        # 3. Size-based estimation from total complexity
        total_cc = summary.get("complexity", {}).get("total", 0)
        size_factor = total_cc / 100  # Rough factor

        total_issues = complexity_issues + mi_issues
        hours = total_issues * debt_rate + size_factor
        days = hours / 8

        return {
            "issues": total_issues,
            "complexity_issues": complexity_issues,
            "maintainability_issues": mi_issues,
            "hours": round(hours, 1),
            "days": round(days, 1),
        }
