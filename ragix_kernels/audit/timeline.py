"""
Kernel: Timeline Analysis
Stage: 1 (Data Collection)
Wraps: ragix_audit.timeline.TimelineScanner

Builds component lifecycle profiles from file timestamps:
- File modification history (mtime/ctime)
- Component lifecycle classification (NEW, ACTIVE, MATURE, LEGACY_HOT, FROZEN)
- Volatility estimation (change frequency)
- Version tracking from Javadoc annotations

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class TimelineKernel(Kernel):
    """
    Build component timeline profiles using TimelineScanner.

    This kernel wraps ragix_audit.timeline.TimelineScanner to analyze
    file timestamps and build lifecycle profiles for components.

    Configuration options:
        project.path: Path to project directory (required)
        extensions: File extensions to scan (default: code/doc types)

    Dependencies:
        ast_scan: Provides project path information

    Output:
        file_timelines: Per-file timeline data
        component_timelines: Aggregated component timelines
        lifecycle_distribution: Count by lifecycle category
        statistics: Summary statistics
    """

    name = "timeline"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Build component lifecycle profiles"

    requires = ["ast_scan"]
    provides = ["file_timelines", "component_timelines", "lifecycle"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Build timeline profiles using TimelineScanner."""

        # Import existing tools (REUSE!)
        from ragix_audit.timeline import TimelineScanner, LifecycleCategory

        # Get configuration
        project_config = input.config.get("project", {})
        extensions = input.config.get("extensions", None)

        # Load ast_scan to get project path
        ast_scan_path = input.dependencies.get("ast_scan")
        if not ast_scan_path or not ast_scan_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")

        with open(ast_scan_path) as f:
            ast_data = json.load(f).get("data", {})

        project_path = project_config.get("path") or ast_data.get("project", {}).get("path", ".")
        project_path = Path(project_path)

        if not project_path.exists():
            raise RuntimeError(f"Project path does not exist: {project_path}")

        logger.info(f"[timeline] Scanning {project_path} for timeline data")

        # Use existing TimelineScanner (REUSE!)
        scanner = TimelineScanner()
        file_timelines = scanner.scan_directory(project_path, extensions)
        component_timelines = scanner.build_component_timelines()

        # Convert to JSON-serializable format
        file_timelines_data = []
        for ft in file_timelines.values():
            file_timelines_data.append(ft.to_dict())

        component_timelines_data = []
        for ct in component_timelines.values():
            component_timelines_data.append(ct.to_dict())

        # Lifecycle distribution
        lifecycle_distribution = {}
        for category in LifecycleCategory:
            count = sum(1 for ct in component_timelines.values() if ct.category == category)
            if count > 0:
                lifecycle_distribution[category.value] = count

        # Age distribution
        age_groups = {
            "new_6mo": 0,
            "young_1yr": 0,
            "mature_3yr": 0,
            "legacy_3yr_plus": 0,
        }
        for ct in component_timelines.values():
            if ct.age_days < 180:
                age_groups["new_6mo"] += 1
            elif ct.age_days < 365:
                age_groups["young_1yr"] += 1
            elif ct.age_days < 1095:
                age_groups["mature_3yr"] += 1
            else:
                age_groups["legacy_3yr_plus"] += 1

        # Find high-volatility components (potential MCO risk)
        high_volatility = [
            {
                "component": ct.component_id,
                "volatility": round(ct.volatility, 3),
                "age_years": round(ct.age_years, 1),
                "category": ct.category.value,
            }
            for ct in sorted(component_timelines.values(), key=lambda x: -x.volatility)[:20]
            if ct.volatility > 0.5
        ]

        # Find legacy hot components (high MCO risk)
        legacy_hot = [
            ct.to_dict()
            for ct in component_timelines.values()
            if ct.category == LifecycleCategory.LEGACY_HOT
        ]

        # Statistics
        statistics = {
            "total_files_scanned": len(file_timelines),
            "total_components": len(component_timelines),
            "lifecycle_distribution": lifecycle_distribution,
            "age_distribution": age_groups,
            "high_volatility_count": len(high_volatility),
            "legacy_hot_count": len(legacy_hot),
            "avg_volatility": round(
                sum(ct.volatility for ct in component_timelines.values()) / len(component_timelines)
                if component_timelines else 0, 3
            ),
        }

        return {
            "file_timelines": file_timelines_data[:500],  # Limit output
            "component_timelines": component_timelines_data,
            "lifecycle_distribution": lifecycle_distribution,
            "high_volatility": high_volatility,
            "legacy_hot": legacy_hot,
            "age_distribution": age_groups,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        lifecycle = data.get("lifecycle_distribution", {})

        # Format lifecycle distribution
        lifecycle_str = ", ".join(f"{k}:{v}" for k, v in lifecycle.items())

        return (
            f"Timeline: {stats.get('total_components', 0)} components, "
            f"{stats.get('total_files_scanned', 0)} files. "
            f"Lifecycle: {lifecycle_str}. "
            f"Legacy hot (MCO risk): {stats.get('legacy_hot_count', 0)}. "
            f"High volatility: {stats.get('high_volatility_count', 0)}. "
            f"Avg volatility: {stats.get('avg_volatility', 0):.2f} changes/month."
        )
