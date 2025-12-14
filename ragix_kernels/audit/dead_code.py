"""
Kernel: Dead Code Detection
Stage: 2 (Analysis)
Wraps: ragix_audit.dead_code.DeadCodeDetector

Detects potentially dead/unused code through:
- Entry point discovery (main, controllers, event handlers)
- BFS reachability analysis from entry points
- Orphan package detection
- Unused class identification

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DeadCodeKernel(Kernel):
    """
    Detect dead/unused code using DeadCodeDetector.

    This kernel wraps ragix_audit.dead_code.DeadCodeDetector to perform
    reachability-based dead code detection from entry points.

    Configuration options:
        include_tests: Include test files in analysis (default: false)

    Dependencies:
        ast_scan: Symbol and file data
        dependency: Dependency graph

    Output:
        entry_points: Detected application entry points
        reachability: Reachability statistics from entry points
        dead_candidates: Unreachable code candidates
        orphan_packages: Packages with no incoming dependencies
        statistics: Summary statistics
    """

    name = "dead_code"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Detect dead/unused code via reachability analysis"

    requires = ["ast_scan", "dependency"]
    provides = ["dead_code", "entry_points", "orphan_packages"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Detect dead code using DeadCodeDetector."""

        # Import existing tool (REUSE!)
        from ragix_audit.dead_code import DeadCodeDetector

        # Load dependencies
        ast_path = input.dependencies.get("ast_scan")
        dep_path = input.dependencies.get("dependency")

        if not ast_path or not ast_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")
        if not dep_path or not dep_path.exists():
            raise RuntimeError("Missing required dependency: dependency")

        with open(ast_path) as f:
            ast_data = json.load(f).get("data", {})
        with open(dep_path) as f:
            dep_data = json.load(f).get("data", {})

        # Get configuration
        include_tests = input.config.get("include_tests", False)

        # Extract files from ast_scan
        files = ast_data.get("files", [])
        symbols = ast_data.get("symbols", [])

        logger.info(f"[dead_code] Analyzing {len(files)} files for dead code")

        # Build file_contents dict for DeadCodeDetector
        # Note: We don't have actual file contents, so we'll synthesize
        # minimal content from symbols for entry point detection
        file_contents: Dict[str, str] = {}
        class_to_file: Dict[str, str] = {}

        for sym in symbols:
            file_path = sym.get("file", "")
            qname = sym.get("qualified_name", "")
            sym_type = sym.get("type", "")

            if not include_tests:
                # Skip test files
                if any(p in file_path for p in ["/test/", "/tests/", "Test.", "_test."]):
                    continue

            # Build class-to-file mapping
            if sym_type == "class" and qname:
                class_to_file[qname] = file_path

            # Synthesize content for entry point detection
            # This is a limitation - ideally we'd have actual file content
            if file_path not in file_contents:
                file_contents[file_path] = ""

            # Add symbol hints for entry point detection
            if sym_type == "class":
                name = sym.get("name", "")
                # Add patterns that help detect entry points
                if "Main" in name or "Application" in name:
                    file_contents[file_path] += f"\npublic class {name}"
                    file_contents[file_path] += "\npublic static void main("
                if "Controller" in name:
                    file_contents[file_path] += f"\n@Controller\npublic class {name}"
                if "Test" in name:
                    file_contents[file_path] += f"\n@Test\npublic class {name}"

        # Build dependency graph from dependency kernel output
        # Format: {class_name: {referenced_classes}}
        dependencies: Dict[str, Set[str]] = {}
        graph_edges = dep_data.get("graph", {}).get("edges", [])

        for edge in graph_edges:
            source = edge.get("source", "")
            target = edge.get("target", "")

            if source not in dependencies:
                dependencies[source] = set()
            dependencies[source].add(target)

        # Use existing DeadCodeDetector (REUSE!)
        detector = DeadCodeDetector()
        analysis = detector.analyze(
            file_contents=file_contents,
            dependencies=dependencies,
            class_to_file=class_to_file,
        )

        # Convert to JSON-serializable format using the analysis's to_dict()
        analysis_dict = analysis.to_dict()

        # Extract structured data for kernel output
        entry_points_data = []
        for ep in analysis.entry_points:
            entry_points_data.append(ep.to_dict())

        dead_candidates_data = []
        for dc in analysis.dead_candidates:
            dead_candidates_data.append(dc.to_dict())

        # Group by confidence
        high_confidence = [d for d in dead_candidates_data if d.get("confidence", 0) >= 0.7]
        medium_confidence = [d for d in dead_candidates_data if 0.5 <= d.get("confidence", 0) < 0.7]
        low_confidence = [d for d in dead_candidates_data if d.get("confidence", 0) < 0.5]

        # Statistics
        statistics = {
            "total_files_analyzed": len(file_contents),
            "total_classes": analysis.total_classes,
            "total_packages": analysis.total_packages,
            "entry_points_found": len(analysis.entry_points),
            "entry_points_by_type": analysis.entry_points_by_type,
            "reachable_classes": len(analysis.reachable_classes),
            "reachability_ratio": round(analysis.reachability_ratio, 3),
            "dead_candidates": len(analysis.dead_candidates),
            "orphan_packages": len(analysis.orphan_packages),
            "high_confidence": len(high_confidence),
            "medium_confidence": len(medium_confidence),
            "low_confidence": len(low_confidence),
        }

        return {
            "entry_points": entry_points_data[:50],  # Limit for output size
            "reachability": {
                "reachable_classes": len(analysis.reachable_classes),
                "total_classes": analysis.total_classes,
                "ratio": round(analysis.reachability_ratio, 3),
                "reachable_packages": len(analysis.reachable_packages),
                "total_packages": analysis.total_packages,
            },
            "dead_candidates": dead_candidates_data[:100],  # Limit for output size
            "orphan_packages": analysis.orphan_packages[:50],
            "confidence_levels": {
                "high": high_confidence[:30],
                "medium": medium_confidence[:30],
                "low": low_confidence[:20],
            },
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        reachability = data.get("reachability", {})

        ep_types = stats.get("entry_points_by_type", {})
        ep_str = ", ".join(f"{t}:{c}" for t, c in ep_types.items()) if ep_types else "none"

        return (
            f"Dead code analysis: {stats.get('total_classes', 0)} classes. "
            f"Entry points: {stats.get('entry_points_found', 0)} ({ep_str}). "
            f"Reachability: {reachability.get('ratio', 0)*100:.1f}% "
            f"({reachability.get('reachable_classes', 0)}/{reachability.get('total_classes', 0)}). "
            f"Dead candidates: {stats.get('dead_candidates', 0)} "
            f"(high:{stats.get('high_confidence', 0)}). "
            f"Orphan pkgs: {stats.get('orphan_packages', 0)}."
        )
