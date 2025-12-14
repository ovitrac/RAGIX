"""
Kernel: Service Detection
Stage: 1 (Data Collection)
Wraps: ragix_audit.service_detector.ServiceDetector

Detects services and components in a codebase:
- Scans for annotation patterns (@Service, @Component)
- Identifies service IDs (SK01, SC02, SG03, spre##)
- Groups files by service
- Provides service metadata

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class ServicesKernel(Kernel):
    """
    Detect services and components in codebase.

    This kernel scans the codebase for service annotations and
    patterns to identify logical service boundaries.

    Configuration options:
        patterns: Additional patterns to search for
        use_ast: Use AST data for enhanced detection (default: true)
        min_confidence: Minimum confidence for service detection (default: 0.5)

    Dependencies:
        ast_scan: Provides symbol data for AST-based detection

    Output:
        services: Detected services with metadata
        summary: Count by service type
        statistics: Detection statistics
    """

    name = "services"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Detect services and components in codebase"

    requires = ["ast_scan"]
    provides = ["services", "service_map"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Detect services in the codebase."""

        # Import here to avoid circular imports
        from ragix_audit.service_detector import ServiceDetector, detect_services
        from ragix_core.dependencies import DependencyGraph

        # Get configuration
        custom_patterns = input.config.get("patterns", [])
        use_ast = input.config.get("use_ast", True)
        min_confidence = input.config.get("min_confidence", 0.5)

        # Get project path from ast_scan
        ast_scan_path = input.dependencies.get("ast_scan")
        if not ast_scan_path or not ast_scan_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")

        with open(ast_scan_path) as f:
            ast_data = json.load(f)

        project_path = ast_data.get("data", {}).get("project", {}).get("path", ".")

        # Optionally reconstruct graph for AST-based detection
        ast_graph = None
        if use_ast:
            symbols = ast_data.get("data", {}).get("symbols", [])
            dependencies = ast_data.get("data", {}).get("dependencies", [])
            if symbols:
                ast_graph = DependencyGraph.from_cached_data(symbols, dependencies)

        logger.info(f"[services] Detecting services in {project_path}")

        # Run detection
        audit_config = detect_services(
            project_path=project_path,
            rag_project=None,  # RAG enrichment not used in kernel
            ast_graph=ast_graph,
        )

        # Build services data
        services_data = {}
        for service_id, service in audit_config.services.items():
            if service.confidence >= min_confidence:
                services_data[service_id] = {
                    "id": service.id,
                    "type": service.type.value if hasattr(service.type, 'value') else str(service.type),
                    "name": service.name,
                    "description": service.description,
                    "confidence": service.confidence,
                    "files": service.files[:50],  # Limit to 50 files
                    "file_count": len(service.files),
                    "packages": list(service.packages)[:20],
                    "main_package": service.main_package,
                    "sources": [s.value if hasattr(s, 'value') else str(s) for s in service.sources],
                    "ast_classes": service.ast_classes[:20],
                    "annotations": service.annotations[:10],
                }

        # Group by type
        by_type: Dict[str, List[str]] = {}
        for service_id, data in services_data.items():
            t = data.get("type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(service_id)

        # Statistics
        statistics = {
            "total_services": len(services_data),
            "by_type": {t: len(ids) for t, ids in by_type.items()},
            "high_confidence": sum(1 for s in services_data.values() if s["confidence"] >= 0.8),
            "medium_confidence": sum(1 for s in services_data.values() if 0.5 <= s["confidence"] < 0.8),
            "total_files": audit_config.total_files,
            "sources_used": audit_config.sources_available,
        }

        return {
            "services": services_data,
            "by_type": by_type,
            "statistics": statistics,
            "project": {
                "path": project_path,
                "name": audit_config.project_name,
                "total_packages": audit_config.total_packages,
                "languages": audit_config.languages,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        services = data.get("services", {})
        by_type = data.get("by_type", {})

        # List top services
        top_services = sorted(
            services.items(),
            key=lambda x: x[1].get("file_count", 0),
            reverse=True
        )[:5]
        top_str = ", ".join(f"{s[0]}({s[1].get('file_count', 0)} files)" for s in top_services)

        return (
            f"Detected {stats.get('total_services', 0)} services. "
            f"Types: {', '.join(f'{t}:{len(ids)}' for t, ids in by_type.items())}. "
            f"High confidence: {stats.get('high_confidence', 0)}. "
            f"Top services: {top_str if top_str else 'none'}."
        )
