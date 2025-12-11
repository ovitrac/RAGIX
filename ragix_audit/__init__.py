"""
RAGIX Audit Module — Code Audit with RAG+AST+Timeline Integration

This module provides enterprise code audit capabilities:
- Timeline analysis (service life profiles without git)
- RAG↔AST connection for traceability
- Service Life Risk Scoring
- Spec-Code Drift Detection
- MCO Estimation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from ragix_core.version import __version__

__all__ = [
    # Timeline
    "FileTimeline",
    "ComponentTimeline",
    "TimelineScanner",
    "LifecycleCategory",
    # Component mapping
    "ComponentMapper",
    "ComponentType",
    # Risk scoring
    "RiskScorer",
    "RiskLevel",
    "ServiceLifeRisk",
    # Drift detection
    "DriftAnalyzer",
    "DriftType",
    "DriftReport",
    # Service detection (auto-config)
    "ServiceDetector",
    "DetectedService",
    "AuditConfig",
    "detect_services",
    # Reports
    "AuditReportGenerator",
    # Statistics (v0.5)
    "DistributionStats",
    "ComponentStats",
    "CodebaseStats",
    "StatisticsComputer",
    # Entropy (v0.5)
    "EntropyMetrics",
    "EntropyComputer",
    "shannon_entropy",
    "gini_coefficient",
    # Coupling (v0.5)
    "PackageCoupling",
    "CouplingAnalysis",
    "CouplingComputer",
    "SDPViolation",
    "PropagationAnalysis",
    "compute_propagation_factor",
    # Dead Code Detection (v0.5)
    "DeadCodeDetector",
    "DeadCodeAnalysis",
    "EntryPointDetector",
    "EntryPoint",
    "DeadCodeCandidate",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("FileTimeline", "ComponentTimeline", "TimelineScanner", "LifecycleCategory"):
        from ragix_audit.timeline import FileTimeline, ComponentTimeline, TimelineScanner, LifecycleCategory
        return locals()[name]
    elif name in ("ComponentMapper", "ComponentType"):
        from ragix_audit.component_mapper import ComponentMapper, ComponentType
        return locals()[name]
    elif name in ("RiskScorer", "RiskLevel", "ServiceLifeRisk"):
        from ragix_audit.risk import RiskScorer, RiskLevel, ServiceLifeRisk
        return locals()[name]
    elif name in ("DriftAnalyzer", "DriftType", "DriftReport"):
        from ragix_audit.drift import DriftAnalyzer, DriftType, DriftReport
        return locals()[name]
    elif name in ("ServiceDetector", "DetectedService", "AuditConfig", "detect_services"):
        from ragix_audit.service_detector import ServiceDetector, DetectedService, AuditConfig, detect_services
        return locals()[name]
    elif name == "AuditReportGenerator":
        from ragix_audit.reports import AuditReportGenerator
        return AuditReportGenerator
    # Statistics (v0.5)
    elif name in ("DistributionStats", "ComponentStats", "CodebaseStats", "StatisticsComputer"):
        from ragix_audit.statistics import DistributionStats, ComponentStats, CodebaseStats, StatisticsComputer
        return locals()[name]
    # Entropy (v0.5)
    elif name in ("EntropyMetrics", "EntropyComputer", "shannon_entropy", "gini_coefficient"):
        from ragix_audit.entropy import EntropyMetrics, EntropyComputer, shannon_entropy, gini_coefficient
        return locals()[name]
    # Coupling (v0.5)
    elif name in ("PackageCoupling", "CouplingAnalysis", "CouplingComputer", "SDPViolation", "PropagationAnalysis", "compute_propagation_factor"):
        from ragix_audit.coupling import PackageCoupling, CouplingAnalysis, CouplingComputer, SDPViolation, PropagationAnalysis, compute_propagation_factor
        return locals()[name]
    # Dead Code Detection (v0.5)
    elif name in ("DeadCodeDetector", "DeadCodeAnalysis", "EntryPointDetector", "EntryPoint", "DeadCodeCandidate"):
        from ragix_audit.dead_code import DeadCodeDetector, DeadCodeAnalysis, EntryPointDetector, EntryPoint, DeadCodeCandidate
        return locals()[name]
    raise AttributeError(f"module 'ragix_audit' has no attribute '{name}'")
