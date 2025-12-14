"""
RAGIX Audit Kernels — Code Audit Stage 1, 2, and 3 Kernels

Stage 1: Data Collection
- ast_scan: AST extraction and symbol enumeration
- metrics: Code metrics (CC, LOC, MI)
- dependency: Dependency graph construction
- partition: Codebase partitioning
- services: Service detection
- timeline: Component lifecycle profiles (wraps TimelineScanner)

Stage 2: Analysis
- stats_summary: Statistical aggregation (wraps StatisticsComputer)
- hotspots: Complexity hotspot identification
- dead_code: Dead code detection (wraps DeadCodeDetector)
- coupling: Coupling metrics (wraps CouplingComputer)
- entropy: Information-theoretic analysis (wraps EntropyComputer)
- risk: Service Life Risk assessment (wraps RiskScorer)

Stage 3: Reporting
- section_*: Report section generators
- report_assemble: Final report assembly

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

# Kernels are registered automatically by the registry
# via package discovery. No explicit imports needed here.

__all__ = []  # Auto-populated by registry
