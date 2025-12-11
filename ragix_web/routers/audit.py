"""
Audit Router — API endpoints for code audit with timeline and risk analysis

Provides REST API for:
- Timeline analysis (component lifecycle)
- Risk scoring (service life risk)
- Drift detection (spec-code misalignment)
- Full audit reports

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit", tags=["audit"])

# Global state for current project
_current_project_path: Optional[str] = None
_cached_results: Dict[str, Any] = {}


def set_current_project(path: str):
    """Set the current project path for audit analysis."""
    global _current_project_path, _cached_results
    if _current_project_path != path:
        _current_project_path = path
        _cached_results = {}  # Clear cache on project change


# =============================================================================
# Request/Response Models
# =============================================================================

class TimelineScanRequest(BaseModel):
    project_path: Optional[str] = None
    extensions: Optional[List[str]] = None


class RiskScorerConfig(BaseModel):
    weights: Optional[Dict[str, float]] = None


# =============================================================================
# Timeline Endpoints
# =============================================================================

def _get_source_path(project_path: Path) -> Path:
    """
    Get the source path for a project, handling multi-module Maven structures.

    For projects like SIAS where src/ exists but contains no code,
    scans from project root instead.
    """
    src_path = project_path / "src"
    if src_path.exists():
        # Check if src has Java files
        java_in_src = list(src_path.rglob("*.java"))[:1]
        if not java_in_src:
            # Multi-module project: scan from root
            return project_path
        return src_path
    return project_path


@router.get("/timeline")
async def get_timeline(
    project_path: Optional[str] = Query(None, description="Project path"),
    refresh: bool = Query(False, description="Force refresh cache"),
) -> Dict[str, Any]:
    """
    Get timeline analysis for a project.

    Returns component lifecycle profiles with age, volatility, and categories.
    """
    from ragix_audit.timeline import TimelineScanner

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    cache_key = f"timeline:{path}"
    if not refresh and cache_key in _cached_results:
        return _cached_results[cache_key]

    try:
        scanner = TimelineScanner()
        src_path = _get_source_path(Path(path))
        scanner.scan_directory(src_path)
        timelines = scanner.build_component_timelines()

        result = {
            "project_path": path,
            "summary": scanner.get_summary(),
            "components": {k: v.to_dict() for k, v in timelines.items()},
        }

        _cached_results[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Timeline scan failed: {e}")
        raise HTTPException(500, f"Timeline scan failed: {e}")


@router.get("/timeline/component/{component_id}")
async def get_component_timeline(
    component_id: str,
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Get detailed timeline for a specific component."""
    result = await get_timeline(project_path)

    comp_id = component_id.upper()
    if comp_id not in result["components"]:
        raise HTTPException(404, f"Component {comp_id} not found")

    return result["components"][comp_id]


# =============================================================================
# Risk Endpoints
# =============================================================================

@router.get("/risk")
async def get_risk_analysis(
    project_path: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    """
    Get risk analysis for all components.

    Returns service life risk scores with factor breakdown and recommendations.
    """
    from ragix_audit.timeline import TimelineScanner
    from ragix_audit.risk import RiskScorer
    from ragix_audit.drift import DriftAnalyzer

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    cache_key = f"risk:{path}"
    if not refresh and cache_key in _cached_results:
        return _cached_results[cache_key]

    try:
        # Get timelines
        scanner = TimelineScanner()
        src_path = _get_source_path(Path(path))
        scanner.scan_directory(src_path)
        timelines = scanner.build_component_timelines()

        # Get drift analysis for doc gap scores
        analyzer = DriftAnalyzer()
        analyzer.scan_docs(Path(path))
        drift_reports = analyzer.analyze_all(timelines)
        gap_scores = analyzer.get_gap_scores(drift_reports)

        # Score risks
        scorer = RiskScorer()
        risks = scorer.score_all(timelines, gap_scores)

        result = {
            "project_path": path,
            "summary": scorer.get_risk_summary(risks),
            "risks": {k: v.to_dict() for k, v in risks.items()},
        }

        _cached_results[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise HTTPException(500, f"Risk analysis failed: {e}")


@router.get("/risk/component/{component_id}")
async def get_component_risk(
    component_id: str,
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Get detailed risk analysis for a specific component."""
    result = await get_risk_analysis(project_path)

    comp_id = component_id.upper()
    if comp_id not in result["risks"]:
        raise HTTPException(404, f"Component {comp_id} not found")

    return result["risks"][comp_id]


@router.get("/risk/matrix")
async def get_risk_matrix(
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Get risk matrix data for visualization.

    Returns components positioned by impact (x) and complexity (y).
    """
    result = await get_risk_analysis(project_path)

    matrix_data = []
    for comp_id, risk in result["risks"].items():
        factors = risk["factors"]
        matrix_data.append({
            "id": comp_id,
            "x": factors["impact"],           # X-axis: impact
            "y": factors["complexity"],       # Y-axis: complexity
            "size": factors["volatility"],    # Size: volatility
            "color": risk["level"],           # Color: risk level
            "score": risk["score"],
            "category": risk["timeline"]["category"] if risk["timeline"] else "unknown",
        })

    return {
        "project_path": result["project_path"],
        "matrix": matrix_data,
        "axes": {
            "x": {"label": "Impact", "min": 0, "max": 1},
            "y": {"label": "Complexity", "min": 0, "max": 1},
        },
    }


# =============================================================================
# Drift Endpoints
# =============================================================================

@router.get("/drift")
async def get_drift_analysis(
    project_path: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    """
    Get spec-code drift analysis.

    Detects misalignment between code changes and documentation updates.
    """
    from ragix_audit.timeline import TimelineScanner
    from ragix_audit.drift import DriftAnalyzer

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    cache_key = f"drift:{path}"
    if not refresh and cache_key in _cached_results:
        return _cached_results[cache_key]

    try:
        # Get timelines
        scanner = TimelineScanner()
        src_path = _get_source_path(Path(path))
        scanner.scan_directory(src_path)
        timelines = scanner.build_component_timelines()

        # Analyze drift
        analyzer = DriftAnalyzer()
        analyzer.scan_docs(Path(path))
        drift_reports = analyzer.analyze_all(timelines)

        result = {
            "project_path": path,
            "summary": analyzer.get_summary(drift_reports),
            "reports": {k: v.to_dict() for k, v in drift_reports.items()},
        }

        _cached_results[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Drift analysis failed: {e}")
        raise HTTPException(500, f"Drift analysis failed: {e}")


@router.get("/drift/alerts")
async def get_drift_alerts(
    project_path: Optional[str] = Query(None),
    min_severity: str = Query("warning", description="Minimum severity: info, warning, error, critical"),
) -> Dict[str, Any]:
    """Get drift alerts filtered by severity."""
    from ragix_audit.drift import Severity

    result = await get_drift_analysis(project_path)

    severity_order = ["info", "warning", "error", "critical"]
    min_idx = severity_order.index(min_severity.lower()) if min_severity.lower() in severity_order else 0

    alerts = []
    for comp_id, report in result["reports"].items():
        sev_idx = severity_order.index(report["severity"]) if report["severity"] in severity_order else 0
        if sev_idx >= min_idx:
            alerts.append({
                "component_id": comp_id,
                "drift_type": report["drift_type"],
                "severity": report["severity"],
                "message": report["message"],
                "drift_days": report["drift_days"],
                "gap_score": report["gap_score"],
            })

    alerts.sort(key=lambda x: (severity_order.index(x["severity"]), -x["drift_days"]), reverse=True)

    return {
        "project_path": result["project_path"],
        "min_severity": min_severity,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


# =============================================================================
# Full Audit Endpoint
# =============================================================================

@router.get("/full")
async def get_full_audit(
    project_path: Optional[str] = Query(None),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    """
    Get comprehensive audit analysis.

    Combines timeline, risk, and drift analysis into a single report.
    """
    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    # Get all analyses
    timeline_result = await get_timeline(path, refresh)
    risk_result = await get_risk_analysis(path, refresh)
    drift_result = await get_drift_analysis(path, refresh)

    # Combine into full audit
    return {
        "project_path": path,
        "audit_date": __import__("datetime").datetime.now().isoformat(),
        "summary": {
            "total_components": timeline_result["summary"]["total_components"],
            "total_files": timeline_result["summary"]["total_files"],
            "risk_summary": risk_result["summary"],
            "drift_summary": drift_result["summary"],
        },
        "timeline": timeline_result,
        "risk": risk_result,
        "drift": drift_result,
    }


@router.post("/clear-cache")
async def clear_audit_cache() -> Dict[str, str]:
    """Clear all cached audit results."""
    global _cached_results
    count = len(_cached_results)
    _cached_results = {}
    return {"status": "ok", "cleared": count}


# =============================================================================
# Service Detection Endpoints (Auto-config)
# =============================================================================

@router.get("/detect-services")
async def api_detect_services(
    project_path: Optional[str] = Query(None, description="Project path to analyze"),
    use_rag: bool = Query(True, description="Use RAG index if available"),
    use_ast: bool = Query(True, description="Use AST analysis if available"),
    refresh: bool = Query(False, description="Force re-detection"),
) -> Dict[str, Any]:
    """
    Auto-detect services/components in a project.

    Combines multiple data sources:
    - Filesystem patterns (paths, filenames)
    - RAG index (concepts, chunks) - if available and enabled
    - AST analysis (classes, packages) - if available and enabled
    - Content patterns (annotations, javadoc)

    Returns an AuditConfig with detected services and suggested configuration.
    """
    from ragix_audit.service_detector import (
        ServiceDetector as SvcDetector,
        load_rag_project,
        load_ast_graph,
    )

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    cache_key = f"detect:{path}:{use_rag}:{use_ast}"
    if not refresh and cache_key in _cached_results:
        return _cached_results[cache_key]

    try:
        # Load optional data sources
        rag_project = None
        ast_graph = None

        if use_rag:
            rag_project = load_rag_project(path)

        if use_ast:
            ast_graph = load_ast_graph(path)

        # Run detection
        detector = SvcDetector(path, rag_project, ast_graph)
        config = detector.detect()

        result = config.to_dict()
        _cached_results[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Service detection failed: {e}")
        raise HTTPException(500, f"Service detection failed: {e}")


@router.get("/config")
async def get_audit_config(
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Get current audit configuration for a project.

    Returns detected services (if available) plus tunable parameters.
    """
    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    # Check if we have cached detection
    cache_key = f"detect:{path}:True:True"
    detected = _cached_results.get(cache_key)

    # Default thresholds
    thresholds = {
        "drift_days": 90,
        "new_component_days": 180,
        "legacy_years": 3,
    }

    # Default risk weights
    risk_weights = {
        "volatility": 0.20,
        "impact": 0.25,
        "complexity": 0.20,
        "maturity": 0.25,
        "doc_gap": 0.10,
    }

    return {
        "project_path": path,
        "has_detection": detected is not None,
        "services_count": len(detected.get("services", {})) if detected else 0,
        "sources_available": detected.get("sources_available", []) if detected else [],
        "thresholds": thresholds,
        "risk_weights": risk_weights,
        "detection_summary": detected.get("summary") if detected else None,
    }


class AuditConfigUpdate(BaseModel):
    """Request body for config update."""
    thresholds: Optional[Dict[str, Any]] = None
    risk_weights: Optional[Dict[str, float]] = None


@router.post("/config")
async def update_audit_config(
    update: AuditConfigUpdate,
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Update audit configuration (session-scoped).

    Allows tuning thresholds and risk weights for the current session.
    """
    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    # Validate weights sum to ~1.0 if provided
    if update.risk_weights:
        total = sum(update.risk_weights.values())
        if abs(total - 1.0) > 0.1:
            raise HTTPException(400, f"Risk weights should sum to ~1.0, got {total:.2f}")

    # Store in session-scoped cache
    config_key = f"config:{path}"
    current = _cached_results.get(config_key, {})

    if update.thresholds:
        current["thresholds"] = {**current.get("thresholds", {}), **update.thresholds}

    if update.risk_weights:
        current["risk_weights"] = update.risk_weights

    _cached_results[config_key] = current

    # Clear dependent caches (risk analysis depends on config)
    for key in list(_cached_results.keys()):
        if key.startswith(f"risk:{path}"):
            del _cached_results[key]

    return {
        "status": "ok",
        "config": current,
        "note": "Risk analysis cache cleared - next request will use new config",
    }


@router.get("/sources-status")
async def get_sources_status(
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Check availability of data sources for a project.

    Returns status of:
    - Filesystem (always available)
    - RAG index (from Project RAG status API)
    - AST cache (from AST cache API)
    """
    from pathlib import Path as PathLib

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    project = PathLib(path)
    sources = {
        "filesystem": {
            "available": True,
            "status": "ready",
        },
        "rag": {
            "available": False,
            "status": "not_indexed",
            "file_count": 0,
        },
        "ast": {
            "available": False,
            "status": "not_cached",
            "node_count": 0,
        },
    }

    # Check RAG - use RAGProject API for accurate status
    rag_roots_to_check = [project, project / "src"]

    for rag_root in rag_roots_to_check:
        rag_dir = rag_root / ".RAG"
        if rag_dir.exists():
            try:
                from ragix_core.rag_project import RAGProject
                rag_project = RAGProject(str(rag_root))
                if rag_project.exists():
                    status = rag_project.get_status()
                    state = status.get("state", {})
                    sources["rag"]["available"] = True
                    sources["rag"]["status"] = "indexed" if status.get("initialized") else "initializing"
                    sources["rag"]["path"] = str(rag_dir)
                    sources["rag"]["file_count"] = state.get("files_indexed", 0)
                    sources["rag"]["chunk_count"] = state.get("chunks_indexed", 0)
                    break
            except Exception:
                # Fallback: just check if directory exists
                chroma_dir = rag_dir / "chroma"
                if chroma_dir.exists():
                    sources["rag"]["available"] = True
                    sources["rag"]["status"] = "indexed"
                    sources["rag"]["path"] = str(rag_dir)
                    break

    # Also check if current Project RAG matches this project
    if not sources["rag"]["available"]:
        try:
            from ragix_web.routers.rag_project import get_current_project
            current_rag = get_current_project()
            if current_rag:
                rag_root = PathLib(current_rag).resolve()
                project_resolved = project.resolve()
                if (rag_root == project_resolved or
                    rag_root == project_resolved / "src" or
                    rag_root.parent == project_resolved):
                    sources["rag"]["available"] = True
                    sources["rag"]["status"] = "active"
                    sources["rag"]["path"] = str(current_rag)
        except Exception:
            pass

    # Check AST cache - check both project path and project/src
    ast_paths_to_check = [project.resolve(), (project / "src").resolve()]

    try:
        from ragix_core.analysis_cache import get_cache
        cache = get_cache()

        # Method 1: Check cache index for matching project paths
        cached_entries = cache.list_cached()
        for entry in cached_entries:
            entry_path = PathLib(entry.get("project_path", "")).resolve()
            if entry_path in ast_paths_to_check:
                sources["ast"]["available"] = True
                sources["ast"]["status"] = "cached"
                sources["ast"]["path"] = str(entry_path)
                sources["ast"]["file_count"] = entry.get("file_count", 0)
                # Load actual analysis to get node count
                fingerprint = entry.get("fingerprint")
                if fingerprint:
                    analysis = cache.load(fingerprint)
                    if analysis:
                        sources["ast"]["node_count"] = len(analysis.symbols)
                break

        # Method 2: If not found in index, compute fingerprint and check
        if not sources["ast"]["available"]:
            for ast_path in ast_paths_to_check:
                if ast_path.exists():
                    fingerprint, file_count, _ = cache.get_fingerprint(ast_path)
                    if cache.is_cached(fingerprint):
                        sources["ast"]["available"] = True
                        sources["ast"]["status"] = "cached"
                        sources["ast"]["path"] = str(ast_path)
                        sources["ast"]["file_count"] = file_count
                        analysis = cache.load(fingerprint)
                        if analysis:
                            sources["ast"]["node_count"] = len(analysis.symbols)
                        break
    except Exception as e:
        logger.debug(f"AST cache check failed: {e}")

    return {
        "project_path": path,
        "sources": sources,
        "recommendation": _get_source_recommendation(sources),
    }


def _get_source_recommendation(sources: Dict) -> str:
    """Generate recommendation based on available sources."""
    available = [k for k, v in sources.items() if v.get("available")]

    if len(available) >= 3:
        return "All sources available - full detection capability"
    elif "rag" in available:
        return "RAG available - good detection with semantic search"
    elif "ast" in available:
        return "AST available - good detection with code analysis"
    else:
        return "Filesystem only - consider indexing with Project RAG for better detection"


# =============================================================================
# AI Insights Endpoint
# =============================================================================

class InsightFocus(str, Enum):
    """Focus modes for AI insights."""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    RECOMMENDATIONS = "recommendations"


class InsightRequest(BaseModel):
    """Request for AI insights generation."""
    focus: str = "executive"
    custom_prompt: Optional[str] = None


@router.post("/insights")
async def generate_insights(
    request: InsightRequest,
    project_path: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Generate AI-powered insights from audit data.

    Focus modes:
    - executive: High-level summary for stakeholders
    - technical: Detailed technical analysis for developers
    - recommendations: Prioritized action items and roadmap
    """
    from datetime import datetime

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    # Gather all audit data
    try:
        timeline_data = await get_timeline(path)
        risk_data = await get_risk_analysis(path)
        drift_data = await get_drift_analysis(path)
    except Exception as e:
        raise HTTPException(500, f"Failed to gather audit data: {e}")

    # Check for detected services
    detect_key = f"detect:{path}:True:True"
    detection_data = _cached_results.get(detect_key, {})

    # Build context for LLM
    context = _build_insight_context(
        path=path,
        timeline=timeline_data,
        risk=risk_data,
        drift=drift_data,
        detection=detection_data,
        focus=request.focus,
    )

    # Get LLM response
    try:
        from ragix_core.llm_backends import OllamaLLM

        # Use mistral model for insights (good balance of speed and quality)
        llm = OllamaLLM(model="mistral")

        system_prompt = _get_insight_system_prompt(request.focus)
        user_prompt = context

        if request.custom_prompt:
            user_prompt += f"\n\n## Additional Focus\n{request.custom_prompt}"

        response = llm.generate(
            system_prompt=system_prompt,
            history=[{"role": "user", "content": user_prompt}],
        )

        return {
            "project_path": path,
            "focus": request.focus,
            "generated_at": datetime.now().isoformat(),
            "insights": response,
            "data_summary": {
                "components": timeline_data.get("summary", {}).get("total_components", 0),
                "services_detected": len(detection_data.get("services", {})),
                "risk_critical": risk_data.get("summary", {}).get("critical", 0),
                "risk_high": risk_data.get("summary", {}).get("high", 0),
                "drift_alerts": drift_data.get("summary", {}).get("total_drifts", 0),
            },
        }

    except ImportError as e:
        logger.error(f"LLM backend not available: {e}")
        raise HTTPException(503, f"LLM backend not available: {e}. Ensure ragix_core.llm_backends is installed.")

    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        # Check if it's a connection error (Ollama not running)
        if "Connection" in str(e) or "connect" in str(e).lower():
            raise HTTPException(503, "Ollama not running. Start it with 'ollama serve'")
        raise HTTPException(500, f"Insight generation failed: {e}")


def _get_insight_system_prompt(focus: str) -> str:
    """Get system prompt based on focus mode."""
    base = """You are an expert software architect and technical auditor specializing in enterprise Java applications.
You analyze code audit data to provide actionable insights for development teams and stakeholders.
Be specific, cite data from the analysis, and prioritize findings by impact."""

    if focus == "executive":
        return base + """

For this EXECUTIVE SUMMARY:
- Lead with the most critical business-impacting findings
- Use clear, non-technical language where possible
- Quantify risks in terms of maintenance cost and project health
- Provide a clear health score interpretation
- Keep the summary concise (3-5 key points)
- End with strategic recommendations for leadership"""

    elif focus == "technical":
        return base + """

For this TECHNICAL ANALYSIS:
- Provide detailed technical assessment of each risk area
- Identify specific components/classes requiring attention
- Analyze architectural patterns and anti-patterns
- Discuss code quality metrics and their implications
- Reference specific timeline data (age, volatility, drift)
- Include dependency and coupling analysis where relevant"""

    elif focus == "recommendations":
        return base + """

For these RECOMMENDATIONS:
- Prioritize actions by impact vs effort (quick wins first)
- Group recommendations by category (refactoring, documentation, testing, etc.)
- Provide specific, actionable items with clear scope
- Suggest a phased approach for larger changes
- Estimate relative effort (low/medium/high)
- Include both immediate actions and long-term improvements"""

    return base


def _build_insight_context(
    path: str,
    timeline: Dict,
    risk: Dict,
    drift: Dict,
    detection: Dict,
    focus: str,
) -> str:
    """Build structured context for LLM prompt."""
    from pathlib import Path as PathLib

    project_name = PathLib(path).name
    lines = [
        f"# CODE AUDIT ANALYSIS: {project_name}",
        f"**Analysis Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Summary metrics
    t_summary = timeline.get("summary", {})
    r_summary = risk.get("summary", {})
    d_summary = drift.get("summary", {})

    lines.extend([
        "## OVERVIEW METRICS",
        f"- **Total Components:** {t_summary.get('total_components', 0)}",
        f"- **Total Files:** {t_summary.get('total_files', 0)}",
        f"- **Services Detected:** {len(detection.get('services', {}))}",
        "",
    ])

    # Risk summary
    lines.extend([
        "## RISK DISTRIBUTION",
        f"- **Critical:** {r_summary.get('critical', 0)}",
        f"- **High:** {r_summary.get('high', 0)}",
        f"- **Medium:** {r_summary.get('medium', 0)}",
        f"- **Low:** {r_summary.get('low', 0)}",
        f"- **Average Score:** {r_summary.get('average_score', 0):.2f}",
        "",
    ])

    # Top risks (limit to 10 for context efficiency)
    risks = risk.get("risks", {})
    if risks:
        sorted_risks = sorted(
            risks.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )[:10]

        lines.append("## TOP RISK COMPONENTS")
        lines.append("| Component | Score | Level | Key Factors |")
        lines.append("|-----------|-------|-------|-------------|")
        for comp_id, r in sorted_risks:
            factors = r.get("factors", {})
            key_factors = ", ".join([
                f"{k}={v:.2f}" for k, v in factors.items()
                if v > 0.5
            ][:3])
            lines.append(f"| {comp_id} | {r.get('score', 0):.2f} | {r.get('level', 'unknown')} | {key_factors} |")
        lines.append("")

    # Timeline categories
    components = timeline.get("components", {})
    if components:
        categories = {}
        for comp_id, comp in components.items():
            cat = comp.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        lines.append("## LIFECYCLE DISTRIBUTION")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat.replace('_', ' ').title()}:** {count}")
        lines.append("")

    # Drift alerts
    drift_reports = drift.get("reports", {})
    if drift_reports:
        high_drift = [
            (k, v) for k, v in drift_reports.items()
            if v.get("severity") in ("error", "critical")
        ][:5]

        if high_drift:
            lines.append("## CRITICAL DRIFT ALERTS")
            for comp_id, d in high_drift:
                lines.append(f"- **{comp_id}**: {d.get('message', 'Documentation drift detected')} ({d.get('drift_days', 0)} days)")
            lines.append("")

    # Detected services (if available)
    services = detection.get("services", {})
    if services:
        lines.append("## DETECTED SERVICES")
        lines.append("| Service | Type | Confidence |")
        lines.append("|---------|------|------------|")
        for svc_id, svc in list(services.items())[:15]:
            lines.append(f"| {svc_id} | {svc.get('service_type', 'unknown')} | {svc.get('confidence', 0):.0%} |")
        lines.append("")

    # Focus-specific request
    focus_prompts = {
        "executive": "Please provide an EXECUTIVE SUMMARY of this audit data, suitable for project stakeholders and management.",
        "technical": "Please provide a DETAILED TECHNICAL ANALYSIS of this audit data, suitable for the development team.",
        "recommendations": "Please provide PRIORITIZED RECOMMENDATIONS based on this audit data, with actionable items and effort estimates.",
    }

    lines.extend([
        "---",
        "",
        f"## REQUEST",
        focus_prompts.get(focus, focus_prompts["executive"]),
    ])

    return "\n".join(lines)


# =============================================================================
# Audit Report Generation
# =============================================================================


class AuditReportType(str, Enum):
    """Types of audit reports."""
    FULL = "full"
    RISK = "risk"
    TIMELINE = "timeline"
    DRIFT = "drift"


@router.get("/report")
async def generate_audit_report(
    project_path: Optional[str] = Query(None),
    report_type: str = Query("full", description="Report type: full, risk, timeline, drift"),
    include_insights: bool = Query(True, description="Include AI insights"),
    insight_focus: str = Query("executive", description="Insight focus: executive, technical, recommendations"),
) -> Dict[str, Any]:
    """
    Generate comprehensive audit report with optional AI insights.

    Returns structured data for rendering, including:
    - All audit analyses (timeline, risk, drift)
    - Detected services
    - AI-generated insights (optional)
    - Visualization data (for charts)
    """
    from datetime import datetime

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    report = {
        "project_path": path,
        "project_name": Path(path).name,
        "generated_at": datetime.now().isoformat(),
        "report_type": report_type,
    }

    try:
        # Always include detection data if available
        detect_key = f"detect:{path}:True:True"
        report["detection"] = _cached_results.get(detect_key, {})

        # Include analyses based on report type
        if report_type in ("full", "timeline"):
            report["timeline"] = await get_timeline(path)

        if report_type in ("full", "risk"):
            report["risk"] = await get_risk_analysis(path)
            # Add matrix data for visualization
            report["risk_matrix"] = await get_risk_matrix(path)

        if report_type in ("full", "drift"):
            report["drift"] = await get_drift_analysis(path)
            report["drift_alerts"] = await get_drift_alerts(path, min_severity="warning")

        # Generate AI insights if requested
        if include_insights:
            try:
                insight_request = InsightRequest(focus=insight_focus)
                insights_result = await generate_insights(insight_request, path)
                report["insights"] = insights_result
            except Exception as e:
                logger.warning(f"Could not generate insights: {e}")
                report["insights"] = {"error": str(e)}

        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(500, f"Report generation failed: {e}")


@router.get("/report/html")
async def generate_audit_report_html(
    project_path: Optional[str] = Query(None),
    report_type: str = Query("full"),
    include_insights: bool = Query(True),
    insight_focus: str = Query("executive"),
):
    """
    Generate audit report as standalone HTML page.

    Suitable for printing, PDF export, or single-page save.
    Includes embedded styles, SVG visualizations, and all content inline.
    """
    from fastapi.responses import HTMLResponse
    from datetime import datetime

    path = project_path or _current_project_path
    if not path:
        raise HTTPException(400, "No project path specified")

    # Get report data
    report_data = await generate_audit_report(
        project_path=path,
        report_type=report_type,
        include_insights=include_insights,
        insight_focus=insight_focus,
    )

    # Generate HTML
    html = _generate_audit_html(report_data)

    return HTMLResponse(content=html, media_type="text/html")


def _generate_audit_html(report: Dict[str, Any]) -> str:
    """
    Generate standalone HTML report with embedded styles and visualizations.

    Designed for:
    - Browser viewing
    - Print to PDF
    - Single-page save (Chrome extension compatible)
    """
    import html as html_lib
    import re

    project_name = report.get("project_name", "Unknown")
    generated_at = report.get("generated_at", "")

    # Risk data
    risk_data = report.get("risk", {})
    risk_summary_raw = risk_data.get("summary", {})
    risks = risk_data.get("risks", {})

    # Extract counts from nested by_level structure
    by_level = risk_summary_raw.get("by_level", {})
    risk_summary = {
        "total_components": risk_summary_raw.get("total_components", 0),
        "critical": by_level.get("critical", {}).get("count", 0),
        "high": by_level.get("high", {}).get("count", 0),
        "medium": by_level.get("medium", {}).get("count", 0),
        "low": by_level.get("low", {}).get("count", 0),
        "average_score": risk_summary_raw.get("avg_risk_score", 0),
        "critical_components": risk_summary_raw.get("critical_components", []),
    }

    # Timeline data
    timeline_data = report.get("timeline", {})
    timeline_summary = timeline_data.get("summary", {})
    components = timeline_data.get("components", {})

    # Drift data
    drift_data = report.get("drift", {})
    drift_alerts = report.get("drift_alerts", {}).get("alerts", [])

    # Detection data
    detection = report.get("detection", {})
    services = detection.get("services", {})

    # Insights
    insights = report.get("insights", {})
    insight_text = insights.get("insights", "")

    # Convert markdown-style formatting in insights to HTML
    def md_to_html(text):
        if not text:
            return ""
        # Headers
        text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Lists
        text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', text, flags=re.MULTILINE)
        # Wrap consecutive <li> in <ul>
        text = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', text)
        # Paragraphs
        text = re.sub(r'\n\n+', '</p><p>', text)
        text = f'<p>{text}</p>'
        text = text.replace('<p></p>', '')
        return text

    # Build risk matrix SVG
    def build_risk_matrix_svg():
        matrix = report.get("risk_matrix", {}).get("matrix", [])
        if not matrix:
            return ""

        svg_parts = [
            '<svg viewBox="0 0 420 320" class="risk-matrix-svg">',
            '  <defs>',
            '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
            '      <stop offset="0%" style="stop-color:#1a1a2e"/>',
            '      <stop offset="100%" style="stop-color:#16213e"/>',
            '    </linearGradient>',
            '  </defs>',
            '  <rect width="420" height="320" fill="url(#bgGrad)" rx="8"/>',
            '  <!-- Grid -->',
            '  <g stroke="#333" stroke-width="0.5">',
        ]

        # Grid lines
        for i in range(5):
            x = 60 + i * 80
            y = 20 + i * 60
            svg_parts.append(f'    <line x1="{x}" y1="20" x2="{x}" y2="260"/>')
            svg_parts.append(f'    <line x1="60" y1="{y}" x2="380" y2="{y}"/>')

        svg_parts.append('  </g>')

        # Zone backgrounds
        svg_parts.extend([
            '  <!-- Risk zones -->',
            '  <rect x="220" y="20" width="160" height="120" fill="rgba(239, 68, 68, 0.2)" rx="4"/>',
            '  <rect x="60" y="140" width="160" height="120" fill="rgba(34, 197, 94, 0.2)" rx="4"/>',
        ])

        # Axis labels
        svg_parts.extend([
            '  <!-- Axis labels -->',
            '  <text x="220" y="295" fill="#888" font-size="12" text-anchor="middle">Impact →</text>',
            '  <text x="25" y="140" fill="#888" font-size="12" text-anchor="middle" transform="rotate(-90, 25, 140)">Complexity →</text>',
        ])

        # Zone labels
        svg_parts.extend([
            '  <text x="300" y="85" fill="rgba(239, 68, 68, 0.6)" font-size="14" text-anchor="middle">HIGH RISK</text>',
            '  <text x="140" y="205" fill="rgba(34, 197, 94, 0.6)" font-size="14" text-anchor="middle">LOW RISK</text>',
        ])

        # Plot points
        color_map = {
            "critical": "#ef4444",
            "high": "#f97316",
            "medium": "#eab308",
            "low": "#22c55e",
        }

        for item in matrix[:50]:  # Limit to 50 points
            x = 60 + item.get("x", 0) * 320
            y = 260 - item.get("y", 0) * 240  # Invert Y
            size = 4 + item.get("size", 0.5) * 8
            color = color_map.get(item.get("color", "medium"), "#888")
            comp_id = html_lib.escape(item.get("id", "")[:10])

            svg_parts.append(
                f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" '
                f'fill="{color}" opacity="0.8">'
                f'<title>{comp_id} (score: {item.get("score", 0):.2f})</title></circle>'
            )

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    # Build lifecycle distribution chart
    def build_lifecycle_svg():
        if not components:
            return ""

        categories = {}
        for comp in components.values():
            cat = comp.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        total = sum(categories.values())
        if total == 0:
            return ""

        # Match actual LifecycleCategory values from timeline.py
        colors = {
            "new": "#22c55e",        # Green - age < 6 months
            "active": "#3b82f6",     # Blue - age >= 6mo, last change < 3mo
            "mature": "#f59e0b",     # Orange - age >= 1yr, last change >= 6mo
            "legacy_hot": "#ef4444", # Red - age >= 3yr, still changing (MCO risk!)
            "frozen": "#6b7280",     # Gray - age >= 2yr, no recent changes
            "unknown": "#888888",    # Fallback
        }

        svg_parts = [
            '<svg viewBox="0 0 400 200" class="lifecycle-svg">',
            '  <rect width="400" height="200" fill="#1a1a2e" rx="8"/>',
        ]

        x = 20
        bar_height = 30
        max_width = 360

        for i, (cat, count) in enumerate(sorted(categories.items(), key=lambda x: -x[1])):
            y = 20 + i * 35
            width = (count / total) * max_width
            color = colors.get(cat, "#888")
            pct = (count / total) * 100

            # Format category name nicely (legacy_hot → Legacy Hot)
            display_name = cat.replace("_", " ").title()
            svg_parts.extend([
                f'  <rect x="{x}" y="{y}" width="{width:.1f}" height="{bar_height}" fill="{color}" rx="4"/>',
                f'  <text x="{x + 5}" y="{y + 20}" fill="white" font-size="11">{display_name} ({count}, {pct:.0f}%)</text>',
            ])

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    # Build services table
    def build_services_table():
        if not services:
            return '<p class="muted">No services detected. Run detection first.</p>'

        rows = []
        for svc_id, svc in sorted(services.items(), key=lambda x: -x[1].get("confidence", 0))[:20]:
            conf = svc.get("confidence", 0) * 100
            svc_type = html_lib.escape(svc.get("service_type", "unknown"))
            sources = ", ".join(svc.get("sources", [])[:3])
            rows.append(f'''
                <tr>
                    <td>{html_lib.escape(svc_id)}</td>
                    <td><span class="badge badge-{svc_type.lower()}">{svc_type}</span></td>
                    <td>{conf:.0f}%</td>
                    <td class="muted">{html_lib.escape(sources)}</td>
                </tr>
            ''')

        return f'''
            <table class="data-table">
                <thead>
                    <tr><th>Service ID</th><th>Type</th><th>Confidence</th><th>Sources</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        '''

    # Build drift alerts table
    def build_drift_table():
        if not drift_alerts:
            return '<p class="success">No significant drift detected.</p>'

        rows = []
        for alert in drift_alerts[:15]:
            severity = alert.get("severity", "info")
            sev_class = f"severity-{severity}"
            rows.append(f'''
                <tr class="{sev_class}">
                    <td>{html_lib.escape(alert.get("component_id", ""))}</td>
                    <td><span class="badge">{alert.get("drift_type", "")}</span></td>
                    <td>{alert.get("drift_days", 0)} days</td>
                    <td>{html_lib.escape(alert.get("message", "")[:60])}</td>
                </tr>
            ''')

        return f'''
            <table class="data-table">
                <thead>
                    <tr><th>Component</th><th>Type</th><th>Drift</th><th>Message</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        '''

    # Build top risks table
    def build_risks_table():
        if not risks:
            return '<p class="muted">No risk data available.</p>'

        sorted_risks = sorted(risks.items(), key=lambda x: -x[1].get("score", 0))[:15]
        rows = []
        for comp_id, risk in sorted_risks:
            level = risk.get("level", "unknown")
            level_class = f"risk-{level}"
            factors = risk.get("factors", {})
            top_factor = max(factors.items(), key=lambda x: x[1])[0] if factors else "-"

            rows.append(f'''
                <tr>
                    <td>{html_lib.escape(comp_id)}</td>
                    <td><span class="badge {level_class}">{level.upper()}</span></td>
                    <td>{risk.get("score", 0):.2f}</td>
                    <td>{top_factor}</td>
                </tr>
            ''')

        return f'''
            <table class="data-table">
                <thead>
                    <tr><th>Component</th><th>Level</th><th>Score</th><th>Top Factor</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        '''

    # Full HTML template
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit Report - {html_lib.escape(project_name)}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #30363d;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .report-header {{
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}

        .report-header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, var(--accent), #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .report-meta {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .summary-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}

        .summary-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }}

        .summary-card .label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }}

        .summary-card.critical .value {{ color: var(--error); }}
        .summary-card.warning .value {{ color: var(--warning); }}
        .summary-card.success .value {{ color: var(--success); }}

        .section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .section h2 .icon {{ font-size: 1.5rem; }}

        .viz-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}

        @media (max-width: 800px) {{
            .viz-row {{ grid-template-columns: 1fr; }}
        }}

        .viz-container {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
        }}

        .viz-container h3 {{
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }}

        .risk-matrix-svg, .lifecycle-svg {{
            width: 100%;
            height: auto;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        .data-table th, .data-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        .data-table th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-secondary);
        }}

        .data-table tr:hover {{
            background: var(--bg-tertiary);
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }}

        .badge.risk-critical {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
        .badge.risk-high {{ background: rgba(249, 115, 22, 0.2); color: #f97316; }}
        .badge.risk-medium {{ background: rgba(234, 179, 8, 0.2); color: #eab308; }}
        .badge.risk-low {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; }}

        .severity-critical {{ background: rgba(239, 68, 68, 0.1); }}
        .severity-error {{ background: rgba(239, 68, 68, 0.08); }}
        .severity-warning {{ background: rgba(234, 179, 8, 0.08); }}

        .insights-content {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1.5rem;
        }}

        .insights-content h2, .insights-content h3, .insights-content h4 {{
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: var(--accent);
        }}

        .insights-content ul {{
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }}

        .insights-content li {{
            margin-bottom: 0.25rem;
        }}

        .insights-content p {{
            margin-bottom: 0.75rem;
        }}

        .muted {{ color: var(--text-secondary); }}
        .success {{ color: var(--success); }}

        .print-only {{ display: none; }}

        @media print {{
            body {{
                background: white;
                color: black;
                padding: 0;
            }}

            .section {{
                break-inside: avoid;
                border: 1px solid #ddd;
            }}

            .no-print {{ display: none; }}
            .print-only {{ display: block; }}
        }}

        /* Export button styles */
        .export-bar {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            display: flex;
            gap: 0.5rem;
            z-index: 1000;
        }}

        .export-btn {{
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            background: var(--accent);
            color: white;
            cursor: pointer;
            font-size: 0.85rem;
            transition: opacity 0.2s;
        }}

        .export-btn:hover {{ opacity: 0.8; }}
        .export-btn.secondary {{ background: var(--bg-tertiary); border: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="export-bar no-print">
        <button class="export-btn" onclick="window.print()">🖨️ Print / PDF</button>
        <button class="export-btn secondary" onclick="saveSinglePage()">💾 Save Page</button>
    </div>

    <div class="report-container">
        <header class="report-header">
            <h1>🔍 Code Audit Report</h1>
            <p class="report-meta">
                <strong>{html_lib.escape(project_name)}</strong><br>
                <span class="muted" style="font-size: 0.85rem;">{html_lib.escape(report.get('project_path', ''))}</span><br>
                Generated: {generated_at[:19].replace('T', ' ')}
            </p>
        </header>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{timeline_summary.get('total_components', 0)}</div>
                <div class="label">Components</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(services)}</div>
                <div class="label">Services Detected</div>
            </div>
            <div class="summary-card critical">
                <div class="value">{risk_summary.get('critical', 0)}</div>
                <div class="label">Critical Risks</div>
            </div>
            <div class="summary-card warning">
                <div class="value">{risk_summary.get('high', 0)}</div>
                <div class="label">High Risks</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(drift_alerts)}</div>
                <div class="label">Drift Alerts</div>
            </div>
            <div class="summary-card success">
                <div class="value">{risk_summary.get('low', 0)}</div>
                <div class="label">Low Risk</div>
            </div>
        </div>

        <!-- AI Insights Section -->
        <section class="section">
            <h2><span class="icon">🤖</span> AI Analysis ({insights.get('focus', 'executive').title()})</h2>
            <div class="insights-content">
                {md_to_html(insight_text) if insight_text else '<p class="muted">No AI insights available. Generate insights to see analysis.</p>'}
            </div>
        </section>

        <!-- Visualizations -->
        <section class="section">
            <h2><span class="icon">📊</span> Risk & Lifecycle Visualizations</h2>
            <div class="viz-row">
                <div class="viz-container">
                    <h3>Risk Matrix (Impact vs Complexity)</h3>
                    {build_risk_matrix_svg()}
                </div>
                <div class="viz-container">
                    <h3>Lifecycle Distribution</h3>
                    {build_lifecycle_svg()}
                </div>
            </div>
        </section>

        <!-- Top Risks -->
        <section class="section">
            <h2><span class="icon">⚠️</span> Top Risk Components</h2>
            {build_risks_table()}
        </section>

        <!-- Detected Services -->
        <section class="section">
            <h2><span class="icon">🔧</span> Detected Services</h2>
            {build_services_table()}
        </section>

        <!-- Drift Alerts -->
        <section class="section">
            <h2><span class="icon">📈</span> Documentation Drift Alerts</h2>
            {build_drift_table()}
        </section>

        <footer class="section" style="text-align: center; color: var(--text-secondary);">
            <p>Generated by RAGIX Audit Module v0.4</p>
            <p>Author: Olivier Vitrac, PhD, HDR | Adservio</p>
        </footer>
    </div>

    <script>
        // Single-page save functionality
        function saveSinglePage() {{
            // Clone the document
            const html = document.documentElement.cloneNode(true);

            // Remove export bar
            const exportBar = html.querySelector('.export-bar');
            if (exportBar) exportBar.remove();

            // Remove scripts
            const scripts = html.querySelectorAll('script');
            scripts.forEach(s => s.remove());

            // Get full HTML
            const fullHtml = '<!DOCTYPE html>\\n' + html.outerHTML;

            // Create blob and download
            const blob = new Blob([fullHtml], {{ type: 'text/html;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'audit-report-{project_name.replace(" ", "_")}.html';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''

    return html
