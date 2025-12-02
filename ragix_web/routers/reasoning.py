"""
RAGIX Reasoning Router - Graph-based Reasoning API endpoints (v0.23)

This router provides endpoints for:
- Reasoning graph state visualization
- Experience corpus management
- Node-by-node execution control
- Reflection history and learning

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-02
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/reasoning", tags=["reasoning"])

# Try to import v0.23 reasoning components
try:
    from ragix_core import (
        # Types
        TaskComplexity,
        StepStatus,
        StopReason,
        ReasoningState,
        ReasoningEvent,
        ReflectionAttempt,
        get_max_reflections,
        # Experience corpus
        ExperienceCorpus,
        HybridExperienceCorpus,
        get_hybrid_corpus,
        # Graph components
        ReasoningGraph,
        create_reasoning_graph,
        ClassifyNode,
        PlanNode,
        ExecuteNode,
        ReflectNode,
        VerifyNode,
        RespondNode,
        # Strategy
        ReasoningStrategy,
        get_reasoning_strategy,
    )
    REASONING_GRAPH_AVAILABLE = True
except ImportError:
    REASONING_GRAPH_AVAILABLE = False
    TaskComplexity = None
    ReasoningState = None
    HybridExperienceCorpus = None

# Storage references (set by server.py)
_active_sessions: Dict[str, Dict[str, Any]] = {}
_session_reasoning_states: Dict[str, Dict[str, Any]] = {}
_session_experience_corpora: Dict[str, Any] = {}
_session_reasoning_config: Dict[str, Dict[str, Any]] = {}  # Per-session reasoning config


def set_stores(
    sessions: Dict,
    reasoning_states: Optional[Dict] = None,
    experience_corpora: Optional[Dict] = None
):
    """Set storage references from server.py."""
    global _active_sessions, _session_reasoning_states, _session_experience_corpora
    _active_sessions = sessions
    if reasoning_states is not None:
        _session_reasoning_states = reasoning_states
    if experience_corpora is not None:
        _session_experience_corpora = experience_corpora


# =============================================================================
# Request/Response Models
# =============================================================================

class ReasoningStateResponse(BaseModel):
    """Response model for reasoning state."""
    session_id: str
    goal: str
    complexity: str
    current_node: Optional[str] = None
    current_step_index: int = 0
    reflection_count: int = 0
    max_reflections: int = 2
    stop_reason: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    node_trace: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None


class PlanStepResponse(BaseModel):
    """Response model for a plan step."""
    num: int
    description: str
    status: str
    tool: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None


class ReflectionResponse(BaseModel):
    """Response model for a reflection attempt."""
    timestamp: str
    failed_step_num: int
    failed_step_description: str
    error: str
    diagnosis: str
    new_plan_summary: str
    context_used: Optional[str] = None


class ExperienceEventResponse(BaseModel):
    """Response model for an experience event."""
    timestamp: str
    session_id: str
    event_type: str
    goal: str
    outcome_status: Optional[str] = None
    step_description: Optional[str] = None
    error: Optional[str] = None
    llm_critique: Optional[str] = None


class ExperienceStatsResponse(BaseModel):
    """Response model for experience corpus statistics."""
    global_count: int = 0
    global_successes: int = 0
    global_failures: int = 0
    project_count: int = 0
    project_successes: int = 0
    project_failures: int = 0
    total_events: int = 0


class GraphNodeInfo(BaseModel):
    """Information about a reasoning graph node."""
    name: str
    description: str
    transitions_to: List[str]


class StartReasoningRequest(BaseModel):
    """Request to start a new reasoning session."""
    goal: str
    complexity: Optional[str] = None  # auto, simple, moderate, complex
    max_reflections: Optional[int] = None


class SearchExperienceRequest(BaseModel):
    """Request to search experience corpus."""
    query: str
    top_k: int = 5
    outcome_filter: Optional[str] = None  # success, failure


# =============================================================================
# Status & Info Endpoints
# =============================================================================

@router.get("/status")
async def get_reasoning_status(session_id: Optional[str] = None):
    """
    Get reasoning graph availability and configuration.

    Model inheritance hierarchy:
    - Reasoning model inherits from Agent Config worker model
    - Agent Config inherits from Session model
    - Session model is the single source of truth
    """
    # Get default strategy from environment
    default_strategy = get_reasoning_strategy() if REASONING_GRAPH_AVAILABLE else "unavailable"
    default_strategy_str = default_strategy.value if hasattr(default_strategy, 'value') else str(default_strategy)

    # Check for session-specific config
    strategy = default_strategy_str
    max_reflections = 2
    experience_corpus_enabled = True

    if session_id and session_id in _session_reasoning_config:
        config = _session_reasoning_config[session_id]
        strategy = config.get("strategy", default_strategy_str)
        max_reflections = config.get("max_reflections", 2)
        experience_corpus_enabled = config.get("experience_corpus_enabled", True)

    # Reasoning model is inherited from agent config / session (not stored separately)
    # Get it from session status if available
    reasoning_model = None
    session_model = None
    if session_id and session_id in _active_sessions:
        session_model = _active_sessions[session_id].get("model", "mistral")
        reasoning_model = session_model  # Inherits from session by default

    return {
        "available": REASONING_GRAPH_AVAILABLE,
        "strategy": strategy,
        "max_reflections": max_reflections,
        "reasoning_model": reasoning_model,  # Now inherited from session/agent
        "session_model": session_model,
        "experience_corpus_enabled": experience_corpus_enabled,
        "is_session_override": session_id is not None and session_id in _session_reasoning_config,
        "model_note": "Reasoning uses the worker model from Agent Config (inherits from Session)",
        "features": {
            "graph_based_reasoning": REASONING_GRAPH_AVAILABLE,
            "experience_corpus": REASONING_GRAPH_AVAILABLE,
            "reflective_learning": REASONING_GRAPH_AVAILABLE,
            "complexity_classification": REASONING_GRAPH_AVAILABLE,
        },
        "nodes": [
            "CLASSIFY", "DIRECT_EXEC", "PLAN", "EXECUTE",
            "REFLECT", "VERIFY", "RESPOND"
        ] if REASONING_GRAPH_AVAILABLE else [],
    }


class ReasoningConfigRequest(BaseModel):
    """Request body for updating reasoning configuration.

    Note: reasoning_model is no longer configurable here.
    It inherits from Agent Config worker model, which inherits from Session model.
    """
    strategy: str = "loop_v1"
    max_reflections: int = 2
    experience_corpus_enabled: bool = True


@router.post("/config")
async def update_reasoning_config(
    request: ReasoningConfigRequest,
    session_id: Optional[str] = None
):
    """
    Update reasoning configuration for a session.

    Note: Reasoning model is inherited from Agent Config / Session.
    To change the model, update the Session or Agent Config settings.
    """
    target_session = session_id or "default"

    # Store config (without reasoning_model - it's inherited)
    _session_reasoning_config[target_session] = {
        "strategy": request.strategy,
        "max_reflections": request.max_reflections,
        "experience_corpus_enabled": request.experience_corpus_enabled,
        "updated_at": datetime.now().isoformat()
    }

    return {
        "status": "ok",
        "session_id": target_session,
        "config": _session_reasoning_config[target_session],
        "message": f"Reasoning config updated for session {target_session}",
        "note": "Reasoning model is inherited from Agent Config / Session"
    }


@router.delete("/config")
async def reset_reasoning_config(session_id: Optional[str] = None):
    """
    Reset reasoning configuration to defaults for a session.
    """
    target_session = session_id or "default"

    if target_session in _session_reasoning_config:
        del _session_reasoning_config[target_session]
        return {
            "status": "ok",
            "session_id": target_session,
            "message": "Reasoning config reset to defaults"
        }
    else:
        return {
            "status": "ok",
            "session_id": target_session,
            "message": "No custom config to reset"
        }


@router.get("/nodes")
async def get_graph_nodes():
    """
    Get information about all reasoning graph nodes.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    nodes = [
        GraphNodeInfo(
            name="CLASSIFY",
            description="Classifies task complexity to determine reasoning strategy",
            transitions_to=["DIRECT_EXEC", "PLAN"]
        ),
        GraphNodeInfo(
            name="DIRECT_EXEC",
            description="Handles simple tasks without planning",
            transitions_to=["END"]
        ),
        GraphNodeInfo(
            name="PLAN",
            description="Generates a structured plan for moderate/complex tasks",
            transitions_to=["EXECUTE"]
        ),
        GraphNodeInfo(
            name="EXECUTE",
            description="Executes plan steps one at a time",
            transitions_to=["EXECUTE", "VERIFY", "REFLECT", "RESPOND"]
        ),
        GraphNodeInfo(
            name="REFLECT",
            description="Reflects on failures and generates improved plans",
            transitions_to=["EXECUTE"]
        ),
        GraphNodeInfo(
            name="VERIFY",
            description="Verifies execution results against plan criteria",
            transitions_to=["RESPOND", "REFLECT"]
        ),
        GraphNodeInfo(
            name="RESPOND",
            description="Generates the final response to the user",
            transitions_to=["END"]
        ),
    ]

    return {"nodes": [n.model_dump() for n in nodes]}


@router.get("/complexity-levels")
async def get_complexity_levels():
    """
    Get task complexity levels and their default configurations.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    return {
        "levels": [
            {
                "name": "simple",
                "value": TaskComplexity.SIMPLE.value,
                "description": "Single command, no planning needed (e.g., 'hello', 'list files')",
                "max_reflections": get_max_reflections(TaskComplexity.SIMPLE),
                "uses_planning": False,
            },
            {
                "name": "moderate",
                "value": TaskComplexity.MODERATE.value,
                "description": "2-3 steps, brief planning (e.g., 'find and read file')",
                "max_reflections": get_max_reflections(TaskComplexity.MODERATE),
                "uses_planning": True,
            },
            {
                "name": "complex",
                "value": TaskComplexity.COMPLEX.value,
                "description": "Multi-step with verification (e.g., 'refactor module')",
                "max_reflections": get_max_reflections(TaskComplexity.COMPLEX),
                "uses_planning": True,
            },
        ]
    }


# =============================================================================
# Session State Endpoints
# =============================================================================

@router.get("/state/{session_id}")
async def get_reasoning_state(session_id: str):
    """
    Get the current reasoning state for a session.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    if session_id not in _session_reasoning_states:
        return {
            "session_id": session_id,
            "active": False,
            "message": "No active reasoning session"
        }

    state_data = _session_reasoning_states[session_id]

    # If we have a ReasoningState object
    if isinstance(state_data, dict) and "state" in state_data:
        state = state_data["state"]
        if hasattr(state, 'to_dict'):
            state_dict = state.to_dict()
        else:
            state_dict = state_data
    else:
        state_dict = state_data

    return {
        "session_id": session_id,
        "active": True,
        "state": state_dict,
        "current_node": state_data.get("current_node", "unknown"),
    }


@router.get("/state/{session_id}/plan")
async def get_reasoning_plan(session_id: str):
    """
    Get the current plan for a reasoning session.
    """
    if session_id not in _session_reasoning_states:
        raise HTTPException(status_code=404, detail="No active reasoning session")

    state_data = _session_reasoning_states[session_id]
    state = state_data.get("state")

    if not state or not hasattr(state, 'plan') or not state.plan:
        return {
            "session_id": session_id,
            "has_plan": False,
            "plan": None
        }

    plan = state.plan
    return {
        "session_id": session_id,
        "has_plan": True,
        "plan": {
            "objective": plan.objective,
            "validation": plan.validation,
            "required_data": plan.required_data,
            "is_complete": plan.is_complete(),
            "has_failures": plan.has_failures(),
            "completed_count": plan.completed_count(),
            "total_steps": len(plan.steps),
            "steps": [
                {
                    "num": s.num,
                    "description": s.description,
                    "status": s.status.value,
                    "tool": s.tool,
                    "result": s.result[:200] if s.result else None,
                    "error": s.error,
                    "returncode": s.returncode,
                }
                for s in plan.steps
            ],
            "text_format": plan.to_text(),
        }
    }


@router.get("/state/{session_id}/reflections")
async def get_reasoning_reflections(session_id: str):
    """
    Get reflection attempts for a reasoning session.
    """
    if session_id not in _session_reasoning_states:
        raise HTTPException(status_code=404, detail="No active reasoning session")

    state_data = _session_reasoning_states[session_id]
    state = state_data.get("state")

    if not state:
        return {"session_id": session_id, "reflections": []}

    reflections = []
    if hasattr(state, 'reflection_attempts'):
        for attempt in state.reflection_attempts:
            reflections.append({
                "timestamp": attempt.timestamp,
                "failed_step_num": attempt.failed_step_num,
                "failed_step_description": attempt.failed_step_description,
                "error": attempt.error,
                "diagnosis": attempt.diagnosis,
                "new_plan_summary": attempt.new_plan_summary,
                "context_used": attempt.context_used[:500] if attempt.context_used else None,
            })

    return {
        "session_id": session_id,
        "reflection_count": state.reflection_count if hasattr(state, 'reflection_count') else 0,
        "max_reflections": state.max_reflections if hasattr(state, 'max_reflections') else 2,
        "can_reflect": state.can_reflect() if hasattr(state, 'can_reflect') else False,
        "reflections": reflections,
    }


@router.get("/state/{session_id}/trace")
async def get_reasoning_trace(session_id: str):
    """
    Get the node execution trace for a reasoning session.
    """
    if session_id not in _session_reasoning_states:
        raise HTTPException(status_code=404, detail="No active reasoning session")

    state_data = _session_reasoning_states[session_id]
    state = state_data.get("state")

    trace = []
    if state and hasattr(state, 'node_trace'):
        for entry in state.node_trace:
            parts = entry.split(":", 1)
            if len(parts) == 2:
                trace.append({"timestamp": parts[0], "node": parts[1]})
            else:
                trace.append({"timestamp": "", "node": entry})

    return {
        "session_id": session_id,
        "trace": trace,
        "trace_length": len(trace),
    }


# =============================================================================
# Experience Corpus Endpoints
# =============================================================================

@router.get("/experience/stats")
async def get_experience_stats(session_id: Optional[str] = None):
    """
    Get statistics for the experience corpus.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    # Get or create corpus for the session
    if session_id and session_id in _active_sessions:
        session = _active_sessions[session_id]
        project_path = Path(session.get("sandbox_root", "."))
    else:
        project_path = Path.cwd()

    corpus = get_hybrid_corpus(project_path=project_path)
    stats = corpus.get_stats()

    global_stats = stats.get("global", {})
    project_stats = stats.get("project", {})

    return ExperienceStatsResponse(
        global_count=global_stats.get("count", 0),
        global_successes=global_stats.get("successes", 0),
        global_failures=global_stats.get("failures", 0),
        project_count=project_stats.get("count", 0) if project_stats else 0,
        project_successes=project_stats.get("successes", 0) if project_stats else 0,
        project_failures=project_stats.get("failures", 0) if project_stats else 0,
        total_events=stats.get("total_events", 0),
    ).model_dump()


@router.post("/experience/search")
async def search_experience(
    request: SearchExperienceRequest,
    session_id: Optional[str] = None
):
    """
    Search the experience corpus for relevant past events.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    # Get or create corpus for the session
    if session_id and session_id in _active_sessions:
        session = _active_sessions[session_id]
        project_path = Path(session.get("sandbox_root", "."))
    else:
        project_path = Path.cwd()

    corpus = get_hybrid_corpus(project_path=project_path)

    # Search
    results = corpus.search(request.query, top_k=request.top_k)

    # Format results
    events = []
    for score, event in results:
        # Apply outcome filter if specified
        if request.outcome_filter and event.outcome_status != request.outcome_filter:
            continue

        events.append({
            "score": round(score, 3),
            "timestamp": event.timestamp,
            "session_id": event.session_id,
            "event_type": event.event_type,
            "goal": event.goal[:100] if event.goal else "",
            "outcome_status": event.outcome_status,
            "step_description": event.step_description[:80] if event.step_description else None,
            "error": event.error[:100] if event.error else None,
            "llm_critique": event.llm_critique[:150] if event.llm_critique else None,
        })

    return {
        "query": request.query,
        "count": len(events),
        "results": events[:request.top_k],
    }


@router.get("/experience/failures")
async def get_failure_lessons(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    session_id: Optional[str] = None
):
    """
    Search for past failures with lessons learned.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    # Get corpus
    if session_id and session_id in _active_sessions:
        session = _active_sessions[session_id]
        project_path = Path(session.get("sandbox_root", "."))
    else:
        project_path = Path.cwd()

    corpus = get_hybrid_corpus(project_path=project_path)

    # Search failures
    result_text = corpus.search_failures(query, top_k=top_k)

    return {
        "query": query,
        "lessons": result_text,
    }


@router.get("/experience/successes")
async def get_success_templates(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    session_id: Optional[str] = None
):
    """
    Search for past successes as templates.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    # Get corpus
    if session_id and session_id in _active_sessions:
        session = _active_sessions[session_id]
        project_path = Path(session.get("sandbox_root", "."))
    else:
        project_path = Path.cwd()

    corpus = get_hybrid_corpus(project_path=project_path)

    # Search successes
    result_text = corpus.search_successes(query, top_k=top_k)

    return {
        "query": query,
        "templates": result_text,
    }


@router.post("/experience/add")
async def add_experience_event(
    event_type: str = Query(..., description="Event type: planning, execution, reflection, verification"),
    goal: str = Query(..., description="The goal/task description"),
    outcome_status: Optional[str] = Query(None, description="success or failure"),
    step_description: Optional[str] = None,
    error: Optional[str] = None,
    llm_critique: Optional[str] = None,
    session_id: Optional[str] = None,
    to_global: bool = False
):
    """
    Add a new event to the experience corpus.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    # Get corpus
    if session_id and session_id in _active_sessions:
        session = _active_sessions[session_id]
        project_path = Path(session.get("sandbox_root", "."))
    else:
        project_path = Path.cwd()

    corpus = get_hybrid_corpus(project_path=project_path)

    # Create event
    event = ReasoningEvent(
        timestamp=datetime.utcnow().isoformat(),
        session_id=session_id or "api",
        event_type=event_type,
        goal=goal,
        outcome_status=outcome_status,
        step_description=step_description,
        error=error,
        llm_critique=llm_critique,
    )

    # Append to corpus
    corpus.append(event, to_global=to_global)

    return {
        "status": "added",
        "event_type": event_type,
        "to_global": to_global,
    }


# =============================================================================
# Classification Endpoint
# =============================================================================

@router.post("/classify")
async def classify_task(goal: str = Query(..., description="The task/goal to classify")):
    """
    Classify a task's complexity without executing it.

    Returns the complexity level and recommended strategy.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    # Use default classifier
    classify_node = ClassifyNode()

    # Create minimal state for classification
    state = ReasoningState(
        goal=goal,
        session_id="classify-only",
    )

    # Run classification
    new_state, next_node = classify_node.run(state)

    return {
        "goal": goal,
        "complexity": new_state.complexity.value,
        "next_node": next_node,
        "max_reflections": new_state.max_reflections,
        "uses_planning": next_node == "PLAN",
        "reasoning": {
            "simple_indicators": ["hello", "hi", "what is", "show me", "list", "pwd"],
            "complex_indicators": ["and then", "after that", "multiple", "refactor", "implement"],
        }
    }


# =============================================================================
# Visualization Data Endpoints
# =============================================================================

@router.get("/visualization/graph")
async def get_graph_visualization_data():
    """
    Get data for visualizing the reasoning graph structure.

    Returns nodes and edges in a format suitable for D3.js or similar.
    """
    if not REASONING_GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Reasoning graph not available")

    nodes = [
        {"id": "START", "label": "Start", "type": "terminal"},
        {"id": "CLASSIFY", "label": "Classify", "type": "decision"},
        {"id": "DIRECT_EXEC", "label": "Direct Exec", "type": "action"},
        {"id": "PLAN", "label": "Plan", "type": "action"},
        {"id": "EXECUTE", "label": "Execute", "type": "action"},
        {"id": "REFLECT", "label": "Reflect", "type": "recovery"},
        {"id": "VERIFY", "label": "Verify", "type": "validation"},
        {"id": "RESPOND", "label": "Respond", "type": "action"},
        {"id": "END", "label": "End", "type": "terminal"},
    ]

    edges = [
        {"source": "START", "target": "CLASSIFY", "label": ""},
        {"source": "CLASSIFY", "target": "DIRECT_EXEC", "label": "simple"},
        {"source": "CLASSIFY", "target": "PLAN", "label": "moderate/complex"},
        {"source": "DIRECT_EXEC", "target": "END", "label": ""},
        {"source": "PLAN", "target": "EXECUTE", "label": ""},
        {"source": "EXECUTE", "target": "EXECUTE", "label": "next step"},
        {"source": "EXECUTE", "target": "VERIFY", "label": "complete"},
        {"source": "EXECUTE", "target": "REFLECT", "label": "failed + can_reflect"},
        {"source": "EXECUTE", "target": "RESPOND", "label": "failed + max_reflections"},
        {"source": "REFLECT", "target": "EXECUTE", "label": "new plan"},
        {"source": "VERIFY", "target": "RESPOND", "label": "pass"},
        {"source": "VERIFY", "target": "REFLECT", "label": "fail + can_reflect"},
        {"source": "RESPOND", "target": "END", "label": ""},
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "node_types": {
            "terminal": {"color": "#888", "shape": "circle"},
            "decision": {"color": "#f0ad4e", "shape": "diamond"},
            "action": {"color": "#5bc0de", "shape": "rect"},
            "recovery": {"color": "#d9534f", "shape": "hexagon"},
            "validation": {"color": "#5cb85c", "shape": "rect"},
        }
    }


@router.get("/visualization/session/{session_id}")
async def get_session_visualization_data(session_id: str):
    """
    Get visualization data for a specific session's reasoning path.
    """
    if session_id not in _session_reasoning_states:
        raise HTTPException(status_code=404, detail="No active reasoning session")

    state_data = _session_reasoning_states[session_id]
    state = state_data.get("state")

    if not state:
        raise HTTPException(status_code=404, detail="No state data available")

    # Get graph structure
    graph_data = await get_graph_visualization_data()

    # Build visited path
    visited_nodes = []
    if hasattr(state, 'node_trace'):
        for entry in state.node_trace:
            parts = entry.split(":", 1)
            node = parts[1] if len(parts) == 2 else entry
            visited_nodes.append(node)

    # Mark active/visited nodes
    for node in graph_data["nodes"]:
        node["visited"] = node["id"] in visited_nodes
        node["visit_count"] = visited_nodes.count(node["id"])
        node["is_current"] = node["id"] == state_data.get("current_node")

    # Build visited edges
    visited_edges = []
    for i in range(len(visited_nodes) - 1):
        visited_edges.append({
            "source": visited_nodes[i],
            "target": visited_nodes[i + 1],
            "order": i + 1,
        })

    return {
        "session_id": session_id,
        "graph": graph_data,
        "visited_path": visited_nodes,
        "visited_edges": visited_edges,
        "current_node": state_data.get("current_node"),
        "complexity": state.complexity.value if hasattr(state, 'complexity') else "unknown",
        "reflection_count": state.reflection_count if hasattr(state, 'reflection_count') else 0,
        "stop_reason": state.stop_reason.value if hasattr(state, 'stop_reason') and state.stop_reason else None,
    }
