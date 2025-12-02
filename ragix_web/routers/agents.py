"""
RAGIX Agents Router - Agent configuration and reasoning API endpoints

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Try to import ragix_core components
try:
    from ragix_core import (
        AgentConfig, AgentMode, AgentRole,
        detect_ollama_models, get_agent_persona, AGENT_PERSONAS
    )
    RAGIX_AVAILABLE = True
except ImportError:
    RAGIX_AVAILABLE = False
    AgentConfig = None
    AgentMode = None
    detect_ollama_models = lambda: []
    AGENT_PERSONAS = {}

# Storage for session-specific configs and traces
_session_agent_configs: Dict[str, Any] = {}
_session_reasoning_traces: Dict[str, List[Dict[str, Any]]] = {}
_active_sessions: Dict[str, Dict[str, Any]] = {}


def set_stores(sessions: Dict, configs: Dict, traces: Dict):
    """Set storage references from server.py"""
    global _active_sessions, _session_agent_configs, _session_reasoning_traces
    _active_sessions = sessions
    _session_agent_configs = configs
    _session_reasoning_traces = traces


class AgentConfigRequest(BaseModel):
    """Request body for agent configuration."""
    mode: str = "minimal"
    single_model_mode: bool = True
    single_model: Optional[str] = None
    planner_model: Optional[str] = None
    worker_model: Optional[str] = None
    verifier_model: Optional[str] = None


@router.get("/config")
async def get_agent_config(session_id: Optional[str] = None):
    """
    Get agent configuration.

    Model inheritance hierarchy:
    1. Session model = default (single source of truth)
    2. Agent Config = optional override (inherits from session if not explicitly set in MINIMAL mode)
    """
    if not RAGIX_AVAILABLE:
        return {"error": "ragix_core not available", "available": False}

    # Get session model from active_sessions (single source of truth)
    session_model = None
    if session_id and session_id in _active_sessions:
        session_model = _active_sessions[session_id].get("model")

    # Fallback to config default
    if session_model is None:
        try:
            from ragix_core.config import get_config
            config = get_config()
            session_model = config.llm.model if hasattr(config, 'llm') and config.llm else "mistral"
        except Exception:
            session_model = "mistral"

    # Detect available models
    available_models = detect_ollama_models()

    # Check for session-specific override
    if session_id and session_id in _session_agent_configs:
        agent_config = _session_agent_configs[session_id]
        is_override = True
    else:
        agent_config = AgentConfig()
        is_override = False

    # Resolve models based on mode - MINIMAL mode inherits from session
    mode = agent_config.mode.value if hasattr(agent_config.mode, 'value') else agent_config.mode
    if mode == "minimal":
        planner = session_model
        worker = session_model
        verifier = session_model
    else:
        planner = agent_config.planner_model
        worker = agent_config.worker_model
        verifier = agent_config.verifier_model

    return {
        "mode": mode,
        "planner_model": planner,
        "worker_model": worker,
        "verifier_model": verifier,
        "session_model": session_model,  # Expose for UI
        "single_model_mode": mode == "minimal",
        "available_models": [
            {"name": m.name, "size_gb": m.size_gb, "parameter_size": m.category, "params_b": m.params_b}
            for m in available_models
        ],
        "is_session_override": is_override,
    }


@router.post("/config")
async def update_agent_config(request: AgentConfigRequest, session_id: Optional[str] = None):
    """
    Update agent configuration for a session.

    This creates a session-specific override without modifying the default config.
    """
    if not RAGIX_AVAILABLE:
        raise HTTPException(status_code=503, detail="ragix_core not available")

    # Determine mode
    if request.single_model_mode:
        mode = AgentMode.MINIMAL
    else:
        mode = AgentMode(request.mode) if request.mode else AgentMode.MINIMAL

    # Create config
    if request.single_model_mode and request.single_model:
        agent_config = AgentConfig(
            mode=mode,
            planner_model=request.single_model,
            worker_model=request.single_model,
            verifier_model=request.single_model,
        )
    else:
        agent_config = AgentConfig(
            mode=mode,
            planner_model=request.planner_model or "granite3.1-moe:3b",
            worker_model=request.worker_model or "granite3.1-moe:3b",
            verifier_model=request.verifier_model or "granite3.1-moe:3b",
        )

    # Store as session override
    target_session = session_id or "default"
    _session_agent_configs[target_session] = agent_config

    return {
        "status": "updated",
        "session_id": target_session,
        "mode": agent_config.mode.value,
        "planner_model": agent_config.planner_model,
        "worker_model": agent_config.worker_model,
        "verifier_model": agent_config.verifier_model,
        "message": f"Agent config updated for session {target_session} (non-destructive override)"
    }


@router.delete("/config")
async def reset_agent_config(session_id: Optional[str] = None):
    """
    Reset session agent config to defaults.

    This removes the session-specific override, reverting to default config.
    """
    target_session = session_id or "default"

    if target_session in _session_agent_configs:
        del _session_agent_configs[target_session]
        return {
            "status": "reset",
            "session_id": target_session,
            "message": "Session config reset to defaults"
        }

    return {
        "session_id": target_session,
        "message": "No session override to reset"
    }


@router.get("/reasoning")
async def get_reasoning_traces(session_id: Optional[str] = None, limit: int = 50):
    """
    Get reasoning/chain-of-thought traces for a session.

    Returns planner/worker/verifier traces for visualization.
    """
    target_session = session_id or "default"

    traces = _session_reasoning_traces.get(target_session, [])

    # Also check episodic memory if session exists
    if target_session in _active_sessions:
        session = _active_sessions[target_session]
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            episodic_path = Path(sandbox_root) / ".agent_logs" / "episodic_log.jsonl"
            if episodic_path.exists():
                try:
                    episodic_entries = []
                    for line in episodic_path.read_text().strip().split('\n'):
                        if line:
                            episodic_entries.append(json.loads(line))
                    # Add episodic entries as traces
                    for entry in episodic_entries[-10:]:
                        traces.append({
                            "type": "episodic",
                            "timestamp": entry.get("timestamp", ""),
                            "content": entry.get("user_goal", ""),
                            "metadata": entry
                        })
                except Exception:
                    pass

    # Apply limit and return most recent
    traces = traces[-limit:] if len(traces) > limit else traces

    return {
        "session_id": target_session,
        "count": len(traces),
        "traces": traces
    }


@router.post("/reasoning")
async def add_reasoning_trace(
    trace_type: str,
    content: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Add a reasoning trace for a session."""
    target_session = session_id or "default"

    if target_session not in _session_reasoning_traces:
        _session_reasoning_traces[target_session] = []

    trace = {
        "type": trace_type,
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "metadata": metadata or {}
    }

    _session_reasoning_traces[target_session].append(trace)

    # Keep only last 100 traces per session
    if len(_session_reasoning_traces[target_session]) > 100:
        _session_reasoning_traces[target_session] = _session_reasoning_traces[target_session][-100:]

    return {"status": "ok", "trace_id": len(_session_reasoning_traces[target_session]) - 1}


@router.get("/personas")
async def get_agent_personas():
    """Get available agent personas."""
    if not RAGIX_AVAILABLE:
        return {"personas": {}, "available": False}

    personas = {}
    for role in [AgentRole.PLANNER, AgentRole.WORKER, AgentRole.VERIFIER]:
        personas[role.value] = get_agent_persona(role)[:200] + "..."

    return {"personas": personas, "available": True}
