"""
RAGIX Sessions Router - Session management API endpoints

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# Reference to global state (set by server.py)
_active_sessions: Dict[str, Dict[str, Any]] = {}


def set_sessions_store(sessions: Dict[str, Dict[str, Any]]):
    """Set the sessions store reference from server.py"""
    global _active_sessions
    _active_sessions = sessions


def get_sessions_store() -> Dict[str, Dict[str, Any]]:
    """Get the sessions store"""
    return _active_sessions


class SessionCreateRequest(BaseModel):
    """Request body for creating a session."""
    session_id: Optional[str] = None
    sandbox_root: Optional[str] = None
    model: str = "granite3.1-moe:3b"
    profile: str = "dev"


@router.get("")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, session_data in _active_sessions.items():
        sessions.append({
            "id": session_id,
            "sandbox_root": session_data.get("sandbox_root", ""),
            "model": session_data.get("model", ""),
            "profile": session_data.get("profile", ""),
            "created_at": session_data.get("created_at", ""),
            "message_count": len(session_data.get("message_history", [])),
        })
    return {"sessions": sessions}


@router.post("")
async def create_session(request: SessionCreateRequest):
    """Create a new session."""
    session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get sandbox root - default to current directory for security
    sandbox_root = request.sandbox_root or ""
    if sandbox_root:
        sandbox_path = Path(sandbox_root).expanduser()
        if not sandbox_path.exists():
            raise HTTPException(status_code=400, detail=f"Sandbox path does not exist: {sandbox_root}")
        sandbox_root = str(sandbox_path.resolve())

    model = request.model
    profile = request.profile

    # Store session
    _active_sessions[session_id] = {
        "id": session_id,
        "sandbox_root": sandbox_root,
        "model": model,
        "profile": profile,
        "created_at": datetime.now().isoformat(),
        "message_history": []
    }

    return {
        "session_id": session_id,
        "sandbox_root": sandbox_root,
        "model": model,
        "profile": profile,
        "status": "created"
    }


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _active_sessions[session_id]
    return {
        "id": session_id,
        "sandbox_root": session.get("sandbox_root", ""),
        "model": session.get("model", ""),
        "profile": session.get("profile", ""),
        "created_at": session.get("created_at", ""),
        "message_count": len(session.get("message_history", [])),
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in _active_sessions:
        del _active_sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/{session_id}/logs")
async def get_session_logs(session_id: str, limit: int = 50):
    """Get command logs for a session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _active_sessions[session_id]
    sandbox_root = session["sandbox_root"]
    log_file = Path(sandbox_root) / ".agent_logs" / "commands.log" if sandbox_root else Path(".agent_logs/commands.log")

    if not log_file.exists():
        return {"logs": [], "count": 0}

    lines = log_file.read_text().strip().split('\n')
    recent_lines = lines[-limit:] if len(lines) > limit else lines
    return {"logs": recent_lines, "count": len(lines)}


@router.get("/{session_id}/events")
async def get_session_events(session_id: str, limit: int = 50):
    """Get JSONL events for a session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _active_sessions[session_id]
    sandbox_root = session["sandbox_root"]
    events_file = Path(sandbox_root) / ".agent_logs" / "events.jsonl" if sandbox_root else Path(".agent_logs/events.jsonl")

    if not events_file.exists():
        return {"events": []}

    events = []
    lines = events_file.read_text().strip().split('\n')
    recent_lines = lines[-limit:] if len(lines) > limit else lines

    for line in recent_lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {"events": events}


# v0.33: Agent config storage for context limits
_agent_configs: Dict[str, Dict[str, Any]] = {}


class AgentConfigRequest(BaseModel):
    """Request body for agent configuration update."""
    context_max_turns: Optional[int] = None
    context_user_limit: Optional[int] = None
    context_assistant_limit: Optional[int] = None


def get_agent_config_store() -> Dict[str, Dict[str, Any]]:
    """Get the agent config store."""
    return _agent_configs


@router.get("/{session_id}/agent-config")
async def get_agent_config(session_id: str):
    """Get agent configuration for a session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    config = _agent_configs.get(session_id, {})
    return {
        "session_id": session_id,
        "context_max_turns": config.get("context_max_turns", 5),
        "context_user_limit": config.get("context_user_limit", 500),
        "context_assistant_limit": config.get("context_assistant_limit", 2000),
    }


@router.post("/{session_id}/agent-config")
async def update_agent_config(session_id: str, request: AgentConfigRequest):
    """Update agent configuration for a session (context limits)."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_id not in _agent_configs:
        _agent_configs[session_id] = {}

    config = _agent_configs[session_id]

    # Update fields if provided
    if request.context_max_turns is not None:
        config["context_max_turns"] = max(1, min(20, request.context_max_turns))

    if request.context_user_limit is not None:
        config["context_user_limit"] = max(100, min(5000, request.context_user_limit))

    if request.context_assistant_limit is not None:
        config["context_assistant_limit"] = max(100, min(10000, request.context_assistant_limit))

    config["updated_at"] = datetime.now().isoformat()

    return {
        "status": "updated",
        "session_id": session_id,
        **config
    }
