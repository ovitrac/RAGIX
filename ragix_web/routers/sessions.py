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
