"""
RAGIX Context Router - User context management API endpoints

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .sessions import get_sessions_store

router = APIRouter(prefix="/api/sessions", tags=["context"])

# User context storage (system instructions, custom prompts)
_user_context: Dict[str, Dict[str, Any]] = {}


def set_context_store(context: Dict[str, Dict[str, Any]]):
    """Set the context store reference from server.py"""
    global _user_context
    _user_context = context


def get_context_store() -> Dict[str, Dict[str, Any]]:
    """Get the context store"""
    return _user_context


class UserContextRequest(BaseModel):
    """Request body for user context update."""
    system_instructions: Optional[str] = None
    custom_prompt: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@router.get("/{session_id}/context")
async def get_user_context(session_id: str):
    """
    Get user context (custom instructions) for a session.

    User context is similar to Claude's system prompts or ChatGPT's custom instructions.
    """
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    context = _user_context.get(session_id, {})

    return {
        "session_id": session_id,
        "has_context": bool(context),
        "system_instructions": context.get("system_instructions", ""),
        "custom_prompt": context.get("custom_prompt", ""),
        "preferences": context.get("preferences", {}),
        "updated_at": context.get("updated_at", "")
    }


@router.post("/{session_id}/context")
async def update_user_context(session_id: str, request: UserContextRequest):
    """
    Update user context (custom instructions) for a session.

    This allows users to set:
    - system_instructions: Persistent instructions prepended to all prompts
    - custom_prompt: Additional context or preferences
    - preferences: Key-value settings (e.g., verbosity, output format)
    """
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get or create context
    if session_id not in _user_context:
        _user_context[session_id] = {}

    context = _user_context[session_id]

    # Update fields if provided
    if request.system_instructions is not None:
        context["system_instructions"] = request.system_instructions

    if request.custom_prompt is not None:
        context["custom_prompt"] = request.custom_prompt

    if request.preferences is not None:
        if "preferences" not in context:
            context["preferences"] = {}
        context["preferences"].update(request.preferences)

    context["updated_at"] = datetime.now().isoformat()

    return {
        "status": "updated",
        "session_id": session_id,
        "system_instructions_length": len(context.get("system_instructions", "")),
        "custom_prompt_length": len(context.get("custom_prompt", "")),
        "preferences_count": len(context.get("preferences", {})),
        "updated_at": context["updated_at"]
    }


@router.delete("/{session_id}/context")
async def delete_user_context(session_id: str):
    """Delete user context for a session."""
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_id in _user_context:
        del _user_context[session_id]
        return {
            "status": "deleted",
            "session_id": session_id
        }

    return {
        "status": "not_found",
        "session_id": session_id,
        "message": "No user context to delete"
    }
