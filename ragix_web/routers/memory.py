"""
RAGIX Memory Router - Session memory management API endpoints

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from .sessions import get_sessions_store

router = APIRouter(prefix="/api/sessions", tags=["memory"])


@router.get("/{session_id}/memory")
async def get_session_memory(session_id: str, limit: int = 100):
    """
    Get session memory (message history).

    Returns all messages exchanged in this session for review/management.
    """
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    message_history = session.get("message_history", [])

    # Build response with truncated content
    messages = [
        {
            "index": i,
            "role": msg.get("role", "unknown"),
            "content": msg.get("content", "")[:500] + ("..." if len(msg.get("content", "")) > 500 else ""),
            "full_content_length": len(msg.get("content", "")),
            "timestamp": msg.get("timestamp", "")
        }
        for i, msg in enumerate(message_history)
    ]

    # Apply limit
    if limit > 0 and len(messages) > limit:
        messages = messages[-limit:]

    return {
        "session_id": session_id,
        "total_messages": len(message_history),
        "returned_messages": len(messages),
        "messages": messages
    }


@router.get("/{session_id}/memory/{message_idx}")
async def get_session_message(session_id: str, message_idx: int):
    """Get full content of a specific message by index."""
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    message_history = session.get("message_history", [])

    if message_idx < 0 or message_idx >= len(message_history):
        raise HTTPException(status_code=404, detail=f"Message index {message_idx} not found")

    msg = message_history[message_idx]
    return {
        "session_id": session_id,
        "index": message_idx,
        "role": msg.get("role", "unknown"),
        "content": msg.get("content", ""),
        "timestamp": msg.get("timestamp", "")
    }


@router.delete("/{session_id}/memory/{message_idx}")
async def delete_session_message(session_id: str, message_idx: int):
    """Delete a specific message from session memory."""
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    message_history = session.get("message_history", [])

    if message_idx < 0 or message_idx >= len(message_history):
        raise HTTPException(status_code=404, detail=f"Message index {message_idx} not found")

    # Remove the message
    deleted_msg = message_history.pop(message_idx)

    return {
        "status": "deleted",
        "session_id": session_id,
        "deleted_index": message_idx,
        "deleted_role": deleted_msg.get("role", "unknown"),
        "remaining_messages": len(message_history)
    }


@router.delete("/{session_id}/memory")
async def clear_session_memory(session_id: str, keep_last: int = 0):
    """
    Clear session memory (message history).

    Args:
        session_id: Session to clear
        keep_last: Number of recent messages to keep (0 = clear all)
    """
    sessions = get_sessions_store()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    previous_count = len(session.get("message_history", []))

    if keep_last > 0:
        session["message_history"] = session.get("message_history", [])[-keep_last:]
    else:
        session["message_history"] = []

    return {
        "status": "cleared",
        "session_id": session_id,
        "previous_count": previous_count,
        "current_count": len(session["message_history"]),
        "kept_last": keep_last
    }
