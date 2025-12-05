"""
RAGIX Threads Router - Thread management API endpoints

Provides REST API for creating, listing, switching, and managing conversation threads.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-05
"""

from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ragix_core.threads import get_thread_manager, Thread, ThreadManager

router = APIRouter(prefix="/api/sessions/{session_id}/threads", tags=["threads"])

# Reference to active sessions (set by server.py)
_active_sessions: Dict[str, Dict[str, Any]] = {}
_storage_root: str = ""


def set_threads_store(sessions: Dict[str, Dict[str, Any]], storage_root: str = ""):
    """Set the sessions store and storage root from server.py."""
    global _active_sessions, _storage_root
    _active_sessions = sessions
    _storage_root = storage_root


def _get_manager(session_id: str) -> ThreadManager:
    """Get thread manager for a session, ensuring session exists."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return get_thread_manager(session_id, _storage_root or None)


# Request/Response Models

class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""
    name: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ThreadRenameRequest(BaseModel):
    """Request body for renaming a thread."""
    name: str


class ThreadMessageRequest(BaseModel):
    """Request body for adding a message to a thread."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ThreadSummary(BaseModel):
    """Summary of a thread for list views."""
    id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int
    model: Optional[str]
    is_active: bool


# Endpoints

@router.get("")
async def list_threads(
    session_id: str,
    sort_by: str = Query("updated_at", regex="^(updated_at|created_at|name)$"),
    descending: bool = True,
) -> Dict[str, Any]:
    """
    List all threads for a session.

    Args:
        session_id: Session identifier
        sort_by: Field to sort by (updated_at, created_at, name)
        descending: Sort in descending order
    """
    manager = _get_manager(session_id)
    threads = manager.list_threads(sort_by=sort_by, descending=descending)
    active_id = manager._active_thread_id

    return {
        "threads": [
            ThreadSummary(
                id=t.id,
                name=t.name,
                created_at=t.created_at,
                updated_at=t.updated_at,
                message_count=len(t.messages),
                model=t.model,
                is_active=(t.id == active_id),
            ).model_dump()
            for t in threads
        ],
        "active_thread_id": active_id,
        "total": len(threads),
    }


@router.post("")
async def create_thread(session_id: str, request: ThreadCreateRequest) -> Dict[str, Any]:
    """
    Create a new thread.

    Args:
        session_id: Session identifier
        request: Thread creation parameters
    """
    manager = _get_manager(session_id)
    thread = manager.create_thread(
        name=request.name,
        model=request.model,
        metadata=request.metadata,
    )

    return {
        "id": thread.id,
        "name": thread.name,
        "created_at": thread.created_at,
        "model": thread.model,
        "status": "created",
    }


@router.get("/summary")
async def get_threads_summary(session_id: str) -> Dict[str, Any]:
    """Get a summary of all threads for this session."""
    manager = _get_manager(session_id)
    return manager.get_summary()


@router.get("/active")
async def get_active_thread(session_id: str) -> Dict[str, Any]:
    """Get the currently active thread."""
    manager = _get_manager(session_id)
    thread = manager.get_active_thread()

    if not thread:
        return {"active_thread": None}

    return {
        "active_thread": {
            "id": thread.id,
            "name": thread.name,
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
            "message_count": len(thread.messages),
            "model": thread.model,
        }
    }


@router.put("/active/{thread_id}")
async def set_active_thread(session_id: str, thread_id: str) -> Dict[str, Any]:
    """Set the active thread."""
    manager = _get_manager(session_id)
    thread = manager.set_active_thread(thread_id)

    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return {
        "active_thread_id": thread_id,
        "name": thread.name,
        "status": "activated",
    }


@router.get("/{thread_id}")
async def get_thread(session_id: str, thread_id: str) -> Dict[str, Any]:
    """Get a thread by ID."""
    manager = _get_manager(session_id)
    thread = manager.get_thread(thread_id)

    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return {
        "id": thread.id,
        "name": thread.name,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "message_count": len(thread.messages),
        "model": thread.model,
        "metadata": thread.metadata,
        "is_active": (thread.id == manager._active_thread_id),
    }


@router.delete("/{thread_id}")
async def delete_thread(session_id: str, thread_id: str) -> Dict[str, Any]:
    """Delete a thread."""
    manager = _get_manager(session_id)

    if not manager.delete_thread(thread_id):
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return {
        "status": "deleted",
        "thread_id": thread_id,
        "remaining_threads": len(manager.list_threads()),
    }


@router.patch("/{thread_id}/rename")
async def rename_thread(
    session_id: str,
    thread_id: str,
    request: ThreadRenameRequest,
) -> Dict[str, Any]:
    """Rename a thread."""
    manager = _get_manager(session_id)
    thread = manager.rename_thread(thread_id, request.name)

    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return {
        "id": thread.id,
        "name": thread.name,
        "status": "renamed",
    }


# Messages within threads

@router.get("/{thread_id}/messages")
async def get_thread_messages(
    session_id: str,
    thread_id: str,
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """Get messages from a thread."""
    manager = _get_manager(session_id)
    thread = manager.get_thread(thread_id)

    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    messages = thread.messages[-limit:] if limit else thread.messages

    return {
        "thread_id": thread_id,
        "thread_name": thread.name,
        "total_messages": len(thread.messages),
        "returned_messages": len(messages),
        "messages": [
            {
                "index": i,
                "role": m.role,
                "content": m.content[:500] + ("..." if len(m.content) > 500 else ""),
                "full_content_length": len(m.content),
                "timestamp": m.timestamp,
            }
            for i, m in enumerate(messages)
        ],
    }


@router.post("/{thread_id}/messages")
async def add_thread_message(
    session_id: str,
    thread_id: str,
    request: ThreadMessageRequest,
) -> Dict[str, Any]:
    """Add a message to a thread."""
    manager = _get_manager(session_id)

    msg = manager.add_message(
        role=request.role,
        content=request.content,
        thread_id=thread_id,
        metadata=request.metadata,
    )

    if not msg:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return {
        "thread_id": thread_id,
        "role": msg.role,
        "timestamp": msg.timestamp,
        "status": "added",
    }


@router.delete("/{thread_id}/messages")
async def clear_thread_messages(
    session_id: str,
    thread_id: str,
    keep_last: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Clear messages from a thread."""
    manager = _get_manager(session_id)

    if thread_id not in [t.id for t in manager.list_threads()]:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    count = manager.clear_thread(thread_id, keep_last)

    return {
        "thread_id": thread_id,
        "messages_cleared": count,
        "kept_last": keep_last,
        "status": "cleared",
    }


# Export endpoints

@router.get("/{thread_id}/export")
async def export_thread(
    session_id: str,
    thread_id: str,
    format: str = Query("markdown", regex="^(markdown|json)$"),
) -> Any:
    """
    Export a thread.

    Args:
        session_id: Session identifier
        thread_id: Thread identifier
        format: Export format (markdown or json)
    """
    manager = _get_manager(session_id)
    content = manager.export_thread(thread_id, format)

    if content is None:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    if format == "json":
        return {"content": content, "format": "json"}
    else:
        return PlainTextResponse(content, media_type="text/markdown")
