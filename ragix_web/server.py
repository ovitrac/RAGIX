"""
FastAPI Server for RAGIX Web UI

Provides local-only web interface with WebSocket chat and REST API.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import os
import sys
import asyncio
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Error: FastAPI dependencies not installed.", file=sys.stderr)
    print("Install with: pip install 'ragix[web]'", file=sys.stderr)
    sys.exit(1)

from ragix_core import OllamaLLM, ShellSandbox, AgentLogger, LogLevel
from ragix_core.version import __version__ as RAGIX_VERSION
from ragix_core.log_integrity import ChainedLogHasher, AuditLogManager, LogIntegrityReport
from ragix_core.agent_config import (
    AgentConfig, AgentMode, AgentRole,
    detect_ollama_models, get_agent_persona, MODEL_REGISTRY
)
from ragix_core.config import get_config, HardwareConfig

# AST imports (optional, may not be installed)
try:
    from ragix_core import (
        build_dependency_graph,
        calculate_metrics_from_graph,
        graph_to_html,
        VizConfig,
        ColorScheme,
        D3Renderer,
        DependencyType,
        NodeType,
        RadialExplorer,
        DSMRenderer,
    )
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False

from ragix_unix import UnixRAGAgent

# Import workflow templates
try:
    from ragix_core.workflow_templates import (
        BUILTIN_TEMPLATES,
        list_builtin_templates,
        TemplateManager,
        get_template_manager,
    )
    WORKFLOWS_AVAILABLE = True
except ImportError:
    BUILTIN_TEMPLATES = {}
    WORKFLOWS_AVAILABLE = False

# Import agents
try:
    from ragix_core.agents import (
        BaseAgent,
        CodeAgent,
        DocAgent,
        GitAgent,
        TestAgent,
        AgentCapability,
    )
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# Import modular routers
try:
    from ragix_web.routers import (
        sessions_router,
        memory_router,
        context_router,
        agents_router,
        logs_router,
    )
    from ragix_web.routers.sessions import set_sessions_store
    from ragix_web.routers.context import set_context_store
    from ragix_web.routers.agents import set_stores as set_agent_stores
    from ragix_web.routers.logs import set_sessions_store as set_logs_sessions
    ROUTERS_AVAILABLE = True
except ImportError:
    ROUTERS_AVAILABLE = False


# FastAPI app
app = FastAPI(
    title="RAGIX Web UI",
    description="Local-first Unix-RAG development assistant with AST analysis",
    version=RAGIX_VERSION
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local only, safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security: Capture launch directory at startup (like Claude Code)
# The sandbox is restricted to the directory where the server was launched
LAUNCH_DIRECTORY = os.getcwd()

# Global state (in production, use proper session management)
active_sessions: Dict[str, Dict[str, Any]] = {}
active_websockets: List[WebSocket] = []


# Request models
class SessionCreateRequest(BaseModel):
    """Request body for creating a session."""
    sandbox_root: str = ""  # Empty means use LAUNCH_DIRECTORY
    model: str = "mistral"
    profile: str = "dev"


# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =============================================================================
# Router Registration (modular endpoints)
# =============================================================================

# Storage for agent configs and reasoning traces (used by routers)
session_agent_configs: Dict[str, AgentConfig] = {}
session_reasoning_traces: Dict[str, List[Dict[str, Any]]] = {}
session_user_context: Dict[str, Dict[str, Any]] = {}

# Register modular routers if available
if ROUTERS_AVAILABLE:
    # Set shared state references
    set_sessions_store(active_sessions)
    set_context_store(session_user_context)
    set_agent_stores(active_sessions, session_agent_configs, session_reasoning_traces)
    set_logs_sessions(active_sessions)

    # Include routers with their prefixes
    # Note: These provide modular, organized endpoints
    # The original inline endpoints are kept for backward compatibility
    # but will be deprecated in a future version
    app.include_router(sessions_router, tags=["Sessions (v2)"])
    app.include_router(memory_router, tags=["Memory (v2)"])
    app.include_router(context_router, tags=["Context (v2)"])
    app.include_router(agents_router, tags=["Agents (v2)"])
    app.include_router(logs_router, tags=["Logs (v2)"])


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main HTML page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    else:
        return """
        <html>
            <head><title>RAGIX Web UI</title></head>
            <body>
                <h1>RAGIX Web UI</h1>
                <p>Frontend not found. Please ensure static files are in ragix_web/static/</p>
            </body>
        </html>
        """


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.10.1",
        "sessions": len(active_sessions),
        "ast_available": AST_AVAILABLE
    }


@app.get("/api/browse")
async def browse_directory(path: str = "", show_files: bool = False):
    """
    Browse filesystem directories for folder selection.

    Security: Only allows browsing within LAUNCH_DIRECTORY and common project roots.
    """
    # Default to launch directory if no path provided
    if not path or path == "~":
        path = os.path.expanduser("~")
    elif path.startswith("~"):
        path = os.path.expanduser(path)

    # Resolve to absolute path
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(LAUNCH_DIRECTORY)

    # Check if path exists
    if not resolved.exists():
        # Try to find closest existing parent
        while not resolved.exists() and resolved != resolved.parent:
            resolved = resolved.parent

    if not resolved.is_dir():
        resolved = resolved.parent

    # List contents
    entries = []
    try:
        for entry in sorted(resolved.iterdir()):
            # Skip hidden files by default
            if entry.name.startswith('.') and entry.name not in ['.git']:
                continue

            entry_info = {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
            }

            if entry.is_dir():
                # Check if it looks like a project (has common markers)
                markers = ['.git', 'pom.xml', 'package.json', 'setup.py', 'pyproject.toml', 'Cargo.toml', 'build.gradle']
                entry_info["is_project"] = any((entry / m).exists() for m in markers)
                entries.append(entry_info)
            elif show_files:
                entries.append(entry_info)

    except PermissionError:
        pass

    return {
        "current": str(resolved),
        "parent": str(resolved.parent) if resolved.parent != resolved else None,
        "entries": entries,
        "is_project": any((resolved / m).exists() for m in ['.git', 'pom.xml', 'package.json', 'setup.py', 'pyproject.toml']),
        "launch_directory": LAUNCH_DIRECTORY
    }


@app.get("/api/ollama/models")
async def get_ollama_models():
    """Get list of available Ollama models installed locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {
                "available": False,
                "error": result.stderr or "Failed to list Ollama models",
                "models": []
            }

        models = []
        lines = result.stdout.strip().split("\n")

        # Skip header line
        for line in lines[1:]:
            if not line.strip():
                continue
            # Parse: NAME  ID  SIZE  MODIFIED
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                model_id = parts[1]
                size = parts[2] + " " + parts[3] if len(parts) > 3 else parts[2]
                modified = " ".join(parts[4:]) if len(parts) > 4 else ""

                models.append({
                    "name": name,
                    "id": model_id,
                    "size": size,
                    "modified": modified
                })

        return {
            "available": True,
            "models": models,
            "count": len(models)
        }

    except FileNotFoundError:
        return {
            "available": False,
            "error": "Ollama not installed or not in PATH",
            "models": []
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "error": "Timeout waiting for Ollama",
            "models": []
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "models": []
        }


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, session_data in active_sessions.items():
        sessions.append({
            "id": session_id,
            "sandbox_root": session_data.get("sandbox_root", ""),
            "model": session_data.get("model", ""),
            "profile": session_data.get("profile", ""),
            "created_at": session_data.get("created_at", "")
        })
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session(request: SessionCreateRequest):
    """Create a new session."""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Extract values from request body
    sandbox_root = request.sandbox_root
    model = request.model
    profile = request.profile

    # Security: Use LAUNCH_DIRECTORY as default (like Claude Code)
    if not sandbox_root or sandbox_root.strip() == "":
        sandbox_root = LAUNCH_DIRECTORY
    else:
        # Expand and validate sandbox root
        sandbox_root = os.path.expanduser(sandbox_root)
        sandbox_root = os.path.abspath(sandbox_root)

        # Security: Ensure sandbox is within or is the launch directory
        if not sandbox_root.startswith(LAUNCH_DIRECTORY):
            raise HTTPException(
                status_code=403,
                detail=f"Security: Sandbox must be within launch directory: {LAUNCH_DIRECTORY}"
            )

    # Create sandbox if it doesn't exist
    os.makedirs(sandbox_root, exist_ok=True)

    # Store session
    active_sessions[session_id] = {
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
        "profile": profile
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions/{session_id}/logs")
async def get_session_logs(session_id: str, limit: int = 50):
    """Get command logs for a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    sandbox_root = session["sandbox_root"]
    log_file = Path(sandbox_root) / ".agent_logs" / "commands.log"

    if not log_file.exists():
        return {"logs": []}

    # Read last N lines
    lines = log_file.read_text().strip().split('\n')
    recent_lines = lines[-limit:] if len(lines) > limit else lines

    return {"logs": recent_lines}


@app.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: str, limit: int = 50):
    """Get JSONL events for a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    sandbox_root = session["sandbox_root"]
    events_file = Path(sandbox_root) / ".agent_logs" / "events.jsonl"

    if not events_file.exists():
        return {"events": []}

    # Parse JSONL
    events = []
    lines = events_file.read_text().strip().split('\n')
    recent_lines = lines[-limit:] if len(lines) > limit else lines

    for line in recent_lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {"events": events}


# =============================================================================
# Session Memory Management API Endpoints
# =============================================================================
# Legacy Memory Management Endpoints (kept for backward compatibility)
# New code should use the /api/sessions/* routers above
# =============================================================================


@app.get("/api/sessions/{session_id}/memory")
async def get_session_memory(session_id: str, limit: int = 100):
    """
    Get session memory (message history).

    Returns all messages exchanged in this session for review/management.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    message_history = session.get("message_history", [])

    # Apply limit
    if limit > 0 and len(message_history) > limit:
        message_history = message_history[-limit:]

    return {
        "session_id": session_id,
        "total_messages": len(session.get("message_history", [])),
        "returned_messages": len(message_history),
        "messages": [
            {
                "index": i,
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", "")[:500] + ("..." if len(msg.get("content", "")) > 500 else ""),
                "full_content_length": len(msg.get("content", "")),
                "timestamp": msg.get("timestamp", "")
            }
            for i, msg in enumerate(session.get("message_history", []))
        ][-limit:] if limit > 0 else [
            {
                "index": i,
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", "")[:500] + ("..." if len(msg.get("content", "")) > 500 else ""),
                "full_content_length": len(msg.get("content", "")),
                "timestamp": msg.get("timestamp", "")
            }
            for i, msg in enumerate(session.get("message_history", []))
        ]
    }


@app.get("/api/sessions/{session_id}/memory/{message_idx}")
async def get_session_message(session_id: str, message_idx: int):
    """Get full content of a specific message by index."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
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


@app.delete("/api/sessions/{session_id}/memory/{message_idx}")
async def delete_session_message(session_id: str, message_idx: int):
    """Delete a specific message from session memory."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
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


@app.delete("/api/sessions/{session_id}/memory")
async def clear_session_memory(session_id: str, keep_last: int = 0):
    """
    Clear session memory (message history).

    Args:
        session_id: Session to clear
        keep_last: Number of recent messages to keep (0 = clear all)
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
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


# =============================================================================
# User Context Management API Endpoints
# =============================================================================

class UserContextRequest(BaseModel):
    """Request body for user context update."""
    system_instructions: Optional[str] = None
    custom_prompt: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@app.get("/api/sessions/{session_id}/context")
async def get_user_context(session_id: str):
    """
    Get user context (custom instructions) for a session.

    User context is similar to Claude's system prompts or ChatGPT's custom instructions.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    context = session_user_context.get(session_id, {})

    return {
        "session_id": session_id,
        "has_context": bool(context),
        "system_instructions": context.get("system_instructions", ""),
        "custom_prompt": context.get("custom_prompt", ""),
        "preferences": context.get("preferences", {}),
        "updated_at": context.get("updated_at", "")
    }


@app.post("/api/sessions/{session_id}/context")
async def update_user_context(session_id: str, request: UserContextRequest):
    """
    Update user context (custom instructions) for a session.

    This allows users to set:
    - system_instructions: Persistent instructions prepended to all prompts
    - custom_prompt: Additional context or preferences
    - preferences: Key-value settings (e.g., verbosity, output format)
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get or create context
    if session_id not in session_user_context:
        session_user_context[session_id] = {}

    context = session_user_context[session_id]

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


@app.delete("/api/sessions/{session_id}/context")
async def delete_user_context(session_id: str):
    """Delete user context for a session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_id in session_user_context:
        del session_user_context[session_id]
        return {
            "status": "deleted",
            "session_id": session_id
        }

    return {
        "status": "not_found",
        "session_id": session_id,
        "message": "No user context to delete"
    }


# =============================================================================
# AST Analysis API Endpoints
# =============================================================================

@app.get("/api/ast/status")
async def ast_status():
    """Check AST analysis availability."""
    return {
        "available": AST_AVAILABLE,
        "message": "AST analysis ready" if AST_AVAILABLE else "Install with: pip install 'ragix[ast]'"
    }


@app.get("/api/ast/graph")
async def get_dependency_graph(
    path: str,
    lang: str = "auto",
    filter_types: Optional[str] = None,
    filter_deps: Optional[str] = None,
    max_nodes: int = 1000
):
    """
    Get dependency graph for a path.

    Args:
        path: Directory or file path to analyze
        lang: Language filter (auto, python, java)
        filter_types: Comma-separated node types (class,interface,method)
        filter_deps: Comma-separated dependency types (inheritance,call,import)
        max_nodes: Maximum nodes to return (default 1000)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Build dependency graph - pass as list, language is auto-detected from extensions
        # Use patterns to filter by language if specified
        patterns = None
        if lang and lang != "auto":
            lang_patterns = {
                "python": ["*.py"],
                "java": ["*.java"],
                "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"],
            }
            patterns = lang_patterns.get(lang)
        graph = build_dependency_graph([target_path], patterns=patterns)

        # Apply filters
        config = VizConfig()
        if filter_types:
            type_list = [NodeType(t.strip()) for t in filter_types.split(',') if t.strip()]
            config.filter_types = type_list
        if filter_deps:
            dep_list = [DependencyType(d.strip()) for d in filter_deps.split(',') if d.strip()]
            config.filter_deps = dep_list

        # Convert to D3.js format
        renderer = D3Renderer(config)
        data = renderer.to_dict(graph)

        # Add total counts before truncation
        data["total_nodes"] = len(data["nodes"])
        data["total_edges"] = len(data["links"])
        data["total_files"] = len(graph._files)

        # Limit nodes if too many
        if len(data["nodes"]) > max_nodes:
            data["nodes"] = data["nodes"][:max_nodes]
            # Filter links to only include nodes within truncated index range
            # source and target are indices into the nodes array
            data["links"] = [l for l in data["links"]
                            if l["source"] < max_nodes and l["target"] < max_nodes]
            data["truncated"] = True

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/metrics")
async def get_project_metrics(
    path: str,
    lang: str = "auto"
):
    """
    Get code metrics for a path.

    Args:
        path: Directory or file path to analyze
        lang: Language filter (auto, python, java)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Build graph and calculate metrics
        patterns = None
        if lang and lang != "auto":
            lang_patterns = {"python": ["*.py"], "java": ["*.java"], "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"]}
            patterns = lang_patterns.get(lang)
        graph = build_dependency_graph([target_path], patterns=patterns)
        metrics = calculate_metrics_from_graph(graph)

        # Convert to JSON-serializable format
        # Calculate blank lines from file metrics
        total_blank_lines = sum(f.blank_lines for f in metrics.file_metrics)

        result = {
            "path": str(target_path),
            "summary": {
                "total_files": metrics.total_files,
                "total_lines": metrics.total_lines,
                "code_lines": metrics.total_code_lines,
                "comment_lines": metrics.total_comment_lines,
                "blank_lines": total_blank_lines,
                "total_classes": metrics.total_classes,
                "total_functions": metrics.total_functions,
            },
            "complexity": {
                "average_cyclomatic": round(metrics.avg_complexity_per_method, 2),
                "total_complexity": metrics.total_complexity,
            },
            "maintainability": {
                "index": metrics.maintainability_index,
            },
            "debt": {
                "estimated_hours": round(metrics.estimated_debt_hours, 1),
                "estimated_days": round(metrics.estimated_debt_days, 1),
            },
            "hotspots": [
                {"name": name, "complexity": cc}
                for name, cc in metrics.get_hotspots(20)
            ],
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/search")
async def search_symbols(
    path: str,
    q: str,
    type: Optional[str] = None,
    limit: int = 50
):
    """
    Search for symbols in codebase.

    Args:
        path: Directory to search in
        q: Search query (supports wildcards: *Service, User*)
        type: Optional type filter (class, interface, method, function)
        limit: Maximum results (default 50)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        import fnmatch
        graph = build_dependency_graph([target_path])
        symbols = graph.get_symbols()

        # Filter by query
        pattern = q.lower() if '*' not in q else q.lower()
        results = []

        for sym in symbols:
            name_match = False
            if '*' in pattern:
                name_match = fnmatch.fnmatch(sym.name.lower(), pattern)
            else:
                name_match = pattern in sym.name.lower() or pattern in sym.qualified_name.lower()

            if not name_match:
                continue

            if type and sym.node_type.value != type:
                continue

            results.append({
                "name": sym.name,
                "qualified_name": sym.qualified_name,
                "type": sym.node_type.value,
                "visibility": sym.visibility.value,
                "file": str(sym.location.file) if sym.location.file else None,
                "line": sym.location.line,
            })

            if len(results) >= limit:
                break

        return JSONResponse(content={"results": results, "total": len(results)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/hotspots")
async def get_hotspots(
    path: str,
    limit: int = 20,
    min_complexity: int = 10
):
    """
    Get complexity hotspots.

    Args:
        path: Directory to analyze
        limit: Maximum results (default 20)
        min_complexity: Minimum cyclomatic complexity (default 10)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        graph = build_dependency_graph([target_path])
        metrics = calculate_metrics_from_graph(graph)

        hotspots = [
            {"name": name, "complexity": cc}
            for name, cc in metrics.get_hotspots(limit * 2)
            if cc >= min_complexity
        ][:limit]

        return JSONResponse(content={
            "hotspots": hotspots,
            "threshold": min_complexity,
            "total_above_threshold": len([h for h in hotspots if h["complexity"] >= min_complexity])
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/visualize", response_class=HTMLResponse)
async def visualize_dependencies(
    path: str,
    title: str = "Dependency Graph",
    color_scheme: str = "default"
):
    """
    Generate interactive HTML visualization.

    Args:
        path: Directory to analyze
        title: Graph title
        color_scheme: Color scheme (default, pastel, dark, monochrome)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        graph = build_dependency_graph([target_path])

        # Configure visualization
        scheme = ColorScheme(color_scheme) if color_scheme in [s.value for s in ColorScheme] else ColorScheme.DEFAULT
        config = VizConfig(color_scheme=scheme)

        html_content = graph_to_html(graph, title=title, config=config)
        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/matrix", response_class=HTMLResponse)
async def visualize_dsm_matrix(
    path: str,
    title: str = "Dependency Structure Matrix",
    level: str = "class"
):
    """
    Generate interactive DSM (Dependency Structure Matrix) HTML visualization.

    Args:
        path: Directory to analyze
        title: Matrix title
        level: Aggregation level ('class', 'package', 'file')
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        graph = build_dependency_graph([target_path])

        # Create DSM renderer and generate HTML
        renderer = DSMRenderer()
        html_content = renderer.render_html(graph, title=title, level=level)
        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/radial")
async def get_radial_graph(
    path: str,
    focal: Optional[str] = None,
    levels: int = 3
):
    """
    Get ego-centric radial graph data for a focal node.

    Args:
        path: Directory to analyze
        focal: Focal node (class name or qualified name). If not provided, auto-selects.
        levels: Maximum depth levels (default 3)

    Returns:
        JSON with nodes organized by level and links
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        graph = build_dependency_graph([target_path])
        symbols = {s.qualified_name: s for s in graph.get_symbols()}

        # Determine focal node
        if not focal:
            # Auto-select highest connectivity class
            deps = graph.get_all_dependencies()
            degree = {}
            for dep in deps:
                degree[dep.source] = degree.get(dep.source, 0) + 1
                degree[dep.target] = degree.get(dep.target, 0) + 1

            excluded_prefixes = (
                'java.lang.', 'java.util.', 'java.io.', 'java.sql.', 'java.time.',
                'org.springframework.', 'javax.', 'lombok.', 'org.joda.',
                'org.slf4j.', 'com.fasterxml.', 'org.apache.', 'com.google.',
            )
            class_degree = {
                k: v for k, v in degree.items()
                if k in symbols and '.' in k and not k.startswith(excluded_prefixes)
            }
            if class_degree:
                focal = max(class_degree.keys(), key=lambda k: class_degree[k])
            elif degree:
                focal = max(degree.keys(), key=lambda k: degree[k])
            else:
                raise HTTPException(status_code=404, detail="No dependencies found")

        # Resolve focal node
        if focal not in symbols:
            matches = [s for s in symbols.keys() if focal in s]
            if len(matches) == 1:
                focal = matches[0]
            elif len(matches) > 1:
                # Prefer shorter qualified names (class over method)
                focal = min(matches, key=len)
            else:
                raise HTTPException(status_code=404, detail=f"Focal node not found: {focal}")

        # Build radial graph
        config = VizConfig()
        explorer = RadialExplorer(config)
        ego_data = explorer.build_ego_graph(graph, focal, max_levels=levels)

        return {
            "focal": focal,
            "path": str(target_path),
            "levels": levels,
            **ego_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/radial/page", response_class=HTMLResponse)
async def radial_explorer_page(
    path: str,
    focal: Optional[str] = None,
    levels: int = 3,
    title: str = "Radial Dependency Explorer"
):
    """
    Serve the live radial explorer page.

    Args:
        path: Directory to analyze
        focal: Initial focal node (optional, auto-selects if not provided)
        levels: Maximum depth levels (default 3)
        title: Page title
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    # Generate HTML page that fetches data dynamically
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            overflow: hidden;
        }}
        #container {{ display: flex; height: 100vh; }}
        #graph-container {{ flex: 1; position: relative; }}
        #graph {{ width: 100%; height: 100%; }}
        #sidebar {{
            width: 350px;
            background: #161b22;
            border-left: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 20px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
        }}
        .sidebar-header h2 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #58a6ff;
        }}
        #search {{
            width: 100%;
            padding: 10px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #0d1117;
            color: #e6edf3;
            font-size: 14px;
        }}
        #search:focus {{ outline: none; border-color: #58a6ff; }}
        .controls {{
            padding: 15px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 6px 12px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #21262d;
            color: #e6edf3;
            cursor: pointer;
            font-size: 12px;
        }}
        .btn:hover {{ background: #30363d; }}
        .btn-primary {{ background: #238636; border-color: #238636; }}
        .btn-primary:hover {{ background: #2ea043; }}
        #node-info {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        .info-section {{
            margin-bottom: 20px;
        }}
        .info-section h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #8b949e;
            text-transform: uppercase;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
        }}
        .info-label {{ color: #8b949e; }}
        .info-value {{ color: #e6edf3; font-weight: 500; }}
        #focal-info {{
            padding: 15px 20px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}
        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        #breadcrumb {{
            padding: 10px 20px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
            font-size: 12px;
            white-space: nowrap;
            overflow-x: auto;
        }}
        .breadcrumb-item {{
            color: #58a6ff;
            cursor: pointer;
        }}
        .breadcrumb-item:hover {{ text-decoration: underline; }}
        .breadcrumb-sep {{ color: #484f58; margin: 0 8px; }}
        #levels-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        #levels-control label {{ font-size: 12px; color: #8b949e; }}
        #levels-input {{
            width: 50px;
            padding: 4px 8px;
            border: 1px solid #30363d;
            border-radius: 4px;
            background: #0d1117;
            color: #e6edf3;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <svg id="graph"></svg>
            <div id="loading">
                <div class="spinner"></div>
                <p>Loading dependency graph...</p>
            </div>
        </div>
        <div id="sidebar">
            <div class="sidebar-header">
                <h2>Radial Explorer</h2>
                <input type="text" id="search" placeholder="Search classes...">
            </div>
            <div id="breadcrumb"></div>
            <div class="controls">
                <button class="btn" onclick="resetView()">Reset View</button>
                <button class="btn" onclick="exportSVG()">Export SVG</button>
                <div id="levels-control">
                    <label>Levels:</label>
                    <input type="number" id="levels-input" value="{levels}" min="1" max="5">
                    <button class="btn btn-primary" onclick="reloadWithLevels()">Apply</button>
                </div>
            </div>
            <div id="focal-info"></div>
            <div id="node-info">
                <p style="color: #8b949e;">Click a node to see details</p>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        const PATH = "{path}";
        let currentFocal = {f'"{focal}"' if focal else 'null'};
        let currentLevels = {levels};
        let egoData = null;
        let selectedNode = null;
        let focalHistory = [];

        const svg = d3.select('#graph');
        const g = svg.append('g');
        const linksLayer = g.append('g').attr('class', 'links');
        const nodesLayer = g.append('g').attr('class', 'nodes');
        const labelsLayer = g.append('g').attr('class', 'labels');

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (e) => g.attr('transform', e.transform));
        svg.call(zoom);

        // Load initial data
        loadData(currentFocal);

        async function loadData(focal) {{
            document.getElementById('loading').style.display = 'block';

            const url = new URL(`${{API_BASE}}/api/ast/radial`);
            url.searchParams.set('path', PATH);
            url.searchParams.set('levels', currentLevels);
            if (focal) url.searchParams.set('focal', focal);

            try {{
                const resp = await fetch(url);
                if (!resp.ok) throw new Error(await resp.text());
                egoData = await resp.json();
                currentFocal = egoData.focal;

                if (focal && focal !== focalHistory[focalHistory.length - 1]) {{
                    focalHistory.push(currentFocal);
                }}

                render();
                updateFocalInfo();
                updateBreadcrumb();
            }} catch (e) {{
                alert('Error loading data: ' + e.message);
            }} finally {{
                document.getElementById('loading').style.display = 'none';
            }}
        }}

        function render() {{
            const width = svg.node().clientWidth;
            const height = svg.node().clientHeight;
            const centerX = width / 2;
            const centerY = height / 2;
            const maxLevel = Math.max(...egoData.nodes.map(n => n.level));
            const levelRadius = Math.min(width, height) / (2 * (maxLevel + 1.5));

            // Position nodes
            const nodesByLevel = {{}};
            egoData.nodes.forEach(n => {{
                nodesByLevel[n.level] = nodesByLevel[n.level] || [];
                nodesByLevel[n.level].push(n);
            }});

            egoData.nodes.forEach(n => {{
                if (n.level === 0) {{
                    n.x = centerX;
                    n.y = centerY;
                }} else {{
                    const nodesAtLevel = nodesByLevel[n.level];
                    const idx = nodesAtLevel.indexOf(n);
                    const angle = (2 * Math.PI * idx) / nodesAtLevel.length - Math.PI / 2;
                    const radius = levelRadius * n.level;
                    n.x = centerX + radius * Math.cos(angle);
                    n.y = centerY + radius * Math.sin(angle);
                }}
            }});

            // Draw level circles
            g.selectAll('.level-circle').remove();
            for (let l = 1; l <= maxLevel; l++) {{
                g.insert('circle', ':first-child')
                    .attr('class', 'level-circle')
                    .attr('cx', centerX)
                    .attr('cy', centerY)
                    .attr('r', levelRadius * l)
                    .attr('fill', 'none')
                    .attr('stroke', '#21262d')
                    .attr('stroke-dasharray', '4,4');
            }}

            // Create node map for links
            const nodeMap = {{}};
            egoData.nodes.forEach(n => nodeMap[n.id] = n);

            // Draw links
            const links = linksLayer.selectAll('.link')
                .data(egoData.links, d => `${{d.source}}|${{d.target}}`);

            links.exit().remove();

            links.enter()
                .append('path')
                .attr('class', 'link')
                .merge(links)
                .attr('d', d => {{
                    const s = nodeMap[d.source];
                    const t = nodeMap[d.target];
                    if (!s || !t) return '';
                    const dx = t.x - s.x;
                    const dy = t.y - s.y;
                    const dr = Math.sqrt(dx * dx + dy * dy) * 0.8;
                    return `M${{s.x}},${{s.y}}A${{dr}},${{dr}} 0 0,1 ${{t.x}},${{t.y}}`;
                }})
                .attr('fill', 'none')
                .attr('stroke', d => getTypeColor(d.type))
                .attr('stroke-width', d => Math.min(d.count, 5))
                .attr('stroke-opacity', 0.6);

            // Draw nodes
            const nodes = nodesLayer.selectAll('.node')
                .data(egoData.nodes, d => d.id);

            nodes.exit().remove();

            nodes.enter()
                .append('circle')
                .attr('class', 'node')
                .merge(nodes)
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', d => d.level === 0 ? 20 : Math.max(8, Math.min(15, 5 + d.total_deps / 5)))
                .attr('fill', d => d.color)
                .attr('stroke', '#0d1117')
                .attr('stroke-width', 2)
                .style('cursor', 'pointer')
                .on('click', (e, d) => selectNode(d))
                .on('dblclick', (e, d) => {{ if (d.level > 0) loadData(d.id); }});

            // Draw labels
            const labels = labelsLayer.selectAll('.node-label')
                .data(egoData.nodes.filter(n => n.level <= 1), d => d.id);

            labels.exit().remove();

            labels.enter()
                .append('text')
                .attr('class', 'node-label')
                .merge(labels)
                .attr('x', d => d.x)
                .attr('y', d => d.y + (d.level === 0 ? 35 : 25))
                .attr('text-anchor', 'middle')
                .attr('fill', '#e6edf3')
                .attr('font-size', d => d.level === 0 ? '14px' : '11px')
                .attr('font-weight', d => d.level === 0 ? 'bold' : 'normal')
                .text(d => d.name);

            // Center view
            svg.call(zoom.transform, d3.zoomIdentity);
        }}

        function getTypeColor(type) {{
            const colors = {{
                'inheritance': '#f97583',
                'implementation': '#b392f0',
                'type_reference': '#79c0ff',
                'call': '#56d364',
                'composition': '#ffa657',
                'instantiation': '#ff7b72'
            }};
            return colors[type] || '#8b949e';
        }}

        function selectNode(node) {{
            selectedNode = node;
            nodesLayer.selectAll('.node')
                .attr('stroke', d => d.id === node.id ? '#58a6ff' : '#0d1117')
                .attr('stroke-width', d => d.id === node.id ? 3 : 2);
            updateNodeInfo(node);
        }}

        function updateNodeInfo(node) {{
            if (!node) {{
                document.getElementById('node-info').innerHTML = '<p style="color: #8b949e;">Click a node to see details</p>';
                return;
            }}
            document.getElementById('node-info').innerHTML = `
                <div class="info-section">
                    <h3>Selected Node</h3>
                    <div class="info-row">
                        <span class="info-label">Name</span>
                        <span class="info-value">${{node.name}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value">${{node.type}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Level</span>
                        <span class="info-value">${{node.level}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Dependencies Out</span>
                        <span class="info-value">${{node.out_deps}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Dependencies In</span>
                        <span class="info-value">${{node.in_deps}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">File</span>
                        <span class="info-value" style="font-size: 11px; word-break: break-all;">${{node.file || 'N/A'}}</span>
                    </div>
                </div>
                ${{node.level > 0 ? `
                <button class="btn btn-primary" onclick="loadData('${{node.id}}')" style="width: 100%;">
                    Explore as Center
                </button>
                ` : ''}}
            `;
        }}

        function updateFocalInfo() {{
            const focal = egoData.nodes.find(n => n.level === 0);
            if (!focal) return;
            const levelCounts = {{}};
            egoData.nodes.forEach(n => levelCounts[n.level] = (levelCounts[n.level] || 0) + 1);

            document.getElementById('focal-info').innerHTML = `
                <div style="font-weight: bold; color: #58a6ff; margin-bottom: 5px;">${{focal.name}}</div>
                <div style="font-size: 12px; color: #8b949e;">
                    ${{Object.entries(levelCounts).map(([l, c]) => `L${{l}}: ${{c}}`).join(' | ')}}
                </div>
            `;
        }}

        function updateBreadcrumb() {{
            const container = document.getElementById('breadcrumb');
            if (focalHistory.length <= 1) {{
                container.innerHTML = '';
                return;
            }}
            container.innerHTML = focalHistory.map((f, i) => {{
                const name = f.split('.').pop();
                const isLast = i === focalHistory.length - 1;
                return isLast
                    ? `<span style="color: #e6edf3;">${{name}}</span>`
                    : `<span class="breadcrumb-item" onclick="navigateTo(${{i}})">${{name}}</span><span class="breadcrumb-sep">›</span>`;
            }}).join('');
        }}

        function navigateTo(index) {{
            const target = focalHistory[index];
            focalHistory = focalHistory.slice(0, index);
            loadData(target);
        }}

        function resetView() {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }}

        function reloadWithLevels() {{
            currentLevels = parseInt(document.getElementById('levels-input').value) || 3;
            loadData(currentFocal);
        }}

        function exportSVG() {{
            const svgData = new XMLSerializer().serializeToString(svg.node());
            const blob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'radial-deps.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Search functionality
        document.getElementById('search').addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase();
            if (!query) {{
                nodesLayer.selectAll('.node').attr('opacity', 1);
                return;
            }}
            nodesLayer.selectAll('.node')
                .attr('opacity', d => d.name.toLowerCase().includes(query) ? 1 : 0.2);
        }});

        // Handle window resize
        window.addEventListener('resize', () => {{ if (egoData) render(); }});
    </script>
</body>
</html>'''
    return HTMLResponse(content=html)


# =============================================================================
# Agents & Workflows API
# =============================================================================

@app.get("/api/agents")
async def list_agents():
    """List available agent types and their capabilities."""
    if not AGENTS_AVAILABLE:
        return {
            "available": False,
            "agents": [],
            "message": "Agent module not available"
        }

    agents = [
        {
            "id": "code",
            "name": "Code Agent",
            "description": "Analyzes, searches, and modifies code",
            "capabilities": ["code_read", "code_write", "code_search", "code_analysis"],
            "icon": "&#128187;"  # computer
        },
        {
            "id": "doc",
            "name": "Doc Agent",
            "description": "Generates and updates documentation",
            "capabilities": ["doc_write", "doc_read"],
            "icon": "&#128196;"  # page
        },
        {
            "id": "git",
            "name": "Git Agent",
            "description": "Manages git operations and versioning",
            "capabilities": ["git_read", "git_write"],
            "icon": "&#128200;"  # chart
        },
        {
            "id": "test",
            "name": "Test Agent",
            "description": "Runs tests and analyzes coverage",
            "capabilities": ["test_run", "code_read"],
            "icon": "&#9989;"  # check
        },
    ]

    return {
        "available": True,
        "agents": agents
    }


@app.get("/api/workflows")
async def list_workflows():
    """List available workflow templates."""
    if not WORKFLOWS_AVAILABLE:
        return {
            "available": False,
            "workflows": [],
            "message": "Workflow templates not available"
        }

    workflows = []
    for name, template in BUILTIN_TEMPLATES.items():
        workflows.append({
            "id": name,
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.param_type,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in template.parameters
            ],
            "steps": [
                {
                    "name": s.name,
                    "agent_type": s.agent_type,
                    "depends_on": s.depends_on,
                }
                for s in template.steps
            ],
        })

    return {
        "available": True,
        "workflows": workflows
    }


# =============================================================================
# Legacy Agent Configuration API (kept for backward compatibility)
# New code should use /api/agents/* routers
# =============================================================================


class AgentConfigRequest(BaseModel):
    """Request body for updating agent configuration."""
    mode: str = "minimal"  # minimal, strict, custom
    planner_model: Optional[str] = None
    worker_model: Optional[str] = None
    verifier_model: Optional[str] = None
    single_model_mode: bool = False  # Use same model for all agents (low VRAM)
    single_model: Optional[str] = None  # Model to use in single-model mode


@app.get("/api/agents/config")
async def get_agent_config(session_id: Optional[str] = None):
    """
    Get current agent configuration.

    Returns session-specific config if available, otherwise returns default.
    """
    # Get default config from ragix.yaml
    try:
        config = get_config()
        default_config = config.agents if hasattr(config, 'agents') and config.agents else AgentConfig()
    except Exception:
        default_config = AgentConfig()

    # Check for session-specific override
    if session_id and session_id in session_agent_configs:
        agent_config = session_agent_configs[session_id]
    else:
        agent_config = default_config

    # Detect available models
    available_models = detect_ollama_models()

    return {
        "mode": agent_config.mode.value,
        "planner_model": agent_config.planner_model,
        "worker_model": agent_config.worker_model,
        "verifier_model": agent_config.verifier_model,
        "fallback_model": agent_config.fallback_model,
        "strict_enforcement": agent_config.strict_enforcement,
        "is_session_override": session_id in session_agent_configs if session_id else False,
        "available_models": [
            {
                "name": m.name,
                "size_gb": m.size_gb,
                "parameter_size": m.category,  # Use category property
                "params_b": m.params_b,
            }
            for m in available_models
        ],
        "model_registry": {
            name: {
                "params_b": info[0],
                "category": info[1],
                "description": info[2],
            }
            for name, info in MODEL_REGISTRY.items()
        },
        "recommended": {
            "minimal": {
                "description": "All agents use 3B model (≤8GB VRAM / CPU)",
                "planner": "granite3.1-moe:3b",
                "worker": "granite3.1-moe:3b",
                "verifier": "granite3.1-moe:3b",
            },
            "strict": {
                "description": "Planner uses 7B+, Worker/Verifier use 3B",
                "planner": "mistral:latest",
                "worker": "granite3.1-moe:3b",
                "verifier": "granite3.1-moe:3b",
            }
        }
    }


@app.post("/api/agents/config")
async def update_agent_config(request: AgentConfigRequest, session_id: Optional[str] = None):
    """
    Update agent configuration for a session.

    This creates a session-specific override without modifying the default config.
    """
    # Validate mode
    try:
        mode = AgentMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    # Handle single-model mode (for low VRAM systems)
    if request.single_model_mode and request.single_model:
        planner = request.single_model
        worker = request.single_model
        verifier = request.single_model
    else:
        planner = request.planner_model
        worker = request.worker_model
        verifier = request.verifier_model

    # Create agent config
    agent_config = AgentConfig(
        mode=mode,
        planner_model=planner or "granite3.1-moe:3b",
        worker_model=worker or "granite3.1-moe:3b",
        verifier_model=verifier or "granite3.1-moe:3b",
        strict_enforcement=mode == AgentMode.STRICT,
        fallback_model=request.single_model or "granite3.1-moe:3b",
    )

    # Store as session override
    target_session = session_id or "default"
    session_agent_configs[target_session] = agent_config

    return {
        "status": "ok",
        "session_id": target_session,
        "config": {
            "mode": agent_config.mode.value,
            "planner_model": agent_config.planner_model,
            "worker_model": agent_config.worker_model,
            "verifier_model": agent_config.verifier_model,
            "single_model_mode": request.single_model_mode,
        },
        "message": f"Agent config updated for session {target_session} (non-destructive override)"
    }


@app.delete("/api/agents/config")
async def reset_agent_config(session_id: Optional[str] = None):
    """
    Reset session agent config to defaults.

    This removes the session-specific override, reverting to default config.
    """
    target_session = session_id or "default"

    if target_session in session_agent_configs:
        del session_agent_configs[target_session]
        return {
            "status": "ok",
            "session_id": target_session,
            "message": "Agent config reset to defaults"
        }

    return {
        "status": "ok",
        "session_id": target_session,
        "message": "No session override to reset"
    }


@app.get("/api/agents/reasoning")
async def get_reasoning_traces(session_id: Optional[str] = None, limit: int = 50):
    """
    Get reasoning/chain-of-thought traces for a session.

    Returns the internal reasoning steps the agent took to reach conclusions.
    """
    target_session = session_id or "default"

    traces = session_reasoning_traces.get(target_session, [])

    # Also check for reasoning in events.jsonl
    if target_session in active_sessions:
        session = active_sessions[target_session]
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            events_file = Path(sandbox_root) / ".agent_logs" / "events.jsonl"
            if events_file.exists():
                try:
                    lines = events_file.read_text().strip().split('\n')
                    for line in lines[-limit:]:
                        event = json.loads(line)
                        # Extract reasoning-related events
                        if event.get("event") in ["reasoning", "thinking", "plan", "analysis"]:
                            traces.append(event)
                        # Also extract LLM responses with reasoning tags
                        if "response" in event and "<thinking>" in str(event.get("response", "")):
                            traces.append({
                                "type": "reasoning",
                                "timestamp": event.get("timestamp"),
                                "content": event.get("response"),
                            })
                except Exception:
                    pass

    return {
        "session_id": target_session,
        "traces": traces[-limit:],
        "total": len(traces),
        "has_traces": len(traces) > 0
    }


@app.post("/api/agents/reasoning")
async def add_reasoning_trace(
    session_id: Optional[str] = None,
    trace_type: str = "reasoning",
    content: str = ""
):
    """
    Add a reasoning trace (internal use by agents).
    """
    target_session = session_id or "default"

    if target_session not in session_reasoning_traces:
        session_reasoning_traces[target_session] = []

    trace = {
        "type": trace_type,
        "timestamp": datetime.now().isoformat(),
        "content": content,
    }

    session_reasoning_traces[target_session].append(trace)

    # Keep only last 100 traces per session
    if len(session_reasoning_traces[target_session]) > 100:
        session_reasoning_traces[target_session] = session_reasoning_traces[target_session][-100:]

    return {"status": "ok", "trace_id": len(session_reasoning_traces[target_session]) - 1}


@app.get("/api/agents/personas")
async def get_agent_personas():
    """
    Get agent personas/system prompts for each role.
    """
    from ragix_core.agent_config import AGENT_PERSONAS

    return {
        "personas": {
            role.value: {
                "role": role.value,
                "system_prompt": persona[:500] + "..." if len(persona) > 500 else persona,
                "full_length": len(persona),
            }
            for role, persona in AGENT_PERSONAS.items()
        }
    }


@app.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get detailed workflow template by ID."""
    if not WORKFLOWS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Workflow templates not available")

    template = BUILTIN_TEMPLATES.get(workflow_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")

    return {
        "id": workflow_id,
        "name": template.name,
        "description": template.description,
        "version": template.version,
        "parameters": [
            {
                "name": p.name,
                "description": p.description,
                "type": p.param_type,
                "required": p.required,
                "default": p.default,
                "enum": p.enum,
            }
            for p in template.parameters
        ],
        "steps": [
            {
                "name": s.name,
                "agent_type": s.agent_type,
                "task_template": s.task_template,
                "capabilities": s.capabilities,
                "tools": s.tools,
                "max_iterations": s.max_iterations,
                "depends_on": s.depends_on,
                "conditions": s.conditions,
            }
            for s in template.steps
        ],
    }


@app.get("/api/logs/server")
async def get_server_logs(limit: int = 100):
    """Get server-level logs (from any sandbox with logs)."""
    all_logs = []

    # Check all active sessions
    for session_id, session in active_sessions.items():
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            log_file = Path(sandbox_root) / ".agent_logs" / "commands.log"
            if log_file.exists():
                lines = log_file.read_text().strip().split('\n')
                for line in lines[-limit:]:
                    all_logs.append({
                        "session": session_id,
                        "line": line
                    })

    # Also check default ragix-workspace
    default_sandbox = Path.home() / "ragix-workspace"
    if default_sandbox.exists():
        log_file = default_sandbox / ".agent_logs" / "commands.log"
        if log_file.exists():
            lines = log_file.read_text().strip().split('\n')
            for line in lines[-limit:]:
                all_logs.append({
                    "session": "default",
                    "line": line
                })

    # Sort by timestamp if possible, return most recent
    all_logs = all_logs[-limit:]

    return {
        "logs": all_logs,
        "total": len(all_logs)
    }


@app.get("/api/traces")
async def get_traces(session_id: Optional[str] = None, limit: int = 50):
    """Get tool execution traces from sessions."""
    traces = []

    # Determine which sessions to check
    sessions_to_check = []
    if session_id and session_id in active_sessions:
        sessions_to_check = [(session_id, active_sessions[session_id])]
    else:
        sessions_to_check = list(active_sessions.items())

    for sid, session in sessions_to_check:
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            events_file = Path(sandbox_root) / ".agent_logs" / "events.jsonl"
            if events_file.exists():
                lines = events_file.read_text().strip().split('\n')
                for line in lines[-limit:]:
                    try:
                        event = json.loads(line)
                        event["session"] = sid
                        traces.append(event)
                    except json.JSONDecodeError:
                        continue

    # Sort by timestamp
    traces.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "traces": traces[:limit],
        "total": len(traces)
    }


# =============================================================================
# Log Integrity API (SHA256 Hash Chain)
# =============================================================================

@app.get("/api/logs/integrity")
async def verify_log_integrity(session_id: Optional[str] = None):
    """
    Verify integrity of log files using SHA256 hash chain.

    Args:
        session_id: Optional session ID (defaults to checking all sessions)

    Returns:
        Integrity verification report
    """
    results = []

    # Determine which sessions to check
    sessions_to_check = []
    if session_id and session_id in active_sessions:
        sessions_to_check = [(session_id, active_sessions[session_id])]
    else:
        sessions_to_check = list(active_sessions.items())

    for sid, session in sessions_to_check:
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            log_dir = Path(sandbox_root) / ".agent_logs"
            if log_dir.exists():
                try:
                    hasher = ChainedLogHasher(log_dir=log_dir)
                    report = hasher.verify_chain()
                    results.append({
                        "session": sid,
                        "sandbox": sandbox_root,
                        "valid": report.valid,
                        "total_entries": report.total_entries,
                        "verified_entries": report.verified_entries,
                        "first_invalid_entry": report.first_invalid_entry,
                        "errors": report.errors,
                        "log_file": report.log_file,
                        "verification_time": report.verification_time
                    })
                except Exception as e:
                    results.append({
                        "session": sid,
                        "sandbox": sandbox_root,
                        "valid": False,
                        "error": str(e)
                    })

    # Overall status
    all_valid = all(r.get("valid", False) for r in results) if results else True

    return {
        "overall_valid": all_valid,
        "sessions_checked": len(results),
        "results": results
    }


@app.get("/api/logs/chain")
async def get_log_chain(session_id: Optional[str] = None, limit: int = 50):
    """
    Get hash chain entries with full metadata.

    Args:
        session_id: Optional session ID
        limit: Maximum entries to return

    Returns:
        List of hash chain entries with cryptographic metadata
    """
    entries = []

    # Determine which sessions to check
    sessions_to_check = []
    if session_id and session_id in active_sessions:
        sessions_to_check = [(session_id, active_sessions[session_id])]
    else:
        sessions_to_check = list(active_sessions.items())

    for sid, session in sessions_to_check:
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            hash_file = Path(sandbox_root) / ".agent_logs" / "commands.log.sha256"
            if hash_file.exists():
                try:
                    lines = hash_file.read_text().strip().split('\n')
                    for line in lines[-limit:]:
                        if line.strip():
                            entry = json.loads(line)
                            entry["session"] = sid
                            entries.append(entry)
                except Exception as e:
                    pass

    # Sort by sequence (most recent first)
    entries.sort(key=lambda x: x.get("sequence", 0), reverse=True)

    return {
        "entries": entries[:limit],
        "total": len(entries),
        "has_integrity": len(entries) > 0
    }


@app.get("/api/logs/stats")
async def get_log_stats(session_id: Optional[str] = None):
    """
    Get log statistics for sessions.

    Args:
        session_id: Optional session ID

    Returns:
        Log statistics including size, entry counts, and chain status
    """
    stats = []

    # Determine which sessions to check
    sessions_to_check = []
    if session_id and session_id in active_sessions:
        sessions_to_check = [(session_id, active_sessions[session_id])]
    else:
        sessions_to_check = list(active_sessions.items())

    for sid, session in sessions_to_check:
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            log_dir = Path(sandbox_root) / ".agent_logs"
            log_file = log_dir / "commands.log"
            hash_file = log_dir / "commands.log.sha256"
            events_file = log_dir / "events.jsonl"

            session_stats = {
                "session": sid,
                "sandbox": sandbox_root,
                "log_dir_exists": log_dir.exists(),
            }

            if log_file.exists():
                session_stats["log_file"] = {
                    "path": str(log_file),
                    "size_bytes": log_file.stat().st_size,
                    "size_kb": round(log_file.stat().st_size / 1024, 2),
                    "entries": sum(1 for _ in open(log_file))
                }

            if hash_file.exists():
                session_stats["hash_chain"] = {
                    "path": str(hash_file),
                    "size_bytes": hash_file.stat().st_size,
                    "entries": sum(1 for _ in open(hash_file)),
                    "algorithm": "sha256"
                }

            if events_file.exists():
                session_stats["events_file"] = {
                    "path": str(events_file),
                    "size_bytes": events_file.stat().st_size,
                    "entries": sum(1 for _ in open(events_file))
                }

            stats.append(session_stats)

    return {
        "sessions": stats,
        "total_sessions": len(stats)
    }


# =============================================================================
# WebSocket Chat
# =============================================================================

# Session agents cache
session_agents: Dict[str, UnixRAGAgent] = {}


def get_or_create_agent(session: Dict[str, Any]) -> UnixRAGAgent:
    """Get or create an agent for a session."""
    session_id = session["id"]

    # Check for session-specific agent config
    agent_config = session_agent_configs.get(session_id)
    if not agent_config:
        # Try to get default from global config
        try:
            config = get_config()
            agent_config = config.agents if hasattr(config, 'agents') and config.agents else AgentConfig()
        except Exception:
            agent_config = AgentConfig()

    # Determine which model to use based on agent config
    # For now, use the worker model as the primary model (it does the most work)
    model_to_use = agent_config.get_model(AgentRole.WORKER)

    # Invalidate cached agent if model changed
    if session_id in session_agents:
        cached_agent = session_agents[session_id]
        if hasattr(cached_agent, 'llm') and cached_agent.llm.model != model_to_use:
            del session_agents[session_id]

    if session_id not in session_agents:
        # Create LLM with the configured model
        llm = OllamaLLM(model=model_to_use)

        # Determine dry_run based on profile
        profile = session.get("profile", "dev")
        dry_run = profile == "strict"

        # Create sandbox
        sandbox = ShellSandbox(
            root=session.get("sandbox_root", LAUNCH_DIRECTORY),
            dry_run=dry_run,
            profile=profile,
            allow_git_destructive=False
        )

        # Create agent (dataclass with llm and shell attributes)
        agent = UnixRAGAgent(
            llm=llm,
            shell=sandbox
        )

        # Store agent config reference for later use
        agent._agent_config = agent_config

        session_agents[session_id] = agent

        # Log reasoning trace for agent creation
        if session_id not in session_reasoning_traces:
            session_reasoning_traces[session_id] = []
        session_reasoning_traces[session_id].append({
            "type": "system",
            "timestamp": datetime.now().isoformat(),
            "content": f"Agent initialized with model: {model_to_use}, mode: {agent_config.mode.value}",
        })

    return session_agents[session_id]


async def run_agent_async(agent: UnixRAGAgent, message: str, session_id: str = "default") -> Tuple[str, List[Dict]]:
    """
    Run agent in a thread pool to avoid blocking.

    Uses step_with_reasoning for complex tasks (Planner/Worker/Verifier loop).

    Returns:
        Tuple of (response_text, reasoning_traces)
    """
    import concurrent.futures

    def execute_step():
        """Execute agent step with reasoning and format response."""
        # Use reasoning-enabled step
        cmd_result, response, traces = agent.step_with_reasoning(message)

        # Build response string
        parts = []
        if response:
            parts.append(response)
        if cmd_result:
            parts.append(f"\n**Command executed:**\n```\n{cmd_result.as_text_block()}\n```")

        result_text = "\n".join(parts) if parts else "No response from agent."
        return result_text, traces

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result_text, traces = await loop.run_in_executor(pool, execute_step)

    # Store reasoning traces in session
    if session_id not in session_reasoning_traces:
        session_reasoning_traces[session_id] = []
    session_reasoning_traces[session_id].extend(traces)

    # Keep only last 100 traces per session
    if len(session_reasoning_traces[session_id]) > 100:
        session_reasoning_traces[session_id] = session_reasoning_traces[session_id][-100:]

    return result_text, traces


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for chat interaction."""
    await websocket.accept()
    active_websockets.append(websocket)

    if session_id not in active_sessions:
        await websocket.send_json({
            "type": "error",
            "message": "Session not found"
        })
        await websocket.close()
        return

    session = active_sessions[session_id]

    # Initialize agent for this session
    try:
        agent = get_or_create_agent(session)

        # Get actual model from agent (not the session's original setting)
        actual_model = agent.llm.model if hasattr(agent, 'llm') else session.get('model', 'mistral')
        agent_mode = agent._agent_config.mode.value if hasattr(agent, '_agent_config') else 'unknown'

        await websocket.send_json({
            "type": "status",
            "message": f"Connected to session {session_id} (model: {actual_model}, mode: {agent_mode})"
        })

        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "chat":
                user_message = data.get("message", "")

                # Echo user message
                await websocket.send_json({
                    "type": "user_message",
                    "message": user_message,
                    "timestamp": datetime.now().isoformat()
                })

                # Store in history
                session["message_history"].append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                })

                # Send thinking indicator
                await websocket.send_json({
                    "type": "thinking",
                    "message": "Agent is processing...",
                    "timestamp": datetime.now().isoformat()
                })

                try:
                    # Run the agent with reasoning (Planner/Worker/Verifier)
                    response, traces = await run_agent_async(agent, user_message, session_id)

                    # Send reasoning traces if any (for Reasoning tab visualization)
                    if traces:
                        await websocket.send_json({
                            "type": "reasoning_traces",
                            "traces": traces,
                            "timestamp": datetime.now().isoformat()
                        })

                    # Send agent response
                    await websocket.send_json({
                        "type": "agent_message",
                        "message": response,
                        "timestamp": datetime.now().isoformat()
                    })

                    session["message_history"].append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })

                except Exception as agent_error:
                    error_msg = f"Agent error: {str(agent_error)}"
                    await websocket.send_json({
                        "type": "error",
                        "message": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })

            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
            await websocket.close()
        except:
            pass
        if websocket in active_websockets:
            active_websockets.remove(websocket)


def main():
    """Main entry point for ragix-web CLI."""
    parser = argparse.ArgumentParser(
        prog="ragix-web",
        description="RAGIX Web UI - Local-first development assistant"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--sandbox-root",
        default=".",
        help="Default sandbox root (default: current directory, like Claude Code)"
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Default Ollama model (default: mistral)"
    )
    parser.add_argument(
        "--profile",
        choices=["strict", "dev", "unsafe"],
        default="dev",
        help="Default safety profile (default: dev)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    args = parser.parse_args()

    # Create default session
    sandbox_root = os.path.expanduser(args.sandbox_root)
    sandbox_root = os.path.abspath(sandbox_root)
    os.makedirs(sandbox_root, exist_ok=True)

    default_session_id = "default"
    active_sessions[default_session_id] = {
        "id": default_session_id,
        "sandbox_root": sandbox_root,
        "model": args.model,
        "profile": args.profile,
        "created_at": datetime.now().isoformat(),
        "message_history": []
    }

    # Print startup info
    print("=" * 60)
    print("RAGIX Web UI")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Sandbox: {sandbox_root}")
    print(f"Model: {args.model}")
    print(f"Profile: {args.profile}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop")
    print()

    # Open browser if requested
    if not args.no_browser:
        import webbrowser
        try:
            webbrowser.open(f"http://{args.host}:{args.port}")
        except Exception:
            pass

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
