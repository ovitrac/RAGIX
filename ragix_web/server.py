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
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from datetime import datetime
import json

# Setup logging
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Error: FastAPI dependencies not installed.", file=sys.stderr)
    print("Install with: pip install 'ragix[web]'", file=sys.stderr)
    sys.exit(1)

from ragix_core import OllamaLLM, ShellSandbox, AgentLogger, LogLevel, extract_json_object
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
        HTMLRenderer,
        VisHTMLRenderer,
        VIS_RENDERER_THRESHOLD,
        get_optimal_renderer,
        DependencyType,
        NodeType,
        RadialExplorer,
        DSMRenderer,
    )
    from ragix_core.analysis_cache import (
        AnalysisCache,
        CachedAnalysis,
        get_cache,
        get_or_analyze,
    )
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False

# Maven imports (optional)
try:
    from ragix_core.maven import (
        MavenParser,
        parse_pom,
        scan_maven_projects,
        find_dependency_conflicts,
    )
    from ragix_core.maven_cache import (
        MavenCache,
        CachedMavenAnalysis,
        get_maven_cache,
    )
    MAVEN_AVAILABLE = True
except ImportError:
    MAVEN_AVAILABLE = False

# SonarQube imports (optional)
try:
    from ragix_core.sonar import (
        SonarClient,
        SonarSeverity,
        get_project_report,
    )
    from ragix_core.sonar_cache import (
        SonarCache,
        CachedSonarAnalysis,
        get_sonar_cache,
    )
    SONAR_AVAILABLE = True
except ImportError:
    SONAR_AVAILABLE = False

from ragix_unix import UnixRAGAgent

# Import prompt database
try:
    from ragix_core.prompts import get_prompt_database, PromptDatabase
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False

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
        reasoning_router,
        threads_router,
        rag_router,
        rag_project_router,
        audit_router,
    )
    from ragix_web.routers.sessions import set_sessions_store
    from ragix_web.routers.context import set_context_store
    from ragix_web.routers.agents import set_stores as set_agent_stores
    from ragix_web.routers.logs import set_sessions_store as set_logs_sessions
    from ragix_web.routers.reasoning import set_stores as set_reasoning_stores
    from ragix_web.routers.threads import set_threads_store
    from ragix_web.routers.rag import set_rag_store, _get_rag_state, _get_index_path
    from ragix_web.routers.rag_project import (
        set_project_rag_store,
        retrieve_project_rag_context,
        check_project_rag_available,
        is_project_rag_enabled,
        get_current_project,
    )
    from ragix_web.routers.audit import set_current_project as set_audit_project
    ROUTERS_AVAILABLE = True
except ImportError:
    ROUTERS_AVAILABLE = False


def retrieve_rag_context(session_id: str, query: str, top_k: int = 5, max_chars_per_chunk: int = 1500) -> Optional[str]:
    """
    Retrieve relevant context from RAG index for a query.

    Args:
        session_id: Session ID to check RAG state
        query: User query to search for
        top_k: Number of results to return
        max_chars_per_chunk: Maximum characters per chunk to include

    Returns:
        Formatted context string or None if RAG not enabled/available
    """
    if not ROUTERS_AVAILABLE:
        return None

    try:
        # Check if RAG is enabled for this session
        rag_state = _get_rag_state(session_id)
        if not rag_state.get("enabled"):
            return None

        index_path = _get_index_path()
        chunks_path = index_path / "chunks.json"

        if not chunks_path.exists():
            return None

        # Load chunks
        with open(chunks_path, 'r') as f:
            all_chunks = json.load(f)

        if not all_chunks:
            return None

        # Simple BM25-like search: score chunks by query term overlap
        query_terms = set(query.lower().split())
        # Also include common summarization terms
        summary_terms = {'summary', 'summarize', 'overview', 'describe', 'explain', 'what', 'about', 'content', 'file', 'document'}

        scored_chunks = []
        for chunk in all_chunks:
            content = chunk.get("content", "").lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            # Boost score if this is a general query (summarize, etc.)
            if query_terms & summary_terms:
                matches += 1  # Give all chunks a base score for summary queries
            if matches > 0:
                scored_chunks.append((matches, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: -x[0])

        # Take top_k results
        top_chunks = scored_chunks[:top_k]

        if not top_chunks:
            # For summary queries with no matches, return first chunks
            if query_terms & summary_terms and all_chunks:
                top_chunks = [(1, chunk) for chunk in all_chunks[:top_k]]
            else:
                return None

        # Format context with clear instructions
        context_parts = [
            "## 📚 DOCUMENT CONTEXT (from RAG Index)\n",
            "**IMPORTANT:** The following content has been retrieved from indexed documents.",
            "Use this content directly to answer the user's question. Do NOT search for files with shell commands.\n"
        ]

        total_chars = 0
        for i, (score, chunk) in enumerate(top_chunks, 1):
            file_path = chunk.get("file_path", "unknown")
            content = chunk.get("content", "")[:max_chars_per_chunk]
            total_chars += len(content)
            context_parts.append(f"### Document {i}: {file_path}\n{content}\n")

        context_parts.append(f"\n---\n*Retrieved {len(top_chunks)} relevant sections ({total_chars} chars) from indexed documents.*\n")

        return "\n".join(context_parts)

    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return None


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
# Cancellation events per session for interrupting reasoning
session_cancellation: Dict[str, threading.Event] = {}


def _extract_message_from_response(response: str) -> str:
    """
    Extract the actual message from a response that may contain raw JSON.

    If the response looks like a JSON action object with a "message" field,
    extract and return just the message content. This handles cases where
    the agent returns raw JSON instead of the extracted message.

    Args:
        response: Raw response string from agent

    Returns:
        Extracted message or original response
    """
    if not response:
        return response

    # Quick check: does it look like JSON?
    stripped = response.strip()
    if not (stripped.startswith('{') and stripped.endswith('}')):
        return response

    # Try to parse as JSON action
    try:
        action = extract_json_object(response)
        if action and isinstance(action, dict):
            action_type = action.get("action", "")
            message = action.get("message", "")

            # If it's a respond action with a message, return the message
            if action_type == "respond" and message:
                return message

            # Also handle bash_and_respond
            if action_type == "bash_and_respond" and message:
                return message
    except Exception:
        pass

    return response


# Request models
class SessionCreateRequest(BaseModel):
    """Request body for creating a session."""
    sandbox_root: str = ""  # Empty means use LAUNCH_DIRECTORY
    model: str = "qwen2.5:7b"
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

# v0.23: Storage for reasoning graph states and experience corpora
session_reasoning_states: Dict[str, Dict[str, Any]] = {}
session_experience_corpora: Dict[str, Any] = {}

# Register modular routers if available
if ROUTERS_AVAILABLE:
    # Set shared state references
    set_sessions_store(active_sessions)
    set_context_store(session_user_context)
    set_agent_stores(active_sessions, session_agent_configs, session_reasoning_traces)
    set_logs_sessions(active_sessions)
    set_reasoning_stores(active_sessions, session_reasoning_states, session_experience_corpora)
    set_threads_store(active_sessions, LAUNCH_DIRECTORY)
    set_rag_store(LAUNCH_DIRECTORY)
    set_project_rag_store(LAUNCH_DIRECTORY)  # v0.33: Project RAG

    # Include routers with their prefixes
    # Note: These provide modular, organized endpoints
    # The original inline endpoints are kept for backward compatibility
    # but will be deprecated in a future version
    app.include_router(sessions_router, tags=["Sessions (v2)"])
    app.include_router(memory_router, tags=["Memory (v2)"])
    app.include_router(context_router, tags=["Context (v2)"])
    app.include_router(agents_router, tags=["Agents (v2)"])
    app.include_router(logs_router, tags=["Logs (v2)"])
    app.include_router(reasoning_router, tags=["Reasoning Graph (v0.23)"])
    app.include_router(threads_router, tags=["Threads (v0.33)"])
    app.include_router(rag_router, tags=["Chat RAG (v0.33)"])
    app.include_router(rag_project_router, tags=["Project RAG (v0.33)"])
    app.include_router(audit_router, tags=["Code Audit (v0.4)"])


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
        "version": RAGIX_VERSION,
        "sessions": len(active_sessions),
        "ast_available": AST_AVAILABLE,
        "maven_available": MAVEN_AVAILABLE,
        "sonar_available": SONAR_AVAILABLE,
        "reports_available": REPORTS_AVAILABLE if 'REPORTS_AVAILABLE' in globals() else False
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


@app.get("/api/ollama/running")
async def get_ollama_running():
    """
    Get currently running/loaded models with VRAM usage from Ollama API.

    Uses /api/ps endpoint with caching (30s TTL).
    """
    try:
        from ragix_core import get_ollama_client
        client = get_ollama_client()

        if not client.is_available():
            return {
                "available": False,
                "error": "Ollama server not available",
                "models": []
            }

        models = client.get_running_models()
        return {
            "available": True,
            "models": models,
            "count": len(models),
            "cache_stats": client.get_cache_stats()
        }
    except Exception as e:
        logger.error(f"Error getting running models: {e}")
        return {
            "available": False,
            "error": str(e),
            "models": []
        }


@app.get("/api/ollama/model/{model_name:path}")
async def get_ollama_model_info(model_name: str, refresh: bool = False):
    """
    Get detailed model information including VRAM, quantization, context size.

    Uses /api/show and /api/ps with caching (5min TTL for details, 30s for VRAM).

    Args:
        model_name: Model name (e.g., "mistral:latest")
        refresh: Force refresh from API
    """
    try:
        from ragix_core import get_ollama_client, get_model_context_limit
        client = get_ollama_client()

        if not client.is_available():
            return {
                "available": False,
                "error": "Ollama server not available",
                "model": None
            }

        info = client.get_model_info(model_name, force_refresh=refresh)

        # Include fallback context limit if not from API
        fallback_context = get_model_context_limit(model_name)

        return {
            "available": True,
            "model": {
                "name": info.name,
                "family": info.family,
                "parameter_size": info.parameter_size,
                "quantization": info.quantization,
                "context_length": info.context_length if info.context_length > 0 else fallback_context,
                "context_source": "ollama_api" if info.context_length > 0 else "hardcoded",
                "size_gb": round(info.size_gb, 2),
                "vram_gb": round(info.vram_gb, 2),
                "vram_bytes": info.vram_bytes,
                "expires_at": info.expires_at,
            },
            "cache_stats": client.get_cache_stats()
        }
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        return {
            "available": False,
            "error": str(e),
            "model": None
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


@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, request: SessionCreateRequest):
    """Update session configuration (model, profile, etc.)."""
    logger.info(f"PUT /api/sessions/{session_id}: model={request.model}, profile={request.profile}")
    if session_id not in active_sessions:
        # Create session if it doesn't exist (e.g., "default")
        active_sessions[session_id] = {
            "id": session_id,
            "sandbox_root": LAUNCH_DIRECTORY,
            "model": "qwen2.5:7b",
            "profile": "dev",
            "created_at": datetime.now().isoformat(),
            "message_history": []
        }

    session = active_sessions[session_id]

    # Update fields if provided
    if request.model:
        old_model = session.get("model", "")
        session["model"] = request.model
        # Clear cached agent if model changed (force recreation)
        if old_model != request.model and session_id in session_agents:
            del session_agents[session_id]
            logger.info(f"Session {session_id}: Model changed from {old_model} to {request.model}, agent cleared")

    if request.profile:
        session["profile"] = request.profile

    if request.sandbox_root:
        # Validate sandbox
        sandbox = os.path.expanduser(request.sandbox_root)
        sandbox = os.path.abspath(sandbox)
        if sandbox.startswith(LAUNCH_DIRECTORY):
            session["sandbox_root"] = sandbox

    return {
        "status": "updated",
        "session_id": session_id,
        "model": session.get("model"),
        "profile": session.get("profile"),
        "sandbox_root": session.get("sandbox_root")
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# Prompts API - Demo prompts database for quick selection
# =============================================================================

@app.get("/api/prompts")
async def list_prompts(complexity: Optional[str] = None):
    """
    List available demo prompts.

    Query params:
        complexity: Filter by complexity level (bypass, simple, moderate, complex)
    """
    if not PROMPTS_AVAILABLE:
        return {"available": False, "prompts": [], "error": "Prompt database not available"}

    db = get_prompt_database()

    if complexity:
        prompts = db.get_by_complexity(complexity)
    else:
        prompts = db.get_all()

    return {
        "available": True,
        "total": len(prompts),
        "prompts": [p.to_dict() for p in prompts],
        "summary": db.get_summary(),
    }


@app.get("/api/prompts/quick-actions")
async def get_quick_action_prompts():
    """Get quick action prompts for UI buttons."""
    if not PROMPTS_AVAILABLE:
        return {"available": False, "quick_actions": []}

    db = get_prompt_database()
    return {
        "available": True,
        "quick_actions": [p.to_dict() for p in db.get_quick_actions()],
    }


@app.get("/api/prompts/search")
async def search_prompts(q: str, complexity: Optional[str] = None):
    """
    Search prompts by keyword.

    Query params:
        q: Search query
        complexity: Optional complexity filter
    """
    if not PROMPTS_AVAILABLE:
        return {"available": False, "results": []}

    db = get_prompt_database()
    results = db.search(q, complexity)

    return {
        "available": True,
        "query": q,
        "total": len(results),
        "results": [p.to_dict() for p in results],
    }


@app.get("/api/prompts/{prompt_id}")
async def get_prompt_by_id(prompt_id: str):
    """Get a specific prompt by ID."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompt database not available")

    db = get_prompt_database()
    prompt = db.get_by_id(prompt_id)

    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")

    return prompt.to_dict()


@app.get("/api/prompts/random/{complexity}")
async def get_random_prompt(complexity: str):
    """Get a random prompt for a given complexity level."""
    if not PROMPTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prompt database not available")

    db = get_prompt_database()
    prompt = db.get_random(complexity)

    if not prompt:
        raise HTTPException(status_code=404, detail=f"No prompts found for complexity: {complexity}")

    return prompt.to_dict()


@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get comprehensive session status including actual agent configuration.

    This returns the real values being used (from agent config overrides),
    not just the originally stored session values.

    If the session doesn't exist (e.g., after server restart), it will be
    auto-created with default settings.
    """
    if session_id not in active_sessions:
        # Auto-create session with defaults (handles server restart case)
        # Get defaults from config or use fallbacks
        try:
            config = get_config()
            default_model = config.llm.model if hasattr(config, 'llm') and config.llm else "qwen2.5:7b"
            default_profile = "dev"
        except Exception:
            default_model = "qwen2.5:7b"
            default_profile = "dev"

        active_sessions[session_id] = {
            "id": session_id,
            "sandbox_root": LAUNCH_DIRECTORY,
            "model": default_model,
            "profile": default_profile,
            "created_at": datetime.now().isoformat(),
            "message_history": [],
            "auto_created": True  # Flag to indicate this was auto-created
        }

    session = active_sessions[session_id]

    # Session model is the single source of truth
    session_model = session.get("model", "qwen2.5:7b")

    # Get agent config (may be session-specific override)
    agent_config = session_agent_configs.get(session_id)
    if not agent_config:
        try:
            config = get_config()
            agent_config = config.agents if hasattr(config, 'agents') and config.agents else None
        except Exception:
            agent_config = None

    # Model resolution hierarchy:
    # 1. Session model = default (single source of truth)
    # 2. Agent Config = optional override (inherits from session if not explicitly set)
    # 3. Reasoning = inherits from Agent Config worker model
    if session_id in session_agents:
        # Agent exists - use its resolved model (most authoritative)
        agent = session_agents[session_id]
        actual_model = getattr(agent, '_resolved_model', None) or \
                       (agent.llm.model if hasattr(agent, 'llm') else session_model)
    elif agent_config:
        # Check if agent config has explicit override
        configured_model = agent_config.get_model(AgentRole.WORKER)
        # Use session model if agent config still has hardcoded default
        if configured_model == "granite3.1-moe:3b" and session_model != "granite3.1-moe:3b":
            actual_model = session_model
        else:
            actual_model = configured_model
    else:
        actual_model = session_model

    # Get reasoning config - inherits from agent worker model
    from ragix_web.routers.reasoning import _session_reasoning_config
    reasoning_config = _session_reasoning_config.get(session_id, {})
    reasoning_strategy = reasoning_config.get("strategy", os.environ.get("RAGIX_REASONING_STRATEGY", "graph_v30"))

    # For planner/worker/verifier display: show actual model if not explicitly overridden
    agent_mode = agent_config.mode.value if agent_config else "minimal"
    planner_model = agent_config.planner_model if agent_config else actual_model
    worker_model = agent_config.worker_model if agent_config else actual_model
    verifier_model = agent_config.verifier_model if agent_config else actual_model

    # If in minimal mode and models are default, show session model instead
    if agent_mode == "minimal" or not agent_config:
        planner_model = actual_model
        worker_model = actual_model
        verifier_model = actual_model

    return {
        "session_id": session_id,
        "sandbox_root": session.get("sandbox_root", ""),
        "model": actual_model,
        "session_model": session_model,  # Original session model for reference
        "profile": session.get("profile", "dev"),
        "reasoning_strategy": reasoning_strategy,
        "agent_mode": agent_mode,
        "planner_model": planner_model,
        "worker_model": worker_model,
        "verifier_model": verifier_model,
        "created_at": session.get("created_at", ""),
        "has_agent_override": session_id in session_agent_configs,
    }


@app.get("/api/sessions/{session_id}/context-window")
async def get_context_window_status(session_id: str):
    """
    Get context window usage for a session.

    Returns:
        - model: Current model name
        - context_limit: Maximum tokens for this model
        - tokens_used: Tokens used in current session
        - tokens_available: Remaining tokens
        - usage_percent: Percentage of context used
        - warning_threshold: Percentage at which to warn user (default 80%)
    """
    from ragix_core.agent_config import get_model_context_limit, get_model_info

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    model = session.get("model", "qwen2.5:7b")

    # Get model context limit
    context_limit = get_model_context_limit(model)
    model_info = get_model_info(model)

    # Get token usage from agent
    # Use cumulative tokens since reasoning loop tracks tokens via _update_token_stats
    # but doesn't update the agent's history directly
    tokens_used = 0
    cumulative_tokens = 0
    history_estimate = 0
    agent_found = session_id in session_agents
    if agent_found:
        agent = session_agents[session_id]
        # Get cumulative token stats (primary metric for reasoning)
        if hasattr(agent, 'get_token_stats'):
            stats = agent.get_token_stats()
            cumulative_tokens = stats.get("total_tokens", 0)
        # Get history-based estimate as fallback
        if hasattr(agent, 'get_history_token_estimate'):
            history_estimate = agent.get_history_token_estimate()
        # Use the higher of cumulative tokens or history estimate
        # This covers both reasoning mode (cumulative) and simple mode (history)
        tokens_used = max(cumulative_tokens, history_estimate + 500)

    # Calculate usage
    tokens_available = max(0, context_limit - tokens_used)
    usage_percent = (tokens_used / context_limit * 100) if context_limit > 0 else 0

    # Warning threshold (configurable, default 80%)
    warning_threshold = 80

    return {
        "session_id": session_id,
        "model": model,
        "model_info": model_info,
        "context_limit": context_limit,
        "tokens_used": tokens_used,  # Current context size
        "tokens_cumulative": cumulative_tokens,  # Total tokens sent this session
        "tokens_available": tokens_available,
        "usage_percent": round(usage_percent, 1),
        "warning_threshold": warning_threshold,
        "is_warning": usage_percent >= warning_threshold,
        "is_critical": usage_percent >= 95,
    }


@app.get("/api/sessions/{session_id}/memory")
async def get_memory_stats(session_id: str):
    """Get memory/history statistics for a session."""
    if session_id not in session_agents:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = session_agents[session_id]

    # Get memory stats from agent
    if hasattr(agent, 'get_memory_stats'):
        memory_stats = agent.get_memory_stats()
    else:
        memory_stats = {
            "message_count": len(agent.history) if hasattr(agent, 'history') else 0,
            "estimated_tokens": 0,
            "is_compacted": False
        }

    # Get context limit for compaction recommendation
    from ragix_core.agent_config import get_model_context_limit
    model = active_sessions.get(session_id, {}).get("model", "mistral")
    context_limit = get_model_context_limit(model)

    # Check if compaction is recommended
    should_compact = False
    if hasattr(agent, 'should_compact'):
        should_compact = agent.should_compact(context_limit)

    return {
        "session_id": session_id,
        **memory_stats,
        "context_limit": context_limit,
        "should_compact": should_compact,
        "compaction_threshold_percent": 80
    }


@app.post("/api/sessions/{session_id}/compact")
async def compact_memory(session_id: str, keep_recent: int = 4):
    """
    Compact conversation history by summarizing older messages.

    This reduces context usage while preserving key information.

    Args:
        session_id: Session to compact
        keep_recent: Number of recent messages to keep (default: 4)
    """
    if session_id not in session_agents:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = session_agents[session_id]

    if not hasattr(agent, 'compact_history'):
        raise HTTPException(
            status_code=400,
            detail="Agent does not support memory compaction"
        )

    # Perform compaction
    result = agent.compact_history(keep_recent=keep_recent)

    return {
        "session_id": session_id,
        **result
    }


# =============================================================================
# Episodic Memory Management API
# =============================================================================

@app.get("/api/sessions/{session_id}/episodic")
async def get_episodic_memory(session_id: str, limit: int = 50, offset: int = 0):
    """
    Get episodic memory entries for a session.

    Args:
        session_id: Session ID
        limit: Max entries to return (default: 50)
        offset: Skip this many entries (default: 0)

    Returns:
        List of episodic memory entries (newest first)
    """
    # Return empty list if agent not initialized yet (no messages sent)
    if session_id not in session_agents:
        return {"entries": [], "total": 0, "stats": {}, "message": "No memories yet - send a message first"}

    agent = session_agents[session_id]

    if not hasattr(agent, '_episodic_memory') or agent._episodic_memory is None:
        return {"entries": [], "total": 0, "stats": {}}

    memory = agent._episodic_memory
    entries = memory.list_entries(limit=limit, offset=offset)
    stats = memory.get_stats()

    return {
        "session_id": session_id,
        "entries": entries,
        "total": stats.get("total_entries", 0),
        "offset": offset,
        "limit": limit,
        "stats": stats
    }


@app.get("/api/sessions/{session_id}/episodic/search")
async def search_episodic_memory(session_id: str, q: str, limit: int = 20):
    """
    Search episodic memory entries.

    Args:
        session_id: Session ID
        q: Search query
        limit: Max results (default: 20)
    """
    # Return empty results if agent not initialized yet (no messages sent)
    if session_id not in session_agents:
        return {"results": [], "query": q, "count": 0, "message": "No memories yet - send a message first"}

    agent = session_agents[session_id]

    if not hasattr(agent, '_episodic_memory') or agent._episodic_memory is None:
        return {"results": [], "query": q, "count": 0}

    memory = agent._episodic_memory
    results = memory.search_entries(q, limit=limit)

    return {
        "session_id": session_id,
        "query": q,
        "results": results,
        "count": len(results)
    }


@app.get("/api/sessions/{session_id}/episodic/{task_id}")
async def get_episodic_entry(session_id: str, task_id: str):
    """Get a specific episodic memory entry."""
    if session_id not in session_agents:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = session_agents[session_id]

    if not hasattr(agent, '_episodic_memory') or agent._episodic_memory is None:
        raise HTTPException(status_code=404, detail="No episodic memory")

    memory = agent._episodic_memory
    entry = memory.get_entry(task_id)

    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"session_id": session_id, "entry": entry}


@app.delete("/api/sessions/{session_id}/episodic/{task_id}")
async def delete_episodic_entry(session_id: str, task_id: str):
    """Delete a specific episodic memory entry."""
    if session_id not in session_agents:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = session_agents[session_id]

    if not hasattr(agent, '_episodic_memory') or agent._episodic_memory is None:
        raise HTTPException(status_code=404, detail="No episodic memory")

    memory = agent._episodic_memory
    deleted = memory.delete_entry(task_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"session_id": session_id, "deleted": task_id}


@app.delete("/api/sessions/{session_id}/episodic")
async def clear_episodic_memory(session_id: str, before: str = None):
    """
    Clear episodic memory entries.

    Args:
        session_id: Session ID
        before: If provided, delete entries before this ISO timestamp.
                If not provided, delete ALL entries.
    """
    if session_id not in session_agents:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = session_agents[session_id]

    if not hasattr(agent, '_episodic_memory') or agent._episodic_memory is None:
        return {"deleted": 0}

    memory = agent._episodic_memory

    if before:
        deleted = memory.delete_entries_before(before)
    else:
        deleted = memory.clear_all_entries()

    return {
        "session_id": session_id,
        "deleted": deleted,
        "before": before
    }


@app.post("/api/files/convert")
async def convert_file(file: UploadFile = File(...)):
    """
    Convert PDF, DOCX, ODT, or RTF files to text.

    Uses pdftotext for PDF files and pandoc for other document formats.
    Configuration is read from ragix.yaml (converters section).
    Returns extracted text content for use in chat context.
    """
    import shutil
    import tempfile

    # Get converter configuration
    config = get_config()
    conv_config = config.converters

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Get file extension
    ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    supported_extensions = ['pdf', 'docx', 'doc', 'odt', 'rtf']

    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: {', '.join(supported_extensions)}"
        )

    # Check file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > conv_config.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.1f}MB (max: {conv_config.max_file_size_mb}MB)"
        )

    # Save uploaded file to temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        extracted_text = ""

        if ext == 'pdf':
            # Use pdftotext for PDF conversion
            pdftotext_cfg = conv_config.pdftotext
            if not pdftotext_cfg.enabled:
                raise HTTPException(status_code=400, detail="PDF conversion is disabled in configuration")

            pdftotext_path = pdftotext_cfg.path or 'pdftotext'
            if not shutil.which(pdftotext_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"pdftotext not found at '{pdftotext_path}'. Install with: sudo apt install poppler-utils"
                )

            # Build command with configured options
            cmd = [pdftotext_path] + pdftotext_cfg.options + [tmp_path, '-']

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=conv_config.timeout
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"PDF conversion failed: {result.stderr}"
                )

            extracted_text = result.stdout

        else:
            # Use pandoc for other document formats (docx, doc, odt, rtf)
            pandoc_cfg = conv_config.pandoc
            if not pandoc_cfg.enabled:
                raise HTTPException(status_code=400, detail="Document conversion is disabled in configuration")

            pandoc_path = pandoc_cfg.path or 'pandoc'
            if not shutil.which(pandoc_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"pandoc not found at '{pandoc_path}'. Install with: sudo apt install pandoc"
                )

            # Build command with configured options and output format
            output_format = pandoc_cfg.output_format or 'plain'
            cmd = [pandoc_path, '-t', output_format] + pandoc_cfg.options + [tmp_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=conv_config.timeout
            )

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Document conversion failed: {result.stderr}"
                )

            extracted_text = result.stdout

        # Clean up extracted text
        extracted_text = extracted_text.strip()

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No text content extracted from document"
            )

        return {
            "filename": file.filename,
            "content": extracted_text,
            "char_count": len(extracted_text),
            "word_count": len(extracted_text.split())
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail=f"Conversion timed out after {conv_config.timeout}s")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


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

def get_cached_data_only(
    target_path: Path,
    use_cache: bool = True
) -> tuple:
    """
    Get cached metrics/cycles data WITHOUT rebuilding the graph.
    Use this for endpoints that don't need the graph object.

    Returns:
        Tuple of (metrics_dict, cycles, symbols, dependencies, was_cached)
        Returns (None, None, None, None, False) if not cached
    """
    if not use_cache:
        return None, None, None, None, False

    cache = get_cache()
    fingerprint, file_count, total_size = cache.get_fingerprint(target_path)

    if cache.is_cached(fingerprint):
        cached = cache.load(fingerprint)
        if cached and cached.metrics:
            return cached.metrics, cached.cycles, cached.symbols, cached.dependencies, True

    return None, None, None, None, False


def get_cached_graph(
    target_path: Path,
    patterns: list = None,
    use_cache: bool = True
) -> tuple:
    """
    Build dependency graph with caching support.

    Caching is now handled internally by build_dependency_graph().
    This helper provides metrics and cycles along with the graph.

    Args:
        target_path: Path to analyze
        patterns: Optional file patterns (e.g., ["*.py", "*.java"])
        use_cache: Whether to use cached results

    Returns:
        Tuple of (graph, metrics_dict, cycles, was_cached)
    """
    from ragix_core.code_metrics import calculate_metrics_from_graph

    cache = get_cache()
    fingerprint, file_count, total_size = cache.get_fingerprint(target_path)

    # Check cache status before building
    was_cached = use_cache and cache.is_cached(fingerprint)

    # Build graph (caching handled internally by build_dependency_graph)
    graph = build_dependency_graph([target_path], patterns=patterns, use_cache=use_cache)

    # Try to get cached metrics/cycles
    if was_cached:
        cached = cache.load(fingerprint)
        if cached and cached.metrics:
            return graph, cached.metrics, cached.cycles, True

    # Calculate fresh metrics and cycles
    cycles = graph.detect_cycles()
    metrics = calculate_metrics_from_graph(graph)
    metrics_dict = metrics.summary() if hasattr(metrics, 'summary') else {}

    # Add hotspots to metrics
    hotspots = [{"name": name, "complexity": cc} for name, cc in metrics.get_hotspots(20)]
    metrics_dict['hotspots'] = hotspots

    return graph, metrics_dict, cycles, was_cached


@app.get("/api/ast/status")
async def ast_status():
    """Check AST analysis availability."""
    return {
        "available": AST_AVAILABLE,
        "message": "AST analysis ready" if AST_AVAILABLE else "Install with: pip install 'ragix[ast]'"
    }


# =============================================================================
# Cache Management API
# =============================================================================

@app.get("/api/ast/cache/info")
async def get_cache_info(path: str):
    """
    Get cache information for a project.

    Args:
        path: Project path to check

    Returns:
        Cache info including fingerprint, whether cached, and metadata
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        cache = get_cache()
        fingerprint, file_count, total_size = cache.get_fingerprint(target_path)
        is_cached = cache.is_cached(fingerprint)
        cache_info = cache.get_cache_info(fingerprint) if is_cached else None

        return {
            "path": str(target_path),
            "fingerprint": fingerprint,
            "file_count": file_count,
            "total_size": total_size,
            "is_cached": is_cached,
            "cache_info": cache_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/cache/stats")
async def get_cache_stats():
    """Get overall cache statistics."""
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    try:
        cache = get_cache()
        stats = cache.get_stats()
        cached_list = cache.list_cached()

        return {
            **stats,
            "entries": cached_list[:20]  # Return last 20 entries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ast/cache/clear")
async def clear_cache(path: Optional[str] = None):
    """
    Clear cache entries.

    Args:
        path: If provided, clear cache for this project only. Otherwise clear all.
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    try:
        cache = get_cache()

        if path:
            target_path = Path(path).expanduser()
            count = cache.invalidate_project(target_path)
            return {"cleared": count, "path": str(target_path)}
        else:
            count = cache.clear()
            return {"cleared": count, "all": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/analyze")
async def analyze_project(
    path: str,
    use_cache: bool = True,
    lang: str = "auto"
):
    """
    Full project analysis with caching support.

    Returns symbols, dependencies, metrics, and cycles in one call.
    Results are cached based on project fingerprint.

    Args:
        path: Project path to analyze
        use_cache: Whether to use cached results (default True)
        lang: Language filter (auto, python, java)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    import time
    start_time = time.time()

    try:
        cache = get_cache()
        fingerprint, file_count, total_size = cache.get_fingerprint(target_path)

        # Check cache first
        if use_cache:
            cached = cache.load(fingerprint)
            if cached:
                return JSONResponse(content={
                    "cached": True,
                    "fingerprint": fingerprint,
                    "cached_at": cached.metadata.created_at,
                    "analysis_time_ms": cached.metadata.analysis_time_ms,
                    "file_count": cached.metadata.file_count,
                    "symbols_count": len(cached.symbols),
                    "dependencies_count": len(cached.dependencies),
                    "metrics": cached.metrics,
                    "cycles_count": len(cached.cycles),
                    "packages": cached.packages
                })

        # Run fresh analysis
        patterns = None
        if lang and lang != "auto":
            lang_patterns = {
                "python": ["*.py"],
                "java": ["*.java"],
                "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"],
            }
            patterns = lang_patterns.get(lang)

        # Disable internal caching - this endpoint handles its own detailed caching
        graph = build_dependency_graph([target_path], patterns=patterns, use_cache=False)
        metrics = calculate_metrics_from_graph(graph)
        cycles = graph.detect_cycles()

        # Convert symbols to serializable format (exclude imports and modules)
        excluded_types = {NodeType.IMPORT, NodeType.IMPORT_FROM, NodeType.MODULE}
        symbols_list = []
        for sym in graph.get_symbols():
            if sym.node_type not in excluded_types:
                symbols_list.append({
                    "name": sym.name,
                    "qualified_name": sym.qualified_name,
                    "type": sym.node_type.value if hasattr(sym.node_type, 'value') else str(sym.node_type),
                    "file": str(sym.file_path) if sym.file_path else None,
                    "line": sym.line_number
                })

        # Convert dependencies to serializable format
        deps_list = []
        for dep in graph.get_all_dependencies():
            deps_list.append({
                "source": dep.source,
                "target": dep.target,
                "type": dep.dep_type.value if hasattr(dep.dep_type, 'value') else str(dep.dep_type)
            })

        # Calculate packages (exclude imports and modules)
        packages: Dict[str, int] = {}
        for sym in graph.get_symbols():
            if sym.node_type not in excluded_types:
                parts = sym.qualified_name.split(".")
                pkg = ".".join(parts[:-1]) if len(parts) > 1 else "(default)"
                packages[pkg] = packages.get(pkg, 0) + 1

        # Convert metrics to serializable format
        total_blank_lines = sum(f.blank_lines for f in metrics.file_metrics)
        metrics_dict = {
            "total_files": metrics.total_files,
            "total_lines": metrics.total_lines,
            "code_lines": metrics.total_code_lines,
            "comment_lines": metrics.total_comment_lines,
            "blank_lines": total_blank_lines,
            "total_classes": metrics.total_classes,
            "total_functions": metrics.total_functions,
            "avg_complexity": round(metrics.avg_complexity_per_method, 2),
            "total_complexity": metrics.total_complexity,
            "maintainability_index": metrics.maintainability_index,
            "debt_hours": round(metrics.estimated_debt_hours, 1),
            "debt_days": round(metrics.estimated_debt_days, 1),
            "hotspots": [{"name": n, "complexity": c} for n, c in metrics.get_hotspots(20)]
        }

        analysis_time_ms = int((time.time() - start_time) * 1000)

        # Save to cache
        cache.save(
            fingerprint=fingerprint,
            project_path=target_path,
            symbols=symbols_list,
            dependencies=deps_list,
            metrics=metrics_dict,
            cycles=cycles,
            packages=packages,
            file_count=file_count,
            total_size=total_size,
            analysis_time_ms=analysis_time_ms,
            settings={"lang": lang}
        )

        return JSONResponse(content={
            "cached": False,
            "fingerprint": fingerprint,
            "analysis_time_ms": analysis_time_ms,
            "file_count": file_count,
            "symbols_count": len(symbols_list),
            "dependencies_count": len(deps_list),
            "metrics": metrics_dict,
            "cycles_count": len(cycles),
            "packages": packages
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/graph")
async def get_dependency_graph(
    path: str,
    lang: str = "auto",
    filter_types: Optional[str] = None,
    filter_deps: Optional[str] = None,
    max_nodes: int = 1000,
    use_cache: bool = True
):
    """
    Get dependency graph for a path.

    Args:
        path: Directory or file path to analyze
        lang: Language filter (auto, python, java)
        filter_types: Comma-separated node types (class,interface,method)
        filter_deps: Comma-separated dependency types (inheritance,call,import)
        max_nodes: Maximum nodes to return (default 1000)
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Get language patterns
        patterns = None
        if lang and lang != "auto":
            lang_patterns = {
                "python": ["*.py"],
                "java": ["*.java"],
                "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"],
            }
            patterns = lang_patterns.get(lang)

        # Use cached graph helper (builds graph but caches metrics/cycles)
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, patterns, use_cache)

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
        data["from_cache"] = was_cached

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
    lang: str = "auto",
    use_cache: bool = True
):
    """
    Get code metrics for a path.

    Args:
        path: Directory or file path to analyze
        lang: Language filter (auto, python, java)
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Try to get cached data first (no graph rebuild needed)
        cached_metrics, cached_cycles, cached_symbols, cached_deps, was_cached = get_cached_data_only(target_path, use_cache)

        if was_cached and cached_metrics:
            # Return cached metrics directly - FAST PATH
            result = {
                "path": str(target_path),
                "from_cache": True,
                "summary": {
                    "total_files": cached_metrics.get('files', 0),
                    "total_lines": cached_metrics.get('lines', {}).get('total', 0),
                    "code_lines": cached_metrics.get('lines', {}).get('code', 0),
                    "comment_lines": cached_metrics.get('lines', {}).get('comments', 0),
                    "blank_lines": cached_metrics.get('lines', {}).get('blank', 0),
                    "total_classes": cached_metrics.get('classes', 0),
                    "total_functions": cached_metrics.get('functions', 0),
                },
                "complexity": {
                    "average_cyclomatic": cached_metrics.get('complexity', {}).get('avg_per_method', 0),
                    "total_complexity": cached_metrics.get('complexity', {}).get('total', 0),
                },
                "maintainability": {
                    "index": cached_metrics.get('maintainability_index', 0),
                },
                "debt": {
                    "estimated_hours": cached_metrics.get('technical_debt', {}).get('hours', 0),
                    "estimated_days": cached_metrics.get('technical_debt', {}).get('days', 0),
                },
                "dependencies": {
                    "total": len(cached_deps) if cached_deps else cached_metrics.get('dependencies', 0),
                },
                "hotspots": cached_metrics.get('hotspots', []),
            }
            return JSONResponse(content=result)

        # Not cached - need to build graph and calculate metrics
        patterns = None
        if lang and lang != "auto":
            lang_patterns = {"python": ["*.py"], "java": ["*.java"], "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"]}
            patterns = lang_patterns.get(lang)

        # Disable internal caching - this endpoint handles its own caching with hotspots
        graph = build_dependency_graph([target_path], patterns=patterns, use_cache=False)
        metrics = calculate_metrics_from_graph(graph)
        cycles = graph.detect_cycles()

        # Convert to JSON-serializable format
        total_blank_lines = sum(f.blank_lines for f in metrics.file_metrics)
        hotspots = [{"name": name, "complexity": cc} for name, cc in metrics.get_hotspots(20)]

        result = {
            "path": str(target_path),
            "from_cache": False,
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
            "dependencies": {
                "total": len(graph._dependencies),
            },
            "hotspots": hotspots,
        }

        # Cache the results with hotspots included (exclude imports and modules)
        cache = get_cache()
        fingerprint, file_count, total_size = cache.get_fingerprint(target_path)
        excluded_types_cache = {NodeType.IMPORT, NodeType.IMPORT_FROM, NodeType.MODULE}
        symbols = [
            {
                'name': s.qualified_name,
                'type': s.node_type.value,
                'file': str(s.location.file) if s.location and s.location.file else '',
                'line': s.location.line if s.location else 0,
                'end_line': s.location.end_line if s.location and s.location.end_line else 0
            }
            for s in graph.get_symbols()
            if s.node_type not in excluded_types_cache
        ]
        dependencies = [
            {'source': d.source, 'target': d.target, 'type': d.dep_type.value}
            for d in graph._dependencies
        ]
        metrics_dict = metrics.summary() if hasattr(metrics, 'summary') else {}
        metrics_dict['hotspots'] = hotspots  # Include hotspots in cache

        packages = {}
        for sym in graph.get_symbols():
            if sym.node_type not in excluded_types_cache:
                pkg = sym.qualified_name.rsplit('.', 1)[0] if '.' in sym.qualified_name else '(default)'
                packages[pkg] = packages.get(pkg, 0) + 1

        cache.save(
            fingerprint=fingerprint,
            project_path=target_path,
            symbols=symbols,
            dependencies=dependencies,
            metrics=metrics_dict,
            cycles=cycles,
            packages=packages,
            file_count=file_count,
            total_size=total_size,
        )

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
    color_scheme: str = "default",
    renderer: str = "auto"
):
    """
    Generate interactive HTML visualization.

    Args:
        path: Directory to analyze
        title: Graph title
        color_scheme: Color scheme (default, pastel, dark, monochrome)
        renderer: Renderer choice (auto, d3, vis). Auto chooses vis.js for large graphs.
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        graph = build_dependency_graph([target_path])
        node_count = len(graph.get_symbols())

        # Configure visualization
        scheme = ColorScheme(color_scheme) if color_scheme in [s.value for s in ColorScheme] else ColorScheme.DEFAULT
        config = VizConfig(color_scheme=scheme)

        # Choose renderer based on node count or explicit choice
        if renderer == "auto":
            optimal = get_optimal_renderer(node_count)
        else:
            optimal = renderer if renderer in ("d3", "vis") else "d3"

        # Use VisHTMLRenderer for large graphs (>2000 nodes)
        if optimal == "vis":
            vis_renderer = VisHTMLRenderer(config)
            html_content = vis_renderer.render(graph, title=title)
        else:
            html_content = graph_to_html(graph, title=title, config=config)

        return HTMLResponse(content=html_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/matrix", response_class=HTMLResponse)
async def visualize_dsm_matrix(
    path: str,
    title: str = "Dependency Structure Matrix",
    level: str = "class",
    use_cache: bool = True
):
    """
    Generate interactive DSM (Dependency Structure Matrix) HTML visualization.

    Args:
        path: Directory to analyze
        title: Matrix title
        level: Aggregation level ('class', 'package', 'file')
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Use cached graph helper
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, None, use_cache)

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
    levels: int = 3,
    use_cache: bool = True
):
    """
    Get ego-centric radial graph data for a focal node.

    Args:
        path: Directory to analyze
        focal: Focal node (class name or qualified name). If not provided, auto-selects.
        levels: Maximum depth levels (default 3)
        use_cache: Whether to use cached results (default True)

    Returns:
        JSON with nodes organized by level and links
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Use cached graph helper
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, None, use_cache)
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
# Maven Analysis API
# =============================================================================

@app.get("/api/ast/maven")
async def maven_analysis(path: str, use_cache: bool = True):
    """
    Analyze Maven projects in a directory.

    Args:
        path: Directory containing pom.xml files
        use_cache: Whether to use cached results (default True)
    """
    if not MAVEN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Maven analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Try cache first - FAST PATH
        if use_cache:
            cache = get_maven_cache()
            fingerprint, pom_count, total_size = cache.get_fingerprint(target_path)

            if cache.is_cached(fingerprint):
                cached = cache.load(fingerprint)
                if cached:
                    return {
                        "found": True,
                        "from_cache": True,
                        "count": len(cached.projects),
                        "projects": cached.projects,
                        "conflicts": cached.conflicts,
                        "conflict_count": cached.conflict_count
                    }

        # Not cached - parse POMs
        pom_path = target_path / "pom.xml" if target_path.is_dir() else target_path
        if target_path.is_dir() and not pom_path.exists():
            projects = scan_maven_projects(target_path)
        elif pom_path.exists() and pom_path.name == "pom.xml":
            projects = [parse_pom(pom_path)]
        else:
            return {"found": False, "message": "No pom.xml found", "projects": []}

        # Convert to JSON-serializable format
        projects_data = []
        for project in projects:
            proj_data = {
                "gav": project.coordinate.gav,
                "group_id": project.coordinate.group_id,
                "artifact_id": project.coordinate.artifact_id,
                "version": project.coordinate.version,
                "name": project.name,
                "packaging": project.packaging,
                "parent": project.parent.gav if project.parent else None,
                "modules": project.modules,
                "dependencies": [
                    {
                        "gav": d.coordinate.gav,
                        "scope": d.scope.value,
                        "optional": d.optional,
                    }
                    for d in project.dependencies
                ],
                "compile_deps": len(project.get_compile_dependencies()),
                "test_deps": len(project.get_test_dependencies()),
            }
            projects_data.append(proj_data)

        # Check for conflicts if multiple projects
        conflicts_data = []
        conflict_count = 0
        if len(projects) > 1:
            conflicts = find_dependency_conflicts(projects)
            conflicts_data = [
                {
                    "artifact": c["artifact"],
                    "versions": {v: len(users) for v, users in c["versions"].items()}
                }
                for c in conflicts[:10]
            ]
            conflict_count = len(conflicts)

        # Cache results
        if use_cache:
            cache = get_maven_cache()
            fingerprint, pom_count, total_size = cache.get_fingerprint(target_path)
            cache.save(
                fingerprint=fingerprint,
                project_path=target_path,
                projects=projects_data,
                conflicts=conflicts_data,
                conflict_count=conflict_count,
                pom_count=pom_count,
                total_size=total_size
            )

        result = {
            "found": True,
            "from_cache": False,
            "count": len(projects_data),
            "projects": projects_data
        }
        if conflicts_data:
            result["conflicts"] = conflicts_data
            result["conflict_count"] = conflict_count

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/maven/page", response_class=HTMLResponse)
async def maven_page(path: str, use_cache: bool = True):
    """Generate Maven analysis HTML page."""
    if not MAVEN_AVAILABLE:
        raise HTTPException(status_code=503, detail="Maven analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Get Maven data (page needs actual objects for methods, cache used for fingerprint check)
        pom_path = target_path / "pom.xml" if target_path.is_dir() else target_path
        if target_path.is_dir() and not pom_path.exists():
            projects = scan_maven_projects(target_path)
        elif pom_path.exists():
            projects = [parse_pom(pom_path)]
        else:
            projects = []

        # Generate HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Maven Analysis - {target_path.name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #e94560; margin-bottom: 20px; }}
        h2 {{ color: #4fc3f7; margin: 20px 0 10px; font-size: 18px; }}
        .project {{ background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        .gav {{ font-family: monospace; background: #0f3460; padding: 8px 12px; border-radius: 4px; display: inline-block; margin-bottom: 10px; }}
        .meta {{ display: flex; gap: 20px; margin: 10px 0; font-size: 14px; color: #888; }}
        .deps-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .deps-table th {{ background: #0f3460; padding: 10px; text-align: left; }}
        .deps-table td {{ padding: 8px 10px; border-bottom: 1px solid #30363d; }}
        .scope {{ padding: 2px 8px; border-radius: 3px; font-size: 11px; }}
        .scope-compile {{ background: #238636; }}
        .scope-test {{ background: #6e40c9; }}
        .scope-provided {{ background: #1f6feb; }}
        .scope-runtime {{ background: #f85149; }}
        .conflict {{ background: #5c2d2d; border: 1px solid #f85149; border-radius: 4px; padding: 10px; margin: 5px 0; }}
        .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; }}
        .footer a {{ color: #4fc3f7; }}
    </style>
</head>
<body>
    <h1>Maven Analysis</h1>
    <p style="color:#888;margin-bottom:20px;">Path: {path}</p>
'''

        if not projects:
            html += '<p style="color:#f85149;">No Maven projects found (no pom.xml)</p>'
        else:
            # Check conflicts
            conflicts = find_dependency_conflicts(projects) if len(projects) > 1 else []

            if conflicts:
                html += f'<div class="conflict"><strong>⚠ {len(conflicts)} Dependency Conflict(s) Detected</strong></div>'

            for proj in projects:
                compile_deps = proj.get_compile_dependencies()
                test_deps = proj.get_test_dependencies()

                html += f'''
    <div class="project">
        <div class="gav">{proj.coordinate.gav}</div>
        {f'<p><strong>{proj.name}</strong></p>' if proj.name else ''}
        <div class="meta">
            <span>Packaging: {proj.packaging}</span>
            <span>Dependencies: {len(proj.dependencies)}</span>
            <span>Compile: {len(compile_deps)}</span>
            <span>Test: {len(test_deps)}</span>
        </div>
        {f'<p style="color:#888;font-size:13px;">Parent: {proj.parent.gav}</p>' if proj.parent else ''}
        {f'<p style="color:#888;font-size:13px;">Modules: {", ".join(proj.modules[:5])}{"..." if len(proj.modules) > 5 else ""}</p>' if proj.modules else ''}

        <h2>Dependencies ({len(proj.dependencies)})</h2>
        <table class="deps-table">
            <tr><th>Artifact</th><th>Version</th><th>Scope</th></tr>
'''
                for dep in proj.dependencies[:30]:
                    scope_class = f"scope-{dep.scope.value}"
                    html += f'''
            <tr>
                <td>{dep.coordinate.group_id}:{dep.coordinate.artifact_id}</td>
                <td>{dep.coordinate.version or 'inherited'}</td>
                <td><span class="scope {scope_class}">{dep.scope.value}</span></td>
            </tr>'''

                if len(proj.dependencies) > 30:
                    html += f'<tr><td colspan="3" style="text-align:center;color:#888;">... and {len(proj.dependencies) - 30} more</td></tr>'

                html += '</table></div>'

        html += '''
    <div class="footer">
        <p>Generated by RAGIX | Olivier Vitrac, PhD, HDR | <a href="mailto:olivier.vitrac@adservio.fr">olivier.vitrac@adservio.fr</a> | Adservio</p>
    </div>
</body>
</html>'''

        return HTMLResponse(content=html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SonarQube API
# =============================================================================

@app.get("/api/ast/sonar")
async def sonar_analysis(
    project: str,
    url: Optional[str] = None,
    token: Optional[str] = None,
    organization: Optional[str] = None,
    use_cache: bool = True
):
    """
    Query SonarQube/SonarCloud for project metrics.

    Args:
        project: Sonar project key
        url: Sonar server URL (defaults to SONAR_URL env or sonarcloud.io)
        token: API token (defaults to SONAR_TOKEN env)
        organization: Organization key for SonarCloud
        use_cache: Whether to use cached results (default True, TTL 5 min)
    """
    if not SONAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SonarQube integration not available")

    import os
    base_url = url or os.environ.get("SONAR_URL", "https://sonarcloud.io")
    api_token = token or os.environ.get("SONAR_TOKEN")
    org = organization or os.environ.get("SONAR_ORGANIZATION")

    try:
        # Try cache first - FAST PATH (TTL-based)
        if use_cache:
            cache = get_sonar_cache()
            cache_key = cache.get_cache_key(project, base_url)

            if cache.is_cached(cache_key):
                cached = cache.load(cache_key)
                if cached:
                    return {
                        "project_key": cached.project_key,
                        "server": cached.server,
                        "from_cache": True,
                        "quality_gate": cached.quality_gate,
                        "metrics": cached.metrics,
                        "issues": cached.issues,
                        "hotspots": cached.hotspots,
                        "top_issues": cached.top_issues
                    }

        # Not cached or expired - fetch from SonarQube
        client = SonarClient(base_url=base_url, token=api_token, organization=org)
        report = get_project_report(client, project)

        proj = report.project

        metrics = {
            "bugs": proj.bugs,
            "vulnerabilities": proj.vulnerabilities,
            "code_smells": proj.code_smells,
            "coverage": proj.coverage,
            "duplicated_lines_density": proj.duplicated_lines_density,
        }

        issues = {
            "total": len(report.issues),
            "by_severity": {}
        }

        # Count by severity
        for issue in report.issues:
            sev = issue.severity.value
            issues["by_severity"][sev] = issues["by_severity"].get(sev, 0) + 1

        # Top issues
        top_issues = [
            {
                "rule": i.rule,
                "message": i.message[:100],
                "severity": i.severity.value,
                "file": i.file_path,
                "line": i.line,
            }
            for i in report.issues[:10]
        ]

        # Cache results
        if use_cache:
            cache = get_sonar_cache()
            cache.save(
                project_key=project,
                server_url=base_url,
                quality_gate=proj.quality_gate.value,
                metrics=metrics,
                issues=issues,
                hotspots=len(report.hotspots),
                top_issues=top_issues
            )

        return {
            "project_key": project,
            "server": base_url,
            "from_cache": False,
            "quality_gate": proj.quality_gate.value,
            "metrics": metrics,
            "issues": issues,
            "hotspots": len(report.hotspots),
            "top_issues": top_issues
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/sonar/page", response_class=HTMLResponse)
async def sonar_page(
    project: str,
    url: Optional[str] = None,
    token: Optional[str] = None,
    organization: Optional[str] = None,
    use_cache: bool = True
):
    """Generate SonarQube analysis HTML page."""
    if not SONAR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SonarQube integration not available")

    import os
    base_url = url or os.environ.get("SONAR_URL", "https://sonarcloud.io")
    api_token = token or os.environ.get("SONAR_TOKEN")
    org = organization or os.environ.get("SONAR_ORGANIZATION")

    try:
        # Try cache first - FAST PATH (TTL-based)
        cached_data = None
        if use_cache:
            cache = get_sonar_cache()
            cache_key = cache.get_cache_key(project, base_url)
            if cache.is_cached(cache_key):
                cached_data = cache.load(cache_key)

        if cached_data:
            # Use cached data for rendering
            quality_gate = cached_data.quality_gate
            metrics = cached_data.metrics
            issues_data = cached_data.issues
            hotspots_count = cached_data.hotspots
            top_issues = cached_data.top_issues
        else:
            # Fetch from SonarQube
            client = SonarClient(base_url=base_url, token=api_token, organization=org)
            report = get_project_report(client, project)
            proj = report.project

            quality_gate = proj.quality_gate.value
            metrics = {
                "bugs": proj.bugs,
                "vulnerabilities": proj.vulnerabilities,
                "code_smells": proj.code_smells,
                "coverage": proj.coverage,
                "duplicated_lines_density": proj.duplicated_lines_density,
            }
            issues_data = {"total": len(report.issues), "by_severity": {}}
            for issue in report.issues:
                sev = issue.severity.value
                issues_data["by_severity"][sev] = issues_data["by_severity"].get(sev, 0) + 1

            hotspots_count = len(report.hotspots)
            top_issues = [
                {
                    "rule": i.rule,
                    "message": i.message[:100],
                    "severity": i.severity.value,
                    "file": i.file_path,
                    "line": i.line,
                }
                for i in report.issues[:30]  # More for page display
            ]

            # Cache results
            if use_cache:
                cache = get_sonar_cache()
                cache.save(
                    project_key=project,
                    server_url=base_url,
                    quality_gate=quality_gate,
                    metrics=metrics,
                    issues=issues_data,
                    hotspots=hotspots_count,
                    top_issues=top_issues
                )

        # Quality gate styling
        gate_colors = {"OK": "#238636", "WARN": "#d29922", "ERROR": "#f85149", "NONE": "#6e7681"}
        gate_color = gate_colors.get(quality_gate, "#6e7681")

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SonarQube - {project}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #e94560; margin-bottom: 10px; }}
        h2 {{ color: #4fc3f7; margin: 20px 0 10px; font-size: 18px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        .quality-gate {{ display: inline-block; padding: 10px 20px; border-radius: 6px; font-size: 18px; font-weight: bold; background: {gate_color}; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: #16213e; border-radius: 8px; padding: 20px; text-align: center; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #4fc3f7; }}
        .metric-label {{ font-size: 12px; color: #888; margin-top: 5px; }}
        .issues-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .issues-table th {{ background: #0f3460; padding: 10px; text-align: left; }}
        .issues-table td {{ padding: 8px 10px; border-bottom: 1px solid #30363d; font-size: 13px; }}
        .severity {{ padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
        .sev-BLOCKER {{ background: #7d1a1a; color: #fff; }}
        .sev-CRITICAL {{ background: #f85149; }}
        .sev-MAJOR {{ background: #d29922; }}
        .sev-MINOR {{ background: #1f6feb; }}
        .sev-INFO {{ background: #6e7681; }}
        .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; }}
        .footer a {{ color: #4fc3f7; }}
    </style>
</head>
<body>
    <h1>SonarQube Analysis</h1>
    <p class="subtitle">Project: {project} | Server: {base_url}</p>

    <div class="quality-gate">Quality Gate: {quality_gate}</div>

    <h2>Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{metrics.get('bugs', 0)}</div>
            <div class="metric-label">Bugs</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('vulnerabilities', 0)}</div>
            <div class="metric-label">Vulnerabilities</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('code_smells', 0)}</div>
            <div class="metric-label">Code Smells</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('coverage') or 'N/A'}{'%' if metrics.get('coverage') else ''}</div>
            <div class="metric-label">Coverage</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('duplicated_lines_density') or 'N/A'}{'%' if metrics.get('duplicated_lines_density') else ''}</div>
            <div class="metric-label">Duplication</div>
        </div>
        <div class="metric">
            <div class="metric-value">{hotspots_count}</div>
            <div class="metric-label">Security Hotspots</div>
        </div>
    </div>

    <h2>Issues ({issues_data.get('total', 0)})</h2>
    <table class="issues-table">
        <tr><th>Severity</th><th>Rule</th><th>Message</th><th>Location</th></tr>
'''
        for issue in top_issues[:30]:
            html += f'''
        <tr>
            <td><span class="severity sev-{issue.get('severity', '')}">{issue.get('severity', '')}</span></td>
            <td>{issue.get('rule', '')}</td>
            <td>{issue.get('message', '')[:80]}{'...' if len(issue.get('message', '')) > 80 else ''}</td>
            <td>{issue.get('file', '')}:{issue.get('line', '')}</td>
        </tr>'''

        total_issues = issues_data.get('total', 0)
        if total_issues > 30:
            html += f'<tr><td colspan="4" style="text-align:center;color:#888;">... and {total_issues - 30} more issues</td></tr>'

        html += '''
    </table>

    <div class="footer">
        <p>Generated by RAGIX | Olivier Vitrac, PhD, HDR | <a href="mailto:olivier.vitrac@adservio.fr">olivier.vitrac@adservio.fr</a> | Adservio</p>
    </div>
</body>
</html>'''

        return HTMLResponse(content=html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Cycles Detection API
# =============================================================================

@app.get("/api/ast/cycles")
async def detect_cycles(path: str, use_cache: bool = True):
    """
    Detect circular dependencies in the codebase.

    Args:
        path: Directory to analyze
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Try to get cached data first (no graph rebuild needed) - FAST PATH
        cached_metrics, cached_cycles, cached_symbols, cached_deps, was_cached = get_cached_data_only(target_path, use_cache)

        if was_cached and cached_cycles is not None:
            cycles = cached_cycles
            from_cache = True
        else:
            # Build graph (caching handled internally by build_dependency_graph)
            graph = build_dependency_graph([target_path], use_cache=use_cache)
            cycles = graph.detect_cycles()
            from_cache = False

        return {
            "path": str(target_path),
            "from_cache": from_cache,
            "has_cycles": len(cycles) > 0,
            "count": len(cycles),
            "cycles": [
                {
                    "length": len(cycle) - 1,
                    "nodes": cycle
                }
                for cycle in cycles[:20]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/cycles/page", response_class=HTMLResponse)
async def cycles_page(path: str, use_cache: bool = True):
    """Generate cycles detection HTML page."""
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Try to get cached data first (no graph rebuild needed) - FAST PATH
        cached_metrics, cached_cycles, cached_symbols, cached_deps, was_cached = get_cached_data_only(target_path, use_cache)

        if was_cached and cached_cycles is not None:
            cycles = cached_cycles
        else:
            # Build graph (caching handled internally by build_dependency_graph)
            graph = build_dependency_graph([target_path], use_cache=use_cache)
            cycles = graph.detect_cycles()

        status_color = "#f85149" if cycles else "#238636"
        status_text = f"{len(cycles)} Circular Dependencies Found" if cycles else "No Circular Dependencies"

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cycles Detection - {target_path.name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #e94560; margin-bottom: 10px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        .status {{ display: inline-block; padding: 10px 20px; border-radius: 6px; font-size: 18px; font-weight: bold; background: {status_color}; margin-bottom: 20px; }}
        .cycle {{ background: #16213e; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #f85149; }}
        .cycle-header {{ font-weight: bold; margin-bottom: 10px; }}
        .cycle-nodes {{ font-family: monospace; font-size: 13px; line-height: 1.8; }}
        .arrow {{ color: #f85149; margin: 0 8px; }}
        .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; }}
        .footer a {{ color: #4fc3f7; }}
        .success {{ text-align: center; padding: 40px; }}
        .success-icon {{ font-size: 64px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Circular Dependencies Detection</h1>
    <p class="subtitle">Path: {path}</p>

    <div class="status">{status_text}</div>
'''

        if not cycles:
            html += '''
    <div class="success">
        <div class="success-icon">✓</div>
        <p style="font-size: 18px; color: #238636;">No circular dependencies detected!</p>
        <p style="color: #888; margin-top: 10px;">Your codebase has a clean dependency structure.</p>
    </div>'''
        else:
            for i, cycle in enumerate(cycles[:20], 1):
                nodes_html = f'<span class="arrow">→</span>'.join(f'<span>{n}</span>' for n in cycle)
                html += f'''
    <div class="cycle">
        <div class="cycle-header">Cycle #{i} ({len(cycle) - 1} dependencies)</div>
        <div class="cycle-nodes">{nodes_html}</div>
    </div>'''

            if len(cycles) > 20:
                html += f'<p style="color:#888;text-align:center;">... and {len(cycles) - 20} more cycles</p>'

        html += '''
    <div class="footer">
        <p>Generated by RAGIX | Olivier Vitrac, PhD, HDR | <a href="mailto:olivier.vitrac@adservio.fr">olivier.vitrac@adservio.fr</a> | Adservio</p>
    </div>
</body>
</html>'''

        return HTMLResponse(content=html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Reports API
# =============================================================================

# Import report generation (optional)
try:
    from ragix_core.report_engine import (
        ReportEngine,
        ReportConfig,
        ReportFormat,
        ReportType,
        ComplianceStandard,
        generate_executive_summary,
        generate_technical_audit,
        generate_compliance_report,
    )
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False


@app.get("/api/reports/status")
async def reports_status():
    """Check if report generation is available."""
    return {
        "available": REPORTS_AVAILABLE,
        "types": ["executive", "technical", "compliance"] if REPORTS_AVAILABLE else [],
        "formats": ["html", "pdf"] if REPORTS_AVAILABLE else [],
        "standards": ["sonarqube", "owasp", "iso25010"] if REPORTS_AVAILABLE else []
    }


@app.get("/api/reports/generate", response_class=HTMLResponse)
async def generate_report(
    path: str,
    report_type: str = "executive",
    project_name: Optional[str] = None,
    standard: str = "sonarqube",
    use_cache: bool = True
):
    """
    Generate analysis report.

    Args:
        path: Directory to analyze
        report_type: executive, technical, or compliance
        project_name: Name for the report
        standard: Compliance standard (for compliance reports)
        use_cache: Ignored - reports always use fresh analysis for accuracy
    """
    if not REPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Report generation not available")

    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Reports need full graph with _files for detailed metrics calculation.
        # Cached "light" graphs lack _files, so we force fresh build for reports.
        # The metrics_data from cache is a summary dict, not a ProjectMetrics object.
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, None, use_cache=False)
        code_metrics = calculate_metrics_from_graph(graph)

        project = project_name or target_path.name

        # Generate appropriate report
        if report_type == "executive":
            html_content = generate_executive_summary(
                metrics=code_metrics,
                graph=graph,
                project_name=project,
                format=ReportFormat.HTML
            )
        elif report_type == "technical":
            html_content = generate_technical_audit(
                metrics=code_metrics,
                graph=graph,
                project_name=project,
                format=ReportFormat.HTML
            )
        elif report_type == "compliance":
            standard_map = {
                "sonarqube": ComplianceStandard.SONARQUBE,
                "owasp": ComplianceStandard.OWASP,
                "iso25010": ComplianceStandard.ISO_25010,
            }
            std = standard_map.get(standard.lower(), ComplianceStandard.SONARQUBE)
            html_content = generate_compliance_report(
                metrics=code_metrics,
                graph=graph,
                project_name=project,
                standard=std,
                format=ReportFormat.HTML
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown report type: {report_type}. Use executive, technical, or compliance"
            )

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/treemap", response_class=HTMLResponse)
async def treemap_visualization(
    path: str,
    metric: str = "loc",
    depth: int = 4,
    title: Optional[str] = None,
    use_cache: bool = True
):
    """
    Generate treemap visualization.

    Args:
        path: Directory to analyze
        metric: Size metric (loc, complexity, count, debt)
        depth: Maximum depth
        title: Custom title
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    try:
        from ragix_core.ast_viz_advanced import (
            TreemapConfig, TreemapMetric, TreemapRenderer
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Advanced visualizations not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Use cached graph helper
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, None, use_cache)

        metric_map = {
            "loc": TreemapMetric.LOC,
            "complexity": TreemapMetric.COMPLEXITY,
            "count": TreemapMetric.COUNT,
            "debt": TreemapMetric.DEBT,
        }
        config = TreemapConfig(
            metric=metric_map.get(metric.lower(), TreemapMetric.LOC),
            title=title or f"Treemap - {target_path.name}",
            max_depth=depth,
        )

        renderer = TreemapRenderer(config)
        html_content = renderer.render(graph)

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/sunburst", response_class=HTMLResponse)
async def sunburst_visualization(
    path: str,
    depth: int = 5,
    title: Optional[str] = None,
    use_cache: bool = True
):
    """
    Generate sunburst visualization.

    Args:
        path: Directory to analyze
        depth: Maximum depth
        title: Custom title
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    try:
        from ragix_core.ast_viz_advanced import SunburstConfig, SunburstRenderer
    except ImportError:
        raise HTTPException(status_code=503, detail="Advanced visualizations not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Use cached graph helper
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, None, use_cache)

        config = SunburstConfig(
            title=title or f"Sunburst - {target_path.name}",
            max_depth=depth,
        )

        renderer = SunburstRenderer(config)
        html_content = renderer.render(graph)

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/chord", response_class=HTMLResponse)
async def chord_visualization(
    path: str,
    group_by: str = "package",
    min_connections: int = 1,
    title: Optional[str] = None,
    use_cache: bool = True
):
    """
    Generate chord diagram visualization.

    Args:
        path: Directory to analyze
        group_by: Grouping method (package, file)
        min_connections: Minimum connections to show
        title: Custom title
        use_cache: Whether to use cached results (default True)
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    try:
        from ragix_core.ast_viz_advanced import ChordConfig, ChordRenderer
    except ImportError:
        raise HTTPException(status_code=503, detail="Advanced visualizations not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Use cached graph helper
        graph, metrics_data, cycles, was_cached = get_cached_graph(target_path, None, use_cache)

        config = ChordConfig(
            title=title or f"Dependencies - {target_path.name}",
            group_by=group_by,
            min_connections=min_connections,
        )

        renderer = ChordRenderer(config)
        html_content = renderer.render(graph)

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Code Tracker API (v0.5)
# =============================================================================

@app.get("/api/ast/tracker")
async def get_tracker_data(path: str, use_cache: bool = True):
    """
    Get code tracker data: outliers, complexity hotspots, dead code candidates, coupling issues.

    This endpoint combines statistical analysis from ragix_audit modules to provide
    interactive exploration of code quality issues.
    """
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        from ragix_core.code_metrics import calculate_metrics_from_graph

        # Build graph - disable cache to ensure full metrics computation
        graph_obj = build_dependency_graph([target_path], patterns=None, use_cache=False)
        metrics = calculate_metrics_from_graph(graph_obj)

        # Convert graph object to dict using D3Renderer
        config = VizConfig()
        renderer = D3Renderer(config)
        graph = renderer.to_dict(graph_obj)

        # Initialize result structure
        result = {
            "outliers": [],
            "complexity": [],
            "deadcode": [],
            "coupling": [],
            "stats": {
                "entropy": {},
                "inequality": {},
                "zones": {}
            }
        }

        # Process file metrics for outliers and complexity
        file_locs = []
        all_methods = []

        for fm in metrics.file_metrics:
            loc = fm.code_lines
            file_locs.append(loc)

            # Check for file-level outliers (size)
            if loc > 300:  # Files over 300 LOC are outliers
                result["outliers"].append({
                    "file": fm.path,
                    "name": Path(fm.path).name,
                    "type": "size",
                    "value": loc,
                    "severity": "high" if loc > 500 else "medium",
                    "line": 1
                })

            # Collect method complexity from class_metrics
            for cm in fm.class_metrics:
                for mm in cm.method_metrics:
                    cc = mm.cyclomatic_complexity
                    line = getattr(mm, 'line', 1)
                    all_methods.append({
                        "file": fm.path,
                        "method": mm.name,
                        "cc": cc,
                        "line": line
                    })

                    # High complexity methods
                    if cc > 5:
                        result["complexity"].append({
                            "file": fm.path,
                            "method": mm.name,
                            "cc": cc,
                            "line": line
                        })

            # Also check function_metrics (non-class methods)
            for mm in fm.function_metrics:
                cc = mm.cyclomatic_complexity
                line = getattr(mm, 'line', 1)
                all_methods.append({
                    "file": fm.path,
                    "method": mm.name,
                    "cc": cc,
                    "line": line
                })

                if cc > 5:
                    result["complexity"].append({
                        "file": fm.path,
                        "method": mm.name,
                        "cc": cc,
                        "line": line
                    })

        # Sort complexity by CC descending
        result["complexity"].sort(key=lambda x: x["cc"], reverse=True)

        # Compute statistical metrics
        try:
            from ragix_audit.entropy import normalized_entropy, compute_inequality_metrics
            from ragix_audit.coupling import CouplingComputer, ZoneType
            import math

            # Entropy - convert list to dict for entropy calculation
            if file_locs:
                # Create dict from list (file index -> LOC)
                file_sizes_dict = {f"file_{i}": loc for i, loc in enumerate(file_locs) if loc > 0}
                if file_sizes_dict:
                    entropy, norm_entropy = normalized_entropy(file_sizes_dict)
                    result["stats"]["entropy"] = {
                        "structural": round(entropy, 3),
                        "structural_pct": round(norm_entropy * 100, 1)
                    }

                # Inequality
                ineq = compute_inequality_metrics(file_locs)
                result["stats"]["inequality"] = {
                    "gini": round(ineq.gini, 3),
                    "cr4": round(ineq.cr4, 1),
                    "hhi": round(ineq.herfindahl, 3)
                }

            # Coupling analysis
            # Build dependency graph from graph data
            dependencies = {}
            package_classes = {}
            nodes_list = graph.get("nodes", [])

            for node in nodes_list:
                node_id = node.get("id", "")
                pkg = node_id.rsplit(".", 1)[0] if "." in node_id else "default"

                if pkg not in dependencies:
                    dependencies[pkg] = set()
                if pkg not in package_classes:
                    package_classes[pkg] = {"total": 0, "abstract": 0, "interfaces": 0}

                package_classes[pkg]["total"] += 1

            # In D3 format, source/target are indices
            for link in graph.get("links", []):
                source_idx = link.get("source", 0)
                target_idx = link.get("target", 0)

                # Get node IDs from indices
                if isinstance(source_idx, int) and isinstance(target_idx, int):
                    if source_idx < len(nodes_list) and target_idx < len(nodes_list):
                        source = nodes_list[source_idx].get("id", "")
                        target = nodes_list[target_idx].get("id", "")
                    else:
                        continue
                else:
                    source = str(source_idx)
                    target = str(target_idx)

                src_pkg = source.rsplit(".", 1)[0] if "." in source else "default"
                tgt_pkg = target.rsplit(".", 1)[0] if "." in target else "default"

                if src_pkg != tgt_pkg:
                    if src_pkg not in dependencies:
                        dependencies[src_pkg] = set()
                    dependencies[src_pkg].add(tgt_pkg)

            if dependencies:
                try:
                    coupling_computer = CouplingComputer()
                    analysis = coupling_computer.compute_from_graph(dependencies, package_classes)

                    result["stats"]["zones"] = {
                        "pain": analysis.packages_in_pain,
                        "useless": analysis.packages_useless,
                        "main_sequence": analysis.packages_on_sequence,
                        "balanced": analysis.packages_balanced
                    }

                    # Add coupling issues
                    zone_labels = {
                        ZoneType.ZONE_OF_PAIN: "Zone of Pain - Rigid",
                        ZoneType.ZONE_OF_USELESSNESS: "Zone of Uselessness",
                        ZoneType.MAIN_SEQUENCE: "Main Sequence",
                        ZoneType.BALANCED: "Balanced"
                    }

                    for pkg_name, pkg in analysis.packages.items():
                        if pkg.zone in (ZoneType.ZONE_OF_PAIN, ZoneType.ZONE_OF_USELESSNESS) or pkg.distance > 0.3:
                            zone_str = pkg.zone.value if hasattr(pkg.zone, 'value') else str(pkg.zone)
                            result["coupling"].append({
                                "package": pkg_name,
                                "type": "coupling",
                                "zone": zone_str.split(".")[-1] if "." in zone_str else zone_str,
                                "zone_label": zone_labels.get(pkg.zone, "Unknown"),
                                "ca": pkg.ca,
                                "ce": pkg.ce,
                                "instability": pkg.instability,
                                "abstractness": pkg.abstractness,
                                "distance": pkg.distance
                            })

                    # Sort by distance descending
                    result["coupling"].sort(key=lambda x: x.get("distance", 0), reverse=True)
                except Exception as coupling_err:
                    logger.warning(f"Coupling analysis error: {coupling_err}")

        except ImportError as e:
            logger.warning(f"Could not compute advanced metrics: {e}")

        # Dead code candidates (simplified heuristic)
        # Classes with no incoming dependencies and not entry points
        nodes = graph.get("nodes", [])
        incoming_deps = set()

        # In D3 format, source/target are indices into nodes array
        for link in graph.get("links", []):
            target_idx = link.get("target", 0)
            if isinstance(target_idx, int) and target_idx < len(nodes):
                incoming_deps.add(target_idx)

        for idx, node in enumerate(nodes):
            node_id = node.get("id", "")
            if idx not in incoming_deps:
                # No incoming dependencies - potential dead code
                # Skip common entry point patterns
                if not node_id or any(p in node_id.lower() for p in ["main", "application", "controller", "test", "config"]):
                    continue

                # Get file path and line from node (D3 nodes include file info)
                node_file = node.get("file", "") or ""
                node_line = node.get("line", 0) or 0
                node_type = node.get("type", "class") or "class"

                result["deadcode"].append({
                    "name": node_id,
                    "type": node_type,
                    "file": node_file,
                    "reason": "No incoming dependencies",
                    "confidence": 0.6,
                    "line": node_line
                })

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tracker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/file-view")
async def get_file_view(path: str):
    """
    Get file content with metrics for the code tracker viewer.

    Returns file content, line count, methods, and complexity metrics.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")

    try:
        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        lines = content.split("\n")
        loc = len([l for l in lines if l.strip() and not l.strip().startswith("//")])

        # Try to get method metrics
        methods = []
        avg_cc = 1.0

        # Simple regex-based method detection for Java
        import re
        method_pattern = re.compile(
            r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?'
            r'(?:[\w<>\[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        )

        for i, line in enumerate(lines, 1):
            match = method_pattern.search(line)
            if match:
                method_name = match.group(1)
                if method_name not in ("if", "for", "while", "switch", "catch"):
                    methods.append({
                        "name": method_name,
                        "line": i,
                        "cc": 1  # Would need real CC computation
                    })

        if methods:
            avg_cc = sum(m.get("cc", 1) for m in methods) / len(methods)

        return {
            "path": str(file_path),
            "content": content,
            "loc": loc,
            "line_count": len(lines),
            "method_count": len(methods),
            "methods": methods,
            "avg_cc": avg_cc
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Codebase Partitioner API (v0.55)
# =============================================================================

# Import partitioner module
try:
    from ragix_audit.partitioner import (
        CodebasePartitioner,
        PartitionConfig,
        ApplicationFingerprint,
        PartitionResult,
        partition_from_graph,
        create_sias_ticc_config,
    )
    PARTITIONER_AVAILABLE = True
except ImportError:
    PARTITIONER_AVAILABLE = False


class PartitionRequest(BaseModel):
    """Request model for partition endpoint."""
    path: str
    config: Optional[Dict[str, Any]] = None


@app.get("/api/ast/partition/status")
async def get_partition_status():
    """Get partitioner module status."""
    return {
        "available": PARTITIONER_AVAILABLE,
        "ast_available": AST_AVAILABLE,
        "presets": ["sias_ticc", "generic_two_apps", "generic_three_apps"] if PARTITIONER_AVAILABLE else []
    }


@app.get("/api/ast/partition/presets")
async def get_partition_presets():
    """Get available partition configuration presets."""
    if not PARTITIONER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Partitioner module not available")

    presets = {
        "sias_ticc": {
            "name": "SIAS/TICC (GRDF Pattern)",
            "description": "Two-application separation pattern used in GRDF audits",
            "config": create_sias_ticc_config().to_dict()
        },
        "generic_two_apps": {
            "name": "Generic Two Applications",
            "description": "Template for separating two applications with custom patterns",
            "config": PartitionConfig.default_two_apps("APP_A", "APP_B").to_dict()
        },
        "generic_three_apps": {
            "name": "Generic Three Applications",
            "description": "Template for separating three applications",
            "config": PartitionConfig(
                applications=[
                    ApplicationFingerprint(app_id="APP_A", package_patterns=[], color="#3498db"),
                    ApplicationFingerprint(app_id="APP_B", package_patterns=[], color="#e74c3c"),
                    ApplicationFingerprint(app_id="APP_C", package_patterns=[], color="#2ecc71"),
                ]
            ).to_dict()
        }
    }

    return {"presets": presets}


@app.post("/api/ast/partition")
async def run_partition(request: PartitionRequest):
    """
    Run codebase partitioning analysis.

    Partitions a Java codebase into logical applications using:
    - Fingerprint-based classification (package patterns, class names)
    - Graph propagation (neighbor majority voting)
    - Evidence chains for traceability

    Returns partition assignments, statistics, and visualization data.
    """
    if not PARTITIONER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Partitioner module not available")
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(request.path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

    try:
        # Build dependency graph
        graph_obj = build_dependency_graph([target_path], patterns=None, use_cache=True)

        # Convert to D3 format
        config = VizConfig()
        renderer = D3Renderer(config)
        graph = renderer.to_dict(graph_obj)

        # Build partition config from request
        partition_config = None
        if request.config:
            apps = []
            for app_data in request.config.get("applications", []):
                apps.append(ApplicationFingerprint(
                    app_id=app_data.get("app_id", "APP"),
                    package_patterns=app_data.get("package_patterns", []),
                    class_patterns=app_data.get("class_patterns", []),
                    annotation_patterns=app_data.get("annotation_patterns", []),
                    keyword_patterns=app_data.get("keyword_patterns", []),
                    entry_point_patterns=app_data.get("entry_point_patterns", []),
                    color=app_data.get("color", "#3498db"),
                ))
            partition_config = PartitionConfig(
                applications=apps,
                shared_patterns=request.config.get("shared_patterns", PartitionConfig().shared_patterns),
                dead_code_threshold=request.config.get("dead_code_threshold", 0.0),
                propagation_iterations=request.config.get("propagation_iterations", 5),
                confidence_threshold=request.config.get("confidence_threshold", 0.6),
            )

        # Run partitioning
        result = partition_from_graph(graph, partition_config)

        # Compute additional statistics
        total_classes = len(result.assignments)
        unknown_count = result.summary.get("UNKNOWN", 0)
        coverage = (total_classes - unknown_count) / total_classes if total_classes > 0 else 0

        # Count cross-partition edges
        cross_partition_edges = sum(1 for e in result.edges if e.get("cross_partition", False))
        total_edges = len(result.edges)
        coupling_density = cross_partition_edges / total_edges if total_edges > 0 else 0

        return {
            "summary": result.summary,
            "coverage": round(coverage, 3),
            "coupling_density": round(coupling_density, 4),
            "cross_partition_edges": cross_partition_edges,
            "total_edges": total_edges,
            "total_classes": total_classes,
            "assignments": {k: v.to_dict() for k, v in result.assignments.items()},
            "nodes": result.nodes,
            "edges": result.edges,
            "config": result.config.to_dict() if result.config else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Partition error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ast/partition/export")
async def export_partition(
    path: str,
    format: str = "json",
    partition: Optional[str] = None,
    preset: Optional[str] = None
):
    """
    Export partition results in various formats.

    Args:
        path: Path to the codebase
        format: Export format (json, csv, xlsx)
        partition: Filter by partition label (optional)
        preset: Use a preset configuration (optional)

    Returns:
        Formatted export data or file download
    """
    if not PARTITIONER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Partitioner module not available")
    if not AST_AVAILABLE:
        raise HTTPException(status_code=503, detail="AST analysis not available")

    target_path = Path(path).expanduser()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    try:
        # Build graph
        graph_obj = build_dependency_graph([target_path], patterns=None, use_cache=True)
        config = VizConfig()
        renderer = D3Renderer(config)
        graph = renderer.to_dict(graph_obj)

        # Get partition config from preset
        partition_config = None
        if preset == "sias_ticc":
            partition_config = create_sias_ticc_config()
        elif preset == "generic_two_apps":
            partition_config = PartitionConfig.default_two_apps()

        # Run partitioning
        result = partition_from_graph(graph, partition_config)

        # Filter by partition if specified
        assignments = result.assignments
        if partition:
            assignments = {k: v for k, v in assignments.items() if v.label == partition}

        # Build export data
        if format == "json":
            export_data = {
                "metadata": {
                    "project": str(target_path.name),
                    "partition_date": datetime.now().isoformat(),
                    "total_classes": len(result.assignments),
                    "exported_classes": len(assignments),
                    "filter": partition,
                },
                "summary": result.summary,
                "classes": []
            }

            for fqn, assignment in assignments.items():
                node = next((n for n in result.nodes if n.get("id") == fqn), {})
                export_data["classes"].append({
                    "fqn": fqn,
                    "partition": assignment.label,
                    "confidence": assignment.confidence,
                    "file": node.get("file", ""),
                    "package": node.get("package", ""),
                    "loc": node.get("loc", 0),
                    "evidence": [e.to_dict() for e in assignment.evidence],
                })

            return export_data

        elif format == "csv":
            import io
            import csv

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "fqn", "partition", "confidence", "file", "package", "loc", "evidence"
            ])

            # Data rows
            for fqn, assignment in assignments.items():
                node = next((n for n in result.nodes if n.get("id") == fqn), {})
                evidence_str = "; ".join(e.details for e in assignment.evidence)
                writer.writerow([
                    fqn,
                    assignment.label,
                    round(assignment.confidence, 3),
                    node.get("file", ""),
                    node.get("package", ""),
                    node.get("loc", 0),
                    evidence_str,
                ])

            csv_content = output.getvalue()
            return JSONResponse(
                content={"csv": csv_content, "filename": f"partition_{target_path.name}.csv"},
                headers={"Content-Type": "application/json"}
            )

        elif format == "xlsx":
            # Return JSON data for XLSX generation on client side
            # (Server-side XLSX would require openpyxl)
            sheets = {
                "Summary": [
                    {"partition": k, "count": v} for k, v in result.summary.items()
                ],
                "Classes": [],
                "Cross-Partition": []
            }

            for fqn, assignment in assignments.items():
                node = next((n for n in result.nodes if n.get("id") == fqn), {})
                sheets["Classes"].append({
                    "fqn": fqn,
                    "partition": assignment.label,
                    "confidence": round(assignment.confidence, 3),
                    "file": node.get("file", ""),
                    "package": node.get("package", ""),
                    "loc": node.get("loc", 0),
                })

            # Cross-partition edges
            for edge in result.edges:
                if edge.get("cross_partition"):
                    sheets["Cross-Partition"].append({
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                        "source_partition": edge.get("source_partition", ""),
                        "target_partition": edge.get("target_partition", ""),
                    })

            return {"sheets": sheets, "filename": f"partition_{target_path.name}.xlsx"}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/file/content")
async def get_file_content(path: str, max_lines: int = 2000):
    """
    Get file content for preview.

    Args:
        path: Absolute path to the file
        max_lines: Maximum number of lines to return (default 2000)

    Returns:
        File content and metadata
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        total_lines = len(lines)

        # Truncate if too large
        if total_lines > max_lines:
            lines = lines[:max_lines]
            content = "\n".join(lines)
            truncated = True
        else:
            truncated = False

        return {
            "content": content,
            "filename": file_path.name,
            "path": str(file_path),
            "total_lines": total_lines,
            "returned_lines": len(lines),
            "truncated": truncated,
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
        }
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


# Alias for compatibility with partition preview
@app.get("/api/audit/file-content")
async def get_audit_file_content(path: str, max_lines: int = 2000):
    """Alias for /api/file/content for audit/partition file previews."""
    return await get_file_content(path, max_lines)


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

    Model inheritance hierarchy:
    1. Session model = default (single source of truth)
    2. Agent Config = optional override (inherits from session if not explicitly set)
    3. Reasoning = inherits from Agent Config worker model
    """
    # Get session model (single source of truth)
    session_model = None  # Will be set if session found
    if session_id:
        if session_id in active_sessions:
            session_model = active_sessions[session_id].get("model", "qwen2.5:7b")
        else:
            # Session not found - log available sessions for debugging
            logger.warning(f"Session {session_id} not in active_sessions. Available: {list(active_sessions.keys())}")

    # Fallback to default model if session not found
    if session_model is None:
        try:
            config = get_config()
            session_model = config.llm.model if hasattr(config, 'llm') and config.llm else "qwen2.5:7b"
        except Exception:
            session_model = "qwen2.5:7b"

    # Get default config from ragix.yaml
    try:
        config = get_config()
        default_config = config.agents if hasattr(config, 'agents') and config.agents else None
    except Exception:
        default_config = None

    # Check for session-specific override
    if session_id and session_id in session_agent_configs:
        agent_config = session_agent_configs[session_id]
        is_override = True
    else:
        agent_config = default_config
        is_override = False

    # Detect available models
    available_models = detect_ollama_models()

    # Resolve actual models (inherit from session if not explicitly set)
    if agent_config:
        mode = agent_config.mode.value
        # In minimal mode or if using default, inherit from session
        if mode == "minimal":
            planner = session_model
            worker = session_model
            verifier = session_model
        else:
            planner = agent_config.planner_model
            worker = agent_config.worker_model
            verifier = agent_config.verifier_model
        fallback = agent_config.fallback_model
    else:
        # No config - use session model for all
        mode = "minimal"
        planner = session_model
        worker = session_model
        verifier = session_model
        fallback = session_model

    return {
        "mode": mode,
        "planner_model": planner,
        "worker_model": worker,
        "verifier_model": verifier,
        "fallback_model": fallback,
        "session_model": session_model,  # Expose for UI to show inheritance
        "strict_enforcement": agent_config.strict_enforcement if agent_config else False,
        "is_session_override": is_override,
        "available_models": [
            {
                "name": m.name,
                "size_gb": m.size_gb,
                "parameter_size": m.category,
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
                "description": "All agents use session model (≤8GB VRAM / CPU)",
                "note": "Inherits from session model setting",
            },
            "strict": {
                "description": "Planner uses 7B+, Worker/Verifier use 3B",
                "planner": "mistral:latest",
                "worker": session_model,
                "verifier": session_model,
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

    # Determine fallback model - use worker model if specified (for MINIMAL mode)
    # This ensures MINIMAL mode uses the user's selected model, not the hardcoded default
    fallback = request.single_model or worker or "granite3.1-moe:3b"

    # Create agent config
    agent_config = AgentConfig(
        mode=mode,
        planner_model=planner or fallback,
        worker_model=worker or fallback,
        verifier_model=verifier or fallback,
        strict_enforcement=mode == AgentMode.STRICT,
        fallback_model=fallback,
    )

    # Store as session override
    target_session = session_id or "default"
    session_agent_configs[target_session] = agent_config

    # Invalidate cached agent so it will be recreated with the new config
    if target_session in session_agents:
        del session_agents[target_session]

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
    """Get or create an agent for a session.

    Model resolution hierarchy:
    1. Session model = default/fallback (single source of truth)
    2. Agent Config = optional override (inherits from session if not set)
    3. Reasoning = inherits from Agent Config worker model
    """
    session_id = session["id"]

    # Session model is the single source of truth
    session_model = session.get("model", "qwen2.5:7b")

    # Check for session-specific agent config
    agent_config = session_agent_configs.get(session_id)
    if not agent_config:
        # Try to get default from global config
        try:
            config = get_config()
            agent_config = config.agents if hasattr(config, 'agents') and config.agents else None
        except Exception:
            agent_config = None

    # Determine model to use:
    # - If agent_config exists and has explicit model set, use it
    # - Otherwise, use session model (inheritance)
    if agent_config:
        # Check if agent config has explicit override (not default)
        configured_model = agent_config.get_model(AgentRole.WORKER)
        # Use session model if agent config still has hardcoded default
        if configured_model == "granite3.1-moe:3b" and session_model != "granite3.1-moe:3b":
            model_to_use = session_model
        else:
            model_to_use = configured_model
    else:
        model_to_use = session_model

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

        # Store agent config reference and session model for later use
        agent._agent_config = agent_config
        agent._session_model = session_model  # Original session model (source of truth)
        agent._resolved_model = model_to_use  # Actually used model

        session_agents[session_id] = agent
        logger.debug(f"Created agent for session={session_id}, model={model_to_use}")

        # Determine mode for logging
        agent_mode = agent_config.mode.value if agent_config else "minimal"

        # Log reasoning trace for agent creation
        if session_id not in session_reasoning_traces:
            session_reasoning_traces[session_id] = []
        session_reasoning_traces[session_id].append({
            "type": "system",
            "timestamp": datetime.now().isoformat(),
            "content": f"Agent initialized with model: {model_to_use} (session: {session_model}), mode: {agent_mode}",
        })

    return session_agents[session_id]


def _build_timeout_summary(accumulated_traces: List[Dict]) -> str:
    """Build a useful summary from accumulated traces on timeout/interruption."""
    if not accumulated_traces:
        return ""

    parts = ["\n\n**What was accomplished:**\n"]

    # Extract successful steps with their results
    successful_steps = []
    failed_steps = []

    for trace in accumulated_traces:
        trace_type = trace.get("type", "")
        metadata = trace.get("metadata", {})
        content = trace.get("content", "")

        if trace_type == "step_complete":
            step_num = metadata.get("step_num", "?")
            status = metadata.get("status", "")

            if status == "success":
                result = metadata.get("result", "")[:200]
                successful_steps.append((step_num, result))
            elif status == "failed":
                error = metadata.get("error", "Unknown error")[:100]
                failed_steps.append((step_num, error))

    # Show successful steps (filter out LLM explanations)
    if successful_steps:
        real_results = []
        llm_chatter = []
        for step_num, result in successful_steps:
            # Detect LLM explanation vs actual output
            is_chatter = result and any(kw in result.lower() for kw in [
                "the objective", "let's", "we need to", "here are the steps",
                "to do this", "assuming", "we will use", "let me"
            ])
            if is_chatter:
                llm_chatter.append(step_num)
            elif result:
                real_results.append((step_num, result))
            else:
                real_results.append((step_num, "(no output)"))

        if real_results:
            parts.append("✅ **Completed steps with output:**\n")
            for step_num, result in real_results:
                parts.append(f"- Step {step_num}: {result[:100]}{'...' if len(result) > 100 else ''}\n")

        if llm_chatter:
            parts.append(f"\n⚠️ **Steps with LLM explanation only (no execution):** {', '.join(map(str, llm_chatter))}\n")

    # Show failed steps with reasons
    if failed_steps:
        parts.append("\n❌ **Failed steps:**\n")
        for step_num, error in failed_steps:
            parts.append(f"- Step {step_num}: {error}\n")

    # If no step info, fall back to last few traces
    if not successful_steps and not failed_steps:
        parts = ["\n\n**Progress before timeout:**\n"]
        for trace in accumulated_traces[-5:]:
            trace_type = trace.get("type", "unknown")
            content = trace.get("content", "")[:100]
            trace_elapsed = trace.get("elapsed", 0)
            parts.append(f"- [{trace_elapsed:.1f}s] {trace_type}: {content}\n")

    return "".join(parts)


async def run_agent_async(
    agent: UnixRAGAgent,
    message: str,
    session_id: str = "default",
    timeout_seconds: int = 180,
    progress_callback: Optional[Callable[[Dict], Awaitable[None]]] = None,
    cancel_event: Optional[threading.Event] = None
) -> Tuple[str, List[Dict]]:
    """
    Run agent in a thread pool to avoid blocking.

    Uses step_with_reasoning for complex tasks (Planner/Worker/Verifier loop).

    Args:
        agent: UnixRAGAgent instance
        message: User message
        session_id: Session identifier
        timeout_seconds: Maximum time to wait for response (default: 180s)
        progress_callback: Optional async callback for streaming progress updates
        cancel_event: Optional threading.Event to signal cancellation

    Returns:
        Tuple of (response_text, reasoning_traces)
    """
    import concurrent.futures
    import time
    import queue

    start_time = time.time()
    progress_queue: queue.Queue = queue.Queue()
    accumulated_traces: List[Dict] = []
    was_cancelled = False

    def emit_progress(event_type: str, content: str, metadata: Optional[Dict] = None):
        """Emit a progress event to the queue."""
        event = {
            "type": event_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - start_time,
            **(metadata or {})
        }
        progress_queue.put(event)
        accumulated_traces.append(event)

    def execute_step():
        """Execute agent step with reasoning and format response."""
        # Set up real-time progress callback on the reasoning loop
        if hasattr(agent, '_reasoning_loop') and agent._reasoning_loop:
            # Create a callback that emits traces to the progress queue in real-time
            def trace_callback(trace: Dict):
                """Real-time callback for reasoning traces."""
                emit_progress(
                    trace.get("type", "trace"),
                    trace.get("content", ""),
                    trace.get("metadata", {})
                )

            # Set the callback on the reasoning loop
            if hasattr(agent._reasoning_loop, 'set_progress_callback'):
                agent._reasoning_loop.set_progress_callback(trace_callback)

            # Note: Classification progress is emitted by the reasoning graph itself
            # Don't emit here to avoid duplicates

        # Use reasoning-enabled step
        # Note: traces are emitted in real-time via trace_callback (set above)
        # so we don't emit them again here to avoid duplicate progress cards
        cmd_result, response, traces = agent.step_with_reasoning(message)

        # Build response string
        parts = []

        # Check if response is a CommandResult repr (shouldn't be displayed raw)
        if response and not (response.startswith('CommandResult(') and response.endswith(')')):
            # Extract message from JSON if response is a raw action object
            clean_response = _extract_message_from_response(response)
            parts.append(clean_response)

        if cmd_result:
            # Format command output cleanly
            output = cmd_result.stdout.strip() if cmd_result.stdout else ""
            stderr = cmd_result.stderr.strip() if cmd_result.stderr else ""

            # Filter out binary file warnings from stderr
            if stderr:
                stderr_lines = [l for l in stderr.split('\n')
                               if not l.startswith('grep:') or 'binary file' not in l.lower()]
                stderr = '\n'.join(stderr_lines)

            # Truncate very long outputs
            max_lines = 50
            output_lines = output.split('\n')
            if len(output_lines) > max_lines:
                output = '\n'.join(output_lines[:max_lines])
                output += f"\n\n... ({len(output_lines) - max_lines} more lines truncated)"

            # Build formatted output
            cmd_output = f"\n**Command:** `{cmd_result.command}`\n"
            if output:
                cmd_output += f"```\n{output}\n```"
            if stderr and cmd_result.returncode != 0:
                cmd_output += f"\n**Errors:**\n```\n{stderr}\n```"

            parts.append(cmd_output)

        result_text = "\n".join(parts) if parts else "No response from agent."
        emit_progress("complete", "Execution completed", {"success": True})
        return result_text, accumulated_traces

    # Process progress updates in background
    async def process_progress():
        """Process and send progress updates."""
        while True:
            try:
                # Check for progress events (non-blocking)
                event = progress_queue.get_nowait()
                if progress_callback:
                    await progress_callback(event)
            except queue.Empty:
                await asyncio.sleep(0.1)  # Brief pause before checking again

    loop = asyncio.get_event_loop()
    result_text = ""
    traces = []

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Start the execution
            future = loop.run_in_executor(pool, execute_step)

            # Poll for progress while waiting for completion
            while not future.done():
                # Process any pending progress events
                try:
                    while True:
                        event = progress_queue.get_nowait()
                        if progress_callback:
                            await progress_callback(event)
                except queue.Empty:
                    pass

                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    was_cancelled = True
                    emit_progress("cancelled", "⛔ Reasoning interrupted by user", {"cancelled": True})
                    # Cancel the future if possible
                    future.cancel()
                    raise asyncio.CancelledError("User cancelled the request")

                # Check if we've exceeded timeout
                if time.time() - start_time > timeout_seconds:
                    # Try to get partial results
                    partial_traces = list(accumulated_traces)
                    raise asyncio.TimeoutError()

                await asyncio.sleep(0.2)  # Brief pause before checking again

            # Get final result
            result_text, traces = future.result()

            # Process any remaining progress events
            try:
                while True:
                    event = progress_queue.get_nowait()
                    if progress_callback:
                        await progress_callback(event)
            except queue.Empty:
                pass

    except asyncio.CancelledError:
        elapsed = time.time() - start_time

        # Build detailed partial results from accumulated traces
        partial_info = _build_timeout_summary(accumulated_traces)

        result_text = f"⛔ **Reasoning interrupted** after {elapsed:.1f}s\n\nThe request was cancelled by user.{partial_info}"
        traces = accumulated_traces + [{"type": "cancelled", "content": f"Request cancelled after {elapsed:.1f}s", "timestamp": datetime.now().isoformat()}]

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time

        # Build detailed partial results from accumulated traces
        partial_info = _build_timeout_summary(accumulated_traces)

        result_text = f"⏱️ **Request timed out** after {elapsed:.1f}s\n\nThe LLM took too long to respond. This can happen with complex queries. Try:\n- Simplifying your question\n- Breaking it into smaller steps\n- Checking if Ollama is responsive (`ollama list`){partial_info}"
        traces = accumulated_traces + [{"type": "timeout", "content": f"Request timed out after {elapsed:.1f}s", "timestamp": datetime.now().isoformat()}]

    # Calculate and append execution time
    elapsed = time.time() - start_time
    result_text = f"{result_text}\n\n---\n⏱️ *Execution time: {elapsed:.2f}s*"

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

        # Get reasoning strategy (check session-specific first, then env)
        from ragix_web.routers.reasoning import _session_reasoning_config
        if session_id in _session_reasoning_config:
            reasoning_strategy = _session_reasoning_config[session_id].get("strategy", "graph_v30")
        else:
            reasoning_strategy = os.environ.get("RAGIX_REASONING_STRATEGY", "graph_v30")

        await websocket.send_json({
            "type": "status",
            "message": f"Connected to session {session_id} (model: {actual_model}, mode: {agent_mode}, reasoning: {reasoning_strategy})"
        })

        # Create cancellation event for this session
        if session_id not in session_cancellation:
            session_cancellation[session_id] = threading.Event()

        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            # Handle cancel request
            if message_type == "cancel":
                if session_id in session_cancellation:
                    session_cancellation[session_id].set()
                    await websocket.send_json({
                        "type": "cancel_ack",
                        "message": "Cancellation requested...",
                        "timestamp": datetime.now().isoformat()
                    })
                continue

            if message_type == "chat":
                user_message = data.get("message", "")

                # Re-fetch agent in case model changed (agent may have been invalidated)
                agent = get_or_create_agent(session)

                # Reset cancellation event for new request
                if session_id in session_cancellation:
                    session_cancellation[session_id].clear()

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

                # Send thinking indicator with cancel hint
                await websocket.send_json({
                    "type": "thinking",
                    "message": "Agent is processing...",
                    "cancellable": True,
                    "timestamp": datetime.now().isoformat()
                })

                # Progress callback for streaming updates
                async def send_progress(event: Dict):
                    """Send progress update to WebSocket."""
                    try:
                        await websocket.send_json({
                            "type": "progress",
                            "event": event,
                            "timestamp": datetime.now().isoformat()
                        })
                        # Also update reasoning traces in real-time
                        await websocket.send_json({
                            "type": "reasoning_trace_update",
                            "trace": event,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        pass  # Ignore errors sending progress

                try:
                    # Get cancellation event for this session
                    cancel_event = session_cancellation.get(session_id)

                    # v0.33: Retrieve RAG context from both Chat RAG and Project RAG
                    # Chat RAG: session-scoped, BM25-based (.ragix/)
                    chat_rag_context = retrieve_rag_context(session_id, user_message, top_k=5, max_chars_per_chunk=1500)

                    # Project RAG: project-wide, ChromaDB-based (.RAG/)
                    # Only retrieve if enabled for this session (default: enabled)
                    project_rag_context = None
                    if ROUTERS_AVAILABLE and is_project_rag_enabled(session_id):
                        try:
                            project_rag_context = retrieve_project_rag_context(
                                user_message, top_k=5, max_chars_per_chunk=1500
                            )
                        except Exception as e:
                            logger.debug(f"Project RAG retrieval failed: {e}")

                    # Merge contexts (Project RAG first, then Chat RAG)
                    combined_context_parts = []
                    context_sources = []

                    if project_rag_context:
                        combined_context_parts.append(project_rag_context)
                        # Count chunks from Project RAG
                        proj_chunks = len(project_rag_context.split('### [')) - 1
                        context_sources.append(f"{proj_chunks} from Project RAG")

                    if chat_rag_context:
                        combined_context_parts.append(chat_rag_context)
                        # Count chunks from Chat RAG
                        chat_chunks = len(chat_rag_context.split('### Document')) - 1
                        context_sources.append(f"{chat_chunks} from Chat RAG")

                    if combined_context_parts:
                        combined_context = "\n\n---\n\n".join(combined_context_parts)
                        # Get current project path for context
                        current_project = get_current_project() if ROUTERS_AVAILABLE else None
                        project_info = f"\n**Current Project:** {current_project}\n" if current_project else ""
                        # Prepend RAG context to user message with clear instructions
                        augmented_message = (
                            f"{combined_context}\n\n"
                            f"---\n\n"
                            f"## User Question{project_info}"
                            f"{user_message}\n\n"
                            f"**Instructions:** Answer the question using ONLY the context provided above. "
                            f"Cite sources when referencing specific files. "
                            f"Do NOT suggest files outside the current project context. "
                            f"Do not use shell commands to search for files unless needed for additional details."
                        )
                        await websocket.send_json({
                            "type": "rag_context",
                            "message": f"📚 Retrieved context: {', '.join(context_sources)}",
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        augmented_message = user_message

                    # Run the agent with reasoning (Planner/Worker/Verifier)
                    response, traces = await run_agent_async(
                        agent, augmented_message, session_id,
                        progress_callback=send_progress,
                        cancel_event=cancel_event
                    )

                    # Send reasoning traces if any (for Reasoning tab visualization)
                    if traces:
                        await websocket.send_json({
                            "type": "reasoning_traces",
                            "traces": traces,
                            "timestamp": datetime.now().isoformat()
                        })

                    # v0.23: Send reasoning graph state if available
                    if session_id in session_reasoning_states:
                        graph_state = session_reasoning_states[session_id]
                        state = graph_state.get("state")
                        if state and hasattr(state, 'to_dict'):
                            state_msg = {
                                "type": "reasoning_graph_state",
                                "state": state.to_dict(),
                                "current_node": graph_state.get("current_node"),
                                "timestamp": datetime.now().isoformat()
                            }
                            # v0.32: Include goal for memory context
                            if hasattr(state, 'goal'):
                                state_msg["goal"] = state.goal
                            await websocket.send_json(state_msg)

                    # Get token statistics if available
                    token_stats = None
                    if hasattr(agent, 'get_token_stats'):
                        token_stats = agent.get_token_stats()

                    # Send agent response with token stats
                    await websocket.send_json({
                        "type": "agent_message",
                        "message": response,
                        "token_stats": token_stats,
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
        default="qwen2.5:7b",
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
