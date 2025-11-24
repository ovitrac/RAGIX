"""
FastAPI Server for RAGIX Web UI

Provides local-only web interface with WebSocket chat and REST API.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("Error: FastAPI dependencies not installed.", file=sys.stderr)
    print("Install with: pip install 'ragix[web]'", file=sys.stderr)
    sys.exit(1)

from ragix_core import OllamaLLM, ShellSandbox, AgentLogger, LogLevel
from ragix_unix import UnixRAGAgent


# FastAPI app
app = FastAPI(
    title="RAGIX Web UI",
    description="Local-first Unix-RAG development assistant",
    version="0.6.0-dev"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local only, safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper session management)
active_sessions: Dict[str, Dict[str, Any]] = {}
active_websockets: List[WebSocket] = []


# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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
        "version": "0.6.0-dev",
        "sessions": len(active_sessions)
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
async def create_session(
    sandbox_root: str = "~/ragix-workspace",
    model: str = "mistral",
    profile: str = "dev"
):
    """Create a new session."""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Expand and validate sandbox root
    sandbox_root = os.path.expanduser(sandbox_root)
    sandbox_root = os.path.abspath(sandbox_root)

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

    # Initialize agent for this session (simplified, in production use proper instantiation)
    try:
        await websocket.send_json({
            "type": "status",
            "message": f"Connected to session {session_id}"
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

                # Simulate agent response (in production, integrate with actual agent)
                # For now, send a mock response
                await websocket.send_json({
                    "type": "agent_message",
                    "message": f"Received: {user_message}\n\nThis is a placeholder response. Full agent integration coming soon.",
                    "timestamp": datetime.now().isoformat()
                })

                session["message_history"].append({
                    "role": "assistant",
                    "content": f"Mock response to: {user_message}",
                    "timestamp": datetime.now().isoformat()
                })

            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        await websocket.close()
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
        default="~/ragix-workspace",
        help="Default sandbox root (default: ~/ragix-workspace)"
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
