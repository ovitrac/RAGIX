#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOAS MCP Demo Server
====================

FastAPI server demonstrating KOAS tools with Ollama LLM integration.
Shows tool call tracing, model selection, and interactive security/audit workflows.

Features:
- Model selection (Ollama backend)
- Real-time tool call visualization
- Security and audit scenario browser
- Dry-run mode for safe testing
- Auto-workspace management

Usage:
    python server.py                    # Start on port 8080
    python server.py --port 8888        # Custom port
    python server.py --host 0.0.0.0     # Allow remote access

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-19
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MCP"))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Error: FastAPI dependencies not installed.", file=sys.stderr)
    print("Install with: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToolCallRequest(BaseModel):
    """Request to execute a KOAS tool."""
    tool_name: str
    parameters: Dict[str, Any]
    dry_run: bool = False


class ChatRequest(BaseModel):
    """Chat request with optional model selection."""
    message: str
    model: str = "mistral:7b-instruct"
    dry_run: bool = False
    workspace: Optional[str] = None


class ScenarioRequest(BaseModel):
    """Execute a predefined scenario."""
    scenario_id: str
    dry_run: bool = True


class MemoryEntry(BaseModel):
    """Single memory entry."""
    role: str  # "user", "assistant", "tool"
    content: str
    timestamp: str
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class MemorySaveRequest(BaseModel):
    """Request to save memory to file."""
    filename: str


# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

class ConversationMemory:
    """Manages conversation context for the chat interface."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.workspace: Optional[str] = None
        self.created_at: str = datetime.now().isoformat()

    def add_user_message(self, message: str):
        """Add a user message to memory."""
        self.entries.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

    def add_assistant_response(self, content: str):
        """Add an assistant response to memory."""
        self.entries.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def add_tool_result(self, tool_name: str, params: Dict[str, Any], result: Dict[str, Any]):
        """Add a tool execution result to memory."""
        self.entries.append({
            "role": "tool",
            "tool_name": tool_name,
            "parameters": params,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        # Track workspace
        if "workspace" in result:
            self.workspace = result["workspace"]

    def get_context(self, max_entries: int = 10) -> str:
        """Get recent conversation context as a formatted string."""
        recent = self.entries[-max_entries:] if len(self.entries) > max_entries else self.entries

        context_parts = []
        for entry in recent:
            if entry["role"] == "user":
                context_parts.append(f"User: {entry['content']}")
            elif entry["role"] == "assistant":
                context_parts.append(f"Assistant: {entry['content'][:500]}...")
            elif entry["role"] == "tool":
                summary = entry.get("result", {}).get("summary", "Completed")
                context_parts.append(f"[Tool: {entry['tool_name']}] {summary}")

        return "\n".join(context_parts)

    def clear(self):
        """Clear all memory entries."""
        self.entries = []
        self.workspace = None
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Export memory to dictionary."""
        return {
            "entries": self.entries,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "entry_count": len(self.entries),
        }

    def save(self, filepath: Path):
        """Save memory to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, filepath: Path):
        """Load memory from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.entries = data.get("entries", [])
            self.workspace = data.get("workspace")
            self.created_at = data.get("created_at", datetime.now().isoformat())


# Global conversation memory
conversation_memory = ConversationMemory()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="KOAS MCP Demo",
    description="Interactive demo for KOAS security and audit tools",
    version="0.62.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =============================================================================
# KOAS TOOL REGISTRY
# =============================================================================

KOAS_TOOLS = {
    # Security tools
    "koas_security_discover": {
        "category": "security",
        "stage": 1,
        "description": "Discover hosts on a network segment",
        "parameters": {
            "target": {"type": "string", "required": True, "example": "192.168.1.0/24"},
            "method": {"type": "string", "default": "ping", "options": ["ping", "arp", "list"]},
            "timeout": {"type": "integer", "default": 120},
            "workspace": {"type": "string", "default": ""},
        },
    },
    "koas_security_scan_ports": {
        "category": "security",
        "stage": 1,
        "description": "Scan ports on target hosts",
        "parameters": {
            "target": {"type": "string", "default": "discovered"},
            "ports": {"type": "string", "default": "common", "options": ["common", "web", "database", "admin", "top100", "full"]},
            "detect_services": {"type": "boolean", "default": True},
            "workspace": {"type": "string", "required": True},
        },
    },
    "koas_security_ssl_check": {
        "category": "security",
        "stage": 2,
        "description": "Analyze SSL/TLS configuration",
        "parameters": {
            "target": {"type": "string", "default": "discovered"},
            "check_ciphers": {"type": "boolean", "default": True},
            "check_vulnerabilities": {"type": "boolean", "default": True},
            "workspace": {"type": "string", "required": True},
        },
    },
    "koas_security_vuln_scan": {
        "category": "security",
        "stage": 2,
        "description": "Scan for known vulnerabilities",
        "parameters": {
            "target": {"type": "string", "default": "discovered"},
            "severity": {"type": "string", "default": "medium", "options": ["info", "low", "medium", "high", "critical"]},
            "templates": {"type": "string", "default": "default"},
            "workspace": {"type": "string", "required": True},
        },
    },
    "koas_security_dns_check": {
        "category": "security",
        "stage": 1,
        "description": "Analyze DNS configuration",
        "parameters": {
            "domain": {"type": "string", "required": True, "example": "example.com"},
            "check_security": {"type": "boolean", "default": True},
            "workspace": {"type": "string", "default": ""},
        },
    },
    "koas_security_compliance": {
        "category": "security",
        "stage": 2,
        "description": "Check compliance against frameworks",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "framework": {"type": "string", "default": "anssi", "options": ["anssi", "nist", "cis"]},
            "level": {"type": "string", "default": "standard"},
        },
    },
    "koas_security_risk": {
        "category": "security",
        "stage": 2,
        "description": "Calculate network risk scores",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "top_hosts": {"type": "integer", "default": 5},
        },
    },
    "koas_security_report": {
        "category": "security",
        "stage": 3,
        "description": "Generate security report",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "format": {"type": "string", "default": "summary", "options": ["summary", "detailed", "executive"]},
            "language": {"type": "string", "default": "en", "options": ["en", "fr"]},
        },
    },
    # Audit tools
    "koas_audit_scan": {
        "category": "audit",
        "stage": 1,
        "description": "Scan codebase and extract AST symbols",
        "parameters": {
            "project_path": {"type": "string", "required": True},
            "language": {"type": "string", "default": "auto", "options": ["auto", "python", "java", "typescript"]},
            "include_tests": {"type": "boolean", "default": False},
            "workspace": {"type": "string", "default": ""},
        },
    },
    "koas_audit_metrics": {
        "category": "audit",
        "stage": 1,
        "description": "Compute code metrics",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "threshold_cc": {"type": "integer", "default": 10},
            "threshold_loc": {"type": "integer", "default": 300},
        },
    },
    "koas_audit_hotspots": {
        "category": "audit",
        "stage": 2,
        "description": "Identify complexity hotspots",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "top_n": {"type": "integer", "default": 20},
        },
    },
    "koas_audit_dependencies": {
        "category": "audit",
        "stage": 1,
        "description": "Analyze code dependencies",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "detect_cycles": {"type": "boolean", "default": True},
        },
    },
    "koas_audit_dead_code": {
        "category": "audit",
        "stage": 2,
        "description": "Detect dead/unused code",
        "parameters": {
            "workspace": {"type": "string", "required": True},
        },
    },
    "koas_audit_risk": {
        "category": "audit",
        "stage": 2,
        "description": "Calculate code risk scores",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "include_volumetry": {"type": "boolean", "default": False},
        },
    },
    "koas_audit_compliance": {
        "category": "audit",
        "stage": 2,
        "description": "Check code quality compliance",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "standard": {"type": "string", "default": "maintainability", "options": ["maintainability", "testability", "documentation"]},
        },
    },
    "koas_audit_report": {
        "category": "audit",
        "stage": 3,
        "description": "Generate audit report",
        "parameters": {
            "workspace": {"type": "string", "required": True},
            "format": {"type": "string", "default": "executive", "options": ["executive", "detailed", "technical"]},
            "language": {"type": "string", "default": "en", "options": ["en", "fr"]},
        },
    },
}

# =============================================================================
# SCENARIOS
# =============================================================================

SCENARIOS = {
    "security_quick": {
        "name": "Quick Security Scan",
        "description": "Fast network discovery and port scan",
        "category": "security",
        "steps": [
            {"tool": "koas_security_discover", "params": {"target": "192.168.1.0/24", "method": "ping"}},
            {"tool": "koas_security_scan_ports", "params": {"target": "discovered", "ports": "web"}},
        ],
    },
    "security_full": {
        "name": "Full Security Assessment",
        "description": "Complete security audit with compliance check",
        "category": "security",
        "steps": [
            {"tool": "koas_security_discover", "params": {"target": "192.168.1.0/24"}},
            {"tool": "koas_security_scan_ports", "params": {"target": "discovered", "ports": "common"}},
            {"tool": "koas_security_ssl_check", "params": {"target": "discovered"}},
            {"tool": "koas_security_vuln_scan", "params": {"target": "discovered", "severity": "medium"}},
            {"tool": "koas_security_compliance", "params": {"framework": "anssi"}},
            {"tool": "koas_security_risk", "params": {}},
            {"tool": "koas_security_report", "params": {"format": "executive"}},
        ],
    },
    "audit_quick": {
        "name": "Quick Code Audit",
        "description": "Fast codebase scan and metrics",
        "category": "audit",
        "steps": [
            {"tool": "koas_audit_scan", "params": {"project_path": ".", "language": "auto"}},
            {"tool": "koas_audit_metrics", "params": {}},
        ],
    },
    "audit_full": {
        "name": "Full Code Audit",
        "description": "Complete code quality assessment",
        "category": "audit",
        "steps": [
            {"tool": "koas_audit_scan", "params": {"project_path": ".", "language": "auto"}},
            {"tool": "koas_audit_metrics", "params": {}},
            {"tool": "koas_audit_dependencies", "params": {"detect_cycles": True}},
            {"tool": "koas_audit_hotspots", "params": {"top_n": 20}},
            {"tool": "koas_audit_dead_code", "params": {}},
            {"tool": "koas_audit_risk", "params": {}},
            {"tool": "koas_audit_compliance", "params": {"standard": "maintainability"}},
            {"tool": "koas_audit_report", "params": {"format": "executive"}},
        ],
    },
}

# =============================================================================
# TOOL EXECUTION
# =============================================================================

async def execute_tool(tool_name: str, parameters: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Execute a KOAS tool and return results."""

    start_time = time.time()

    if dry_run:
        # Simulate tool execution
        await asyncio.sleep(0.5)  # Simulate processing
        return {
            "status": "dry_run",
            "tool": tool_name,
            "parameters": parameters,
            "summary": f"[DRY RUN] Would execute {tool_name} with params: {json.dumps(parameters)[:100]}...",
            "duration_seconds": round(time.time() - start_time, 2),
        }

    try:
        # Import the MCP server module from the correct path
        import importlib.util
        mcp_server_path = PROJECT_ROOT / "MCP" / "ragix_mcp_server.py"

        spec = importlib.util.spec_from_file_location("ragix_mcp_server", mcp_server_path)
        if spec is None or spec.loader is None:
            return {
                "status": "error",
                "error": f"Could not load MCP server from {mcp_server_path}",
                "summary": "MCP server module not found.",
            }

        ragix_mcp = importlib.util.module_from_spec(spec)
        sys.modules['ragix_mcp_server'] = ragix_mcp
        spec.loader.exec_module(ragix_mcp)

        # Get the tool function
        tool_func = getattr(ragix_mcp, tool_name, None)

        if tool_func is None:
            return {
                "status": "error",
                "error": f"Tool not found: {tool_name}",
                "summary": f"Tool {tool_name} is not available in MCP server.",
            }

        # Execute the tool (run in thread pool since it may be blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: tool_func(**parameters))

        if isinstance(result, dict):
            result["duration_seconds"] = round(time.time() - start_time, 2)
        else:
            result = {
                "status": "completed",
                "result": result,
                "duration_seconds": round(time.time() - start_time, 2),
            }
        return result

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": tool_name,
            "summary": f"Error executing {tool_name}: {str(e)[:100]}",
            "duration_seconds": round(time.time() - start_time, 2),
        }


# =============================================================================
# OLLAMA INTEGRATION
# =============================================================================

async def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        models = []
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        for line in lines:
            parts = line.split()
            if parts:
                models.append({
                    "name": parts[0],
                    "size": parts[1] if len(parts) > 1 else "unknown",
                    "modified": parts[2] if len(parts) > 2 else "unknown",
                })
        return models
    except Exception:
        return []


async def chat_with_llm(message: str, model: str = "mistral:7b-instruct") -> str:
    """Send a message to the LLM and get a response."""
    try:
        # Simple Ollama API call
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": message,
                    "stream": False,
                },
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"


def _format_tool_name(tool_name: str) -> str:
    """Format tool name for display."""
    # Remove prefix and format nicely
    name = tool_name.replace("koas_security_", "").replace("koas_audit_", "")
    name = name.replace("_", " ").title()

    # Add emoji based on tool type
    emoji_map = {
        "discover": "🔍",
        "scan": "📡",
        "ports": "🔌",
        "ssl": "🔒",
        "vuln": "⚠️",
        "dns": "🌐",
        "compliance": "✅",
        "risk": "📊",
        "report": "📋",
        "metrics": "📈",
        "hotspots": "🔥",
        "dependencies": "🔗",
        "dead": "💀",
    }

    for key, emoji in emoji_map.items():
        if key in tool_name.lower():
            return f"{emoji} {name}"

    return f"🔧 {name}"


def extract_tool_calls(llm_response: str) -> List[Dict[str, Any]]:
    """Extract tool calls from LLM response.

    Looks for JSON blocks with tool_call format:
    {"tool_call": {"name": "tool_name", "parameters": {...}}}
    """
    import re

    tool_calls = []

    # Look for JSON blocks in the response
    json_pattern = r'\{[^{}]*"tool_call"[^{}]*\{[^{}]*\}[^{}]*\}'
    matches = re.findall(json_pattern, llm_response, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if "tool_call" in data:
                tool_calls.append(data["tool_call"])
        except json.JSONDecodeError:
            pass

    # Also look for simpler format: {"name": "...", "parameters": {...}}
    simple_pattern = r'\{\s*"name"\s*:\s*"(koas_\w+)"[^}]*"parameters"\s*:\s*(\{[^}]*\})'
    simple_matches = re.findall(simple_pattern, llm_response, re.DOTALL)

    for name, params_str in simple_matches:
        try:
            params = json.loads(params_str)
            tool_calls.append({"name": name, "parameters": params})
        except json.JSONDecodeError:
            pass

    return tool_calls


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main demo page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>KOAS MCP Demo</title></head>
    <body>
        <h1>KOAS MCP Demo</h1>
        <p>Static files not found. Please check the installation.</p>
        <p><a href="/api/tools">View available tools</a></p>
        <p><a href="/api/models">View available models</a></p>
    </body>
    </html>
    """)


@app.get("/api/tools")
async def list_tools():
    """List all available KOAS tools."""
    return {
        "tools": KOAS_TOOLS,
        "count": len(KOAS_TOOLS),
        "categories": {
            "security": [k for k, v in KOAS_TOOLS.items() if v["category"] == "security"],
            "audit": [k for k, v in KOAS_TOOLS.items() if v["category"] == "audit"],
        },
    }


@app.get("/api/scenarios")
async def list_scenarios():
    """List all available scenarios."""
    return {
        "scenarios": SCENARIOS,
        "count": len(SCENARIOS),
    }


@app.get("/api/models")
async def list_models():
    """List available Ollama models."""
    models = await get_available_models()
    return {
        "models": models,
        "count": len(models),
        "recommended": ["mistral:7b-instruct", "llama3.1:70b", "qwen2.5:72b"],
    }


@app.post("/api/tool")
async def execute_tool_endpoint(request: ToolCallRequest):
    """Execute a single KOAS tool."""
    if request.tool_name not in KOAS_TOOLS:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {request.tool_name}")

    result = await execute_tool(request.tool_name, request.parameters, request.dry_run)
    return result


@app.post("/api/scenario")
async def execute_scenario(request: ScenarioRequest):
    """Execute a predefined scenario."""
    if request.scenario_id not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {request.scenario_id}")

    scenario = SCENARIOS[request.scenario_id]
    results = []
    workspace = None

    for step in scenario["steps"]:
        params = step["params"].copy()

        # Use workspace from previous step if available
        if workspace and "workspace" in KOAS_TOOLS[step["tool"]]["parameters"]:
            if not params.get("workspace"):
                params["workspace"] = workspace

        result = await execute_tool(step["tool"], params, request.dry_run)
        results.append({
            "step": len(results) + 1,
            "tool": step["tool"],
            "params": params,
            "result": result,
        })

        # Capture workspace for subsequent steps
        if "workspace" in result and result["workspace"]:
            workspace = result["workspace"]

    return {
        "scenario": request.scenario_id,
        "name": scenario["name"],
        "dry_run": request.dry_run,
        "steps": results,
        "workspace": workspace,
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with LLM about security/audit tasks with automatic tool execution."""

    # Add user message to memory
    conversation_memory.add_user_message(request.message)

    # Use workspace from memory if not provided
    workspace = request.workspace or conversation_memory.workspace

    # Build detailed tool context with parameters
    tool_details = []
    for name, info in KOAS_TOOLS.items():
        params_str = ", ".join([
            f"{p}={info['parameters'][p].get('default', 'required')}"
            for p in info['parameters']
        ])
        tool_details.append(f"- {name}({params_str}): {info['description']}")

    tool_context = "\n".join(tool_details)

    # Get conversation context from memory
    memory_context = conversation_memory.get_context(max_entries=6)

    system_prompt = f"""You are a security and code audit assistant. You have access to KOAS tools that you MUST use to help users.

IMPORTANT: When the user asks you to perform a task, you MUST output a tool call in this EXACT JSON format:
{{"tool_call": {{"name": "tool_name", "parameters": {{"param1": "value1"}}}}}}

Available tools:
{tool_context}

EXAMPLES:
- User: "scan my network" → {{"tool_call": {{"name": "koas_security_discover", "parameters": {{"target": "192.168.1.0/24", "method": "ping"}}}}}}
- User: "check ports on localhost" → {{"tool_call": {{"name": "koas_security_scan_ports", "parameters": {{"target": "127.0.0.1", "ports": "common"}}}}}}
- User: "audit this code" → {{"tool_call": {{"name": "koas_audit_scan", "parameters": {{"project_path": ".", "language": "auto"}}}}}}

Current workspace: {workspace or 'auto-created'}
Dry run mode: {request.dry_run}

CONVERSATION HISTORY:
{memory_context if memory_context else 'No previous conversation.'}

When the user asks to scan, audit, check, or analyze something, OUTPUT THE JSON TOOL CALL. Do not just explain - EXECUTE!
Use the workspace from previous tool results when chaining operations."""

    full_prompt = f"{system_prompt}\n\nUser: {request.message}\n\nAssistant:"

    # Get LLM response
    llm_response = await chat_with_llm(full_prompt, request.model)

    # Extract and execute any tool calls
    tool_calls = extract_tool_calls(llm_response)
    tool_results = []

    workspace = request.workspace

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        params = tool_call.get("parameters", {})

        if tool_name in KOAS_TOOLS:
            # Use existing workspace if available
            if workspace and "workspace" in KOAS_TOOLS[tool_name]["parameters"]:
                if not params.get("workspace"):
                    params["workspace"] = workspace

            # Execute the tool
            result = await execute_tool(tool_name, params, request.dry_run)
            tool_results.append({
                "tool": tool_name,
                "parameters": params,
                "result": result,
            })

            # Add tool result to memory
            conversation_memory.add_tool_result(tool_name, params, result)

            # Update workspace for chaining
            if "workspace" in result:
                workspace = result["workspace"]

    # If tools were executed, format a nice response with results
    if tool_results:
        results_parts = []
        for tr in tool_results:
            result = tr["result"]
            tool_name = tr["tool"]
            summary = result.get("summary", "Completed")

            # Format header
            results_parts.append(f"### {_format_tool_name(tool_name)}")
            results_parts.append(f"_{summary}_\n")

            # Format detailed results based on tool type
            if "hosts" in result and result["hosts"]:
                results_parts.append("**Discovered Hosts:**")
                for i, host in enumerate(result["hosts"][:20], 1):
                    ip = host.get("ip", "unknown")
                    hostname = host.get("hostname") or ""
                    status = host.get("status", "unknown")
                    hostname_str = f" ({hostname})" if hostname else ""
                    results_parts.append(f"  {i}. `{ip}`{hostname_str} - {status}")
                if len(result["hosts"]) > 20:
                    results_parts.append(f"  ... and {len(result['hosts']) - 20} more")
                results_parts.append("")

            if "ports" in result and result["ports"]:
                results_parts.append("**Open Ports:**")
                for port_info in result["ports"][:15]:
                    port = port_info.get("port", "?")
                    service = port_info.get("service", "unknown")
                    state = port_info.get("state", "open")
                    results_parts.append(f"  - Port `{port}` ({service}) - {state}")
                results_parts.append("")

            if "hotspots" in result and result["hotspots"]:
                results_parts.append("**Code Hotspots:**")
                for hs in result["hotspots"][:10]:
                    name = hs.get("name", "unknown")
                    score = hs.get("score", 0)
                    results_parts.append(f"  - `{name}` (risk score: {score})")
                results_parts.append("")

            if "findings" in result and result["findings"]:
                results_parts.append("**Findings:**")
                for f in result["findings"][:10]:
                    if isinstance(f, dict):
                        msg = f.get("message", f.get("rule_id", str(f)))
                        severity = f.get("severity", "info")
                        results_parts.append(f"  - [{severity.upper()}] {msg}")
                results_parts.append("")

            if "compliance_score" in result:
                score = result["compliance_score"]
                framework = result.get("framework", "")
                results_parts.append(f"**Compliance Score:** {score:.0f}% ({framework})")
                results_parts.append("")

            if "risk_score" in result:
                score = result["risk_score"]
                level = result.get("risk_level", "")
                results_parts.append(f"**Risk Score:** {score}/10 ({level})")
                results_parts.append("")

            if "action_items" in result and result["action_items"]:
                results_parts.append("**Recommended Actions:**")
                for item in result["action_items"][:5]:
                    if isinstance(item, dict):
                        action = item.get("action", str(item))
                        priority = item.get("priority", "medium")
                        results_parts.append(f"  - [{priority}] {action}")
                    else:
                        results_parts.append(f"  - {item}")
                results_parts.append("")

        # Build clean response (hide raw JSON from LLM)
        response_text = "\n".join(results_parts)
    else:
        # Clean up LLM response - remove JSON blocks for cleaner output
        import re
        clean_response = re.sub(r'```json\s*\{[^}]*"tool_call"[^}]*\{[^}]*\}[^}]*\}\s*```', '', llm_response)
        clean_response = re.sub(r'\{[^{}]*"tool_call"[^{}]*\}', '', clean_response)
        response_text = clean_response.strip() or llm_response

    # Add assistant response to memory
    conversation_memory.add_assistant_response(response_text)

    return {
        "response": response_text,
        "model": request.model,
        "workspace": workspace,
        "tool_calls": tool_results,
        "memory_entries": len(conversation_memory.entries),
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    # Check Ollama
    ollama_ok = False
    try:
        result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, timeout=5)
        ollama_ok = result.returncode == 0
    except Exception:
        pass

    # Check KOAS helpers
    koas_ok = False
    try:
        from koas_helpers import get_or_create_workspace
        koas_ok = True
    except ImportError:
        pass

    return {
        "status": "ok" if ollama_ok and koas_ok else "degraded",
        "ollama": ollama_ok,
        "koas_helpers": koas_ok,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# MEMORY MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/memory")
async def get_memory():
    """Get current conversation memory."""
    return conversation_memory.to_dict()


@app.delete("/api/memory")
async def clear_memory():
    """Clear conversation memory."""
    conversation_memory.clear()
    return {"status": "cleared", "timestamp": datetime.now().isoformat()}


@app.post("/api/memory/save")
async def save_memory(request: MemorySaveRequest):
    """Save conversation memory to a file."""
    # Save to workspace or temp directory
    save_dir = Path(conversation_memory.workspace) if conversation_memory.workspace else Path("/tmp")
    save_path = save_dir / f"{request.filename}.json"

    try:
        conversation_memory.save(save_path)
        return {
            "status": "saved",
            "path": str(save_path),
            "entries": len(conversation_memory.entries),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save memory: {str(e)}")


@app.post("/api/memory/load")
async def load_memory(request: MemorySaveRequest):
    """Load conversation memory from a file."""
    # Try workspace first, then tmp
    save_dir = Path(conversation_memory.workspace) if conversation_memory.workspace else Path("/tmp")
    load_path = save_dir / f"{request.filename}.json"

    # Also try as absolute path
    if not load_path.exists():
        load_path = Path(request.filename)
        if not load_path.exists():
            load_path = Path(f"{request.filename}.json")

    if not load_path.exists():
        raise HTTPException(status_code=404, detail=f"Memory file not found: {request.filename}")

    try:
        conversation_memory.load(load_path)
        return {
            "status": "loaded",
            "path": str(load_path),
            "entries": len(conversation_memory.entries),
            "workspace": conversation_memory.workspace,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load memory: {str(e)}")


@app.get("/api/memory/saved")
async def list_saved_memories():
    """List saved memory files."""
    saved = []

    # Check workspace directory
    if conversation_memory.workspace:
        ws_path = Path(conversation_memory.workspace)
        if ws_path.exists():
            for f in ws_path.glob("*.json"):
                if "memory" in f.stem.lower() or f.stem.startswith("session"):
                    saved.append({
                        "name": f.stem,
                        "path": str(f),
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })

    # Check tmp for koas sessions
    for f in Path("/tmp").glob("koas_*/session*.json"):
        saved.append({
            "name": f.stem,
            "path": str(f),
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })

    return {"saved_memories": saved, "count": len(saved)}


# =============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time tool execution updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "tool_call":
                # Broadcast start
                await manager.broadcast({
                    "type": "tool_start",
                    "tool": data["tool"],
                    "params": data.get("params", {}),
                    "timestamp": datetime.now().isoformat(),
                })

                # Execute tool
                result = await execute_tool(
                    data["tool"],
                    data.get("params", {}),
                    data.get("dry_run", False),
                )

                # Broadcast result
                await manager.broadcast({
                    "type": "tool_result",
                    "tool": data["tool"],
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                })

            elif data.get("type") == "scenario":
                scenario_id = data.get("scenario_id")
                if scenario_id in SCENARIOS:
                    scenario = SCENARIOS[scenario_id]
                    workspace = None

                    await manager.broadcast({
                        "type": "scenario_start",
                        "scenario": scenario_id,
                        "name": scenario["name"],
                        "steps": len(scenario["steps"]),
                    })

                    for i, step in enumerate(scenario["steps"]):
                        params = step["params"].copy()
                        if workspace:
                            params.setdefault("workspace", workspace)

                        await manager.broadcast({
                            "type": "step_start",
                            "step": i + 1,
                            "tool": step["tool"],
                            "params": params,
                        })

                        result = await execute_tool(step["tool"], params, data.get("dry_run", False))

                        if "workspace" in result:
                            workspace = result["workspace"]

                        await manager.broadcast({
                            "type": "step_result",
                            "step": i + 1,
                            "tool": step["tool"],
                            "result": result,
                        })

                    await manager.broadcast({
                        "type": "scenario_complete",
                        "scenario": scenario_id,
                        "workspace": workspace,
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Run the demo server."""
    parser = argparse.ArgumentParser(description="KOAS MCP Demo Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    KOAS MCP Demo Server                       ║
║                       Version 0.62.0                          ║
╠═══════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                               ║
║  API: http://{args.host}:{args.port}/api/tools                     ║
║  Docs: http://{args.host}:{args.port}/docs                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
