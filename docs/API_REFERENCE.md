# RAGIX Web API Reference

**Version:** 0.20.0 | **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Updated:** 2025-11-28

---

This document provides a comprehensive reference for the REST API exposed by the RAGIX web server (`ragix-web`). The API enables programmatic interaction with RAGIX sessions, the AST analysis engine, agent configurations, and workflow management.

## Table of Contents

1. [Overview](#1-overview)
2. [Server & Session Management](#2-server--session-management)
3. [AST Analysis API](#3-ast-analysis-api)
4. [Agents & Workflows API](#4-agents--workflows-api)
5. [Memory & Context API](#5-memory--context-api)
6. [Error Handling](#6-error-handling)

---

## 1. Overview

**Base URL:** `http://localhost:8000/api`

**Authentication:** The API is designed for local-first use and does not require authentication by default. For production deployments, consider adding authentication middleware.

**Content-Type:** All requests and responses use `application/json`.

**CORS:** Enabled for local development (`localhost` origins).

---

## 2. Server & Session Management

Endpoints for managing the server lifecycle and user sessions.

### `GET /api/health`

Checks the health of the server and its dependencies.

- **Response `200 OK`:**
  ```json
  {
    "status": "ok",
    "version": "0.20.0",
    "sessions": 1,
    "ast_available": true,
    "ollama_status": "connected"
  }
  ```

---

### `POST /api/sessions`

Creates a new project session with a dedicated sandbox.

- **Request Body:**
  ```json
  {
    "sandbox_root": "/path/to/your/project",
    "model": "mistral",
    "profile": "dev"
  }
  ```

  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | `sandbox_root` | string | No | Project directory (defaults to server launch directory) |
  | `model` | string | No | Ollama model name (default: `mistral`) |
  | `profile` | string | No | Safety profile: `strict`, `dev`, `unsafe` (default: `dev`) |

- **Response `200 OK`:**
  ```json
  {
    "session_id": "session_20251128_103000",
    "sandbox_root": "/path/to/your/project",
    "model": "mistral",
    "profile": "dev"
  }
  ```

---

### `GET /api/sessions`

Lists all currently active sessions.

- **Response `200 OK`:**
  ```json
  {
    "sessions": [
      {
        "id": "session_20251128_103000",
        "sandbox_root": "/path/to/your/project",
        "model": "mistral",
        "profile": "dev",
        "created_at": "2025-11-28T10:30:00.123456"
      }
    ]
  }
  ```

---

### `DELETE /api/sessions/{session_id}`

Deletes a specific session and cleans up associated resources.

- **URL Parameters:**
  - `session_id` (string, required): The ID of the session to delete.

- **Response `200 OK`:**
  ```json
  {
    "status": "deleted",
    "session_id": "session_20251128_103000"
  }
  ```

---

### `GET /api/sessions/{session_id}/logs`

Retrieves the command logs for a specific session.

- **URL Parameters:**
  - `session_id` (string, required): The session ID.

- **Query Parameters:**
  - `limit` (int, optional, default: 50): Maximum number of log lines to return.
  - `offset` (int, optional, default: 0): Number of lines to skip.

- **Response `200 OK`:**
  ```json
  {
    "logs": [
      "[2025-11-28 10:31:05] Thinking: The user wants to know the file structure.",
      "[2025-11-28 10:31:05] Executing: ls -F",
      "[2025-11-28 10:31:05] Command finished with code 0"
    ],
    "total": 150,
    "limit": 50,
    "offset": 0
  }
  ```

---

## 3. AST Analysis API

Endpoints for the Abstract Syntax Tree analysis engine. These require the `[ast]` optional dependency.

### `GET /api/ast/status`

Checks if the AST analysis module is available and ready.

- **Response `200 OK`:**
  ```json
  {
    "available": true,
    "message": "AST analysis ready",
    "supported_languages": ["python", "java"]
  }
  ```

---

### `GET /api/ast/graph`

Builds and returns a dependency graph for a given path.

- **Query Parameters:**

  | Parameter | Type | Required | Default | Description |
  |-----------|------|----------|---------|-------------|
  | `path` | string | Yes | - | Absolute path to directory or file |
  | `max_nodes` | int | No | 1000 | Maximum nodes to return |
  | `cluster` | bool | No | true | Group nodes by package |

- **Response `200 OK`:**
  ```json
  {
    "nodes": [
      {
        "id": "src.main.App",
        "name": "App",
        "type": "class",
        "file": "/path/to/src/main.py",
        "line": 15,
        "package": "src.main"
      }
    ],
    "links": [
      {
        "source": "src.main.App",
        "target": "src.utils.helpers",
        "type": "import",
        "weight": 1
      }
    ],
    "total_nodes": 150,
    "total_edges": 300,
    "truncated": false
  }
  ```

---

### `GET /api/ast/metrics`

Calculates and returns a professional code metrics report.

- **Query Parameters:**
  - `path` (string, required): Absolute path to the directory to analyze.

- **Response `200 OK`:**
  ```json
  {
    "path": "/path/to/your/project",
    "summary": {
      "total_files": 50,
      "total_lines": 10000,
      "code_lines": 7500,
      "comment_lines": 1500,
      "blank_lines": 1000
    },
    "complexity": {
      "total": 350,
      "average": 3.5,
      "max": 25,
      "max_symbol": "src.main.process_data"
    },
    "maintainability": {
      "index": 75.5,
      "rating": "Good"
    },
    "debt": {
      "estimated_hours": 40.5,
      "rating": "B"
    },
    "coupling": {
      "afferent": 234,
      "efferent": 567,
      "instability": 0.71
    },
    "hotspots": [
      {"name": "src.main.process_data", "complexity": 25, "file": "src/main.py", "line": 45}
    ]
  }
  ```

---

### `GET /api/ast/search`

Searches for code symbols using the RAGIX query language.

- **Query Parameters:**

  | Parameter | Type | Required | Description |
  |-----------|------|----------|-------------|
  | `path` | string | Yes | Directory to search in |
  | `q` | string | Yes | Search query (supports predicates and wildcards) |
  | `type` | string | No | Filter by symbol type |
  | `limit` | int | No | Maximum results (default: 50) |

- **Response `200 OK`:**
  ```json
  {
    "query": "type:class name:*Service*",
    "results": [
      {
        "name": "UserService",
        "qualified_name": "com.example.UserService",
        "type": "class",
        "file": "/path/to/project/src/com/example/UserService.java",
        "line": 10,
        "annotations": ["@Service", "@Transactional"]
      }
    ],
    "total": 5
  }
  ```

---

### `GET /api/ast/matrix`

Generates a Dependency Structure Matrix (DSM) for architectural analysis.

- **Query Parameters:**
  - `path` (string, required): Directory to analyze.
  - `level` (string, optional): Aggregation level (`class`, `package`, `file`). Default: `package`.

- **Response `200 OK`:**
  ```json
  {
    "labels": ["com.example.api", "com.example.service", "com.example.db"],
    "matrix": [
      [0, 5, 0],
      [2, 0, 8],
      [0, 3, 0]
    ],
    "cycles": [
      ["com.example.service", "com.example.db"]
    ]
  }
  ```

---

## 4. Agents & Workflows API

Endpoints for configuring agents and executing workflows.

### `GET /api/agents`

Lists the available agent types and their capabilities.

- **Response `200 OK`:**
  ```json
  {
    "available": true,
    "agents": [
      {
        "id": "code",
        "name": "Code Agent",
        "description": "Analyzes, searches, and modifies code",
        "capabilities": ["code_read", "code_write", "code_search", "ast_analysis"]
      },
      {
        "id": "test",
        "name": "Test Agent",
        "description": "Runs tests and analyzes coverage",
        "capabilities": ["test_run", "coverage_analysis"]
      },
      {
        "id": "doc",
        "name": "Documentation Agent",
        "description": "Generates and updates documentation",
        "capabilities": ["doc_generate", "doc_update"]
      }
    ]
  }
  ```

---

### `GET /api/workflows`

Lists all available workflow templates.

- **Response `200 OK`:**
  ```json
  {
    "available": true,
    "workflows": [
      {
        "id": "bug_fix",
        "name": "Bug Fix Workflow",
        "description": "Locates, diagnoses, fixes, and tests a bug.",
        "parameters": [
          {"name": "bug_description", "type": "string", "required": true},
          {"name": "affected_files", "type": "string", "required": false}
        ]
      }
    ]
  }
  ```

---

### `POST /api/workflows/{template_id}/execute`

Executes a workflow template with the provided parameters.

- **URL Parameters:**
  - `template_id` (string, required): The workflow template ID.

- **Request Body:**
  ```json
  {
    "params": {
      "bug_description": "TypeError in handler.py line 45",
      "affected_files": "src/handlers/"
    }
  }
  ```

- **Response `200 OK`:**
  ```json
  {
    "execution_id": "exec_20251128_103500",
    "template": "bug_fix",
    "status": "running",
    "steps": [
      {"id": "analyze", "status": "completed"},
      {"id": "fix", "status": "running"},
      {"id": "test", "status": "pending"}
    ]
  }
  ```

---

### `GET /api/agents/config`

Retrieves the current multi-agent LLM configuration.

- **Response `200 OK`:**
  ```json
  {
    "mode": "strict",
    "planner_model": "mistral:latest",
    "worker_model": "granite3.1-moe:3b",
    "verifier_model": "granite3.1-moe:3b",
    "available_models": [
      {"name": "mistral:latest", "size_gb": 4.1, "parameters": "7B"}
    ]
  }
  ```

---

### `POST /api/agents/config`

Updates the multi-agent LLM configuration for the current session.

- **Request Body:**
  ```json
  {
    "mode": "custom",
    "planner_model": "qwen2.5",
    "worker_model": "mistral"
  }
  ```

- **Response `200 OK`:**
  ```json
  {
    "status": "ok",
    "session_id": "session_20251128_103000",
    "message": "Agent config updated for session"
  }
  ```

---

## 5. Memory & Context API

Endpoints for managing conversation history and user context.

### `GET /api/memory`

Retrieves the conversation memory for a session.

- **Query Parameters:**
  - `session_id` (string, optional): Session ID (uses active session if not provided).
  - `limit` (int, optional, default: 100): Maximum messages to return.

- **Response `200 OK`:**
  ```json
  {
    "messages": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "Find all Python files",
        "timestamp": "2025-11-28T10:30:00.000000"
      },
      {
        "id": "msg_002",
        "role": "assistant",
        "content": "I found 25 Python files...",
        "timestamp": "2025-11-28T10:30:05.000000"
      }
    ],
    "total": 50
  }
  ```

---

### `DELETE /api/memory/{message_id}`

Deletes a specific message from memory.

- **URL Parameters:**
  - `message_id` (string, required): The message ID to delete.

- **Response `200 OK`:**
  ```json
  {
    "status": "deleted",
    "message_id": "msg_001"
  }
  ```

---

### `DELETE /api/memory`

Clears all conversation memory for a session.

- **Query Parameters:**
  - `session_id` (string, optional): Session ID (uses active session if not provided).

- **Response `200 OK`:**
  ```json
  {
    "status": "cleared",
    "session_id": "session_20251128_103000"
  }
  ```

---

### `GET /api/context`

Retrieves the user context (custom instructions).

- **Response `200 OK`:**
  ```json
  {
    "context": "Focus on Python best practices. Prefer type hints.",
    "updated_at": "2025-11-28T09:00:00.000000"
  }
  ```

---

### `POST /api/context`

Updates the user context.

- **Request Body:**
  ```json
  {
    "context": "Focus on Python best practices. Use type hints. Follow PEP 8."
  }
  ```

- **Response `200 OK`:**
  ```json
  {
    "status": "updated",
    "context": "Focus on Python best practices. Use type hints. Follow PEP 8."
  }
  ```

---

## 6. Error Handling

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "Session 'session_xyz' does not exist",
    "details": {}
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `SESSION_NOT_FOUND` | 404 | The specified session does not exist |
| `AST_NOT_AVAILABLE` | 503 | AST module not installed or unavailable |
| `INVALID_PATH` | 400 | The provided path is invalid or outside sandbox |
| `MODEL_NOT_FOUND` | 404 | The specified Ollama model is not available |
| `WORKFLOW_NOT_FOUND` | 404 | The specified workflow template does not exist |
| `VALIDATION_ERROR` | 422 | Request body validation failed |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

*For more information, see the [CLI Guide](CLI_GUIDE.md) or the [Architecture](ARCHITECTURE.md) documentation.*
