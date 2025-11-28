# RAGIX MCP Server Guide

**Version:** 0.20.0 | **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Updated:** 2025-11-28

---

This document provides instructions and a technical reference for the RAGIX MCP (Mission Control Protocol) server. The server acts as a bridge, allowing any MCP-compatible client (like Claude Desktop, Claude Code, or IDE extensions) to use RAGIX's powerful, local-first capabilities.

## Table of Contents

1. [Overview](#1-overview)
2. [Setup and Usage](#2-setup-and-usage)
3. [Tool Reference](#3-tool-reference)
4. [Configuration](#4-configuration)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Overview

When you run the RAGIX MCP server, it exposes a set of tools that an MCP client can call. This allows a primary AI assistant (like Claude) to delegate tasks such as local file searching, code analysis, and workflow execution to RAGIX.

This setup provides the best of both worlds: the advanced reasoning of a frontier model like Claude, combined with the secure, local, and context-aware execution of RAGIX.

**Key Features:**
- **Local Execution:** Your code and data never leave your machine
- **Sandboxed Environment:** All actions are performed within the secure RAGIX sandbox
- **Hybrid Search:** Combines keyword and semantic search for accurate code retrieval
- **Workflow Automation:** Enables the primary AI to trigger complex, multi-step local workflows
- **Complete Auditability:** Every action is logged with hash-chain integrity

### Directory Structure

```text
MCP/
  ragix_mcp_server.py   # MCP server exposing the Unix-RAG agent
  ragix_tools_spec.json # JSON spec for the RAGIX Unix toolbox (rt-*)
  README_MCP.md         # This documentation
```

---

## 2. Setup and Usage

### Prerequisites

- Python 3.10+
- Ollama installed and running
- RAGIX installed with MCP support

### Step 1: Install Dependencies

```bash
# Install RAGIX with MCP support
pip install -e ".[mcp]"

# Verify MCP CLI is available
mcp --version
```

### Step 2: Configure Environment

Set environment variables to configure the agent:

```bash
# Required: Project directory for the sandbox
export UNIX_RAG_SANDBOX="/path/to/your/project"

# Required: Ollama model to use
export UNIX_RAG_MODEL="mistral"

# Optional: Safety profile (strict, dev, unsafe)
export UNIX_RAG_PROFILE="dev"
```

### Step 3: Install the MCP Server

```bash
# Register RAGIX as an MCP server
mcp install MCP/ragix_mcp_server.py --name "RAGIX"
```

### Step 4: Enable in Your Client

**For Claude Desktop:**
1. Open Claude Desktop → Settings → MCP Servers
2. Find "RAGIX" in the server list
3. Toggle it ON
4. Restart Claude Desktop

**For VS Code (with MCP extension):**
1. Open VS Code Settings
2. Search for "MCP Servers"
3. Add RAGIX to enabled servers
4. Reload VS Code window

---

## 3. Tool Reference

The following tools are exposed by the RAGIX MCP server.

### `ragix_chat`

Run a single, stateful Unix-RAG reasoning step. This is the most general-purpose tool.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Natural language instruction |

**Returns:**
```json
{
  "response": "The agent's natural language response.",
  "last_command": {
    "command": "ls -la",
    "cwd": "/path/to/project",
    "stdout": "...",
    "stderr": "",
    "returncode": 0
  }
}
```

---

### `ragix_scan_repo`

Performs a quick scan of the project directory structure.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_depth` | integer | No | 4 | Maximum directory depth |
| `include_hidden` | boolean | No | false | Include hidden files |

**Returns:** Array of file objects with path, size, and extension.

---

### `ragix_read_file`

Reads the content of a single text file from within the sandbox.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | Yes | - | Relative path within sandbox |
| `max_bytes` | integer | No | 65536 | Maximum bytes to read |

**Returns:** File content as string (truncated if exceeds max_bytes).

---

### `ragix_search`

Performs a hybrid (BM25 + vector) search across the codebase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `k` | integer | No | 10 | Number of results |
| `strategy` | string | No | "rrf" | Fusion strategy |

**Returns:**
```json
{
  "query": "database connection",
  "strategy": "rrf",
  "results": [
    {
      "file": "src/db.py",
      "name": "connect_to_db",
      "type": ".py",
      "score": 0.85,
      "matched_terms": ["database", "connection"]
    }
  ]
}
```

---

### `ragix_workflow`

Executes a pre-defined, multi-step workflow template.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `template` | string | Yes | Template name (e.g., `bug_fix`) |
| `params` | object | Yes | Template parameters |

**Returns:** Workflow execution result with status and step outputs.

---

### `ragix_templates`

Lists all available workflow templates and their parameters.

**Returns:**
```json
{
  "templates": [
    {
      "name": "bug_fix",
      "description": "Locate, diagnose, fix, and test a bug.",
      "parameters": [
        {"name": "bug_description", "required": true}
      ]
    }
  ]
}
```

---

### `ragix_config`

Retrieves the current RAGIX configuration.

**Returns:**
```json
{
  "version": "0.20.0",
  "llm": {"backend": "ollama", "model": "mistral"},
  "safety": {"profile": "dev", "air_gapped": true}
}
```

---

### `ragix_health`

Provides a comprehensive system health status report.

**Returns:**
```json
{
  "status": "healthy",
  "checks": {
    "ollama": {"status": "ok", "details": "Ollama is running"},
    "disk": {"status": "ok", "details": "Disk space sufficient"}
  },
  "timestamp": "2025-11-28T12:00:00.000000"
}
```

---

### `ragix_logs`

Retrieves the most recent log entries from the agent's command log.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `n` | integer | No | 50 | Number of entries |

**Returns:**
```json
{
  "entries": ["[2025-11-28 12:00:00] Executing: ls -l"],
  "total": 100,
  "log_file": "/path/to/project/.agent_logs/commands.log"
}
```

---

### `ragix_verify_logs`

Verifies the integrity of the command log using its SHA256 hash chain.

**Returns:**
```json
{
  "valid": true,
  "total_entries": 100,
  "verified_entries": 100,
  "first_invalid_entry": null,
  "errors": []
}
```

---

## 4. Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UNIX_RAG_SANDBOX` | Project sandbox root | Current directory |
| `UNIX_RAG_MODEL` | Ollama model name | `mistral` |
| `UNIX_RAG_PROFILE` | Safety profile | `dev` |
| `RAGIX_LOG_LEVEL` | Logging verbosity | `INFO` |

### Example Configuration

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# RAGIX MCP Configuration
export UNIX_RAG_SANDBOX="$HOME/projects/current"
export UNIX_RAG_MODEL="mistral:instruct"
export UNIX_RAG_PROFILE="dev"
```

---

## 5. Troubleshooting

### Server Not Appearing in Client

1. Verify installation: `mcp list`
2. Check server status: `mcp status RAGIX`
3. Restart the MCP server and client

### Connection Errors

```bash
# Check Ollama is running
ollama list

# Test RAGIX standalone
ragix-unix-agent --model mistral
```

### Permission Issues

Ensure the sandbox directory is readable:

```bash
ls -la "$UNIX_RAG_SANDBOX"
```

### Log Verification Failed

If log verification fails, the audit trail may have been tampered with:

```bash
# Check log file
cat "$UNIX_RAG_SANDBOX/.agent_logs/commands.log"

# Verify manually
ragix_verify_logs
```

### Performance Issues

- Use smaller models for faster response: `granite3.1-moe:3b`
- Increase Ollama memory: `OLLAMA_NUM_GPU=1`
- Pre-build search index: `ragix-index $UNIX_RAG_SANDBOX`

---

*For more information, see the [CLI Guide](../docs/CLI_GUIDE.md) or the [Architecture](../docs/ARCHITECTURE.md) documentation.*
