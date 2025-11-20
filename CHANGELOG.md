# 📘 CHANGELOG — RAGIX

All notable changes to the **RAGIX** project will be documented here.

---

## ⭐ v0.4 — MCP Integration & Unix Toolbox (2025-11-20)

### 🔧 New Features
- Full **MCP server** added (`MCP/ragix_mcp_server.py`)
  - Exposes: `ragix_chat`, `ragix_scan_repo`, `ragix_read_file`
  - Compatible with Claude Desktop, Claude Code, Codex, and all MCP clients
- Added **ragix_tools.py**: sovereign Unix toolbox  
  (`rt-find`, `rt-grep`, `rt-stats`, `rt-lines`, `rt-top`, `rt-replace`, `rt-doc2md`)
- Added **bash surrogates** (`rt.sh`, `rt-find.sh`, `rt-grep.sh`) for LLM friendliness
- Added `MCP/ragix_tools_spec.json` for agent integration

### 📦 Improved Architecture
- Unified naming convention (RAGIX everywhere, removed RADIX remnants)
- New environment variable configuration:
  - `UNIX_RAG_MODEL`
  - `UNIX_RAG_SANDBOX`
  - `UNIX_RAG_PROFILE`
  - `UNIX_RAG_ALLOW_GIT_DESTRUCTIVE`
- Added modular scanning + safe-shell improvements in `unix-rag-agent.py`
- Added project overview pre-scan at agent startup
- Added better denylist enforcement and profile support

### 📚 Documentation
- Completely rewritten **README.md** (v0.4)
- Added **README_RAGIX_TOOLS.md**
- Added **MCP/README_MCP.md**
- Updated **demo.md** with RAGIX v0.4 usage
- Introduced consistent ASCII diagrams & mermaid blocks
- Added installation, setup, tooling, safety, and MCP instructions

---

## v0.3 — Original Release (2025-11)
- Initial version of `unix-rag-agent.py`  
- JSON action protocol (`bash`, `bash_and_respond`, `edit_file`, `respond`)
- Git awareness (status, diff, log)
- Sandboxed shell with denylist
- Structured logging (`.agent_logs/commands.log`)
- Basic Unix-RAG retrieval capabilities
- No MCP tooling
- No sovereign toolbox yet

---

## v0.2 — Experimental (2025-10)
- First drafts of shell sandbox
- Local LLM integration (Ollama)
- Minimal Unix-RAG prompt engineering

---

## v0.1 — Prototype (2025-09)
- First conceptual prototype: run bash via LLM locally
- Pure sandbox experiment, no JSON tools
- Hardcoded reasoning

