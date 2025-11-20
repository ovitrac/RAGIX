# RAGIX MCP Server

This folder contains the **RAGIX MCP server** and tool specs so that any
MCP-compatible client (Claude Desktop, Claude Code, OpenAI Code / Codex, etc.)
can use RAGIX as a local Unix-RAG assistant.

Current structure:

```text
MCP/
  ragix_mcp_server.py   # MCP server exposing the Unix-RAG agent
  ragix_tools_spec.json # JSON spec for the RAGIX Unix toolbox (rt-*)