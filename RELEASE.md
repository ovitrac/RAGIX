# RAGIX Release v0.11.1

**Build:** 202511281400
**Date:** 2025-11-28

## 1. Overview

Version 0.20.0 marks a major leap forward for RAGIX, evolving it from a development assistant into a production-ready, multi-agent orchestration platform. This release introduces a powerful AST-based code analysis engine, a comprehensive Web UI, and a full suite of documentation, solidifying its position as a sovereign, local-first tool for professional software engineering.

## 2. 🚀 Major Features & Capabilities

This release is packed with new, deeply integrated capabilities that provide a powerful and cohesive experience.

### 2.1. Advanced AST Code Analysis

RAGIX now parses your entire codebase (Python & Java) into an Abstract Syntax Tree (AST), enabling deep, structural analysis and visualization.

- **Key Features:** Dependency graph generation, cycle detection, code metrics (cyclomatic complexity, technical debt), and an interactive query language.
- **Example (`ragix-ast` CLI):** Find all classes that extend `BaseService` and are decorated with `@Transactional`.
  ```bash
  ragix-ast search ./src "type:class extends:BaseService @Transactional"
  ```

### 2.2. Multi-Agent Orchestration

Define, execute, and stream complex workflows with a dependency-aware graph executor.

- **Key Features:** Pre-built templates for common tasks (`bug_fix`, `code_review`, `feature_addition`), parallel execution, and a clear separation of Planner, Worker, and Verifier agents.
- **Example (Python):**
  ```python
  from ragix_core import get_template_manager
  
  manager = get_template_manager()
  graph = manager.instantiate("bug_fix", {
      "bug_description": "TypeError in handler.py",
  })
  # ... then run with GraphExecutor
  ```

### 2.3. Hybrid Search Engine

Combine the best of keyword-based (BM25) and semantic (Vector) search for highly accurate code retrieval.

- **Key Features:** Multiple fusion strategies (`RRF`, `Weighted`, `Interleave`), clear source attribution for results, and a simple API.

### 2.4. Flexible LLM Backends

Choose the right LLM for the job, with a clear distinction between sovereign and cloud-based models.

- **Sovereign (Default):** 🟢 **Ollama**. 100% local, private, and free. Recommended for sensitive codebases.
- **Cloud-Based:** 🔴 **Claude & OpenAI**. Higher reasoning quality at the cost of sending data to third-party APIs.

### 2.5. Production-Grade Infrastructure

- **Monitoring:** Health checks for system components (`/api/health`).
- **Resilience:** Built-in patterns like `CircuitBreaker` and `retry_async` for robust tool execution.
- **Caching:** In-memory and disk-based caching for LLM responses and tool calls to improve performance.

## 3. 💻 Developer Experience & Usability

### 3.1. Comprehensive Documentation Suite

A full suite of documentation has been created to make RAGIX accessible and easy to learn.

| Document | Location | Description |
|---|---|---|
| **CLI Guide** | `docs/CLI_GUIDE.md` | A complete reference for all `ragix-*` commands. |
| **AST Guide** | `docs/AST_GUIDE.md` | A deep dive into code analysis with `ragix-ast`. |
| **API Reference** | `docs/API_REFERENCE.md` | REST API documentation for the `ragix-web` server. |
| **Architecture** | `docs/ARCHITECTURE.md` | An overview of the system architecture. |
| **Playbook Guide** | `docs/PLAYBOOK_GUIDE.md`| How to write `ragix-batch` automation playbooks. |

### 3.2. New Web UI

A new, comprehensive Web UI (`ragix-web`) provides a graphical interface for:
- Interactive agent chat.
- Live AST visualizations (dependency graphs, DSM matrices, radial explorers).
- Session management and log viewing.

### 3.3. Project Reorganization

- The project root has been cleaned significantly.
- Old development and planning documents are now in `docs/archive/`.
- Internal technical notes are consolidated in `docs/developer/`.
- Every project folder now contains a `README.md`.

## 4. 🐞 Bug Fixes

- **Web UI:** Fixed a critical `AttributeError` on the `/api/agents/config` endpoint that caused a 500 Internal Server Error.
- Numerous other minor stability improvements.
