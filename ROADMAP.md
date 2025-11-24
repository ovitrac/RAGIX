# ROADMAP to move from 0.5 to 0.7

Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | 2025-11-24

------

## Axes

I’ll use the same axes for each version:

1. **Orchestrator core** (RAGIX/RAGGAE, agents, profiles)
2. **Interface** (CLI, Unix-RAG, Web UI, WASM shell)
3. **Multi-agent workflows**
4. **Retrieval** (embeddings, hybrid search on codebases / docs)
5. **DevOps integration** (CI, tests, packaging)
6. **Security & secrets** (vault, MCP adapter, policies)

------

## v0.5 – “Core orchestrator + modular tooling”

**Goal:** solidify the architecture and the *shared orchestrator* so RAGIX and RAGGAE are just two faces of the same core. No fancy UI yet; focus on **package structure, abstractions, and safety**.

### 1. Orchestrator core

- Extract a **shared orchestrator layer**:
  - `ragix_core/` (or similar) with:
    - `orchestrator.py` – generic loop: “read → think → act” over tools.
    - `profiles/` – profiles like `dev`, `safe-read-only`, `unsafe` as pure config.
    - `llm_backends/` – local LLM abstraction (Ollama, later others).
  - Define **RAGIX** = “Unix-local dev assistant” profile
     and **RAGGAE** = “enterprise / doc-centric RAG” profile, but both using the same orchestrator primitives.
- Stabilise the **JSON action protocol**:
  - Single schema for tools/actions across CLI, MCP server, and Unix agent.
  - Add robust JSON repair + clear error reporting.

### 2. Interface (CLI + Unix-RAG)

- Turn the current scripts into a **proper package + entry points**:
  - `ragix` CLI (high-level) and `ragix-unix-agent` (Unix development assistant).
- Config precedence: `CLI > env > config file (> defaults)`
   (e.g. `~/.ragix.toml` / `ragix.yaml`).
- Tighten the **Unix-RAG agent**:
  - Sandbox root + profile defined in config.
  - Policy layer for dangerous commands (configurable denylist).

### 3. Multi-agent (v0.5 minimal)

- Introduce the **concept** but not full-blown graphs yet:
  - Single “primary agent” that can internally call **subroutines**:
    - e.g. `code_explorer`, `doc_explorer`, `git_helper` as *sub-agents* implemented as functions/classes using same tool API.
- Support a **simple linear task list**:
  - `orchestrator.run_playbook([...steps...])` where steps are LLM or tool calls, but still single-agent from the user POV.

### 4. Retrieval (first bricks)

- Define a minimal **retrieval interface**:
  - `Retriever` ABC with methods like `index(path)`, `query(text) → chunks`.
- Implement a **local codebase retriever (v0.5 minimal)**:
  - Pure keyword / ripgrep-based + simple ranking.
  - Integrated as a tool: “search in project” returning file+line+snippet.
- Lay the **embedding hooks** but don’t require a full vector DB yet:
  - Place for `EmbeddingsBackend` (e.g. local HF model, sentence-transformers).
  - But allow it to be “disabled” so v0.5 stays light.

### 5. DevOps & packaging

- **Restructure** repo:
  - `ragix_core/`, `profiles/`, `unix_agent/`, `MCP/`, `tools/`, `docs/`.
- Add **basic tests**:
  - JSON protocol + denylist + edit_file behaviour.
- Publish a **Python package** (local install at least): `pip install -e .`.
- Keep a clean **CHANGELOG** entry: describe v0.4 → v0.5 changes along these axes.

### 6. Security & secrets (foundation only)

- Make sure **no secrets** are ever logged by default:
  - Mask environment variables with patterns (`KEY`, `TOKEN`, `SECRET`).
- Add the abstraction for a **future vault**:
  - `SecretProvider` interface (but implemented initially as “dummy / env only”).
- Tighten **policy profiles**:
  - `strict`: no network tools, no git write, only read-only commands.
  - `dev`: allows edit_file/git but still denies rm -rf etc.
  - `unsafe`: full shell access (for your own lab only).

> **Summary v0.5:** A clean, modular **orchestrator + CLI + Unix-RAG agent**, with clear profiles, a pluggable retrieval and LLM layer, and a place for RAGGAE and a secrets vault — but minimal implementations to keep things small and robust.

------

## v0.6 – “Web UI + real multi-agent + hybrid retrieval”

**Goal:** build on the solid core from 0.5 to introduce **Web UI, proper multi-agent workflows, and serious retrieval** (embeddings + hybrid search). This is where RAGGAE starts to look like a real orchestration layer.

### 1. Orchestrator core

- Extend orchestrator to **graph-based workflows**:
  - A small DSL or Python API for defining agent graphs (e.g. LangGraph-style but lighter).
  - Nodes = agents, edges = transitions based on tool outcomes.
- RAGIX = “local dev graph” (code agent ↔ shell agent ↔ doc agent).
   RAGGAE = “RAG graph” (retriever ↔ summarizer ↔ planner ↔ answerer).

### 2. Interface – **Web UI (local, WASM shell, no cloud)**

- Local-only **Web UI**:
  - Single-page app served by RAGIX (FastAPI / Flask / simple HTTP).
  - Chat window + tool traces (commands, files touched, RAG chunks).
- **WASM shell**:
  - Embed a terminal-like client in the browser:
    - Commands are *proxied* to the orchestrator, not a real browser shell.
    - Respect the same policy profiles (strict/dev/unsafe).
- Multi-session support:
  - Tabs/workspaces named after projects or repos.

### 3. Multi-agent workflows (real version)

- Formalise **agent roles**:
  - `CodeAgent`, `DocAgent`, `GitAgent`, `InfraAgent`, etc., each with its own tool subset.
- Allow **task delegation**:
  - Primary agent can assign sub-tasks (“analyse this file”, “prepare patch”) to specialised agents and aggregate their results.
- Implement **multi-step “recipes”**:
  - e.g. “Refactor module X” → plan → inspect → edit → run tests → summarise diff.

### 4. Retrieval – **Embeddings & hybrid retrieval for codebases**

- Implement **embedding-backed retrievers**:

  - Code-aware chunking (files → functions/classes → smaller blocks).

  - Hybrid search:

    - keyword (grep / ripgrep)

    - semantic (vector search)
    - structure (e.g. function/class names).

- Provide **indexers**:

  - CLI command: `ragix index . --profile code` → builds index in `.ragix_index/`.

- Use retrieval for:

  - “jump to definition / usage” as tool calls.
  - RAG answers in RAGGAE profile for docs & READMEs.

### 5. DevOps integration (early CI hooks)

- Design RAGIX as **a service usable in CI**:
  - A “batch mode” CLI:
    - `ragix run-ci-check --playbook ci_checks.yaml`
  - Typical steps:
    - lint / docstring generation proposals / TODO scanning / simple reasoning checks.
- Provide **starter templates**:
  - GitHub Actions / GitLab CI `.yaml` that:
    - spins up a local model (optional, via Ollama),
    - runs RAGIX in non-interactive mode,
    - posts results as an artifact or PR comment draft (no auto-commit yet).

### 6. Security & secrets (towards vaults)

- Implement a **local secrets vault** abstraction:
  - File-based encrypted store (e.g. age/gpg) or pluggable backend.
  - Access requires explicit user confirmation / policy rule.
- Start the **“credential MCP adapter”**:
  - RAGIX exposes a MCP tool that other MCP clients can call to get ephemeral credentials (never raw secrets if you don’t want to).
  - Policies to limit scope (per project, per service).

> **Summary v0.6:** RAGIX gets a local **Web UI + WASM shell**, a true **multi-agent orchestrator (RAGGAE)**, and **hybrid retrieval** for code and docs. CI integration is “first-class citizen but optional”, and security starts to use a real vault abstraction.

------

## v0.7 – “Plug-and-play CI + enterprise-grade secrets & observability”

**Goal:** harden everything for **team / enterprise use**: CI plug-ins, vault integration, observability, and a mature secrets MCP adapter. Functionally, RAGIX becomes a *platform*.

### 1. Orchestrator core

- Stabilise the **public API**:
  - `orchestrator` interfaces documented for:
    - external Python clients,
    - MCP & HTTP APIs.
- Versioned protocol:
  - `ragix_protocol_version = "0.7"` included in metadata for future compatibility.

### 2. Interface

- Web UI v2:
  - Workspace history (projects, sessions, logs).
  - Diff viewer embedded (for patches).
  - Visual graph of multi-agent workflows (nodes, edges, status).

### 3. Multi-agent workflows

- Graph orchestration v2:
  - Conditional branches, retries, timeouts.
  - “Human-in-the-loop” nodes (require confirmation before continuing).
- Library of **ready-made flows**:
  - “Code review assistant”, “Repo onboarding summariser”, “Doc QA bot on repo”, etc.
- Export/import of workflows as **YAML/JSON recipes**.

### 4. Retrieval

- Mature **retrieval layer**:
  - Multiple backends (local vector store, SQLite/pgvector, etc., but still local-first).
  - Metrics: recall proxy, chunk hit ratios, retrieval logs.
- Allow **shared indices** across projects (e.g. same knowledge base for many repos).

### 5. DevOps – **Plug-and-play CI integration**

- “One-liner” CI integration:
  - For GitHub:
    - Provide a **reusable Action** (`ovitrac/ragix-ci@v0.7`).
  - For GitLab:
    - Example `.gitlab-ci.yml` templates.
- Modes:
  - **advisory** (comments, suggestions),
  - **gating** (fail pipeline if conditions not met, with explicit rules).
- Documentation for:
  - how to pin models,
  - how to run fully offline,
  - how to cache indices between CI runs.

### 6. Security & secrets – **Secrets vault + credential MCP adapter**

- Full **secrets vault integration**:
  - Pluggable drivers:
    - local encrypted store,
    - system keyring,
    - (optionally) external secret managers if self-hosted.
- **Credential MCP adapter**:
  - Exposes safe, scoped credentials to other MCP clients:
    - short-lived tokens,
    - scoped to a repo / environment.
  - Auditing:
    - who requested what, for which task, which workflow.
- Policy engine:
  - Central config to define:
    - which agents can access which secrets,
    - which tools are allowed in which profile,
    - what is forbidden in CI vs interactive.

> **Summary v0.7:** RAGIX becomes “drop-in” for CI pipelines, with a strong **secrets story** (vault + MCP adapter), reproducible workflows, and full observability. It’s no longer just a dev toy; it’s a small platform.

------

## How to use this concretely:

- Turn it into a **roadmap section** in `README.md` with a short table:
   version × axes (core / UI / agents / retrieval / CI / secrets).
- Or into **structured `CHANGELOG.md` targets** (what 0.5, 0.6, 0.7 will add/change/deprecate) so it stays aligned with your last nightly edits.