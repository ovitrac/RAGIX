# RAGIX –  Workload (v0.5 → v0.7)

**Repository:** `ovitrac/RAGIX`  
**Scope:** Implement RAGIX roadmap from v0.4 to v0.5, and prepare scaffolding for v0.6–0.7.  
**Style:** Local-first, modular, *Unix-RAG*, minimal magic, transparent orchestration.

---

## 0. Meta-rules for Claude Code

When working on this repo, always follow these rules:

1. **Authorship & attribution**
   - Do **not** overwrite author headers.  
   - New files must clearly indicate:

     ```text
     Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-23+
     ```

     (You can appear in comments as “LLM assistant (Claude)” in a short note, but never as the main author.)

2. **Safety & scope**
   - Never run destructive shell commands (`rm -rf`, `truncate`, massive `sed -i`, etc.) in this repo.
   - Prefer **small, localized edits** with clear diffs.
   - Use search/grep/find to understand first; avoid blind refactors.

3. **Philosophy of RAGIX**
   - RAGIX is:
     - **Local** (Ollama / local LLM, no cloud calls).
     - **Unix-first** (grep, find, sed, head, tail…).
     - **RAG-like loop** for code & docs, not yet a heavy framework.
   - Keep code:
     - simple, readable,
     - explicit rather than “magic”,
     - suitable for being orchestrated by another LLM agent.

4. **Workflow**
   - For each task:
     - Read the relevant files first.
     - Draft changes in a small subset of files.
     - Run basic checks (lint, `python -m`, or ad-hoc tests where available).
     - Summarize what changed in a short bullet list (for CHANGELOG alignment).

---

## 1. Milestone v0.5 – Core orchestrator + Unix agent

**Goal:** Turn the current prototype into a **modular package** with a shared orchestrator, cleaner CLI, safety profiles, and a clear path for RAGGAE & future agents. No Web UI yet; retrieval remains simple (text/grep-based).

### 1.1. Task: Create core package skeleton

**Goal:** Introduce a `ragix_core` (or similar) package and move core logic out of `unix-rag-agent.py`.

**Files to inspect first**

- `unix-rag-agent.py` (uploaded to you as context).
- `README.md` (root).
- `CHANGELOG.md` (root).
- Any `MCP/` files: `MCP/ragix_mcp_server.py`, `MCP/ragix_tools_spec.json`.
- Shell helpers: `rt.sh`, `rt-find.sh`, `rt-grep.sh`.

**Steps**

1. **Create package structure** under `ragix_core/`:

   ```text
   ragix_core/
     __init__.py
     orchestrator.py
     llm_backends.py
     tools_shell.py
     profiles.py
     logging_utils.py
   ragix_unix/
     __init__.py
     agent.py
     cli.py
   ```

2. **Extract logic** from `unix-rag-agent.py`:
   - Orchestrator loop → `ragix_core/orchestrator.py`.
   - LLM call + system prompt handling → `ragix_core/llm_backends.py`.
   - Shell sandbox & denylist → `ragix_core/tools_shell.py`.
   - Profile definition (strict/dev/unsafe) → `ragix_core/profiles.py`.
   - Logging to `.agent_logs` → `ragix_core/logging_utils.py`.
3. Leave a **thin wrapper** in `unix-rag-agent.py` (or deprecate it) that calls the new package entry points.

**Done when**

- Code runs via a **new CLI** (see next task) and the old script is either:
  - a small wrapper that imports the package, or
  - clearly marked as legacy in the README.

---

### 1.2. Task: Implement CLI entry points and config

**Goal:** Provide a clean CLI interface (`ragix` / `ragix-unix-agent`) with configuration from CLI, env, and config file.

**Steps**

1. Add an **entry point** in `pyproject.toml` or `setup.cfg` (depending on repo style, or create `pyproject.toml` if missing):

   ```toml
   [project.scripts]
   ragix-unix-agent = "ragix_unix.cli:main"
   ```

2. Implement `ragix_unix/cli.py` with:
   - `--sandbox-root`
   - `--model`
   - `--profile {strict, dev, unsafe}`
   - `--dry-run / --no-dry-run`
   - `--config PATH` (optional)
3. Implement **config resolution**:
   - Look for `~/.ragix.toml` or `ragix.yaml` in the repo root.
   - Precedence: `CLI > env > config > defaults`.
4. Update `README.md`:
   - New installation block: `pip install -e .`.
   - Usage examples:

     ```bash
     ragix-unix-agent --profile dev --model mistral
     ```

**Done when**

- Running `ragix-unix-agent` from a fresh clone + editable install works and configuration precedence is documented.

---

### 1.3. Task: Policy profiles and sandbox safety layer

**Goal:** Centralise safety rules and make them easy to extend without editing code.

**Steps**

1. In `ragix_core/profiles.py`, define profiles:

   ```python
   class Profile(Enum):
       STRICT = "strict"
       DEV = "dev"
       UNSAFE = "unsafe"
   ```

2. Associate each profile with:
   - Allowed tools (bash/shell, read-only vs edit).
   - Denylisted patterns (e.g. rm -rf, destructive git commands).
   - Whether network access is allowed (for future use).

3. In `ragix_core/tools_shell.py`, wrap shell execution with:
   - `check_policy(profile, command) -> allowed / reason`.
   - If denied, return a clear message to the orchestrator.

4. Expose a **config patch mechanism**:
   - In config file, allow adding extra denylist entries per profile without touching Python code.

**Done when**

- Profiles are clearly represented in code.
- A command forbidden in a given profile is rejected with an explicit message, and this behaviour is stable.

---

### 1.4. Task: JSON action protocol & error handling

**Goal:** Stabilise the JSON tool protocol used between the LLM and the orchestrator, with robust error handling but no over-complicated schemas.

**Steps**

1. In `ragix_core/orchestrator.py`, define a small **Action model**:

   ```python
   {"action": "bash", "command": "..."}
   {"action": "respond", "message": "..."}
   {"action": "edit_file", "path": "...", "old": "...", "new": "..."}
   ```

2. Implement:
   - A function `parse_action(raw_text: str) -> Action | Error`:
     - Try `json.loads`.
     - If it fails, one retry with a short “please return pure JSON” system message.
   - On persistent failure, produce a **clean error** for the user without crashing.

3. Document the protocol in `README_RAGIX_TOOLS.md` (or a new `PROTOCOL.md`) so humans and LLMs know the expected shape.

**Done when**

- Malformed JSON from the LLM doesn’t crash the loop; instead, user sees an intelligible error and the tool can continue.
- The action types currently used (`bash`, `respond`, `edit_file`) are clearly documented.

---

### 1.5. Task: Logging and observability

**Goal:** Provide lightweight logging suitable for debugging RAGIX behaviour, keeping in mind future multi-agent and CI use.

**Steps**

1. In `ragix_core/logging_utils.py`, implement:
   - A text log: `.agent_logs/commands.log`.
   - A JSONL event log: `.agent_logs/events.jsonl` with fields:
     - timestamp, profile, action type, command/path, return code, duration.

2. Add an option `--log-level` or `--debug` in CLI.
3. Optionally, create a small helper CLI subcommand:

   ```bash
   ragix-unix-agent --show-log tail --n 50
   ```

   that prints the last N events.

4. Ensure no secrets are logged:
   - Basic masking on environment variables and known patterns.

**Done when**

- Typical sessions produce meaningful logs.
- The log format is stable enough to be reused by future tools (v0.6 Web UI, v0.7 observability).

---

### 1.6. Task: Minimal retrieval abstraction (v0.5)

**Goal:** Prepare the ground for hybrid retrieval in v0.6, without committing to a heavy vector DB yet.

**Steps**

1. Create `ragix_core/retrieval.py`:

   ```python
   class Retriever(Protocol):
       def index(self, root: Path) -> None: ...
       def query(self, text: str, limit: int = 20) -> list[Result]: ...
   ```

2. Implement a **grep-based retriever**:
   - Use `rg`/`grep` under the hood or Python `subprocess`.
   - Return structured results: file, line number, snippet.
3. Integrate as a tool:
   - `"action": "search_project", "query": "..."` mapped to retriever.
4. Mention in README under “Unix-RAG retrieval loop” that this is the first layer; semantic embeddings will come in v0.6.

**Done when**

- The agent can answer simple questions like “Where is `ragix_mcp_server` defined?” via the retrieval tool.
- The retriever abstraction is simple but ready to be swapped for a semantic backend later.

---

### 1.7. Task: Tests and CHANGELOG update

**Goal:** Provide a minimal test suite and document the v0.5 changes.

**Steps**

1. Add a `tests/` folder with at least:
   - Tests for policy/denylist.
   - Tests for JSON action parsing.
   - Tests for retrieval returning expected hits on a tiny synthetic repo.

2. Update `CHANGELOG.md`:
   - Add a section for `v0.5.0` with “Added / Changed / Fixed” referencing:
     - new package structure,
     - CLI,
     - profiles,
     - retrieval abstraction,
     - logging.

3. Adjust `README.md`:
   - Version badge / version mention → v0.5.
   - Short “Migration from v0.4 to v0.5” note.

**Done when**

- `pytest` (or chosen test runner) can be executed on a fresh clone.
- CHANGELOG reflects these tasks accurately.

---

## 2. Milestone v0.6 – Web UI, multi-agent, hybrid retrieval

For now, only **prepare scaffolding** and TODOs so they don’t block v0.5.

### 2.1. Task: Stub multi-agent orchestration

- In `ragix_core/orchestrator.py`, introduce the concept of **Agents**:

  ```python
  class Agent(Protocol):
      name: str
      tools: list[str]
      def step(self, state): ...
  ```

- Implement a simple registry:
  - `CodeAgent`, `DocAgent`, `GitAgent` as thin wrappers around current tools.
- Keep the current single-agent flow, but allow choosing an agent by name in config.

### 2.2. Task: Prepare Web UI entry point

- Add a `ragix_web/` package (empty skeleton is enough for now):
  - `app.py` or `main.py` with placeholder FastAPI/Flask app.
  - A config option like `--ui` that later will start the web server.
- Add TODOs / docstrings describing the intended future Web UI:
  - Local SPA, WASM shell, multi-session.

### 2.3. Task: Prepare embedding hooks

- Extend `retrieval.py` with a placeholder `EmbeddingsBackend`:

  ```python
  class EmbeddingsBackend(Protocol):
      def embed(self, texts: list[str]) -> list[list[float]]: ...
  ```

- Provide a dummy implementation that raises `NotImplementedError` but is fully wired and documented so future code can plug in easily.

---

## 3. Milestone v0.7 – CI integration, secrets vault & CloakMCP

For now: **integration points and TODO markers**, not implementations.

### 3.1. Task: CI entry hooks

- Add a `ragix_ci/` module with:
  - A CLI subcommand `ragix ci-run --playbook ci_checks.yaml`.
  - Stub functions with TODO comments: “lint”, “doc summary”, etc.
- Document in README a sketch of GitHub/GitLab CI usage, but clearly marked as **experimental / v0.7 target**.

### 3.2. Task: Secret vault & CloakMCP integration points

- Add `ragix_core/secrets.py`:
  - `SecretProvider` protocol.
  - Dummy implementation for now (env vars only).
  - Clearly mark integration point for **CloakMCP**:
    - comments / TODO: “use CloakMCP to cleanse code before sending to LLM”, “use CloakMCP as secret-scrubbing pre-processor in future MCP agent”.

- In `MCP/`, add short comments referencing CloakMCP’s role for:
  - secret scrubbing before tool calls,
  - future credential MCP adapter.

---

## 4. Suggested execution order

If you need a **practical order** to work in multiple sessions:

1. **Session 1–2**
   - 1.1 Core package skeleton.
   - 1.2 CLI entry points & config.
2. **Session 3**
   - 1.3 Profiles & sandbox safety.
3. **Session 4**
   - 1.4 JSON action protocol, 1.5 logging.
4. **Session 5**
   - 1.6 Retrieval abstraction.
5. **Session 6**
   - 1.7 Tests + CHANGELOG + README update.
6. **Session 7+ (lightweight, optional)**
   - 2.1 Agent protocol stubs.
   - 2.2 Web UI placeholder.
   - 2.3 Embedding hooks.
   - 3.1, 3.2 CI & secrets integration points, with TODO comments referencing CloakMCP.

---

