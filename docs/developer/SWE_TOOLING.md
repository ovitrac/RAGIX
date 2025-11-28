# SWE_TOOLING.md — SWE-style Tooling for RAGIX

> Status: draft design for v0.4  
> Scope: navigation, search, editing tooling for large codebases in **Unix-RAG** style, SWE-Agent compatible in spirit, without forcing JSON interaction in the default mode.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-23

---

## 1. Purpose and Scope

This document specifies **SWE-style tooling** for the RAGIX ecosystem, aligned with:

- `unix-rag-agent.py` (Unix-RAG / Ollama local dev assistant),
- the existing shell helpers `rt.sh`, `rt-grep.sh`, `rt-find.sh`,
- the MCP integration in `MCP/`.

The goal is to bring **SWE-Agent-like capabilities** (systematic navigation, search, structured edits) into RAGIX **without** forcing JSON actions in daily use.

Instead, the default interaction pattern stays:

- human + LLM chatting,
- LLM issues **bash commands** (primarily `rt …` helpers) in fenced code blocks,
- RAGIX runs those commands in a sandboxed project directory.

JSON actions (e.g. `{"action": "edit_file", ...}`) remain available for:

- internal use,
- automated evaluation (SWE-Bench, harnesses),
- MCP / tool-calling integration.

They are **not** the primary interface for human users.

---

## 2. Design Principles

### 2.1 Unix-RAG First

RAGIX is a **Unix-RAG development assistant**:

- Retrieval and exploration via classic Unix tools:
  - `find`, `ls`, `tree`, `grep -R -n`, `sed -n`, `awk`, `wc -l`, `python` one-liners, etc.
- Git-aware workflows:
  - `git status -sb`, `git diff`, `git log --oneline --graph`, `git show`, `git grep`.
- High-level file editing via a safe helper (`edit_file` right now).
- Sandboxed shell (`ShellSandbox`) bound to a project root.

SWE tooling must **augment** this style, not replace it. Wherever possible:

- **Tools should look like commands** a senior Unix developer might invent:
  - `rt open path:line`, `rt scroll path +`, `rt edit file start end <<EOF … EOF`, etc.
- The LLM and the human can both read and understand these commands.

### 2.2 SWE-Agent Capabilities, Minus Friction

SWE-Bench / SWE-Agent tools typically include:

- **Navigation**: `open`, `goto`, `scroll_up`, `scroll_down`,
- **Search**: `find_file`, `search_dir`, `search_file`,
- **Edit**: `edit`, `insert`.

We mimic these capabilities in RAGIX via *shell commands* (and optional JSON mapping), with the same behavioral guidelines:

- View **100 lines at a time**, with a small **overlap** between windows.
- Use **direct jumps** (`open path:line` / `goto`) instead of many scrolls.
- Keep edits **small, localized, and diff-friendly**.

### 2.3 Robustness on Large Codebases

RAGIX must stay usable on:

- repositories with **thousands of files**,
- mixed languages,
- deep directory trees.

Guidelines:

- All `rt` commands must:
  - respect the sandbox root,
  - avoid traversing system directories,
  - allow scoping by sub-directory where useful,
  - have sane limits (e.g. max number of matches shown by default).
- Output must be:
  - dense enough for LLM reasoning,
  - but not so verbose that it overwhelms context.

### 2.4 Modularity and Extensibility

The SWE tooling must be:

- **Modular**:
  - each command has a single responsibility,
  - small shell wrappers call Python helpers when needed.
- **Composable**:
  - commands can be chained (`rt find … | rt grep …` if desired),
  - semantics are predictable and documented.
- **Extensible**:
  - easy to add new commands (e.g. `rt symbol`, `rt callgraph`) without changing the core philosophy.

---

## 3. Architectural Overview

### 3.1 Repository Context

RAGIX root (simplified):

```text
.
├── assets/
├── MCP/
│   ├── ragix_mcp_server.py
│   └── ragix_tools_spec.json
├── ragix_tools.py
├── unix-rag-agent.py
├── rt.sh
├── rt-grep.sh
├── rt-find.sh
├── README.md
├── README_RAGIX_TOOLS.md
└── ...
````

### 3.2 Components

* **`unix-rag-agent.py`**

  * Provides `UnixRAGAgent`:

    * `OllamaLLM` client,
    * `ShellSandbox` (subprocess runner, logging, safety),
    * chat history + system prompt (`AGENT_SYSTEM_PROMPT`),
    * JSON `edit_file` action (line-anchored, snippet replacement).
  * Initial “project overview” using `find` and `grep`.

* **Shell helpers (`rt*.sh`)**

  * Provide convenient **Unix-style commands** that the agent can call via `bash` actions.
  * Currently include:

    * `rt.sh` (dispatcher / entrypoint),
    * `rt-grep.sh` (search content),
    * `rt-find.sh` (search files).

* **`ragix_tools.py`**

  * Central place to define Python helpers for:

    * search,
    * file parsing,
    * future MCP-style tools.

* **`MCP/`**

  * MCP server implementation and JSON tool spec for use with Claude Code / Desktop, etc.

### 3.3 Where SWE Tooling Lives

We introduce SWE-style tooling in three layers:

1. **Shell commands** (primary interface)

   * Implemented in `rt.sh` and new helpers (e.g. `rt-open.sh`, `rt-edit.sh`) or as subcommands inside `rt.sh`.
   * Called by the agent via `bash` actions.

2. **Python helpers** (logic + robustness)

   * Implemented in `ragix_tools.py` (navigation, search, edits).
   * Shell commands call into these via `python -m ragix_tools …` or a direct CLI entrypoint.

3. **Agent prompt and behavior**

   * Documented and encouraged in `AGENT_SYSTEM_PROMPT` (inside `unix-rag-agent.py`).
   * Specifies how and when to use SWE commands.

Optionally, a 4th layer:

4. **JSON tool mapping / MCP tools**

   * For benchmarks and MCP integration:

     * each SWE shell command can be mirrored by a JSON tool (`tool: "open_file_chunk", args: {...}`),
     * kept out of the default human-facing flow.

---

## 4. Navigation Commands (SWE-style File Viewing)

### 4.1 Requirements

* View files in **windows of ~100 lines**.
* Provide **line numbers** in output.
* Support:

  * open from top,
  * open around a target line,
  * scroll up/down in fixed steps with a slight overlap.

### 4.2 Proposed Interface

We define the following **shell commands** (as subcommands of `rt.sh`, or separate scripts):

#### 4.2.1 `rt open`

* **Synopsis**

  ```bash
  rt open <path>[:<line>|:<start>-<end>]
  ```

* **Modes**

  * `rt open path`
    → show lines `1–100` of `path`.
  * `rt open path:line`
    → show a 100-line window centered around `line` (e.g. `line-50` to `line+49`, clipped at file boundaries).
  * `rt open path:start-end`
    → show lines `start` to `end` explicitly (with a max enforced window size, e.g. 200 lines).

* **Behavior**

  * Output format:

    ```text
    --- <path> (lines <start>-<end>) ---
    <start>: first line
    <start+1>: second line
    ...
    <end>: last line
    ```

  * Always respect the **project root sandbox**.

  * If file is too short, just show what’s available.

#### 4.2.2 `rt scroll`

* **Synopsis**

  ```bash
  rt scroll <path> [+|-]
  ```

* **Behavior**

  * Uses a small **state file** (e.g. `.ragix_view_state.json` in the sandbox root) to remember the **last viewed window** for each path.
  * A window is defined by:

    * `start_line`,
    * `chunk_size` (default 100),
    * `overlap` (default 2).
  * `rt scroll path +`:

    * new `start_line = previous.start_line + chunk_size - overlap`.
  * `rt scroll path -`:

    * new `start_line = max(1, previous.start_line - chunk_size + overlap)`.
  * If no prior view exists, `rt scroll path +` behaves like `rt open path`.

* **Output**

  Same as `rt open` (header + numbered lines).

#### 4.2.3 LLM Usage Guidelines

In `AGENT_SYSTEM_PROMPT`:

* Encourage the model to:

  * use `rt open path:line` to jump to relevant sections,
  * **avoid repeated `rt scroll`**; prefer direct jumps when a target line is known (e.g. from `grep -n`),
  * never dump entire files.

Examples for the model:

```bash
# Find where "safety_margin" appears
rt-grep.sh "safety_margin" src/

# Suppose result shows: src/sim/gas_flow_model.py:123: ...
# Jump around that line:
rt open src/sim/gas_flow_model.py:123
```

---

## 5. Search Commands (SWE-style Retrieval)

### 5.1 Requirements

* Discover files by **name pattern**.
* Search **directory trees** by content.
* Search **within a single file** with line numbers.
* Scale to large repos; avoid excessive output.

### 5.2 Existing Commands

* `rt-find.sh`

  * Currently wraps `find` or `fd` for name-based search.
* `rt-grep.sh`

  * Currently wraps `grep` / `rg` for content search.

These should be preserved and possibly expanded.

### 5.3 Proposed Interface

#### 5.3.1 `rt find` (file name search)

Implement as:

* either a subcommand `rt find` in `rt.sh`, or reuse `rt-find.sh` with updated semantics.

* **Synopsis**

  ```bash
  rt find <pattern> [<root>]
  ```

* **Semantics**

  * Look for files with names matching `<pattern>` (substring or glob).
  * Default `root = .` (project root).
  * Output:

    ```text
    <relative/path/to/file1>
    <relative/path/to/file2>
    ...
    ```
  * Respect `.gitignore` if possible (e.g. via `fd`).

#### 5.3.2 `rt grep` (recursive search)

* **Synopsis**

  ```bash
  rt grep "<pattern>" [<root>]
  ```

* **Semantics**

  * `pattern` is typically a regex; treat it literally if quoted.
  * Search recursively under `root` (default `.`).
  * Output lines à la `grep -R -n`:

    ```text
    path/to/file.py:123: matching code snippet
    ```
  * Limit max results (e.g. 200–500) by default; show a final “truncated” line if cut.

#### 5.3.3 `rt grep-file` (single file search)

* **Synopsis**

  ```bash
  rt grep-file "<pattern>" <path>
  ```

* **Semantics**

  * Search only within `path`.
  * Output:

    ```text
    <line> : matching code snippet
    ```
  * This is the conceptual equivalent of SWE’s `search_file`.

### 5.4 LLM Usage Guidelines

In the system prompt, encourage:

* **Discovery phase**:

  * Use `rt find` to identify candidate files by name,
  * Then `rt grep` / `rt grep-file` to localize code responsible for a behavior.
* **Navigation phase**:

  * Use results (line numbers) to drive `rt open path:line`.

Example:

```bash
# Step 1: find routes or controllers
rt find "routes" src/

# Step 2: inspect usage of "get_user" in a specific file
rt grep-file "get_user" src/backend/routes.py

# Step 3: open the file around a line of interest
rt open src/backend/routes.py:210
```

---

## 6. Edit Commands (SWE-style Localized Edits)

### 6.1 Requirements

* Perform **precise edits** on line ranges or at specific line positions.
* Be **diff-friendly**:

  * edits should be small and traceable,
  * encourage immediate `git diff` after modifications.
* Avoid messing with file encodings or line endings.

### 6.2 Existing JSON `edit_file` Tool

`unix-rag-agent.py` specifies an action:

```jsonc
{"action": "edit_file", "path": "...", "old": "...", "new": "..."}
```

Semantics:

* open file at `path`,
* replace **first** occurrence of exact snippet `old`,
* with `new`,
* then typically run `git diff -- path`.

This is still useful as a **fallback**, especially for:

* small config file changes,
* simple replacements that do not depend on line numbers.

### 6.3 Proposed Shell-Level Edit Commands

We introduce **two line-based commands**:

#### 6.3.1 `rt edit` (replace a range of lines)

* **Synopsis**

  ```bash
  rt edit <path> <start_line> <end_line> << 'EOF'
  <replacement text, any lines>
  EOF
  ```

* **Semantics**

  * Replace lines `[start_line, end_line]` in `path` with the provided block.
  * Lines are **1-based indexing**.
  * If `end_line < start_line` or out of bounds → error.
  * Implementation should:

    * read file into memory as a list of lines,
    * do range replacement,
    * write back atomically if possible (e.g. via temp file then move).

* **Example**

  ```bash
  rt edit src/sim/gas_flow_model.py 120 135 << 'EOF'
  def compute_safety_margin(flow_rate: float, pressure: float) -> float:
      """
      Recompute safety margins with updated regulation constraints.
      """
      margin = max(0.0, pressure - 1.2 * flow_rate)
      return margin
  EOF
  ```

#### 6.3.2 `rt insert` (insert before a line)

* **Synopsis**

  ```bash
  rt insert <path> <line> << 'EOF'
  <text to insert>
  EOF
  ```

* **Semantics**

  * Insert the given block **before** line `<line>` in `path`.
  * Accept `line = N+1` to append at end (where `N` is number of lines).
  * Error if `line < 1` or `line > N+1`.

* **Example**

  ```bash
  rt insert src/sim/gas_flow_model.py 10 << 'EOF'
  import logging
  logger = logging.getLogger(__name__)
  EOF
  ```

### 6.4 Implementation Notes

* **Where**:

  * Shell interface: new subcommands in `rt.sh` (or dedicated scripts `rt-edit.sh`, `rt-insert.sh`).
  * Logic: call into `ragix_tools.py` (e.g. `python -m ragix_tools edit_range path start end`).

* **Python helpers (ragix_tools.py)**

  Pseudo-API:

  ```python
  def edit_range(path: str, start: int, end: int, new_text: str) -> str:
      """
      Replace lines [start, end] (1-based inclusive) in file `path` with `new_text`.
      Returns a short summary string.
      """

  def insert_at(path: str, line: int, new_text: str) -> str:
      """
      Insert `new_text` before line `line` (1-based) in file `path`.
      Returns a short summary string.
      """
  ```

* **Safety**

  * Work only inside sandbox root,
  * Optionally create a backup (`.bak`) or rely on git for rollback,
  * Log operations in the same structured way as `edit_file`.

### 6.5 LLM Usage Guidelines

In `AGENT_SYSTEM_PROMPT`, encourage:

* **Before editing**:

  * inspect relevant sections with `rt open` / `rt grep-file`,
  * describe verbally what will change.
* **Perform edit** with `rt edit` / `rt insert` (or fallback to JSON `edit_file` for simple “find/replace”).
* **After editing**:

  * run a diff:

    ```bash
    git diff -- src/sim/gas_flow_model.py
    ```
  * or re-open the modified region:

    ```bash
    rt open src/sim/gas_flow_model.py:125
    ```

Example:

```bash
# Inspect
rt open src/sim/gas_flow_model.py:120

# Edit
rt edit src/sim/gas_flow_model.py 120 128 << 'EOF'
def compute_safety_margin(flow_rate: float, pressure: float) -> float:
    """
    Updated safety margin for new specs (2025).
    """
    return max(0.0, pressure - 1.1 * flow_rate)
EOF

# Verify
git diff -- src/sim/gas_flow_model.py
```

---

## 7. Integration Points and Extension “Where”

### 7.1 `unix-rag-agent.py`

**What to extend:**

1. **System prompt (`AGENT_SYSTEM_PROMPT`)**
   Add a section describing SWE tools:

   * `rt open`: open file windows, including `path:line` syntax.
   * `rt scroll`: continue view with +/- windows.
   * `rt find`, `rt grep`, `rt grep-file`: structured search commands.
   * `rt edit`, `rt insert`: line-range and line-insert edits.

   Explicitly document:

   * 100-line window convention,
   * 2-line overlap for scrolling,
   * preference for `rt open path:line` over repeated `rt scroll`.

2. **Edit strategy section**
   Emphasize:

   * inspect → edit → diff → explain loop,
   * when to use `rt edit` / `rt insert` vs JSON `edit_file`.

3. **Optional SWE profile**
   Introduce a command-line flag or env var (e.g. `--profile swe`) to:

   * slightly bias the prompt to mention SWE-style commands first,
   * keep default profile more minimal if desired.

### 7.2 `rt.sh` and Friends

**Where to extend:**

* `rt.sh`:

  * add subcommands:

    * `open`, `scroll`, `find`, `grep`, `grep-file`, `edit`, `insert`,
  * centralize validation and help text.

* `rt-grep.sh` and `rt-find.sh`:

  * either:

    * keep them as internal helpers called by `rt.sh`,
    * or deprecate direct use and re-export them via `rt grep` / `rt find`.

* New helper scripts (optional):

  * `rt-open.sh`: implement file windowing and state tracking.
  * `rt-edit.sh`: implement `edit` / `insert` wrappers around `ragix_tools.py`.

### 7.3 `ragix_tools.py`

**What to add:**

* Functions for:

  * file chunk reading (`open_window(path, start, size)`),
  * view state (if you prefer state in Python rather than shell),
  * line-range editing (`edit_range`, `insert_at`),
  * structured search (if not fully delegated to `grep`/`rg`).

**Design hints:**

* Keep functions pure where possible; state (e.g. last view window) can be managed via:

  * a JSON file in the repo root (e.g. `.ragix_view_state.json`),
  * or environment variables if needed for ephemeral sessions.

### 7.4 MCP / JSON Mapping (Optional)

In `MCP/ragix_tools_spec.json` and `ragix_mcp_server.py`:

* Define tools mirroring the SWE functionality:

  * `open_file_chunk(path, line=None, start=None, end=None)`,
  * `scroll_file(path, direction)`,
  * `find_file(pattern, root)`,
  * `search_dir(pattern, root)`,
  * `search_file(path, pattern)`,
  * `edit_range(path, start_line, end_line, new_text)`,
  * `insert_at(path, line, new_text)`.

* Back them with the same Python helpers as the shell commands.

This keeps:

* **daily use**: text + `bash` + `rt …`,
* **programmatic use**: MCP tools with clearly defined args and JSON.

---

## 8. Robustness and Testing Strategy

### 8.1 Large Repository Scenarios

Test on:

* monorepos with:

  * 10k+ files,
  * nested `src`, `tests`, `tools`, etc.

Check:

* `rt find pattern` returns results quickly and is easily scannable,
* `rt grep` truncates output gracefully (with an indication),
* `rt open` remains responsive for very large files.

### 8.2 Common Task Patterns

Design unit / integration tests for typical SWE tasks:

1. **Bug localization**

   * `rt grep "error message" .`
   * `rt open path:line`
   * verify that the window contains the expected context.

2. **Small feature addition**

   * locate extension point via `rt grep` / `rt find`,
   * `rt insert` new function or route,
   * `rt edit` to wire it into an existing map / registry,
   * `git diff` shows only relevant changes.

3. **Refactoring**

   * rename a function (old name → new name) using `rt edit` in several files,
   * ensure no unintended replacements (validate with `rt grep`).

### 8.3 Regression and Safety

* Confirm all tools respect the sandbox root and do not traverse upwards.
* Confirm that malformed arguments:

  * produce clear errors,
  * do not corrupt files,
  * are easily explainable in the agent’s replies.

---

## 9. Summary

This SWE tooling spec aims to:

* Give RAGIX **SWE-Agent-level capabilities** (navigation, search, edits),
* While preserving its **Unix-RAG character**:

  * the model mostly writes shell commands (`rt open`, `rt grep`, `rt edit`),
  * humans can read, understand, and reproduce those commands.

The central idea:

> *SWE-style tools are expressed as composable `rt` commands and backed by Python helpers; JSON/MCP mappings are an optional outer layer, not the default conversational protocol.*

This ensures:

* robustness on large codebases,
* modular tooling you can extend,
* flexibility to support both:

  * interactive local dev workflows,
  * automated SWE benchmarks and MCP-style integrations.

---

## 10. Approved conventions
Nice, these are exactly the right questions to settle before touching code. Here’s what I’d recommend, with rationale so you can see the trade-offs.

---

### 1. Backward compatibility for `ragix_tools.py`

**Recommendation:**
👉 *Preserve the existing CLI interface exactly* (command names, positional args, main behaviors), and **extend it** rather than refactor/rename.

* Add new SWE-related commands as **new subcommands** or **new options**, but:

  * Don’t rename existing commands.
  * Don’t change the meaning of existing flags/positional parameters.
* If you really need to change something later:

  * Keep a backward-compatible shim (old name still works, prints a short “deprecated, use X instead” note).

This keeps all your existing scripts, notes, and muscle memory valid, and it’s very much in line with your “regulated, auditable” mindset: behavior never silently changes.

---

### 2. `.ragix_view_state.json` behavior

**Recommendation:**
👉 *Persistent per repo* **and git-ignored** — treated as ephemeral state/cache.

Concretely:

* **Persistent:** it survives agent restarts so navigation & SWE context (e.g. last opened file, last line range) can be reused by Claude / RAGIX.
* **Per repo:** store it at repo root (`.ragix_view_state.json`) or under a small namespaced dir like `.ragix/state.json`.
* **Git-ignored:** add it to `.gitignore` so it never pollutes commits or diffs.

Semantics: “safe to delete at any time, but helpful when kept”. That matches how you treat logs and caches.

If you want to be extra clean later, you can support an env var like `RAGIX_VIEW_STATE=off` to disable persistence entirely, but default should be persistent+ignored.

---

### 3. Profile integration for SWE tools

**Recommendation:**
👉 *Available by default in `dev` and `unsafe`, **read-only subset only** in `safe-read-only`*, with a simple env toggle.

* **`dev` profile:**

  * Full SWE toolkit enabled: search, navigation, and editing.
* **`unsafe` profile:**

  * Same as `dev`, but you might allow more powerful commands (e.g. mass edits) if you later decide so.
* **`safe-read-only` profile:**

  * SWE tools that *edit* (`rt edit`, `rt insert`, etc.) should:

    * Either be disabled (fail with a clear message), or
    * Run in strict dry-run mode (show patch but do not apply).
  * SWE tools that only *read* (search, find, view) stay available.

Plus a global kill switch:

* `RAGIX_ENABLE_SWE=0` → disables all SWE tools regardless of profile.
* `RAGIX_ENABLE_SWE=1` (or unset) → normal behavior.

This gives you:

* Safety guarantees for “audit only” sessions.
* A clean story: “editing tools are only allowed where the profile explicitly allows mutations”.

No need for an extra `swe` profile now; that would add complexity without much gain.

---

### 4. Post-edit behavior for `rt edit` / `rt insert`

**Recommendation:**
👉 *By default: return the edited region + a concise success message, and offer an optional flag/env for automatic `git diff`.*

So:

* **Default behavior:**

  * Apply the edit.
  * Print:

    * A short confirmation line:
      `✓ Edited <file>:<start_line>-<end_line>`
    * The **edited region** (a small snippet around the change).
  * Do **not** automatically run `git diff` (avoids noise and keeps it usable in scripts).

* **Optional diff:**

  * `rt edit --show-diff …` or `rt insert --show-diff …` runs:
    `git diff -- <file>` after a successful edit.
  * Or a global env var: `RAGIX_AUTO_DIFF=1` to always show the per-file diff after edits.

Why this way?

* For **Claude / SWE-agent style** workflows, the edited region is usually enough for immediate reasoning.
* For you as a human, when you want full diffs, you opt in via flag or env.
* It avoids spamming large diffs when tools are used in scripts or tight loops.

