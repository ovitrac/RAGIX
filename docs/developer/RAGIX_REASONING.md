# RAGIX – Reasoning & Agents Guide (Planner / Worker / Verifier)

## 0. Purpose

This document defines how Claude (or any LLM backend) must **reason inside RAGIX**:

- Use a **Planner Agent** to structure complex tasks.
- Use a **Worker Agent** to execute steps with Unix/RAG tools.
- Use a **Verifier Agent** to check results and enforce rules.
- Maintain an **episodic memory** (what happened) and a **knowledge memory** (rules, conventions).
- Always use a **clear [PLAN] template** for multi-step tasks.

This is **behavioral guidance for the LLM**, not a Python API spec.

---

## 1. Global Roles & Philosophy

You operate as an orchestrator for a **sovereign Unix-RAG bot**:

- You **do not** have direct internet access.
- You **do** have:
  - Access to local files and repositories via RAGIX tools.
  - A set of **rules and conventions** stored in `rules/` (knowledge memory).
  - A log of past episodes in `.ragix/` (episodic memory).
- You are expected to be:
  - Systematic
  - Traceable
  - Minimalist in context usage
  - Compatible with Unix workflows (small, composable steps).

You must **not** try to “do everything in one shot” for non-trivial tasks.  
You must use **plan → execute → verify → summarize**.

---

## 2. Agent Modes

RAGIX conceptually exposes three “modes” of reasoning. In practice this may be done with different model configs, prompts, or profiles.

### 2.1 Planner Agent

**Goal**: Turn a vague or complex user request into a **concrete, structured [PLAN]**.

**When to use**:

- Any task involving multiple files.
- Any task involving persistent changes (code, docs, config).
- Any task with several deliverables (README, CHANGELOG, MCP spec, etc.).

**Output format** (mandatory):

```text
[PLAN]
1. Define the objective.
2. Identify required data.
3. List steps.
4. Solve steps one by one.
5. Validate.
6. Summarize.

You must fill each point **with task-specific content**, not just repeat the generic wording.

**Example (short)** – extending RAGIX SWE tooling:

```text
[PLAN]
1. Define the objective.
   - Extend RAGIX with planner/worker/verifier instructions and SWE tooling rules.
2. Identify required data.
   - Files: README.md, SWE_TOOLING.md, MCP/ragix_mcp_server.py, ragix_tools.py, CHANGELOG.md.
   - Rules: existing coding conventions in rules/RAGIX_RULES.md (if present).
3. List steps.
   - 3.1 Inspect current SWE tools documentation and CLI.
   - 3.2 Design agent pattern (planner/worker/verifier) for large codebases.
   - 3.3 Specify episodic + knowledge memory structures.
   - 3.4 Update documentation and examples.
   - 3.5 Propose tests / validation steps.
4. Solve steps one by one.
   - Use Worker Agent to read/edit files and craft concrete content.
5. Validate.
   - Use Verifier Agent to check alignment with rules and user constraints.
6. Summarize.
   - Provide final summary + explicit next actions for maintainers.
```

------

### 2.2 Worker Agent

**Goal**: Execute each step of the `[PLAN]` using available tools and files.

**Behavior**:

For each plan step:

1. **Restate** the step briefly.
2. **Decide** what is needed:
   - File read/write,
   - Search (rt-grep / rt-find / ripgrep),
   - Tool execution,
   - Local reasoning (design text, code, spec).
3. **Perform** the action (or specify the exact tool call / command).
4. **Log** the important decisions in an episodic entry (see §3.1).

You must:

- Keep actions **atomic** (small, verifiable changes).
- Explicitly mention where you are in the plan:
  - e.g. `Step 3/6 – Listing steps and prioritizing edits…`
- Avoid “hidden jumps”: always say when you move from one step to another.

------

### 2.3 Verifier Agent

**Goal**: Check that the result is:

- Coherent with the original user request.
- Consistent with RAGIX rules and conventions.
- Technically sound at a reasonable level (given no direct execution).

**Checks**:

- Did we address **all points** from `[PLAN]`?
- Did we respect the relevant **rules** in `rules/*.md`?
- For code:
  - Any obvious syntax or structural issues?
  - Any missing updates (e.g. README, CHANGELOG, tests)?
- For docs:
  - Required sections present?
  - Clarity, structure, and consistency?

**Output format**:

```text
[VERIFY]
- Scope checked: ...
- Conformity to rules: ...
- Potential issues / uncertainties: ...
- Recommended follow-up: ...
```

You must be honest about uncertainties (e.g. “Tests not run”, “Runtime behavior unverified”).

------

## 3. Memory Systems

### 3.1 Episodic Memory – Execution History

**Purpose**: Remember *what we did* across sessions to support longer reasoning and continuity.

**Storage**:

- File: `.ragix/episodic_log.jsonl` (or similar).
- Format: one JSON object per line, for example:

```json
{
  "task_id": "2025-11-28-RAGIX-SWE-001",
  "timestamp": "2025-11-28T09:30:00Z",
  "user_goal": "Extend RAGIX with planner/worker/verifier reasoning.",
  "plan_summary": "Defined agent roles, memory structures, and doc updates.",
  "key_decisions": [
    "Created CLAUDE_RAGIX_REASONING.md.",
    "Introduced episodic_log.jsonl and rules directory."
  ],
  "files_touched": [
    "CLAUDE_RAGIX_REASONING.md",
    "rules/RAGIX_RULES.md"
  ],
  "open_questions": [
    "How to expose agent modes in MCP config?",
    "Which tests to automate first?"
  ],
  "result_summary": "Drafted reasoning guide and proposed file structure."
}
```

**Lifecycle**: `write → compress → store → retrieve → inject`

1. **Write**

   - After non-trivial tasks, produce a concise JSON summary (as above).

2. **Compress**

   - If your internal content is long, compress to a short paragraph + bullets before storing.

3. **Store**

   - Append to the log file via the appropriate RAGIX tool / script.

4. **Retrieve**

   - At the start of a new task, search episodic memory by:
     - file names,
     - tags / keywords,
     - task type (e.g. “MCP”, “SWE”, “DOC”).

5. **Inject**

   - Provide a short contextual block in the prompt:

   ```text
   [EPISODIC CONTEXT]
   - Last RAGIX SWE change: ...
   - Files previously modified: ...
   - Open questions: ...
   ```

   **Never** dump the full log; only inject a compact synthesis relevant to the task.

------

### 3.2 Knowledge Memory – Rule-Based Space

**Purpose**: Store **stable, reusable knowledge**, such as:

- Coding and documentation conventions.
- Naming rules.
- Versioning policies.
- MCP / SWE tooling guidelines.
- Domain-specific standards.

**Location** (example):

- `rules/RAGIX_RULES.md`
- `rules/SWE_TOOLING.md`
- `rules/SEMANTICS.md`
- Additional rule files as needed.

**Structure** (suggested):

- Use headings and tags:

```markdown
## MCP Conventions #MCP

- All MCP servers must expose a `health` tool.
- Config files must be described in README.md.

## SWE Tools for Large Codebases #SWE

- Prefer read-only tools by default.
- Provide clear commands for 'find', 'open', 'edit', 'diff'.
```

**Behavior**:

- At planning time, look for relevant rule files and sections (e.g. by tag `#MCP`, `#SWE`, `#DOC`).
- Quote or paraphrase key rules in your `[PLAN]` and `[VERIFY]`.
- If you detect missing rules, suggest adding them to the appropriate `rules/*.md` file.

------

## 4. The [PLAN] Template – Mandatory for Complex Tasks

For any non-trivial request, you **must** start with a filled-in `[PLAN]`:

```text
[PLAN]
1. Define the objective.
2. Identify required data.
3. List steps.
4. Solve steps one by one.
5. Validate.
6. Summarize.
```

You must adapt each line to the specific task, e.g.:

```text
[PLAN]
1. Define the objective.
   - Clarify how to integrate planner/worker/verifier roles into RAGIX docs and config.
2. Identify required data.
   - Files: CLAUDE.md (if present), README.md, MCP/ragix_mcp_server.py, rules/*.md.
   - Prior knowledge: guidelines already recorded in RAGIX_RULES.md.
3. List steps.
   - 3.1 Review current documentation and tooling.
   - 3.2 Design agent responsibilities and message formats.
   - 3.3 Define episodic and knowledge memory locations and formats.
   - 3.4 Draft or update documentation files.
   - 3.5 Propose minimal tests or usage examples.
4. Solve steps one by one.
   - Use Worker Agent logic, calling RAGIX tools as needed.
5. Validate.
   - Verifier Agent checks alignment with rules and identifies remaining risks.
6. Summarize.
   - Provide a short recap, files changed, and recommended next actions.
```

------

## 5. Recommended Implementation Path in RAGIX

From the perspective of the **maintainer**, here is how to implement this spec in RAGIX:

1. **Define State Structures**
   - Implement helpers for:
     - `[PLAN]` objects (simple dict).
     - Episodic entries (JSON schema as in §3.1).
     - Knowledge rules metadata (optional index).
2. **Install Episodic Memory**
   - Create `.ragix/episodic_log.jsonl`.
   - Provide CLI or MCP tools:
     - `ragix-log-episode`
     - `ragix-search-episodes <query>`
3. **Structure Knowledge Memory**
   - Create a `rules/` directory in the repo.
   - Seed with:
     - `rules/RAGIX_RULES.md`
     - `rules/SWE_TOOLING.md`
     - `rules/SEMANTICS.md`
   - Add tags (#MCP, #SWE, #DOC, etc.) for searchability.
4. **Expose Agent Modes**
   - In MCP config or RAGIX profiles, define:
     - `planner-mode` (short, high-level, generates `[PLAN]`).
     - `worker-mode` (more verbose, executes steps).
     - `verifier-mode` (concise, critical, outputs `[VERIFY]`).
5. **Wire the Reasoning Loop**
   - For complex user requests:
     1. Call **Planner Agent** → produce `[PLAN]`.
     2. Retrieve relevant **knowledge memory** and inject.
     3. Call **Worker Agent** step-by-step, logging episodes.
     4. Call **Verifier Agent** → produce `[VERIFY]`.
     5. Store episodic entry and return final summary to user.
6. **Create Canonical Examples**
   - Example 1: Code refactoring across multiple files.
   - Example 2: Extending MCP server with a new tool.
   - Example 3: Updating docs (README + CHANGELOG + rules file).

Each example should:

- Show the `[PLAN]`,
- Show worker reasoning for at least 2 steps,
- Show a `[VERIFY]` block,
- Show the episodic log entry created.

------

## 6. Summary for the Model

When you (Claude or another LLM) run inside RAGIX, remember:

1. **Always plan first** for complex tasks → produce a detailed `[PLAN]`.
2. **Use tools**, do not rely purely on internal reasoning for codebase tasks.
3. **Maintain memory**:
   - Episodic: what was done.
   - Knowledge: what rules apply.
4. **Execute step-by-step** with a Worker mindset.
5. **Verify explicitly** with a `[VERIFY]` block.
6. **Summarize clearly** and indicate next possible actions.

This pattern is mandatory to ensure traceable, reproducible, and robust reasoning in a sovereign Unix-RAG environment.



---

## 7.  Recommendation

- Draft a **matching `SWE_TOOLING.md`** aligned with this reasoning spec, or  
- Propose a **minimal `.ragix/` layout + helper CLI commands** (names + behavior) to make this operational.





---



# Best LLM for Planer/Verifier/worker agents - discussion between 3B and 7B models

## 🟥 FIRST: What a 3B *cannot* do

Let’s clear the illusion:

A 3B model **cannot**:

- run deep implicit chain-of-thought
- solve multi-step problems in a single shot
- manipulate long contexts (beyond 4–8K reliably)
- do abstract reasoning without scaffolding
- self-correct complex outputs
- carry implicit memory over long sequences

**BUT** this does *not* mean it cannot “think”.
 It means: **it cannot think alone.**

A 3B needs **structure**, not intuition.

------

## 🟩 THEN: What a 3B *can do extremely well*

With good scaffolding, a 3B can:

### ✔ 1. Follow strict templates

If you provide:

- step instructions
- IR (intermediate representation)
- controlled vocabulary
   It will execute reliably.

### ✔ 2. Break down simple tasks *when guided*

If you enforce a `[PLAN]` template, a 3B can produce:

- objective
- required data
- substeps
- validations

But it won’t invent it by itself.

### ✔ 3. Execute Worker Agent steps

A small model is **very good** at:

- reading files
- locating functions
- extracting text
- summarizing a diff
- creating precise micro-instructions

This is why Claude Code integrates *small* models for file navigation.

### ✔ 4. Perform Verifier Agent checks

Especially when constrained:

- “Check only structure”
- “Check only consistency”
- “Check if all steps were addressed”

3B models are excellent deterministic validators.

### ✔ 5. Operate with external memory

A 3B becomes a “small deterministic processor” if you provide:

- succinct rules from knowledge memory
- structured episodic context
- 1–2 facts per step (not more)

### ✔ 6. Generate tool calls reliably

3B excels at:

- filling JSON schemas
- producing CLI commands
- composing Unix pipelines
- generating MCP calls

This is precisely what you want in RAGIX.

------

## 🟦 AND: What changes a 3B into a **thinking system**

The model itself doesn’t “reason deeply”.
 But the **system** does, if:

### 1. You enforce a Planner–Worker–Verifier architecture

The Planner can even be a *bigger* model; the Worker and Verifier can be 3B.

Typical configuration:

| Agent    | Recommended size  |
| -------- | ----------------- |
| Planner  | 7B–14B or R1-tiny |
| Worker   | 3B                |
| Verifier | 3B                |

Result:
 The heavy model produces the creative structure; the light models execute reproducibly.

------

### 2. You use external reasoning primitives

A 3B with:

- symbolic IR
- rule-based knowledge
- episodic memory
- file tools
- calculators
- search/grep tools
- domain kernels (SFPPy, sig2dna…)

→ **This is where the actual reasoning happens.**

The 3B is the **glue agent**, not the brain.

------

### 3. You force explicit chain-of-thought

Small models can’t maintain *implicit* reasoning.
 But if you force a step-by-step pattern:

```
[THINK]
Step 1: ...
Step 2: ...
Step 3: ...
```

You basically *externalize* the chain of thought.

This is how synthetic agents using 1–3B models (EVALPLUS, E2B) achieve high performance on code tasks.

------

### 🟧 Practical: How Granite 3B performs vs Mistral 7B / Mixtral / Llama

### Granite 3B strengths:

- deterministic output
- low hallucination when rules are strict
- excellent tool-use reliability
- cheap → can spawn many agents concurrently
- perfect as Worker / Verifier

### Weaknesses:

- poor general reasoning without scaffolding
- low context retention
- weak implicit chain of thought
- limited problem-solving creativity

### The trick:

Use Granite 3B *inside a thinking architecture*, not as the thinking core.

------

## 🟩 REAL-WORLD AGENT ARCHITECTURES USING SMALL MODELS

Several production systems already do this:

### Claude Code

Uses small specialized models for:

- file search
- diff analysis
- command generation
   The heavy model is the planner.

### Voyager / SWE-agent-style systems

Workers are **tiny** models.
 The planning and self-reflection are delegated to larger LLMs.

### Vale / SmolAgents

Use 1.3B to 3B models for:

- tool choice
- action formatting
- step execution

------

## 🟦  RAGIX on Unix

A 3B model inside a **Unix agent stack** is *ideal*:

- Worker:
   Executes atomic UNIX + RAG instructions.
- Verifier:
   Checks structure, steps, coherence.
- Episodic memory:
   Gives continuity.
- Knowledge memory:
   Gives context rules, IR, conventions.

**This amplifies the 3B beyond its natural cognitive bandwidth.**

This is exactly how your GS / RAGIX architecture is supposed to operate.

------

## 🟩 Final answer

### 👉 **YES** — a Granite 3B *can* be very useful for “thinking”,

### but only inside a **planned, memory-augmented, multi-agent architecture**.

If you try to use it as a standalone reasoning model, it will fail.
 If you integrate it as a **Worker/Verifier** in the Planner–Worker–Verifier loop with external memory and rules:

### → It becomes a *surprisingly strong reasoning engine*.

### → Very stable.

### → Very deterministic.

### → Perfect for sovereign on-device or self-hosted RAG.

------

## 8. Agent LLM Configuration Specification

**Updated:** 2025-11-28

### 8.1 Agent-to-Model Assignment

Each agent can be configured independently with a specific LLM. The configuration supports two modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Minimal** | All agents use 3B model | Low VRAM (≤8GB), CPU-only, fast response |
| **Strict** | Planner uses ≥7B, Worker/Verifier use ≥3B | Better planning quality, adequate VRAM (≥12GB) |

### 8.2 Agent Assignment Tables

#### Minimal Mode (Default) — 8GB VRAM / CPU

| Agent | Model | Size | Rationale |
|-------|-------|------|-----------|
| **Planner** | `granite3.1-moe:3b` | 2.0 GB | Fast, deterministic, structured output |
| **Worker** | `granite3.1-moe:3b` | 2.0 GB | Excellent tool-use reliability |
| **Verifier** | `granite3.1-moe:3b` | 2.0 GB | Low hallucination with strict rules |

**Total VRAM:** ~2.5 GB (single model instance, shared)
**Speed:** ★★★★★ (sub-second on CPU)

#### Strict Mode — 12GB+ VRAM

| Agent | Model | Size | Rationale |
|-------|-------|------|-----------|
| **Planner** | `mistral:latest` or `dolphin-mistral:7b` | 4.1–4.4 GB | Better creative structuring |
| **Worker** | `granite3.1-moe:3b` | 2.0 GB | Fast execution, reliable tool calls |
| **Verifier** | `granite3.1-moe:3b` | 2.0 GB | Deterministic validation |

**Total VRAM:** ~6.5 GB (two model instances)
**Speed:** ★★★☆☆ (planner slower, worker/verifier fast)

#### Advanced Mode — 16GB+ VRAM

| Agent | Model | Size | Rationale |
|-------|-------|------|-----------|
| **Planner** | `deepseek-r1:14b` | 9.0 GB | Deep reasoning, complex plans |
| **Worker** | `mistral:latest` | 4.4 GB | Strong execution with context |
| **Verifier** | `granite3.1-moe:3b` | 2.0 GB | Fast, deterministic checks |

**Total VRAM:** ~15.5 GB (three model instances)

### 8.3 Installed Models Detection

Current system models (auto-detected via `ollama list`):

| Model | Size | Category | Recommended For |
|-------|------|----------|-----------------|
| `granite3.1-moe:3b` | 2.0 GB | 3B | Worker, Verifier, Minimal Planner |
| `dolphin-mistral:7b-v2.6-dpo-laser` | 4.1 GB | 7B | Strict Planner |
| `mistral:latest` | 4.4 GB | 7B | Strict Planner, Advanced Worker |
| `llama3:latest` | 4.7 GB | 8B | Alternative Planner |
| `deepseek-r1:14b` | 9.0 GB | 14B | Advanced Planner (reasoning-focused) |

### 8.4 Configuration File Format

```yaml
# ragix.yaml - Agent LLM Configuration
agents:
  # Mode switch: "minimal" | "strict" | "custom"
  mode: minimal

  # Strict mode enforces: planner ≥7B, worker/verifier ≥3B
  strict_enforcement: false

  # Per-agent model assignment (used when mode: custom)
  models:
    planner: granite3.1-moe:3b
    worker: granite3.1-moe:3b
    verifier: granite3.1-moe:3b

  # Model size requirements (for strict mode validation)
  size_requirements:
    planner_min_params: 7b   # 7 billion minimum
    worker_min_params: 3b    # 3 billion minimum
    verifier_min_params: 3b  # 3 billion minimum

  # Fallback if assigned model unavailable
  fallback_model: granite3.1-moe:3b

# Hardware profile (auto-detected or manual)
hardware:
  vram_gb: 8
  prefer_cpu: false
  max_concurrent_models: 1
```

### 8.5 UI Toggle Specification

The web interface should provide a simple toggle:

```
┌─────────────────────────────────────────────────────────┐
│ Agent LLM Mode                                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ○ Minimal (3B for all)    ● Strict (7B Planner)      │
│                                                         │
│   Current Assignment:                                   │
│   ┌─────────┬──────────────────────┬────────┐          │
│   │ Agent   │ Model                │ Status │          │
│   ├─────────┼──────────────────────┼────────┤          │
│   │ Planner │ granite3.1-moe:3b    │ ✓ OK   │          │
│   │ Worker  │ granite3.1-moe:3b    │ ✓ OK   │          │
│   │ Verifier│ granite3.1-moe:3b    │ ✓ OK   │          │
│   └─────────┴──────────────────────┴────────┘          │
│                                                         │
│   [Advanced: Custom Assignment...]                      │
└─────────────────────────────────────────────────────────┘
```

### 8.6 Implementation Requirements

#### Python API

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class AgentMode(Enum):
    MINIMAL = "minimal"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class AgentConfig:
    mode: AgentMode = AgentMode.MINIMAL
    planner_model: str = "granite3.1-moe:3b"
    worker_model: str = "granite3.1-moe:3b"
    verifier_model: str = "granite3.1-moe:3b"
    strict_enforcement: bool = False

    def get_model(self, agent: str) -> str:
        """Get model for agent, applying mode rules."""
        if self.mode == AgentMode.MINIMAL:
            return "granite3.1-moe:3b"
        elif self.mode == AgentMode.STRICT:
            if agent == "planner":
                return self.planner_model or "mistral:latest"
            return "granite3.1-moe:3b"
        return getattr(self, f"{agent}_model", "granite3.1-moe:3b")

    @classmethod
    def detect_optimal(cls, vram_gb: int) -> "AgentConfig":
        """Auto-configure based on available VRAM."""
        if vram_gb <= 8:
            return cls(mode=AgentMode.MINIMAL)
        elif vram_gb <= 12:
            return cls(mode=AgentMode.STRICT, planner_model="mistral:latest")
        else:
            return cls(
                mode=AgentMode.CUSTOM,
                planner_model="deepseek-r1:14b",
                worker_model="mistral:latest",
                verifier_model="granite3.1-moe:3b"
            )
```

#### Model Validation

```python
def validate_model_size(model: str, min_params: str) -> bool:
    """Check if model meets minimum parameter requirement."""
    size_map = {
        "granite3.1-moe:3b": 3,
        "dolphin-mistral:7b": 7,
        "mistral:latest": 7,
        "llama3:latest": 8,
        "deepseek-r1:14b": 14,
    }
    min_val = int(min_params.replace("b", ""))
    return size_map.get(model, 0) >= min_val
```

### 8.7 Granite 3B Persona (Optimized for Worker/Verifier)

When using Granite 3B as Worker or Verifier, prepend this system prompt:

```text
You are a RAGIX Agent operating in {role} mode.

CRITICAL CONSTRAINTS:
- You MUST follow the provided [PLAN] exactly
- You MUST output valid JSON for tool calls
- You MUST NOT invent information outside provided context
- You MUST complete one atomic step at a time

OUTPUT FORMAT:
{format_spec}

AVAILABLE TOOLS:
{tools_list}

Execute the assigned step with precision. No creativity. No improvisation.
```

### 8.8 Recommendations Summary

| Hardware | Mode | Why |
|----------|------|-----|
| Laptop ≤8GB VRAM | **Minimal** | Granite 3B is fast, deterministic, fits in memory |
| Workstation 12GB | **Strict** | Better plans from 7B, fast workers from 3B |
| Server 16GB+ | **Advanced** | Full reasoning stack with DeepSeek-R1 |

**Default recommendation for development:** `Minimal` mode with `granite3.1-moe:3b` for all agents.

---

## Conclusion: Assignment table

The configuration file should enable different LLMs according to the considered agent. Recommendations should be proposed.

- ✅ An exact **agent assignment table** for 3B, 7B, 14B (see §8.2)
- ✅ A **RAGIX multi-model routing strategy** (see §8.4, §8.6)
- ✅ A **Granite 3B persona** optimized for Worker and Verifier roles (see §8.7)
- ⏳ Benchmarks / reasoning patterns to stress-test 3B thinking abilities (future work)