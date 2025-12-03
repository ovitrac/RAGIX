# RAGIX v0.30 — Reflective Reasoning Graph (graph_v3)

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-12-03
**Status:** APPROVED FOR IMPLEMENTATION
**Target:** v0.30.0

---

## 0. Changes vs v0.23

This version refines the v0.23 Reflective Reasoning Graph with:

1. **New complexity level:** `BYPASS` ("just answer, no tools / no plan").
2. **Confidence wiring:** optional `confidence` on plan and state, surfaced by nodes.
3. **Strict reflection budget:** per-complexity reflection budget + global `max_iterations`.
4. **Canonical experience corpus layout:** per-session trace + `events.jsonl` at global and project levels.
5. **Unified tool action schema:** `ToolCall` / `ToolResult` for all Unix tools.
6. **Versioned module layout:** `ragix_core/reasoning_v30/` to coexist with previous graph versions.

---

## 1. Graph Overview

### 1.1 High-level structure

```text
USER PROMPT
   │
   ▼
+----------+     +-----------+     +-------------------+
|  START   | --> | CLASSIFY  | --> |       PLAN        |
+----------+     +-----------+     +-------------------+
                      |                     |
                 (BYPASS)                   v
                      v              +-------------+
               +------------+        |   EXECUTE   |<----+
               | DIRECT_EXEC|        +-------------+     |
               +------------+         |    |    |       |
                      |          fail |    | ok |       |
                      v               v    |    v       |
               +---------+     +---------+ | +--------+ |
               | RESPOND |<----| REFLECT |-+ | VERIFY | |
               +---------+     +---------+   +--------+ |
                    |               |             |     |
                    v               | (new plan)  |     |
                 +-----+            +-------------+-----+
                 | END |
                 +-----+
```

**New:** `TaskComplexity.BYPASS` routes `CLASSIFY → DIRECT_EXEC → RESPOND` with *no plan, no tools*.

### 1.2 Key design decisions (v0.30)

| Decision          | Choice                                              | Rationale                                                |
| ----------------- | --------------------------------------------------- | -------------------------------------------------------- |
| New complexity    | `BYPASS`                                            | Avoid tool use when pure conversational answer is enough |
| Confidence        | `Plan.confidence`, `ReasoningState.confidence`      | For future safety / model selection                      |
| Reflection budget | Per-complexity `max_reflections` + `max_iterations` | Prevent loops, bound latency                             |
| Tool protocol     | `ToolCall` / `ToolResult` dataclasses               | Single schema for all Unix tools                         |
| Experience corpus | Canonical layout (session trace + events.jsonl)     | Easier RAG + analytics                                   |
| Versioning        | `reasoning_v30` package                             | Coexist with earlier loop/graph versions                 |

---

## 2. Core Types (`ragix_core/reasoning_v30/types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class TaskComplexity(Enum):
    BYPASS = "bypass"      # no tools, no plan, direct answer
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    """Unified schema for tool invocations."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Unified schema for tool results."""
    tool: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None        # Python exception, if any


@dataclass
class PlanStep:
    num: int
    description: str
    tool_call: Optional[ToolCall] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None


@dataclass
class Plan:
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    validation: str = ""
    confidence: Optional[float] = None   # 0.0–1.0 (LLM self-estimate)

    def get_current_step(self, index: int) -> Optional[PlanStep]:
        return self.steps[index] if 0 <= index < len(self.steps) else None

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
                   for s in self.steps)


@dataclass
class ReflectionAttempt:
    timestamp: str
    failed_step: int
    error: str
    diagnosis: str
    new_plan_summary: str


@dataclass
class ReasoningState:
    goal: str
    session_id: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    plan: Optional[Plan] = None
    current_step_index: int = 0
    last_error: Optional[str] = None
    reflection_count: int = 0
    reflection_attempts: List[ReflectionAttempt] = field(default_factory=list)
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None   # "success", "max_reflections", "max_iterations", "no_plan", "bypass"
    confidence: Optional[float] = None  # State-level confidence for the overall answer


@dataclass
class ReasoningEvent:
    timestamp: str
    session_id: str
    event_type: Literal["planning", "execution", "reflection", "verification", "respond"]
    goal: str
    step_num: Optional[int] = None
    step_description: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[str] = None
    outcome_status: Optional[str] = None  # "success", "failure"
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    llm_critique: Optional[str] = None
    context_used: Optional[str] = None   # RAG / experience context
    meta: Dict[str, Any] = field(default_factory=dict)
```

---

## 3. Graph Orchestrator (`reasoning_v30/graph.py`)

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

from .types import ReasoningState


class BaseNode(ABC):
    name: str

    @abstractmethod
    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        ...


class ReasoningGraph:
    def __init__(self, nodes: Dict[str, BaseNode], start: str = "CLASSIFY", end: str = "END"):
        self.nodes = nodes
        self.start = start
        self.end = end
        self.trace: List[str] = []

    def run(self, state: ReasoningState, max_iterations: int = 50) -> ReasoningState:
        current = self.start
        iterations = 0

        while current != self.end and iterations < max_iterations:
            self.trace.append(current)
            node = self.nodes[current]
            state, next_node = node.run(state)
            current = next_node
            iterations += 1

        if iterations >= max_iterations and not state.stop_reason:
            state.stop_reason = "max_iterations"

        return state
```

---

## 4. Node Implementations (`reasoning_v30/nodes.py`)

```python
from datetime import datetime
from typing import Callable, Dict, Tuple

from .types import (
    ReasoningState,
    TaskComplexity,
    StepStatus,
    Plan,
    PlanStep,
    ToolResult,
    ReasoningEvent,
    ReflectionAttempt,
)
from .graph import BaseNode
from .experience import HybridExperienceCorpus


class ClassifyNode(BaseNode):
    name = "CLASSIFY"

    def __init__(self, classify_fn: Callable[[str], TaskComplexity]):
        self.classify_fn = classify_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        state.complexity = self.classify_fn(state.goal)

        if state.complexity == TaskComplexity.BYPASS:
            state.stop_reason = "bypass"
            return state, "DIRECT_EXEC"

        if state.complexity == TaskComplexity.SIMPLE:
            return state, "DIRECT_EXEC"

        return state, "PLAN"


class DirectExecNode(BaseNode):
    """
    For BYPASS and SIMPLE tasks:
    - BYPASS → answer directly without tools
    - SIMPLE → optionally allow a single tool call in the LLM prompt but no Plan object
    """
    name = "DIRECT_EXEC"

    def __init__(self, llm_answer_fn: Callable[[str], Dict[str, str]]):
        """
        llm_answer_fn(goal) -> {"answer": str, "confidence": float (0..1)}
        """
        self.llm_answer_fn = llm_answer_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        res = self.llm_answer_fn(state.goal)
        state.final_answer = res.get("answer", "")
        state.confidence = res.get("confidence")  # may be None
        if not state.stop_reason:
            state.stop_reason = "success"
        return state, "RESPOND"


class PlanNode(BaseNode):
    name = "PLAN"

    def __init__(self, generate_plan_fn: Callable[[str, str], Dict], parse_plan_fn: Callable[[Dict, str], Plan]):
        """
        generate_plan_fn(goal, reflection_context) -> raw LLM JSON/dict
        parse_plan_fn(raw, goal) -> Plan
        """
        self.generate_plan_fn = generate_plan_fn
        self.parse_plan_fn = parse_plan_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        reflection_context = ""
        if state.reflection_attempts:
            reflection_context = "\n\nPrevious attempts:\n" + "\n".join(
                f"- {att.diagnosis}" for att in state.reflection_attempts
            )

        raw = self.generate_plan_fn(state.goal, reflection_context)
        plan = self.parse_plan_fn(raw, state.goal)
        state.plan = plan
        state.current_step_index = 0
        state.confidence = plan.confidence
        return state, "EXECUTE"


class ExecuteNode(BaseNode):
    name = "EXECUTE"

    def __init__(self,
                 execute_step_fn: Callable[[PlanStep, ReasoningState], PlanStep],
                 max_reflections_by_complexity: Dict[TaskComplexity, int]):
        self.execute_step_fn = execute_step_fn
        self.max_reflections = max_reflections_by_complexity

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if state.plan is None:
            state.stop_reason = "no_plan"
            return state, "RESPOND"

        step = state.plan.get_current_step(state.current_step_index)
        if step is None:
            # Plan complete
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            state.stop_reason = "success"
            return state, "RESPOND"

        step = self.execute_step_fn(step, state)

        if step.status == StepStatus.FAILED:
            max_refl = self.max_reflections.get(state.complexity, 0)
            if state.reflection_count < max_refl:
                return state, "REFLECT"
            state.stop_reason = "max_reflections"
            return state, "RESPOND"

        # Next step
        state.current_step_index += 1

        if state.plan.is_complete():
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            state.stop_reason = "success"
            return state, "RESPOND"

        return state, "EXECUTE"


class ReflectNode(BaseNode):
    name = "REFLECT"

    ALLOWED_TOOLS = ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd"]

    def __init__(self,
                 llm_reflect_fn: Callable[[Dict], Dict],
                 experience_corpus: HybridExperienceCorpus,
                 shell_executor: Callable[[str], ToolResult]):
        self.llm_reflect_fn = llm_reflect_fn
        self.experience = experience_corpus
        self.shell = shell_executor

    def _safe_shell(self, command: str) -> ToolResult:
        cmd = (command or "").split()[0]
        if cmd not in self.ALLOWED_TOOLS:
            return ToolResult(tool=cmd, returncode=1, stderr=f"[BLOCKED: {cmd} not allowed]")
        return self.shell(command)

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if not state.plan:
            state.stop_reason = "no_plan"
            return state, "RESPOND"

        step = state.plan.get_current_step(state.current_step_index)

        # 1) Context via read-only tools
        context_commands = [
            "pwd",
            "ls -la",
            "find . -maxdepth 2 -type f -name '*.py' | head -20",
        ]
        file_context = "\n".join(
            f"$ {cmd}\n{self._safe_shell(cmd).stdout}"
            for cmd in context_commands
        )

        # 2) Experience
        query = f"goal: {state.goal}\nerror: {state.last_error}"
        experience_context = self.experience.search(query, top_k=5)

        # 3) Build reflection prompt payload
        payload = {
            "goal": state.goal,
            "failed_step_num": step.num if step else None,
            "failed_step_description": step.description if step else None,
            "error": state.last_error,
            "file_context": file_context,
            "experience_context": experience_context,
            "previous_attempts": [
                {
                    "failed_step": a.failed_step,
                    "error": a.error,
                    "diagnosis": a.diagnosis,
                    "new_plan_summary": a.new_plan_summary,
                }
                for a in state.reflection_attempts
            ],
        }

        res = self.llm_reflect_fn(payload)
        diagnosis = res.get("diagnosis", "")
        new_plan_summary = res.get("new_plan_summary", "")

        state.reflection_count += 1
        state.reflection_attempts.append(
            ReflectionAttempt(
                timestamp=datetime.utcnow().isoformat(),
                failed_step=step.num if step else -1,
                error=state.last_error or "",
                diagnosis=diagnosis,
                new_plan_summary=new_plan_summary,
            )
        )

        # Hand off to PLAN with enriched reflection context
        return state, "PLAN"


class VerifyNode(BaseNode):
    name = "VERIFY"

    def __init__(self, llm_verify_fn: Callable[[ReasoningState], Dict[str, str]]):
        """
        llm_verify_fn(state) -> {"answer": str, "confidence": float}
        """
        self.llm_verify_fn = llm_verify_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        res = self.llm_verify_fn(state)
        state.final_answer = res.get("answer", state.final_answer)
        if res.get("confidence") is not None:
            state.confidence = res["confidence"]
        state.stop_reason = state.stop_reason or "success"
        return state, "RESPOND"


class RespondNode(BaseNode):
    name = "RESPOND"

    def __init__(self, emit_event_fn: Callable[[ReasoningEvent], None]):
        self.emit_event = emit_event_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        # Final corpus event
        ev = ReasoningEvent(
            timestamp=datetime.utcnow().isoformat(),
            session_id=state.session_id,
            event_type="respond",
            goal=state.goal,
            outcome_status=state.stop_reason,
            llm_critique=None,
            context_used=None,
            meta={"confidence": state.confidence},
        )
        self.emit_event(ev)
        return state, "END"
```

---

## 5. Experience Corpus (`reasoning_v30/experience.py`)

### 5.1 Canonical layout

```text
~/.ragix/
  experience/
    events.jsonl        # global
    traces/
      {session_id}.jsonl

.ragix/
  experience/
    events.jsonl        # project-specific
  reasoning_traces/
    {session_id}.jsonl
```

### 5.2 Implementation

```python
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

from .types import ReasoningEvent


class ExperienceCorpus:
    def __init__(self, root: Path, max_age_days: int = 90):
        self.root = root
        self.events_path = self.root / "experience" / "events.jsonl"
        self.max_age = timedelta(days=max_age_days)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: ReasoningEvent) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def _load_recent(self) -> Iterable[dict]:
        if not self.events_path.exists():
            return []
        cutoff = datetime.utcnow() - self.max_age
        out: List[dict] = []
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ts = datetime.fromisoformat(obj.get("timestamp"))
                    if ts >= cutoff:
                        out.append(obj)
                except Exception:
                    continue
        return out

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Very simple keyword-based retrieval for now.
        Returns a text block suitable for LLM context.
        """
        query_l = query.lower()
        candidates = []
        for ev in self._load_recent():
            blob = " ".join(
                [
                    ev.get("goal", ""),
                    ev.get("step_description", "") or "",
                    ev.get("error", "") or "",
                    ev.get("llm_critique", "") or "",
                ]
            ).lower()
            score = sum(1 for token in query_l.split() if token in blob)
            if score > 0:
                candidates.append((score, ev))
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = [ev for _, ev in candidates[:top_k]]

        lines = []
        for i, ev in enumerate(selected, 1):
            lines.append(f"[{i}] {ev.get('timestamp')} — goal={ev.get('goal')}")
            if ev.get("error"):
                lines.append(f"    error: {ev['error']}")
            if ev.get("llm_critique"):
                lines.append(f"    critique: {ev['llm_critique']}")
        return "\n".join(lines)


class HybridExperienceCorpus:
    def __init__(self, global_root: Path, project_root: Path,
                 global_max_age_days: int = 90,
                 project_max_age_days: int = 30):
        self.global_corpus = ExperienceCorpus(global_root, max_age_days=global_max_age_days)
        self.project_corpus = ExperienceCorpus(project_root, max_age_days=project_max_age_days)

    def append(self, event: ReasoningEvent, project: bool = True, global_: bool = True) -> None:
        if project:
            self.project_corpus.append(event)
        if global_:
            self.global_corpus.append(event)

    def search(self, query: str, top_k: int = 5) -> str:
        g = self.global_corpus.search(query, top_k=top_k)
        p = self.project_corpus.search(query, top_k=top_k)
        out = []
        if p:
            out.append("### Project experience\n" + p)
        if g:
            out.append("### Global experience\n" + g)
        return "\n\n".join(out)
```

---

## 6. Configuration (`ragix.yaml` section)

```yaml
reasoning:
  strategy: "graph_v30"       # "loop_v1" | "graph_v2" | "graph_v30"

  max_reflections:
    bypass: 0
    simple: 0
    moderate: 1
    complex: 3

  graph:
    max_iterations: 50

  reflect:
    allowed_tools: ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd"]
    max_context_chars: 2000

  experience:
    global_root: "~/.ragix"
    project_root: ".ragix"
    global_max_age_days: 90
    project_max_age_days: 30
    top_k: 5

  traces:
    enabled: true
    path: ".ragix/reasoning_traces"
    max_per_session: 1000
```

---

## 7. Agent Profiles Matrix

| Profile   | Tools Allowed | Models Allowed | Reflection | Memory |
| --------- | ------------- | -------------- | ---------- | ------ |
| safe      | none          | cloud only     | simple     | off    |
| dev       | read + write  | cloud/local    | full       | on     |
| sovereign | full          | local only     | full       | on     |

---

## 8. Tests (`tests/reasoning_v30/`)

### 8.1 Folder skeleton

```text
tests/reasoning_v30/
├── __init__.py
├── harness.py
├── fixtures/
│   └── mock_repo/
│       ├── a.py
│       └── b.py
├── scenarios/
│   ├── file_search.yaml
│   ├── code_analysis.yaml
│   └── bypass_question.yaml
└── test_reasoning_graph_v30.py
```

### 8.2 Example scenario YAML

```yaml
# tests/reasoning_v30/scenarios/file_search.yaml
- id: largest-md-file
  description: Find largest markdown file by line count
  input: "Find the largest markdown file in this repo by line count"
  expected_patterns:
    - "largest"
    - ".md"
    - "lines"
  must_run_commands:
    - "find"
    - "wc"
  max_steps: 5
  complexity: MODERATE

# tests/reasoning_v30/scenarios/bypass_question.yaml
- id: conceptual-question
  description: Pure reasoning, no tools
  input: "Explain the difference between cyclomatic complexity and cognitive complexity."
  expected_patterns:
    - "cyclomatic"
    - "cognitive"
    - "control flow"
  must_run_commands: []
  max_steps: 1
  complexity: BYPASS
```

### 8.3 Minimal test

```python
# tests/reasoning_v30/test_reasoning_graph_v30.py

from pathlib import Path

from ragix_core.reasoning_v30.graph import ReasoningGraph
from ragix_core.reasoning_v30.nodes import (
    ClassifyNode,
    DirectExecNode,
    PlanNode,
    ExecuteNode,
    ReflectNode,
    VerifyNode,
    RespondNode,
)
from ragix_core.reasoning_v30.types import ReasoningState, TaskComplexity
from ragix_core.reasoning_v30.experience import HybridExperienceCorpus


def dummy_classify(goal: str) -> TaskComplexity:
    if "Explain the difference" in goal:
        return TaskComplexity.BYPASS
    if "largest markdown file" in goal:
        return TaskComplexity.MODERATE
    return TaskComplexity.SIMPLE


def test_bypass_flow(tmp_path: Path):
    state = ReasoningState(
        goal="Explain the difference between cyclomatic complexity and cognitive complexity.",
        session_id="test-bypass"
    )

    experience = HybridExperienceCorpus(
        global_root=tmp_path / "global",
        project_root=tmp_path / "project"
    )

    nodes = {
        "CLASSIFY": ClassifyNode(dummy_classify),
        "DIRECT_EXEC": DirectExecNode(lambda goal: {"answer": "dummy", "confidence": 0.8}),
        "PLAN": PlanNode(lambda g, ctx: {}, lambda raw, g: None),
        "EXECUTE": ExecuteNode(lambda step, st: step, {}),
        "REFLECT": ReflectNode(lambda payload: {}, experience, lambda cmd: None),
        "VERIFY": VerifyNode(lambda st: {"answer": st.final_answer, "confidence": st.confidence or 0.5}),
        "RESPOND": RespondNode(lambda ev: None),
        "END": RespondNode(lambda ev: None),
    }

    graph = ReasoningGraph(nodes=nodes, start="CLASSIFY", end="END")
    final_state = graph.run(state)

    assert final_state.final_answer == "dummy"
    assert final_state.stop_reason == "success"
```

---

## 9. Node Prompt Drafts (LLM-side)

### 9.1 CLASSIFY (TaskComplexity)

```text
You are a task complexity classifier for a Unix-native code assistant (RAGIX).

Decide how much structure and tooling is needed to solve the user's request.

Complexity levels:

- BYPASS: Pure reasoning / explanation / summarization. No tools, no plan.
- SIMPLE: 1–2 shell commands or a single file edit, no reflections.
- MODERATE: 2–4 steps with shell commands and/or multiple edits, may need 1 reflection.
- COMPLEX: Multi-step investigation across files and tools, with possible failures and up to 3 reflections.

User goal:
---
{{goal}}
---

Answer with exactly one token:
BYPASS, SIMPLE, MODERATE, or COMPLEX.
```

### 9.2 PLAN

```text
You are a planning agent for RAGIX, a Unix-native reasoning assistant.

Goal:
{{goal}}

Previous attempts (if any):
{{reflection_context}}

Produce a step-by-step plan to achieve the goal using Unix tools and minimal file edits.

Rules:
- 2–4 steps for MODERATE tasks, 3–7 for COMPLEX.
- Each step must have:
  - num: integer step number starting from 1
  - description: natural language description of the action
  - tool: name of the main Unix tool to use (e.g. "find", "grep", "sed", "python", "none")
  - args: brief JSON object describing key arguments (path, pattern, etc.)
- If a step does not require a tool, use tool="none" and args={}.

Also output:
- validation: a short sentence describing how we will know the task is done.
- confidence: a number between 0 and 1 for how confident you are that this plan will work.

Respond in JSON with keys: objective, steps, validation, confidence.
```

### 9.3 REFLECT

```text
You are a reflection agent for RAGIX.

The previous execution failed.

Goal:
{{goal}}

Failed step {{failed_step_num}}: {{failed_step_description}}

Error:
{{error}}

Current directory structure and files (truncated):
{{file_context}}

Similar past experiences:
{{experience_context}}

Previous attempts:
{{previous_attempts}}

Your tasks (maximum 3 bullet points each):

1. **Diagnose** precisely what went wrong.
2. **Explain** why the original approach is fragile or incorrect.
3. **Propose** a revised plan at a high level (3–5 bullet summary).

Return JSON with:
- diagnosis: short paragraph
- new_plan_summary: bullet-point text describing the revised strategy
```

### 9.4 VERIFY

```text
You are a verification agent for RAGIX.

You will:
1. Read the goal.
2. Inspect the executed steps and their outputs.
3. Decide if the answer is correct, coherent, and safe.
4. Optionally refine the final answer.

Goal:
{{goal}}

Executed plan:
{{plan_and_steps}}

Current final answer:
{{current_answer}}

If the answer is correct but can be better structured, rewrite it.
If there are gaps or potential issues, mention them explicitly and give a cautious answer.

Return JSON:
{
  "answer": "...",
  "confidence": 0.0–1.0
}
```

### 9.5 DIRECT_EXEC answer

```text
You are a conversational expert for RAGIX.

The user is asking a question that does NOT require running tools or editing files.

Goal:
{{goal}}

Provide a clear, concise answer in one or two paragraphs, plus bullet points if helpful.

Also estimate your confidence (0.0–1.0). High confidence if the topic is standard and unambiguous.

Return JSON:
{
  "answer": "...",
  "confidence": 0.0–1.0
}
```

---

## 10. Folder Layout Summary for v0.30

```text
ragix_core/
├── reasoning.py                  # legacy loop / adapter
├── reasoning_v30/
│   ├── __init__.py
│   ├── types.py
│   ├── graph.py
│   ├── nodes.py
│   ├── experience.py
│   └── config.py                 # optional loader from ragix.yaml

tests/
└── reasoning_v30/
    ├── __init__.py
    ├── harness.py
    ├── fixtures/
    │   └── mock_repo/
    ├── scenarios/
    │   ├── file_search.yaml
    │   ├── code_analysis.yaml
    │   └── bypass_question.yaml
    └── test_reasoning_graph_v30.py
```

---

## 11. Migration Path

| Version | Changes |
|---------|---------|
| v0.23.x | Current - Model inheritance, Web UI fixes |
| v0.30.0 | Add `graph_v30` with BYPASS, confidence, unified tools |
| v0.31.0 | Stabilize, add evaluation harness, dry-run preview |
| v0.32.0 | Default to `graph_v30`, deprecate `loop_v1` |
| v1.0.0 | Remove legacy, advanced features (tool autodiscovery, multi-agent) |

---

## 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Plan success rate | >80% | Scenarios passing without reflection |
| Recovery rate | >60% | Failed plans recovered by REFLECT |
| Max reflections hit | <10% | Tasks exhausting reflection budget |
| Avg steps per task | <6 | Efficiency measure |
| BYPASS accuracy | >90% | Correctly classified pure-reasoning tasks |

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| REFLECT loops infinitely | Hard cap on reflections + iteration limit |
| Experience corpus grows unbounded | TTL-based pruning (30/90 days) |
| Read-only tools misused | Strict allowlist, no write operations |
| Graph adds latency | Profile, optimize hot paths |
| BYPASS misclassification | Test harness, feedback loop |

---

*Plan prepared for RAGIX v0.30 implementation.*
*Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio*
