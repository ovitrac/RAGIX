# RAGIX v0.23 Implementation Plan: Reflective Reasoning Graph

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-12-02
**Status:** APPROVED FOR IMPLEMENTATION
**Target:** v0.23.0

---

## Executive Summary

This plan transforms RAGIX's reasoning system from a linear Planner/Worker/Verifier loop into a **Reflective Reasoning Graph** that learns from its own experience. Key innovations:

1. **Graph-based reasoning** with explicit state transitions
2. **REFLECT node** with read-only tool access (not just plan repair)
3. **Hybrid experience corpus** (global `~/.ragix/` + project `.ragix/`)
4. **Graceful degradation** with attempt summaries when max_reflections reached

---

## 1. Architecture Overview

### 1.1 Graph Structure

```
                 +-----------------+
                 |   USER PROMPT   |
                 +-----------------+
                        |
                        v
+----------+     +-----------+     +-------------------+
|  START   | --> | CLASSIFY  | --> |       PLAN        |
+----------+     +-----------+     +-------------------+
                      |                     |
                      | (SIMPLE)            v
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

### 1.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| REFLECT tool access | **Read-only tools allowed** | Needs `ls`, `find`, `grep` to understand context before replanning |
| Experience corpus | **Hybrid (global + project)** | Like CLAUDE.md: global patterns + project-specific history |
| Max reflections | **Configurable per complexity + attempt summary** | SIMPLE=0, MODERATE=1, COMPLEX=2-3; summary on exhaustion |
| Graph vs Loop | **Graph wraps Loop methods** | Reuse existing code, no duplication |

---

## 2. Implementation Phases

### Phase 1: Foundation (Schema + Hardening)

**Files:** `ragix_core/reasoning.py`, `ragix_core/reasoning_types.py` (new)

#### 2.1.1 Define Core Types

```python
# ragix_core/reasoning_types.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime

class TaskComplexity(Enum):
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
class PlanStep:
    num: int
    description: str
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None

@dataclass
class Plan:
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    validation: str = ""

    def get_current_step(self, index: int) -> Optional[PlanStep]:
        return self.steps[index] if index < len(self.steps) else None

    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED) for s in self.steps)

@dataclass
class ReflectionAttempt:
    """Record of a single reflection attempt."""
    timestamp: str
    failed_step: int
    error: str
    diagnosis: str
    new_plan_summary: str

@dataclass
class ReasoningState:
    """State object passed between graph nodes."""
    goal: str
    session_id: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    plan: Optional[Plan] = None
    current_step_index: int = 0
    last_error: Optional[str] = None
    reflection_count: int = 0
    reflection_attempts: List[ReflectionAttempt] = field(default_factory=list)
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None  # "success", "max_reflections", "user_abort"

@dataclass
class ReasoningEvent:
    """Unified event schema for experience corpus."""
    timestamp: str
    session_id: str
    event_type: Literal["planning", "execution", "reflection", "verification"]
    goal: str
    step_description: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[str] = None
    outcome_status: Optional[str] = None  # "success", "failure"
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    llm_critique: Optional[str] = None
    context_used: Optional[str] = None  # RAG context if REFLECT
```

#### 2.1.2 Harden execute_step

```python
# In ragix_core/reasoning.py - modify execute_step

def execute_step(self, step: PlanStep, state: ReasoningState) -> PlanStep:
    """Execute a single plan step with full error capture."""
    step.status = StepStatus.RUNNING

    try:
        result = self.execute_fn(step.description)

        # Parse structured result if available
        if isinstance(result, dict):
            step.returncode = result.get("returncode", 0)
            step.result = result.get("stdout", str(result))
            step.error = result.get("stderr", "")
        else:
            step.returncode = 0
            step.result = str(result)

        # Determine success based on returncode
        if step.returncode == 0:
            step.status = StepStatus.SUCCESS
        else:
            step.status = StepStatus.FAILED
            state.last_error = f"Step {step.num} failed (rc={step.returncode}): {step.error}"

    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.returncode = -1
        state.last_error = f"Step {step.num} exception: {e}"

    # Emit event for experience corpus
    self._emit_event(ReasoningEvent(
        timestamp=datetime.utcnow().isoformat(),
        session_id=state.session_id,
        event_type="execution",
        goal=state.goal,
        step_description=step.description,
        tool=step.tool,
        tool_input=str(step.args) if step.args else None,
        outcome_status="success" if step.status == StepStatus.SUCCESS else "failure",
        stdout=step.result,
        stderr=step.error,
        returncode=step.returncode,
        error=state.last_error
    ))

    return step
```

---

### Phase 2: Reasoning Graph

**Files:** `ragix_core/reasoning_graph.py` (new)

#### 2.2.1 Base Node Class

```python
# ragix_core/reasoning_graph.py

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Dict, Any

class BaseNode(ABC):
    """Base class for reasoning graph nodes."""

    name: str

    @abstractmethod
    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """
        Execute node logic.

        Returns:
            Tuple of (updated_state, next_node_name)
        """
        pass

class ReasoningGraph:
    """Orchestrates reasoning as a directed graph of nodes."""

    def __init__(
        self,
        nodes: Dict[str, BaseNode],
        start: str = "CLASSIFY",
        end: str = "END"
    ):
        self.nodes = nodes
        self.start = start
        self.end = end
        self.trace: List[str] = []

    def run(self, state: ReasoningState, max_iterations: int = 50) -> ReasoningState:
        """Execute the graph until END or max_iterations."""
        current = self.start
        iterations = 0

        while current != self.end and iterations < max_iterations:
            self.trace.append(current)
            node = self.nodes[current]
            state, next_node = node.run(state)
            current = next_node
            iterations += 1

        if iterations >= max_iterations:
            state.stop_reason = "max_iterations"

        return state
```

#### 2.2.2 Node Implementations

```python
class ClassifyNode(BaseNode):
    name = "CLASSIFY"

    def __init__(self, classify_fn: Callable):
        self.classify_fn = classify_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        state.complexity = self.classify_fn(state.goal)

        if state.complexity == TaskComplexity.SIMPLE:
            return state, "DIRECT_EXEC"
        return state, "PLAN"

class PlanNode(BaseNode):
    name = "PLAN"

    def __init__(self, generate_plan_fn: Callable, parse_plan_fn: Callable):
        self.generate_plan_fn = generate_plan_fn
        self.parse_plan_fn = parse_plan_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        # Include reflection context if replanning
        context = ""
        if state.reflection_attempts:
            context = "\n\nPrevious attempts:\n"
            for att in state.reflection_attempts:
                context += f"- {att.diagnosis}\n"

        plan_response = self.generate_plan_fn(state.goal + context)
        state.plan = self.parse_plan_fn(plan_response, state.goal)
        state.current_step_index = 0

        return state, "EXECUTE"

class ExecuteNode(BaseNode):
    name = "EXECUTE"

    def __init__(
        self,
        execute_step_fn: Callable,
        max_reflections_by_complexity: Dict[TaskComplexity, int]
    ):
        self.execute_step_fn = execute_step_fn
        self.max_reflections = max_reflections_by_complexity

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if state.plan is None:
            state.stop_reason = "no_plan"
            return state, "RESPOND"

        # Execute current step
        step = state.plan.get_current_step(state.current_step_index)
        if step is None:
            # Plan complete
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            return state, "RESPOND"

        step = self.execute_step_fn(step, state)

        if step.status == StepStatus.FAILED:
            max_refl = self.max_reflections.get(state.complexity, 1)
            if state.reflection_count < max_refl:
                return state, "REFLECT"
            else:
                # Max reflections reached - graceful degradation
                state.stop_reason = "max_reflections"
                return state, "RESPOND"

        # Move to next step
        state.current_step_index += 1

        # Check if plan complete
        if state.plan.is_complete():
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            return state, "RESPOND"

        # Continue execution
        return state, "EXECUTE"
```

---

### Phase 3: REFLECT Node with Tool Access

**Key innovation:** REFLECT can use read-only tools to understand context.

```python
class ReflectNode(BaseNode):
    """
    Reflection node that can:
    1. Use read-only tools to understand file structure
    2. Query experience corpus for similar past failures
    3. Generate improved plan with diagnosis
    """
    name = "REFLECT"

    ALLOWED_TOOLS = ["ls", "find", "grep", "head", "tail", "wc", "cat"]

    def __init__(
        self,
        llm_generate: Callable,
        experience_corpus: 'HybridExperienceCorpus',
        shell_executor: Callable,
        episodic_memory: 'EpisodicMemory'
    ):
        self.llm_generate = llm_generate
        self.experience = experience_corpus
        self.shell = shell_executor
        self.episodic = episodic_memory

    def _safe_shell(self, command: str) -> str:
        """Execute only read-only commands."""
        cmd_name = command.split()[0] if command else ""
        if cmd_name not in self.ALLOWED_TOOLS:
            return f"[BLOCKED: {cmd_name} not in allowed tools]"
        return self.shell(command)

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        failed_step = state.plan.get_current_step(state.current_step_index)

        # 1. Gather context with read-only tools
        context_commands = [
            "ls -la",
            "find . -maxdepth 2 -type f -name '*.py' | head -20",
            "pwd"
        ]
        file_context = "\n".join([
            f"$ {cmd}\n{self._safe_shell(cmd)}"
            for cmd in context_commands
        ])

        # 2. Query experience corpus (hybrid: global + project)
        query = f"goal: {state.goal}, error: {state.last_error}"
        experience_context = self.experience.search(query, top_k=5)

        # 3. Get episodic context
        episodic_context = self.episodic.get_current_session_context()

        # 4. Build reflection prompt
        reflection_prompt = f"""## Reflection Required

**Goal:** {state.goal}

**Failed Step {failed_step.num}:** {failed_step.description}

**Error:**
```
{state.last_error}
```

**Current Directory Structure:**
```
{file_context}
```

**Similar Past Experiences:**
{experience_context}

**Session Context:**
{episodic_context}

## Your Task

1. **Diagnose** what went wrong (be specific)
2. **Explain** why the original approach failed
3. **Generate** a new [PLAN] that avoids this failure

Consider:
- File paths and existence
- Command syntax for this OS
- Alternative approaches if the tool is unavailable
- Lessons from past similar failures
"""

        # 5. Generate new plan
        response = self.llm_generate(
            get_agent_persona(AgentRole.PLANNER),
            [{"role": "user", "content": reflection_prompt}]
        )

        # 6. Extract diagnosis and new plan
        diagnosis = self._extract_diagnosis(response)
        new_plan = self._parse_plan(response, state.goal)

        # 7. Record reflection attempt
        state.reflection_attempts.append(ReflectionAttempt(
            timestamp=datetime.utcnow().isoformat(),
            failed_step=failed_step.num,
            error=state.last_error or "",
            diagnosis=diagnosis,
            new_plan_summary=f"{len(new_plan.steps)} steps"
        ))

        # 8. Update state
        state.plan = new_plan
        state.current_step_index = 0
        state.last_error = None
        state.reflection_count += 1

        # 9. Emit reflection event
        self._emit_event(ReasoningEvent(
            timestamp=datetime.utcnow().isoformat(),
            session_id=state.session_id,
            event_type="reflection",
            goal=state.goal,
            error=str(state.last_error),
            llm_critique=diagnosis,
            context_used=experience_context[:500]  # Truncate for storage
        ))

        return state, "EXECUTE"
```

---

### Phase 4: Hybrid Experience Corpus

**Files:** `ragix_core/experience_corpus.py` (new)

```python
# ragix_core/experience_corpus.py

from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime, timedelta

class ExperienceCorpus:
    """Single experience corpus (either global or project)."""

    def __init__(self, events_path: Path, max_age_days: int = 30):
        self.events_path = events_path
        self.max_age_days = max_age_days
        self._events: List[ReasoningEvent] = []
        self._load()

    def _load(self):
        if not self.events_path.exists():
            return

        cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)

        with open(self.events_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    event_time = datetime.fromisoformat(data.get("timestamp", ""))
                    if event_time > cutoff:
                        self._events.append(ReasoningEvent(**data))
                except (json.JSONDecodeError, ValueError):
                    continue

    def search(self, query: str, top_k: int = 5) -> List[ReasoningEvent]:
        """Simple keyword + recency search."""
        query_terms = set(query.lower().split())

        scored = []
        for event in self._events:
            # Keyword score
            event_text = f"{event.goal} {event.step_description} {event.error} {event.llm_critique}".lower()
            keyword_score = len(query_terms & set(event_text.split()))

            # Recency score (0-1, higher = more recent)
            try:
                event_time = datetime.fromisoformat(event.timestamp)
                age_days = (datetime.utcnow() - event_time).days
                recency_score = max(0, 1 - age_days / self.max_age_days)
            except:
                recency_score = 0

            # Success bonus (prefer successful recoveries)
            success_bonus = 1.5 if event.outcome_status == "success" else 1.0

            total_score = (keyword_score * 2 + recency_score) * success_bonus
            scored.append((total_score, event))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


class HybridExperienceCorpus:
    """
    Combines global (~/.ragix/experience/) and project (.ragix/experience/) corpora.

    Similar to CLAUDE.md pattern:
    - Global: General patterns, cross-project learnings
    - Project: Project-specific history and context
    """

    GLOBAL_PATH = Path.home() / ".ragix" / "experience" / "events.jsonl"

    def __init__(self, project_path: Optional[Path] = None):
        self.global_corpus = ExperienceCorpus(self.GLOBAL_PATH, max_age_days=90)

        if project_path:
            project_events = project_path / ".ragix" / "experience" / "events.jsonl"
            self.project_corpus = ExperienceCorpus(project_events, max_age_days=30)
        else:
            self.project_corpus = None

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search both corpora with project results prioritized.

        Returns formatted context string for LLM.
        """
        results = []

        # Project-specific (higher priority, more recent)
        if self.project_corpus:
            project_results = self.project_corpus.search(query, top_k=3)
            for event in project_results:
                results.append(("PROJECT", event))

        # Global patterns
        global_results = self.global_corpus.search(query, top_k=3)
        for event in global_results:
            results.append(("GLOBAL", event))

        # Format for LLM
        if not results:
            return "[No relevant past experiences found]"

        formatted = []
        for source, event in results[:top_k]:
            formatted.append(f"""
[{source}] {event.timestamp[:10]}
Goal: {event.goal}
Step: {event.step_description}
Outcome: {event.outcome_status}
{f'Lesson: {event.llm_critique}' if event.llm_critique else ''}
""".strip())

        return "\n\n".join(formatted)

    def append(self, event: ReasoningEvent, to_global: bool = False):
        """Append event to appropriate corpus."""
        path = self.GLOBAL_PATH if to_global else (
            self.project_corpus.events_path if self.project_corpus else self.GLOBAL_PATH
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            f.write(json.dumps(event.__dict__) + "\n")
```

---

### Phase 5: Graceful Degradation with Attempt Summary

When `max_reflections` is reached, provide a useful summary:

```python
class RespondNode(BaseNode):
    name = "RESPOND"

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if state.stop_reason == "max_reflections":
            # Build attempt summary
            summary = self._build_attempt_summary(state)
            state.final_answer = f"""## Partial Results

I attempted to complete your request but encountered persistent issues after {state.reflection_count} recovery attempts.

### What Was Tried

{summary}

### Recommendation

{self._generate_recommendation(state)}

### Partial Output

{self._collect_partial_results(state)}
"""
        else:
            state.final_answer = self._build_success_response(state)

        state.stop_reason = state.stop_reason or "success"
        return state, "END"

    def _build_attempt_summary(self, state: ReasoningState) -> str:
        lines = []
        for i, attempt in enumerate(state.reflection_attempts, 1):
            lines.append(f"""
**Attempt {i}** (Step {attempt.failed_step})
- Error: {attempt.error[:100]}...
- Diagnosis: {attempt.diagnosis[:200]}...
""")
        return "\n".join(lines)

    def _generate_recommendation(self, state: ReasoningState) -> str:
        # Analyze failure patterns
        errors = [a.error for a in state.reflection_attempts]

        if any("not found" in e.lower() for e in errors):
            return "Some required files or commands may not exist. Please verify the project structure."
        if any("permission" in e.lower() for e in errors):
            return "There may be permission issues. Try running with appropriate access."

        return "Consider breaking this task into smaller steps or providing more specific guidance."
```

---

### Phase 6: Evaluation Harness

**Files:** `tests/reasoning/harness.py`, `tests/reasoning/scenarios/`

```yaml
# tests/reasoning/scenarios/file_search.yaml
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

- id: grep-pattern
  description: Search for pattern across files
  input: "Find all files containing 'def execute'"
  expected_patterns:
    - "execute"
    - ".py"
  must_run_commands:
    - "grep"
  max_steps: 3
  complexity: SIMPLE
```

---

## 3. File Structure for v0.23

```
ragix_core/
├── reasoning.py              # Existing - hardened
├── reasoning_types.py        # NEW - Pydantic schemas
├── reasoning_graph.py        # NEW - Graph + nodes
├── reasoning_nodes.py        # NEW - Node implementations
├── experience_corpus.py      # NEW - Hybrid corpus
└── __init__.py               # Export new classes

tests/reasoning/
├── __init__.py
├── harness.py                # Evaluation runner
├── scenarios/
│   ├── file_search.yaml
│   ├── code_analysis.yaml
│   └── multi_step.yaml
├── fixtures/
│   └── mock_repo/            # Test fixture
└── test_reasoning_graph.py

~/.ragix/
├── experience/
│   └── events.jsonl          # Global experience corpus
├── cache/                    # Existing
└── config.yaml               # Existing

.ragix/  (per-project)
├── experience/
│   └── events.jsonl          # Project experience corpus
├── reasoning_traces/
│   └── {session_id}.jsonl    # Debug traces
└── cache/                    # Existing
```

---

## 4. Configuration

```python
# In AgentConfig or ragix.yaml

reasoning:
  strategy: "graph_v2"  # "loop_v1" | "graph_v2"

  max_reflections:
    simple: 0
    moderate: 1
    complex: 3

  reflect:
    allowed_tools: ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd"]
    max_context_chars: 2000

  experience:
    global_max_age_days: 90
    project_max_age_days: 30
    top_k: 5

  traces:
    enabled: true
    path: ".ragix/reasoning_traces"
    max_per_session: 1000
```

---

## 5. Migration Path

| Version | Changes |
|---------|---------|
| v0.22.x | Current - loop-based reasoning |
| v0.23.0 | Add `graph_v2` behind feature flag, default `loop_v1` |
| v0.23.1 | Stabilize, add evaluation harness |
| v0.24.0 | Default to `graph_v2`, deprecate `loop_v1` |
| v0.25.0 | Remove `loop_v1`, advanced features (tool reliability, curricula) |

---

## 6. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Plan success rate | >80% | Scenarios passing without reflection |
| Recovery rate | >60% | Failed plans recovered by REFLECT |
| Max reflections hit | <10% | Tasks exhausting reflection budget |
| Avg steps per task | <6 | Efficiency measure |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| REFLECT loops infinitely | Hard cap on reflections + iteration limit |
| Experience corpus grows unbounded | TTL-based pruning (30/90 days) |
| Read-only tools misused | Strict allowlist, no write operations |
| Graph adds latency | Profile, optimize hot paths |

---

*Plan prepared for RAGIX v0.23 implementation.*
*Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio*
