"""
RAGIX Reasoning Types - Pydantic schemas for the Reflective Reasoning Graph

This module defines the core data types for:
- ReasoningState: State object passed between graph nodes
- PlanStep / Plan: Structured plan representation
- ReasoningEvent: Unified event schema for experience corpus
- ReflectionAttempt: Record of reflection attempts

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-02
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
import json


class TaskComplexity(Enum):
    """Task complexity levels determining reasoning strategy."""
    BYPASS = "bypass"      # Pure conceptual/conversational, no tools needed
    SIMPLE = "simple"      # Single command, no planning needed
    MODERATE = "moderate"  # 2-3 steps, brief planning
    COMPLEX = "complex"    # Multi-step, full [PLAN] required


class StepStatus(Enum):
    """Execution status for a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StopReason(Enum):
    """Reason for reasoning loop termination."""
    SUCCESS = "success"
    MAX_REFLECTIONS = "max_reflections"
    MAX_ITERATIONS = "max_iterations"
    USER_ABORT = "user_abort"
    NO_PLAN = "no_plan"
    ERROR = "error"


@dataclass
class PlanStep:
    """
    Single step in a reasoning plan.

    Attributes:
        num: Step number (1-indexed)
        description: Human-readable description of what to do
        tool: Optional tool name (e.g., "bash", "grep", "find")
        args: Optional tool arguments
        status: Current execution status
        result: Output from successful execution
        error: Error message if failed
        returncode: Shell return code if applicable
        stdout: Standard output
        stderr: Standard error
    """
    num: int
    description: str
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num": self.num,
            "description": self.description,
            "tool": self.tool,
            "args": self.args,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        data = data.copy()
        if "status" in data and isinstance(data["status"], str):
            data["status"] = StepStatus(data["status"])
        return cls(**data)


@dataclass
class Plan:
    """
    Structured plan for task execution.

    Attributes:
        objective: High-level goal
        steps: Ordered list of steps to execute
        validation: Criteria for success verification
        required_data: Data needed for execution
    """
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    validation: str = ""
    required_data: List[str] = field(default_factory=list)

    def get_current_step(self, index: int) -> Optional[PlanStep]:
        """Get step at given index, or None if out of bounds."""
        return self.steps[index] if 0 <= index < len(self.steps) else None

    def get_step_by_num(self, num: int) -> Optional[PlanStep]:
        """Get step by its number (1-indexed)."""
        for step in self.steps:
            if step.num == num:
                return step
        return None

    def is_complete(self) -> bool:
        """Check if all steps are completed or skipped."""
        return all(
            s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
            for s in self.steps
        )

    def has_failures(self) -> bool:
        """Check if any step has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def completed_count(self) -> int:
        """Count of successfully completed steps."""
        return sum(1 for s in self.steps if s.status == StepStatus.SUCCESS)

    def to_text(self) -> str:
        """Convert plan to human-readable [PLAN] format."""
        lines = ["[PLAN]"]
        lines.append(f"Objective: {self.objective}")
        if self.required_data:
            lines.append(f"Required: {', '.join(self.required_data)}")
        lines.append("Steps:")

        status_icons = {
            StepStatus.PENDING: "[ ]",
            StepStatus.RUNNING: "[>]",
            StepStatus.SUCCESS: "[x]",
            StepStatus.FAILED: "[!]",
            StepStatus.SKIPPED: "[-]",
        }

        for step in self.steps:
            icon = status_icons.get(step.status, "[ ]")
            lines.append(f"  {step.num}. {icon} {step.description}")

        if self.validation:
            lines.append(f"Validation: {self.validation}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "validation": self.validation,
            "required_data": self.required_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        data = data.copy()
        if "steps" in data:
            data["steps"] = [PlanStep.from_dict(s) for s in data["steps"]]
        return cls(**data)


@dataclass
class ReflectionAttempt:
    """
    Record of a single reflection attempt.

    Captured when REFLECT node generates a new plan after failure.
    """
    timestamp: str
    failed_step_num: int
    failed_step_description: str
    error: str
    diagnosis: str
    new_plan_summary: str
    context_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "failed_step_num": self.failed_step_num,
            "failed_step_description": self.failed_step_description,
            "error": self.error,
            "diagnosis": self.diagnosis,
            "new_plan_summary": self.new_plan_summary,
            "context_used": self.context_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionAttempt":
        return cls(**data)


@dataclass
class ReasoningState:
    """
    State object passed between reasoning graph nodes.

    This is the central data structure that flows through the graph,
    accumulating results and tracking progress.
    """
    goal: str
    session_id: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    plan: Optional[Plan] = None
    current_step_index: int = 0
    last_error: Optional[str] = None
    reflection_count: int = 0
    reflection_attempts: List[ReflectionAttempt] = field(default_factory=list)
    final_answer: Optional[str] = None
    stop_reason: Optional[StopReason] = None

    # Configuration (can be overridden per-task)
    max_reflections: int = 2

    # Accumulated results
    step_results: List[str] = field(default_factory=list)

    # Trace for debugging
    node_trace: List[str] = field(default_factory=list)

    def record_node_visit(self, node_name: str):
        """Record visiting a node for trace."""
        self.node_trace.append(f"{datetime.utcnow().isoformat()}:{node_name}")

    def get_failed_step(self) -> Optional[PlanStep]:
        """Get the step that caused the current failure."""
        if self.plan and self.current_step_index < len(self.plan.steps):
            step = self.plan.steps[self.current_step_index]
            if step.status == StepStatus.FAILED:
                return step
        return None

    def can_reflect(self) -> bool:
        """Check if reflection is still allowed."""
        return self.reflection_count < self.max_reflections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "session_id": self.session_id,
            "complexity": self.complexity.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "current_step_index": self.current_step_index,
            "last_error": self.last_error,
            "reflection_count": self.reflection_count,
            "reflection_attempts": [r.to_dict() for r in self.reflection_attempts],
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "max_reflections": self.max_reflections,
            "step_results": self.step_results,
            "node_trace": self.node_trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningState":
        data = data.copy()
        if "complexity" in data and isinstance(data["complexity"], str):
            data["complexity"] = TaskComplexity(data["complexity"])
        if "plan" in data and data["plan"]:
            data["plan"] = Plan.from_dict(data["plan"])
        if "reflection_attempts" in data:
            data["reflection_attempts"] = [
                ReflectionAttempt.from_dict(r) for r in data["reflection_attempts"]
            ]
        if "stop_reason" in data and data["stop_reason"]:
            data["stop_reason"] = StopReason(data["stop_reason"])
        return cls(**data)


@dataclass
class ReasoningEvent:
    """
    Unified event schema for the experience corpus.

    Every significant action is logged as a ReasoningEvent to build
    the "experience" that REFLECT queries for learning.
    """
    timestamp: str
    session_id: str
    event_type: Literal["planning", "execution", "reflection", "verification", "classification"]
    goal: str

    # Step-level details (for execution events)
    step_num: Optional[int] = None
    step_description: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[str] = None

    # Outcome
    outcome_status: Optional[str] = None  # "success", "failure"
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None

    # LLM analysis
    llm_critique: Optional[str] = None

    # Context (for reflection events)
    context_used: Optional[str] = None

    # Plan details (for planning events)
    plan_steps_count: Optional[int] = None
    complexity: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "goal": self.goal,
            "step_num": self.step_num,
            "step_description": self.step_description,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "outcome_status": self.outcome_status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "error": self.error,
            "llm_critique": self.llm_critique,
            "context_used": self.context_used,
            "plan_steps_count": self.plan_steps_count,
            "complexity": self.complexity,
        }.items() if v is not None}

    def to_jsonl(self) -> str:
        """Convert to JSON line for appending to events.jsonl."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningEvent":
        # Filter to only known fields
        known_fields = {
            "timestamp", "session_id", "event_type", "goal",
            "step_num", "step_description", "tool", "tool_input",
            "outcome_status", "stdout", "stderr", "returncode", "error",
            "llm_critique", "context_used", "plan_steps_count", "complexity"
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def get_searchable_text(self) -> str:
        """Get text representation for keyword search."""
        parts = [
            self.goal or "",
            self.step_description or "",
            self.error or "",
            self.llm_critique or "",
            self.tool or "",
        ]
        return " ".join(p for p in parts if p).lower()


# Default max reflections by complexity
DEFAULT_MAX_REFLECTIONS = {
    TaskComplexity.SIMPLE: 0,
    TaskComplexity.MODERATE: 1,
    TaskComplexity.COMPLEX: 3,
}


def get_max_reflections(complexity: TaskComplexity) -> int:
    """Get default max reflections for a complexity level."""
    return DEFAULT_MAX_REFLECTIONS.get(complexity, 1)
