"""
RAGIX v0.30 Reasoning Types

Core dataclasses for the Reflective Reasoning Graph:
- TaskComplexity: BYPASS, SIMPLE, MODERATE, COMPLEX
- ToolCall/ToolResult: Unified tool invocation schema
- Plan, PlanStep: Structured planning objects
- ReasoningState: Graph traversal state
- ReasoningEvent: Experience corpus events

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class TaskComplexity(Enum):
    """
    Task complexity classification for routing through the reasoning graph.

    - BYPASS: Pure reasoning/explanation/summarization. No tools, no plan.
    - SIMPLE: 1-2 shell commands or a single file edit, no reflections.
    - MODERATE: 2-4 steps with shell commands and/or multiple edits, may need 1 reflection.
    - COMPLEX: Multi-step investigation across files and tools, up to 3 reflections.
    """
    BYPASS = "bypass"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class StepStatus(Enum):
    """Execution status for a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    """
    Unified schema for tool invocations.

    All Unix tools (rt-*, edit_file, ragix-ast, etc.) should use this format
    for consistent handling and logging.
    """
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"tool": self.tool, "args": self.args}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(tool=data.get("tool", ""), args=data.get("args", {}))


@dataclass
class ToolResult:
    """
    Unified schema for tool results.

    Captures returncode, stdout, stderr for deterministic output handling.
    """
    tool: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None  # Python exception message, if any
    duration_ms: Optional[float] = None  # Execution time in milliseconds

    @property
    def success(self) -> bool:
        """Check if tool execution was successful."""
        return self.returncode == 0 and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool": self.tool,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create from dictionary."""
        return cls(
            tool=data.get("tool", ""),
            returncode=data.get("returncode", -1),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
        )


@dataclass
class PlanStep:
    """
    A single step in an execution plan.

    Each step has:
    - num: Step number (1-indexed)
    - description: Natural language description
    - tool_call: Optional ToolCall for execution
    - status: Current execution status
    - result: Optional ToolResult after execution
    """
    num: int
    description: str
    tool_call: Optional[ToolCall] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num": self.num,
            "description": self.description,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Create from dictionary."""
        tool_call = None
        if data.get("tool_call"):
            tool_call = ToolCall.from_dict(data["tool_call"])

        result = None
        if data.get("result"):
            result = ToolResult.from_dict(data["result"])

        return cls(
            num=data.get("num", 0),
            description=data.get("description", ""),
            tool_call=tool_call,
            status=StepStatus(data.get("status", "pending")),
            result=result,
        )


@dataclass
class Plan:
    """
    A structured execution plan.

    Contains:
    - objective: What the plan aims to achieve
    - steps: List of PlanStep objects
    - validation: How to verify the plan succeeded
    - confidence: LLM self-estimate (0.0-1.0)
    """
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    validation: str = ""
    confidence: Optional[float] = None

    def get_current_step(self, index: int) -> Optional[PlanStep]:
        """Get step at index, or None if out of bounds."""
        return self.steps[index] if 0 <= index < len(self.steps) else None

    def is_complete(self) -> bool:
        """Check if all steps are done (SUCCESS or SKIPPED)."""
        return all(
            s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED)
            for s in self.steps
        )

    def get_failed_steps(self) -> List[PlanStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def get_completed_steps(self) -> List[PlanStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.SUCCESS]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "validation": self.validation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Create from dictionary."""
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            objective=data.get("objective", ""),
            steps=steps,
            validation=data.get("validation", ""),
            confidence=data.get("confidence"),
        )


@dataclass
class ReflectionAttempt:
    """
    Record of a single reflection attempt.

    Captures:
    - timestamp: When reflection occurred
    - failed_step: Which step failed
    - error: The error message
    - diagnosis: LLM's analysis of what went wrong
    - new_plan_summary: Brief description of the revised approach
    """
    timestamp: str
    failed_step: int
    error: str
    diagnosis: str
    new_plan_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "failed_step": self.failed_step,
            "error": self.error,
            "diagnosis": self.diagnosis,
            "new_plan_summary": self.new_plan_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionAttempt":
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", ""),
            failed_step=data.get("failed_step", -1),
            error=data.get("error", ""),
            diagnosis=data.get("diagnosis", ""),
            new_plan_summary=data.get("new_plan_summary", ""),
        )


@dataclass
class ReasoningState:
    """
    State object passed between graph nodes.

    This is the central state machine for the reasoning graph,
    tracking goal, plan, execution progress, reflections, and final answer.
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
    stop_reason: Optional[str] = None  # "success", "max_reflections", "max_iterations", "no_plan", "bypass"
    confidence: Optional[float] = None  # State-level confidence for overall answer
    tool_result: Optional["ToolResult"] = None  # For SIMPLE tasks: the executed command result

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow().isoformat()

    def add_reflection(self, attempt: ReflectionAttempt) -> None:
        """Add a reflection attempt and increment counter."""
        self.reflection_attempts.append(attempt)
        self.reflection_count += 1
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal": self.goal,
            "session_id": self.session_id,
            "complexity": self.complexity.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "current_step_index": self.current_step_index,
            "last_error": self.last_error,
            "reflection_count": self.reflection_count,
            "reflection_attempts": [a.to_dict() for a in self.reflection_attempts],
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason,
            "confidence": self.confidence,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningState":
        """Create from dictionary."""
        plan = None
        if data.get("plan"):
            plan = Plan.from_dict(data["plan"])

        attempts = [
            ReflectionAttempt.from_dict(a)
            for a in data.get("reflection_attempts", [])
        ]

        return cls(
            goal=data.get("goal", ""),
            session_id=data.get("session_id", ""),
            complexity=TaskComplexity(data.get("complexity", "moderate")),
            plan=plan,
            current_step_index=data.get("current_step_index", 0),
            last_error=data.get("last_error"),
            reflection_count=data.get("reflection_count", 0),
            reflection_attempts=attempts,
            final_answer=data.get("final_answer"),
            stop_reason=data.get("stop_reason"),
            confidence=data.get("confidence"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


@dataclass
class ReasoningEvent:
    """
    Unified event schema for experience corpus.

    All reasoning events (planning, execution, reflection, verification, respond)
    are recorded in this format for later retrieval and learning.
    """
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
    context_used: Optional[str] = None  # RAG / experience context
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
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
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningEvent":
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id", ""),
            event_type=data.get("event_type", "execution"),
            goal=data.get("goal", ""),
            step_num=data.get("step_num"),
            step_description=data.get("step_description"),
            tool=data.get("tool"),
            tool_input=data.get("tool_input"),
            outcome_status=data.get("outcome_status"),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            returncode=data.get("returncode"),
            error=data.get("error"),
            llm_critique=data.get("llm_critique"),
            context_used=data.get("context_used"),
            meta=data.get("meta", {}),
        )

    @classmethod
    def create_now(
        cls,
        session_id: str,
        event_type: Literal["planning", "execution", "reflection", "verification", "respond"],
        goal: str,
        **kwargs
    ) -> "ReasoningEvent":
        """Factory method to create event with current timestamp."""
        return cls(
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            event_type=event_type,
            goal=goal,
            **kwargs
        )
