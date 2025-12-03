"""
Tests for reasoning_v30 types module.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import pytest
from datetime import datetime

from ragix_core.reasoning_v30.types import (
    TaskComplexity,
    StepStatus,
    ToolCall,
    ToolResult,
    PlanStep,
    Plan,
    ReflectionAttempt,
    ReasoningState,
    ReasoningEvent,
)


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_simple(self):
        tc = ToolCall(tool="find", args={"path": "."})
        assert tc.tool == "find"
        assert tc.args == {"path": "."}

    def test_to_dict(self):
        tc = ToolCall(tool="grep", args={"pattern": "TODO"})
        d = tc.to_dict()
        assert d["tool"] == "grep"
        assert d["args"]["pattern"] == "TODO"

    def test_from_dict(self):
        d = {"tool": "ls", "args": {"path": "/tmp"}}
        tc = ToolCall.from_dict(d)
        assert tc.tool == "ls"
        assert tc.args["path"] == "/tmp"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success(self):
        tr = ToolResult(tool="ls", returncode=0, stdout="file1.py\nfile2.py")
        assert tr.success is True

    def test_failure(self):
        tr = ToolResult(tool="cat", returncode=1, stderr="No such file")
        assert tr.success is False

    def test_exception(self):
        tr = ToolResult(tool="python", returncode=0, error="RuntimeError")
        assert tr.success is False

    def test_to_dict(self):
        tr = ToolResult(tool="wc", returncode=0, stdout="100", duration_ms=15.5)
        d = tr.to_dict()
        assert d["tool"] == "wc"
        assert d["duration_ms"] == 15.5


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_create(self):
        step = PlanStep(
            num=1,
            description="Find Python files",
            tool_call=ToolCall(tool="find", args={"pattern": "*.py"}),
        )
        assert step.num == 1
        assert step.status == StepStatus.PENDING

    def test_with_result(self):
        step = PlanStep(num=1, description="List files")
        step.status = StepStatus.SUCCESS
        step.result = ToolResult(tool="ls", returncode=0, stdout="files")
        assert step.status == StepStatus.SUCCESS
        assert step.result.success


class TestPlan:
    """Tests for Plan dataclass."""

    def test_create(self):
        plan = Plan(
            objective="Find largest file",
            steps=[
                PlanStep(num=1, description="List files"),
                PlanStep(num=2, description="Sort by size"),
            ],
            validation="Show the largest file",
            confidence=0.8,
        )
        assert len(plan.steps) == 2
        assert plan.confidence == 0.8

    def test_get_current_step(self):
        plan = Plan(
            objective="Test",
            steps=[
                PlanStep(num=1, description="Step 1"),
                PlanStep(num=2, description="Step 2"),
            ],
        )
        assert plan.get_current_step(0).num == 1
        assert plan.get_current_step(1).num == 2
        assert plan.get_current_step(5) is None

    def test_is_complete(self):
        plan = Plan(
            objective="Test",
            steps=[
                PlanStep(num=1, description="Step 1", status=StepStatus.SUCCESS),
                PlanStep(num=2, description="Step 2", status=StepStatus.SKIPPED),
            ],
        )
        assert plan.is_complete() is True

    def test_is_not_complete(self):
        plan = Plan(
            objective="Test",
            steps=[
                PlanStep(num=1, description="Step 1", status=StepStatus.SUCCESS),
                PlanStep(num=2, description="Step 2", status=StepStatus.PENDING),
            ],
        )
        assert plan.is_complete() is False

    def test_get_failed_steps(self):
        plan = Plan(
            objective="Test",
            steps=[
                PlanStep(num=1, description="Step 1", status=StepStatus.SUCCESS),
                PlanStep(num=2, description="Step 2", status=StepStatus.FAILED),
            ],
        )
        failed = plan.get_failed_steps()
        assert len(failed) == 1
        assert failed[0].num == 2


class TestReasoningState:
    """Tests for ReasoningState dataclass."""

    def test_create(self):
        state = ReasoningState(
            goal="Test goal",
            session_id="test-session",
        )
        assert state.goal == "Test goal"
        assert state.complexity == TaskComplexity.MODERATE
        assert state.reflection_count == 0

    def test_add_reflection(self):
        state = ReasoningState(goal="Test", session_id="test")
        attempt = ReflectionAttempt(
            timestamp=datetime.utcnow().isoformat(),
            failed_step=1,
            error="Test error",
            diagnosis="Test diagnosis",
            new_plan_summary="New plan",
        )
        state.add_reflection(attempt)
        assert state.reflection_count == 1
        assert len(state.reflection_attempts) == 1

    def test_to_dict_from_dict(self):
        state = ReasoningState(
            goal="Test goal",
            session_id="test-session",
            complexity=TaskComplexity.COMPLEX,
            confidence=0.75,
        )
        d = state.to_dict()
        restored = ReasoningState.from_dict(d)
        assert restored.goal == state.goal
        assert restored.complexity == state.complexity
        assert restored.confidence == state.confidence


class TestReasoningEvent:
    """Tests for ReasoningEvent dataclass."""

    def test_create_now(self):
        event = ReasoningEvent.create_now(
            session_id="test",
            event_type="execution",
            goal="Test goal",
            step_num=1,
            outcome_status="success",
        )
        assert event.session_id == "test"
        assert event.event_type == "execution"
        assert event.timestamp  # Should be set

    def test_to_dict(self):
        event = ReasoningEvent(
            timestamp="2025-12-03T10:00:00",
            session_id="test",
            event_type="reflection",
            goal="Test",
            llm_critique="Diagnosis text",
        )
        d = event.to_dict()
        assert d["event_type"] == "reflection"
        assert d["llm_critique"] == "Diagnosis text"
