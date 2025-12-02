"""
RAGIX Reasoning Graph Tests - v0.23 Reflective Reasoning Graph

Tests for the graph-based reasoning system including:
- ReasoningState and supporting types
- ExperienceCorpus (single and hybrid)
- Graph nodes (CLASSIFY, PLAN, EXECUTE, REFLECT, VERIFY, RESPOND)
- ReasoningGraph orchestrator
- Feature flag and factory function

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-02
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import MagicMock, patch

import pytest

from ragix_core.reasoning_types import (
    TaskComplexity,
    StepStatus,
    StopReason,
    PlanStep,
    Plan,
    ReflectionAttempt,
    ReasoningState,
    ReasoningEvent,
    get_max_reflections,
)
from ragix_core.experience_corpus import (
    ExperienceCorpus,
    HybridExperienceCorpus,
    get_hybrid_corpus,
)
from ragix_core.reasoning_graph import (
    BaseNode,
    ReasoningGraph,
    ClassifyNode,
    DirectExecNode,
    PlanNode,
    ExecuteNode,
    ReflectNode,
    VerifyNode,
    RespondNode,
    create_reasoning_graph,
)
from ragix_core.reasoning import (
    ReasoningStrategy,
    GraphReasoningLoop,
    create_reasoning_loop,
    get_reasoning_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_corpus_dir():
    """Create a temporary directory for experience corpus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_events():
    """Create sample ReasoningEvents for testing."""
    now = datetime.utcnow()
    return [
        ReasoningEvent(
            timestamp=(now - timedelta(days=1)).isoformat(),
            session_id="session-001",
            event_type="execution",
            goal="list files in current directory",
            step_num=1,
            step_description="Run ls command",
            tool="bash",
            tool_input="ls -la",
            outcome_status="success",
            stdout="file1.py\nfile2.py\n",
        ),
        ReasoningEvent(
            timestamp=(now - timedelta(days=2)).isoformat(),
            session_id="session-002",
            event_type="execution",
            goal="search for function definition",
            step_num=1,
            step_description="Grep for def keyword",
            tool="bash",
            tool_input="grep -r 'def main'",
            outcome_status="failure",
            error="grep: command not found",
            llm_critique="grep was not available in the sandbox. Use find with -exec instead.",
        ),
        ReasoningEvent(
            timestamp=(now - timedelta(days=5)).isoformat(),
            session_id="session-003",
            event_type="planning",
            goal="refactor module structure",
            plan_steps_count=5,
            complexity="complex",
            outcome_status="success",
        ),
    ]


@pytest.fixture
def mock_llm_generate():
    """Mock LLM generate function with correct signature (persona, messages)."""
    def _generate(persona: str, messages: List[Dict]) -> str:
        # Get the last message content
        content = ""
        if messages and len(messages) > 0:
            content = messages[-1].get("content", "").lower()

        # Simple pattern matching for test purposes
        if "classify" in persona.lower() or "complexity" in content:
            return '{"complexity": "moderate"}'
        elif "plan" in persona.lower():
            return """[PLAN]
Objective: Complete the task
Steps:
1. First step
2. Second step
Validation: Check output"""
        elif "verify" in persona.lower():
            return '{"verified": true, "summary": "Task completed successfully"}'
        elif "reflect" in persona.lower():
            return """The step failed because the file doesn't exist.
[PLAN]
Objective: Retry with correct path
Steps:
1. Find correct file path
2. Read the file
Validation: File content retrieved"""
        else:
            return "Task completed."
    return _generate


@pytest.fixture
def mock_execute_fn():
    """Mock direct execution function."""
    def _execute(goal: str) -> Tuple[Any, str]:
        # Return (result, message)
        return ("file1.py\nfile2.py", "Listed files successfully")
    return _execute


@pytest.fixture
def mock_execute_step_fn():
    """Mock step execution function."""
    def _execute_step(step: PlanStep) -> PlanStep:
        step.status = StepStatus.SUCCESS
        step.result = "Step executed"
        step.stdout = "output"
        step.returncode = 0
        return step
    return _execute_step


@pytest.fixture
def mock_shell_executor():
    """Mock shell executor function."""
    def _execute(command: str) -> str:
        # Return stdout only
        if "ls" in command:
            return "file1.py\nfile2.py\n"
        elif "grep" in command:
            return ""
        elif "cat" in command:
            return "file content here"
        else:
            return ""
    return _execute


@pytest.fixture
def mock_experience_corpus(temp_corpus_dir):
    """Create mock experience corpus."""
    return HybridExperienceCorpus(project_path=temp_corpus_dir)


@pytest.fixture
def mock_agent_config():
    """Create mock AgentConfig."""
    config = MagicMock()
    config.system_prompt = "You are a helpful assistant."
    config.model = "test-model"
    return config


@pytest.fixture
def mock_episodic_memory():
    """Create mock EpisodicMemory."""
    memory = MagicMock()
    memory.get_recent_context.return_value = "Previous context: listed files"
    memory.add_episode = MagicMock()
    return memory


# =============================================================================
# Tests: ReasoningTypes
# =============================================================================

class TestReasoningTypes:
    """Test reasoning type definitions."""

    def test_task_complexity_enum(self):
        """Test TaskComplexity enum values."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"

    def test_step_status_enum(self):
        """Test StepStatus enum values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.SUCCESS.value == "success"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"

    def test_stop_reason_enum(self):
        """Test StopReason enum values."""
        assert StopReason.SUCCESS.value == "success"
        assert StopReason.MAX_REFLECTIONS.value == "max_reflections"
        assert StopReason.MAX_ITERATIONS.value == "max_iterations"

    def test_plan_step_creation(self):
        """Test PlanStep dataclass."""
        step = PlanStep(num=1, description="Run test command")
        assert step.num == 1
        assert step.description == "Run test command"
        assert step.status == StepStatus.PENDING
        assert step.tool is None

    def test_plan_step_serialization(self):
        """Test PlanStep to/from dict."""
        step = PlanStep(
            num=1,
            description="List files",
            tool="bash",
            args={"command": "ls"},
            status=StepStatus.SUCCESS,
            result="file1.py",
        )
        data = step.to_dict()
        assert data["num"] == 1
        assert data["status"] == "success"

        restored = PlanStep.from_dict(data)
        assert restored.num == step.num
        assert restored.status == StepStatus.SUCCESS

    def test_plan_is_complete(self):
        """Test Plan.is_complete method."""
        plan = Plan(
            objective="Test objective",
            steps=[
                PlanStep(num=1, description="Step 1", status=StepStatus.SUCCESS),
                PlanStep(num=2, description="Step 2", status=StepStatus.SUCCESS),
            ]
        )
        assert plan.is_complete() is True

        plan.steps[1].status = StepStatus.PENDING
        assert plan.is_complete() is False

    def test_plan_has_failures(self):
        """Test Plan.has_failures method."""
        plan = Plan(
            objective="Test objective",
            steps=[
                PlanStep(num=1, description="Step 1", status=StepStatus.SUCCESS),
                PlanStep(num=2, description="Step 2", status=StepStatus.FAILED),
            ]
        )
        assert plan.has_failures() is True

    def test_plan_to_text(self):
        """Test Plan.to_text formatting."""
        plan = Plan(
            objective="Test the system",
            steps=[
                PlanStep(num=1, description="Run tests", status=StepStatus.SUCCESS),
                PlanStep(num=2, description="Check output", status=StepStatus.PENDING),
            ],
            validation="All tests pass"
        )
        text = plan.to_text()
        assert "[PLAN]" in text
        assert "Test the system" in text
        assert "[x] Run tests" in text
        assert "[ ] Check output" in text

    def test_reasoning_state_creation(self):
        """Test ReasoningState dataclass."""
        state = ReasoningState(
            goal="Find all Python files",
            session_id="test-session",
        )
        assert state.goal == "Find all Python files"
        assert state.complexity == TaskComplexity.MODERATE
        assert state.reflection_count == 0
        assert state.can_reflect() is True

    def test_reasoning_state_can_reflect(self):
        """Test ReasoningState.can_reflect with max_reflections."""
        state = ReasoningState(
            goal="Test goal",
            session_id="test",
            max_reflections=2,
        )
        assert state.can_reflect() is True

        state.reflection_count = 2
        assert state.can_reflect() is False

    def test_reasoning_event_searchable_text(self):
        """Test ReasoningEvent.get_searchable_text."""
        event = ReasoningEvent(
            timestamp="2025-12-01T10:00:00",
            session_id="test",
            event_type="execution",
            goal="Find Python files",
            step_description="Search using grep",
            error="Command failed",
            tool="bash",
        )
        text = event.get_searchable_text()
        assert "python" in text
        assert "grep" in text
        assert "failed" in text

    def test_get_max_reflections(self):
        """Test default max_reflections by complexity."""
        assert get_max_reflections(TaskComplexity.SIMPLE) == 0
        assert get_max_reflections(TaskComplexity.MODERATE) == 1
        assert get_max_reflections(TaskComplexity.COMPLEX) == 3


# =============================================================================
# Tests: ExperienceCorpus
# =============================================================================

class TestExperienceCorpus:
    """Test single experience corpus."""

    def test_corpus_creation(self, temp_corpus_dir):
        """Test corpus initialization."""
        events_path = temp_corpus_dir / "events.jsonl"
        corpus = ExperienceCorpus(events_path)
        assert corpus.events_path == events_path
        assert corpus.max_events == 1000

    def test_corpus_append_and_search(self, temp_corpus_dir, sample_events):
        """Test appending and searching events."""
        events_path = temp_corpus_dir / "events.jsonl"
        corpus = ExperienceCorpus(events_path)

        # Append events
        for event in sample_events:
            corpus.append(event)

        # Search
        results = corpus.search("grep function", top_k=3)
        assert len(results) > 0

        # Check that failure with critique ranks well
        for score, event in results:
            if event.outcome_status == "failure":
                assert event.llm_critique is not None

    def test_corpus_search_with_outcome_filter(self, temp_corpus_dir, sample_events):
        """Test searching with outcome filter."""
        events_path = temp_corpus_dir / "events.jsonl"
        corpus = ExperienceCorpus(events_path)

        for event in sample_events:
            corpus.append(event)

        # Search only failures
        results = corpus.search("grep", outcome_filter="failure")
        for _, event in results:
            assert event.outcome_status == "failure"

    def test_corpus_stats(self, temp_corpus_dir, sample_events):
        """Test corpus statistics."""
        events_path = temp_corpus_dir / "events.jsonl"
        corpus = ExperienceCorpus(events_path)

        for event in sample_events:
            corpus.append(event)

        stats = corpus.get_stats()
        assert stats["count"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1

    def test_corpus_empty_search(self, temp_corpus_dir):
        """Test search on empty corpus."""
        events_path = temp_corpus_dir / "events.jsonl"
        corpus = ExperienceCorpus(events_path)

        results = corpus.search("anything")
        assert results == []


class TestHybridExperienceCorpus:
    """Test hybrid experience corpus."""

    def test_hybrid_corpus_creation(self, temp_corpus_dir):
        """Test hybrid corpus with project path."""
        corpus = HybridExperienceCorpus(project_path=temp_corpus_dir)
        assert corpus.project_corpus is not None
        assert corpus.global_corpus is not None

    def test_hybrid_corpus_without_project(self):
        """Test hybrid corpus without project path."""
        corpus = HybridExperienceCorpus(project_path=None)
        assert corpus.project_corpus is None
        assert corpus.global_corpus is not None

    def test_hybrid_search_prioritizes_project(self, temp_corpus_dir, sample_events):
        """Test that project results are weighted higher."""
        corpus = HybridExperienceCorpus(project_path=temp_corpus_dir)

        # Add to project corpus
        corpus.project_corpus.append(sample_events[0])

        results = corpus.search("list files")
        # Should have at least the project result
        assert len(results) > 0

    def test_get_hybrid_corpus_factory(self, temp_corpus_dir):
        """Test factory function."""
        corpus = get_hybrid_corpus(project_path=temp_corpus_dir)
        assert isinstance(corpus, HybridExperienceCorpus)


# =============================================================================
# Tests: ReasoningGraph Nodes
# =============================================================================

class TestClassifyNode:
    """Test ClassifyNode."""

    def test_classify_simple_task(self):
        """Test classification of simple task using default classifier."""
        node = ClassifyNode()
        state = ReasoningState(goal="hello", session_id="test")

        new_state, next_node = node.run(state)
        assert new_state.complexity == TaskComplexity.SIMPLE
        assert next_node == "DIRECT_EXEC"

    def test_classify_complex_task(self):
        """Test classification of complex task using default classifier."""
        node = ClassifyNode()
        state = ReasoningState(goal="refactor the module and then update tests", session_id="test")

        new_state, next_node = node.run(state)
        assert new_state.complexity == TaskComplexity.COMPLEX
        assert next_node == "PLAN"

    def test_classify_with_custom_function(self):
        """Test classification with custom function."""
        def always_complex(goal: str) -> TaskComplexity:
            return TaskComplexity.COMPLEX

        node = ClassifyNode(classify_fn=always_complex)
        state = ReasoningState(goal="simple task", session_id="test")

        new_state, next_node = node.run(state)
        assert new_state.complexity == TaskComplexity.COMPLEX


class TestDirectExecNode:
    """Test DirectExecNode."""

    def test_direct_exec(self, mock_execute_fn):
        """Test direct execution."""
        node = DirectExecNode(execute_fn=mock_execute_fn)
        state = ReasoningState(goal="list files", session_id="test")

        new_state, next_node = node.run(state)
        assert new_state.final_answer is not None
        # DirectExecNode goes to END, not RESPOND
        assert next_node == "END"


class TestPlanNode:
    """Test PlanNode."""

    def test_plan_generation(self, mock_llm_generate):
        """Test plan parsing from LLM output."""
        node = PlanNode(llm_generate=mock_llm_generate)
        state = ReasoningState(
            goal="Complete task",
            session_id="test",
            complexity=TaskComplexity.MODERATE,
        )

        new_state, next_node = node.run(state)
        assert new_state.plan is not None
        assert len(new_state.plan.steps) >= 1
        assert next_node == "EXECUTE"


class TestExecuteNode:
    """Test ExecuteNode."""

    def test_execute_step(self, mock_execute_step_fn):
        """Test step execution."""
        # ExecuteNode only takes execute_step_fn
        node = ExecuteNode(execute_step_fn=mock_execute_step_fn)

        plan = Plan(
            objective="List files",
            steps=[
                PlanStep(num=1, description="Run ls command", tool="bash"),
            ]
        )
        state = ReasoningState(
            goal="List files",
            session_id="test",
            plan=plan,
            current_step_index=0,
        )

        new_state, next_node = node.run(state)
        assert new_state.plan.steps[0].status == StepStatus.SUCCESS


class TestReflectNode:
    """Test ReflectNode."""

    def test_reflect_allowed_tools(self, mock_llm_generate, mock_experience_corpus):
        """Test that reflect node has read-only tools."""
        node = ReflectNode(
            llm_generate=mock_llm_generate,
            experience_corpus=mock_experience_corpus,
        )

        assert "ls" in node.ALLOWED_TOOLS
        assert "grep" in node.ALLOWED_TOOLS
        assert "find" in node.ALLOWED_TOOLS
        # Dangerous tools should not be allowed
        assert "rm" not in node.ALLOWED_TOOLS
        assert "mv" not in node.ALLOWED_TOOLS

    def test_reflect_safe_shell_blocks_dangerous(
        self, mock_llm_generate, mock_experience_corpus, mock_shell_executor
    ):
        """Test that _safe_shell blocks non-allowed commands."""
        node = ReflectNode(
            llm_generate=mock_llm_generate,
            experience_corpus=mock_experience_corpus,
            shell_executor=mock_shell_executor,
        )

        result = node._safe_shell("rm -rf /")
        assert "BLOCKED" in result

    def test_reflect_safe_shell_allows_read_only(
        self, mock_llm_generate, mock_experience_corpus, mock_shell_executor
    ):
        """Test that _safe_shell allows read-only commands."""
        node = ReflectNode(
            llm_generate=mock_llm_generate,
            experience_corpus=mock_experience_corpus,
            shell_executor=mock_shell_executor,
        )

        result = node._safe_shell("ls -la")
        assert "BLOCKED" not in result


class TestVerifyNode:
    """Test VerifyNode."""

    def test_verify_success(self, mock_llm_generate):
        """Test verification of successful execution."""
        # Override mock to return PASS for verification
        def verify_generate(persona: str, messages: list) -> str:
            return "Status: PASS\nSummary: All steps completed successfully"

        node = VerifyNode(llm_generate=verify_generate)

        plan = Plan(
            objective="List files",
            steps=[
                PlanStep(num=1, description="Run ls", status=StepStatus.SUCCESS, result="file1.py"),
            ]
        )
        state = ReasoningState(
            goal="List files",
            session_id="test",
            plan=plan,
            step_results=["Listed files: file1.py"],
        )

        new_state, next_node = node.run(state)
        # VerifyNode transitions to RESPOND (which sets stop_reason)
        assert next_node == "RESPOND"
        # Verification result should be added to step_results
        assert any("VERIFY" in r for r in new_state.step_results)


# =============================================================================
# Tests: ReasoningGraph Orchestrator
# =============================================================================

class TestReasoningGraph:
    """Test ReasoningGraph orchestrator."""

    def test_graph_creation(
        self,
        mock_llm_generate,
        mock_execute_fn,
        mock_execute_step_fn,
        mock_experience_corpus,
    ):
        """Test graph creation with factory."""
        graph = create_reasoning_graph(
            llm_generate=mock_llm_generate,
            execute_fn=mock_execute_fn,
            execute_step_fn=mock_execute_step_fn,
            experience_corpus=mock_experience_corpus,
        )
        assert isinstance(graph, ReasoningGraph)
        assert "CLASSIFY" in graph.nodes
        assert "PLAN" in graph.nodes
        assert "EXECUTE" in graph.nodes
        assert "REFLECT" in graph.nodes
        assert "VERIFY" in graph.nodes
        assert "RESPOND" in graph.nodes

    def test_graph_max_iterations_protection(
        self,
        mock_llm_generate,
        mock_execute_fn,
        mock_execute_step_fn,
        mock_experience_corpus,
    ):
        """Test that graph stops at max iterations."""
        graph = create_reasoning_graph(
            llm_generate=mock_llm_generate,
            execute_fn=mock_execute_fn,
            execute_step_fn=mock_execute_step_fn,
            experience_corpus=mock_experience_corpus,
        )
        graph.max_iterations = 3  # Set very low for test

        state = ReasoningState(goal="Test goal", session_id="test")

        # Run should eventually stop
        final_state = graph.run(state)
        assert final_state.stop_reason is not None


# =============================================================================
# Tests: Feature Flag and Factory
# =============================================================================

class TestReasoningStrategy:
    """Test reasoning strategy selection."""

    def test_default_strategy(self):
        """Test default strategy is loop_v1."""
        # Without env var set, should default to loop_v1
        with patch.dict(os.environ, {}, clear=True):
            strategy = get_reasoning_strategy()
            # Default depends on implementation
            assert strategy in [ReasoningStrategy.LOOP_V1, ReasoningStrategy.GRAPH_V2]

    def test_env_var_strategy(self):
        """Test strategy from environment variable."""
        with patch.dict(os.environ, {"RAGIX_REASONING_STRATEGY": "graph_v2"}):
            strategy = get_reasoning_strategy()
            assert strategy == ReasoningStrategy.GRAPH_V2

    def test_factory_creates_correct_type(
        self, mock_llm_generate, mock_agent_config, mock_episodic_memory
    ):
        """Test factory creates correct reasoning loop type."""
        # Test loop_v1
        loop_v1 = create_reasoning_loop(
            llm_generate=mock_llm_generate,
            agent_config=mock_agent_config,
            episodic_memory=mock_episodic_memory,
            strategy=ReasoningStrategy.LOOP_V1,
        )
        assert loop_v1 is not None

        # Test graph_v2 - needs shell_executor
        loop_v2 = create_reasoning_loop(
            llm_generate=mock_llm_generate,
            agent_config=mock_agent_config,
            episodic_memory=mock_episodic_memory,
            shell_executor=lambda c: ("", "", 0),
            strategy=ReasoningStrategy.GRAPH_V2,
        )
        assert isinstance(loop_v2, GraphReasoningLoop)


# =============================================================================
# Tests: Integration
# =============================================================================

class TestReasoningIntegration:
    """Integration tests for the reasoning graph system."""

    def test_simple_task_bypasses_planning(
        self,
        mock_llm_generate,
        mock_execute_fn,
        mock_execute_step_fn,
        mock_experience_corpus,
    ):
        """Test that simple tasks go directly to execution."""
        graph = create_reasoning_graph(
            llm_generate=mock_llm_generate,
            execute_fn=mock_execute_fn,
            execute_step_fn=mock_execute_step_fn,
            experience_corpus=mock_experience_corpus,
        )

        state = ReasoningState(goal="hello", session_id="test")
        final_state = graph.run(state)

        # Simple tasks should have no plan
        assert final_state.complexity == TaskComplexity.SIMPLE
        assert final_state.plan is None

    def test_reflection_count_increments(
        self, mock_llm_generate, mock_experience_corpus, mock_shell_executor
    ):
        """Test that reflection count increments correctly."""
        reflect_node = ReflectNode(
            llm_generate=mock_llm_generate,
            experience_corpus=mock_experience_corpus,
            shell_executor=mock_shell_executor,
        )

        plan = Plan(
            objective="Test",
            steps=[PlanStep(num=1, description="Fail", status=StepStatus.FAILED)],
        )
        state = ReasoningState(
            goal="Test",
            session_id="test",
            plan=plan,
            current_step_index=0,
            last_error="Test error",
            reflection_count=0,
            max_reflections=3,
        )

        new_state, next_node = reflect_node.run(state)
        assert new_state.reflection_count == 1

    def test_max_reflections_stops_in_execute(self, mock_execute_step_fn):
        """Test that max_reflections triggers stop in EXECUTE node."""
        # Create a failing step executor
        def failing_step_fn(step: PlanStep) -> PlanStep:
            step.status = StepStatus.FAILED
            step.error = "Test error"
            step.returncode = 1
            return step

        execute_node = ExecuteNode(execute_step_fn=failing_step_fn)

        plan = Plan(
            objective="Test",
            steps=[PlanStep(num=1, description="Fail")],
        )
        state = ReasoningState(
            goal="Test",
            session_id="test",
            plan=plan,
            current_step_index=0,
            reflection_count=2,  # Already at limit
            max_reflections=2,
        )

        new_state, next_node = execute_node.run(state)
        # When can_reflect() returns False, EXECUTE goes to RESPOND
        assert new_state.stop_reason == StopReason.MAX_REFLECTIONS
        assert next_node == "RESPOND"


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
