"""
Tests for reasoning_v30 graph module.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import pytest
from typing import Tuple

from ragix_core.reasoning_v30.graph import (
    BaseNode,
    ReasoningGraph,
    GraphBuilder,
    EndNode,
)
from ragix_core.reasoning_v30.types import (
    ReasoningState,
    TaskComplexity,
)


class SimpleNode(BaseNode):
    """Simple test node that transitions to a fixed next node."""
    name = "SIMPLE"

    def __init__(self, next_node: str = "END"):
        self.next_node = next_node

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        state.final_answer = "Simple answer"
        return state, self.next_node


class CounterNode(BaseNode):
    """Node that counts invocations and transitions after N calls."""
    name = "COUNTER"

    def __init__(self, max_count: int = 3):
        self.max_count = max_count
        self.count = 0

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        self.count += 1
        if self.count >= self.max_count:
            return state, "END"
        return state, "COUNTER"


class ErrorNode(BaseNode):
    """Node that raises an exception."""
    name = "ERROR"

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        raise ValueError("Test error")


class TestReasoningGraph:
    """Tests for ReasoningGraph class."""

    def test_simple_execution(self):
        """Test simple graph execution."""
        nodes = {
            "START": SimpleNode(next_node="END"),
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="START", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        result = graph.run(state)

        assert result.final_answer == "Simple answer"
        assert graph.trace == ["START"]

    def test_multi_node_execution(self):
        """Test execution through multiple nodes."""
        nodes = {
            "A": SimpleNode(next_node="B"),
            "B": SimpleNode(next_node="C"),
            "C": SimpleNode(next_node="END"),
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="A", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        result = graph.run(state)

        assert graph.trace == ["A", "B", "C"]
        assert result.final_answer == "Simple answer"

    def test_max_iterations(self):
        """Test max_iterations limit."""
        counter = CounterNode(max_count=100)  # Would loop many times
        nodes = {
            "COUNTER": counter,
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="COUNTER", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        result = graph.run(state, max_iterations=5)

        assert result.stop_reason == "max_iterations"
        assert counter.count == 5

    def test_node_exception(self):
        """Test handling of node exceptions."""
        nodes = {
            "ERROR": ErrorNode(),
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="ERROR", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        result = graph.run(state)

        assert "node_exception" in result.stop_reason
        assert result.last_error == "Test error"

    def test_missing_node(self):
        """Test handling of missing nodes."""
        nodes = {
            "START": SimpleNode(next_node="MISSING"),
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="START", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        result = graph.run(state)

        assert "node_not_found" in result.stop_reason

    def test_trace_with_timestamps(self):
        """Test that timing information is recorded."""
        nodes = {
            "A": SimpleNode(next_node="B"),
            "B": SimpleNode(next_node="END"),
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="A", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        graph.run(state)

        assert len(graph.trace_with_timestamps) == 2
        for node_name, next_name, duration_ms in graph.trace_with_timestamps:
            assert isinstance(duration_ms, float)
            assert duration_ms >= 0

    def test_get_trace_summary(self):
        """Test trace summary generation."""
        nodes = {
            "A": SimpleNode(next_node="END"),
            "END": EndNode(),
        }
        graph = ReasoningGraph(nodes=nodes, start="A", end="END")
        state = ReasoningState(goal="Test", session_id="test")

        graph.run(state)
        summary = graph.get_trace_summary()

        assert "Graph Execution Trace" in summary
        assert "A" in summary


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    def test_build_simple_graph(self):
        """Test building a simple graph."""
        graph = (GraphBuilder()
            .add_node(SimpleNode(next_node="END"))
            .add_node(EndNode())
            .set_start("SIMPLE")
            .set_end("END")
            .build())

        state = ReasoningState(goal="Test", session_id="test")
        result = graph.run(state)

        assert result.final_answer == "Simple answer"

    def test_build_with_event_emitter(self):
        """Test building graph with event emitter."""
        events = []

        def emit(event):
            events.append(event)

        graph = (GraphBuilder()
            .add_node(SimpleNode())
            .add_node(EndNode())
            .set_start("SIMPLE")
            .set_end("END")
            .set_event_emitter(emit)
            .build())

        assert graph.emit_event_fn is not None


class TestValidation:
    """Tests for graph validation."""

    def test_invalid_start_node(self):
        """Test validation of missing start node."""
        with pytest.raises(ValueError, match="Start node"):
            ReasoningGraph(
                nodes={"END": EndNode()},
                start="MISSING",
                end="END"
            )

    def test_invalid_node_type(self):
        """Test validation of invalid node types."""
        with pytest.raises(TypeError, match="must be a BaseNode"):
            ReasoningGraph(
                nodes={"START": "not a node", "END": EndNode()},
                start="START",
                end="END"
            )
