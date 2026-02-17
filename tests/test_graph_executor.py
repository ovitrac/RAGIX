"""
Tests for Graph Executor with Streaming

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from ragix_core.graph_executor import (
    GraphExecutor,
    SyncGraphExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    StreamEvent,
)
from ragix_core.agent_graph import (
    AgentGraph,
    AgentNode,
    AgentEdge,
    NodeStatus,
    TransitionCondition,
    create_linear_workflow,
)


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_event_creation(self):
        """Test creating a stream event."""
        event = StreamEvent(
            event_type="node_started",
            node_id="step1",
            data={"task": "Test task"},
        )

        assert event.event_type == "node_started"
        assert event.node_id == "step1"
        assert event.timestamp is not None

    def test_event_to_dict(self):
        """Test converting event to dict."""
        event = StreamEvent(
            event_type="node_completed",
            node_id="step1",
            data={"result": "success"},
        )

        d = event.to_dict()

        assert d["event_type"] == "node_completed"
        assert d["node_id"] == "step1"
        assert "timestamp" in d


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_context_creation(self):
        """Test creating an execution context."""
        context = ExecutionContext(workflow_id="wf_001")

        assert context.workflow_id == "wf_001"
        assert context.state == {}
        assert context.results == {}

    def test_set_get_result(self):
        """Test setting and getting node results."""
        context = ExecutionContext(workflow_id="wf_001")

        context.set_result("node1", {"output": "test"})
        result = context.get_result("node1")

        assert result == {"output": "test"}

    def test_get_nonexistent_result(self):
        """Test getting non-existent result returns None."""
        context = ExecutionContext(workflow_id="wf_001")
        result = context.get_result("nonexistent")
        assert result is None

    def test_set_get_error(self):
        """Test setting and getting node errors."""
        context = ExecutionContext(workflow_id="wf_001")

        error = ValueError("Test error")
        context.set_error("node1", error)

        assert context.get_error("node1") is error

    def test_has_failed_nodes(self):
        """Test checking for failed nodes."""
        context = ExecutionContext(workflow_id="wf_001")

        assert not context.has_failed_nodes()

        context.set_error("node1", Exception("Error"))
        assert context.has_failed_nodes()


class TestGraphExecutor:
    """Tests for GraphExecutor."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple 3-node linear graph."""
        graph = AgentGraph(name="test_workflow")
        graph.add_node(AgentNode(id="step1", agent_type="code", name="step1", config={"task": "Task 1"}))
        graph.add_node(AgentNode(id="step2", agent_type="code", name="step2", config={"task": "Task 2"}))
        graph.add_node(AgentNode(id="step3", agent_type="code", name="step3", config={"task": "Task 3"}))
        graph.add_edge(AgentEdge(source_id="step1", target_id="step2"))
        graph.add_edge(AgentEdge(source_id="step2", target_id="step3"))
        return graph

    @pytest.fixture
    def parallel_graph(self):
        """Create a graph with parallel branches."""
        graph = AgentGraph(name="parallel_workflow")
        graph.add_node(AgentNode(id="start", agent_type="code", name="start", config={"task": "Start"}))
        graph.add_node(AgentNode(id="branch1", agent_type="code", name="branch1", config={"task": "Branch 1"}))
        graph.add_node(AgentNode(id="branch2", agent_type="code", name="branch2", config={"task": "Branch 2"}))
        graph.add_node(AgentNode(id="end", agent_type="code", name="end", config={"task": "End"}))

        graph.add_edge(AgentEdge(source_id="start", target_id="branch1"))
        graph.add_edge(AgentEdge(source_id="start", target_id="branch2"))
        graph.add_edge(AgentEdge(source_id="branch1", target_id="end"))
        graph.add_edge(AgentEdge(source_id="branch2", target_id="end"))

        return graph

    @pytest.fixture
    def mock_agent_factory(self):
        """Create a mock agent factory."""
        def factory(workflow_id, node):
            agent = Mock()
            agent.run = AsyncMock(return_value=f"Result from {node.id}")
            return agent
        return factory

    @pytest.fixture
    def failing_agent_factory(self):
        """Create a factory that produces failing agents."""
        def factory(workflow_id, node):
            agent = Mock()
            if node.id == "step2":
                agent.run = AsyncMock(side_effect=Exception("Test failure"))
            else:
                agent.run = AsyncMock(return_value=f"Result from {node.id}")
            return agent
        return factory

    @pytest.mark.asyncio
    async def test_execute_simple_graph(self, simple_graph, mock_agent_factory):
        """Test executing a simple linear graph."""
        executor = GraphExecutor(simple_graph)
        result = await executor.execute(mock_agent_factory)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.completed_nodes) == 3
        assert "step1" in result.completed_nodes
        assert "step2" in result.completed_nodes
        assert "step3" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_execute_parallel_graph(self, parallel_graph, mock_agent_factory):
        """Test executing a graph with parallel branches."""
        executor = GraphExecutor(parallel_graph)
        result = await executor.execute(mock_agent_factory, max_parallel=2)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.completed_nodes) == 4

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, simple_graph, failing_agent_factory):
        """Test execution with a failing node."""
        executor = GraphExecutor(simple_graph)
        result = await executor.execute(failing_agent_factory)

        assert result.status == ExecutionStatus.FAILED
        assert "step2" in result.failed_nodes
        assert "step1" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_stop_on_failure(self, simple_graph, failing_agent_factory):
        """Test stop_on_failure option."""
        executor = GraphExecutor(simple_graph, stop_on_failure=True)
        result = await executor.execute(failing_agent_factory)

        assert result.status == ExecutionStatus.FAILED
        # step3 should not be executed when stop_on_failure is True
        assert "step3" not in result.completed_nodes

    @pytest.mark.asyncio
    async def test_cancel_execution(self, simple_graph):
        """Test cancelling execution via streaming (which supports cancel)."""
        async def slow_run(ctx):
            await asyncio.sleep(5)
            return "Done"

        def slow_factory(wid, node):
            agent = Mock(spec=["run"])
            agent.run = slow_run
            return agent

        executor = GraphExecutor(simple_graph)

        async def run_streaming():
            async for item in executor.execute_streaming(slow_factory):
                pass

        # Start streaming execution in background
        task = asyncio.create_task(run_streaming())

        # Cancel after short delay
        await asyncio.sleep(0.1)
        executor.cancel()

        await task
        assert executor.status == ExecutionStatus.CANCELLED


class TestGraphExecutorStreaming:
    """Tests for streaming execution."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple 2-node graph."""
        graph = AgentGraph(name="test_workflow")
        graph.add_node(AgentNode(id="step1", agent_type="code", name="step1", config={"task": "Task 1"}))
        graph.add_node(AgentNode(id="step2", agent_type="code", name="step2", config={"task": "Task 2"}))
        graph.add_edge(AgentEdge(source_id="step1", target_id="step2"))
        return graph

    @pytest.fixture
    def mock_agent_factory(self):
        """Create a mock agent factory (without run_streaming to avoid async iter issues)."""
        def factory(workflow_id, node):
            agent = Mock(spec=["run"])
            agent.run = AsyncMock(return_value=f"Result from {node.id}")
            return agent
        return factory

    @pytest.mark.asyncio
    async def test_streaming_yields_events(self, simple_graph, mock_agent_factory):
        """Test that streaming execution yields events."""
        executor = GraphExecutor(simple_graph)

        events = []
        async for item in executor.execute_streaming(mock_agent_factory):
            if isinstance(item, StreamEvent):
                events.append(item)

        # Should have various events
        event_types = [e.event_type for e in events]
        assert "workflow_started" in event_types
        assert "node_started" in event_types
        assert "node_completed" in event_types
        assert "workflow_completed" in event_types

    @pytest.mark.asyncio
    async def test_streaming_yields_final_result(self, simple_graph, mock_agent_factory):
        """Test that streaming yields final ExecutionResult."""
        executor = GraphExecutor(simple_graph)

        final_result = None
        async for item in executor.execute_streaming(mock_agent_factory):
            if isinstance(item, ExecutionResult):
                final_result = item

        assert final_result is not None
        assert final_result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_streaming_node_events_order(self, simple_graph, mock_agent_factory):
        """Test that node events come in correct order."""
        executor = GraphExecutor(simple_graph)

        node_events = []
        async for item in executor.execute_streaming(mock_agent_factory):
            if isinstance(item, StreamEvent) and item.node_id:
                node_events.append((item.node_id, item.event_type))

        # step1 events should come before step2 events
        step1_started = next(
            i for i, (n, e) in enumerate(node_events)
            if n == "step1" and e == "node_started"
        )
        step1_completed = next(
            i for i, (n, e) in enumerate(node_events)
            if n == "step1" and e == "node_completed"
        )
        step2_started = next(
            i for i, (n, e) in enumerate(node_events)
            if n == "step2" and e == "node_started"
        )

        assert step1_started < step1_completed < step2_started


class TestSyncGraphExecutor:
    """Tests for synchronous graph executor."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph."""
        return create_linear_workflow(
            name="test_sync",
            agent_sequence=[
                ("step1", "code", ["read_file"]),
                ("step2", "code", ["read_file"]),
            ],
        )

    def test_sync_execution(self, simple_graph):
        """Test synchronous execution."""
        def agent_factory(workflow_id, node):
            agent = Mock()
            # For sync tests, we need a sync-like async mock
            async def mock_run(ctx):
                return f"Result from {node.id}"
            agent.run = mock_run
            return agent

        executor = SyncGraphExecutor(simple_graph)
        result = executor.execute(agent_factory)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.completed_nodes) == 2


class TestConditionalTransitions:
    """Tests for conditional edge transitions."""

    @pytest.fixture
    def conditional_graph(self):
        """Create a graph with conditional transitions."""
        graph = AgentGraph(name="conditional_workflow")
        graph.add_node(AgentNode(id="analyze", agent_type="code", name="analyze", config={"task": "Analyze"}))
        graph.add_node(AgentNode(id="fix", agent_type="code", name="fix", config={"task": "Fix (on success)"}))
        graph.add_node(AgentNode(id="report", agent_type="doc", name="report", config={"task": "Report (on failure)"}))

        graph.add_edge(AgentEdge(
            source_id="analyze",
            target_id="fix",
            condition=TransitionCondition.ON_SUCCESS,
        ))
        graph.add_edge(AgentEdge(
            source_id="analyze",
            target_id="report",
            condition=TransitionCondition.ON_FAILURE,
        ))

        return graph

    @pytest.mark.asyncio
    async def test_on_success_transition(self, conditional_graph):
        """Test ON_SUCCESS transition is taken on success."""
        def factory(workflow_id, node):
            agent = Mock()
            agent.run = AsyncMock(return_value="Success")
            return agent

        executor = GraphExecutor(conditional_graph)
        result = await executor.execute(factory)

        assert "analyze" in result.completed_nodes
        assert "fix" in result.completed_nodes
        assert "report" in result.skipped_nodes

    @pytest.mark.asyncio
    async def test_on_failure_transition(self, conditional_graph):
        """Test ON_FAILURE transition is taken on failure."""
        def factory(workflow_id, node):
            agent = Mock()
            if node.id == "analyze":
                agent.run = AsyncMock(side_effect=Exception("Analysis failed"))
            else:
                agent.run = AsyncMock(return_value="Success")
            return agent

        executor = GraphExecutor(conditional_graph)
        result = await executor.execute(factory)

        assert "analyze" in result.failed_nodes
        assert "report" in result.completed_nodes
        assert "fix" in result.skipped_nodes
