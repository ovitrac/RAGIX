"""
Graph Executor for Multi-Agent Workflows

Executes agent graphs with state management, parallel execution, and error handling.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import asyncio
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .agent_graph import AgentGraph, AgentNode, AgentEdge, NodeStatus, TransitionCondition


class ExecutionStatus(str, Enum):
    """Overall execution status for workflow."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """
    Shared context for workflow execution.

    Contains state that persists across agent executions and can be
    shared between agents.

    Attributes:
        workflow_id: Unique identifier for this execution
        state: Shared state dictionary (agents can read/write)
        results: Results from completed nodes
        errors: Errors from failed nodes
        start_time: Workflow start timestamp
        end_time: Workflow end timestamp
        metadata: Additional execution metadata
    """
    workflow_id: str
    state: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Exception] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, node_id: str) -> Optional[Any]:
        """Get result from a completed node."""
        return self.results.get(node_id)

    def set_result(self, node_id: str, result: Any) -> None:
        """Store result from a node."""
        self.results[node_id] = result

    def get_error(self, node_id: str) -> Optional[Exception]:
        """Get error from a failed node."""
        return self.errors.get(node_id)

    def set_error(self, node_id: str, error: Exception) -> None:
        """Store error from a node."""
        self.errors[node_id] = error

    def has_failed_nodes(self) -> bool:
        """Check if any nodes have failed."""
        return len(self.errors) > 0


@dataclass
class ExecutionResult:
    """
    Result of workflow execution.

    Attributes:
        status: Overall execution status
        context: Execution context with results and state
        completed_nodes: List of successfully completed node IDs
        failed_nodes: List of failed node IDs
        skipped_nodes: List of skipped node IDs
        duration_seconds: Total execution time
        error: Exception if execution failed
    """
    status: ExecutionStatus
    context: ExecutionContext
    completed_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    skipped_nodes: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[Exception] = None


class GraphExecutor:
    """
    Executes agent workflow graphs.

    Handles:
    - Topological execution order
    - Parallel execution of independent branches
    - State management across agents
    - Error handling and rollback
    - Conditional transitions

    Example:
        executor = GraphExecutor(graph)
        result = await executor.execute(agent_factory)
    """

    def __init__(self, graph: AgentGraph):
        """
        Initialize executor.

        Args:
            graph: AgentGraph to execute
        """
        self.graph = graph
        self.status = ExecutionStatus.IDLE

        # Validate graph before execution
        errors = self.graph.validate()
        if errors:
            raise ValueError(f"Invalid graph: {', '.join(errors)}")

    async def execute(
        self,
        agent_factory: Callable[[str, AgentNode], Any],
        context: Optional[ExecutionContext] = None,
        max_parallel: int = 3
    ) -> ExecutionResult:
        """
        Execute the workflow graph.

        Args:
            agent_factory: Function that creates agent instances
                          Signature: (workflow_id, node) -> agent
            context: Optional existing execution context
            max_parallel: Maximum number of nodes to execute in parallel

        Returns:
            ExecutionResult with status and results
        """
        # Initialize context
        if context is None:
            context = ExecutionContext(
                workflow_id=f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                start_time=datetime.now()
            )

        self.status = ExecutionStatus.RUNNING
        result = ExecutionResult(
            status=ExecutionStatus.RUNNING,
            context=context
        )

        try:
            # Get execution order
            execution_order = self._compute_execution_order()

            # Execute nodes in order (with parallelization where possible)
            for level_nodes in execution_order:
                await self._execute_level(
                    level_nodes,
                    agent_factory,
                    context,
                    result,
                    max_parallel
                )

                # Check if we should stop early due to failures
                if self._should_stop_execution(context, result):
                    break

            # Finalize result
            context.end_time = datetime.now()
            if context.start_time:
                duration = (context.end_time - context.start_time).total_seconds()
                result.duration_seconds = duration

            # Determine final status
            if result.failed_nodes:
                result.status = ExecutionStatus.FAILED
                self.status = ExecutionStatus.FAILED
            else:
                result.status = ExecutionStatus.COMPLETED
                self.status = ExecutionStatus.COMPLETED

        except Exception as e:
            context.end_time = datetime.now()
            result.status = ExecutionStatus.FAILED
            result.error = e
            self.status = ExecutionStatus.FAILED

        return result

    def _compute_execution_order(self) -> List[List[AgentNode]]:
        """
        Compute execution order as levels of independent nodes.

        Each level contains nodes that can be executed in parallel.

        Returns:
            List of levels, where each level is a list of nodes
        """
        # Build dependency tracking
        in_degree = {node_id: 0 for node_id in self.graph.nodes}
        for edge in self.graph.edges:
            in_degree[edge.target_id] += 1

        # Find entry nodes (level 0)
        levels = []
        remaining_nodes = set(self.graph.nodes.keys())

        while remaining_nodes:
            # Find nodes with no remaining dependencies
            level_nodes = [
                self.graph.nodes[node_id]
                for node_id in remaining_nodes
                if in_degree[node_id] == 0
            ]

            if not level_nodes:
                # Should not happen with valid DAG
                raise ValueError("Circular dependency detected in graph")

            levels.append(level_nodes)

            # Update dependencies
            for node in level_nodes:
                remaining_nodes.remove(node.id)
                for successor in self.graph.get_successors(node.id):
                    in_degree[successor.id] -= 1

        return levels

    async def _execute_level(
        self,
        nodes: List[AgentNode],
        agent_factory: Callable,
        context: ExecutionContext,
        result: ExecutionResult,
        max_parallel: int
    ) -> None:
        """
        Execute all nodes in a level (potentially in parallel).

        Args:
            nodes: Nodes to execute
            agent_factory: Factory for creating agents
            context: Execution context
            result: Execution result to update
            max_parallel: Maximum parallel executions
        """
        # Create tasks for all nodes in this level
        tasks = []
        for node in nodes:
            # Check if node should be executed based on edge conditions
            if not self._should_execute_node(node, context):
                node.status = NodeStatus.SKIPPED
                result.skipped_nodes.append(node.id)
                continue

            task = self._execute_node(node, agent_factory, context, result)
            tasks.append(task)

        # Execute tasks with concurrency limit
        if tasks:
            # Use semaphore to limit parallelism
            semaphore = asyncio.Semaphore(max_parallel)

            async def bounded_task(task):
                async with semaphore:
                    return await task

            await asyncio.gather(*[bounded_task(t) for t in tasks], return_exceptions=True)

    async def _execute_node(
        self,
        node: AgentNode,
        agent_factory: Callable,
        context: ExecutionContext,
        result: ExecutionResult
    ) -> None:
        """
        Execute a single node.

        Args:
            node: Node to execute
            agent_factory: Factory for creating agent
            context: Execution context
            result: Execution result to update
        """
        node.status = NodeStatus.RUNNING

        try:
            # Create agent instance
            agent = agent_factory(context.workflow_id, node)

            # Execute agent (agent should have a step() or run() method)
            if hasattr(agent, 'run'):
                node_result = await agent.run(context)
            elif hasattr(agent, 'step'):
                node_result = await agent.step(context)
            else:
                raise AttributeError(f"Agent for node '{node.id}' has no run() or step() method")

            # Store result
            node.status = NodeStatus.COMPLETED
            node.result = node_result
            context.set_result(node.id, node_result)
            result.completed_nodes.append(node.id)

        except Exception as e:
            node.status = NodeStatus.FAILED
            context.set_error(node.id, e)
            result.failed_nodes.append(node.id)

    def _should_execute_node(self, node: AgentNode, context: ExecutionContext) -> bool:
        """
        Check if a node should be executed based on edge conditions.

        Args:
            node: Node to check
            context: Execution context

        Returns:
            True if node should execute, False to skip
        """
        # Get incoming edges
        incoming_edges = [
            edge for edge in self.graph.edges if edge.target_id == node.id
        ]

        # If no incoming edges (entry node), always execute
        if not incoming_edges:
            return True

        # Check each incoming edge condition
        for edge in incoming_edges:
            source_node = self.graph.get_node(edge.source_id)
            if not source_node:
                continue

            # Check condition
            if edge.condition == TransitionCondition.ON_SUCCESS:
                if source_node.status == NodeStatus.COMPLETED:
                    return True
            elif edge.condition == TransitionCondition.ON_FAILURE:
                if source_node.status == NodeStatus.FAILED:
                    return True
            elif edge.condition == TransitionCondition.ON_COMPLETION:
                if source_node.status in (NodeStatus.COMPLETED, NodeStatus.FAILED):
                    return True
            elif edge.condition == TransitionCondition.CONDITIONAL:
                # Evaluate condition expression
                if self._evaluate_condition(edge.condition_expr, source_node, context):
                    return True

        # If no conditions matched, skip this node
        return False

    def _evaluate_condition(
        self,
        condition_expr: Optional[str],
        source_node: AgentNode,
        context: ExecutionContext
    ) -> bool:
        """
        Evaluate a conditional expression.

        Args:
            condition_expr: Python expression to evaluate
            source_node: Source node for condition
            context: Execution context

        Returns:
            True if condition is satisfied
        """
        if not condition_expr:
            return True

        # Build evaluation context
        eval_context = {
            "result": source_node.result,
            "status": source_node.status.value,
            "context": context.state,
            "node": source_node
        }

        try:
            # Evaluate expression
            return bool(eval(condition_expr, {"__builtins__": {}}, eval_context))
        except Exception:
            # If evaluation fails, default to False (skip)
            return False

    def _should_stop_execution(
        self,
        context: ExecutionContext,
        result: ExecutionResult
    ) -> bool:
        """
        Check if execution should stop early.

        Args:
            context: Execution context
            result: Current execution result

        Returns:
            True if execution should stop
        """
        # For now, continue even if some nodes fail
        # In future, could add policies like "stop_on_first_failure"
        return False

    def cancel(self) -> None:
        """Cancel execution (if running)."""
        if self.status == ExecutionStatus.RUNNING:
            self.status = ExecutionStatus.CANCELLED


# Synchronous wrapper for simple use cases
class SyncGraphExecutor:
    """
    Synchronous wrapper for GraphExecutor.

    Provides a simpler interface for non-async code.
    """

    def __init__(self, graph: AgentGraph):
        self.executor = GraphExecutor(graph)

    def execute(
        self,
        agent_factory: Callable[[str, AgentNode], Any],
        context: Optional[ExecutionContext] = None,
        max_parallel: int = 1  # Default to sequential for sync
    ) -> ExecutionResult:
        """
        Execute workflow synchronously.

        Args:
            agent_factory: Function that creates agent instances
            context: Optional existing execution context
            max_parallel: Maximum parallel executions (default 1 for sync)

        Returns:
            ExecutionResult
        """
        # Run async executor in new event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.executor.execute(agent_factory, context, max_parallel)
        )
