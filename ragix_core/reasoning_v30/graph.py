"""
RAGIX v0.30 Reasoning Graph

Core graph orchestration for the Reflective Reasoning Graph:
- BaseNode: Abstract base class for all graph nodes
- ReasoningGraph: Orchestrates node execution with tracing

The graph implements a state machine where each node:
1. Receives a ReasoningState
2. Performs its logic
3. Returns (updated_state, next_node_name)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Callable
from datetime import datetime
import logging

from .types import ReasoningState, ReasoningEvent

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """
    Abstract base class for reasoning graph nodes.

    Each node must implement:
    - name: Class attribute identifying the node
    - run(state) -> (state, next_node_name): Execute node logic

    Nodes should be stateless - all state flows through ReasoningState.
    """
    name: str = "BASE"

    @abstractmethod
    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """
        Execute node logic.

        Args:
            state: Current reasoning state

        Returns:
            Tuple of (updated_state, next_node_name)
            Use "END" as next_node_name to terminate the graph.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


class ReasoningGraph:
    """
    Orchestrates reasoning as a directed graph of nodes.

    The graph:
    1. Starts at the 'start' node (default: "CLASSIFY")
    2. Executes nodes in sequence based on their return values
    3. Stops when reaching 'end' node (default: "END") or max_iterations

    Features:
    - Full execution trace for debugging
    - Event emission for experience corpus
    - Configurable iteration limits
    """

    def __init__(
        self,
        nodes: Dict[str, BaseNode],
        start: str = "CLASSIFY",
        end: str = "END",
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None,
    ):
        """
        Initialize the reasoning graph.

        Args:
            nodes: Dictionary mapping node names to BaseNode instances
            start: Name of the starting node
            end: Name of the terminal node
            emit_event_fn: Optional callback for emitting events to experience corpus
        """
        self.nodes = nodes
        self.start = start
        self.end = end
        self.emit_event_fn = emit_event_fn

        # Execution trace
        self.trace: List[str] = []
        self.trace_with_timestamps: List[Tuple[str, str, float]] = []

        # Validate graph structure
        self._validate()

    def _validate(self) -> None:
        """Validate graph structure."""
        if self.start not in self.nodes:
            raise ValueError(f"Start node '{self.start}' not found in nodes")

        # Check that all nodes are properly typed
        for name, node in self.nodes.items():
            if not isinstance(node, BaseNode):
                raise TypeError(f"Node '{name}' must be a BaseNode instance")

    def run(
        self,
        state: ReasoningState,
        max_iterations: int = 50
    ) -> ReasoningState:
        """
        Execute the graph until END or max_iterations.

        Args:
            state: Initial reasoning state
            max_iterations: Maximum number of node transitions (prevents infinite loops)

        Returns:
            Final reasoning state after graph execution
        """
        current = self.start
        iterations = 0

        # Reset trace for this run
        self.trace = []
        self.trace_with_timestamps = []

        logger.info(f"Starting reasoning graph from '{current}' with goal: {state.goal[:100]}...")

        while current != self.end and iterations < max_iterations:
            # Record trace
            start_time = datetime.utcnow()
            self.trace.append(current)

            # Get and execute node
            if current not in self.nodes:
                logger.error(f"Node '{current}' not found in graph")
                state.stop_reason = f"node_not_found:{current}"
                break

            node = self.nodes[current]
            logger.debug(f"Executing node: {current}")

            try:
                state, next_node = node.run(state)
                state.touch()
            except Exception as e:
                logger.error(f"Node '{current}' raised exception: {e}")
                state.last_error = str(e)
                state.stop_reason = f"node_exception:{current}"
                break

            # Record timing
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self.trace_with_timestamps.append((current, next_node, duration_ms))

            logger.debug(f"Node '{current}' -> '{next_node}' ({duration_ms:.1f}ms)")

            current = next_node
            iterations += 1

        # Check termination reason
        if iterations >= max_iterations:
            logger.warning(f"Reached max_iterations ({max_iterations})")
            if not state.stop_reason:
                state.stop_reason = "max_iterations"

        if current == self.end:
            logger.info(f"Graph completed normally with stop_reason: {state.stop_reason}")

        return state

    def get_trace_summary(self) -> str:
        """Get human-readable trace summary."""
        if not self.trace_with_timestamps:
            return "No trace available"

        lines = ["Graph Execution Trace:", "=" * 40]
        total_ms = 0

        for i, (node, next_node, duration_ms) in enumerate(self.trace_with_timestamps, 1):
            lines.append(f"{i:2d}. {node:15s} -> {next_node:15s} ({duration_ms:6.1f}ms)")
            total_ms += duration_ms

        lines.append("=" * 40)
        lines.append(f"Total: {len(self.trace)} nodes, {total_ms:.1f}ms")

        return "\n".join(lines)

    def emit_event(self, event: ReasoningEvent) -> None:
        """Emit an event to the experience corpus."""
        if self.emit_event_fn:
            try:
                self.emit_event_fn(event)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")


class GraphBuilder:
    """
    Builder pattern for constructing ReasoningGraph instances.

    Example:
        graph = (GraphBuilder()
            .add_node(ClassifyNode(...))
            .add_node(PlanNode(...))
            .add_node(ExecuteNode(...))
            .set_start("CLASSIFY")
            .set_end("END")
            .build())
    """

    def __init__(self):
        self._nodes: Dict[str, BaseNode] = {}
        self._start: str = "CLASSIFY"
        self._end: str = "END"
        self._emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None

    def add_node(self, node: BaseNode) -> "GraphBuilder":
        """Add a node to the graph."""
        self._nodes[node.name] = node
        return self

    def set_start(self, start: str) -> "GraphBuilder":
        """Set the starting node name."""
        self._start = start
        return self

    def set_end(self, end: str) -> "GraphBuilder":
        """Set the terminal node name."""
        self._end = end
        return self

    def set_event_emitter(
        self,
        emit_fn: Callable[[ReasoningEvent], None]
    ) -> "GraphBuilder":
        """Set the event emission callback."""
        self._emit_event_fn = emit_fn
        return self

    def build(self) -> ReasoningGraph:
        """Build and return the ReasoningGraph."""
        return ReasoningGraph(
            nodes=self._nodes,
            start=self._start,
            end=self._end,
            emit_event_fn=self._emit_event_fn,
        )


# Sentinel node for graph termination
class EndNode(BaseNode):
    """
    Terminal node that marks the end of graph execution.

    This node should never actually be executed - the graph stops
    when transitioning TO this node, not when executing it.
    """
    name = "END"

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """This should never be called - graph stops before reaching here."""
        return state, "END"
