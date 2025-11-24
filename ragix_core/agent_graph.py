"""
Agent Graph Data Structures for Multi-Agent Workflows

This module provides DAG-based workflow orchestration for RAGIX multi-agent systems.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path


class NodeStatus(str, Enum):
    """Execution status for agent nodes."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TransitionCondition(str, Enum):
    """Conditions for transitioning between nodes."""
    ON_SUCCESS = "on_success"  # Execute next node if current succeeds
    ON_FAILURE = "on_failure"  # Execute next node if current fails
    ON_COMPLETION = "on_completion"  # Execute regardless of success/failure
    CONDITIONAL = "conditional"  # Execute based on output condition


@dataclass
class AgentNode:
    """
    A node in the agent workflow graph.

    Each node represents an agent with specific tools and execution conditions.

    Attributes:
        id: Unique identifier for this node
        agent_type: Type of agent (e.g., "code_agent", "doc_agent")
        name: Human-readable name
        tools: List of tool names this agent can use
        config: Agent-specific configuration
        entry_condition: Condition to start this node
        exit_condition: Condition to finish this node
        status: Current execution status
        result: Result data from execution (populated at runtime)
    """
    id: str
    agent_type: str
    name: str
    tools: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    entry_condition: Optional[str] = None
    exit_condition: Optional[str] = None
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "name": self.name,
            "tools": self.tools,
            "config": self.config,
            "entry_condition": self.entry_condition,
            "exit_condition": self.exit_condition,
            "status": self.status.value,
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentNode":
        """Create node from dictionary."""
        status = NodeStatus(data.get("status", "pending"))
        return cls(
            id=data["id"],
            agent_type=data["agent_type"],
            name=data["name"],
            tools=data.get("tools", []),
            config=data.get("config", {}),
            entry_condition=data.get("entry_condition"),
            exit_condition=data.get("exit_condition"),
            status=status,
            result=data.get("result")
        )


@dataclass
class AgentEdge:
    """
    An edge connecting two nodes in the workflow graph.

    Edges define transitions between agents based on conditions.

    Attributes:
        source_id: ID of source node
        target_id: ID of target node
        condition: Transition condition
        condition_expr: Optional expression for conditional transitions
        metadata: Additional edge metadata
    """
    source_id: str
    target_id: str
    condition: TransitionCondition = TransitionCondition.ON_SUCCESS
    condition_expr: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "condition": self.condition.value,
            "condition_expr": self.condition_expr,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentEdge":
        """Create edge from dictionary."""
        condition = TransitionCondition(data.get("condition", "on_success"))
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            condition=condition,
            condition_expr=data.get("condition_expr"),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentGraph:
    """
    Directed Acyclic Graph (DAG) for multi-agent workflows.

    Represents a workflow as a graph of agents (nodes) and transitions (edges).

    Attributes:
        name: Workflow name
        description: Workflow description
        nodes: Dictionary of node_id → AgentNode
        edges: List of AgentEdge connections
        metadata: Workflow-level metadata
    """
    name: str
    description: str = ""
    nodes: Dict[str, AgentNode] = field(default_factory=dict)
    edges: List[AgentEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: AgentNode) -> None:
        """Add a node to the graph."""
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists")
        self.nodes[node.id] = node

    def add_edge(self, edge: AgentEdge) -> None:
        """
        Add an edge to the graph.

        Validates that source and target nodes exist.
        """
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node '{edge.source_id}' not found")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node '{edge.target_id}' not found")
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[AgentNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_entry_nodes(self) -> List[AgentNode]:
        """
        Get entry nodes (nodes with no incoming edges).

        These are the starting points of the workflow.
        """
        target_ids = {edge.target_id for edge in self.edges}
        return [node for node in self.nodes.values() if node.id not in target_ids]

    def get_successors(self, node_id: str) -> List[AgentNode]:
        """Get all successor nodes for a given node."""
        successor_ids = [edge.target_id for edge in self.edges if edge.source_id == node_id]
        return [self.nodes[nid] for nid in successor_ids if nid in self.nodes]

    def get_predecessors(self, node_id: str) -> List[AgentNode]:
        """Get all predecessor nodes for a given node."""
        predecessor_ids = [edge.source_id for edge in self.edges if edge.target_id == node_id]
        return [self.nodes[nid] for nid in predecessor_ids if nid in self.nodes]

    def validate(self) -> List[str]:
        """
        Validate graph structure.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check for cycles (DAG requirement)
        if self._has_cycle():
            errors.append("Graph contains cycles (must be a DAG)")

        # Check for orphaned nodes (except entry nodes)
        entry_nodes = set(node.id for node in self.get_entry_nodes())
        for node_id in self.nodes:
            if node_id not in entry_nodes:
                predecessors = self.get_predecessors(node_id)
                if not predecessors:
                    errors.append(f"Node '{node_id}' is orphaned (no incoming edges and not an entry node)")

        # Check edge validity
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                errors.append(f"Edge source '{edge.source_id}' not found in nodes")
            if edge.target_id not in self.nodes:
                errors.append(f"Edge target '{edge.target_id}' not found in nodes")

        return errors

    def _has_cycle(self) -> bool:
        """Check if graph contains cycles using DFS."""
        visited = set()
        rec_stack = set()

        def visit(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for successor in self.get_successors(node_id):
                if successor.id not in visited:
                    if visit(successor.id):
                        return True
                elif successor.id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if visit(node_id):
                    return True

        return False

    def topological_sort(self) -> List[AgentNode]:
        """
        Perform topological sort on the graph.

        Returns nodes in execution order (dependencies first).
        Raises ValueError if graph contains cycles.
        """
        if self._has_cycle():
            raise ValueError("Cannot perform topological sort: graph contains cycles")

        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(self.nodes[node_id])

            for successor in self.get_successors(node_id):
                in_degree[successor.id] -= 1
                if in_degree[successor.id] == 0:
                    queue.append(successor.id)

        if len(result) != len(self.nodes):
            raise ValueError("Topological sort failed: graph contains cycles")

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGraph":
        """Create graph from dictionary."""
        graph = cls(
            name=data["name"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )

        # Add nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = AgentNode.from_dict(node_data)
            graph.nodes[node_id] = node

        # Add edges
        for edge_data in data.get("edges", []):
            edge = AgentEdge.from_dict(edge_data)
            graph.edges.append(edge)

        return graph

    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Serialize graph to JSON.

        Args:
            path: Optional path to write JSON file

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str, encoding="utf-8")

        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> "AgentGraph":
        """
        Deserialize graph from JSON.

        Args:
            json_str_or_path: JSON string or path to JSON file

        Returns:
            AgentGraph instance
        """
        # Try to parse as JSON string first
        try:
            data = json.loads(str(json_str_or_path))
            return cls.from_dict(data)
        except json.JSONDecodeError:
            # If parsing fails, try as file path
            path = Path(json_str_or_path)
            if path.exists():
                json_str = path.read_text(encoding="utf-8")
                data = json.loads(json_str)
                return cls.from_dict(data)
            else:
                raise ValueError(f"Invalid JSON string or file path: {json_str_or_path}")


def create_linear_workflow(
    name: str,
    agent_sequence: List[tuple[str, str, List[str]]],
    description: str = ""
) -> AgentGraph:
    """
    Convenience function to create a simple linear workflow.

    Args:
        name: Workflow name
        agent_sequence: List of (node_id, agent_type, tools) tuples
        description: Workflow description

    Returns:
        AgentGraph with linear chain of agents

    Example:
        graph = create_linear_workflow(
            "Code Review",
            [
                ("analyze", "code_agent", ["bash", "edit_file"]),
                ("test", "test_agent", ["bash"]),
                ("document", "doc_agent", ["edit_file"])
            ]
        )
    """
    graph = AgentGraph(name=name, description=description)

    # Create nodes
    for node_id, agent_type, tools in agent_sequence:
        node = AgentNode(
            id=node_id,
            agent_type=agent_type,
            name=node_id.replace("_", " ").title(),
            tools=tools
        )
        graph.add_node(node)

    # Create edges (linear chain)
    for i in range(len(agent_sequence) - 1):
        source_id = agent_sequence[i][0]
        target_id = agent_sequence[i + 1][0]
        edge = AgentEdge(source_id=source_id, target_id=target_id)
        graph.add_edge(edge)

    return graph
