"""
Base Agent Class for Multi-Agent System

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

from ..agent_graph import AgentNode
from ..graph_executor import ExecutionContext


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""
    CODE_ANALYSIS = "code_analysis"
    CODE_EDITING = "code_editing"
    DOCUMENTATION = "documentation"
    GIT_OPERATIONS = "git_operations"
    TEST_EXECUTION = "test_execution"
    SEARCH = "search"
    FILE_OPERATIONS = "file_operations"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    verbose: bool = False
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.

    All agents must implement the run() method which performs the agent's task
    and returns a result dictionary.

    Attributes:
        node: AgentNode configuration for this agent
        capabilities: Set of capabilities this agent has
        config: Agent configuration
    """

    def __init__(self, workflow_id: str, node: AgentNode, config: Optional[AgentConfig] = None):
        """
        Initialize agent.

        Args:
            workflow_id: ID of the workflow this agent belongs to
            node: AgentNode containing configuration
            config: Optional agent configuration
        """
        self.workflow_id = workflow_id
        self.node = node
        self.config = config or AgentConfig()
        self.capabilities: List[AgentCapability] = []

    @abstractmethod
    async def run(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            context: Execution context with shared state

        Returns:
            Dictionary with results (structure depends on agent type)
        """
        pass

    def can_use_tool(self, tool_name: str) -> bool:
        """
        Check if agent can use a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is in agent's tool list
        """
        return tool_name in self.node.tools

    def get_shared_state(self, context: ExecutionContext, key: str) -> Optional[Any]:
        """
        Get value from shared workflow state.

        Args:
            context: Execution context
            key: State key

        Returns:
            Value or None if not found
        """
        return context.state.get(key)

    def set_shared_state(self, context: ExecutionContext, key: str, value: Any) -> None:
        """
        Set value in shared workflow state.

        Args:
            context: Execution context
            key: State key
            value: Value to store
        """
        context.state[key] = value

    def get_predecessor_result(self, context: ExecutionContext, node_id: str) -> Optional[Any]:
        """
        Get result from a predecessor node.

        Args:
            context: Execution context
            node_id: ID of predecessor node

        Returns:
            Result or None if not found
        """
        return context.get_result(node_id)

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message (can be overridden for custom logging).

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        if self.config.verbose:
            print(f"[{level}] {self.node.id}: {message}")
