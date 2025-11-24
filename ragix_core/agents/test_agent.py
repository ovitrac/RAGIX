"""
Test Agent - Specialized for running and analyzing tests

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..agent_graph import AgentNode
from ..graph_executor import ExecutionContext


TEST_AGENT_SYSTEM_PROMPT = """You are a TestAgent specialized in running tests and analyzing results.

Your capabilities:
- Run test suites (pytest, unittest, etc.)
- Analyze test failures and errors
- Identify failing tests and their causes
- Check test coverage
- Suggest test improvements

Guidelines:
- Run tests before making conclusions
- Analyze failure patterns
- Provide clear summaries of test results
- Identify root causes of failures
- Suggest fixes for failing tests
- Track test coverage metrics

Available tools: {tools}

Task: {task_description}
"""


class TestAgent(BaseAgent):
    """
    Agent specialized for test execution and analysis.

    Capabilities:
    - Test execution (pytest, unittest, etc.)
    - Failure analysis
    - Coverage checking
    - Test result interpretation
    """

    def __init__(self, workflow_id: str, node: AgentNode, config: Optional[AgentConfig] = None):
        super().__init__(workflow_id, node, config)
        self.capabilities = [
            AgentCapability.TEST_EXECUTION
        ]

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        tools_str = ", ".join(self.node.tools)
        task_desc = self.node.config.get("task_description", "Run and analyze tests")

        return TEST_AGENT_SYSTEM_PROMPT.format(
            tools=tools_str,
            task_description=task_desc
        )

    async def run(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute test agent task."""
        self.log("Starting test agent execution")

        result = {
            "status": "success",
            "agent_type": "test_agent",
            "actions_taken": [],
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": [],
            "error": None
        }

        try:
            task = self.node.config.get("task", "")
            if not task:
                task = self.get_shared_state(context, "current_task")

            self.log(f"Task: {task}")

            # Placeholder implementation
            result["actions_taken"].append(f"Test execution: {task}")
            result["message"] = f"TestAgent completed task: {task}"

        except Exception as e:
            self.log(f"Error in test agent: {e}", "ERROR")
            result["status"] = "error"
            result["error"] = str(e)

        return result
