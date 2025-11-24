"""
Git Agent - Specialized for git operations and repository analysis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..agent_graph import AgentNode
from ..graph_executor import ExecutionContext


GIT_AGENT_SYSTEM_PROMPT = """You are a GitAgent specialized in git operations and repository analysis.

Your capabilities:
- Check repository status (git status, git diff)
- Analyze commit history (git log, git show)
- Search commit messages and code history (git grep)
- Create commits with meaningful messages
- Manage branches
- Review changes before committing

Guidelines:
- Always check status before committing
- Write clear, descriptive commit messages
- Review diffs carefully before commits
- Follow git best practices
- Respect branch naming conventions
- Never force-push without explicit permission

Available tools: {tools}

Task: {task_description}
"""


class GitAgent(BaseAgent):
    """
    Agent specialized for git operations.

    Capabilities:
    - Git status and diff analysis
    - Commit history exploration
    - Branch management
    - Commit creation
    """

    def __init__(self, workflow_id: str, node: AgentNode, config: Optional[AgentConfig] = None):
        super().__init__(workflow_id, node, config)
        self.capabilities = [
            AgentCapability.GIT_OPERATIONS,
            AgentCapability.SEARCH
        ]

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        tools_str = ", ".join(self.node.tools)
        task_desc = self.node.config.get("task_description", "Perform git operations")

        return GIT_AGENT_SYSTEM_PROMPT.format(
            tools=tools_str,
            task_description=task_desc
        )

    async def run(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute git agent task."""
        self.log("Starting git agent execution")

        result = {
            "status": "success",
            "agent_type": "git_agent",
            "actions_taken": [],
            "git_operations": [],
            "changes_detected": [],
            "error": None
        }

        try:
            task = self.node.config.get("task", "")
            if not task:
                task = self.get_shared_state(context, "current_task")

            self.log(f"Task: {task}")

            # Placeholder implementation
            result["actions_taken"].append(f"Git operation: {task}")
            result["message"] = f"GitAgent completed task: {task}"

        except Exception as e:
            self.log(f"Error in git agent: {e}", "ERROR")
            result["status"] = "error"
            result["error"] = str(e)

        return result
