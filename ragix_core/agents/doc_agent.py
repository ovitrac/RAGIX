"""
Documentation Agent - Specialized for documentation analysis and generation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..agent_graph import AgentNode
from ..graph_executor import ExecutionContext


DOC_AGENT_SYSTEM_PROMPT = """You are a DocAgent specialized in documentation analysis and generation.

Your capabilities:
- Analyze README files and documentation
- Check documentation quality and completeness
- Generate or improve docstrings
- Create user guides and API documentation
- Ensure documentation consistency

Guidelines:
- Focus on clarity and completeness
- Use consistent formatting and style
- Include examples where appropriate
- Keep documentation up-to-date with code
- Follow project documentation standards

Available tools: {tools}

Task: {task_description}
"""


class DocAgent(BaseAgent):
    """
    Agent specialized for documentation operations.

    Capabilities:
    - Documentation analysis
    - Docstring generation/improvement
    - README file management
    - Documentation quality checks
    """

    def __init__(self, workflow_id: str, node: AgentNode, config: Optional[AgentConfig] = None):
        super().__init__(workflow_id, node, config)
        self.capabilities = [
            AgentCapability.DOCUMENTATION,
            AgentCapability.FILE_OPERATIONS,
            AgentCapability.SEARCH
        ]

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        tools_str = ", ".join(self.node.tools)
        task_desc = self.node.config.get("task_description", "Analyze and improve documentation")

        return DOC_AGENT_SYSTEM_PROMPT.format(
            tools=tools_str,
            task_description=task_desc
        )

    async def run(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute documentation agent task."""
        self.log("Starting documentation agent execution")

        result = {
            "status": "success",
            "agent_type": "doc_agent",
            "actions_taken": [],
            "files_analyzed": [],
            "suggestions": [],
            "error": None
        }

        try:
            task = self.node.config.get("task", "")
            if not task:
                task = self.get_shared_state(context, "current_task")

            self.log(f"Task: {task}")

            # Placeholder implementation
            result["actions_taken"].append(f"Analyzed documentation for: {task}")
            result["message"] = f"DocAgent completed task: {task}"

        except Exception as e:
            self.log(f"Error in doc agent: {e}", "ERROR")
            result["status"] = "error"
            result["error"] = str(e)

        return result
