"""
Code Agent - Specialized for codebase exploration and editing

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentCapability, AgentConfig
from ..agent_graph import AgentNode
from ..graph_executor import ExecutionContext


# System prompt for CodeAgent
CODE_AGENT_SYSTEM_PROMPT = """You are a CodeAgent specialized in code exploration, analysis, and editing.

Your capabilities:
- Search code using grep/ripgrep patterns
- Semantic code search using vector index (search_project)
- Navigate files using rt tools (open, scroll, grep-file)
- Edit code using edit_file or rt edit commands
- Analyze code structure and dependencies
- Identify bugs and propose fixes
- Suggest refactorings
- Execute bash commands in sandboxed environment

Guidelines:
- Always inspect code before editing
- Use semantic search (search_project) for concept-based queries
- Use grep/ripgrep for exact pattern matching
- Prefer targeted edits over large-scale refactoring
- Verify changes with git diff after editing
- Keep changes focused on the task at hand
- Follow security policies and sandbox restrictions

Multi-Agent Workflow Context:
- You may be part of a larger workflow graph
- Share results via execution context for downstream agents
- Check shared state for inputs from upstream agents
- Document key findings in shared state

Available tools: {tools}
Workflow ID: {workflow_id}
Node ID: {node_id}

Task: {task_description}
"""


class CodeAgent(BaseAgent):
    """
    Agent specialized for codebase operations.

    Capabilities:
    - Code analysis and navigation
    - Bug localization
    - Code editing and refactoring
    - Pattern searching
    """

    def __init__(self, workflow_id: str, node: AgentNode, config: Optional[AgentConfig] = None):
        super().__init__(workflow_id, node, config)
        self.capabilities = [
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.CODE_EDITING,
            AgentCapability.SEARCH,
            AgentCapability.FILE_OPERATIONS
        ]

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        tools_str = ", ".join(self.node.tools)
        task_desc = self.node.config.get("task_description", "Analyze and edit code")

        return CODE_AGENT_SYSTEM_PROMPT.format(
            tools=tools_str,
            workflow_id=self.workflow_id,
            node_id=self.node.id,
            task_description=task_desc
        )

    async def run(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute code agent task.

        Args:
            context: Execution context

        Returns:
            Dictionary with:
            - status: "success" or "error"
            - actions_taken: List of actions performed
            - files_modified: List of modified files
            - error: Error message if failed
        """
        self.log("Starting code agent execution")

        result = {
            "status": "success",
            "agent_type": "code_agent",
            "actions_taken": [],
            "files_modified": [],
            "error": None
        }

        try:
            # Get task from node config
            task = self.node.config.get("task", "")
            if not task:
                task = self.get_shared_state(context, "current_task")

            self.log(f"Task: {task}")

            # Store task in shared state for other agents
            self.set_shared_state(context, f"{self.node.id}_task", task)

            # For now, this is a stub that would integrate with actual LLM/tools
            # In real implementation, this would:
            # 1. Use the system prompt
            # 2. Call LLM with available tools
            # 3. Execute tool calls (bash, search_project, edit_file)
            # 4. Iterate until task is complete

            # Placeholder: Mark as successful
            result["actions_taken"].append(f"Analyzed task: {task}")
            result["message"] = f"CodeAgent completed task: {task}"

        except Exception as e:
            self.log(f"Error in code agent: {e}", "ERROR")
            result["status"] = "error"
            result["error"] = str(e)

        return result
