"""
Batch Mode Executor for CI/CD Integration

Enables running RAGIX agent workflows in CI pipelines with YAML configuration.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json
from datetime import datetime

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .agent_graph import AgentGraph, NodeStatus, create_linear_workflow
from .graph_executor import GraphExecutor, ExecutionContext, ExecutionResult, ExecutionStatus
from .agents import BaseAgent

logger = logging.getLogger(__name__)


class BatchExitCode(int, Enum):
    """Exit codes for batch execution."""

    SUCCESS = 0
    FAILURE = 1
    PARTIAL_SUCCESS = 2
    CONFIGURATION_ERROR = 10
    EXECUTION_ERROR = 20
    TIMEOUT = 30


@dataclass
class BatchConfig:
    """
    Configuration for batch execution.

    Loaded from YAML file or constructed programmatically.
    """

    name: str
    description: str = ""
    timeout: Optional[int] = None  # Overall timeout in seconds
    fail_fast: bool = True  # Stop on first failure
    max_parallel: int = 1  # Max parallel agents
    sandbox_root: Optional[str] = None
    model: str = "mistral:instruct"
    profile: str = "dev"
    environment: Dict[str, str] = field(default_factory=dict)
    workflows: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "BatchConfig":
        """
        Load batch configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            BatchConfig instance

        Raises:
            ImportError: If PyYAML not installed
            FileNotFoundError: If file doesn't exist
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            timeout=data.get("timeout"),
            fail_fast=data.get("fail_fast", True),
            max_parallel=data.get("max_parallel", 1),
            sandbox_root=data.get("sandbox_root"),
            model=data.get("model", "mistral:instruct"),
            profile=data.get("profile", "dev"),
            environment=data.get("environment", {}),
            workflows=data.get("workflows", []),
        )

    def to_yaml(self, path: Path):
        """
        Save batch configuration to YAML file.

        Args:
            path: Path to output YAML file
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        data = {
            "name": self.name,
            "description": self.description,
            "timeout": self.timeout,
            "fail_fast": self.fail_fast,
            "max_parallel": self.max_parallel,
            "sandbox_root": self.sandbox_root,
            "model": self.model,
            "profile": self.profile,
            "environment": self.environment,
            "workflows": self.workflows,
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class WorkflowResult:
    """Result of a single workflow execution."""

    workflow_name: str
    status: ExecutionStatus
    duration: float
    nodes_completed: int
    nodes_failed: int
    nodes_skipped: int
    errors: List[str]
    output: Dict[str, Any]

    @property
    def success(self) -> bool:
        """Check if workflow succeeded."""
        return self.status == ExecutionStatus.COMPLETED and self.nodes_failed == 0


@dataclass
class BatchResult:
    """Result of batch execution."""

    config_name: str
    start_time: datetime
    end_time: datetime
    total_workflows: int
    successful_workflows: int
    failed_workflows: int
    workflow_results: List[WorkflowResult]
    exit_code: BatchExitCode

    @property
    def duration(self) -> float:
        """Total execution duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success(self) -> bool:
        """Check if batch succeeded."""
        return self.exit_code == BatchExitCode.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config_name": self.config_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration,
            "total_workflows": self.total_workflows,
            "successful_workflows": self.successful_workflows,
            "failed_workflows": self.failed_workflows,
            "exit_code": self.exit_code.value,
            "workflow_results": [
                {
                    "workflow_name": wr.workflow_name,
                    "status": wr.status.value,
                    "duration": wr.duration,
                    "nodes_completed": wr.nodes_completed,
                    "nodes_failed": wr.nodes_failed,
                    "nodes_skipped": wr.nodes_skipped,
                    "errors": wr.errors,
                    "success": wr.success,
                }
                for wr in self.workflow_results
            ],
        }

    def save_json(self, path: Path):
        """Save results to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'=' * 60}")
        print(f"Batch Execution Summary: {self.config_name}")
        print(f"{'=' * 60}")
        print(f"Duration: {self.duration:.2f}s")
        print(f"Workflows: {self.total_workflows} total, "
              f"{self.successful_workflows} successful, "
              f"{self.failed_workflows} failed")
        print(f"Exit Code: {self.exit_code.name} ({self.exit_code.value})")
        print(f"{'=' * 60}\n")

        for wr in self.workflow_results:
            status_icon = "✅" if wr.success else "❌"
            print(f"{status_icon} {wr.workflow_name}")
            print(f"   Status: {wr.status.name}")
            print(f"   Duration: {wr.duration:.2f}s")
            print(f"   Nodes: {wr.nodes_completed} completed, "
                  f"{wr.nodes_failed} failed, {wr.nodes_skipped} skipped")
            if wr.errors:
                print(f"   Errors: {len(wr.errors)}")
                for error in wr.errors[:3]:  # Show first 3 errors
                    print(f"     - {error}")
            print()


class BatchExecutor:
    """
    Executor for batch workflows in CI/CD.

    Runs multiple agent workflows sequentially or in parallel,
    tracks results, and generates reports.
    """

    def __init__(
        self,
        config: BatchConfig,
        agent_factory: Callable[[str, Any], BaseAgent],
        verbose: bool = False,
    ):
        """
        Initialize batch executor.

        Args:
            config: Batch configuration
            agent_factory: Factory function to create agents
            verbose: Enable verbose logging
        """
        self.config = config
        self.agent_factory = agent_factory
        self.verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)

    async def execute(self) -> BatchResult:
        """
        Execute all workflows in the batch.

        Returns:
            BatchResult with execution summary
        """
        start_time = datetime.now()
        workflow_results: List[WorkflowResult] = []
        successful = 0
        failed = 0

        logger.info(f"Starting batch execution: {self.config.name}")
        logger.info(f"Workflows: {len(self.config.workflows)}")

        for i, workflow_config in enumerate(self.config.workflows, 1):
            workflow_name = workflow_config.get("name", f"workflow_{i}")
            logger.info(f"[{i}/{len(self.config.workflows)}] Running workflow: {workflow_name}")

            try:
                result = await self._execute_workflow(workflow_config)
                workflow_results.append(result)

                if result.success:
                    successful += 1
                    logger.info(f"✅ Workflow '{workflow_name}' completed successfully")
                else:
                    failed += 1
                    logger.error(f"❌ Workflow '{workflow_name}' failed")

                    # Fail fast if configured
                    if self.config.fail_fast:
                        logger.warning("Fail-fast enabled. Stopping batch execution.")
                        break

            except Exception as e:
                logger.error(f"❌ Workflow '{workflow_name}' crashed: {e}", exc_info=True)
                failed += 1

                # Create failed result
                workflow_results.append(
                    WorkflowResult(
                        workflow_name=workflow_name,
                        status=ExecutionStatus.FAILED,
                        duration=0.0,
                        nodes_completed=0,
                        nodes_failed=1,
                        nodes_skipped=0,
                        errors=[str(e)],
                        output={},
                    )
                )

                if self.config.fail_fast:
                    logger.warning("Fail-fast enabled. Stopping batch execution.")
                    break

        end_time = datetime.now()

        # Determine exit code
        if failed == 0:
            exit_code = BatchExitCode.SUCCESS
        elif successful == 0:
            exit_code = BatchExitCode.FAILURE
        else:
            exit_code = BatchExitCode.PARTIAL_SUCCESS

        result = BatchResult(
            config_name=self.config.name,
            start_time=start_time,
            end_time=end_time,
            total_workflows=len(workflow_results),
            successful_workflows=successful,
            failed_workflows=failed,
            workflow_results=workflow_results,
            exit_code=exit_code,
        )

        logger.info(f"Batch execution completed: {exit_code.name}")
        return result

    async def _execute_workflow(self, workflow_config: Dict[str, Any]) -> WorkflowResult:
        """
        Execute a single workflow.

        Args:
            workflow_config: Workflow configuration dict

        Returns:
            WorkflowResult
        """
        workflow_name = workflow_config.get("name", "unnamed")
        workflow_type = workflow_config.get("type", "linear")

        # Build graph
        if workflow_type == "linear":
            steps = workflow_config.get("steps", [])
            graph = self._build_linear_graph(workflow_name, steps)
        elif workflow_type == "graph":
            graph_data = workflow_config.get("graph", {})
            graph = AgentGraph.from_dict(graph_data)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Create execution context
        context = ExecutionContext(
            workflow_id=workflow_name, shared_state=workflow_config.get("context", {})
        )

        # Execute
        executor = GraphExecutor(graph)
        exec_result: ExecutionResult = await executor.execute(
            self.agent_factory, context, max_parallel=self.config.max_parallel
        )

        # Count results
        nodes_completed = sum(
            1 for node_id, status in exec_result.node_results.items() if status == NodeStatus.COMPLETED
        )
        nodes_failed = sum(
            1 for node_id, status in exec_result.node_results.items() if status == NodeStatus.FAILED
        )
        nodes_skipped = sum(
            1 for node_id, status in exec_result.node_results.items() if status == NodeStatus.SKIPPED
        )

        return WorkflowResult(
            workflow_name=workflow_name,
            status=exec_result.status,
            duration=exec_result.duration,
            nodes_completed=nodes_completed,
            nodes_failed=nodes_failed,
            nodes_skipped=nodes_skipped,
            errors=list(exec_result.errors.values()),
            output=exec_result.results,
        )

    def _build_linear_graph(self, name: str, steps: List[Dict[str, Any]]) -> AgentGraph:
        """
        Build linear workflow graph from step list.

        Args:
            name: Workflow name
            steps: List of step configurations

        Returns:
            AgentGraph
        """
        step_specs = []
        for step in steps:
            step_name = step.get("name", "step")
            agent_type = step.get("agent", "code_agent")
            tools = step.get("tools", [])
            step_specs.append((step_name, agent_type, tools))

        return create_linear_workflow(name, step_specs)


def run_batch_sync(
    config: BatchConfig,
    agent_factory: Callable[[str, Any], BaseAgent],
    verbose: bool = False,
) -> BatchResult:
    """
    Synchronous wrapper for batch execution.

    Args:
        config: Batch configuration
        agent_factory: Factory function to create agents
        verbose: Enable verbose logging

    Returns:
        BatchResult
    """
    executor = BatchExecutor(config, agent_factory, verbose)
    return asyncio.run(executor.execute())
