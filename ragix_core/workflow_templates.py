"""
Workflow Templates - Pre-built workflows for common development tasks

Provides a template system for defining and instantiating multi-agent
workflows for bug fixing, feature development, code review, etc.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .agent_graph import AgentGraph, AgentNode, AgentEdge, TransitionCondition
from .agents import AgentCapability

logger = logging.getLogger(__name__)


@dataclass
class WorkflowParameter:
    """Definition of a workflow parameter."""

    name: str
    description: str
    param_type: str = "string"  # string, int, float, bool, list, path
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None

    def validate(self, value: Any) -> bool:
        """Validate a parameter value."""
        if value is None:
            return not self.required

        if self.enum and value not in self.enum:
            return False

        type_validators = {
            "string": lambda v: isinstance(v, str),
            "int": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, (int, float)),
            "bool": lambda v: isinstance(v, bool),
            "list": lambda v: isinstance(v, list),
            "path": lambda v: isinstance(v, (str, Path)),
        }

        validator = type_validators.get(self.param_type, lambda v: True)
        return validator(value)


@dataclass
class WorkflowStep:
    """A step in a workflow template."""

    name: str
    agent_type: str  # code, doc, git, test
    task_template: str
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    depends_on: List[str] = field(default_factory=list)
    conditions: Dict[str, str] = field(default_factory=dict)  # status -> target_step


@dataclass
class WorkflowTemplate:
    """A workflow template definition."""

    name: str
    description: str
    version: str = "1.0"
    parameters: List[WorkflowParameter] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters against template.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    errors.append(f"Missing required parameter: {param.name}")

        # Validate provided parameters
        for name, value in params.items():
            param = next((p for p in self.parameters if p.name == name), None)
            if param is None:
                errors.append(f"Unknown parameter: {name}")
            elif not param.validate(value):
                errors.append(f"Invalid value for {name}: {value}")

        return errors

    def substitute_params(self, text: str, params: Dict[str, Any]) -> str:
        """Substitute parameters in text."""
        result = text
        for name, value in params.items():
            placeholder = f"${{{name}}}"
            result = result.replace(placeholder, str(value))
        return result

    def instantiate(self, params: Dict[str, Any]) -> AgentGraph:
        """
        Create an AgentGraph from this template with given parameters.

        Args:
            params: Parameter values to substitute

        Returns:
            Configured AgentGraph
        """
        # Apply defaults
        full_params = {}
        for param in self.parameters:
            if param.default is not None:
                full_params[param.name] = param.default
        full_params.update(params)

        # Validate
        errors = self.validate_params(full_params)
        if errors:
            raise ValueError(f"Parameter validation failed: {errors}")

        # Create graph
        graph = AgentGraph(
            name=self.substitute_params(self.name, full_params),
            description=self.substitute_params(self.description, full_params),
        )

        # Add nodes
        node_map = {}
        for step in self.steps:
            # Parse capabilities
            capabilities = set()
            for cap_str in step.capabilities:
                try:
                    capabilities.add(AgentCapability[cap_str.upper()])
                except KeyError:
                    logger.warning(f"Unknown capability: {cap_str}")

            node = AgentNode(
                id=step.name,
                agent_type=step.agent_type,
                name=step.name,
                tools=step.tools,
                config={
                    "task": self.substitute_params(step.task_template, full_params),
                    "capabilities": [c.value for c in capabilities],
                    "max_iterations": step.max_iterations,
                },
            )
            graph.add_node(node)
            node_map[step.name] = node

        # Add edges based on dependencies and conditions
        for step in self.steps:
            # Dependency edges (must complete before this step)
            for dep_name in step.depends_on:
                if dep_name in node_map:
                    edge = AgentEdge(
                        source_id=dep_name,
                        target_id=step.name,
                        condition=TransitionCondition.ON_SUCCESS,
                    )
                    graph.add_edge(edge)

            # Conditional edges (status-based transitions)
            for status, target_name in step.conditions.items():
                if target_name in node_map:
                    try:
                        condition = TransitionCondition[status.upper()]
                    except KeyError:
                        condition = TransitionCondition.ON_SUCCESS
                    edge = AgentEdge(
                        source_id=step.name,
                        target_id=target_name,
                        condition=condition,
                    )
                    graph.add_edge(edge)

        return graph


class TemplateManager:
    """
    Manages workflow templates.

    Loads templates from YAML files and provides instantiation.
    """

    def __init__(self, template_dirs: Optional[List[Path]] = None):
        """
        Initialize template manager.

        Args:
            template_dirs: Directories to search for templates
        """
        self.template_dirs = template_dirs or []
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._builtin_loaded = False

    def add_template_dir(self, path: Path):
        """Add a directory to search for templates."""
        if path.exists() and path.is_dir():
            self.template_dirs.append(path)

    def load_builtin_templates(self):
        """Load built-in templates."""
        if self._builtin_loaded:
            return

        for name, template in BUILTIN_TEMPLATES.items():
            self.templates[name] = template

        self._builtin_loaded = True
        logger.info(f"Loaded {len(BUILTIN_TEMPLATES)} built-in templates")

    def load_from_file(self, path: Path) -> Optional[WorkflowTemplate]:
        """
        Load a template from a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            WorkflowTemplate or None if invalid
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            template = self._parse_template(data)
            self.templates[template.name] = template
            logger.info(f"Loaded template: {template.name} from {path}")
            return template

        except Exception as e:
            logger.error(f"Failed to load template from {path}: {e}")
            return None

    def load_from_directory(self, path: Path):
        """Load all templates from a directory."""
        if not path.exists():
            return

        for yaml_file in path.glob("*.yaml"):
            self.load_from_file(yaml_file)

        for yml_file in path.glob("*.yml"):
            self.load_from_file(yml_file)

    def load_all(self):
        """Load built-in templates and all from configured directories."""
        self.load_builtin_templates()

        for template_dir in self.template_dirs:
            self.load_from_directory(template_dir)

    def _parse_template(self, data: Dict[str, Any]) -> WorkflowTemplate:
        """Parse a template from dictionary data."""
        # Parse parameters
        parameters = []
        for param_data in data.get("parameters", []):
            param = WorkflowParameter(
                name=param_data["name"],
                description=param_data.get("description", ""),
                param_type=param_data.get("type", "string"),
                required=param_data.get("required", True),
                default=param_data.get("default"),
                enum=param_data.get("enum"),
            )
            parameters.append(param)

        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            step = WorkflowStep(
                name=step_data["name"],
                agent_type=step_data.get("agent", "code"),
                task_template=step_data["task"],
                capabilities=step_data.get("capabilities", []),
                tools=step_data.get("tools", []),
                max_iterations=step_data.get("max_iterations", 10),
                depends_on=step_data.get("depends_on", []),
                conditions=step_data.get("conditions", {}),
            )
            steps.append(step)

        return WorkflowTemplate(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            parameters=parameters,
            steps=steps,
            metadata=data.get("metadata", {}),
        )

    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())

    def instantiate(self, name: str, params: Dict[str, Any]) -> AgentGraph:
        """
        Instantiate a template by name.

        Args:
            name: Template name
            params: Parameter values

        Returns:
            Configured AgentGraph
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Unknown template: {name}")

        return template.instantiate(params)


# === BUILT-IN TEMPLATES ===

BUILTIN_TEMPLATES: Dict[str, WorkflowTemplate] = {
    "bug_fix": WorkflowTemplate(
        name="Bug Fix Workflow",
        description="Locate, diagnose, fix, and test a bug: ${bug_description}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="bug_description",
                description="Description of the bug to fix",
                param_type="string",
                required=True,
            ),
            WorkflowParameter(
                name="affected_files",
                description="Files likely affected by the bug",
                param_type="string",
                required=False,
                default="",
            ),
        ],
        steps=[
            WorkflowStep(
                name="locate",
                agent_type="code",
                task_template="Search the codebase to locate code related to: ${bug_description}. "
                "Look for error patterns, stack traces, or suspicious logic. "
                "Focus on files: ${affected_files}",
                capabilities=["code_read", "code_search"],
                tools=["grep_search", "semantic_search", "read_file", "find_files"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="diagnose",
                agent_type="code",
                task_template="Analyze the code found in the previous step to diagnose the root cause "
                "of: ${bug_description}. Identify the exact location and nature of the bug.",
                capabilities=["code_read", "code_analysis"],
                tools=["read_file", "semantic_search"],
                max_iterations=5,
                depends_on=["locate"],
            ),
            WorkflowStep(
                name="fix",
                agent_type="code",
                task_template="Implement a fix for: ${bug_description}. Make minimal, focused changes. "
                "Preserve existing behavior for non-buggy cases.",
                capabilities=["code_read", "code_write"],
                tools=["read_file", "edit_file", "write_file"],
                max_iterations=10,
                depends_on=["diagnose"],
            ),
            WorkflowStep(
                name="test",
                agent_type="test",
                task_template="Run existing tests and verify the fix for: ${bug_description}. "
                "Check for regressions.",
                capabilities=["test_run"],
                tools=["bash", "read_file"],
                max_iterations=5,
                depends_on=["fix"],
            ),
        ],
    ),
    "feature_addition": WorkflowTemplate(
        name="Feature Addition Workflow",
        description="Design, implement, and document a new feature: ${feature_name}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="feature_name",
                description="Name of the feature to implement",
                param_type="string",
                required=True,
            ),
            WorkflowParameter(
                name="feature_spec",
                description="Detailed specification of the feature",
                param_type="string",
                required=True,
            ),
            WorkflowParameter(
                name="target_module",
                description="Module where the feature should be implemented",
                param_type="string",
                required=False,
                default="",
            ),
        ],
        steps=[
            WorkflowStep(
                name="analyze",
                agent_type="code",
                task_template="Analyze the codebase to understand where ${feature_name} should be implemented. "
                "Target module: ${target_module}. Study existing patterns and conventions.",
                capabilities=["code_read", "code_search"],
                tools=["grep_search", "semantic_search", "read_file", "find_files", "list_directory"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="implement",
                agent_type="code",
                task_template="Implement ${feature_name} according to specification: ${feature_spec}. "
                "Follow existing code patterns. Add appropriate error handling.",
                capabilities=["code_read", "code_write"],
                tools=["read_file", "edit_file", "write_file", "bash"],
                max_iterations=15,
                depends_on=["analyze"],
            ),
            WorkflowStep(
                name="test",
                agent_type="test",
                task_template="Create tests for ${feature_name}. Cover happy path, edge cases, and error scenarios.",
                capabilities=["code_write", "test_run"],
                tools=["read_file", "write_file", "bash"],
                max_iterations=10,
                depends_on=["implement"],
            ),
            WorkflowStep(
                name="document",
                agent_type="doc",
                task_template="Document ${feature_name}. Add docstrings, update README if needed, "
                "add usage examples.",
                capabilities=["doc_write"],
                tools=["read_file", "edit_file", "write_file"],
                max_iterations=5,
                depends_on=["implement"],
            ),
        ],
    ),
    "code_review": WorkflowTemplate(
        name="Code Review Workflow",
        description="Perform automated code review on: ${target_path}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="target_path",
                description="Path to review (file or directory)",
                param_type="path",
                required=True,
            ),
            WorkflowParameter(
                name="review_focus",
                description="Specific aspects to focus on",
                param_type="string",
                required=False,
                default="quality, security, performance",
            ),
        ],
        steps=[
            WorkflowStep(
                name="inventory",
                agent_type="code",
                task_template="List and categorize files to review in ${target_path}. "
                "Identify file types, sizes, and structure.",
                capabilities=["code_read", "code_search"],
                tools=["find_files", "list_directory", "bash"],
                max_iterations=3,
            ),
            WorkflowStep(
                name="quality_review",
                agent_type="code",
                task_template="Review code quality in ${target_path}. Check for: ${review_focus}. "
                "Look for code smells, complexity issues, maintainability problems.",
                capabilities=["code_read", "code_analysis"],
                tools=["read_file", "grep_search"],
                max_iterations=10,
                depends_on=["inventory"],
            ),
            WorkflowStep(
                name="security_review",
                agent_type="code",
                task_template="Review security of code in ${target_path}. "
                "Check for vulnerabilities: injection, auth issues, data exposure.",
                capabilities=["code_read", "code_analysis"],
                tools=["read_file", "grep_search"],
                max_iterations=10,
                depends_on=["inventory"],
            ),
            WorkflowStep(
                name="report",
                agent_type="doc",
                task_template="Generate a code review report for ${target_path}. "
                "Summarize findings, prioritize issues, provide recommendations.",
                capabilities=["doc_write"],
                tools=["write_file"],
                max_iterations=5,
                depends_on=["quality_review", "security_review"],
            ),
        ],
    ),
    "refactoring": WorkflowTemplate(
        name="Refactoring Workflow",
        description="Identify and refactor code smells in: ${target_path}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="target_path",
                description="Path to refactor",
                param_type="path",
                required=True,
            ),
            WorkflowParameter(
                name="refactor_goal",
                description="Specific refactoring goal",
                param_type="string",
                required=False,
                default="improve readability and maintainability",
            ),
        ],
        steps=[
            WorkflowStep(
                name="analyze",
                agent_type="code",
                task_template="Analyze ${target_path} to identify refactoring opportunities. "
                "Goal: ${refactor_goal}. Look for duplicated code, long methods, complex conditionals.",
                capabilities=["code_read", "code_analysis"],
                tools=["read_file", "grep_search", "find_files"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="plan",
                agent_type="code",
                task_template="Create a refactoring plan for ${target_path}. "
                "List specific changes in priority order. Minimize risk.",
                capabilities=["code_read"],
                tools=["read_file"],
                max_iterations=3,
                depends_on=["analyze"],
            ),
            WorkflowStep(
                name="refactor",
                agent_type="code",
                task_template="Execute refactoring of ${target_path}. Make incremental changes. "
                "Preserve behavior. Goal: ${refactor_goal}.",
                capabilities=["code_read", "code_write"],
                tools=["read_file", "edit_file", "write_file"],
                max_iterations=15,
                depends_on=["plan"],
            ),
            WorkflowStep(
                name="verify",
                agent_type="test",
                task_template="Run tests to verify refactoring of ${target_path} didn't break anything.",
                capabilities=["test_run"],
                tools=["bash", "read_file"],
                max_iterations=5,
                depends_on=["refactor"],
            ),
        ],
    ),
    "documentation": WorkflowTemplate(
        name="Documentation Workflow",
        description="Generate or update documentation for: ${target_path}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="target_path",
                description="Path to document",
                param_type="path",
                required=True,
            ),
            WorkflowParameter(
                name="doc_type",
                description="Type of documentation",
                param_type="string",
                required=False,
                default="api",
                enum=["api", "readme", "tutorial", "reference"],
            ),
        ],
        steps=[
            WorkflowStep(
                name="analyze",
                agent_type="code",
                task_template="Analyze code structure in ${target_path}. "
                "Identify public APIs, classes, functions to document.",
                capabilities=["code_read", "code_search"],
                tools=["read_file", "grep_search", "find_files"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="generate",
                agent_type="doc",
                task_template="Generate ${doc_type} documentation for ${target_path}. "
                "Include descriptions, parameters, return values, examples.",
                capabilities=["doc_write"],
                tools=["read_file", "write_file", "edit_file"],
                max_iterations=10,
                depends_on=["analyze"],
            ),
            WorkflowStep(
                name="review",
                agent_type="doc",
                task_template="Review generated documentation for ${target_path}. "
                "Check accuracy, completeness, clarity.",
                capabilities=["doc_write"],
                tools=["read_file", "edit_file"],
                max_iterations=5,
                depends_on=["generate"],
            ),
        ],
    ),
    "security_audit": WorkflowTemplate(
        name="Security Audit Workflow",
        description="Perform security audit on: ${target_path}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="target_path",
                description="Path to audit",
                param_type="path",
                required=True,
            ),
            WorkflowParameter(
                name="severity_threshold",
                description="Minimum severity to report",
                param_type="string",
                required=False,
                default="medium",
                enum=["low", "medium", "high", "critical"],
            ),
        ],
        steps=[
            WorkflowStep(
                name="inventory",
                agent_type="code",
                task_template="Inventory ${target_path}. List files, dependencies, entry points. "
                "Identify sensitive areas (auth, crypto, data handling).",
                capabilities=["code_read", "code_search"],
                tools=["find_files", "grep_search", "read_file", "bash"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="static_analysis",
                agent_type="code",
                task_template="Perform static security analysis on ${target_path}. "
                "Check for: injection, XSS, auth bypass, data exposure, insecure configs. "
                "Report ${severity_threshold} and above.",
                capabilities=["code_read", "code_analysis"],
                tools=["read_file", "grep_search"],
                max_iterations=15,
                depends_on=["inventory"],
            ),
            WorkflowStep(
                name="dependency_check",
                agent_type="code",
                task_template="Check dependencies in ${target_path} for known vulnerabilities. "
                "Review package versions and security advisories.",
                capabilities=["code_read"],
                tools=["read_file", "bash"],
                max_iterations=5,
                depends_on=["inventory"],
            ),
            WorkflowStep(
                name="report",
                agent_type="doc",
                task_template="Generate security audit report for ${target_path}. "
                "List vulnerabilities by severity (${severity_threshold}+). "
                "Include remediation recommendations.",
                capabilities=["doc_write"],
                tools=["write_file"],
                max_iterations=5,
                depends_on=["static_analysis", "dependency_check"],
            ),
        ],
    ),
    "test_coverage": WorkflowTemplate(
        name="Test Coverage Workflow",
        description="Analyze and improve test coverage for: ${target_path}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="target_path",
                description="Path to analyze",
                param_type="path",
                required=True,
            ),
            WorkflowParameter(
                name="coverage_target",
                description="Target coverage percentage",
                param_type="int",
                required=False,
                default=80,
            ),
        ],
        steps=[
            WorkflowStep(
                name="analyze",
                agent_type="test",
                task_template="Analyze existing test coverage for ${target_path}. "
                "Identify untested code paths and low-coverage areas.",
                capabilities=["code_read", "test_run"],
                tools=["bash", "read_file", "grep_search"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="generate_tests",
                agent_type="test",
                task_template="Generate tests to improve coverage of ${target_path}. "
                "Target: ${coverage_target}%. Focus on critical paths first.",
                capabilities=["code_read", "code_write", "test_run"],
                tools=["read_file", "write_file", "bash"],
                max_iterations=15,
                depends_on=["analyze"],
            ),
            WorkflowStep(
                name="verify",
                agent_type="test",
                task_template="Run all tests for ${target_path} and verify coverage. "
                "Report final coverage metrics.",
                capabilities=["test_run"],
                tools=["bash", "read_file"],
                max_iterations=5,
                depends_on=["generate_tests"],
            ),
        ],
    ),
    "exploration": WorkflowTemplate(
        name="Codebase Exploration Workflow",
        description="Explore and understand: ${query}",
        version="1.0",
        parameters=[
            WorkflowParameter(
                name="query",
                description="What to explore or understand",
                param_type="string",
                required=True,
            ),
            WorkflowParameter(
                name="scope",
                description="Scope of exploration",
                param_type="string",
                required=False,
                default=".",
            ),
        ],
        steps=[
            WorkflowStep(
                name="overview",
                agent_type="code",
                task_template="Get a high-level overview of the codebase in ${scope}. "
                "Understand project structure, main components, key patterns.",
                capabilities=["code_read", "code_search"],
                tools=["find_files", "list_directory", "read_file", "project_overview"],
                max_iterations=5,
            ),
            WorkflowStep(
                name="search",
                agent_type="code",
                task_template="Search the codebase (${scope}) for: ${query}. "
                "Use semantic and keyword search to find relevant code.",
                capabilities=["code_read", "code_search"],
                tools=["semantic_search", "grep_search", "read_file"],
                max_iterations=10,
                depends_on=["overview"],
            ),
            WorkflowStep(
                name="analyze",
                agent_type="code",
                task_template="Analyze the code found to answer: ${query}. "
                "Trace control flow, understand data structures, identify patterns.",
                capabilities=["code_read", "code_analysis"],
                tools=["read_file", "grep_search"],
                max_iterations=10,
                depends_on=["search"],
            ),
            WorkflowStep(
                name="summarize",
                agent_type="doc",
                task_template="Summarize findings about: ${query}. "
                "Provide clear explanation with code references.",
                capabilities=["doc_write"],
                tools=["read_file", "write_file"],
                max_iterations=5,
                depends_on=["analyze"],
            ),
        ],
    ),
}


def get_template_manager(template_dirs: Optional[List[Path]] = None) -> TemplateManager:
    """
    Get a configured template manager.

    Args:
        template_dirs: Additional directories to search for templates

    Returns:
        TemplateManager with built-in templates loaded
    """
    manager = TemplateManager(template_dirs)
    manager.load_all()
    return manager


def list_builtin_templates() -> Dict[str, str]:
    """
    List built-in template names and descriptions.

    Returns:
        Dict of name -> description
    """
    return {name: template.description for name, template in BUILTIN_TEMPLATES.items()}
