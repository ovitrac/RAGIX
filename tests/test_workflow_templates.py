"""
Tests for Workflow Template System

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import pytest
from pathlib import Path

from ragix_core.workflow_templates import (
    WorkflowParameter,
    WorkflowStep,
    WorkflowTemplate,
    TemplateManager,
    get_template_manager,
    list_builtin_templates,
    BUILTIN_TEMPLATES,
)


class TestWorkflowParameter:
    """Tests for WorkflowParameter."""

    def test_parameter_creation(self):
        """Test creating a parameter."""
        param = WorkflowParameter(
            name="bug_description",
            description="Description of the bug",
            param_type="string",
            required=True,
        )

        assert param.name == "bug_description"
        assert param.required is True

    def test_parameter_with_default(self):
        """Test parameter with default value."""
        param = WorkflowParameter(
            name="priority",
            description="Bug priority",
            param_type="string",
            required=False,
            default="medium",
        )

        assert param.default == "medium"

    def test_parameter_with_enum(self):
        """Test parameter with enum values."""
        param = WorkflowParameter(
            name="severity",
            description="Bug severity",
            param_type="string",
            required=True,
            enum=["low", "medium", "high", "critical"],
        )

        assert "high" in param.enum


class TestWorkflowStep:
    """Tests for WorkflowStep."""

    def test_step_creation(self):
        """Test creating a workflow step."""
        step = WorkflowStep(
            name="find_bug",
            agent_type="code",
            task_template="Search for the bug in ${target_file}",
            tools=["grep_search", "read_file"],
            max_iterations=10,
        )

        assert step.name == "find_bug"
        assert step.agent_type == "code"
        assert "grep_search" in step.tools

    def test_step_with_dependencies(self):
        """Test step with dependencies."""
        step = WorkflowStep(
            name="fix_bug",
            agent_type="code",
            task_template="Fix the bug",
            tools=["edit_file"],
            depends_on=["find_bug", "analyze_bug"],
        )

        assert "find_bug" in step.depends_on
        assert "analyze_bug" in step.depends_on


class TestWorkflowTemplate:
    """Tests for WorkflowTemplate."""

    def test_template_creation(self):
        """Test creating a workflow template."""
        template = WorkflowTemplate(
            name="test_workflow",
            description="Test workflow",
            version="1.0",
            parameters=[
                WorkflowParameter(
                    name="target",
                    description="Target file",
                    param_type="string",
                    required=True,
                ),
            ],
            steps=[
                WorkflowStep(
                    name="step1",
                    agent_type="code",
                    task_template="Process ${target}",
                    tools=["read_file"],
                ),
            ],
        )

        assert template.name == "test_workflow"
        assert len(template.parameters) == 1
        assert len(template.steps) == 1

    def test_template_validation_missing_required(self):
        """Test validation fails for missing required parameters."""
        template = WorkflowTemplate(
            name="test_workflow",
            description="Test",
            parameters=[
                WorkflowParameter(
                    name="required_param",
                    description="Required",
                    param_type="string",
                    required=True,
                ),
            ],
            steps=[],
        )

        errors = template.validate_params({})
        assert len(errors) > 0
        assert any("required_param" in e for e in errors)

    def test_template_validation_passes(self):
        """Test validation passes with valid parameters."""
        template = WorkflowTemplate(
            name="test_workflow",
            description="Test",
            parameters=[
                WorkflowParameter(
                    name="required_param",
                    description="Required",
                    param_type="string",
                    required=True,
                ),
            ],
            steps=[],
        )

        errors = template.validate_params({"required_param": "value"})
        assert len(errors) == 0

    def test_template_instantiation(self):
        """Test instantiating a template creates a graph."""
        template = WorkflowTemplate(
            name="test_workflow",
            description="Test workflow for ${target}",
            parameters=[
                WorkflowParameter(
                    name="target",
                    description="Target",
                    param_type="string",
                    required=True,
                ),
            ],
            steps=[
                WorkflowStep(
                    name="analyze",
                    agent_type="code",
                    task_template="Analyze ${target}",
                    tools=["read_file"],
                ),
                WorkflowStep(
                    name="process",
                    agent_type="code",
                    task_template="Process results from analyze",
                    tools=["edit_file"],
                    depends_on=["analyze"],
                ),
            ],
        )

        graph = template.instantiate({"target": "src/main.py"})

        assert graph is not None
        assert len(graph.nodes) == 2
        assert "analyze" in graph.nodes
        assert "process" in graph.nodes

    def test_parameter_substitution(self):
        """Test parameter substitution in tasks."""
        template = WorkflowTemplate(
            name="test_workflow",
            description="Test",
            parameters=[
                WorkflowParameter(
                    name="file_path",
                    description="File path",
                    param_type="string",
                    required=True,
                ),
                WorkflowParameter(
                    name="pattern",
                    description="Search pattern",
                    param_type="string",
                    required=True,
                ),
            ],
            steps=[
                WorkflowStep(
                    name="search",
                    agent_type="code",
                    task_template="Search for ${pattern} in ${file_path}",
                    tools=["grep_search"],
                ),
            ],
        )

        graph = template.instantiate({
            "file_path": "src/utils.py",
            "pattern": "def main",
        })

        # Check that the task has substituted values
        search_node = graph.nodes["search"]
        assert "src/utils.py" in search_node.config["task"]
        assert "def main" in search_node.config["task"]


class TestBuiltinTemplates:
    """Tests for built-in workflow templates."""

    def test_builtin_templates_exist(self):
        """Test that expected built-in templates exist."""
        templates = list_builtin_templates()

        expected = [
            "bug_fix",
            "feature_addition",
            "code_review",
            "refactoring",
            "documentation",
            "security_audit",
            "test_coverage",
            "exploration",
        ]

        for name in expected:
            assert name in templates, f"Missing template: {name}"

    def test_bug_fix_template(self):
        """Test bug_fix template structure."""
        template = BUILTIN_TEMPLATES["bug_fix"]

        assert template.name == "Bug Fix Workflow"
        assert len(template.parameters) > 0
        assert len(template.steps) > 0

        # Check required parameters
        param_names = [p.name for p in template.parameters]
        assert "bug_description" in param_names

    def test_feature_addition_template(self):
        """Test feature_addition template structure."""
        template = BUILTIN_TEMPLATES["feature_addition"]

        assert template.name == "Feature Addition Workflow"
        assert len(template.steps) >= 3  # Design, implement, test

    def test_all_templates_instantiate(self):
        """Test that all templates can be instantiated with defaults."""
        for name, template in BUILTIN_TEMPLATES.items():
            # Build params with defaults where possible
            params = {}
            for param in template.parameters:
                if param.required:
                    # Provide a test value
                    params[param.name] = "test_value"
                elif param.default:
                    params[param.name] = param.default

            try:
                graph = template.instantiate(params)
                assert graph is not None, f"Template {name} returned None"
                assert len(graph.nodes) > 0, f"Template {name} has no nodes"
            except Exception as e:
                pytest.fail(f"Template {name} failed to instantiate: {e}")


class TestTemplateManager:
    """Tests for TemplateManager."""

    def test_get_template_manager(self):
        """Test getting template manager."""
        manager = get_template_manager()

        assert isinstance(manager, TemplateManager)
        assert len(manager.list_templates()) > 0

    def test_list_templates(self):
        """Test listing available templates."""
        manager = get_template_manager()
        templates = manager.list_templates()

        assert len(templates) > 0
        assert "bug_fix" in templates

    def test_get_template(self):
        """Test getting a template by name."""
        manager = get_template_manager()
        template = manager.get_template("bug_fix")

        assert template is not None
        assert template.name == "Bug Fix Workflow"

    def test_get_nonexistent_template(self):
        """Test getting non-existent template returns None."""
        manager = get_template_manager()

        assert manager.get_template("nonexistent_template") is None

    def test_instantiate_template(self):
        """Test instantiating template through manager."""
        manager = get_template_manager()
        graph = manager.instantiate(
            "bug_fix",
            {"bug_description": "TypeError in handler"},
        )

        assert graph is not None
        assert len(graph.nodes) > 0

    def test_load_yaml_template(self, temp_dir: Path):
        """Test loading template from YAML file."""
        yaml_content = """
name: custom_workflow
description: A custom test workflow
version: "1.0"
parameters:
  - name: target
    description: Target to process
    type: string
    required: true
steps:
  - name: analyze
    agent: code
    task: "Analyze ${target}"
    tools:
      - read_file
      - grep_search
  - name: report
    agent: doc
    task: "Generate report for ${target}"
    depends_on:
      - analyze
"""
        yaml_path = temp_dir / "custom.yaml"
        yaml_path.write_text(yaml_content)

        manager = TemplateManager()
        manager.load_from_file(yaml_path)

        template = manager.get_template("custom_workflow")
        assert template is not None
        assert template.name == "custom_workflow"
        assert len(template.steps) == 2


class TestListBuiltinTemplates:
    """Tests for list_builtin_templates function."""

    def test_returns_dict(self):
        """Test that list_builtin_templates returns a dict."""
        templates = list_builtin_templates()
        assert isinstance(templates, dict)

    def test_contains_descriptions(self):
        """Test that returned dict has descriptions."""
        templates = list_builtin_templates()

        for name, description in templates.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 0
