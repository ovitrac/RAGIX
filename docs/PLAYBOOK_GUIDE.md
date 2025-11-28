# RAGIX Playbook Authoring Guide

**Version:** 0.20.0 | **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Updated:** 2025-11-28

---

RAGIX Playbooks are YAML files that define automated, multi-step workflows executed by the `ragix-batch` command-line tool. They are ideal for CI/CD pipelines, automated code reviews, and other non-interactive, repeatable tasks.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Top-Level Structure](#2-top-level-structure)
3. [Defining Workflows](#3-defining-workflows)
4. [Defining Steps](#4-defining-steps)
5. [Variables and Parameters](#5-variables-and-parameters)
6. [Complete Examples](#6-complete-examples)
7. [Built-in Templates](#7-built-in-templates)
8. [Best Practices](#8-best-practices)

---

## 1. Introduction

A Playbook allows you to define one or more workflows, where each workflow consists of a series of steps. Each step is assigned to a specialized agent (e.g., `code_agent` or `test_agent`) that carries out a specific task.

**Key Benefits:**
- **Reproducibility:** Define once, run consistently
- **Automation:** Integrate with CI/CD pipelines
- **Modularity:** Compose complex workflows from simple steps
- **Traceability:** Complete audit trail of all actions

---

## 2. Top-Level Structure

A playbook YAML file starts with top-level configuration fields that apply to the entire run.

```yaml
# playbook.yaml

# Required: The overall name of the playbook
name: "CI Pipeline for MyProject"

# Optional: A brief description
description: "Runs linting, testing, and security checks."

# Optional: Default Ollama model for all agents
# Can be overridden at workflow or step level
model: "mistral:instruct"

# Optional: Default safety profile
# Options: "strict", "dev", "unsafe"
# Default: "dev"
profile: "strict"

# Optional: Stop on first failure
# Default: false
fail_fast: true

# Optional: Number of parallel workflows
# Default: 1
max_parallel: 2

# Optional: Global timeout in seconds
# Default: 3600 (1 hour)
timeout: 1800

# Required: List of workflows to execute
workflows:
  - # ... workflow definitions ...
```

### Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Human-readable playbook name |
| `description` | string | No | - | Short explanation of purpose |
| `model` | string | No | `mistral` | Default LLM model |
| `profile` | string | No | `dev` | Safety profile |
| `fail_fast` | boolean | No | `false` | Stop on first failure |
| `max_parallel` | integer | No | `1` | Concurrent workflows |
| `timeout` | integer | No | `3600` | Global timeout (seconds) |
| `workflows` | list | Yes | - | Workflow definitions |

---

## 3. Defining Workflows

Each item in the `workflows` list is a workflow object—a sequence of steps designed to accomplish a larger goal.

```yaml
workflows:
  - name: "Lint and Format"
    description: "Check code formatting and style."

    # Execution strategy
    # "linear": Steps run sequentially
    # "parallel": Independent steps run concurrently
    # "graph": Respect dependency declarations
    type: "linear"

    # Optional: Override model for this workflow
    model: "granite3.1-moe:3b"

    # Optional: Workflow-specific timeout
    timeout: 600

    # Optional: Continue to next workflow on failure
    continue_on_error: false

    # Required: Steps in this workflow
    steps:
      - # ... step definitions ...
```

### Workflow Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Workflow name |
| `description` | string | No | - | Workflow purpose |
| `type` | string | No | `linear` | Execution strategy |
| `model` | string | No | (inherited) | Override LLM model |
| `timeout` | integer | No | (inherited) | Workflow timeout |
| `continue_on_error` | boolean | No | `false` | Continue on step failure |
| `steps` | list | Yes | - | Step definitions |

---

## 4. Defining Steps

Each item in the `steps` list is a step object—the smallest unit of work, performed by a single agent.

```yaml
steps:
  - name: "run_ruff"
    description: "Run the ruff linter on all Python files."

    # Required: Agent type
    # Options: code_agent, test_agent, doc_agent, git_agent
    agent: "code_agent"

    # Required: Tools the agent can use
    tools: ["bash"]

    # Required: Natural language instruction
    prompt: |
      Run the ruff linter on all Python files in the project.
      Report any errors found and their locations.

    # Optional: Expected exit code for success
    # Default: 0
    expected_exit_code: 0

    # Optional: Step-specific timeout
    timeout: 120

    # Optional: Retry on failure
    retries: 2
    retry_delay: 5

    # Optional: Dependencies (for type: "graph")
    depends_on: ["previous_step_name"]

    # Optional: Condition for execution
    condition: "{{ previous_step.exit_code == 0 }}"
```

### Step Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Unique step name |
| `description` | string | No | - | Step purpose |
| `agent` | string | Yes | - | Agent type |
| `tools` | list | Yes | - | Allowed tool categories |
| `prompt` | string | Yes | - | Agent instruction |
| `expected_exit_code` | integer | No | `0` | Success exit code |
| `timeout` | integer | No | (inherited) | Step timeout |
| `retries` | integer | No | `0` | Retry attempts |
| `retry_delay` | integer | No | `0` | Seconds between retries |
| `depends_on` | list | No | `[]` | Step dependencies |
| `condition` | string | No | - | Execution condition |

### Agent Types

| Agent | Description | Typical Use |
|-------|-------------|-------------|
| `code_agent` | General code analysis and modification | Linting, refactoring, search |
| `test_agent` | Test execution and coverage | Running pytest, coverage |
| `doc_agent` | Documentation generation | README updates, docstrings |
| `git_agent` | Git operations | Commits, branch management |

### Tool Categories

| Tool | Description |
|------|-------------|
| `bash` | Shell command execution |
| `file` | File read/write operations |
| `search` | Code search (grep, find) |
| `ast` | AST analysis tools |

---

## 5. Variables and Parameters

Playbooks support variable substitution for dynamic configuration.

### Defining Parameters

```yaml
name: "Configurable Pipeline"

# Define parameters with defaults
parameters:
  - name: "target_dir"
    type: "string"
    default: "./src"
    description: "Directory to analyze"

  - name: "coverage_threshold"
    type: "integer"
    default: 80
    description: "Minimum coverage percentage"

workflows:
  - name: "Test with Coverage"
    steps:
      - name: "run_tests"
        agent: "test_agent"
        tools: ["bash"]
        prompt: |
          Run pytest with coverage on {{ target_dir }}.
          Fail if coverage is below {{ coverage_threshold }}%.
```

### Using Parameters

Pass parameters via CLI:

```bash
ragix-batch playbook.yaml --params "target_dir=./lib,coverage_threshold=90"
```

### Environment Variables

Access environment variables in prompts:

```yaml
prompt: |
  Deploy to {{ env.DEPLOY_ENV }} environment.
  Use API key from environment.
```

---

## 6. Complete Examples

### Example 1: CI/CD Pipeline

```yaml
name: "RAGIX CI/CD Pipeline"
description: "Full CI pipeline with linting, testing, and security checks."
fail_fast: true
max_parallel: 3
profile: "strict"

workflows:
  - name: "Code Quality"
    description: "Linting and formatting checks."
    type: "linear"
    steps:
      - name: "check_formatting"
        agent: "code_agent"
        tools: ["bash"]
        prompt: |
          Check if the codebase is correctly formatted with black.
          Exit with non-zero status if changes are needed.
        expected_exit_code: 0

      - name: "lint_code"
        agent: "code_agent"
        tools: ["bash"]
        prompt: |
          Run ruff linter on all Python files.
          Report any errors with file locations.

  - name: "Testing"
    description: "Unit and integration tests."
    type: "linear"
    steps:
      - name: "unit_tests"
        agent: "test_agent"
        tools: ["bash"]
        prompt: "Run pytest on tests/unit/ directory."
        timeout: 300
        retries: 1

      - name: "integration_tests"
        agent: "test_agent"
        tools: ["bash"]
        prompt: "Run pytest on tests/integration/ directory."
        timeout: 600

  - name: "Security"
    description: "Security scanning."
    type: "linear"
    steps:
      - name: "dependency_scan"
        agent: "code_agent"
        tools: ["bash"]
        prompt: |
          Run safety check on Python dependencies.
          Report any known vulnerabilities.

      - name: "code_security"
        agent: "code_agent"
        tools: ["bash"]
        prompt: |
          Run bandit security scanner on the codebase.
          Report any high-severity findings.
```

### Example 2: Code Review Workflow

```yaml
name: "Automated Code Review"
description: "Comprehensive code review for pull requests."
model: "mistral:instruct"
profile: "strict"

parameters:
  - name: "pr_branch"
    type: "string"
    required: true
    description: "Branch to review"

workflows:
  - name: "Review"
    type: "graph"
    steps:
      - name: "fetch_changes"
        agent: "git_agent"
        tools: ["bash"]
        prompt: "Fetch and list all changed files in {{ pr_branch }}."

      - name: "analyze_complexity"
        agent: "code_agent"
        tools: ["bash", "ast"]
        depends_on: ["fetch_changes"]
        prompt: |
          Analyze cyclomatic complexity of changed files.
          Flag any functions with complexity > 10.

      - name: "check_tests"
        agent: "test_agent"
        tools: ["bash"]
        depends_on: ["fetch_changes"]
        prompt: |
          Verify that new code has corresponding tests.
          Report coverage for changed files.

      - name: "generate_report"
        agent: "doc_agent"
        tools: ["file"]
        depends_on: ["analyze_complexity", "check_tests"]
        prompt: |
          Generate a code review summary report.
          Include complexity findings and test coverage.
```

### Example 3: Documentation Update

```yaml
name: "Documentation Refresh"
description: "Update project documentation from code."
model: "mistral"

workflows:
  - name: "Update Docs"
    steps:
      - name: "extract_api"
        agent: "doc_agent"
        tools: ["bash", "ast"]
        prompt: |
          Extract all public API functions and classes.
          Generate API reference in Markdown format.

      - name: "update_readme"
        agent: "doc_agent"
        tools: ["file"]
        prompt: |
          Update the README.md with:
          - Current feature list
          - Installation instructions
          - Quick start examples

      - name: "generate_changelog"
        agent: "git_agent"
        tools: ["bash"]
        prompt: |
          Generate changelog entries from recent commits.
          Group by type: features, fixes, documentation.
```

---

## 7. Built-in Templates

RAGIX provides built-in templates for common tasks.

### List Available Templates

```bash
ragix-batch --list-templates
```

### Available Templates

| Template | Description | Required Parameters |
|----------|-------------|---------------------|
| `bug_fix` | Locate, diagnose, fix, test | `bug_description` |
| `feature_addition` | Design, implement, test, document | `feature_description` |
| `code_review` | Quality and security review | `target_path` |
| `refactoring` | Analyze, plan, refactor, verify | `target_symbol`, `goal` |
| `documentation` | Generate/update docs | `target_path` |
| `security_audit` | Static analysis, dependency scan | `target_path` |
| `test_coverage` | Coverage analysis, test generation | `target_path` |

### Using Templates

```bash
# Run bug fix workflow
ragix-batch --template bug_fix \
  --params "bug_description=TypeError in api/handlers.py line 45"

# Run code review
ragix-batch --template code_review \
  --params "target_path=./src"

# Run with verbose output
ragix-batch --template security_audit \
  --params "target_path=." \
  --verbose
```

---

## 8. Best Practices

### Design Principles

1. **Single Responsibility:** Each step should do one thing well
2. **Idempotency:** Steps should be safe to re-run
3. **Explicit Dependencies:** Use `depends_on` for clear ordering
4. **Meaningful Names:** Use descriptive step and workflow names

### Error Handling

```yaml
steps:
  - name: "critical_step"
    agent: "code_agent"
    tools: ["bash"]
    prompt: "Run critical operation."
    retries: 3
    retry_delay: 10

  - name: "cleanup"
    agent: "code_agent"
    tools: ["bash"]
    prompt: "Clean up temporary files."
    condition: "always"  # Run even if previous steps failed
```

### Performance Optimization

```yaml
# Parallel independent workflows
max_parallel: 4

workflows:
  # These will run in parallel
  - name: "Lint"
    # ...
  - name: "Type Check"
    # ...
  - name: "Unit Tests"
    # ...
```

### Security Considerations

1. **Use `strict` profile for CI/CD:** Prevents accidental modifications
2. **Validate inputs:** Check parameter values before use
3. **Limit tool access:** Only grant necessary tools to each step
4. **Review prompts:** Avoid dynamic command injection

### Debugging

```bash
# Dry run - show plan without executing
ragix-batch playbook.yaml --dry-run

# Verbose output for debugging
ragix-batch playbook.yaml --verbose

# Run specific workflow only
ragix-batch playbook.yaml --workflow "Testing"
```

---

*For more information, see the [CLI Guide](CLI_GUIDE.md) or the [Architecture](ARCHITECTURE.md) documentation.*
