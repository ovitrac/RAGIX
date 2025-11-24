# RAGIX CI/CD Integration Guide

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24

---

## Overview

RAGIX Batch Mode enables running agent workflows in CI/CD pipelines. This allows you to:

- **Automate code quality checks** with AI-powered analysis
- **Run tests with intelligent debugging** when failures occur
- **Generate documentation** automatically from code changes
- **Perform security audits** on commits and pull requests
- **Execute custom workflows** defined in YAML playbooks

---

## Quick Start

### 1. Install RAGIX with CI dependencies

```bash
pip install ragix[ci]
```

### 2. Create a playbook (`.ragix/ci_checks.yaml`)

```yaml
name: "CI Quality Checks"
description: "Automated linting and testing"

fail_fast: false
max_parallel: 2
model: "mistral:instruct"
profile: "safe-read-only"

workflows:
  - name: "Lint Python"
    type: "linear"
    steps:
      - name: "run_black"
        agent: "code_agent"
        tools: ["bash"]

  - name: "Run Tests"
    type: "linear"
    steps:
      - name: "pytest"
        agent: "test_agent"
        tools: ["bash"]
```

### 3. Run locally

```bash
ragix-batch .ragix/ci_checks.yaml --verbose
```

### 4. Integrate with CI platform

See platform-specific guides below.

---

## Playbook Configuration

### Top-level Options

```yaml
name: string               # Playbook name
description: string        # Optional description
fail_fast: boolean         # Stop on first failure (default: true)
max_parallel: int          # Max parallel agents (default: 1)
model: string              # LLM model (default: "mistral:instruct")
profile: string            # Security profile (default: "dev")
sandbox_root: string       # Working directory (default: ".")
timeout: int               # Overall timeout in seconds (optional)

environment:               # Environment variables
  KEY: "value"

workflows:                 # List of workflows
  - name: "..."
    # ...workflow configuration
```

### Workflow Types

#### Linear Workflow

Sequential execution of steps:

```yaml
workflows:
  - name: "My Workflow"
    type: "linear"
    steps:
      - name: "step1"
        agent: "code_agent"
        tools: ["bash", "search_project"]
        config:
          task: "Check code quality"

      - name: "step2"
        agent: "doc_agent"
        tools: ["bash"]
```

#### Graph Workflow

Complex workflows with parallel execution and conditionals:

```yaml
workflows:
  - name: "Complex Workflow"
    type: "graph"
    graph:
      name: "Parallel Checks"
      nodes:
        - id: "lint"
          agent_type: "code_agent"
          name: "Linting"
          tools: ["bash"]

        - id: "test"
          agent_type: "test_agent"
          name: "Testing"
          tools: ["bash"]

        - id: "report"
          agent_type: "doc_agent"
          name: "Generate Report"
          tools: ["bash", "edit_file"]

      edges:
        - from: "lint"
          to: "report"
          condition: "ON_SUCCESS"

        - from: "test"
          to: "report"
          condition: "ON_SUCCESS"
```

### Agent Types

- **`code_agent`**: Code analysis, editing, search
- **`doc_agent`**: Documentation generation, analysis
- **`git_agent`**: Git operations, history analysis
- **`test_agent`**: Test execution, debugging

### Available Tools

- **`bash`**: Execute shell commands
- **`edit_file`**: Modify files
- **`search_project`**: Semantic code search (requires index)

---

## GitHub Actions Integration

### Setup Workflow (`.github/workflows/ragix-ci.yml`)

```yaml
name: RAGIX CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  ragix-checks:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Ollama
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          ollama pull mistral:instruct

      - name: Install RAGIX
        run: pip install -e ".[ci]"

      - name: Run RAGIX checks
        run: ragix-batch .ragix/ci_checks.yaml --json-report results.json

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: ragix-results
          path: results.json
```

See `templates/ci/github_actions.yaml` for full template.

---

## GitLab CI Integration

### Setup Configuration (`.gitlab-ci.yml`)

```yaml
ragix-ci:
  image: python:3.10-slim
  stage: test

  before_script:
    - curl -fsSL https://ollama.com/install.sh | sh
    - ollama serve &
    - ollama pull mistral:instruct
    - pip install -e ".[ci]"

  script:
    - ragix-batch .ragix/ci_checks.yaml --json-report results.json

  artifacts:
    paths:
      - results.json
    when: always
```

See `templates/ci/gitlab_ci.yaml` for full template.

---

## Exit Codes

RAGIX batch execution uses standard exit codes:

| Code | Name | Description |
|------|------|-------------|
| 0 | SUCCESS | All workflows passed |
| 1 | FAILURE | All workflows failed |
| 2 | PARTIAL_SUCCESS | Some workflows passed |
| 10 | CONFIGURATION_ERROR | Invalid playbook |
| 20 | EXECUTION_ERROR | Runtime error |
| 30 | TIMEOUT | Execution timeout |

---

## CLI Options

```bash
ragix-batch <playbook.yaml> [options]

Options:
  --output-dir DIR       Output directory for reports
  --json-report FILE     Save JSON report to file
  --fail-fast            Stop on first failure (override)
  --no-fail-fast         Continue on failures (override)
  --max-parallel N       Max parallel agents (override)
  --verbose, -v          Verbose output
  --quiet, -q            Suppress output (errors only)
```

---

## Example Playbooks

### Linting Only

```yaml
name: "Linting"
fail_fast: false
max_parallel: 3

workflows:
  - name: "Black"
    type: "linear"
    steps:
      - name: "check_format"
        agent: "code_agent"
        tools: ["bash"]
        config:
          command: "black --check ."

  - name: "Ruff"
    type: "linear"
    steps:
      - name: "lint"
        agent: "code_agent"
        tools: ["bash"]
        config:
          command: "ruff check ."
```

### Full CI Pipeline

```yaml
name: "Full CI"
description: "Lint, test, build, deploy"

fail_fast: true
max_parallel: 2

workflows:
  - name: "Lint"
    type: "linear"
    steps:
      - name: "black"
        agent: "code_agent"
        tools: ["bash"]

  - name: "Test"
    type: "linear"
    steps:
      - name: "pytest"
        agent: "test_agent"
        tools: ["bash"]

  - name: "Security Scan"
    type: "linear"
    steps:
      - name: "bandit"
        agent: "code_agent"
        tools: ["bash"]
        config:
          command: "bandit -r . -f json"

  - name: "Build Docs"
    type: "linear"
    steps:
      - name: "sphinx"
        agent: "doc_agent"
        tools: ["bash"]
```

---

## Best Practices

### 1. Use Read-Only Profile for Safety

```yaml
profile: "safe-read-only"  # No file modifications in CI
```

### 2. Enable Fail-Fast for Critical Checks

```yaml
fail_fast: true  # Stop immediately on failure
```

### 3. Parallelize Independent Checks

```yaml
max_parallel: 3  # Run linters in parallel
```

### 4. Save JSON Reports

```bash
ragix-batch playbook.yaml --json-report results.json
```

Parse with `jq`:

```bash
jq '.successful_workflows' results.json
jq '.workflow_results[] | select(.success == false)' results.json
```

### 5. Use Timeouts

```yaml
timeout: 600  # 10 minutes max
```

### 6. Version Lock Models

```yaml
model: "mistral:7b-instruct-v0.2"  # Specific version
```

---

## Troubleshooting

### Issue: Ollama not found

**Solution:** Ensure Ollama is installed and in PATH:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:instruct
```

### Issue: Import errors

**Solution:** Install RAGIX with CI dependencies:

```bash
pip install ragix[ci]
```

### Issue: Playbook syntax error

**Solution:** Validate YAML syntax:

```bash
python -c "import yaml; yaml.safe_load(open('.ragix/ci_checks.yaml'))"
```

### Issue: Slow execution

**Solution:** Increase parallelism or use smaller model:

```yaml
max_parallel: 5
model: "mistral:7b-instruct"  # Smaller model
```

---

## Advanced: Custom Agents

Create custom agent types by extending `BaseAgent`:

```python
from ragix_core.agents import BaseAgent, AgentCapability

class SecurityAgent(BaseAgent):
    def __init__(self, workflow_id: str, node: AgentNode):
        super().__init__(workflow_id, node)
        self.capabilities = {AgentCapability.CODE_ANALYSIS}

    async def run(self, context: ExecutionContext) -> Dict[str, Any]:
        # Custom security logic
        pass
```

Register in agent factory:

```python
def factory(workflow_id: str, node: Any) -> BaseAgent:
    if node.agent_type == "security_agent":
        return SecurityAgent(workflow_id, node)
    # ...
```

---

## Resources

- **Example playbooks**: `templates/ci/`
- **CI templates**: `templates/ci/github_actions.yaml`, `templates/ci/gitlab_ci.yaml`
- **Core documentation**: `README.md`

---

**Questions?** Contact: olivier.vitrac@adservio.fr
