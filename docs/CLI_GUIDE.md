# RAGIX CLI Guide

**Version:** 0.20.0 | **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Updated:** 2025-11-28

---

Welcome to the RAGIX Command-Line Interface (CLI) suite. This guide provides a comprehensive reference for all `ragix-*` commands that form the backbone of the RAGIX ecosystem.

## Table of Contents

1. [Installation](#1-installation)
2. [Command Overview](#2-command-overview)
3. [Command Reference](#3-command-reference)
4. [Common Workflows](#4-common-workflows)
5. [Environment Variables](#5-environment-variables)

---

## 1. Installation

The RAGIX CLI tools are automatically installed when you install the `ragix` package:

```bash
# Standard installation
pip install -e .

# With all optional features
pip install -e ".[all]"

# Verify installation
ragix --version
ragix-ast --help
```

After installation, the following commands are available in your shell.

---

## 2. Command Overview

| Command | Purpose | Primary Use Case |
|---------|---------|------------------|
| `ragix` | Main entry point, system info | Check installation, version |
| `ragix-unix-agent` | Interactive chat agent | Conversational development |
| `ragix-ast` | Code analysis (AST) | Static analysis, metrics |
| `ragix-web` | Web UI server | Visual interface |
| `ragix-index` | Build search indexes | Enable semantic search |
| `ragix-batch` | Run workflows | CI/CD automation |
| `ragix-vault` | Secret management | Secure credential storage |
| `ragix-wasp` | Plugin management | Manage WASP tools |

---

## 3. Command Reference

### `ragix`

The main entry point for RAGIX. Displays version and system information.

```bash
ragix --version          # Show version
ragix --check            # Verify dependencies
ragix --help             # Show help
```

---

### `ragix-unix-agent`

The primary, interactive RAGIX agent providing a chat-like interface for natural language interactions.

**Usage:**
```bash
ragix-unix-agent [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--model MODEL` | Ollama model to use | `mistral` |
| `--profile PROFILE` | Safety profile (`strict`, `dev`, `unsafe`) | `dev` |
| `--sandbox-root PATH` | Working directory | Current directory |
| `--max-tokens N` | Maximum response tokens | `4096` |
| `--temperature T` | LLM temperature (0.0-1.0) | `0.7` |
| `--verbose` | Enable verbose output | Disabled |

**Examples:**
```bash
# Start with default settings
ragix-unix-agent

# Use a specific model with strict safety
ragix-unix-agent --model mistral:instruct --profile strict

# Work in a specific project directory
ragix-unix-agent --sandbox-root ~/projects/myapp
```

**Interactive Commands:**
- Type your request in natural language
- `exit` or `quit` to end the session
- `clear` to clear conversation history
- `help` for in-session help

---

### `ragix-ast`

A powerful tool for deep code analysis using Abstract Syntax Trees (AST). This is one of RAGIX's most sophisticated features.

**Usage:**
```bash
ragix-ast <subcommand> [OPTIONS]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `scan` | Full project analysis |
| `parse` | Parse single file |
| `search` | Query symbols |
| `graph` | Generate dependency graph |
| `metrics` | Calculate code metrics |
| `matrix` | Generate DSM |
| `radial` | Radial explorer |
| `hotspots` | Find complex code |
| `cycles` | Detect circular dependencies |
| `maven` | Parse Maven POM files |
| `sonar` | Query SonarQube |

> **Note:** For comprehensive `ragix-ast` documentation, see the dedicated [AST Guide](AST_GUIDE.md).

**Quick Examples:**
```bash
# Full project scan
ragix-ast scan ./src --lang java

# Get code metrics
ragix-ast metrics ./src

# Find complexity hotspots
ragix-ast hotspots ./src --top 20

# Generate interactive dependency graph
ragix-ast graph ./src --format html --output deps.html
```

---

### `ragix-web`

Launches the RAGIX Web UI, a comprehensive graphical interface for chat, code analysis, and workflow management.

**Usage:**
```bash
ragix-web [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--host HOST` | Server host address | `127.0.0.1` |
| `--port PORT` | Server port | `8000` |
| `--reload` | Auto-reload on changes | Disabled |
| `--workers N` | Number of worker processes | `1` |

**Examples:**
```bash
# Start with defaults (localhost:8000)
ragix-web

# Start on all interfaces with custom port
ragix-web --host 0.0.0.0 --port 8080

# Development mode with auto-reload
ragix-web --reload
```

Once running, open **http://localhost:8000** in your browser.

---

### `ragix-index`

Creates and manages semantic and keyword-based indexes of your codebase for efficient retrieval-augmented generation (RAG).

**Usage:**
```bash
ragix-index <path_to_project> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir PATH` | Index output directory | `.ragix/index` |
| `--model MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `--no-bm25` | Skip BM25 keyword index | Disabled |
| `--no-vector` | Skip vector index | Disabled |
| `--chunk-size N` | Text chunk size | `512` |
| `--exclude PATTERN` | Exclude file patterns | None |

**Examples:**
```bash
# Index a project with defaults
ragix-index ./my-project

# Custom output directory and model
ragix-index ./src --output-dir ./.ragix --model all-mpnet-base-v2

# Index only keyword (BM25)
ragix-index ./src --no-vector
```

---

### `ragix-batch`

Executes automated, multi-step workflows defined in YAML playbooks. Ideal for CI/CD pipelines and non-interactive automation.

**Usage:**
```bash
ragix-batch <playbook.yaml> [OPTIONS]
# or
ragix-batch --template <template_name> --params "key=value,..."
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--template NAME` | Use built-in template | None |
| `--params PARAMS` | Parameters for template | None |
| `--verbose` | Detailed output | Disabled |
| `--dry-run` | Show plan without executing | Disabled |
| `--list-templates` | List available templates | - |

**Examples:**
```bash
# List available templates
ragix-batch --list-templates

# Run built-in bug fix workflow
ragix-batch --template bug_fix --params "bug_description=TypeError in api.py"

# Run custom playbook
ragix-batch ./ci_playbook.yaml --verbose

# Preview without executing
ragix-batch ./deploy.yaml --dry-run
```

> **Note:** For playbook authoring, see the [Playbook Guide](PLAYBOOK_GUIDE.md).

---

### `ragix-vault`

A secure, encrypted vault for managing API keys, tokens, and other secrets.

**Usage:**
```bash
ragix-vault <subcommand> [OPTIONS]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `init` | Initialize a new vault |
| `set KEY VALUE` | Store a secret |
| `get KEY` | Retrieve a secret |
| `list` | List stored keys |
| `delete KEY` | Remove a secret |

**Examples:**
```bash
# Initialize vault (interactive password prompt)
ragix-vault init

# Store a secret
ragix-vault set GITHUB_TOKEN "ghp_xxxx..."

# Retrieve a secret
ragix-vault get GITHUB_TOKEN

# List all keys
ragix-vault list
```

---

### `ragix-wasp`

Manages WebAssembly Sandboxed Plugins (WASP) - secure, deterministic tools for agents.

**Usage:**
```bash
ragix-wasp <subcommand> [OPTIONS]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List available WASP tools |
| `info TOOL` | Show tool details |
| `run TOOL ARGS` | Execute a tool |
| `validate FILE` | Validate a WASP definition |

**Examples:**
```bash
# List all available WASP tools
ragix-wasp list

# Get info about a specific tool
ragix-wasp info validate_json

# Run a tool directly
ragix-wasp run format_json '{"key": "value"}'
```

---

## 4. Common Workflows

### First-Time Setup
```bash
# 1. Install Ollama and a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# 2. Clone and install RAGIX
git clone https://github.com/ovitrac/RAGIX.git
cd RAGIX
pip install -e ".[all]"

# 3. Verify installation
ragix --check
```

### Analyze a New Codebase
```bash
# 1. Index the project
ragix-index ./project --output-dir ./project/.ragix

# 2. Get code metrics
ragix-ast metrics ./project

# 3. Find hotspots
ragix-ast hotspots ./project --top 10

# 4. Generate visualization
ragix-ast graph ./project --format html --output analysis.html
```

### Interactive Development Session
```bash
# Start agent in project directory
cd ~/projects/myapp
ragix-unix-agent --profile dev
```

### CI/CD Integration
```bash
# Run quality checks
ragix-batch --template code_review --params "target=./src"

# Run with strict safety
UNIX_RAG_PROFILE=strict ragix-batch ./ci.yaml
```

---

## 5. Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UNIX_RAG_MODEL` | Default Ollama model | `mistral` |
| `UNIX_RAG_SANDBOX` | Sandbox root directory | Current directory |
| `UNIX_RAG_PROFILE` | Safety profile | `dev` |
| `RAGIX_CACHE_TYPE` | Cache backend (`memory`, `disk`) | `memory` |
| `RAGIX_LOG_LEVEL` | Logging level | `INFO` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

**Example `.bashrc` configuration:**
```bash
export UNIX_RAG_MODEL="mistral:instruct"
export UNIX_RAG_SANDBOX="$HOME/projects"
export UNIX_RAG_PROFILE="dev"
export RAGIX_LOG_LEVEL="INFO"
```

---

*For more information, see the [main documentation](../README.md) or run any command with `--help`.*
