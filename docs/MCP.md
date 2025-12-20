# Model Context Protocol (MCP) in RAGIX

**A Protocol-First Perspective for Collective AI Intelligence**

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.62.0
**Updated:** 2025-12-20

---

## Table of Contents

1. [What MCP Really Is](#1-what-mcp-really-is)
2. [Common Misconceptions](#2-common-misconceptions)
3. [Architecture Principles](#3-architecture-principles)
4. [RAGIX MCP Implementation](#4-ragix-mcp-implementation)
5. [Complete Tool Reference](#5-complete-tool-reference)
6. [Deployment Topologies](#6-deployment-topologies)
7. [Hybrid Backends: Stochastic, Deterministic, and Combined](#7-hybrid-backends)
8. [Collective Intelligence Patterns](#8-collective-intelligence-patterns)
9. [Frequently Asked Questions](#9-frequently-asked-questions)
10. [Related Documentation](#10-related-documentation)

---

## 1. What MCP Really Is

### 1.1 MCP is a Protocol, Not a System

**Model Context Protocol (MCP)** is a standardized communication protocol for tool invocation between AI agents and external services. It defines:

- **Message formats** for tool discovery, invocation, and response
- **Schema conventions** for parameter and return type descriptions
- **Transport mechanisms** (stdio, HTTP, WebSocket)
- **Error handling** semantics

**MCP is NOT:**
- An orchestration engine
- A reasoning system
- An execution runtime
- A model hosting solution

Think of MCP like HTTP for the web: HTTP doesn't decide what content to serve or how to render it—it only defines how clients and servers communicate. Similarly, MCP doesn't decide which tools to call or how to interpret results—it only standardizes the communication interface.

### 1.2 The Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Agent (Orchestrator)                  │
│         Reasoning, Planning, Tool Selection, Memory         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ MCP Protocol
                              │ (JSON-RPC over stdio/HTTP/WS)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                             │
│              Tool Registry, Schema Exposure                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Native Calls
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Services                         │
│     LLMs, Scripts, APIs, Databases, Kernels, Programs       │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Separation of Concerns

| Layer | Responsibility | Examples |
|-------|----------------|----------|
| **Orchestration** | Decides *what* to do and *when* | Claude Code, LangChain, custom agents |
| **Protocol (MCP)** | Standardizes *how* to communicate | Tool schemas, invocation format |
| **Execution** | Performs the actual *work* | Kernels, scripts, LLM calls, APIs |

**Key insight:** MCP servers can host any combination of backends—the protocol doesn't care whether a tool calls an LLM, runs a shell command, queries a database, or performs pure computation.

---

## 2. Common Misconceptions

### Misconception 1: "MCP is an AI system"

**Reality:** MCP is a communication protocol. The AI reasoning happens in the orchestrating agent (e.g., Claude Code), not in MCP itself. MCP servers are passive—they expose tools and respond to invocations.

### Misconception 2: "MCP requires cloud services"

**Reality:** MCP is transport-agnostic. RAGIX's MCP server runs 100% locally:
- Local Ollama for LLM inference
- Local file system for workspaces
- Local network for security scans
- No external API calls required

### Misconception 3: "MCP tools must be LLM-powered"

**Reality:** MCP tools can be:
- **Stochastic (LLM):** Natural language processing, reasoning
- **Deterministic (Scripts):** AST parsing, metrics computation, network scanning
- **Hybrid:** LLM-guided tool selection with deterministic execution

RAGIX intentionally uses **deterministic kernels** for most operations—ensuring reproducibility, auditability, and predictability.

### Misconception 4: "MCP manages tool orchestration"

**Reality:** Orchestration is the agent's responsibility. MCP only provides:
- Tool discovery (`tools/list`)
- Tool invocation (`tools/call`)
- Response formatting

The agent decides which tools to call, in what order, with what parameters, and how to interpret results.

### Misconception 5: "One MCP server per application"

**Reality:** Multiple MCP servers can collaborate:
- Server A: Code analysis tools
- Server B: Security scanning tools
- Server C: Documentation generation
- Server D: External API integrations

The orchestrating agent can connect to multiple servers simultaneously, enabling **distributed collective intelligence**.

---

## 3. Architecture Principles

### 3.1 Local-First, Sovereign AI

RAGIX's MCP implementation follows sovereign AI principles:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Infrastructure                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Ollama    │  │   RAGIX     │  │    Your Code        │  │
│  │  (Local LLM)│  │ MCP Server  │  │   (Sandboxed)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │             │
│         └────────────────┴────────────────────┘             │
│                          │                                  │
│              No External Dependencies                       │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**

- **Data sovereignty:** Code never leaves your machine
- **Air-gapped operation:** Works without internet
- **Auditability:** Complete command logs with SHA256 verification
- **Reproducibility:** Deterministic kernels produce identical outputs

### 3.2 Kernel-Centric Design

RAGIX separates concerns between:

| Component | Nature | Responsibility |
|-----------|--------|----------------|
| **LLM (Ollama)** | Stochastic | Reasoning, planning, natural language |
| **Kernels** | Deterministic | Computation, analysis, scanning |
| **MCP Server** | Protocol | Tool exposure, invocation routing |
| **Agent** | Hybrid | Orchestration, memory, tool selection |

**Why this matters:** LLMs hallucinate. Kernels don't. By isolating computation in deterministic kernels, we get:
- Reproducible audits
- Verifiable results
- Predictable performance
- Testable components

### 3.3 Three-Stage Pipeline (KOAS)

The Kernel-Orchestrated Audit System exemplifies this architecture:

```
Stage 1: Data Collection       Stage 2: Analysis          Stage 3: Reporting
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ ast_scan            │     │ stats_summary       │     │ section_executive   │
│ metrics             │ ──▶ │ hotspots            │ ──▶ │ section_overview    │
│ dependency          │     │ coupling            │     │ section_recs        │
│ partition           │     │ dead_code           │     │ report_assemble     │
│ services            │     │ entropy             │     └─────────────────────┘
│ timeline            │     │ risk                │
└─────────────────────┘     └─────────────────────┘

    Pure extraction           Pure computation           Pure generation
    (no LLM inside)           (no LLM inside)            (optional LLM)
```



---

## 4. RAGIX MCP Implementation

### 4.1 Server Overview

**File:** `MCP/ragix_mcp_server.py`
**Framework:** FastMCP (Anthropic MCP SDK)
**Lines:** ~3,600

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="RAGIX",
    instructions="RAGIX is a sovereign Unix-RAG development assistant...",
    website_url="https://github.com/ovitrac/RAGIX",
)
```

### 4.2 Tool Categories (38 Tools)

| Category | Count | Purpose |
|----------|-------|---------|
| **Core RAGIX** | 11 | Unix-RAG reasoning, search, workflows |
| **System** | 5 | Model management, AST, system info |
| **KOAS Base** | 4 | Workspace orchestration |
| **KOAS Security** | 8 | Network discovery, scanning, compliance |
| **KOAS Audit** | 8 | Code analysis, metrics, risk assessment |
| **Utilities** | 2 | Logging, health checks |

### 4.3 Configuration

**Environment Variables:**
```bash
# Core
UNIX_RAG_SANDBOX=/path/to/project      # Sandbox root
UNIX_RAG_MODEL=mistral                  # Ollama model
UNIX_RAG_PROFILE=dev                    # Safety profile (strict/dev/unsafe)

# MCP
RAGIX_LOG_LEVEL=INFO                    # Logging verbosity
OLLAMA_NUM_GPU=1                        # GPU allocation
```

**Profiles:**
| Profile | Description | Use Case |
|---------|-------------|----------|
| `strict` | Read-only, maximum safety | Production audits |
| `dev` | Normal editing, protected destructive ops | Development |
| `unsafe` | Minimal restrictions | Expert/testing |

---

## 5. Complete Tool Reference

### 5.1 Core RAGIX Tools

#### `ragix_chat`
Execute single Unix-RAG reasoning step.

```python
# Parameters
prompt: str          # Natural language instruction

# Returns
{
    "response": "Analysis complete...",
    "last_command": {
        "command": "grep -rn 'TODO' src/",
        "stdout": "...",
        "returncode": 0
    }
}
```

#### `ragix_scan_repo`
Quick project structure overview.

```python
# Parameters
max_depth: int = 4           # Directory recursion depth
include_hidden: bool = False # Include dot files

# Returns
[
    {"path": "src/main.py", "size": 1245, "ext": ".py"},
    {"path": "docs/README.md", "size": 5432, "ext": ".md"}
]
```

#### `ragix_search`
Hybrid BM25 + vector search across codebase.

```python
# Parameters
query: str              # Search query
k: int = 10             # Number of results
strategy: str = "rrf"   # Fusion strategy

# Returns
{
    "query": "database connection",
    "results": [
        {"file": "db.py", "name": "connect", "score": 0.85}
    ]
}
```

#### `ragix_read_file`
Read file with size limit.

```python
# Parameters
path: str                  # Relative path within sandbox
max_bytes: int = 65536     # Maximum bytes to read
```

#### `ragix_workflow`
Execute pre-defined multi-step workflow.

```python
# Parameters
template: str              # Workflow name
params: Dict[str, str]     # Template parameters
```

#### `ragix_templates`
List available workflow templates.

#### `ragix_config`
Get current RAGIX configuration.

#### `ragix_health`
Comprehensive system health status.

#### `ragix_logs`
Retrieve recent command log entries.

#### `ragix_verify_logs`
Verify command log integrity using SHA256 chain.

#### `ragix_agent_step`
Enhanced Unix-RAG step with optional history persistence.

### 5.2 System Tools

#### `ragix_ast_scan`
Extract AST symbols from source code.

```python
# Parameters
path: str                  # File or directory
language: str = "auto"     # python, java, typescript, go, cpp
```

#### `ragix_ast_metrics`
Compute code metrics (complexity, LOC, maintainability).

#### `ragix_models_list`
List available Ollama models.

#### `ragix_model_info`
Get detailed model information.

#### `ragix_system_info`
Comprehensive system information (CPU, GPU, memory, disk).

### 5.3 KOAS Security Tools

#### `koas_security_discover`
Network host discovery.

```python
# Parameters
target: str              # IP, CIDR, or hostname
method: str = "ping"     # ping, arp, list
timeout: int = 120       # Scan timeout
workspace: str = ""      # Auto-created if empty

# Returns
{
    "summary": "Found 12 hosts on 192.168.1.0/24",
    "hosts": [{"ip": "192.168.1.1", "hostname": "router", "status": "up"}],
    "workspace": "/tmp/koas_security_..."
}
```

#### `koas_security_scan_ports`
Port scanning with preset groups.

```python
# Parameters
target: str = "discovered"    # Target or "discovered" keyword
ports: str = "common"         # common, web, database, admin, top100, full
detect_services: bool = True
```

**Port Presets:**
| Preset | Ports | Use Case |
|--------|-------|----------|
| `common` | 21,22,23,25,53,80,110,143,443,445,3306,3389,5432,8080 | Quick scan |
| `web` | 80,443,8000,8080,8443,8888 | Web servers |
| `database` | 1433,1521,3306,5432,6379,27017 | Database servers |
| `admin` | 22,3389,5900,5985,5986 | Remote admin |
| `top100` | Top 100 most common | Comprehensive |
| `full` | 1-65535 | Complete (slow) |

#### `koas_security_ssl_check`
SSL/TLS certificate and cipher analysis.

#### `koas_security_vuln_scan`
Vulnerability assessment with nuclei templates.

```python
# Parameters
target: str
severity: str = "all"       # critical, high, medium, low, all
templates: str = ""         # Custom template selection
```

#### `koas_security_dns_check`
DNS enumeration and security record analysis (DNSSEC, SPF, DMARC, DKIM).

#### `koas_security_compliance`
Compliance checking against security frameworks.

```python
# Parameters
workspace: str
framework: str = "anssi"    # anssi, nist, cis
level: str = "standard"     # essential, standard, reinforced
```

#### `koas_security_risk`
Network security risk scoring.

#### `koas_security_report`
Security assessment report generation.

### 5.4 KOAS Audit Tools

#### `koas_audit_scan`
AST scanning and symbol extraction.

```python
# Parameters
project_path: str
language: str = "auto"
include_tests: bool = False
workspace: str = ""
```

#### `koas_audit_metrics`
Code metrics analysis (cyclomatic complexity, LOC, maintainability).

#### `koas_audit_hotspots`
Complexity and risk hotspot identification.

#### `koas_audit_dependencies`
Dependency graph analysis and cycle detection.

#### `koas_audit_dead_code`
Dead/unused code detection.

#### `koas_audit_risk`
Code risk scoring (Service Life perspective).

#### `koas_audit_compliance`
Code quality compliance checking.

#### `koas_audit_report`
Comprehensive audit report generation.

### 5.5 KOAS Base Tools

#### `koas_init`
Initialize audit workspace.

#### `koas_run`
Execute kernel stages with parallelization.

#### `koas_status`
Get workspace status.

#### `koas_summary`
Get stage summaries.

#### `koas_list_kernels`
List available kernels.

#### `koas_report`
Get generated report.

---

## 6. Deployment Topologies

### 6.1 Local Standalone

```
┌─────────────────────────────────────────┐
│              Local Machine              │
│  ┌─────────┐    ┌──────────────────┐    │
│  │ Claude  │◀──▶│ RAGIX MCP Server │    │
│  │  Code   │    └────────┬─────────┘    │
│  └─────────┘             │              │
│                   ┌──────▼──────┐       │
│                   │   Ollama    │       │
│                   │ (Local LLM) │       │
│                   └─────────────┘       │
└─────────────────────────────────────────┘
```

**Use case:** Individual developer, full sovereignty.

### 6.2 Team Server

```
┌─────────────────────────────────────────┐
│            Development Server           │
│  ┌──────────────────────────────────┐   │
│  │       RAGIX MCP Server           │   │
│  │  (Shared workspace, GPU access)  │   │
│  └──────────────┬───────────────────┘   │
│                 │                       │
│          ┌──────▼──────┐                │
│          │   Ollama    │                │
│          │  (70B+ LLM) │                │
│          └─────────────┘                │
└─────────────────────────────────────────┘
         ▲           ▲           ▲
         │           │           │
    ┌────┴───┐  ┌────┴───┐  ┌────┴───┐
    │ Dev 1  │  │ Dev 2  │  │ Dev 3  │
    │(Claude)│  │(Claude)│  │(Custom)│
    └────────┘  └────────┘  └────────┘
```

**Use case:** Team sharing expensive GPU resources.

### 6.3 Hybrid Local/Remote

```
┌─────────────────┐     ┌─────────────────┐
│  Local Machine  │     │  Remote Server  │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ MCP Local │  │     │  │MCP Remote │  │
│  │(Security) │  │     │  │ (Compute) │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │
│  ┌─────▼─────┐  │     │  ┌─────▼─────┐  │
│  │  Ollama   │  │     │  │  Ollama   │  │
│  │  (7B)     │  │     │  │  (70B+)   │  │
│  └───────────┘  │     │  └───────────┘  │
└─────────────────┘     └─────────────────┘
         ▲                       ▲
         │                       │
         └───────────┬───────────┘
                     │
             ┌───────▼───────┐
             │ Orchestrator  │
             │ (Claude Code) │
             └───────────────┘
```

**Use case:** Sensitive operations local, heavy compute remote.

### 6.4 Multi-Server Collective

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ MCP Server A│  │ MCP Server B│  │ MCP Server C│
│ Code Audit  │  │  Security   │  │    Docs     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                ┌───────▼───────┐
                │  Orchestrator │
                │    (Agent)    │
                └───────────────┘
```

**Use case:** Specialized services collaborating.

---

## 7. Hybrid Backends

### 7.1 Backend Types

Behind every MCP tool is an **execution backend**. These can be:

| Type | Characteristics | RAGIX Examples |
|------|-----------------|----------------|
| **Stochastic (LLM)** | Non-deterministic, creative, may hallucinate | `ragix_chat`, natural language responses |
| **Deterministic (Scripts)** | Reproducible, predictable, verifiable | Kernels, AST parsing, metrics |
| **Hybrid** | LLM for planning, scripts for execution | Unix-RAG pattern |

### 7.2 Why Deterministic Kernels Matter

```
User: "Analyze my codebase for security issues"

┌─ Stochastic Approach (Pure LLM) ─────────────────────────┐
│                                                          │
│  LLM reads code → "I think there might be SQL injection" │
│                                                          │
│  Problems:                                               │
│  - May hallucinate vulnerabilities                       │
│  - Different runs give different results                 │
│  - Cannot verify findings                                │
│  - No audit trail                                        │
└──────────────────────────────────────────────────────────┘

┌─ Deterministic Approach (RAGIX Kernels) ─────────────────┐
│                                                          │
│  Kernel executes → "Found SQL injection at db.py:42"     │
│                                                          │
│  Benefits:                                               │
│  - Verifiable: Same code → Same findings                 │
│  - Traceable: Exact file, line, pattern                  │
│  - Auditable: SHA256-verified execution log              │
│  - Testable: Kernel output is unit-testable              │
└──────────────────────────────────────────────────────────┘
```

### 7.3 The Unix-RAG Hybrid Pattern

RAGIX uses LLMs for **reasoning and planning**, but delegates execution to **deterministic tools**:

```
┌─────────────────────────────────────────────────────────┐
│                     Unix-RAG Agent                      │
├─────────────────────────────────────────────────────────┤
│  1. LLM receives user query                             │
│  2. LLM reasons about approach (stochastic)             │
│  3. LLM selects tools to call (stochastic)              │
│  4. Tools execute commands (DETERMINISTIC)              │
│  5. LLM interprets results (stochastic)                 │
│  6. LLM formulates response (stochastic)                │
├─────────────────────────────────────────────────────────┤
│  Key: Steps 1-3, 5-6 may vary                           │
│       Step 4 is ALWAYS reproducible                     │
└─────────────────────────────────────────────────────────┘
```

### 7.4 Choosing Backend Types

| Task | Recommended Backend | Rationale |
|------|---------------------|-----------|
| Code metrics | Deterministic | Must be reproducible |
| Vulnerability scanning | Deterministic | Legal/compliance requirements |
| Natural language summary | Stochastic | Creativity needed |
| Risk interpretation | Hybrid | Numbers from kernel, prose from LLM |
| Report generation | Hybrid | Structure deterministic, narrative LLM |

---

## 8. Collective Intelligence Patterns

### 8.1 Beyond Text: Multi-Modal Information Exchange

MCP tools exchange more than text. RAGIX tools communicate:

| Data Type | Examples | Use Case |
|-----------|----------|----------|
| **Numbers** | Risk scores, complexity metrics, compliance % | Quantitative analysis |
| **Graphs** | Dependency trees, call graphs, AST | Structural analysis |
| **Vectors** | Embeddings, feature representations | Semantic search |
| **Schemas** | JSON schemas, type definitions | Interoperability |
| **Binary** | Images, compiled artifacts | Visual analysis |

### 8.2 Tool Chaining Patterns

**Sequential Pipeline:**
```
discover → scan_ports → ssl_check → compliance → report
    │           │            │           │          │
    └─ hosts ───┴── ports ───┴── certs ──┴─ score ──┴─▶ Final Report
```

**Parallel Execution:**

```
         ┌─▶ ast_scan ──────┐
         │                  │
discover ├─▶ metrics ───────┼─▶ risk_analysis
         │                  │
         └─▶ dependency ────┘
```

**Conditional Branching:**
```
scan_ports ──┬─▶ [port 443 open?] ─▶ ssl_check
             │
             └─▶ [port 3306 open?] ─▶ db_security_check
```

### 8.3 Multi-Agent Collaboration

Multiple RAGIX instances can collaborate:

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator Agent                    │
│         (Planning, Coordination, Synthesis)             │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ RAGIX Node 1│  │ RAGIX Node 2│  │ RAGIX Node 3│
│ Code Audit  │  │  Security   │  │   Docs      │
│             │  │  Scanning   │  │ Generation  │
└─────────────┘  └─────────────┘  └─────────────┘
       │               │               │
       └───────────────┴───────────────┘
                       │
                       ▼
              Unified Assessment
```

### 8.4 Synchronous vs Asynchronous Execution

| Pattern | When to Use | RAGIX Support |
|---------|-------------|---------------|
| **Synchronous** | Sequential dependencies | Default execution |
| **Asynchronous** | Independent parallel tasks | `koas_run(parallel=True)` |
| **Streaming** | Long-running operations | WebSocket trace |
| **Batch** | Multiple similar operations | Workflow templates |

---

## 9. Frequently Asked Questions

### Q1: Do I need an MCP server if tools are installed locally?

**Answer:** It depends on your orchestrator.

- **With Claude Code/Desktop:** Yes, MCP provides the standard interface Claude expects
- **With custom scripts:** No, you can call tools directly
- **For interoperability:** Yes, MCP enables tool sharing across different agents

**Key insight:** MCP adds value through standardization, not through functionality. Your tools work the same way—MCP just makes them accessible to any MCP-compatible agent.

### Q2: Can I use MCP with commercial LLMs (OpenAI, Anthropic API)?

**Answer:** Absolutely. MCP is model-agnostic:

```
┌─────────────────┐     ┌─────────────────┐
│  Claude API     │     │  RAGIX MCP      │
│  (Commercial)   │◀───▶│  (Local Tools)  │
└─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  Ollama         │     │  RAGIX MCP      │
│  (Sovereign)    │◀───▶│  (Same Tools)   │
└─────────────────┘     └─────────────────┘
```

The **same MCP server** works with both. The orchestrating agent chooses which LLM to use for reasoning.

### Q3: How do I ensure reproducibility with stochastic backends?

**Answer:** RAGIX uses multiple strategies:

1. **Deterministic kernels:** Core computation never uses LLM
2. **Seed control:** Ollama supports temperature=0 for deterministic generation
3. **Logging:** SHA256-verified command logs for audit trails
4. **Workspace isolation:** Each run gets a unique workspace

### Q4: Can multiple MCP servers collaborate?

**Answer:** Yes, through the orchestrating agent:

```python
# Conceptual: Agent connects to multiple servers
code_server = MCPClient("http://localhost:8080")      # Code analysis
security_server = MCPClient("http://server2:8080")   # Security scanning
docs_server = MCPClient("http://server3:8080")       # Documentation

# Agent orchestrates across servers
code_issues = await code_server.call("koas_audit_scan", ...)
vulns = await security_server.call("koas_security_vuln_scan", ...)
report = await docs_server.call("generate_report", code_issues, vulns)
```

### Q5: What's the difference between RAGIX tools and KOAS tools?

**Answer:**

| Aspect | RAGIX Tools | KOAS Tools |
|--------|-------------|------------|
| **Scope** | General-purpose Unix-RAG | Structured audit/security |
| **Execution** | Direct shell commands | Kernel orchestration |
| **State** | Stateless | Workspace-based |
| **Output** | Raw command results | Simplified summaries |

KOAS tools are **opinionated wrappers** that make complex multi-step operations accessible to local LLMs through simplified interfaces.

### Q6: How do I add custom tools to the MCP server?

**Answer:** Use the FastMCP decorator pattern:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my_server")

@mcp.tool()
def my_custom_tool(param1: str, param2: int = 10) -> dict:
    """
    Tool description for LLM.

    Args:
        param1: Description of param1
        param2: Description with default

    Returns:
        Dictionary with results
    """
    # Your implementation
    return {"result": "..."}
```

### Q7: How does RAGIX handle tool errors?

**Answer:** Standardized error responses:

```python
{
    "error": "Tool execution failed",
    "error_type": "KernelExecutionError",
    "details": "nmap not found in PATH",
    "suggestions": ["Install nmap: sudo apt install nmap"],
    "workspace": "/tmp/koas_security_..."  # For debugging
}
```

Kernels never raise exceptions to the MCP layer—all errors are caught and formatted.

### Q8: Can I use RAGIX without Ollama?

**Answer:** Partially. Core functionality requires an LLM, but:

- **KOAS kernels:** Work without LLM (pure computation)
- **AST tools:** Work without LLM (static analysis)
- **Search:** Vector search needs embeddings (requires model)

For full functionality, Ollama (or compatible API) is recommended.

---

## 10. Related Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [REASONING.md](REASONING.md) | Reasoning engines: ContractiveReasoner, Reasoning v30 |
| [KOAS.md](KOAS.md) | Kernel-Orchestrated Audit System details |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Overall RAGIX architecture |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation |

### Specialized Guides

| Document | Description |
|----------|-------------|
| [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) | Detailed KOAS MCP tool reference |
| [CLI_GUIDE.md](CLI_GUIDE.md) | Command-line interface guide |
| [AST_GUIDE.md](AST_GUIDE.md) | AST analysis tools |
| [PLAYBOOK_GUIDE.md](PLAYBOOK_GUIDE.md) | Workflow playbooks |

### Component Documentation

| Document | Description |
|----------|-------------|
| [RAGIX_TOOLS_INVENTORY.md](RAGIX_TOOLS_INVENTORY.md) | Tool component reference |
| [WASP_GUIDE.md](WASP_GUIDE.md) | WASP integration guide |

### External Resources

| Resource | URL |
|----------|-----|
| MCP Specification | https://modelcontextprotocol.io/ |
| Ollama | https://ollama.ai/ |
| RAGIX Repository | https://github.com/ovitrac/RAGIX |

---

## Appendix A: Tool Quick Reference

```
RAGIX Core (11 tools)
├── ragix_chat              # Unix-RAG reasoning step
├── ragix_scan_repo         # Project structure scan
├── ragix_read_file         # File reading
├── ragix_search            # Hybrid search
├── ragix_workflow          # Workflow execution
├── ragix_templates         # List workflows
├── ragix_config            # Configuration
├── ragix_health            # Health check
├── ragix_logs              # Log retrieval
├── ragix_verify_logs       # Log verification
└── ragix_agent_step        # Enhanced reasoning step

System (5 tools)
├── ragix_ast_scan          # AST extraction
├── ragix_ast_metrics       # Code metrics
├── ragix_models_list       # Ollama models
├── ragix_model_info        # Model details
└── ragix_system_info       # System information

KOAS Base (6 tools)
├── koas_init               # Initialize workspace
├── koas_run                # Execute stages
├── koas_status             # Workspace status
├── koas_summary            # Stage summaries
├── koas_list_kernels       # Available kernels
└── koas_report             # Get report

KOAS Security (8 tools)
├── koas_security_discover  # Network discovery
├── koas_security_scan_ports# Port scanning
├── koas_security_ssl_check # SSL/TLS analysis
├── koas_security_vuln_scan # Vulnerability assessment
├── koas_security_dns_check # DNS enumeration
├── koas_security_compliance# Framework compliance
├── koas_security_risk      # Risk scoring
└── koas_security_report    # Security report

KOAS Audit (8 tools)
├── koas_audit_scan         # AST scanning
├── koas_audit_metrics      # Code metrics
├── koas_audit_hotspots     # Hotspot identification
├── koas_audit_dependencies # Dependency analysis
├── koas_audit_dead_code    # Dead code detection
├── koas_audit_risk         # Risk scoring
├── koas_audit_compliance   # Quality compliance
└── koas_audit_report       # Audit report
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-20
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
