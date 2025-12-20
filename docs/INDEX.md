# RAGIX Documentation Index

**Retrieval-Augmented Generative Interactive eXecution Agent**

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.62.0
**Updated:** 2025-12-20

---

## Overview

RAGIX is a **sovereign AI development assistant** that combines local LLM reasoning with deterministic computation kernels. This documentation covers all aspects of the system, from high-level architecture to specific tool references.

---

## Documentation Map

```
                           ┌─────────────────────┐
                           │    INDEX.md         │
                           │   (You are here)    │
                           └──────────┬──────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
    │ Architecture  │         │   Protocols   │         │   Reasoning   │
    │               │         │               │         │               │
    │ ARCHITECTURE  │         │     MCP       │         │  REASONING    │
    │ KOAS          │         │               │         │               │
    └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
            │                         │                         │
            ▼                         ▼                         ▼
    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
    │   Guides      │         │  References   │         │   Advanced    │
    │               │         │               │         │               │
    │ CLI_GUIDE     │         │ API_REFERENCE │         │ WASP_GUIDE    │
    │ AST_GUIDE     │         │ TOOLS_INV     │         │ PLAYBOOK      │
    │               │         │ KOAS_MCP_REF  │         │               │
    └───────────────┘         └───────────────┘         └───────────────┘
```

---

## Quick Navigation

### Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Overall system architecture | Everyone |
| [MCP.md](MCP.md) | Model Context Protocol explained | Developers, Architects |
| [CLI_GUIDE.md](CLI_GUIDE.md) | Command-line usage | Users |

### Core Concepts

| Document | Description | Audience |
|----------|-------------|----------|
| [REASONING.md](REASONING.md) | Reasoning engines deep dive | Researchers, Developers |
| [KOAS.md](KOAS.md) | Kernel-Orchestrated Audit System | Auditors, Developers |
| [AST_GUIDE.md](AST_GUIDE.md) | Abstract Syntax Tree tools | Developers |

### Reference

| Document | Description | Audience |
|----------|-------------|----------|
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation | Developers |
| [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) | KOAS MCP tool reference | Developers |
| [RAGIX_TOOLS_INVENTORY.md](RAGIX_TOOLS_INVENTORY.md) | Tool component reference | Developers |

### Advanced Topics

| Document | Description | Audience |
|----------|-------------|----------|
| [PLAYBOOK_GUIDE.md](PLAYBOOK_GUIDE.md) | Workflow playbooks | Power Users |
| [WASP_GUIDE.md](WASP_GUIDE.md) | WASP integration | Specialists |

---

## Documentation by Topic

### Model Context Protocol (MCP)

**Start here to understand RAGIX's tool exposure mechanism.**

- **[MCP.md](MCP.md)** — Comprehensive MCP guide
  - What MCP really is (protocol, not system)
  - Common misconceptions addressed
  - 38 tools reference
  - Deployment topologies
  - Hybrid backends (stochastic/deterministic)
  - Collective intelligence patterns
  - FAQ

- **[KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md)** — Detailed KOAS tools

### Reasoning Systems

**Understand how RAGIX transforms local LLMs into deep reasoners.**

- **[REASONING.md](REASONING.md)** — Reasoning engines guide
  - ContractiveReasoner (tree-based decomposition)
  - Reasoning v30 (graph-based state machine)
  - Mathematical foundations
  - Integration with MCP tools
  - Multi-agent patterns
  - Configuration reference

- **Source Documentation:**
  - `ragix_core/reasoning_slim/README.md` — ContractiveReasoner (1200+ lines)
  - `ragix_core/reasoning_v30/__init__.py` — Reasoning v30 API

### Kernel System (KOAS)

**Learn about deterministic computation kernels.**

- **[KOAS.md](KOAS.md)** — KOAS architecture
  - Three-stage pipeline
  - Kernel development
  - Security kernels
  - Audit kernels

- **Source Documentation:**
  - `ragix_kernels/README.md` — Kernel development guide

### Tools and APIs

- **[API_REFERENCE.md](API_REFERENCE.md)** — Complete API docs
- **[RAGIX_TOOLS_INVENTORY.md](RAGIX_TOOLS_INVENTORY.md)** — Component reference
- **[AST_GUIDE.md](AST_GUIDE.md)** — AST analysis tools

---

## Document Relationships

### Conceptual Flow

```
User Goal
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      ARCHITECTURE                           │
│            (How RAGIX is structured)                        │
└─────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
┌─────────────────────┐                         ┌─────────────────────┐
│        MCP          │                         │     REASONING       │
│   (Tool Protocol)   │                         │  (Thinking Engines) │
└─────────────────────┘                         └─────────────────────┘
    │                                                         │
    │    ┌────────────────────────────────────────────────────┘
    │    │
    ▼    ▼
┌─────────────────────────────────────────────────────────────┐
│                         KOAS                                │
│          (Deterministic Kernel Execution)                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Execution                           │
│        Security Scans │ Code Audits │ Analysis              │
└─────────────────────────────────────────────────────────────┘
```

### Reading Order by Goal

**"I want to audit a codebase"**
1. [ARCHITECTURE.md](ARCHITECTURE.md) → Overview
2. [KOAS.md](KOAS.md) → Kernel system
3. [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) → Audit tools
4. [CLI_GUIDE.md](CLI_GUIDE.md) → Command usage

**"I want to understand MCP integration"**
1. [MCP.md](MCP.md) → Protocol concepts
2. [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) → Tool reference
3. [API_REFERENCE.md](API_REFERENCE.md) → API details

**"I want to build custom reasoning"**
1. [REASONING.md](REASONING.md) → Engine concepts
2. `ragix_core/reasoning_slim/README.md` → ContractiveReasoner
3. `ragix_core/reasoning_v30/__init__.py` → Reasoning v30
4. [MCP.md](MCP.md) → Tool integration

**"I want to perform security scans"**
1. [KOAS.md](KOAS.md) → Overview
2. [MCP.md](MCP.md) § Security Tools → Tool reference
3. [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) → Detailed parameters

---

## Key Concepts Glossary

| Term | Definition | Documentation |
|------|------------|---------------|
| **MCP** | Model Context Protocol — standardized tool communication | [MCP.md](MCP.md) |
| **KOAS** | Kernel-Orchestrated Audit System — deterministic computation | [KOAS.md](KOAS.md) |
| **Kernel** | Deterministic computation unit (no LLM inside) | [KOAS.md](KOAS.md) |
| **ContractiveReasoner** | Tree-based reasoning with entropy decomposition | [REASONING.md](REASONING.md) |
| **Reasoning v30** | Graph-based state machine reasoning | [REASONING.md](REASONING.md) |
| **Unix-RAG** | Unix tools for retrieval, LLM for generation | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Workspace** | Isolated directory for audit/scan results | [KOAS.md](KOAS.md) |
| **Entropy** | Uncertainty measure for decomposition decisions | [REASONING.md](REASONING.md) |
| **Peer Review** | External LLM validation of reasoning | [REASONING.md](REASONING.md) |

---

## External Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| **RAGIX Repository** | https://github.com/ovitrac/RAGIX | Source code |
| **Ollama** | https://ollama.ai/ | Local LLM runtime |
| **MCP Specification** | https://modelcontextprotocol.io/ | Protocol standard |
| **FastMCP** | Anthropic SDK | MCP server framework |

---

## Contributing to Documentation

Documentation follows these principles:

1. **Academic rigor** — Cite sources, include mathematical foundations
2. **Practical examples** — Show real usage, not just theory
3. **Clear structure** — Use consistent headings and formatting
4. **Cross-references** — Link related documents
5. **Version tracking** — Include version and date

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.62.0 | 2025-12-20 | Added MCP.md, REASONING.md, INDEX.md |
| 0.61.0 | 2025-12-19 | KOAS security kernels |
| 0.60.0 | 2025-12-18 | Initial documentation structure |

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-20
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
