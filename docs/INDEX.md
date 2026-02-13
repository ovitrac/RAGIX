# RAGIX Documentation Index

**Retrieval-Augmented Generative Interactive eXecution Agent**

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.66.0
**Updated:** 2026-02-13

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
    │ KOAS ─────────┼─┐       │               │         │               │
    └───────┬───────┘ │       └───────┬───────┘         └───────┬───────┘
            │         │               │                         │
            ▼         │               ▼                         ▼
    ┌───────────────┐ │       ┌───────────────┐         ┌───────────────┐
    │   Guides      │ │       │  References   │         │   Advanced    │
    │               │ │       │               │         │               │
    │ CLI_GUIDE     │ │       │ API_REFERENCE │         │ SOVEREIGN_LLM │
    │ AST_GUIDE     │ │       │ TOOLS_INV     │         │ WASP_GUIDE    │
    │               │ │       │ KOAS_MCP_REF  │         │ PLAYBOOK      │
    └───────────────┘ │       └───────────────┘         └───────────────┘
                      │
    ┌─────────────────▼──────────────────────────────────┐
    │              KOAS Kernel Families                   │
    │                                                    │
    │  KOAS_DOCS      (17 kernels — document analysis)   │
    │  KOAS_PRESENTER (8 kernels — slide generation)     │
    │  KOAS_REVIEW    (13 kernels — Markdown review)     │
    │  KOAS_ACTIVITY  (centralized activity logging)     │
    │  + audit (27) and security (10) in KOAS.md         │
    └────────────────────────────────────────────────────┘
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
| [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) | Confidential AI operations | Auditors, Compliance |
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

**Learn about deterministic computation kernels — 75 kernels across 5 families.**

- **[KOAS.md](KOAS.md)** — KOAS philosophy and architecture
  - Three-stage pipeline, 5 kernel families
  - Kernel development patterns
  - Deterministic guarantee: kernels compute, LLMs reason

- **[KOAS_DOCS.md](KOAS_DOCS.md)** — Document summarization (17 kernels)
  - Hierarchical analysis, dual clustering (Pyramidal + Leiden)
  - Worker + Tutor LLM pattern

- **[KOAS_PRESENTER.md](KOAS_PRESENTER.md)** — Slide deck generation (8 kernels)
  - MARP output, 3 compression modes (full/compressed/executive)
  - Deterministic default, optional LLM normalization

- **[KOAS_REVIEW.md](KOAS_REVIEW.md)** — Traceable Markdown review (13 kernels)
  - Chunk-level edits, selective revert, preflight pipeline
  - Append-only ledger with RVW-NNNN change IDs

- **[KOAS_ACTIVITY.md](KOAS_ACTIVITY.md)** — Centralized activity logging
  - Event schema (koas.event/1.0), actor model, hash chain
  - Sovereignty attestation per event

- **Source Documentation:**
  - `ragix_kernels/README.md` — Kernel developer reference (v1.4.0, all 75 kernels)

### Sovereignty and Compliance

**Understanding data sovereignty and confidential operations.**

- **[SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md)** — Confidential AI guide
  - Gray environment deployment
  - Policy enforcement mechanisms
  - Audit trail and attestation
  - Air-gapped operation
  - Sovereignty verification

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
│    ┌──────────┬──────────┬──────────┬──────────┐            │
│    │  audit   │   docs   │presenter │ reviewer │ security   │
│    │  (27)    │   (17)   │  (8)     │  (13)    │  (10)      │
│    └──────────┴──────────┴──────────┴──────────┘            │
└─────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│      Tool Execution         │         │      Activity Logging       │
│  Security │ Audits │ Docs   │         │  Events │ Sovereignty │ ACL │
└─────────────────────────────┘         └─────────────────────────────┘
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

**"I want to summarize large documents"**
1. [KOAS_DOCS.md](KOAS_DOCS.md) → 17-kernel pipeline
2. [KOAS.md](KOAS.md) → Three-stage architecture
3. [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) → MCP tool interface

**"I want to generate slide decks from documents"**
1. [KOAS_PRESENTER.md](KOAS_PRESENTER.md) → 8-kernel pipeline, presenterctl CLI
2. [KOAS_DOCS.md](KOAS_DOCS.md) → Upstream document analysis (required input)
3. [KOAS.md](KOAS.md) → Kernel orchestration

**"I want to review and edit Markdown documents"**
1. [KOAS_REVIEW.md](KOAS_REVIEW.md) → 13-kernel pipeline, selective revert
2. [KOAS.md](KOAS.md) → KOAS philosophy
3. [KOAS_MCP_REFERENCE.md](KOAS_MCP_REFERENCE.md) → MCP tool interface

**"I want to understand activity logging and compliance"**
1. [KOAS_ACTIVITY.md](KOAS_ACTIVITY.md) → Event schema, actor model, hash chain
2. [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) → Sovereignty architecture
3. [KOAS.md](KOAS.md) → Audit trail mechanisms

**"I need to ensure data sovereignty/compliance"**
1. [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) → Complete guide
2. [KOAS_ACTIVITY.md](KOAS_ACTIVITY.md) → Sovereignty attestation per event
3. [ARCHITECTURE.md](ARCHITECTURE.md) → System topology
4. [KOAS.md](KOAS.md) → Audit trail mechanisms

---

## Key Concepts Glossary

| Term | Definition | Documentation |
|------|------------|---------------|
| **MCP** | Model Context Protocol — standardized tool communication | [MCP.md](MCP.md) |
| **KOAS** | Kernel-Orchestrated Audit System — deterministic computation | [KOAS.md](KOAS.md) |
| **Kernel** | Deterministic computation unit (no LLM inside) | [KOAS.md](KOAS.md) |
| **Kernel Family** | Group of kernels sharing a scope (audit, docs, presenter, reviewer, security) | [KOAS.md](KOAS.md) |
| **Three-Stage Pipeline** | Data collection → Analysis → Reporting architecture | [KOAS.md](KOAS.md) |
| **Worker + Tutor** | Dual-LLM pattern: small model generates, larger model refines | [KOAS_DOCS.md](KOAS_DOCS.md) |
| **Broker Gateway** | Core-Shell access layer mediating external orchestrator requests | [KOAS_ACTIVITY.md](KOAS_ACTIVITY.md) |
| **Activity Logging** | Centralized append-only event stream (`koas.event/1.0`) | [KOAS_ACTIVITY.md](KOAS_ACTIVITY.md) |
| **Hash Chain** | SHA256 chain across kernel executions for tamper evidence | [KOAS_ACTIVITY.md](KOAS_ACTIVITY.md) |
| **Merkle Root** | Cryptographic root hash for pyramidal provenance | [KOAS_DOCS.md](KOAS_DOCS.md) |
| **ContractiveReasoner** | Tree-based reasoning with entropy decomposition | [REASONING.md](REASONING.md) |
| **Reasoning v30** | Graph-based state machine reasoning | [REASONING.md](REASONING.md) |
| **Unix-RAG** | Unix tools for retrieval, LLM for generation | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Workspace** | Isolated `.KOAS/` directory for pipeline results and audit trail | [KOAS.md](KOAS.md) |
| **Entropy** | Uncertainty measure for decomposition decisions | [REASONING.md](REASONING.md) |
| **Peer Review** | External LLM validation of reasoning | [REASONING.md](REASONING.md) |
| **Sovereignty** | Data processing within controlled perimeter | [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) |
| **Gray Environment** | Networked but untrusted context | [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) |
| **Attestation** | Cryptographic proof of sovereignty (`sovereignty.local_only: true`) | [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) |
| **Output Sanitizer** | 4-level isolation (Internal/External/Orchestrator/Compliance) | [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) |

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
| 0.66.0 | 2026-02-13 | Added KOAS_PRESENTER, KOAS_REVIEW, KOAS_ACTIVITY; 5-family kernel map; new glossary entries and reading paths |
| 0.64.2 | 2026-01-29 | Added SOVEREIGN_LLM_OPERATIONS.md |
| 0.62.0 | 2025-12-20 | Added MCP.md, REASONING.md, INDEX.md |
| 0.61.0 | 2025-12-19 | KOAS security kernels |
| 0.60.0 | 2025-12-18 | Initial documentation structure |

---

**Document Version:** 2.0.0
**Last Updated:** 2026-02-13
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
