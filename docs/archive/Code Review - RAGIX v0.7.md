# Code Review - RAGIX v0.7

> This document is not prescriptive. It is based on an external review.



### **v0.7 (current state)**

From the unix agent code and README, RAGIX v0.7 is now:

- A fully-local, sandboxed **Claude-Code–style** development agent
- JSON action protocol with strong structure
- Explicit **Unix-RAG pattern** enforced in system prompt
- Git-aware safety with destructive-command gating
- Structured `edit_file` tool
- Detailed command/event audit logs
- Automatic project overview on startup
- MCP folder present but not yet fully unified
- Clear philosophy: *local, auditable, safety-first, minimalistic orchestrator*

The v0.7 design is extremely solid. The next step is not to add features blindly, but to **move toward a stable, modular, multi-back-end, multi-agent architecture** while keeping the minimalistic Unix-RAG heart.

------

# **1. Is the project split into clear internal modules (i.e., core–agents–io)**

The single monolithic `unix-rag-agent.py` already includes the separation:

- **LLM backend** (Ollama wrapper) 
- **Shell sandbox** with command logging and denylist
- **JSON protocol**
- **Agent** (logic + conversation history)

**v0.8 objective:**
 Move these into a package structure:

```
ragix/
   core/
       llm_backend.py
       system_prompts.py
       json_protocol.py
       audit_logging.py
   shells/
       sandbox.py
       profiles.py
   agents/
       unix_agent.py
       python_agent.py (future)
       mcp_agent.py (future)
   io/
       file_edit_tools.py
       git_tools.py
   cli/
       ragix
```

Why:

- Enables plug-and-play agents (Unix, Python, MCP, multi-agent)
- Enables testing each layer
- Prepares RAGGAE-compatible orchestrator

------

# **2. Native MCP integration as a first-class backend (not separate)**

You already have:

```
MCP/
   ragix_mcp_server.py
   ragix_tools_spec.json
```

But they are not yet unified with the Unix agent.

**v0.8 objective:**

- Add a **unified adapter layer** that exposes RAGIX tools (bash, grep, edit_file, logs) as MCP “tools”.
- Invert control: the MCP server should be able to **instantiate a RAGIX agent** inside itself.
- Create a single config file:

```
ragix.yaml
   mcp:
      enabled: true
      port: 5173
   llm:
      backend: ollama
      model: mistral
   profiles:
      default: dev
```

This matches your RAGIX philosophy: a *thin layer, highly configurable, no bloat*.

------

# **3. Move from single-agent to \**multi-agent orchestration\** (RAGGAE alignment)**

The README and past roadmaps mention a shared orchestrator with RAGGAE.

**v0.8 objective: introduce lightweight agent chaining**:

### **3.1 Message routing**

Add a simple internal router:

- Unix agent → Python agent
- Unix agent → Git agent
- Unix agent → WASM agent (future)

### **3.2 Tool unification**

Expose each agent’s capabilities under one namespace:

```
ragix.tools:
   - search
   - edit
   - git
   - shell
   - python
   - wasm
```

### **3.3 Mini-RAGGAE**

Version 0.8 can introduce:

- A minimal task graph (DAG) executed locally
- Deterministic execution
- Logging of tool transitions

This remains consistent with your minimalistic Unix-RAG identity.

------

# **4. Add hybrid retrieval: structured RAG for large codebases**

RAGIX is “Unix-RAG”: no embeddings.
 But per v0.4 roadmap: hybrid retrieval was planned.

**v0.8 objective: optional retrieval layer** (disabled by default)

- A small local embeddings index (e.g., HuggingFace MiniLM, or local Mistral embedding model)
- Integrated with `grep`, `find`, and diff outputs
- Exposed as a tool:
  - `ragix search semantic "<query>"`
  - fallback to RAGIX native grep/head/tail workflows

**Design constraint:**
 It must *not* replace Unix-RAG, only complement it.

------

# **5. Web UI (local, WASM-enabled, no cloud)**

Your README v0.4 roadmap already listed a local Web UI.

**v0.8 objective: minimal Web UI**:

- A local static SPA using WebAssembly-enhanced shell (wasm exec)
- Embed the JSON agent protocol directly in the UI
- Local logs displayed (from `.agent_logs`)
- Live diff viewer
- This is *not* a rewrite — only a display layer.

------

# **6. End-to-end reproducibility and packaging**

Introduce a **RAGIX environment installer**:

```
ragix install
ragix doctor
ragix upgrade
```

Include:

- checks for Ollama
- checks for GPU VRAM
- verifying safety policies
- initializing `.agent_logs/`
- generating a base config file
- launching a local MCP server if configured

This transforms RAGIX from "a script" to "a developer tool".

------

# **7. Security & compliance strengthening**

Based on unix_rag_agent.py's denylist :

- Expand denylist with dangerous Python modules (`os.remove`, networking)
- Add optional “air-gapped mode” (force LLM = only local models)
- Add signature-based log hashing:

```
.agent_logs/
   commands.log
   commands.log.sha256
```

- Enable deterministic execution mode (for regulated industries)

This matches your professional environment (food safety, regulated R&D, sovereign AI).

------

# **8. Beyond v0.8: what v0.9 could be**

Just to help you prepare:

- distributed agents (network-capable RAGIX swarm?)
- WASM sandbox for safe plugin execution
- support for multi-user sessions
- full RAGGAE orchestrator integration
- codebase-scale planning (like SWE-Bench but local)

------

# ✔ Summary roadmap table

| Version            | Focus                          | Deliverables                                                 |
| ------------------ | ------------------------------ | ------------------------------------------------------------ |
| **0.7 (current)**  | Stable local Claude-Code clone | Unix-RAG, sandbox, JSON actions, logs                        |
| **0.8 (proposed)** | **Modular RAGIX Framework**    | package structure, MCP integration, multi-agent routing, optional semantic search, Web UI v1 |
| **0.9**            | Distributed / WASM agents      | WASM tools, network agents, advanced RAGGAE orchestrator     |

------

# ✔ One-sentence definition of **RAGIX 0.8**

> **RAGIX 0.8 becomes a modular, multi-agent, locally-orchestrated development assistant framework, combining Unix-RAG, structured JSON tools, compliant audit logging, and optional semantic search — all in a fully local, MCP-compatible architecture.**

If you want, I can immediately:

✅ generate the new **README for v0.8**,
 ✅ generate the new **CHANGELOG**,
 ✅ generate the **folder restructuring diff**,
 ✅ or generate a **new CLAUDE.md** aligned with the modular framework.