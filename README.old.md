# `RAGIX` Unix-RAG Local Agent

**A Claude-Code–style Local Development Assistant**
 **Author:** *Olivier Vitrac, PhD, HDR | Adservio Innovation Lab*
 **Contact:** *[olivier.vitrac@adservio.fr](mailto:olivier.vitrac@adservio.fr)*

------

# 1. Overview

This document presents a fully local, open-source alternative to Claude Code, based on:

- **A local LLM** (e.g. Mistral, run through **Ollama**, suitable for 8 GB VRAM)
- **A controlled bash environment** (sandbox)
- **A Unix-driven “RAG-like” retrieval loop** using:
  - `grep`, `find`, `head`, `tail`, `sed`
  - `awk`, `cut`, `sort`, `uniq`
  - small `python` scripts to pre-process data
- **A JSON action protocol** enabling the model to:
  - explore,
  - reason,
  - run commands,
  - edit files,
  - summarize results,
  - and iterate.

This architecture reproduces Claude Code’s strengths:

- Efficient contextual retrieval from a codebase (Unix-RAG)
- Strong iterative reasoning over shell outputs
- Fine-grained control (git, builds, logs, file edits, etc.)
- Full **local execution** and **privacy**

The agent uses a *sandboxed* directory, ensuring safety and reproducibility.

------

# 2. Features

## ✔ Unix-RAG Retrieval

The model naturally uses Unix tools to “retrieve” context:

- `grep -R -n`
- `head`, `tail`
- `sed -n 'start,endp'`
- `wc -l` (file size estimation)
- `find` for structure
- `python` snippets for parsing logs, JSON, CSV, etc.

Equivalent to RAG, but filesystem-native and zero-infrastructure.

------

## ✔ Safe Shell Execution

- All commands executed only inside a **sandbox** directory.
- A denylist blocks dangerous patterns:
  - `rm -rf /`
  - `dd if=...`
  - `mkfs`
  - `shutdown`, `reboot`
- A **dry-run mode** to test behavior.

------

## ✔ File Editing Tools

The agent can:

- Show diffs (`diff -u`)
- Edit files via:
  - Bash redirection (`cat << 'EOF' > file`)
  - Python patching scripts
- Apply structured edit actions:
  - Replace blocks between line numbers
  - Insert new imports
  - Append configuration
  - Apply JSON modifications

------

## ✔ Git Tooling

The agent understands:

- `git status -sb`
- `git diff`
- `git log --oneline --graph -20`
- `git show <commit>`
- `git grep`

Optional protection against:

- `git reset --hard`
- `git push --force`

------

## ✔ Project Exploration Helpers

- File tree snapshots
- Directory summaries
- Language-aware scans (Python, Bash, SQL, JSON, YAML)
- Auto-detection of:
  - Python packages
  - Makefiles
  - Dockerfiles
  - Config directories

------

## ✔ Conversational Loop

- You speak naturally
- The agent decides:
  - whether to run bash,
  - whether to summarize,
  - how to proceed
- Results are fed back into its next iteration.

------

# 3. Requirements

Laptop/machine:

- **8 GB VRAM GPU**
- **Mistral via Ollama** (excellent choice)

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull a Mistral model:

```bash
ollama pull mistral
```

Or a larger one (if RAM allows):

```bash
ollama pull mistral:instruct
```

------

# 4. Start Ollama Server

In one terminal:

```bash
ollama serve
```

------

# 5. Complete Python Script (Ollama Version)

This version is adapted to **Ollama**.
 The LLM backend was replaced with a simple HTTP call to `localhost:11434`.

Save as:

```
unix_rag_agent.py
```

------

*The full script appears below.*

------

## 5.1 Minimal RAGIX design

The implementation of `RAGIX` is exemplified in this mini RAGIX 

```python
# Mini RAGIX by Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr
import os, json, subprocess

SANDBOX = os.path.expanduser("~/agent-sandbox")
os.makedirs(SANDBOX, exist_ok=True)

SYSTEM_PROMPT = """
You are a local assistant.
Respond ONLY with JSON:
{"action": "bash", "command": "..."} or {"action": "respond", "message": "..."}.
Use grep/head/tail/wc to inspect files; never dump huge files.
"""

def call_llm(system, history):
    # Placeholder: call Ollama/OpenAI/etc. and return raw text response
    ...

def run_bash(cmd: str) -> str:
    cp = subprocess.run(
        cmd, shell=True, cwd=SANDBOX,
        capture_output=True, text=True
    )
    return f"$ {cmd}\n{cp.stdout}{cp.stderr}"

history = []
while True:
    user = input("You: ").strip()
    if not user: break
    history.append({"role": "user", "content": user})
    raw = call_llm(SYSTEM_PROMPT, history)

    action = json.loads(raw)        # model must return pure JSON
    if action["action"] == "bash":
        out = run_bash(action["command"])
        print(out)
        history.append({"role": "user", "content": "Command result:\n" + out})
    elif action["action"] == "respond":
        print("Assistant:", action["message"])
        history.append({"role": "assistant", "content": action["message"]})

```

------

# 6. Running the Agent

1. Make sure `ollama serve` is running
2. Run:

```bash
python3 unix_rag_agent.py
```

1. Interact:

```
You: explore the project
You: search where database is initialized
You: show me 30 lines around the init function
You: run the tests and summarize failures
You: fix the error in config.py
```

------

# 7. Demo Suggestions

Here are recommended scenarios:

### **A. Exploration of a codebase**

```
You: list important files
You: search all references to "gas_flow"
You: show me 20 lines around each match
```

### **B. Analyse logs**

```
You: load the logs directory
You: find ERROR lines
You: summarize error patterns using awk
You: produce a Python summary table
```

### **C. Build & test loop**

```
You: run make
You: show me the last 50 lines of build.log
You: grep for warnings and classify them
```

### **D. Git operations**

```
You: git status -sb
You: show the diff of modified files
You: summarize changes
```

### **E. Safe file editing**

```
You: open config.yaml
You: show the first 80 lines
You: propose an edit to adjust max_pressure
You: apply the edit using a patch
You: show the diff
```

------

# 8. Conclusion

This “Unix-RAG Agent” provides a **fully private, fully local** Claude Code equivalent using **Mistral + Ollama** and a sandboxed shell environment.

It is:

- reproducible
- extensible
- safe
- efficient
- perfect for industrial demonstrations



# 9. Comparison with existing solutions

There *are* similar open-source projects/orchestrators—but none that are *exactly* the same design choices implemented in `RAGIX` (Unix-RAG + Ollama + explicit JSON tools + safety model). They are more like close cousins.

Here are the closest ones, with how they relate.

------

## 9.1 Open Interpreter

**What it is:**
 A local agent that lets an LLM run Python, JavaScript, and shell on your machine through a ChatGPT-like terminal. It’s basically “Code Interpreter, but open-source and local.” ([GitHub](https://github.com/openinterpreter/open-interpreter?utm_source=chatgpt.com))

**Similarities:**

- Local code + shell execution.
- Works as a natural-language interface to your computer.
- Often combined with Ollama for the LLM, very close to `RAGIX`. ([Medium](https://medium.com/@karunsharma1920/building-a-local-ai-agent-with-ollama-and-open-interpreter-an-offline-hybrid-assistant-6c8b0ac470b9?utm_source=chatgpt.com))

**Differences vs. `RAGIX` agent:**

- Less opinionated about **Unix-RAG patterns** (grep/head/tail/awk) in the prompt; more generic “run whatever code.”
- No built-in **profiles** (safe-read-only/dev/unsafe) or explicit **git-destructive denylist** in the core design (you can layer it on, but it’s not part of the canonical flow).
- No first-class **JSON tool protocol with `edit_file`** like defined in `RAGIX`; editing is usually done via general shell/Python snippets, not a dedicated structured tool.

It’s the *closest* conceptually.

------

## 9.2 OpenHands (formerly OpenDevin)

**What it is:**
 A full-blown open-source “AI software engineer” platform: the agent can modify code, run commands, browse the web, call APIs, etc. ([GitHub](https://github.com/OpenHands/OpenHands?utm_source=chatgpt.com))

**Similarities:**

- Agents can:
  - run shell commands,
  - edit files,
  - work over a git repo.
- Designed for “agent as developer” scenarios, like our use case.

**Differences:**

- Much heavier stack (web UI, orchestrator, task management).
- Less “minimalist Unix-RAG shell with explicit JSON actions”, more a full research platform.
- Strong focus on evaluation benchmarks and complex workflows; `RAGIX` is a *deliberately light* orchestration layer.

Think of OpenHands as the “Kubernetes cluster” version of `RAGIX`, but `RAGIX` is the “single well-engineered bash+Python tool.”

------

## 9.3 Continue.dev

**What it is:**
 An open-source AI code agent with IDE integrations (VS Code, JetBrains) and a CLI mode for running agents. It can integrate with multiple models (including local ones via Ollama) and automate code tasks across IDE/terminal/CI. ([continue.dev](https://www.continue.dev/?utm_source=chatgpt.com))

**Similarities:**

- Open-source “AI code agent”.
- Can connect to local models and tools.
- Has concept of *rules, prompts, models, and workflows* very similar in spirit to `RAGIX` orchestrator.

**Differences:**

- IDE-centric; the primary UX is VS Code/JetBrains, not a generic terminal REPL.
- Less opinionated about Unix-RAG and explicit structured `edit_file` tool; typically uses model-driven edits within the IDE.

------

## 9.4 Smol Developer & similar “junior dev” agents

Projects like **smol-developer**, **Mentat**, **Aider**, etc., are CLI/agent tools meant to work over a local git repo and edit code via LLMs. ([GitHub](https://github.com/smol-ai/developer?utm_source=chatgpt.com))

**Similarities:**

- LLM-driven repo editing.
- Reasoning + shell/git integration.
- Some use explicit “plan → edit → diff → commit” cycles.

**Differences:**

- Many of them focus on **scaffolding** new codebases or applying patches driven by natural language, more than the kind of **Unix-tool-native observability** and logging we added.
- Less emphasis (out of the box) on structured high-level file tools and safety profiles; more on productivity.

------

## 9.5 How `RAGIX` agent is *distinct*

So, conceptually `RAGIX` is in the same ecosystem as:

- Open Interpreter (local code/shell agent)
- OpenHands (generalist dev agent)
- Continue.dev (IDE/CLI coding agent)
- Smol-developer/Aider/Mentat (repo-editing agents)

But `RAGIX` has a few specific twists:

1. **Explicit Unix-RAG bias in the system prompt**
   - `RAGIX` *force*s the model to rely on `find/grep/head/tail/wc/awk/python` as its retrieval and summarization machinery, rather than generic “just cat the whole file” behavior.
2. **Structured `edit_file` tool with sandbox & git-diff expectation**
   - High-level `{"action": "edit_file", ...}` with a simple, transparent implementation.
   - Model is encouraged to follow up with `git diff -- <path>` for auditability.
3. **Profiles / modes baked in**
   - `safe-read-only`, `dev`, `unsafe` via env vars.
   - Separate handling of `git reset --hard`, `git push --force`, etc.
4. **Command/event logging in `.agent_logs/commands.log`**
   - Every command and edit is logged with timestamp and RC.
   - Very aligned with industrial safety/compliance thinking.
5. **Automatic project overview at startup**
   - `find . -maxdepth 4 ... | head -n 200` captured as an initial “user” message to seed the model’s internal world model of the repo.

`RAGIX` is a compact, opinionated variant tuned for:

- local Unix expertise,
- safety/logging,
- reproducibility,
- and specific workflows (scientific/industrial code, git, configs, logs).

