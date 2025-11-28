# Local Unix-RAG Agent Demo — RAGIX v0.4

**Author:** Olivier Vitrac, PhD, HDR — *Adservio Innovation Lab*  
**Contact:** [olivier.vitrac@adservio.fr](mailto:olivier.vitrac@adservio.fr)

---

## 1. Context & Objective

This demo illustrates the usage of **RAGIX v0.4**, a fully local, sovereign AI
agent integrating:

- Local LLM reasoning (Ollama)
- A safe, sandboxed Unix environment
- JSON-based action protocol
- Unix-RAG retrieval (grep, find, head, tail, awk, python)
- Optional **RAGIX Unix Toolbox** (`rt-find`, `rt-grep`, etc.)
- Optional **MCP integration** (Claude Desktop, Claude Code, Codex)

The agent can:

- Explore codebases  
- Read & summarize logs  
- Inspect configuration  
- Safely edit files  
- Interact with Git  
- Provide fully logged, reproducible execution trails  

Everything runs **locally**, with no cloud dependency.

---

## 2. Pre-Requisites (v0.4)

### 2.1 Software

- Python 3.10+
- Ollama installed & running
- `mistral` model pulled:

```bash
ollama pull mistral
~~~

### 2.2 RAGIX Installation

```bash
git clone https://github.com/ovitrac/RAGIX.git
cd RAGIX
pip install -r requirements.txt
```

### 2.3 Environment Variables (v0.4)

```bash
export UNIX_RAG_SANDBOX=~/agent-sandbox
export UNIX_RAG_MODEL=mistral
export UNIX_RAG_PROFILE=dev          # safe-read-only | dev | unsafe
export UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=0
```

------

## 3. Launching the Agent

### Terminal A — Start Ollama

```bash
ollama serve
```

### Terminal B — Start RAGIX

```bash
cd ~/RAGIX
python3 unix-rag-agent.py
```

You will see a banner and project overview scan.

------

## 4. Demo Workflow

### 4.1 Project Structure Summary

```text
You: Summarize the structure of this project.
```

The agent internally uses:

- `find`
- limited scanning depth
- ASCII summary

### 4.2 Log Analysis

```text
You: Analyze logs/system.log and summarize the anomalies.
```

Agent uses:

- `grep -n ERROR logs/system.log`
- `tail`/`head` depending on context

### 4.3 Inspect Configuration & Models

```text
You: Inspect config/system.yaml and relate it to the safety margin logic.
```

Internally:

- `sed -n '1,80p' config/system.yaml`
- `sed -n '1,120p' src/simulation/gas_flow_model.py`

### 4.4 Apply a Fix (Structured Editing)

```text
You: Lower the alert threshold from 0.20 to 0.10 in config/system.yaml.
```

The agent emits:

```json
{
  "action": "edit_file",
  "path": "config/system.yaml",
  "old": "alert_threshold: 0.20",
  "new": "alert_threshold: 0.10"
}
```

Then uses Git to diff the change.

### 4.5 Optional Tests

```text
You: Run pytest and summarize the results.
```

Agent:

- `pytest -q | tail -n 40`

### 4.6 Git Summary

```text
You: Show me git status and what has changed.
```

------

## 5. Using the RAGIX Toolbox (Optional)

Examples:

```bash
rt-find ~/agent-sandbox --ext py
rt-grep --root . -e ERROR
rt-grep --root . -e password -l | rt-grep --from-stdin -e DEBUG -l
rt-replace --root src --ext py --old foo --new bar --dry-run
rt-doc2md docs/ --json
```

------

## 6. Using RAGIX via MCP (v0.4)

Install server:

```bash
mcp install MCP/ragix_mcp_server.py --name "RAGIX"
```

Available tools:

### `ragix_chat(prompt)`

Perform one reasoning + shell step.

### `ragix_scan_repo(max_depth=4)`

Return structure of SANDBOX_ROOT.

### `ragix_read_file(path)`

Safe file reading.

------

## 7. Closing

RAGIX v0.4 provides:

- Sovereign LLM orchestration
- Deterministic shell execution
- Unix-RAG retrieval
- Git-aware safe productivity
- MCP integration
- Reproducible workflows

A fully local alternative to cloud-based Claude Code.
