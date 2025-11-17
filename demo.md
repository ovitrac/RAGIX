------

# Local Unix-RAG Agent Demo

**Author:** *Olivier Vitrac, PhD, HDR | Adservio Innovation Lab*
 **Contact:** *[olivier.vitrac@adservio.fr](mailto:olivier.vitrac@adservio.fr)*

------

## 1. Context & Objective

This demo showcases a **local AI assistant** that can:

- Explore and understand an existing **industrial codebase** (simulation, monitoring, analytics).
- Inspect and summarize **logs** and **configuration**.
- Interact with **git** (status, diff, log, show).
- Safely **edit files** through a high-level API.
- Keep a structured log of all commands and edits.

Everything runs:

- Locally (no data leaves the machine).
- On a standard workstation with ~8 GB GPU VRAM.
- Using a **local LLM** (Mistral via Ollama) as an **orchestrator** over bash and Python tools.

The assistant uses a **Unix-RAG pattern**: `find`, `grep`, `head`, `tail`, `awk`, `python`, etc., to retrieve just the relevant context instead of building a separate vector index.

------

## 2. Pre-requisites

### 2.1 Software

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running
- A Mistral model pulled, e.g.:

```bash
ollama pull mistral
```

### 2.2 Agent Script

You should have the extended `unix_rag_agent.py` (the version with:

- project overview,
- git-aware mode,
- `edit_file` tool,
- structured logging,
- profiles/modes).

Place it anywhere convenient, e.g.:

```bash
~/unix-rag-demo/unix_rag_agent.py
```

### 2.3 Demo Project (Sandbox Content)

Create a sandbox directory that simulates a typical industrial project:

```bash
mkdir -p ~/agent-sandbox
cd ~/agent-sandbox
```

Example structure (you can adapt to your real code):

```bash
mkdir -p src/monitoring src/simulation config logs tests
touch Makefile
```

Create a small Python module:

```bash
cat > src/simulation/gas_flow_model.py << 'EOF'
import math

def compute_flow(pressure_in, pressure_out, resistance):
    """Simple placeholder model for demo purposes."""
    if resistance <= 0:
        raise ValueError("resistance must be positive")
    return (pressure_in - pressure_out) / resistance

def safety_margin(flow, max_flow):
    return (max_flow - flow) / max_flow
EOF
```

Create a config file:

```bash
cat > config/system.yaml << 'EOF'
max_flow: 1200.0
alert_threshold: 0.20   # fraction of safety margin triggering an alert
log_level: INFO
EOF
```

Create a log file simulating anomalies:

```bash
cat > logs/system.log << 'EOF'
2025-11-10 08:01:12 INFO  pipeline starting
2025-11-10 08:03:45 WARNING flow close to limit: 1150.0
2025-11-10 09:10:11 ERROR safety margin below threshold: 0.15
2025-11-10 09:10:12 ERROR alert failed to trigger
EOF
```

Set up git (optional but strongly recommended for the demo):

```bash
git init
git add .
git commit -m "Initial demo project: simulation, config, logs"
```

------

## 3. Launching the Agent

Open **terminal A** and start Ollama:

```bash
ollama serve
```

Open **terminal B** and run the agent:

```bash
cd ~/unix-rag-demo   # where unix_rag_agent.py lives

# Example environment (profile and sandbox root)
export UNIX_RAG_SANDBOX=~/agent-sandbox
export UNIX_RAG_MODEL=mistral
export UNIX_RAG_PROFILE=dev          # or "safe-read-only"
export UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=0

python3 unix_rag_agent.py
```

You should see:

```text
Unix-RAG Agent (Ollama Edition)
---------------------------------------------
Sandbox root : /home/you/agent-sandbox
Profile      : dev
Dry-run      : False
Model        : mistral
=============================================

Initial project overview has been collected (if available).
You can start by asking, for example:
  - "Summarize the structure of this project."
  - "Search where a certain function or variable is defined."
```

------

## 4. Demo Flow (Step-by-Step Scenario)

### 4.1 Step 1 — Project Overview & Structure

**Goal:** Show that the agent *starts* by scanning the project and can summarize it.

You say:

```text
You: Summarize the structure of this project and identify main components.
```

The agent internally has already run:

```bash
find . -maxdepth 4 -type f \
  \( -name '*.py' -o -name '*.ipynb' -o -name 'Makefile' -o -name 'CMakeLists.txt' \) \
  2>/dev/null | head -n 200
```

Expected response (approximate, not exact):

```text
Assistant:
The project appears to contain:

- src/simulation/gas_flow_model.py: a simple simulation module.
- config/system.yaml: configuration (max_flow, alert_threshold, log_level).
- logs/system.log: runtime log with warnings and errors.
- Makefile: build or test entry point (not yet inspected).
- tests/: currently empty or not yet populated.

Broad roles:
- simulation: core model (flow computation, safety margin).
- config: operational parameters (limits, thresholds).
- logs: historical events and anomalies.
```

**Point to highlight to the audience:**

- The overview came from **Unix tools** (`find`, limited by depth and count).
- No external indexing or database is required.

------

### 4.2 Step 2 — Log Analysis (Unix-RAG)

**Goal:** Show grep/head/tail + basic analysis.

You say:

```text
You: Analyze logs/system.log, find all errors, and summarize what is going wrong.
```

Internally, the agent may issue commands like:

```json
{"action": "bash", "command": "grep -n 'ERROR' logs/system.log | head"}
```

Then possibly:

```json
{"action": "bash", "command": "sed -n '1,40p' logs/system.log"}
```

Expected user-visible answer:

```text
Assistant:
The log contains two ERROR entries:

- 2025-11-10 09:10:11: "safety margin below threshold: 0.15"
- 2025-11-10 09:10:12: "alert failed to trigger"

Interpretation:
- The system detects that the safety margin has fallen below the configured threshold.
- Immediately after, it logs that an alert failed to trigger, suggesting a bug or misconfiguration
  in the alerting mechanism or in the threshold logic.
```

**Highlight:**

- No large file dump, only targeted extraction.
- Behavior is reproducible and token-efficient.

------

### 4.3 Step 3 — Inspect Config & Simulation Code

**Goal:** Connect logs to configuration and model.

You say:

```text
You: Show me the relevant configuration parameters and how they relate to the safety margin.
```

Agent will first inspect `config/system.yaml`:

```json
{"action": "bash", "command": "sed -n '1,80p' config/system.yaml"}
```

Then inspect the simulation module:

```json
{"action": "bash", "command": "sed -n '1,120p' src/simulation/gas_flow_model.py"}
```

Expected answer:

```text
Assistant:
Configuration (config/system.yaml):
- max_flow: 1200.0
- alert_threshold: 0.20 (fraction of safety margin)
- log_level: INFO

Simulation code (gas_flow_model.py):
- compute_flow(pressure_in, pressure_out, resistance): simple linear model for flow.
- safety_margin(flow, max_flow): (max_flow - flow) / max_flow

Relationship:
- safety_margin is a fraction between 0 and 1.
- alert_threshold = 0.20 means an alert should trigger when the safety margin is 20% or less.
- In the logs, the safety margin is 0.15 (< 0.20), yet the alert fails to trigger, which is consistent with the ERROR messages.
```

------

### 4.4 Step 4 — Propose & Apply a Fix (File Editing Tool)

**Goal:** Use the `edit_file` action to adjust configuration or fix a bug, and show git diff.

You ask:

```text
You: Lower the alert threshold from 0.20 to 0.10 in config/system.yaml, apply the change, and show me the diff.
```

The agent *strategy* (explained to the audience):

1. Inspect `config/system.yaml` to make sure the pattern is correct.
2. Use `{"action": "edit_file", ...}` to perform a **precise** text replacement.
3. Run `git diff -- config/system.yaml` via a bash action.

Under the hood, it might send:

```json
{
  "action": "edit_file",
  "path": "config/system.yaml",
  "old": "alert_threshold: 0.20   # fraction of safety margin triggering an alert",
  "new": "alert_threshold: 0.10   # fraction of safety margin triggering an alert"
}
```

You’ll see a user-visible summary:

```text
Assistant:
Edited file 'config/system.yaml': replaced the first occurrence of the specified line.
Now showing the diff:
```

Then a bash command:

```json
{"action": "bash", "command": "git diff -- config/system.yaml"}
```

Output in the terminal:

```text
[Command executed]
$ git diff -- config/system.yaml
(cwd: /home/you/agent-sandbox)

STDOUT:
diff --git a/config/system.yaml b/config/system.yaml
index 1234567..89abcde 100644
--- a/config/system.yaml
+++ b/config/system.yaml
@@ -1,3 +1,3 @@
 max_flow: 1200.0
-alert_threshold: 0.20   # fraction of safety margin triggering an alert
+alert_threshold: 0.10   # fraction of safety margin triggering an alert
 log_level: INFO

Return code: 0
```

**Highlight:**

- The edit is done via a **structured tool** (`edit_file`) – not arbitrary `sed -i` magic.
- The agent logs the edit in `.agent_logs/commands.log`.
- Git diff provides a clear, auditable view of what changed.

------

### 4.5 Step 5 — Run Checks / Tests (Optional)

If you add a small test script, e.g.:

```bash
cat > tests/test_margin.py << 'EOF'
from src.simulation.gas_flow_model import safety_margin

def test_thresholds():
    flow = 1100.0
    max_flow = 1200.0
    margin = safety_margin(flow, max_flow)
    assert 0 <= margin <= 1.0
EOF

git add tests/test_margin.py
git commit -m "Add basic safety margin test"
```

You can demo:

```text
You: Run pytest (or python -m pytest) and summarize the results.
```

Agent might run:

```json
{"action": "bash", "command": "pytest -q 2>&1 | tail -n 40"}
```

Then summarize:

```text
Assistant:
The test suite ran successfully:

- 1 test passed: tests/test_margin.py::test_thresholds
- No failures.

This indicates that the current safety_margin implementation behaves as expected for the tested case.
```

------

### 4.6 Step 6 — Git History & Summary of Changes

**Goal:** Show git awareness: `git status`, `git diff`, `git log`.

You:

```text
You: Show me the current git status and summarize what has changed from the last commit.
```

Expected sequence:

```json
{"action": "bash", "command": "git status -sb"}
{"action": "bash", "command": "git diff"}
```

Then a natural language summary:

```text
Assistant:
Git status:
- On branch master
- One modified file: config/system.yaml

Summary of changes:
- The alert_threshold parameter was reduced from 0.20 to 0.10 in config/system.yaml.
This narrows the margin required to trigger alerts and may increase system sensitivity.
```

You can follow with:

```text
You: Show me a compact history of recent commits.
```

Agent:

```json
{"action": "bash", "command": "git log --oneline --graph -n 5"}
```

Terminal shows a compact history graph.

------

### 4.7 Step 7 — Inspect Command Logs (Auditability)

**Goal:** Show `.agent_logs/commands.log` as an audit trail.

You say:

```text
You: Show me the last 10 commands you ran, from your own logs.
```

Agent runs:

```json
{"action": "bash", "command": "tail -n 10 .agent_logs/commands.log"}
```

You see entries like:

```text
[2025-11-16T10:21:03.123456] CMD="find . -maxdepth 4 -type f ..." CWD='/home/you/agent-sandbox' RC=0
[2025-11-16T10:23:17.654321] CMD="grep -n 'ERROR' logs/system.log | head" CWD='/home/you/agent-sandbox' RC=0
[2025-11-16T10:25:00.111111] EVENT=EDIT_FILE_OK DETAILS=Edited file 'config/system.yaml': replaced first occurrence of given 'old' text.
...
```

**Highlight:**

- Every command and edit is logged with a timestamp and return code.
- This supports **traceability** and **compliance** needs.

------

## 5. Profiles / Modes (Optional Slide / Talking Point)

Explain how to switch modes:

```bash
# Safe read-only profile (no real changes)
export UNIX_RAG_PROFILE=safe-read-only

# Developer mode (normal behavior, denylist still active)
export UNIX_RAG_PROFILE=dev

# "Unsafe" mode (intended only for expert use)
export UNIX_RAG_PROFILE=unsafe

# Allow destructive git operations (only if necessary and explicitly agreed)
export UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=1
```

Key messages:

- In **safe-read-only**, everything is effectively dry-run.
- In **dev**, file edits and normal git commands work; hard-dangerous commands are blocked.
- Destructive git operations (reset --hard, push --force, clean -fd) are blocked unless explicitly allowed by an environment variable.

------

## 6. Closing Message (for Presentation)

You can close with something like:

> What we’ve seen is a local AI assistant that:
>
> - Understands and navigates an industrial codebase using Unix tools.
> - Analyzes logs and configuration to diagnose anomalies.
> - Safely modifies configuration with a structured file edit tool.
> - Integrates with existing git workflows.
> - Logs every command and edit for traceability.
>
> All of this happens locally, under our control, using an open-source LLM and a thin, auditable orchestration layer.

