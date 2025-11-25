# RAGIX v0.7 Examples

This directory contains examples demonstrating RAGIX v0.7 features.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25

---

## Quick Start: Launcher

The easiest way to run RAGIX is via the launcher script:

```bash
# From RAGIX root directory
./launch_ragix.sh           # Interactive menu
./launch_ragix.sh gui       # Launch Web GUI directly
./launch_ragix.sh demo      # Run Claude demo
./launch_ragix.sh mcp       # Start MCP server
./launch_ragix.sh test      # Run LLM backend test
```

The launcher will:
1. Initialize Conda automatically
2. Create `ragix-env` environment if missing
3. Install all dependencies
4. Check Ollama status and available models
5. Launch the selected component

---

## Web Interface

RAGIX includes a Streamlit-based web interface (`ragix_app.py`):

```bash
# Via launcher (recommended)
./launch_ragix.sh gui

# Or directly
streamlit run ragix_app.py
```

**Features:**
- Dashboard with sovereignty status
- Hybrid search interface (BM25 + Vector)
- Chat with local LLMs
- Workflow template browser
- System health monitoring

**URL:** http://localhost:8501

---

## Examples

### 1. Claude Demo (`claude_demo.py`) ⭐

**Comprehensive demonstration for Claude integration** - the complete showcase of RAGIX v0.7:

- Full RAGIX capabilities accessible via MCP
- Multi-agent workflow execution
- Hybrid search (BM25 + vector)
- Production monitoring and health checks
- Resilience patterns (retry, circuit breaker, fallback)

```bash
# Run full demo
python examples/claude_demo.py

# Run specific component
python examples/claude_demo.py --component health
python examples/claude_demo.py --component search --query "error handling"
python examples/claude_demo.py --component templates
python examples/claude_demo.py --component tools

# Output as JSON
python examples/claude_demo.py --json
```

This demo can also be called through Claude via MCP tools:
- `ragix_health()` - System health status
- `ragix_search(query)` - Hybrid search
- `ragix_templates()` - List workflow templates
- `ragix_workflow(template, params)` - Execute workflows

### 2. LLM Backend Test (`test_llm_backends.sh`) 🆕

**Real integration test** that makes actual calls to Ollama and compares models:

```bash
# Make executable and run
chmod +x examples/test_llm_backends.sh
./examples/test_llm_backends.sh
```

What it tests:
- 🟢 Verifies Ollama is running
- 📊 Lists available models
- ⚡ Compares response times between models
- 🧪 Tests with 3 different prompts (greeting, code explanation, code generation)
- 🏆 Shows speed ranking

Example output:
```
🏆 Speed Ranking (fastest first):
   🥇 granite3.1-moe:3b: 2.01s avg
   🥈 mistral:latest: 2.43s avg

⚡ granite3.1-moe:3b is 1.2x faster than mistral:latest
```

**Key insight:** Both models are 🟢 SOVEREIGN - no data leaves your machine!

---

### 3. Workflow Example (`workflow_example.py`)

Demonstrates multi-agent workflows:
- Workflow templates for common tasks
- Graph-based execution with dependencies
- Streaming events for real-time updates
- Agent monitoring and health checks

```bash
python examples/workflow_example.py
```

### 4. Hybrid Search Example (`hybrid_search_example.py`)

Demonstrates hybrid retrieval:
- BM25 sparse keyword search
- Code-aware tokenization (camelCase, snake_case)
- Multiple fusion strategies (RRF, weighted, interleave)
- Source tracking for provenance

```bash
python examples/hybrid_search_example.py
```

### 5. Resilience Example (`resilience_example.py`)

Demonstrates fault tolerance patterns:
- Retry with exponential backoff
- Circuit breaker for failure protection
- Rate limiting with token bucket
- Graceful degradation
- Bulkhead for concurrency control
- Fallback chains

```bash
python examples/resilience_example.py
```

## Quick Start

1. Install dependencies:
```bash
pip install -e .
```

2. Run an example:
```bash
cd /path/to/RAGIX
python examples/workflow_example.py
```

## Key Features Demonstrated

### Workflow Templates

```python
from ragix_core import get_template_manager

manager = get_template_manager()
graph = manager.instantiate("bug_fix", {
    "bug_description": "TypeError in handler.py",
    "affected_files": "src/handlers/",
})
```

Available templates:
- `bug_fix` - Locate, diagnose, fix, and test bugs
- `feature_addition` - Design, implement, test, and document features
- `code_review` - Quality and security review
- `refactoring` - Analyze, plan, refactor, verify
- `documentation` - Analyze code and generate docs
- `security_audit` - Static analysis and dependency checks
- `test_coverage` - Analyze and improve test coverage
- `exploration` - Codebase exploration and analysis

### Streaming Execution

```python
from ragix_core import GraphExecutor, StreamEvent

executor = GraphExecutor(graph)

async for event in executor.execute_streaming(agent_factory):
    if isinstance(event, StreamEvent):
        print(f"[{event.event_type}] {event.node_id}")
```

### Hybrid Search

```python
from ragix_core import create_hybrid_engine, FusionStrategy

engine = create_hybrid_engine(
    index_path=Path(".ragix/index"),
    embedding_model="all-MiniLM-L6-v2",
)

results = engine.search(
    "database connection error",
    k=10,
    strategy=FusionStrategy.RRF,
)
```

### Caching

```python
from ragix_core import create_llm_cache

cache = create_llm_cache(
    cache_type="memory",
    max_size=1000,
    ttl=3600,
)

cached = cache.get("What is Python?")
if not cached:
    response = llm.generate(query)
    cache.set(query, response)
```

### Monitoring

```python
from ragix_core import get_health_checker, get_agent_monitor

# Health checks
checker = get_health_checker()
report = checker.get_status_report()

# Agent monitoring
monitor = get_agent_monitor()
monitor.start_execution("exec_001", "Fix bug", "code")
monitor.record_tool_call("read_file", success=True, duration=0.1)
monitor.end_execution("exec_001", success=True, agent_type="code")
```

### Resilience

```python
from ragix_core import retry_async, RetryConfig, BackoffStrategy, CircuitBreaker

# Retry with backoff
@retry_async(RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    strategy=BackoffStrategy.EXPONENTIAL_JITTER,
))
async def unreliable_call():
    ...

# Circuit breaker
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
if breaker.is_allowed():
    try:
        result = await api.call()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
```

## Requirements

- Python 3.10+
- RAGIX core package
- Optional: sentence-transformers (for semantic search)
- Optional: psutil (for memory monitoring)
