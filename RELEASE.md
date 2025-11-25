## What's New

### 🚀 Launcher & Web GUI
 - `./launch_ragix.sh` – One-command setup with conda
 - `ragix_app.py` – Streamlit dashboard at [http://localhost:8501](http://localhost:8501/)

### 🤖 Multi-Agent Workflows
 - 8 built-in templates: `bug_fix`, `feature_addition`, `code_review`, and more
 - Graph-based execution with explicit dependencies
 - Streaming events for real-time progress and status updates

### 🔍 Hybrid Search
 - BM25 keyword + vector semantic search pipeline
 - 5 fusion strategies (RRF, weighted, and others) for flexible ranking
 - Code-aware tokenization optimized for repositories

### 🔒 Sovereignty
 - 🟢 Ollama: 100% local, air-gapped capable
 - 🔴 Claude / OpenAI: optional cloud backends with explicit warnings

## Quick Start

```bash
./launch_ragix.sh
```

or

```
./launch_ragix.sh gui
```





> 🤖🟢 All features remain 100% sovereign (local-first)

  Enjoy sovereignty. [Olivier Vitrac, PhD, HDR](olivier.vitrac@adservio.fr)
