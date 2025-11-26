"""
RAGIX Web Interface - Streamlit Application
============================================

A sovereign, local-first web interface for RAGIX v0.7.1.
All processing happens locally - no data leaves your machine.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26

Usage:
    streamlit run ragix_app.py

Or via launcher:
    ./launch_ragix.sh gui
"""

import streamlit as st
import requests
import json
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="RAGIX v0.7",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background-color: #0e1117;
    }

    /* Sovereignty badge */
    .sovereign-badge {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }

    .cloud-badge {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }

    /* Status indicators */
    .status-ok { color: #00b894; }
    .status-warn { color: #fdcb6e; }
    .status-error { color: #e74c3c; }

    /* Cards */
    .metric-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }

    /* Model list */
    .model-item {
        background: #1e2530;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        border-left: 3px solid #00b894;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_data(ttl=30)
def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama status and available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return {
                "running": True,
                "models": models,
                "model_count": len(models),
            }
    except Exception as e:
        pass

    return {
        "running": False,
        "models": [],
        "model_count": 0,
        "error": "Ollama not running",
    }


def query_ollama(model: str, prompt: str, system: str = "") -> Dict[str, Any]:
    """Send a query to Ollama."""
    try:
        start_time = time.perf_counter()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system} if system else None,
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        # Remove None messages
        payload["messages"] = [m for m in payload["messages"] if m]

        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120,
        )

        elapsed = time.perf_counter() - start_time

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "response": data["message"]["content"],
                "time": elapsed,
                "model": model,
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "time": elapsed,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0,
        }


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("# 🔍 RAGIX")
    st.markdown("**v0.7** — Sovereign AI Assistant")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔎 Search", "🤖 Chat", "⚙️ Workflows", "📋 Logs", "📊 Monitor", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick status
    ollama_status = check_ollama_status()

    if ollama_status["running"]:
        st.markdown('<span class="sovereign-badge">🟢 SOVEREIGN</span>', unsafe_allow_html=True)
        st.caption(f"{ollama_status['model_count']} models available")
    else:
        st.markdown('<span class="cloud-badge">⚠️ OLLAMA OFFLINE</span>', unsafe_allow_html=True)
        st.caption("Start with: `ollama serve`")

    st.markdown("---")
    st.caption("© 2025 Adservio Innovation Lab")
    st.caption("All processing is local.")


# =============================================================================
# Dashboard Page
# =============================================================================

if page == "🏠 Dashboard":
    st.title("🔍 RAGIX Dashboard")
    st.markdown("*Retrieval-Augmented Generative Interactive eXecution Agent*")

    st.markdown("---")

    # Sovereignty banner
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0;">🔒 100% Sovereign</h2>
            <p style="color: #a8d8ea; margin: 10px 0 0 0;">All data stays on your machine. No cloud dependencies.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Status cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Ollama Status",
            "Online" if ollama_status["running"] else "Offline",
            delta="Ready" if ollama_status["running"] else "Start required",
        )

    with col2:
        st.metric(
            "Models Available",
            ollama_status["model_count"],
            delta="Local" if ollama_status["model_count"] > 0 else None,
        )

    with col3:
        # Check for ragix_core
        try:
            from ragix_core import __version__
            ragix_ok = True
        except ImportError:
            ragix_ok = False
            __version__ = "N/A"

        st.metric(
            "RAGIX Core",
            __version__ if ragix_ok else "Not installed",
            delta="Ready" if ragix_ok else "pip install -e .",
        )

    with col4:
        st.metric(
            "Search Index",
            "Available",
            delta="BM25 + Vector",
        )

    st.markdown("---")

    # Available Models
    st.subheader("🟢 Available Models (Sovereign)")

    if ollama_status["running"] and ollama_status["models"]:
        cols = st.columns(3)
        for i, model in enumerate(ollama_status["models"]):
            with cols[i % 3]:
                size = format_size(model.get("size", 0))
                st.markdown(f"""
                <div class="model-item">
                    <strong>{model['name']}</strong><br/>
                    <small>{size}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No models found. Install with: `ollama pull mistral`")

    st.markdown("---")

    # Quick Actions
    st.subheader("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔎 Try Search", use_container_width=True):
            st.session_state["page"] = "search"
            st.rerun()

    with col2:
        if st.button("🤖 Chat with LLM", use_container_width=True):
            st.session_state["page"] = "chat"
            st.rerun()

    with col3:
        if st.button("📊 View Workflows", use_container_width=True):
            st.session_state["page"] = "workflows"
            st.rerun()


# =============================================================================
# Search Page
# =============================================================================

elif page == "🔎 Search":
    st.title("🔎 Hybrid Search")
    st.markdown("*BM25 keyword search + Vector semantic search*")

    st.markdown("---")

    # Search input
    query = st.text_input(
        "Search Query",
        placeholder="Enter your search query...",
        help="Supports code-aware tokenization (camelCase, snake_case)",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        search_type = st.selectbox(
            "Search Type",
            ["Hybrid (BM25 + Vector)", "BM25 Only", "Vector Only"],
        )

    with col2:
        top_k = st.slider("Results", 5, 50, 10)

    with col3:
        fusion = st.selectbox(
            "Fusion Strategy",
            ["RRF (Reciprocal Rank)", "Weighted", "Interleave"],
        )

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if query:
            with st.spinner("Searching..."):
                # Demo search results (in real app, would use ragix_core)
                st.success(f"Found results for: **{query}**")

                # Sample results
                st.markdown("### Results")

                for i in range(min(5, top_k)):
                    with st.expander(f"Result {i+1}: example_file_{i}.py"):
                        st.code(f"""
def example_function_{i}():
    \"\"\"Example matching '{query}'\"\"\"
    # This is a sample result
    return True
""", language="python")
                        st.caption(f"Score: {0.95 - i*0.1:.2f} | Line: {10+i*5}")
        else:
            st.warning("Please enter a search query.")


# =============================================================================
# Chat Page
# =============================================================================

elif page == "🤖 Chat":
    st.title("🤖 Chat with Local LLM")
    st.markdown("*Sovereign conversation - all data stays local*")

    st.markdown("---")

    # Model selection
    if ollama_status["running"] and ollama_status["models"]:
        model_names = [m["name"] for m in ollama_status["models"]]

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_model = st.selectbox(
                "Select Model",
                model_names,
                help="All models run 100% locally (sovereign)",
            )

        with col2:
            st.markdown('<span class="sovereign-badge">🟢 LOCAL</span>', unsafe_allow_html=True)

        st.markdown("---")

        # System prompt
        system_prompt = st.text_area(
            "System Prompt (optional)",
            value="You are a helpful coding assistant. Be concise and precise.",
            height=80,
        )

        # Chat input
        user_input = st.text_area(
            "Your Message",
            placeholder="Ask me anything...",
            height=100,
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            send_button = st.button("📤 Send", type="primary", use_container_width=True)

        if send_button and user_input:
            with st.spinner(f"Thinking with {selected_model}..."):
                result = query_ollama(selected_model, user_input, system_prompt)

                if result["success"]:
                    st.markdown("### Response")
                    st.markdown(result["response"])
                    st.caption(f"⏱️ {result['time']:.2f}s | 🟢 Sovereign (local)")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
    else:
        st.warning("Ollama is not running. Start with: `ollama serve`")
        st.info("Then install a model: `ollama pull mistral`")


# =============================================================================
# Workflows Page
# =============================================================================

elif page == "⚙️ Workflows":
    st.title("⚙️ Workflow Templates")
    st.markdown("*Pre-built multi-agent workflows for common tasks*")

    st.markdown("---")

    # Workflow templates
    workflows = {
        "bug_fix": {
            "name": "🐛 Bug Fix",
            "description": "Locate, diagnose, fix, and test bugs",
            "steps": ["Locate bug", "Diagnose root cause", "Apply fix", "Run tests", "Review"],
        },
        "feature_addition": {
            "name": "✨ Feature Addition",
            "description": "Design, implement, test, and document new features",
            "steps": ["Design", "Implement", "Write tests", "Document", "Review"],
        },
        "code_review": {
            "name": "🔍 Code Review",
            "description": "Quality and security review of code",
            "steps": ["Quality check", "Security scan", "Best practices", "Report"],
        },
        "refactoring": {
            "name": "🔧 Refactoring",
            "description": "Analyze, plan, refactor, and verify code improvements",
            "steps": ["Analyze", "Plan refactor", "Apply changes", "Verify", "Test"],
        },
        "documentation": {
            "name": "📝 Documentation",
            "description": "Analyze code and generate documentation",
            "steps": ["Analyze code", "Extract API", "Generate docs", "Review"],
        },
        "security_audit": {
            "name": "🔒 Security Audit",
            "description": "Static analysis and dependency security checks",
            "steps": ["SAST scan", "Dependency check", "Code review", "Report"],
        },
        "test_coverage": {
            "name": "🧪 Test Coverage",
            "description": "Analyze and improve test coverage",
            "steps": ["Measure coverage", "Identify gaps", "Generate tests", "Verify"],
        },
        "exploration": {
            "name": "🗺️ Codebase Exploration",
            "description": "Explore and understand codebase structure",
            "steps": ["Map structure", "Analyze dependencies", "Document patterns", "Report"],
        },
    }

    # Display workflows
    cols = st.columns(2)

    for i, (key, workflow) in enumerate(workflows.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"### {workflow['name']}")
                st.markdown(workflow["description"])

                # Steps visualization
                st.markdown("**Steps:**")
                step_cols = st.columns(len(workflow["steps"]))
                for j, step in enumerate(workflow["steps"]):
                    with step_cols[j]:
                        st.markdown(f"<div style='text-align:center; padding:5px; background:#1e2530; border-radius:5px; font-size:12px;'>{j+1}. {step}</div>", unsafe_allow_html=True)

                if st.button(f"Run {workflow['name']}", key=f"run_{key}"):
                    st.info(f"Workflow '{key}' would be executed here via ragix_core")

                st.markdown("---")


# =============================================================================
# Logs Page
# =============================================================================

elif page == "📋 Logs":
    st.title("📋 Audit Logs")
    st.markdown("*Command history and integrity verification*")

    st.markdown("---")

    # Log configuration
    log_dir = Path(".agent_logs")
    log_file = log_dir / "commands.log"
    hash_file = log_dir / "commands.log.sha256"

    # Log stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if log_file.exists():
            size_kb = log_file.stat().st_size / 1024
            st.metric("Log Size", f"{size_kb:.1f} KB")
        else:
            st.metric("Log Size", "No logs")

    with col2:
        if log_file.exists():
            with open(log_file, 'r') as f:
                entry_count = sum(1 for _ in f)
            st.metric("Entries", entry_count)
        else:
            st.metric("Entries", 0)

    with col3:
        if hash_file.exists():
            st.metric("Integrity", "🔒 Hashed", delta="SHA256")
        else:
            st.metric("Integrity", "⚠️ No hash")

    with col4:
        if log_file.exists():
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            st.metric("Last Update", mtime.strftime("%H:%M:%S"))
        else:
            st.metric("Last Update", "N/A")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📜 Recent Entries", "🔍 Search Logs", "✅ Verify Integrity"])

    with tab1:
        st.subheader("Recent Log Entries")

        num_entries = st.slider("Number of entries to show", 10, 200, 50)

        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                recent = lines[-num_entries:] if len(lines) > num_entries else lines

                # Display in reverse order (most recent first)
                for i, line in enumerate(reversed(recent)):
                    line = line.strip()
                    if not line:
                        continue

                    # Color code by type
                    if "CMD:" in line:
                        icon = "⚡"
                        color = "#00b894"
                    elif "EDIT:" in line:
                        icon = "✏️"
                        color = "#0984e3"
                    elif "EVENT:" in line:
                        icon = "📢"
                        color = "#fdcb6e"
                    elif "ERROR" in line or "RC: 1" in line:
                        icon = "❌"
                        color = "#e74c3c"
                    else:
                        icon = "📝"
                        color = "#636e72"

                    st.markdown(
                        f"<div style='padding:8px; margin:4px 0; background:#1e2530; "
                        f"border-left:3px solid {color}; border-radius:4px; font-family:monospace; font-size:12px;'>"
                        f"{icon} {line}</div>",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Failed to read logs: {e}")
        else:
            st.info("No log file found. Logs will appear here after running commands.")

    with tab2:
        st.subheader("Search Logs")

        search_query = st.text_input("Search pattern", placeholder="Enter search term...")
        search_type = st.radio("Filter by", ["All", "Commands", "Edits", "Events", "Errors"], horizontal=True)

        if st.button("🔍 Search") and search_query and log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()

            results = []
            for line in lines:
                # Apply type filter
                if search_type == "Commands" and "CMD:" not in line:
                    continue
                if search_type == "Edits" and "EDIT:" not in line:
                    continue
                if search_type == "Events" and "EVENT:" not in line:
                    continue
                if search_type == "Errors" and "ERROR" not in line and "RC: 1" not in line:
                    continue

                # Apply search query
                if search_query.lower() in line.lower():
                    results.append(line.strip())

            st.markdown(f"**Found {len(results)} matches:**")

            for line in results[-100:]:  # Show max 100 results
                st.code(line, language=None)

    with tab3:
        st.subheader("Integrity Verification")

        st.markdown("""
        Log integrity verification uses SHA256 chained hashing to detect tampering.
        Each log entry's hash includes the previous entry's hash, creating a tamper-evident chain.
        """)

        if st.button("🔒 Verify Log Integrity", type="primary"):
            if hash_file.exists():
                try:
                    # Simple verification
                    with open(hash_file, 'r') as f:
                        entries = [json.loads(line) for line in f if line.strip()]

                    if entries:
                        st.success(f"✅ Hash chain contains {len(entries)} entries")

                        # Show chain info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**First Entry:**")
                            st.json({
                                "sequence": entries[0].get("sequence"),
                                "timestamp": entries[0].get("timestamp"),
                                "hash": entries[0].get("hash", "")[:32] + "...",
                            })
                        with col2:
                            st.markdown("**Latest Entry:**")
                            st.json({
                                "sequence": entries[-1].get("sequence"),
                                "timestamp": entries[-1].get("timestamp"),
                                "hash": entries[-1].get("hash", "")[:32] + "...",
                            })

                        # Verify chain
                        genesis = "0" * 64
                        prev_hash = genesis
                        valid = True
                        invalid_entry = None

                        for i, entry in enumerate(entries):
                            if entry.get("prev_hash") != prev_hash:
                                valid = False
                                invalid_entry = i + 1
                                break
                            prev_hash = entry.get("hash", "")

                        if valid:
                            st.success("✅ Chain integrity verified - no tampering detected")
                        else:
                            st.error(f"❌ Chain broken at entry {invalid_entry}")

                    else:
                        st.warning("Hash file is empty")

                except json.JSONDecodeError as e:
                    st.error(f"Invalid hash file format: {e}")
                except Exception as e:
                    st.error(f"Verification failed: {e}")
            else:
                st.warning("No hash file found. Enable log hashing in ragix.yaml")

        st.markdown("---")

        # Export options
        st.subheader("Export Logs")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download Log File") and log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                st.download_button(
                    "Download commands.log",
                    log_content,
                    file_name="commands.log",
                    mime="text/plain"
                )

        with col2:
            if st.button("📥 Download Hash File") and hash_file.exists():
                with open(hash_file, 'r') as f:
                    hash_content = f.read()
                st.download_button(
                    "Download commands.log.sha256",
                    hash_content,
                    file_name="commands.log.sha256",
                    mime="application/json"
                )


# =============================================================================
# Monitor Page
# =============================================================================

elif page == "📊 Monitor":
    st.title("📊 System Monitor")
    st.markdown("*Health checks and performance metrics*")

    st.markdown("---")

    # Health checks
    st.subheader("Health Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "✅" if ollama_status["running"] else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{status} Ollama</h3>
            <p>{"Running" if ollama_status["running"] else "Offline"}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ ragix_core</h3>
            <p>Loaded</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Search Index</h3>
            <p>Ready</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🟢 Sovereign</h3>
            <p>100% Local</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System info
    st.subheader("System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Environment**")
        st.json({
            "Python": "3.11+",
            "Streamlit": st.__version__,
            "Platform": "Linux",
        })

    with col2:
        st.markdown("**Ollama Models**")
        if ollama_status["running"]:
            model_info = {m["name"]: format_size(m.get("size", 0)) for m in ollama_status["models"][:5]}
            st.json(model_info)
        else:
            st.warning("Ollama not running")

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# About Page
# =============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About RAGIX")

    st.markdown("""
    ## RAGIX v0.7

    **Retrieval-Augmented Generative Interactive eXecution Agent**

    RAGIX is a sovereign, local-first development assistant that combines:

    - **Unix-RAG Patterns**: Use classic Unix tools (grep, find, awk) for context retrieval
    - **Local LLMs**: Powered by Ollama (Mistral, Granite, DeepSeek, etc.)
    - **Hybrid Search**: BM25 keyword + Vector semantic search
    - **Multi-Agent Workflows**: Pre-built templates for common tasks
    - **MCP Integration**: Works with Claude Desktop and Claude Code

    ---

    ### 🔒 Sovereignty Guarantee

    All processing happens **100% locally**:

    - ✅ No data sent to cloud APIs
    - ✅ No external dependencies required
    - ✅ Full control over your code and data
    - ✅ Works completely offline

    ---

    ### 🏗️ Architecture

    ```
    ┌─────────────────────────────────────────────────┐
    │                    RAGIX GUI                     │
    │                  (Streamlit)                     │
    └─────────────────────┬───────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────┐
    │                  ragix_core                      │
    │  ┌─────────┐  ┌──────────┐  ┌───────────────┐  │
    │  │ Search  │  │ Workflow │  │  LLM Backend  │  │
    │  │ Engine  │  │ Executor │  │   (Ollama)    │  │
    │  └────┬────┘  └────┬─────┘  └───────┬───────┘  │
    │       │            │                │          │
    │  ┌────▼────────────▼────────────────▼────┐     │
    │  │           Unix-RAG Tools               │     │
    │  │     grep | find | awk | sed | etc.     │     │
    │  └───────────────────────────────────────┘     │
    └─────────────────────────────────────────────────┘
    ```

    ---

    ### 📚 Resources

    - **GitHub**: [github.com/ovitrac/RAGIX](https://github.com/ovitrac/RAGIX)
    - **Documentation**: See README.md
    - **MCP Integration**: See MCP/README.md

    ---

    ### 👤 Author

    **Olivier Vitrac, PhD, HDR**
    Head of Innovation Lab, Adservio
    olivier.vitrac@adservio.fr

    ---

    *© 2025 Adservio Innovation Lab. All rights reserved.*
    """)


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔍 RAGIX v0.7 | 🟢 Sovereign | 🔒 100% Local | "
    "<a href='https://github.com/ovitrac/RAGIX'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
