# TODO — RAGIX Roadmap

**Updated:** 2025-12-05 (v0.32.1 - Dynamic Model Info & Context Fixes)
**Reference:** See `PLAN_v0.30_REASONING.md` for full implementation plan
**Review:** See `REVIEW_current_reasoning_towardsv0.30.md` for colleague feedback

---

## Session Completed (2025-12-05 - v0.32.1)

### Dynamic Model Info & Context Counter Fix

| Task | Status |
|------|--------|
| **Ollama API Client** - New `ragix_core/ollama_client.py` with caching | ✅ Done |
| **Dynamic Context Limits** - Fetch from Ollama `/api/show` instead of hardcoded | ✅ Done |
| **VRAM Display** - Show VRAM usage from `/api/ps` in sidebar | ✅ Done |
| **Quantization Display** - Show model quantization (Q4_K_M, etc.) in sidebar | ✅ Done |
| **Parameter Size Display** - Show model size (7B, 14B, etc.) in sidebar | ✅ Done |
| **Context Counter Fix** - Now shows current context size, not cumulative tokens | ✅ Done |
| **API Caching** - 5min TTL for model details, 30s for running models | ✅ Done |
| **Classification Bug Fix** - Tasks with "audit", "quality", "analyze" now properly classified as COMPLEX | ✅ Fixed |

**Key Files Created/Modified:**
- `ragix_core/ollama_client.py` - NEW: `OllamaClient` class with `get_model_info()`, `get_running_models()`, caching
- `ragix_core/__init__.py` - Export `OllamaClient`, `ModelInfo`, `get_ollama_client`, `get_dynamic_context_limit`
- `ragix_web/server.py` - NEW endpoints: `/api/ollama/running`, `/api/ollama/model/{name}`; Fixed `/api/sessions/{id}/context-window` to use history estimate
- `ragix_web/static/app.js` - `updateModelInfo()`, `_displayModelInfo()` methods with caching
- `ragix_web/static/index.html` - Model info row with Quant/VRAM/Size display
- `ragix_web/static/style.css` - `.model-info-row`, `.model-info-item`, `.model-info-value` styles

**Features:**
- Sidebar now shows: Quant (e.g., Q4_K_M), VRAM (e.g., 4.2G), Size (e.g., 7B)
- Context indicator shows **current** context size (what would be sent to LLM)
- After compaction, context counter properly resets to reflect reduced history
- API caching reduces Ollama API calls
- Graceful fallback to hardcoded limits if Ollama API unavailable

**API Endpoints:**
```
GET /api/ollama/running     # Running models with VRAM usage
GET /api/ollama/model/{name}  # Detailed model info (quantization, context, etc.)
```

---

## Session Completed (2025-12-04 - v0.31.1)

### Context Window Indicator & File Drop Zone

| Task | Status |
|------|--------|
| **Context Window Indicator** - Progress bar showing tokens used vs available | ✅ Done |
| **Model Context Limits** - Config per model (qwen2.5:7b=32k, mistral=128k, etc.) | ✅ Done |
| **File Drop Zone** - Drag & drop files into chat input area | ✅ Done |
| **Text File Detection** - Auto-detect text vs binary file types | ✅ Done |
| **PDF Conversion** - `pdftotext` integration for PDF→text | ✅ Done |
| **Document Conversion** - `pandoc` for docx/odt/rtf→text | ✅ Done |
| **File Context in Messages** - Prepend attached file contents to user messages | ✅ Done |
| **Converter Config** - `ragix.yaml` section for pdftotext/pandoc paths & options | ✅ Done |
| **Token Tracking Fix** - Track tokens for BYPASS/SIMPLE tasks (not just complex) | ✅ Done |
| **Duplicate Progress Fix** - Remove redundant trace emission causing duplicate cards | ✅ Done |
| **Memory Compaction** - LLM-based summarization of older messages to free context | ✅ Done |
| **Compaction Button** - Manual trigger at 80%+, auto-trigger at 95% usage | ✅ Done |
| **Memory Stats API** - `/api/sessions/{id}/memory` and `/api/sessions/{id}/compact` | ✅ Done |

**Key Files Modified:**
- `ragix_core/agent_config.py` - `MODEL_CONTEXT_LIMITS`, `get_model_context_limit()`, `get_model_info()`
- `ragix_core/config.py` - `ConvertersConfig`, `ConverterToolConfig` dataclasses
- `ragix_unix/agent.py` - Token tracking in `step()`, memory compaction methods (`compact_history()`, `should_compact()`, `get_memory_stats()`)
- `ragix_web/server.py` - `/api/sessions/{id}/context-window`, `/api/files/convert`, `/api/sessions/{id}/memory`, `/api/sessions/{id}/compact` endpoints
- `ragix_web/static/app.js` - `setupFileHandling()`, `compactMemory()`, auto-compact at 95%, file handling methods
- `ragix_web/static/index.html` - Context indicator, compact button, file drop zone, file preview UI
- `ragix_web/static/style.css` - Styles for context bar, compact button, file drop zone, preview items
- `ragix.yaml` - Added `converters` section for pdftotext/pandoc configuration

**Features:**
- Context window progress bar in sidebar with warning (≥80%) and critical (≥95%) states
- Model context limits from configuration (32k-128k depending on model)
- Memory compaction: LLM summarizes older messages to free context space
- Compact button appears at ≥80% usage, auto-compacts at ≥95%
- Drag & drop files onto chat area or use 📎 button
- Supported text files: py, js, ts, json, yaml, md, txt, html, css, sql, etc.
- Conversion for: pdf (pdftotext), docx/doc/odt/rtf (pandoc)
- File preview with icons, sizes, and remove button
- Truncation for large files (>10k chars) with indication
- Configurable converters in ragix.yaml (paths, options, timeouts)

---

## Session Completed (2025-12-04 - v0.32.0)

### Memory Management UI & Reasoning Integration

| Task | Status |
|------|--------|
| **Memory Explorer Panel** - Sidebar section with episodic memory entries | ✅ Done |
| **Memory Stats Display** - Episode count, files touched, commands run | ✅ Done |
| **Memory Search** - Debounced keyword search across all entry fields | ✅ Done |
| **Memory Entry Details** - Click to view full episode with plan, result, files, commands | ✅ Done |
| **Memory Pruning** - Delete individual entries or clear all with confirmation | ✅ Done |
| **Episodic Memory API** - CRUD endpoints for episodic memory management | ✅ Done |
| **Memory Context in Reasoning Tab** - Panel showing episodic memories during reasoning | ✅ Done |
| **Memory-Reasoning Integration** - Track which memories used per reasoning session | ✅ Done |
| **Memory Details Modal** - Proper modal with formatted sections for memory entry details | ✅ Done |
| **Memory Search in Reasoning** - Search input with Enter key support in Memory Context | ✅ Done |

**Key Files Modified:**
- `ragix_core/reasoning.py` - `EpisodicMemory` class: `list_entries()`, `search_entries()`, `delete_entry()`, `clear_all_entries()`, `get_stats()`
- `ragix_web/server.py` - New endpoints: `/api/sessions/{id}/episodic`, `/api/sessions/{id}/episodic/search`, DELETE endpoints; Enhanced reasoning state emission with goal for memory context
- `ragix_web/static/app.js` - `refreshMemory()`, `searchMemory()`, `showMemoryEntry()`, `deleteMemoryEntry()`, `clearMemory()`; WebSocket handler for memory context
- `ragix_web/static/index.html` - Memory Explorer sidebar section; Memory Context section in Reasoning tab with stats, controls, entry list; `memoryContext` JavaScript object
- `ragix_web/static/style.css` - Styles for memory explorer, entries, search input; Memory context section styles

**Features:**
- Memory Explorer sidebar section between Agents and Workflows
- Real-time stats: episode count, files touched, commands run
- Debounced search (300ms) across goals, plans, results, files, commands
- Entry cards with goal preview, timestamp, file count
- Hover-to-reveal delete button on entries
- Click entry to view full details in chat
- Refresh and Clear All buttons
- **Reasoning Tab Integration:**
  - Memory Context panel showing episodic memories
  - Stats: total memories, sessions, current goal display
  - Search input with Enter key support
  - "All" button refreshes full list
  - "Relevant" button filters by current goal, search input, or last user message
  - "Used (N)" button shows memories accessed during reasoning with count
  - Visual "Used" badges on memory entries used in current session
  - WebSocket updates when reasoning goal changes
- **Memory Details Modal:**
  - Proper modal display instead of alert()
  - Color-coded sections: Plan (purple), Result (green), Decisions (yellow), Files (blue), Commands (orange)
  - Metadata header with timestamp, file count, command count
  - Code formatting for files and commands
  - Delete button to remove entry from modal
  - Command truncation at 100 chars

---

## Session Completed (2025-12-04 - v0.31.0)

### Interrupt Reasoning, Token Counter, Progress Cards & Bug Fixes

| Task | Status |
|------|--------|
| **Interrupt Reasoning Button** - Stop button in thinking indicator to abort reasoning | ✅ Done |
| **Cancellation Token** - Thread-safe `threading.Event` in `run_agent_async()` | ✅ Done |
| **Token Counter Display** - Show ↑prompt ↓completion Σtotal, requests, tok/s | ✅ Done |
| **Progress Cards** - Inline cards for classification, steps with status badges | ✅ Done |
| **JSON Escape Fix** - Handle invalid `\(` escapes in shell commands | ✅ Done |
| **Multiline JSON Fix** - Handle unescaped newlines in LLM responses | ✅ Done |
| **Classification Fix** - Fixed "they are" matching "hey " false positive | ✅ Done |
| **Response Extraction** - Extract message from raw JSON action objects | ✅ Done |

**Key Files Modified:**
- `ragix_core/llm_backends.py` - `generate_with_stats()` for token tracking
- `ragix_core/orchestrator.py` - Fix 4 (invalid escapes), Fix 5 (multiline strings)
- `ragix_core/reasoning.py` - Word-boundary-aware bypass classification
- `ragix_unix/agent.py` - `_token_stats` tracking, `get_token_stats()`
- `ragix_web/server.py` - `session_cancellation`, `_extract_message_from_response()`
- `ragix_web/static/app.js` - Cancel button, token stats display, progress cards
- `ragix_web/static/style.css` - New styles for cancel, tokens, progress cards

**Bug Fixes:**
1. **Invalid JSON escapes**: LLMs write `\(` instead of `\\(` in bash commands
2. **Multiline JSON**: LLMs write literal newlines in strings instead of `\n`
3. **False BYPASS**: "they are" was matching "hey " pattern
4. **Raw JSON display**: BYPASS responses showed `{"action": "respond", ...}`

---

## Internal Review Feedback (2025-12-03) — Roadmap v0.31+

### Priority 1: Enhanced Reasoning Visibility (v0.31) ✅ COMPLETED

| Task | Effort | Status |
|------|--------|--------|
| **Intermediate Results Display** - Show step outputs in chat with success/warning/error styling | 4h | ✅ Done |
| **Step Status Indicators** - ✅ success, ⚠️ warning, ❌ error badges in reasoning panel | 2h | ✅ Done |
| **Interrupt Reasoning** - Stop button to abort before next Ollama call | 3h | ✅ Done |
| **Cancellation Token** - Thread-safe cancellation in `run_agent_async()` | 2h | ✅ Done |

### Priority 2: Token & Context Management (v0.31) ✅ COMPLETED

| Task | Effort | Status |
|------|--------|--------|
| **Token Counter Display** - Show input/output/reasoning tokens per request | 4h | ✅ Done |
| **Context Window Indicator** - Progress bar showing tokens used vs available | 3h | ✅ Done |
| **Model Context Limits** - Config per model (qwen2.5:7b=32k, mistral=128k, etc.) | 2h | ✅ Done |
| **Memory Compaction** - Manual/auto summarization when context fills | 6h | ✅ Done |
| **Compaction Trigger** - Button + auto-trigger at 80%/95% context usage | 2h | ✅ Done |

### Priority 3: File Handling & Conversion (v0.31) ✅ COMPLETED

| Task | Effort | Status |
|------|--------|--------|
| **File Drop Zone** - Drag & drop files into chat | 4h | ✅ Done |
| **Text File Detection** - Auto-detect text vs binary | 1h | ✅ Done |
| **PDF Conversion** - `pdftotext` integration (configurable) | 3h | ✅ Done |
| **Document Conversion** - `pandoc` for docx/odt/rtf→text | 3h | ✅ Done |
| **Converter Config** - `ragix.yaml` section for converter paths | 1h | ✅ Done |

### Priority 4: Memory Management UI (v0.32) ✅ COMPLETED

| Task | Effort | Status |
|------|--------|--------|
| **Memory Explorer Panel** - View/edit episodic memory entries | 6h | ✅ Done |
| **Memory Usage Visualization** - Stats showing episodes, files, commands | 4h | ✅ Done |
| **Memory Search** - Search through past episodes with debounce | 2h | ✅ Done |
| **Memory Pruning UI** - Delete individual entries or clear all | 2h | ✅ Done |
| **Memory Context in Reasoning** - Show episodic memory in Reasoning tab | 3h | ✅ Done |
| **RAG Context Display** - Show which RAG chunks are retrieved and used | 4h | Pending |

### Priority 5: Dynamic Model Info & Context Fixes (v0.32.1) — ✅ COMPLETED

**Problem:** Model context limits are hardcoded. Context counter doesn't reset after compaction. No VRAM/quantization info.

| Task | Effort | Status |
|------|--------|--------|
| **Ollama API Integration** - Fetch model info from `/api/ps` and `/api/show` | 3h | ✅ Done |
| **Dynamic Context Limits** - Cache and use actual model context size | 2h | ✅ Done |
| **VRAM Usage Display** - Show memory usage in sidebar/settings | 2h | ✅ Done |
| **Quantization Info** - Display model quantization (Q4_K_M, Q8, F16, etc.) | 1h | ✅ Done |
| **Fix Context Reset** - Context shows current size, not cumulative | 1h | ✅ Fixed |
| **Fix SIMPLE Classification** - "audit", "quality", "analyze" now trigger COMPLEX | 0.5h | ✅ Fixed |
| **File Context Management** - UI to summarize or remove files from context | 4h | Pending |

**Ollama API Reference:**
```bash
curl http://localhost:11434/api/ps          # Running models, VRAM usage
curl http://localhost:11434/api/show -d '{"name":"mistral"}'  # Model details, quantization
```

### Priority 6: Session & Thread Management (v0.33) — FROM REVIEW 2025-12-05

**Problem:** RAGIX feels stateless. Can't create threads. Unclear memory lifecycle between sessions.

| Task | Effort | Status |
|------|--------|--------|
| **Thread Creation UI** - Create new conversation threads | 4h | Pending |
| **Thread Switching** - Switch between active threads in sidebar | 3h | Pending |
| **Thread Persistence** - Save/restore thread state to disk | 4h | Pending |
| **Session Export** - Export conversation as markdown/JSON | 2h | Pending |
| **Global Context Editor** - Edit user profile, project context ("who I am") | 4h | Pending |
| **Episodic Memory Clarity** - UI to show active vs archived, session binding | 3h | Pending |

### Priority 7: RAG System Management (v0.33) — FROM REVIEW 2025-12-05

**Problem:** RAG activation unclear. No way to manage, feed, or sanitize the index independently.

| Task | Effort | Status |
|------|--------|--------|
| **RAG Activation UI** - Toggle switch and status indicator | 3h | Pending |
| **RAG Index Browser** - View indexed documents and chunks | 4h | Pending |
| **RAG Feed Interface** - Add documents to index (independent of LLM) | 4h | Pending |
| **RAG Sanitize/Rebuild** - Remove/update documents, rebuild index | 3h | Pending |
| **RAG Context Display** - Show retrieved chunks during reasoning | 4h | Pending |
| **RAG Documentation** - How it works, when activated, how to configure | 2h | Pending |

### Priority 8: Git & Diff Integration (v0.34)

| Task | Effort | Status |
|------|--------|--------|
| **Git Diff Tool** - Compare current vs committed versions | 4h | Pending |
| **Diff Visualization** - Side-by-side or unified diff in UI | 4h | Pending |
| **Commit History Browser** - Select commits to compare | 3h | Pending |
| **Change Summary** - AI-generated summary of changes | 2h | Pending |

### Priority 9: CLI & IDE Integration (v0.35)

| Task | Effort | Status |
|------|--------|--------|
| **`ragix` CLI Command** - Shell interface like Claude/Codex | 8h | Pending |
| **CLI Parameters** - `--resume`, `-c "prompt"`, `--model`, etc. | 4h | Pending |
| **Colored Output** - Claude-style blue theme (RAGIX brand colors) | 3h | Pending |
| **Interactive REPL** - Continuous conversation mode | 4h | Pending |
| **VS Code Extension** - RAGIX sidebar panel | 16h | Pending |
| **VS Code Commands** - Ask RAGIX about selection, file, project | 8h | Pending |

### Priority 10: Semantic Task Classification (v0.36) — Internationalization

**Problem:** Current task classification uses English keyword matching, which:
- Fails for non-English queries (French, German, Spanish, etc.)
- Causes false positives (e.g., "they are" matching "hey " pattern)
- Is brittle and requires constant pattern maintenance
- Does not scale to new languages or domains

| Task | Effort | Status |
|------|--------|--------|
| **LLM-Based Classification** - Use small/fast model for intent detection | 6h | Pending |
| **Intent Categories** - Define semantic categories independent of language | 4h | Pending |
| **Embedding Similarity** - Use sentence embeddings for query similarity | 8h | Pending |
| **Classification Cache** - Cache common query patterns to reduce latency | 3h | Pending |
| **Fallback Heuristics** - Keep keyword fallback for offline/fast mode | 2h | Pending |
| **Multi-Language Testing** - Test suite with FR/DE/ES/PT/IT/NL queries | 4h | Pending |

**Proposed Architecture:**

```
User Query (any language)
    │
    ├─► [FAST] Embedding similarity to known intents
    │       └─► If confidence > 0.85: return classification
    │
    ├─► [MEDIUM] LLM mini-classifier (qwen2.5:0.5b or similar)
    │       Prompt: "Classify intent: BYPASS | SIMPLE | MODERATE | COMPLEX"
    │       └─► Return classification with confidence
    │
    └─► [FALLBACK] Keyword heuristics (current system)
            └─► Only if LLM unavailable or timeout
```

**Intent Categories (Language-Independent):**

| Intent | Description | Examples (EN/FR/DE) |
|--------|-------------|---------------------|
| `GREETING` | Conversational opener | "hello" / "bonjour" / "hallo" |
| `CONCEPTUAL` | Theory/explanation request | "what is X" / "qu'est-ce que" / "was ist" |
| `FILE_SEARCH` | Find files by pattern | "find files" / "trouver fichiers" / "Dateien finden" |
| `CODE_SEARCH` | Search code content | "grep for" / "chercher dans" / "suchen nach" |
| `FILE_READ` | Read file contents | "show me" / "montre-moi" / "zeig mir" |
| `FILE_EDIT` | Modify file | "change X to Y" / "modifier" / "ändern" |
| `MULTI_STEP` | Complex operation | "refactor and test" / "refactoriser et tester" |

**Embedding Model Options:**
- `all-MiniLM-L6-v2` (384d, 22M params, multilingual support)
- `paraphrase-multilingual-MiniLM-L12-v2` (384d, 118M params, 50+ languages)
- Local Ollama embedding: `nomic-embed-text` or `mxbai-embed-large`

**Benefits:**
- Language-agnostic classification
- Semantic understanding vs brittle keywords
- Graceful degradation (embedding → LLM → keywords)
- Caching for common patterns reduces latency
- Extensible to new languages without code changes

### CLI Design Reference

```bash
# Like Claude Code / Codex
ragix                           # Start interactive REPL
ragix -c "find all TODO comments"  # Single prompt
ragix --resume                  # Resume last session
ragix --session abc123          # Resume specific session
ragix --model qwen2.5:7b        # Override model
ragix --profile safe            # Use safe profile

# Output: Claude-style colored layout (RAGIX blue theme)
```

### Architecture Notes

**Token Counting:**
- Ollama API returns `prompt_eval_count` and `eval_count`
- Store per-request and cumulative in session
- Display: `📊 Tokens: ↑1.2k ↓0.8k | Context: 4.2k/32k`

**Memory Compaction:**
- Use LLM to summarize older messages
- Keep recent N messages verbatim + summary of older
- Trigger: manual button or auto at 80% context

**File Conversion Pipeline:**
```
Drop file → Detect type → Convert if needed → Insert as context
           ├─ .txt/.md/.py → Direct insert
           ├─ .pdf → pdftotext → markdown
           ├─ .docx/.odt → pandoc → markdown
           └─ binary → Error message
```

---

## Session Completed (2025-12-03 - v0.30.0 Streaming Progress)

### Real-Time Progress Streaming & Robust JSON Handling

| Task | Status |
|------|--------|
| **Progress Callback Pipeline** - ReasoningGraph → GraphReasoningLoop → WebSocket | ✅ Done |
| **Graph Progress Events** - Classification, planning, step execution, reflection | ✅ Done |
| **Server Progress Setup** - Set callback before execution for real-time updates | ✅ Done |
| **Tool Traces Panel** - New icons for plan_ready, plan_step, step_complete, etc. | ✅ Done |
| **Robust JSON Parsing** - Handle unquoted keys, single quotes, trailing commas | ✅ Done |
| **JSON Action Execution** - Direct command execution from malformed LLM JSON | ✅ Done |
| **Output Filtering** - Filter raw JSON from RespondNode final response | ✅ Done |

**Key Files Modified:**
- `ragix_core/reasoning_graph.py` - `_emit_progress()`, enhanced `run()` loop, `RespondNode` filtering
- `ragix_core/reasoning.py` - `graph_progress_callback`, `_execute_step_wrapper` JSON handling
- `ragix_core/orchestrator.py` - `extract_json_object()` with JSON repair
- `ragix_web/server.py` - `trace_callback` setup in `execute_step()`
- `ragix_web/static/app.js` - Extended icon mapping, `handleProgressUpdate()` cases

**Progress Pipeline:**
```
ReasoningGraph._emit_progress(node_name, message, metadata)
    → graph_progress_callback(node, msg, meta)
        → GraphReasoningLoop._add_trace(trace_type, content, meta)
            → self._progress_callback(trace)  [set by server]
                → emit_progress() → progress_queue
                    → WebSocket send_progress()
                        → app.js handleProgressUpdate() / addSingleReasoningTrace()
```

**JSON Repair in `extract_json_object()`:**
- Unquoted keys: `{action: "bash"}` → `{"action": "bash"}`
- Single quotes: `{'action': 'bash'}` → `{"action": "bash"}`
- Mixed quotes: `{"action": "respond", message: "test"}` → fixed
- Trailing commas: `{"a": 1,}` → `{"a": 1}`

---

## Session Completed (2025-12-02 - v0.23.0)

### Unified Model Inheritance & Web UI Fixes

| Task | Status |
|------|--------|
| **Model Inheritance Hierarchy** - Session → Agent Config → Reasoning | ✅ Done |
| **Fixed Agent Config Router** - `/api/agents/config` reads from `active_sessions` | ✅ Done |
| **MINIMAL Mode Inheritance** - Planner/Worker/Verifier inherit session model | ✅ Done |
| **UI Consistency** - All panels show correct model | ✅ Done |
| **Session Auto-Creation** - Handle server restart gracefully | ✅ Done |
| **Settings Modal Sync** - Session ID consistent between Chat and Settings | ✅ Done |
| **Removed Redundant Settings** - Reasoning model selector now inherits | ✅ Done |
| **Version Bump** - 0.23.0 centralized in `ragix_core/version.py` | ✅ Done |

**Key Files Modified:**
- `ragix_web/routers/agents.py` - Major fix for model inheritance
- `ragix_web/server.py` - Session management and auto-creation
- `ragix_web/static/app.js` - Settings modal fixes
- `ragix_web/static/index.html` - UI consistency updates

**Model Inheritance Architecture:**
```
Session (Settings → Session tab)
    └── Agent Config (in MINIMAL mode, inherits session model)
            └── Reasoning (Planner/Worker/Verifier inherit from Agent Config)
```

---

## Next Steps (v0.30.0) — Reflective Reasoning Graph v3

**Reference:** See `PLAN_v0.30_REASONING.md` for full specification
**Based on:** Colleague review in `REVIEW_current_reasoning_towardsv0.30.md`

### Key Improvements vs v0.23 Plan

| Feature | v0.23 | v0.30 |
|---------|-------|-------|
| Complexity levels | SIMPLE, MODERATE, COMPLEX | + **BYPASS** (no tools, no plan) |
| Confidence | Not tracked | **Plan.confidence**, **State.confidence** |
| Tool schema | Ad-hoc | **ToolCall/ToolResult** unified |
| Module layout | ragix_core/reasoning*.py | **ragix_core/reasoning_v30/** versioned |
| Reflection | Limited | Strict budget + **3-bullet max** prompts |

### Priority 1: Core Reasoning Graph (~20h)

| Task | Effort | Status |
|------|--------|--------|
| **Create `reasoning_v30/types.py`** - TaskComplexity (BYPASS/SIMPLE/MODERATE/COMPLEX), ToolCall, ToolResult, Plan, ReasoningState | 4h | Pending |
| **Create `reasoning_v30/graph.py`** - BaseNode, ReasoningGraph orchestrator | 3h | Pending |
| **Create `reasoning_v30/nodes.py`** - ClassifyNode, DirectExecNode, PlanNode, ExecuteNode, ReflectNode, VerifyNode, RespondNode | 8h | Pending |
| **Wire confidence** - Track Plan.confidence → State.confidence | 2h | Pending |
| **Add BYPASS flow** - CLASSIFY → DIRECT_EXEC → RESPOND | 3h | Pending |

### Priority 2: Unified Tool Protocol (~8h)

| Task | Effort | Status |
|------|--------|--------|
| **Standardize tool schema** - All tools (rt-*, edit_file, ragix-ast) use ToolCall/ToolResult | 4h | Pending |
| **Deterministic output format** - Explicit error codes, max output size | 2h | Pending |
| **Dry-run preview** - `rt_edit --dry-run` for dev profile | 2h | Pending |

### Priority 3: Experience Corpus (~10h)

| Task | Effort | Status |
|------|--------|--------|
| **Create `reasoning_v30/experience.py`** - ExperienceCorpus, HybridExperienceCorpus | 4h | Pending |
| **Canonical layout** - `~/.ragix/experience/events.jsonl` + `.ragix/experience/events.jsonl` | 2h | Pending |
| **Per-session traces** - `{session_id}.jsonl` in traces folder | 2h | Pending |
| **TTL pruning** - 90 days global, 30 days project | 2h | Pending |

### Priority 4: Test Harness (~12h)

| Task | Effort | Status |
|------|--------|--------|
| **Create `tests/reasoning_v30/`** - Folder structure | 1h | Pending |
| **Scenario YAML format** - id, input, expected_patterns, must_run_commands, complexity | 2h | Pending |
| **Harness runner** - Execute scenarios, collect metrics | 4h | Pending |
| **Mock repo fixtures** - Test file structures | 2h | Pending |
| **Test cases** - file_search.yaml, code_analysis.yaml, bypass_question.yaml | 3h | Pending |

### Priority 5: Configuration & Profiles (~6h)

| Task | Effort | Status |
|------|--------|--------|
| **Update ragix.yaml** - `reasoning.strategy: graph_v30` section | 2h | Pending |
| **Agent profiles matrix** - safe/dev/sovereign with tools/models/reflection/memory | 2h | Pending |
| **Create `reasoning_v30/config.py`** - Config loader | 2h | Pending |

### Priority 6: LLM Prompts (~8h)

| Task | Effort | Status |
|------|--------|--------|
| **CLASSIFY prompt** - Output exactly BYPASS/SIMPLE/MODERATE/COMPLEX | 1h | Pending |
| **PLAN prompt** - JSON with objective, steps, validation, confidence | 2h | Pending |
| **REFLECT prompt** - 3-bullet max constraint for stability | 2h | Pending |
| **VERIFY prompt** - Check correctness, refine answer, output confidence | 1h | Pending |
| **DIRECT_EXEC prompt** - Conversational answer with confidence | 1h | Pending |
| **Prompt templates** - Jinja2 or f-string templates | 1h | Pending |

**Total Estimated Effort:** ~64 hours

### File Structure for v0.30

```
ragix_core/
├── reasoning.py                  # Legacy loop / adapter
├── reasoning_v30/
│   ├── __init__.py
│   ├── types.py                  # TaskComplexity, ToolCall, ToolResult, Plan, State
│   ├── graph.py                  # BaseNode, ReasoningGraph
│   ├── nodes.py                  # All node implementations
│   ├── experience.py             # ExperienceCorpus, HybridExperienceCorpus
│   └── config.py                 # Config loader from ragix.yaml

tests/reasoning_v30/
├── __init__.py
├── harness.py                    # Scenario runner
├── fixtures/
│   └── mock_repo/
├── scenarios/
│   ├── file_search.yaml
│   ├── code_analysis.yaml
│   └── bypass_question.yaml
└── test_reasoning_graph_v30.py

~/.ragix/experience/              # Global experience corpus
.ragix/experience/                # Project experience corpus
.ragix/reasoning_traces/          # Per-session traces
```

### Success Metrics (v0.30)

| Metric | Target |
|--------|--------|
| Plan success rate | >80% |
| Recovery rate (REFLECT) | >60% |
| Max reflections hit | <10% |
| BYPASS accuracy | >90% |
| Avg steps per task | <6 |

---

## Session Completed (2025-11-28 - v0.20.0)

### Report Generation & Documentation Coverage

| Task | Status |
|------|--------|
| **Report Engine** - Jinja2 templates for PDF/HTML reports | ✅ Done |
| **Executive Summary** - High-level metrics, risks, recommendations | ✅ Done |
| **Technical Audit** - Detailed code metrics and hotspots | ✅ Done |
| **Compliance Report** - Security findings and coverage | ✅ Done |
| **Treemap Visualization** - Package hierarchy by LOC/complexity | ✅ Done |
| **Sunburst Visualization** - Module structure drill-down | ✅ Done |
| **Chord Diagram** - Inter-module dependency visualization | ✅ Done |
| **Maven Integration** - POM parsing in reports | ✅ Done |
| **SonarQube Integration** - Quality gate data in reports | ✅ Done |
| **Documentation Coverage Fix** - Filter placeholder Javadocs | ✅ Done |
| **Separate Doc Metrics** - Class vs Method coverage (50/50 weighted) | ✅ Done |
| **Methods Count Fix** - Include class methods in total | ✅ Done |
| **Web UI Defensive JS** - Handle undefined API responses | ✅ Done |

**Implemented in:**
- `ragix_core/report_engine.py` - ReportEngine, generators, templates
- `ragix_core/ast_viz_advanced.py` - TreemapRenderer, SunburstRenderer, ChordRenderer
- `ragix_core/code_metrics.py` - total_methods, doc_coverage, class_doc_coverage
- `ragix_core/ast_java.py` - _get_javadoc() filters placeholders
- `ragix_unix/ast_cli.py` - New CLI commands (treemap, sunburst, chord, report)
- `ragix_web/server.py` - New API endpoints
- `ragix_web/static/index.html` - New cards for visualizations and reports

**New API Endpoints:**
```
GET /api/ast/treemap?path=...      # Treemap visualization
GET /api/ast/sunburst?path=...     # Sunburst visualization
GET /api/ast/chord?path=...        # Chord diagram
GET /api/ast/maven?path=...        # Maven analysis
GET /api/ast/sonar?url=...         # SonarQube integration
GET /api/ast/cycles?path=...       # Cycle detection
GET /api/reports/generate          # Generate reports
```

---

## TOP PRIORITY — Multi-Agent LLM Configuration (v0.11.0)

### Augmented Reasoning with Light Models (3B/7B)

| Task | Effort | Status |
|------|--------|--------|
| **AgentConfig class** - mode/model per agent | 4h | ✅ Done |
| **Auto-detect installed Ollama models** | 2h | ✅ Done |
| **UI toggle** - Minimal (3B) vs Strict (7B+) mode | 3h | ✅ Done |
| **Validate model size** against requirements | 1h | ✅ Done |
| **Granite 3B persona** - Worker/Verifier prompts | 2h | ✅ Done |
| **ragix.yaml agents section** - schema + loader | 2h | ✅ Done |
| **KnowledgeBase** - Pattern storage for 7B models | 4h | ✅ Done |

**Implemented in:**
- `ragix_core/agent_config.py` - Full AgentConfig, model detection, personas
- `ragix.yaml` - Agent configuration section (lines 139-159)

**Reference:** See `RAGIX_REASONING.md` §8 for full specification

**Agent Architecture:**
```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Planner  │───▶│  Worker  │───▶│ Verifier │
│ (≥7B/3B) │    │   (3B)   │    │   (3B)   │
└──────────┘    └──────────┘    └──────────┘
```

**Modes:**
- **Minimal** (default): All 3B — for 8GB VRAM / CPU
- **Strict**: Planner ≥7B, Worker/Verifier 3B — for 12GB+ VRAM

**Installed Models:**
| Model | Size | Recommended Role |
|-------|------|------------------|
| `granite3.1-moe:3b` | 2.0 GB | Worker, Verifier, Minimal Planner |
| `mistral:latest` | 4.4 GB | Strict Planner |
| `deepseek-r1:14b` | 9.0 GB | Advanced Planner |

---

## Session Completed (2025-11-28 Session 5)

### Knowledge Base & Web UI Consolidation

| Task | Status |
|------|--------|
| **KnowledgeBase system** - Pattern/rule storage for 7B models | ✅ Done |
| **Session Memory viewer** - View/delete/clear message history | ✅ Done |
| **User Context management** - System instructions like Claude/ChatGPT | ✅ Done |
| **Modular routers** - sessions, memory, context, agents, logs | ✅ Done |

**Implemented in:**
- `ragix_core/knowledge_base.py` - CommandPattern, ReasoningRule, KnowledgeBase
- `ragix_core/knowledge_rules.yaml` - Extensible YAML patterns
- `ragix_web/routers/` - Modular API routers
- `ragix_web/static/index.html` - Memory & Context tabs in Settings
- `ragix_web/static/style.css` - Memory/Context panel styles
- `ragix_web/server.py` - Router registration

**New API Endpoints:**
```
GET/DELETE /api/sessions/{session_id}/memory      # Session memory
GET/DELETE /api/sessions/{session_id}/memory/{i}  # Specific message
GET/POST/DELETE /api/sessions/{session_id}/context # User context
```

---

## Session Completed (2025-11-28 Session 4)

### Reasoning Loop & Web UI Improvements

| Task | Status |
|------|--------|
| **Direct conversational responses** (greeting, identity, help) | ✅ Done |
| **Improved Planner prompts** - line count filtering examples | ✅ Done |
| **Improved Worker prompts** - JSON templates for commands | ✅ Done |
| **CommandResult formatting** - clean output with newlines | ✅ Done |
| **JSON copy button** - optional JSON copy in details section | ✅ Done |
| **Collapsible details** - show/hide execution details | ✅ Done |

**Implemented in:**
- `ragix_core/reasoning.py` - Direct responses, improved prompts, formatting
- `ragix_web/static/app.js` - JSON copy button, details toggle
- `ragix_web/static/style.css` - JSON button styling

**Command Examples Added:**
```bash
# Files > N lines
find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | awk '$1 > 1000'
# Largest file
find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | sort -n | tail -1
```

---

## Session Completed (2025-11-27)

### HTMLRenderer Improvements (ast_viz.py)

| Task | Status |
|------|--------|
| Large Graph Mode (>5000 nodes) - auto-hide packages on load | ✅ Done |
| Package search with OR support (`pkg1\|pkg2\|pkg3`) | ✅ Done |
| Large Graph Mode informational message overlay | ✅ Done |
| Search match highlighting (red border on matches) | ✅ Done |

**Technical Details:**
- Large Graph Mode threshold: 5000 nodes
- Package search: case-insensitive, `|` for OR
- Tested on GRDF codebase (18,210 nodes)

---

## Immediate (v0.11.0) — ragix-web Consolidation

### Critical Priority

| Task | Effort | Status |
|------|--------|--------|
| Fix ragix-web server startup | 2h | Pending |
| Connect trace viewer to log_integrity.py | 4h | Pending |
| Integrate radial explorer into ragix-web | 2h | Pending |
| Add project selector to dashboard | 3h | Pending |
| Dashboard with quick stats from AST | 4h | Pending |

### High Priority

| Task | Effort | Status |
|------|--------|--------|
| Modular router structure (sessions, memory, context, agents, logs) | 6h | ✅ Done |
| WebSocket for live updates | 6h | Pending |
| Full MCP wrapper for rt-* tools | 4h | Pending |
| Add `rt-checksum` and `rt-metadata` tools | 3h | Pending |

---

## Phase 2 (v0.20.0) — Visualization Completion ✅ COMPLETE

### New Visualization Types

| Visualization | Purpose | Effort | Status |
|---------------|---------|--------|--------|
| **Treemap** | Package hierarchy by LOC/complexity | 8h | ✅ Done |
| **Sunburst** | Module structure drill-down | 8h | ✅ Done |
| **Chord Diagram** | Inter-module dependencies | 8h | ✅ Done |

### All Visualizations (Complete)

- [x] Force-directed graph with package clustering
- [x] Dependency Structure Matrix (DSM)
- [x] Radial ego-centric explorer
- [x] Treemap (package hierarchy)
- [x] Sunburst (module drill-down)
- [x] Chord diagram (inter-module deps)
- [x] D3.js interactive HTML

---

## Phase 3 (v0.20.0) — Report Generation ✅ COMPLETE

| Report Type | Format | Effort | Status |
|-------------|--------|--------|--------|
| Executive Summary | PDF/HTML | 8h | ✅ Done |
| Technical Audit | PDF/HTML | 8h | ✅ Done |
| Compliance Report | PDF | 6h | ✅ Done |
| Report Engine + Templates | — | 10h | ✅ Done |
| Maven Integration | — | 4h | ✅ Done |
| SonarQube Integration | — | 4h | ✅ Done |
| Documentation Coverage Fix | — | 4h | ✅ Done |

**Dependencies:** `weasyprint`, `jinja2` (added to pyproject.toml)

---

## Phase 4 (v0.12.0) — Git Integration

| Feature | Description | Effort | Status |
|---------|-------------|--------|--------|
| Complexity Evolution | Track CC over commits | 12h | Pending |
| Hotspot Emergence | Files becoming complex | 8h | Pending |
| Debt Accumulation | Technical debt timeline | 8h | Pending |
| Churn Analysis | Most-changed files | 6h | Pending |

---

## Future (v1.0+)

### Agent Reasoning Improvements (from Gemini 2.5 Review)
- [ ] **Automated Self-Correction Loop** — Verifier failure → Planner re-plan → Re-execute (High priority)
- [ ] **Dynamic Tool Selection** — Agent decides which ragix-* tool to use (High priority)
- [ ] **Structured Episodic Memory** — Store successful workflows for reference (Medium priority)
- [ ] **Confidence Scoring** — Agent outputs confidence (1-10), pauses on low (Medium priority)
- [x] Autonomous multi-step reasoning with self-correction (Planner/Worker/Verifier loop)
- [x] Memory and context persistence across sessions (EpisodicMemory in reasoning.py)
- [ ] Agent specialization profiles (security, performance, refactoring)
- [ ] Inter-agent communication protocol

### Tool Usability Improvements (from Gemini 2.5 Review)
- [ ] **Unified CLI Entry Point** — `ragix ast search` instead of `ragix-ast search` (High priority)
- [ ] **Interactive ragix-ast Mode** — `--interactive` REPL to avoid re-parsing (Medium priority)
- [ ] **Cross-Panel Context** — Click node in graph → show file in code panel (Medium priority)
- [ ] **`ragix config view`** — Show resolved config with sources (High priority)
- [ ] **Visual Workflow Builder** — Drag-and-drop workflow creation (Low priority)

### Tool Enhancements
- [ ] WASM-compiled tools for browser execution
- [ ] AST-aware search (tree-sitter integration)
- [ ] Pyodide sandbox for browser-in-agent (not urgent)

### Integrations
- [ ] VS Code Extension
- [ ] GitHub Actions integration
- [ ] GitLab CI/CD integration
- [ ] Jupyter notebook support

### Performance
- [ ] GPU acceleration for embeddings (CUDA/MPS)
- [ ] Distributed index sharding
- [ ] Response streaming for large outputs
- [ ] Persistent connection pooling for Ollama

### Security & Compliance
- [ ] Audit log export (JSON, CSV)
- [ ] Role-based access control for tools
- [ ] Secrets scanning integration
- [ ] SBOM generation for analyzed repos

---

## Completed (v0.10.1 → v0.20.0)

### v0.20.0 (Current Release)
- [x] Report Generation (Executive Summary, Technical Audit, Compliance)
- [x] Advanced Visualizations (Treemap, Sunburst, Chord Diagram)
- [x] Maven integration in reports
- [x] SonarQube integration in reports
- [x] Documentation coverage fix (filters placeholder Javadocs)
- [x] Separate class/method doc coverage metrics
- [x] Methods count fix (includes class methods)
- [x] Web UI defensive JavaScript (handles undefined responses)
- [x] New API endpoints (treemap, sunburst, chord, maven, sonar, cycles, reports)

### v0.11.0
- [x] AgentConfig class with mode/model per agent
- [x] Auto-detect installed Ollama models
- [x] Model size validation against requirements
- [x] Agent persona prompts (Planner/Worker/Verifier)
- [x] ragix.yaml agents section with schema
- [x] EpisodicMemory for session context persistence
- [x] ReasoningLoop (Planner/Worker/Verifier orchestration)
- [x] Direct conversational responses (greetings, identity, help)
- [x] Improved planner/worker prompts with Unix command templates
- [x] CommandResult formatting with proper newlines
- [x] Web UI: JSON copy button, collapsible details
- [x] Web UI: Reasoning traces display
- [x] Web UI: Agent mode selector (Minimal/Strict/Custom)
- [x] KnowledgeBase for 7B model reasoning improvement
- [x] Session memory viewer with selective delete/clear
- [x] User context management (system instructions, preferences)
- [x] Modular router structure (ragix_web/routers/)

### v0.10.1
- [x] Multi-language AST (Python + Java via javalang)
- [x] Dependency graph with cycle detection
- [x] AST query language (pattern-based search)
- [x] Professional code metrics (cyclomatic complexity, technical debt)
- [x] Maven POM parsing
- [x] SonarQube/SonarCloud client
- [x] Enhanced HTML visualization (package clustering, edge bundling)
- [x] Dependency Structure Matrix (DSM) with heatmap
- [x] Radial ego-centric explorer
- [x] Standalone radial server
- [x] 8 AST API endpoints in ragix-web
- [x] CLI: `ragix-ast` with 12 subcommands
- [x] Plugin system with tool/workflow types
- [x] WASP tools (18 deterministic tools)
- [x] SWE chunked workflows
- [x] Unified configuration (ragix.yaml)
- [x] Log integrity (SHA256 hash chain)
- [x] Large Graph Mode for HTMLRenderer (>5000 nodes threshold)
- [x] Package search with OR support in HTMLRenderer

---

## Quick Reference

```bash
# Current CLI capabilities
ragix-ast scan ./project          # Full AST analysis
ragix-ast metrics ./project       # Code quality metrics
ragix-ast graph ./project -f html # Force-directed graph
ragix-ast matrix ./project        # DSM visualization
ragix-ast radial ./project        # Radial explorer

# Live radial server
python -m ragix_unix.radial_server --path ./project --port 8090

# ragix-web (needs consolidation)
ragix-web --port 8080
```

---

*See `ACTION_PLAN.md` for implementation timeline and architecture.*
