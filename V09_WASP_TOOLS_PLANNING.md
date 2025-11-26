# RAGIX v0.9 Planning: WASP Tools & Browser Runtime

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
**Status:** RELEASED (2025-11-26)
**Reference:** `V08_WASP_PLANNING.md` for foundation, `WASM.md` for specifications

---

## Executive Summary

v0.9 delivers the **WASP tools and browser runtime** building on v0.8's sandbox abstraction:

| Feature | Description | Priority |
|---------|-------------|----------|
| WASP Registry CLI | `ragix-wasp` command for tool management | HIGH |
| JSON Protocol Extension | `wasp_task` action for WASM execution | HIGH |
| Priority WASM Tools | validate, mdparse, search (Python-based) | HIGH |
| Browser WASM Runtime | Client-side execution | MEDIUM |
| Testing Suite | Comprehensive WASP tests | HIGH |
| Documentation | WASP guides and migration docs | MEDIUM |

---

## What v0.8 Delivered (Foundation)

```
ragix_core/sandbox_base.py     - BaseSandbox protocol, HybridSandbox
ragix_core/wasm_sandbox.py     - WasmSandbox, WasmToolRegistry
ragix_core/plugin_system.py    - PluginManager, trust levels
ragix_core/swe_workflows.py    - ChunkedWorkflow, checkpoints
```

---

## Task Breakdown

### Task 4.3: WASP Registry CLI

**Goal:** Complete tool registry with CLI management.

**Components:**

1. **WASP CLI** (`ragix_unix/wasp_cli.py` ~200 lines)
   ```bash
   ragix-wasp list                    # List installed tools
   ragix-wasp info <tool>             # Show tool details
   ragix-wasp install <path>          # Install tool from manifest
   ragix-wasp run <tool> [args]       # Run tool directly
   ragix-wasp validate <manifest>     # Validate tool manifest
   ```

2. **Enhanced Registry** (update `wasm_sandbox.py` ~100 lines)
   - Tool versioning
   - Dependency checking
   - Capability validation

---

### Task 4.4: JSON Protocol Extension

**Goal:** Add `wasp_task` action to orchestrator.

**Components:**

1. **Action Handler** (`ragix_core/orchestrator.py` ~100 lines)
   ```python
   elif kind == "wasp_task":
       tool = action["tool"]
       inputs = action.get("inputs", {})
       result = self.wasp_sandbox.execute_tool(tool, inputs)
       return None, result
   ```

2. **Schema Update** (`ragix_core/orchestrator.py` ~50 lines)
   ```python
   WASP_TASK_SCHEMA = {
       "action": "wasp_task",
       "tool": str,      # Required: tool name
       "inputs": dict,   # Optional: input parameters
   }
   ```

3. **Agent Prompt Updates** (~50 lines)
   - Add `wasp_task` to available actions
   - Guidelines for when to use WASP vs bash

---

### Task 4.5: Priority WASP Tools

**Goal:** Implement deterministic tools as Python modules (WASM later).

**Strategy:** Start with Python implementations that can be wrapped as WASM later:

1. **JSON/YAML Validator** (`wasp_tools/validate.py` ~200 lines)
   ```python
   def validate_json(content: str, schema: Optional[str] = None) -> dict
   def validate_yaml(content: str, schema: Optional[str] = None) -> dict
   def format_json(content: str, indent: int = 2) -> dict
   ```

2. **Markdown Parser** (`wasp_tools/mdparse.py` ~250 lines)
   ```python
   def parse_markdown(content: str) -> dict  # AST structure
   def extract_headers(content: str) -> list
   def extract_code_blocks(content: str) -> list
   def renumber_sections(content: str) -> str
   ```

3. **Text Search** (`wasp_tools/search.py` ~200 lines)
   ```python
   def search_pattern(pattern: str, content: str, options: dict) -> dict
   def search_files(pattern: str, paths: list, options: dict) -> dict
   ```

4. **Tool Manifest Format** (`wasp_tools/manifest.yaml`)
   ```yaml
   tools:
     - name: validate_json
       entry: validate:validate_json
       description: Validate JSON content
       parameters:
         - name: content
           type: string
           required: true
         - name: schema
           type: string
           required: false
   ```

---

### Task 4.6: Browser WASM Runtime

**Goal:** Run WASP tools client-side in browser.

**Components:**

1. **Browser Runtime** (`ragix_web/static/js/wasp_runtime.js` ~300 lines)
   ```javascript
   class WaspRuntime {
       constructor() { this.tools = new Map(); }
       async loadTool(name, config) { ... }
       async execute(toolName, inputs) { ... }
   }
   ```

2. **Virtual Filesystem** (`ragix_web/static/js/virtual_fs.js` ~200 lines)
   ```javascript
   class VirtualFS {
       constructor() { this.files = new Map(); }
       readFile(path) { ... }
       writeFile(path, content) { ... }
       listDir(path) { ... }
   }
   ```

3. **UI Integration** (update `ragix_web/static/js/app.js` ~100 lines)
   - WASP tool selector
   - Client-side execution toggle
   - Result display

---

### Task 4.7: Testing Suite

**Goal:** Comprehensive tests for WASP.

**Components:**

1. **Unit Tests** (`tests/test_wasp_tools.py` ~400 lines)
   - validate.py tests
   - mdparse.py tests
   - search.py tests

2. **Integration Tests** (`tests/test_wasp_integration.py` ~300 lines)
   - Protocol tests (wasp_task action)
   - End-to-end execution
   - Hybrid sandbox fallback

3. **CLI Tests** (`tests/test_wasp_cli.py` ~200 lines)
   - ragix-wasp commands

---

### Task 4.8: Documentation

**Goal:** Complete WASP documentation.

**Components:**

1. **WASP Guide** (`docs/WASP_GUIDE.md` ~400 lines)
   - Architecture overview
   - Tool development guide
   - Usage examples

2. **Tool Reference** (`docs/WASP_TOOLS.md` ~300 lines)
   - Built-in tools reference
   - Parameter documentation

3. **Migration Guide** (update `CHANGELOG.md`)
   - v0.8 → v0.9 changes
   - New features

---

## File Structure (v0.9)

```
RAGIX/
├── ragix_core/
│   ├── orchestrator.py          UPDATED - wasp_task action
│   └── wasm_sandbox.py          UPDATED - enhanced registry
│
├── ragix_unix/
│   └── wasp_cli.py              NEW - WASP CLI
│
├── wasp_tools/                   NEW - WASP tool implementations
│   ├── __init__.py
│   ├── manifest.yaml            Tool registry manifest
│   ├── validate.py              JSON/YAML validation
│   ├── mdparse.py               Markdown parsing
│   └── search.py                Text search
│
├── ragix_web/static/js/
│   ├── wasp_runtime.js          NEW - Browser runtime
│   └── virtual_fs.js            NEW - Virtual filesystem
│
├── tests/
│   ├── test_wasp_tools.py       NEW
│   ├── test_wasp_integration.py NEW
│   └── test_wasp_cli.py         NEW
│
└── docs/
    ├── WASP_GUIDE.md            NEW
    └── WASP_TOOLS.md            NEW
```

---

## Estimated Lines of Code

| Task | Component | Lines |
|------|-----------|-------|
| 4.3 | WASP CLI | ~300 |
| 4.4 | Protocol Extension | ~200 |
| 4.5 | WASP Tools | ~650 |
| 4.6 | Browser Runtime | ~600 |
| 4.7 | Testing | ~900 |
| 4.8 | Documentation | ~700 |
| **Total** | | **~3,350** |

---

## Implementation Order

1. **Phase 1: Core Tools** (Tasks 4.5, 4.3)
   - Implement Python-based WASP tools
   - Create WASP CLI
   - Test tools standalone

2. **Phase 2: Protocol Integration** (Task 4.4)
   - Add wasp_task action
   - Update agent prompts
   - Integration testing

3. **Phase 3: Browser** (Task 4.6)
   - Browser runtime
   - Virtual filesystem
   - UI integration

4. **Phase 4: Quality** (Tasks 4.7, 4.8)
   - Complete test suite
   - Documentation
   - v0.9 release

---

## Dependencies

No new Python dependencies required (uses stdlib).

Browser runtime uses native WebAssembly APIs.

---
