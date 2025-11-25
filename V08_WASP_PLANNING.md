# RAGIX v0.8 Planning: WASP Integration

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
**Status:** PLANNING - Scheduled after v0.7 completion
**Reference:** `WASM.md` for full specifications and rationale

---

## Executive Summary

**WASP** = WebAssembly Agentic System Protocol / Pipelines

v0.8 introduces WASM as a **secure, portable execution layer** for RAGIX:
- Deterministic tool execution (no shell unpredictability)
- Stronger sandboxing (WASI capabilities model)
- Cross-platform portability (Linux/macOS/Windows/browser)
- Full auditability (reproducible, logged execution)

**Development Strategy:** Server-side and browser-side in parallel.

---

## Context: v0.7 → v0.8

### v0.7 Delivers (prerequisite for v0.8)
- LLM integration with tool execution loop
- Tool registry system
- Enhanced retrieval (BM25 + vector)
- Production-ready testing and monitoring

### v0.8 Adds
- WASM-based tool execution alongside shell
- Browser-native RAGIX (full WASM stack)
- Deterministic micro-tools library
- Unified protocol for server + browser

---

## Architecture Overview

### Current (v0.6/v0.7)
```
LLM → RAGIX Orchestrator → ShellSandbox → bash/grep/git
                                ↓
                          CommandResult
```

### Target (v0.8)
```
LLM → RAGIX Orchestrator → ┬→ ShellSandbox (legacy)
                           │      ↓
                           │  CommandResult
                           │
                           └→ WasmSandbox (new)
                                  ↓
                             .wasm modules
                                  ↓
                             WasmResult
```

### Browser Architecture (v0.8)
```
Browser
   │
   ├→ ragix_web/static/ (existing)
   │      ↓
   │   WebSocket → Python server → ShellSandbox
   │
   └→ ragix_wasm/ (new)
          ↓
      JS + WASI runtime
          ↓
      .wasm modules (client-side)
          ↓
      Local execution (no server for tools)
```

---

## Task Breakdown

### Task 4.1: Sandbox Abstraction Layer ⏸️

**Goal:** Refactor sandbox into pluggable interface for WASM integration.

**Components:**

1. **Base Sandbox Protocol** (`ragix_core/sandbox_base.py` ~100 lines)
   ```python
   class BaseSandbox(Protocol):
       def run(self, cmd: str, timeout: int = 60) -> CommandResult:
           ...

       def supports_command(self, cmd: str) -> bool:
           ...
   ```

2. **Shell Sandbox Refactor** (update `tools_shell.py` ~50 lines)
   - Inherit from `BaseSandbox`
   - No functional changes

3. **Sandbox Factory** (`ragix_core/sandbox_factory.py` ~80 lines)
   ```python
   def create_sandbox(
       sandbox_type: str = "shell",  # "shell" | "wasm" | "hybrid"
       root: Path,
       config: Optional[SandboxConfig] = None
   ) -> BaseSandbox:
       ...
   ```

**Estimated:** ~230 lines

---

### Task 4.2: WasmSandbox Implementation ⏸️

**Goal:** Create WASM-based sandbox using wasmtime Python bindings.

**Components:**

1. **WASM Runtime Wrapper** (`ragix_core/wasm_runtime.py` ~300 lines)
   ```python
   class WasmRuntime:
       def __init__(self, engine_config: Optional[dict] = None):
           self.engine = wasmtime.Engine()
           self.linker = wasmtime.Linker(self.engine)
           self._setup_wasi()

       def load_module(self, path: Path) -> WasmModule:
           ...

       def execute(self, module: WasmModule, func: str, args: dict) -> WasmResult:
           ...
   ```

2. **WASM Sandbox** (`ragix_core/wasm_sandbox.py` ~250 lines)
   ```python
   class WasmSandbox(BaseSandbox):
       def __init__(self, root: Path, modules_dir: Path):
           self.runtime = WasmRuntime()
           self.modules = self._load_modules(modules_dir)

       def run(self, cmd: str, timeout: int = 60) -> CommandResult:
           tool, args = self._parse_command(cmd)
           if tool in self.modules:
               return self._run_wasm(tool, args)
           return CommandResult(cmd, self.root, "", f"Unknown tool: {tool}", 1)
   ```

3. **Hybrid Sandbox** (`ragix_core/hybrid_sandbox.py` ~150 lines)
   - Routes commands to WASM or shell based on availability
   - Transparent fallback

**Estimated:** ~700 lines

**Dependencies:**
```toml
[project.optional-dependencies]
wasm = ["wasmtime>=14.0.0"]
```

---

### Task 4.3: WASP Tool Registry ⏸️

**Goal:** Registry for .wasm modules with metadata and capabilities.

**Components:**

1. **Tool Manifest Schema** (`ragix_core/wasp_manifest.py` ~100 lines)
   ```python
   @dataclass
   class WaspToolManifest:
       name: str
       version: str
       description: str
       wasm_path: str
       entry_func: str = "main"
       capabilities: List[str] = field(default_factory=list)  # ["fs:read", "fs:write"]
       input_schema: Optional[dict] = None
       output_schema: Optional[dict] = None
   ```

2. **Tool Registry** (`ragix_core/wasp_registry.py` ~200 lines)
   ```python
   class WaspRegistry:
       def __init__(self, registry_dir: Path):
           self.tools: Dict[str, WaspToolManifest] = {}
           self._scan_registry(registry_dir)

       def get_tool(self, name: str) -> Optional[WaspToolManifest]:
           ...

       def list_tools(self) -> List[str]:
           ...

       def install_tool(self, wasm_path: Path, manifest: WaspToolManifest):
           ...
   ```

3. **Registry CLI** (`ragix_unix/wasp_cli.py` ~150 lines)
   ```bash
   ragix-wasp list                    # List installed tools
   ragix-wasp info <tool>             # Show tool details
   ragix-wasp install <path>          # Install .wasm tool
   ragix-wasp run <tool> [args]       # Run tool directly
   ```

**Estimated:** ~450 lines

---

### Task 4.4: JSON Protocol Extension ⏸️

**Goal:** Add `wasp_task` action to JSON protocol.

**Components:**

1. **Action Handler** (update orchestrator ~100 lines)
   ```python
   elif kind == "wasp_task":
       module = action["module"]
       func = action.get("func", "run")
       inputs = action.get("inputs", {})
       result = self.wasp_sandbox.execute(module, func, inputs)
       return None, result
   ```

2. **Agent Prompt Updates** (update prompts ~50 lines)
   - Add `wasp_task` to available actions
   - Guidelines for when to use WASM vs bash

3. **Schema Validation** (update validation ~50 lines)
   ```python
   WASP_TASK_SCHEMA = {
       "action": "wasp_task",
       "module": str,  # Required: tool name or path
       "func": str,    # Optional: function to call (default: "run")
       "inputs": dict  # Optional: input parameters
   }
   ```

**Estimated:** ~200 lines

---

### Task 4.5: Priority WASM Tools ⏸️

**Goal:** Implement first three WASM tools for RAGIX.

**Tools (in priority order):**

1. **JSON/YAML Validator** (`wasp_tools/validate.wasm` ~500 lines Rust)
   ```rust
   // Validate JSON/YAML against schema
   #[wasm_bindgen]
   pub fn validate_json(input: &str, schema: &str) -> ValidationResult { ... }

   #[wasm_bindgen]
   pub fn validate_yaml(input: &str, schema: &str) -> ValidationResult { ... }
   ```

2. **Markdown Parser** (`wasp_tools/mdparse.wasm` ~600 lines Rust)
   ```rust
   // Parse Markdown to structured AST
   #[wasm_bindgen]
   pub fn parse_markdown(input: &str) -> MarkdownAST { ... }

   #[wasm_bindgen]
   pub fn extract_headers(input: &str) -> Vec<Header> { ... }

   #[wasm_bindgen]
   pub fn renumber_sections(input: &str) -> String { ... }
   ```

3. **ripgrep.wasm** (~existing, ~100 lines wrapper)
   - Use existing ripgrep WASM build
   - Create wrapper for RAGIX integration
   ```rust
   #[wasm_bindgen]
   pub fn search(pattern: &str, path: &str, options: &SearchOptions) -> SearchResults { ... }
   ```

**Build System** (`wasp_tools/Cargo.toml`, build scripts ~200 lines)
```toml
[package]
name = "ragix-wasp-tools"
version = "0.8.0"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
pulldown-cmark = "0.9"  # Markdown parsing
```

**Estimated:** ~1,400 lines (Rust + build config)

---

### Task 4.6: Browser WASM Runtime ⏸️

**Goal:** Run RAGIX tools client-side in browser.

**Components:**

1. **WASI Runtime** (`ragix_web/static/wasm_runtime.js` ~400 lines)
   ```javascript
   class WasmRuntime {
       constructor() {
           this.modules = new Map();
           this.fs = new VirtualFS();
       }

       async loadModule(name, wasmUrl) {
           const response = await fetch(wasmUrl);
           const bytes = await response.arrayBuffer();
           const module = await WebAssembly.compile(bytes);
           // Setup WASI imports
           const instance = await WebAssembly.instantiate(module, this.wasiImports());
           this.modules.set(name, instance);
       }

       async execute(name, func, args) {
           const instance = this.modules.get(name);
           return instance.exports[func](args);
       }
   }
   ```

2. **Virtual Filesystem** (`ragix_web/static/virtual_fs.js` ~300 lines)
   ```javascript
   class VirtualFS {
       constructor() {
           this.files = new Map();
       }

       // File System Access API integration
       async loadFromLocal(dirHandle) { ... }

       // In-memory operations
       readFile(path) { ... }
       writeFile(path, content) { ... }

       // Export results
       async exportZip() { ... }
   }
   ```

3. **Browser Tool Integration** (`ragix_web/static/browser_tools.js` ~250 lines)
   ```javascript
   class BrowserTools {
       constructor(runtime, fs) {
           this.runtime = runtime;
           this.fs = fs;
       }

       // Map JSON actions to WASM calls
       async executeAction(action) {
           if (action.action === 'wasp_task') {
               return this.runtime.execute(action.module, action.func, action.inputs);
           }
           // ... other actions
       }
   }
   ```

4. **UI Integration** (update `app.js` ~150 lines)
   - Toggle between server and browser execution
   - Status indicator for WASM availability
   - File loading from local filesystem

**Estimated:** ~1,100 lines

---

### Task 4.7: Testing and Validation ⏸️

**Goal:** Comprehensive tests for WASM integration.

**Components:**

1. **Unit Tests** (`tests/test_wasm/` ~800 lines)
   - WasmRuntime tests
   - WasmSandbox tests
   - Tool registry tests
   - JSON protocol tests

2. **Integration Tests** (`tests/integration/test_wasp.py` ~500 lines)
   - End-to-end WASM tool execution
   - Hybrid sandbox fallback
   - Browser simulation tests

3. **Tool Tests** (`wasp_tools/tests/` ~400 lines)
   - JSON/YAML validator tests
   - Markdown parser tests
   - ripgrep.wasm tests

4. **Performance Benchmarks** (`tests/benchmarks/wasm_bench.py` ~300 lines)
   - WASM vs shell execution time
   - Memory usage comparison
   - Browser performance metrics

**Estimated:** ~2,000 lines

---

### Task 4.8: Documentation ⏸️

**Goal:** Complete WASP documentation.

**Components:**

1. **WASP Guide** (`docs/WASP_GUIDE.md` ~600 lines)
   - Architecture overview
   - Tool development guide
   - Deployment options

2. **Tool Development** (`docs/WASP_TOOL_DEV.md` ~400 lines)
   - Rust setup for WASM
   - Manifest format
   - Testing tools

3. **Browser Integration** (`docs/BROWSER_RAGIX.md` ~400 lines)
   - Setup guide
   - Capabilities and limitations
   - Offline usage

4. **Migration Guide** (`docs/MIGRATION_V07_V08.md` ~200 lines)
   - Breaking changes
   - Upgrade steps
   - Compatibility notes

**Estimated:** ~1,600 lines

---

## Total v0.8 Estimates

| Task | Description | Lines | Priority |
|------|-------------|-------|----------|
| 4.1 | Sandbox Abstraction | ~230 | CRITICAL |
| 4.2 | WasmSandbox | ~700 | CRITICAL |
| 4.3 | WASP Registry | ~450 | HIGH |
| 4.4 | JSON Protocol | ~200 | HIGH |
| 4.5 | WASM Tools | ~1,400 | HIGH |
| 4.6 | Browser Runtime | ~1,100 | MEDIUM |
| 4.7 | Testing | ~2,000 | HIGH |
| 4.8 | Documentation | ~1,600 | MEDIUM |
| **TOTAL** | | **~7,680** | |

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
1. Task 4.1: Sandbox Abstraction
2. Task 4.2: WasmSandbox (server-side)
3. Basic wasmtime integration tests

### Phase 2: Tools & Protocol (Weeks 3-4)
4. Task 4.3: WASP Registry
5. Task 4.4: JSON Protocol Extension
6. Task 4.5: First WASM tool (JSON/YAML validator)

### Phase 3: Browser & More Tools (Weeks 5-6)
7. Task 4.6: Browser WASM Runtime (parallel with server)
8. Task 4.5 continued: Markdown parser, ripgrep.wasm

### Phase 4: Quality & Launch (Weeks 7-8)
9. Task 4.7: Testing Suite
10. Task 4.8: Documentation
11. v0.8 release

---

## Dependencies

### Python (server-side)
```toml
[project.optional-dependencies]
wasm = ["wasmtime>=14.0.0"]
```

### Rust (tool development)
```toml
# wasp_tools/Cargo.toml
[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
pulldown-cmark = "0.9"
```

### JavaScript (browser)
- No npm dependencies (use browser-native WebAssembly)
- Optional: File System Access API polyfill

---

## Risk Assessment

### High Risk
1. **WASM tool compilation** - Rust-to-WASM toolchain complexity
   - Mitigation: Start with simple tools, use existing ripgrep.wasm

2. **Browser compatibility** - WASI support varies
   - Mitigation: Feature detection, graceful fallback to server

### Medium Risk
3. **Performance** - WASM may be slower for some operations
   - Mitigation: Benchmark early, keep shell fallback

4. **Tool interop** - Different calling conventions
   - Mitigation: Unified manifest format, strict schemas

### Low Risk
5. **wasmtime stability** - Well-maintained project
6. **Documentation** - WASM ecosystem well-documented

---

## Success Criteria

### Functional
- ✅ WasmSandbox executes all priority tools
- ✅ Browser RAGIX runs without Python server (for tools)
- ✅ Hybrid sandbox seamlessly falls back
- ✅ JSON protocol supports `wasp_task` action

### Quality
- ✅ All WASM tools have >90% test coverage
- ✅ Browser works on Chrome, Firefox, Safari
- ✅ Performance within 2x of native shell

### Security
- ✅ WASI capabilities properly restricted
- ✅ No filesystem access outside sandbox
- ✅ All executions logged and auditable

---

## File Structure (v0.8)

```
RAGIX/
├── ragix_core/
│   ├── sandbox_base.py         NEW - Base protocol
│   ├── sandbox_factory.py      NEW - Factory function
│   ├── wasm_runtime.py         NEW - wasmtime wrapper
│   ├── wasm_sandbox.py         NEW - WasmSandbox
│   ├── hybrid_sandbox.py       NEW - Hybrid routing
│   ├── wasp_manifest.py        NEW - Tool manifest
│   └── wasp_registry.py        NEW - Tool registry
│
├── ragix_unix/
│   └── wasp_cli.py             NEW - WASP CLI
│
├── ragix_web/static/
│   ├── wasm_runtime.js         NEW - Browser WASM
│   ├── virtual_fs.js           NEW - Virtual filesystem
│   └── browser_tools.js        NEW - Tool integration
│
├── wasp_tools/                  NEW - WASM tool sources
│   ├── Cargo.toml
│   ├── src/
│   │   ├── validate.rs         JSON/YAML validator
│   │   ├── mdparse.rs          Markdown parser
│   │   └── rg_wrapper.rs       ripgrep wrapper
│   └── tests/
│
├── .ragix_modules/wasp/         Runtime tool storage
│   ├── validate.wasm
│   ├── mdparse.wasm
│   └── rg.wasm
│
└── docs/
    ├── WASP_GUIDE.md           NEW
    ├── WASP_TOOL_DEV.md        NEW
    ├── BROWSER_RAGIX.md        NEW
    └── MIGRATION_V07_V08.md    NEW
```

---

## Quick Reference: WASM Tools Priority

| Tool | Purpose | Complexity | Value |
|------|---------|------------|-------|
| JSON/YAML Validator | Schema validation | Low | High |
| Markdown Parser | Doc structure | Medium | High |
| ripgrep.wasm | Code search | Low (existing) | Very High |
| tree-sitter.wasm | AST parsing | High | Medium |
| SFPPy-lite.wasm | Scientific compute | Very High | Specialized |

---

## Relationship to WASM.md

This planning document implements the vision outlined in `WASM.md`:

| WASM.md Section | v0.8 Task |
|-----------------|-----------|
| §1 WASM as safer shell | Task 4.2 WasmSandbox |
| §2 Browser front-end | Task 4.6 Browser Runtime |
| §3 MCP + WASM | Task 4.4 Protocol Extension |
| §4 Minimal path | Tasks 4.1-4.2 (abstraction + sandbox) |
| §5 WASM benefits | All tasks |
| §6 WASP concept | Tasks 4.3-4.5 (registry + tools) |

---

**v0.8 Status:** PLANNED - Ready after v0.7 completion

**Next Action:** Complete v0.7, then start Task 4.1 (Sandbox Abstraction)
