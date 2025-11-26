"""
WASM Sandbox - WebAssembly-based secure tool execution

Provides sandboxed execution using WebAssembly:
- WASI-compliant tool execution
- Capability-based security model
- Cross-platform deterministic execution

Requires: wasmtime>=14.0.0 (optional dependency)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .sandbox_base import (
    BaseSandbox,
    ExecutionResult,
    SandboxCapability,
    SandboxConfig,
    SandboxType,
)

logger = logging.getLogger(__name__)

# Check for wasmtime availability
WASMTIME_AVAILABLE = False
try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    logger.debug("wasmtime not installed, WASM sandbox unavailable")


@dataclass
class WasmToolManifest:
    """
    Manifest for a WASM tool module.

    Describes the tool's capabilities and interface.
    """
    name: str
    version: str
    description: str
    wasm_path: Path
    entry_func: str = "main"
    capabilities: Set[SandboxCapability] = field(default_factory=set)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_path: Path) -> "WasmToolManifest":
        """Load manifest from dictionary."""
        caps = set()
        for cap_str in data.get("capabilities", []):
            try:
                caps.add(SandboxCapability(cap_str))
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")

        wasm_path = Path(data["wasm_path"])
        if not wasm_path.is_absolute():
            wasm_path = base_path / wasm_path

        return cls(
            name=data["name"],
            version=data.get("version", "0.0.1"),
            description=data.get("description", ""),
            wasm_path=wasm_path,
            entry_func=data.get("entry_func", "main"),
            capabilities=caps,
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
        )


@dataclass
class WasmResult:
    """Result from WASM module execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: int = 0


class WasmModule:
    """
    Wrapper for a loaded WASM module.

    Handles module lifecycle and function calls.
    """

    def __init__(
        self,
        manifest: WasmToolManifest,
        store: "wasmtime.Store",
        instance: "wasmtime.Instance",
    ):
        """
        Initialize WASM module wrapper.

        Args:
            manifest: Tool manifest
            store: Wasmtime store
            instance: Instantiated module
        """
        self.manifest = manifest
        self._store = store
        self._instance = instance
        self._exports = instance.exports(store)

    def call(self, func_name: str, *args) -> Any:
        """
        Call a function in the module.

        Args:
            func_name: Function name to call
            *args: Function arguments

        Returns:
            Function result
        """
        func = getattr(self._exports, func_name, None)
        if func is None:
            raise ValueError(f"Function not found: {func_name}")

        return func(self._store, *args)

    @property
    def memory(self) -> Optional["wasmtime.Memory"]:
        """Get module memory if exported."""
        return getattr(self._exports, "memory", None)

    def read_string(self, ptr: int, length: int) -> str:
        """Read string from module memory."""
        if not self.memory:
            raise RuntimeError("Module has no memory export")

        data = self.memory.data_ptr(self._store)[ptr:ptr + length]
        return bytes(data).decode('utf-8')

    def write_string(self, ptr: int, data: str) -> int:
        """Write string to module memory."""
        if not self.memory:
            raise RuntimeError("Module has no memory export")

        encoded = data.encode('utf-8')
        mem = self.memory.data_ptr(self._store)
        mem[ptr:ptr + len(encoded)] = encoded
        return len(encoded)


class WasmRuntime:
    """
    WASM runtime environment using wasmtime.

    Manages engine, linker, and module instances.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize WASM runtime.

        Args:
            config: Sandbox configuration
        """
        if not WASMTIME_AVAILABLE:
            raise ImportError("wasmtime is required for WASM sandbox")

        self.config = config or SandboxConfig(sandbox_type=SandboxType.WASM)

        # Create engine
        engine_config = wasmtime.Config()
        engine_config.wasm_multi_memory = True
        self._engine = wasmtime.Engine(engine_config)

        # Create linker with WASI
        self._linker = wasmtime.Linker(self._engine)

        # Setup WASI imports based on capabilities
        self._setup_wasi()

        # Cache of loaded modules
        self._modules: Dict[str, WasmModule] = {}

    def _setup_wasi(self):
        """Configure WASI imports based on capabilities."""
        wasi_config = wasmtime.WasiConfig()

        # File system access
        if self.config.has_capability(SandboxCapability.FILE_READ):
            # Pre-open directories for reading
            root = str(self.config.root_path)
            wasi_config.preopen_dir(root, "/workspace")

        if self.config.has_capability(SandboxCapability.FILE_WRITE):
            # Enable stdout/stderr
            wasi_config.inherit_stdout()
            wasi_config.inherit_stderr()

        # Environment
        if self.config.has_capability(SandboxCapability.ENV_READ):
            wasi_config.inherit_env()

        # Define WASI in linker
        self._linker.define_wasi()

        # Store WASI config for later
        self._wasi_config = wasi_config

    def load_module(self, manifest: WasmToolManifest) -> WasmModule:
        """
        Load a WASM module.

        Args:
            manifest: Tool manifest

        Returns:
            Loaded WasmModule
        """
        if manifest.name in self._modules:
            return self._modules[manifest.name]

        if not manifest.wasm_path.exists():
            raise FileNotFoundError(f"WASM file not found: {manifest.wasm_path}")

        # Validate capabilities
        for cap in manifest.capabilities:
            if not self.config.has_capability(cap):
                raise PermissionError(
                    f"Tool {manifest.name} requires capability {cap.value} "
                    "which is not granted"
                )

        # Load and compile module
        module = wasmtime.Module.from_file(self._engine, str(manifest.wasm_path))

        # Create store with WASI
        store = wasmtime.Store(self._engine)
        store.set_wasi(self._wasi_config)

        # Instantiate
        instance = self._linker.instantiate(store, module)

        # Wrap and cache
        wasm_module = WasmModule(manifest, store, instance)
        self._modules[manifest.name] = wasm_module

        logger.info(f"Loaded WASM module: {manifest.name} v{manifest.version}")
        return wasm_module

    def execute(
        self,
        module: WasmModule,
        func_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> WasmResult:
        """
        Execute a function in a WASM module.

        Args:
            module: Loaded module
            func_name: Function to call (default: entry_func from manifest)
            args: Arguments as dictionary

        Returns:
            WasmResult
        """
        func_name = func_name or module.manifest.entry_func
        args = args or {}

        start_time = time.time()

        try:
            # Convert args to JSON string for simple interface
            args_json = json.dumps(args)

            # Allocate and write args to memory
            if module.memory:
                # Simple convention: write args at offset 1024
                args_ptr = 1024
                args_len = module.write_string(args_ptr, args_json)

                # Call with pointer and length
                result_ptr = module.call(func_name, args_ptr, args_len)

                # Read result (assuming NULL-terminated string at result_ptr)
                # This is a simplified protocol - real impl would be more robust
                if isinstance(result_ptr, int) and result_ptr > 0:
                    # Read result length from convention location
                    result_json = module.read_string(result_ptr, 4096)
                    result_json = result_json.split('\0')[0]  # Trim at null
                    output = json.loads(result_json) if result_json else None
                else:
                    output = result_ptr
            else:
                # Direct call without memory interface
                output = module.call(func_name)

            return WasmResult(
                success=True,
                output=output,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"WASM execution failed: {e}")
            return WasmResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def unload_module(self, name: str):
        """Unload a cached module."""
        if name in self._modules:
            del self._modules[name]
            logger.debug(f"Unloaded WASM module: {name}")


class WasmToolRegistry:
    """
    Registry for WASM tool modules.

    Discovers and manages available WASM tools.
    """

    def __init__(self, modules_dir: Optional[Path] = None):
        """
        Initialize registry.

        Args:
            modules_dir: Directory containing WASM modules
        """
        self.modules_dir = modules_dir or Path(".ragix/wasm_modules")
        self._manifests: Dict[str, WasmToolManifest] = {}
        self._scan()

    def _scan(self):
        """Scan for available WASM modules."""
        if not self.modules_dir.exists():
            return

        # Look for manifest.json files
        for manifest_path in self.modules_dir.rglob("manifest.json"):
            try:
                with open(manifest_path) as f:
                    data = json.load(f)

                manifest = WasmToolManifest.from_dict(
                    data,
                    manifest_path.parent
                )
                self._manifests[manifest.name] = manifest
                logger.debug(f"Found WASM tool: {manifest.name}")

            except Exception as e:
                logger.warning(f"Failed to load manifest {manifest_path}: {e}")

        # Also look for standalone .wasm files with sidecar manifests
        for wasm_path in self.modules_dir.rglob("*.wasm"):
            name = wasm_path.stem
            if name in self._manifests:
                continue

            # Check for .manifest.json sidecar
            manifest_path = wasm_path.with_suffix(".manifest.json")
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        data = json.load(f)
                    data["wasm_path"] = str(wasm_path.name)
                    manifest = WasmToolManifest.from_dict(data, wasm_path.parent)
                    self._manifests[manifest.name] = manifest
                except Exception as e:
                    logger.warning(f"Failed to load sidecar manifest: {e}")
            else:
                # Create basic manifest
                self._manifests[name] = WasmToolManifest(
                    name=name,
                    version="0.0.0",
                    description=f"WASM tool: {name}",
                    wasm_path=wasm_path,
                )

    def get(self, name: str) -> Optional[WasmToolManifest]:
        """Get tool manifest by name."""
        return self._manifests.get(name)

    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self._manifests.keys())

    def install(self, wasm_path: Path, manifest: Optional[Dict] = None) -> bool:
        """
        Install a WASM tool.

        Args:
            wasm_path: Path to .wasm file
            manifest: Optional manifest data

        Returns:
            True if installed
        """
        if not wasm_path.exists():
            logger.error(f"WASM file not found: {wasm_path}")
            return False

        # Create modules directory
        self.modules_dir.mkdir(parents=True, exist_ok=True)

        # Copy WASM file
        dest = self.modules_dir / wasm_path.name
        import shutil
        shutil.copy2(wasm_path, dest)

        # Write manifest
        name = wasm_path.stem
        manifest_data = manifest or {
            "name": name,
            "version": "1.0.0",
            "description": f"Installed tool: {name}",
            "wasm_path": wasm_path.name,
        }

        manifest_path = self.modules_dir / f"{name}.manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)

        # Reload
        self._scan()
        logger.info(f"Installed WASM tool: {name}")
        return True


class WasmSandbox(BaseSandbox):
    """
    WASM-based sandbox for secure tool execution.

    Uses WebAssembly for isolated, deterministic execution.
    """

    def __init__(self, config: Optional[SandboxConfig] = None, **kwargs):
        """
        Initialize WASM sandbox.

        Args:
            config: Sandbox configuration
            **kwargs: Additional arguments
        """
        if not WASMTIME_AVAILABLE:
            raise ImportError(
                "wasmtime is required for WASM sandbox. "
                "Install with: pip install wasmtime>=14.0.0"
            )

        config = config or SandboxConfig(sandbox_type=SandboxType.WASM)
        super().__init__(config)

        # Initialize runtime
        self._runtime = WasmRuntime(config)

        # Initialize registry
        modules_dir = config.wasm_modules_dir or Path(".ragix/wasm_modules")
        self._registry = WasmToolRegistry(modules_dir)

        # Command -> tool mapping
        self._command_map: Dict[str, str] = {}
        self._setup_command_map()

    def _setup_command_map(self):
        """Setup mapping from commands to WASM tools."""
        for tool_name in self._registry.list_tools():
            # Basic mapping: tool name as command
            self._command_map[tool_name] = tool_name

            # Also map common aliases
            manifest = self._registry.get(tool_name)
            if manifest:
                # Handle tool-specific aliases
                if "grep" in tool_name.lower():
                    self._command_map["grep"] = tool_name
                    self._command_map["rg"] = tool_name
                elif "validate" in tool_name.lower():
                    self._command_map["validate"] = tool_name

    @property
    def sandbox_type(self) -> SandboxType:
        return SandboxType.WASM

    def supports_command(self, command: str) -> bool:
        """Check if WASM sandbox supports the command."""
        if not command.strip():
            return False

        cmd_name = command.split()[0]
        return cmd_name in self._command_map

    def run(
        self,
        command: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute command via WASM tool.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            **kwargs: Additional arguments

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        # Parse command
        parts = command.strip().split()
        if not parts:
            return ExecutionResult(
                command=command,
                sandbox_type=SandboxType.WASM,
                root_path=self._config.root_path,
                stderr="Empty command",
                return_code=1,
            )

        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # Find tool
        tool_name = self._command_map.get(cmd_name)
        if not tool_name:
            return ExecutionResult(
                command=command,
                sandbox_type=SandboxType.WASM,
                root_path=self._config.root_path,
                stderr=f"Unknown WASM tool: {cmd_name}",
                return_code=127,
            )

        # Get manifest
        manifest = self._registry.get(tool_name)
        if not manifest:
            return ExecutionResult(
                command=command,
                sandbox_type=SandboxType.WASM,
                root_path=self._config.root_path,
                stderr=f"Tool manifest not found: {tool_name}",
                return_code=1,
            )

        try:
            # Load module
            module = self._runtime.load_module(manifest)

            # Execute
            result = self._runtime.execute(
                module,
                args={"command": command, "args": args},
            )

            return ExecutionResult(
                command=command,
                sandbox_type=SandboxType.WASM,
                root_path=self._config.root_path,
                stdout=json.dumps(result.output) if result.output else "",
                stderr=result.error or "",
                return_code=0 if result.success else 1,
                execution_time=time.time() - start_time,
                metadata={"tool": tool_name, "wasm": True},
            )

        except Exception as e:
            logger.error(f"WASM execution error: {e}")
            return ExecutionResult(
                command=command,
                sandbox_type=SandboxType.WASM,
                root_path=self._config.root_path,
                stderr=str(e),
                return_code=1,
                execution_time=time.time() - start_time,
            )

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available WASM tools."""
        tools = []
        for name in self._registry.list_tools():
            manifest = self._registry.get(name)
            if manifest:
                tools.append({
                    "name": manifest.name,
                    "version": manifest.version,
                    "description": manifest.description,
                    "capabilities": [c.value for c in manifest.capabilities],
                })
        return tools

    def install_tool(self, wasm_path: Path, manifest: Optional[Dict] = None) -> bool:
        """Install a WASM tool."""
        success = self._registry.install(wasm_path, manifest)
        if success:
            self._setup_command_map()
        return success
