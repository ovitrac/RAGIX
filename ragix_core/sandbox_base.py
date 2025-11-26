"""
Sandbox Base - Abstract sandbox protocol for pluggable execution backends

Provides a unified interface for different sandbox implementations:
- ShellSandbox (existing) - subprocess-based shell execution
- WasmSandbox (new) - WebAssembly-based tool execution
- HybridSandbox (new) - Routes to appropriate backend

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)


class SandboxType(str, Enum):
    """Types of sandbox backends."""
    SHELL = "shell"
    WASM = "wasm"
    HYBRID = "hybrid"


class SandboxCapability(str, Enum):
    """Capabilities that can be granted to sandbox execution."""
    FILE_READ = "fs:read"
    FILE_WRITE = "fs:write"
    NET_CONNECT = "net:connect"
    ENV_READ = "env:read"
    CLOCK = "clock"
    RANDOM = "random"
    PROC_EXEC = "proc:exec"


@dataclass
class SandboxConfig:
    """
    Configuration for sandbox execution.

    Defines security boundaries and resource limits.
    """
    sandbox_type: SandboxType = SandboxType.SHELL
    root_path: Path = field(default_factory=Path.cwd)

    # Capabilities
    capabilities: Set[SandboxCapability] = field(default_factory=lambda: {
        SandboxCapability.FILE_READ,
        SandboxCapability.FILE_WRITE,
        SandboxCapability.CLOCK,
        SandboxCapability.RANDOM,
    })

    # Resource limits
    max_execution_time: float = 60.0  # seconds
    max_memory_mb: int = 512
    max_output_size: int = 1024 * 1024  # 1MB

    # Path restrictions
    allowed_paths: List[Path] = field(default_factory=list)
    denied_paths: List[Path] = field(default_factory=list)

    # Command restrictions (for shell)
    command_allowlist: Optional[List[str]] = None
    command_denylist: List[str] = field(default_factory=list)

    # WASM-specific
    wasm_modules_dir: Optional[Path] = None

    def has_capability(self, cap: SandboxCapability) -> bool:
        """Check if capability is granted."""
        return cap in self.capabilities


@dataclass
class ExecutionResult:
    """
    Result of sandbox command execution.

    Unified result format across all sandbox types.
    """
    command: str
    sandbox_type: SandboxType
    root_path: Path
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.return_code == 0

    @property
    def output(self) -> str:
        """Combined output (stdout + stderr if failed)."""
        if self.success:
            return self.stdout
        return f"{self.stdout}\n{self.stderr}".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "sandbox_type": self.sandbox_type.value,
            "root_path": str(self.root_path),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "truncated": self.truncated,
            "success": self.success,
            "metadata": self.metadata,
        }


@runtime_checkable
class SandboxProtocol(Protocol):
    """
    Protocol defining the sandbox interface.

    All sandbox implementations must satisfy this protocol.
    """

    def run(
        self,
        command: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a command in the sandbox.

        Args:
            command: Command to execute
            timeout: Optional timeout override
            **kwargs: Additional backend-specific arguments

        Returns:
            ExecutionResult
        """
        ...

    def supports_command(self, command: str) -> bool:
        """
        Check if the sandbox supports a given command.

        Args:
            command: Command to check

        Returns:
            True if supported
        """
        ...

    @property
    def sandbox_type(self) -> SandboxType:
        """Get the sandbox type."""
        ...

    @property
    def config(self) -> SandboxConfig:
        """Get sandbox configuration."""
        ...


class BaseSandbox(ABC):
    """
    Abstract base class for sandbox implementations.

    Provides common functionality and enforces the protocol.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox.

        Args:
            config: Sandbox configuration
        """
        self._config = config or SandboxConfig()

    @property
    def config(self) -> SandboxConfig:
        """Get sandbox configuration."""
        return self._config

    @property
    @abstractmethod
    def sandbox_type(self) -> SandboxType:
        """Get the sandbox type."""
        pass

    @abstractmethod
    def run(
        self,
        command: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute a command in the sandbox."""
        pass

    @abstractmethod
    def supports_command(self, command: str) -> bool:
        """Check if the sandbox supports a given command."""
        pass

    def validate_command(self, command: str) -> tuple[bool, Optional[str]]:
        """
        Validate command against security policies.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check denylist
        for denied in self._config.command_denylist:
            if denied in command:
                return False, f"Command contains denied pattern: {denied}"

        # Check allowlist if specified
        if self._config.command_allowlist:
            cmd_name = command.split()[0] if command.strip() else ""
            if cmd_name not in self._config.command_allowlist:
                return False, f"Command not in allowlist: {cmd_name}"

        return True, None

    def check_path_access(self, path: Path, write: bool = False) -> bool:
        """
        Check if path access is allowed.

        Args:
            path: Path to check
            write: True if write access needed

        Returns:
            True if access allowed
        """
        # Check capability
        cap = SandboxCapability.FILE_WRITE if write else SandboxCapability.FILE_READ
        if not self._config.has_capability(cap):
            return False

        # Resolve path
        try:
            resolved = path.resolve()
        except Exception:
            return False

        # Check denied paths
        for denied in self._config.denied_paths:
            try:
                if resolved.is_relative_to(denied.resolve()):
                    return False
            except (ValueError, RuntimeError):
                pass

        # Check allowed paths (if specified)
        if self._config.allowed_paths:
            for allowed in self._config.allowed_paths:
                try:
                    if resolved.is_relative_to(allowed.resolve()):
                        return True
                except (ValueError, RuntimeError):
                    pass
            return False

        # Check within root
        try:
            resolved.relative_to(self._config.root_path.resolve())
            return True
        except ValueError:
            return False


class SandboxRegistry:
    """
    Registry for sandbox backend implementations.

    Allows dynamic registration and selection of sandbox types.
    """

    def __init__(self):
        """Initialize registry."""
        self._backends: Dict[SandboxType, type] = {}
        self._instances: Dict[str, BaseSandbox] = {}

    def register(self, sandbox_type: SandboxType, backend_class: type):
        """
        Register a sandbox backend.

        Args:
            sandbox_type: Type identifier
            backend_class: Class implementing BaseSandbox
        """
        if not issubclass(backend_class, BaseSandbox):
            raise TypeError(f"{backend_class} must inherit from BaseSandbox")

        self._backends[sandbox_type] = backend_class
        logger.debug(f"Registered sandbox backend: {sandbox_type.value}")

    def create(
        self,
        sandbox_type: SandboxType,
        config: Optional[SandboxConfig] = None,
        **kwargs
    ) -> BaseSandbox:
        """
        Create a sandbox instance.

        Args:
            sandbox_type: Type of sandbox to create
            config: Sandbox configuration
            **kwargs: Additional arguments for backend

        Returns:
            Sandbox instance
        """
        if sandbox_type not in self._backends:
            raise ValueError(f"Unknown sandbox type: {sandbox_type}")

        backend_class = self._backends[sandbox_type]

        config = config or SandboxConfig(sandbox_type=sandbox_type)
        return backend_class(config=config, **kwargs)

    def get_or_create(
        self,
        key: str,
        sandbox_type: SandboxType,
        config: Optional[SandboxConfig] = None,
        **kwargs
    ) -> BaseSandbox:
        """
        Get existing sandbox or create new one.

        Args:
            key: Unique key for caching
            sandbox_type: Type of sandbox
            config: Configuration
            **kwargs: Backend arguments

        Returns:
            Sandbox instance
        """
        if key not in self._instances:
            self._instances[key] = self.create(sandbox_type, config, **kwargs)

        return self._instances[key]

    @property
    def available_types(self) -> List[SandboxType]:
        """Get list of available sandbox types."""
        return list(self._backends.keys())


def create_sandbox(
    sandbox_type: str = "shell",
    root_path: Optional[Path] = None,
    config: Optional[SandboxConfig] = None,
    **kwargs
) -> BaseSandbox:
    """
    Create a sandbox of the specified type.

    Factory function for easy sandbox creation.

    Args:
        sandbox_type: "shell", "wasm", or "hybrid"
        root_path: Sandbox root directory
        config: Full configuration (overrides other params)
        **kwargs: Additional backend arguments

    Returns:
        Configured sandbox instance
    """
    from .tools_shell import ShellSandbox

    # Parse type
    try:
        stype = SandboxType(sandbox_type)
    except ValueError:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")

    # Build config
    if config is None:
        config = SandboxConfig(
            sandbox_type=stype,
            root_path=root_path or Path.cwd(),
        )

    # Create appropriate sandbox
    if stype == SandboxType.SHELL:
        # Adapt existing ShellSandbox
        shell_sandbox = ShellSandbox(str(config.root_path))
        return ShellSandboxAdapter(shell_sandbox, config)

    elif stype == SandboxType.WASM:
        try:
            from .wasm_sandbox import WasmSandbox
            return WasmSandbox(config, **kwargs)
        except ImportError:
            logger.warning("WASM sandbox not available, falling back to shell")
            shell_sandbox = ShellSandbox(str(config.root_path))
            return ShellSandboxAdapter(shell_sandbox, config)

    elif stype == SandboxType.HYBRID:
        try:
            from .wasm_sandbox import WasmSandbox
            wasm = WasmSandbox(config, **kwargs)
        except ImportError:
            wasm = None

        shell_sandbox = ShellSandbox(str(config.root_path))
        shell = ShellSandboxAdapter(shell_sandbox, config)

        return HybridSandbox(shell, wasm, config)

    raise ValueError(f"Unsupported sandbox type: {sandbox_type}")


class ShellSandboxAdapter(BaseSandbox):
    """
    Adapter to make existing ShellSandbox conform to BaseSandbox protocol.

    Wraps the legacy ShellSandbox with the new interface.
    """

    def __init__(self, shell_sandbox, config: Optional[SandboxConfig] = None):
        """
        Initialize adapter.

        Args:
            shell_sandbox: Existing ShellSandbox instance
            config: Sandbox configuration
        """
        super().__init__(config)
        self._shell = shell_sandbox

    @property
    def sandbox_type(self) -> SandboxType:
        return SandboxType.SHELL

    def run(
        self,
        command: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute command via shell sandbox."""
        import time

        # Validate command
        valid, error = self.validate_command(command)
        if not valid:
            return ExecutionResult(
                command=command,
                sandbox_type=SandboxType.SHELL,
                root_path=self._config.root_path,
                stderr=error or "Command validation failed",
                return_code=1,
            )

        # Execute
        start = time.time()
        timeout = timeout or self._config.max_execution_time

        result = self._shell.run(command, timeout=int(timeout))

        return ExecutionResult(
            command=command,
            sandbox_type=SandboxType.SHELL,
            root_path=Path(result.cwd),
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            execution_time=time.time() - start,
        )

    def supports_command(self, command: str) -> bool:
        """Shell supports most commands."""
        valid, _ = self.validate_command(command)
        return valid


class HybridSandbox(BaseSandbox):
    """
    Hybrid sandbox that routes commands to appropriate backend.

    Prefers WASM when available, falls back to shell.
    """

    def __init__(
        self,
        shell_sandbox: BaseSandbox,
        wasm_sandbox: Optional[BaseSandbox],
        config: Optional[SandboxConfig] = None,
    ):
        """
        Initialize hybrid sandbox.

        Args:
            shell_sandbox: Shell backend
            wasm_sandbox: WASM backend (optional)
            config: Configuration
        """
        super().__init__(config)
        self._shell = shell_sandbox
        self._wasm = wasm_sandbox

    @property
    def sandbox_type(self) -> SandboxType:
        return SandboxType.HYBRID

    def run(
        self,
        command: str,
        timeout: Optional[float] = None,
        prefer_wasm: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute command via most appropriate backend.

        Args:
            command: Command to execute
            timeout: Timeout override
            prefer_wasm: Prefer WASM if available
            **kwargs: Additional arguments
        """
        # Try WASM first if preferred and available
        if prefer_wasm and self._wasm and self._wasm.supports_command(command):
            logger.debug(f"Routing to WASM: {command[:50]}")
            return self._wasm.run(command, timeout, **kwargs)

        # Fall back to shell
        logger.debug(f"Routing to shell: {command[:50]}")
        return self._shell.run(command, timeout, **kwargs)

    def supports_command(self, command: str) -> bool:
        """Check if either backend supports the command."""
        if self._wasm and self._wasm.supports_command(command):
            return True
        return self._shell.supports_command(command)


# Global registry
_sandbox_registry: Optional[SandboxRegistry] = None


def get_sandbox_registry() -> SandboxRegistry:
    """Get global sandbox registry."""
    global _sandbox_registry
    if _sandbox_registry is None:
        _sandbox_registry = SandboxRegistry()
        # Register default backends
        _sandbox_registry.register(SandboxType.SHELL, ShellSandboxAdapter)

    return _sandbox_registry
