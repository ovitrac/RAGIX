"""
KOAS Kernel Base Classes

Following the VirtualHybridLab pattern:
- Kernels do computation (no LLM inside)
- Produce structured output + summary for LLM consumption
- Fully traceable and reproducible

Design Principles:
1. No LLM inside — pure computation
2. Deterministic — same input = same output
3. Traceable — full audit trail
4. Composable — clear dependencies

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class KernelInput:
    """Standard input for any kernel.

    Attributes:
        workspace: Audit workspace root directory
        config: Kernel-specific configuration from manifest
        dependencies: Required input files from previous stages
    """
    workspace: Path
    config: Dict[str, Any]
    dependencies: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure workspace is a Path object."""
        if isinstance(self.workspace, str):
            self.workspace = Path(self.workspace)


@dataclass
class KernelOutput:
    """Standard output from any kernel.

    Attributes:
        success: Whether execution completed without errors
        data: Full structured data (JSON-serializable)
        summary: Human-readable summary (<500 chars) for LLM consumption
        output_file: Path where JSON was persisted

    Traceability:
        kernel_name: Name of the kernel that produced this output
        kernel_version: Version of the kernel
        execution_time_ms: Execution time in milliseconds
        input_hash: SHA256 hash of inputs for reproducibility
        dependencies_used: List of dependency names used

    Diagnostics:
        warnings: Non-fatal issues encountered
        errors: Errors that caused failure (if success=False)
    """
    success: bool
    data: Dict[str, Any]
    summary: str
    output_file: Path

    # Traceability
    kernel_name: str
    kernel_version: str
    execution_time_ms: int
    input_hash: str
    dependencies_used: List[str]

    # Diagnostics
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "summary": self.summary,
            "output_file": str(self.output_file),
            "kernel_name": self.kernel_name,
            "kernel_version": self.kernel_version,
            "execution_time_ms": self.execution_time_ms,
            "input_hash": self.input_hash,
            "dependencies_used": self.dependencies_used,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class Kernel(ABC):
    """
    Abstract base class for all KOAS kernels.

    A kernel is a specialized computation unit that:
    1. Encapsulates domain logic (AST parsing, metrics, graph analysis)
    2. Wraps existing tools (RAGIX CLI, Python libraries, shell scripts)
    3. Produces structured output (JSON data + human-readable summary)
    4. Is fully deterministic (same input = same output)
    5. Contains no LLM (pure computation, no AI reasoning inside)

    Subclasses must implement:
        - compute(): Core computation logic
        - summarize(): Generate human-readable summary

    Subclasses should override:
        - name: Unique kernel identifier
        - version: Semantic version
        - category: Category (audit, transform, test, docs)
        - stage: Pipeline stage (1=collection, 2=analysis, 3=reporting)
        - requires: List of kernel names this depends on
        - provides: List of capabilities this kernel provides

    Example:
        class MyKernel(Kernel):
            name = "my_kernel"
            version = "1.0.0"
            category = "audit"
            stage = 1
            requires = []
            provides = ["data_type"]

            def compute(self, input: KernelInput) -> Dict[str, Any]:
                # Pure computation here
                return {"result": ...}

            def summarize(self, data: Dict[str, Any]) -> str:
                return f"Processed {len(data)} items."
    """

    # Kernel metadata — override in subclasses
    name: str = "base"
    version: str = "1.0.0"
    category: str = "base"
    stage: int = 0
    description: str = "Base kernel"

    # Dependency declaration
    requires: List[str] = []   # Kernel names this depends on
    provides: List[str] = []   # What this kernel provides

    @abstractmethod
    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """
        Core computation logic. Must be deterministic.

        Args:
            input: KernelInput with workspace, config, and dependencies

        Returns:
            Raw data dictionary (JSON-serializable)

        Raises:
            RuntimeError: If computation fails
        """
        pass

    @abstractmethod
    def summarize(self, data: Dict[str, Any]) -> str:
        """
        Generate human-readable summary for LLM consumption.

        The summary should be:
        - Less than 500 characters
        - Concise and informative
        - Highlight key findings

        Args:
            data: Output from compute()

        Returns:
            Human-readable summary string
        """
        pass

    def validate_input(self, input: KernelInput) -> List[str]:
        """
        Validate input before computation.

        Override to add custom validation.

        Args:
            input: KernelInput to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check workspace exists
        if not input.workspace.exists():
            errors.append(f"Workspace does not exist: {input.workspace}")

        # Check required dependencies
        for dep in self.requires:
            if dep not in input.dependencies:
                errors.append(f"Missing required dependency: {dep}")
            elif not input.dependencies[dep].exists():
                errors.append(f"Dependency file does not exist: {input.dependencies[dep]}")

        return errors

    def run(self, input: KernelInput) -> KernelOutput:
        """
        Execute kernel with full traceability.

        Do NOT override — override compute() and summarize() instead.

        This method:
        1. Validates input
        2. Computes input hash for reproducibility
        3. Runs computation
        4. Generates summary
        5. Persists output to JSON
        6. Returns KernelOutput with full metadata

        Args:
            input: KernelInput with workspace, config, and dependencies

        Returns:
            KernelOutput with data, summary, and traceability info
        """
        start_time = datetime.now()
        warnings = []
        errors = []

        # Validate input
        validation_errors = self.validate_input(input)
        if validation_errors:
            for err in validation_errors:
                logger.error(f"[{self.name}] Validation error: {err}")
            return KernelOutput(
                success=False,
                data={"validation_errors": validation_errors},
                summary=f"Kernel {self.name} failed validation: {validation_errors[0]}",
                output_file=input.workspace / f"stage{self.stage}" / f"{self.name}.json",
                kernel_name=self.name,
                kernel_version=self.version,
                execution_time_ms=0,
                input_hash="",
                dependencies_used=[],
                errors=validation_errors
            )

        # Compute input hash for reproducibility
        input_hash = self._hash_input(input)
        logger.info(f"[{self.name}] Starting computation (input_hash={input_hash[:8]})")

        try:
            # Run computation
            data = self.compute(input)

            # Generate summary
            summary = self.summarize(data)

            # Truncate summary if too long
            if len(summary) > 500:
                summary = summary[:497] + "..."
                warnings.append("Summary truncated to 500 characters")

            success = True
            logger.info(f"[{self.name}] Computation successful")

        except Exception as e:
            logger.error(f"[{self.name}] Computation failed: {e}")
            data = {"error": str(e), "error_type": type(e).__name__}
            summary = f"Kernel {self.name} failed: {str(e)[:100]}"
            success = False
            errors.append(str(e))

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Determine output path
        output_file = input.workspace / f"stage{self.stage}" / f"{self.name}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Persist output with metadata
        output_data = {
            "_meta": {
                "kernel_name": self.name,
                "kernel_version": self.version,
                "execution_time_ms": execution_time_ms,
                "input_hash": input_hash,
                "timestamp": datetime.now().isoformat(),
                "success": success,
            },
            "data": data
        }
        output_file.write_text(json.dumps(output_data, indent=2, default=str))

        # Also write summary for LLM consumption
        summary_file = output_file.with_suffix('.summary.txt')
        summary_file.write_text(summary)

        logger.info(f"[{self.name}] Output saved to {output_file} ({execution_time_ms}ms)")

        return KernelOutput(
            success=success,
            data=data,
            summary=summary,
            output_file=output_file,
            kernel_name=self.name,
            kernel_version=self.version,
            execution_time_ms=execution_time_ms,
            input_hash=input_hash,
            dependencies_used=list(input.dependencies.keys()),
            warnings=warnings,
            errors=errors
        )

    def _hash_input(self, input: KernelInput) -> str:
        """Generate SHA256 hash of inputs for reproducibility.

        The hash captures:
        - Configuration (sorted keys for determinism)
        - Dependency paths
        - Kernel name and version

        Args:
            input: KernelInput to hash

        Returns:
            16-character hexadecimal hash prefix
        """
        content = json.dumps({
            "kernel": f"{self.name}@{self.version}",
            "config": input.config,
            "dependencies": {k: str(v) for k, v in sorted(input.dependencies.items())}
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"<Kernel {self.name}@{self.version} stage={self.stage}>"


# Type alias for kernel classes
KernelClass = type[Kernel]
