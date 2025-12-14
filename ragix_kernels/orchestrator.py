"""
KOAS Orchestrator — Kernel execution coordination and manifest management.

Following the KOAS design document:
- Parses manifest.yaml configuration
- Resolves kernel dependencies via topological sort
- Executes kernels by stage with progress reporting
- Collects summaries for LLM consumption
- Full audit trail with SHA256 chain

Usage:
    python -m ragix_kernels.orchestrator run --workspace /path/to/audit --stage 1
    python -m ragix_kernels.orchestrator init --workspace /path/to/audit --project /path/to/code
    python -m ragix_kernels.orchestrator summary --workspace /path/to/audit

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from ragix_kernels.base import Kernel, KernelInput, KernelOutput
from ragix_kernels.registry import KernelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# GPU and System Detection
# =============================================================================

def check_gpu() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check for GPU availability (NVIDIA CUDA or AMD ROCm).

    Returns:
        Tuple of (available, description, details_dict)
    """
    details = {"vendor": None, "name": None, "memory": None, "cuda_version": None}

    # Check NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            gpu_name = parts[0].strip() if len(parts) > 0 else "Unknown"
            gpu_memory = parts[1].strip() if len(parts) > 1 else "Unknown"
            driver_version = parts[2].strip() if len(parts) > 2 else "Unknown"

            details["vendor"] = "NVIDIA"
            details["name"] = gpu_name
            details["memory"] = gpu_memory
            details["driver_version"] = driver_version

            # Check CUDA version
            cuda_result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if cuda_result.returncode == 0:
                for line in cuda_result.stdout.split("\n"):
                    if "release" in line.lower():
                        details["cuda_version"] = line.split("release")[-1].strip().split(",")[0]
                        break

            return True, f"NVIDIA {gpu_name} ({gpu_memory})", details
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check AMD GPU
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            details["vendor"] = "AMD"
            details["name"] = "ROCm GPU"
            return True, "AMD GPU detected (ROCm)", details
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, "No GPU detected (CPU-only mode)", details


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for KOAS status."""
    import platform
    import shutil

    info = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_version": platform.release(),
        "cpu_count": None,
        "memory_total_gb": None,
        "disk_free_gb": None,
        "gpu_available": False,
        "gpu_info": {},
        "ollama_available": False,
        "ollama_models": [],
        "rag_available": False,
    }

    # CPU count
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # Memory
    try:
        import psutil
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    # Disk space
    try:
        stat = shutil.disk_usage(Path.cwd())
        info["disk_free_gb"] = round(stat.free / (1024**3), 1)
    except Exception:
        pass

    # GPU
    gpu_avail, gpu_desc, gpu_details = check_gpu()
    info["gpu_available"] = gpu_avail
    info["gpu_info"] = gpu_details
    info["gpu_description"] = gpu_desc

    # Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            info["ollama_available"] = True
            info["ollama_models"] = [m["name"] for m in models]
    except Exception:
        pass

    # RAG module availability
    try:
        from ragix_core.rag_project import RAGProject
        info["rag_available"] = True
    except ImportError:
        pass

    return info


# =============================================================================
# Manifest Parsing and Validation
# =============================================================================

@dataclass
class ManifestConfig:
    """Parsed manifest configuration."""

    # Metadata
    name: str
    version: str
    date: str
    author: str

    # Project
    project_name: str
    project_path: Path
    language: str
    modules: List[Dict[str, str]] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    # Stage configurations
    stage1: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stage2: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stage3: Dict[str, Any] = field(default_factory=dict)

    # Output
    output_format: str = "markdown"
    output_template: str = "default"
    output: Dict[str, Any] = field(default_factory=dict)

    # LLM configuration (optional, for agent coordination)
    llm: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Sovereignty settings
    sovereignty: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ManifestConfig":
        """Load manifest from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse audit metadata
        audit = data.get("audit", {})

        # Parse project
        project = data.get("project", {})
        project_path = project.get("path", ".")
        if not Path(project_path).is_absolute():
            # Make relative to manifest directory
            project_path = path.parent / project_path

        return cls(
            name=audit.get("name", "Unnamed Audit"),
            version=audit.get("version", "1.0"),
            date=audit.get("date", datetime.now().strftime("%Y-%m-%d")),
            author=audit.get("author", "Unknown"),
            project_name=project.get("name", "Unknown"),
            project_path=Path(project_path),
            language=project.get("language", "java"),
            modules=project.get("modules", []),
            exclude=project.get("exclude", []),
            stage1=data.get("stage1", {}),
            stage2=data.get("stage2", {}),
            stage3=data.get("stage3", {}),
            output_format=data.get("output", {}).get("format", "markdown"),
            output_template=data.get("output", {}).get("template", "default"),
            output=data.get("output", {}),
            llm=data.get("llm", {}),
            sovereignty=data.get("sovereignty", {"mode": "local"}),
        )

    def get_kernel_config(self, kernel_name: str, stage: int) -> Dict[str, Any]:
        """Get configuration for a specific kernel."""
        stage_config = {1: self.stage1, 2: self.stage2, 3: self.stage3}.get(stage, {})
        kernel_config = stage_config.get(kernel_name, {})

        # Base config with project info
        config = {
            "project": {
                "name": self.project_name,
                "path": str(self.project_path),
                "language": self.language,
                "modules": self.modules,
                "exclude": self.exclude,
            },
            **kernel_config.get("options", {}),
        }

        # For Stage 3, add output/report configuration
        if stage == 3:
            config["language"] = self.output.get("language", "en")
            config["template"] = self.output.get("template", "default")
            config["frontmatter"] = self.output.get("frontmatter", {})

        return config

    def is_kernel_enabled(self, kernel_name: str, stage: int) -> bool:
        """Check if a kernel is enabled in the manifest."""
        stage_config = {1: self.stage1, 2: self.stage2, 3: self.stage3}.get(stage, {})
        kernel_config = stage_config.get(kernel_name, {})

        # Default to enabled if not specified
        return kernel_config.get("enabled", True)


def validate_manifest(config: ManifestConfig) -> List[str]:
    """Validate manifest configuration."""
    errors = []

    # Check project path exists
    if not config.project_path.exists():
        errors.append(f"Project path does not exist: {config.project_path}")

    # Check modules exist (if specified)
    for module in config.modules:
        module_path = config.project_path / module.get("path", "")
        if not module_path.exists():
            errors.append(f"Module path does not exist: {module_path}")

    return errors


# =============================================================================
# Execution Context and Audit Trail
# =============================================================================

@dataclass
class ExecutionContext:
    """Execution context for kernel orchestration."""

    workspace: Path
    manifest: ManifestConfig
    outputs: Dict[str, KernelOutput] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def get_dependency_paths(self, kernel_name: str) -> Dict[str, Path]:
        """Get output paths for a kernel's dependencies."""
        kernel_class = KernelRegistry.get(kernel_name)
        dependencies = {}

        for dep_name in kernel_class.requires:
            if dep_name in self.outputs:
                dependencies[dep_name] = self.outputs[dep_name].output_file
            else:
                # Try to find existing output file
                dep_class = KernelRegistry.get(dep_name)
                dep_path = self.workspace / f"stage{dep_class.stage}" / f"{dep_name}.json"
                if dep_path.exists():
                    dependencies[dep_name] = dep_path

        return dependencies

    def add_to_audit_trail(self, kernel_name: str, output: KernelOutput):
        """Add kernel execution to audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "kernel": kernel_name,
            "success": output.success,
            "execution_time_ms": output.execution_time_ms,
            "input_hash": output.input_hash,
            "output_file": str(output.output_file),
        }

        # Compute chain hash
        if self.audit_trail:
            prev_hash = self.audit_trail[-1].get("chain_hash", "")
        else:
            prev_hash = ""

        chain_content = json.dumps(entry, sort_keys=True) + prev_hash
        entry["chain_hash"] = hashlib.sha256(chain_content.encode()).hexdigest()[:16]

        self.audit_trail.append(entry)

    def save_audit_trail(self):
        """Save audit trail to workspace."""
        logs_dir = self.workspace / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        trail_file = logs_dir / "audit_trail.json"
        with open(trail_file, "w") as f:
            json.dump({
                "audit_name": self.manifest.name,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "entries": self.audit_trail,
            }, f, indent=2)

        logger.info(f"Audit trail saved to {trail_file}")


# =============================================================================
# Orchestrator Core
# =============================================================================

class Orchestrator:
    """
    KOAS Orchestrator — Coordinates kernel execution.

    Responsibilities:
    1. Parse and validate manifest
    2. Discover and register kernels
    3. Resolve dependencies via topological sort
    4. Execute kernels with progress reporting
    5. Collect summaries for LLM consumption
    6. Maintain audit trail
    """

    def __init__(self, workspace: Path, manifest_path: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            workspace: Audit workspace directory
            manifest_path: Path to manifest.yaml (default: workspace/manifest.yaml)
        """
        self.workspace = Path(workspace)
        self.manifest_path = manifest_path or (self.workspace / "manifest.yaml")

        # Ensure kernels are discovered
        KernelRegistry.discover()

        # Load manifest if exists
        self.manifest: Optional[ManifestConfig] = None
        if self.manifest_path.exists():
            self.manifest = ManifestConfig.from_yaml(self.manifest_path)

    def run_stage(
        self,
        stage: int,
        kernels: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> List[KernelOutput]:
        """
        Execute all kernels in a stage.

        Args:
            stage: Stage number (1, 2, or 3)
            kernels: Specific kernels to run (default: all enabled in stage)
            progress_callback: Optional callback(kernel_name, status, progress)
            parallel: Enable parallel execution for independent kernels
            max_workers: Maximum concurrent kernel executions (default: 4)

        Returns:
            List of KernelOutput from executed kernels
        """
        if not self.manifest:
            raise RuntimeError("No manifest loaded. Run init first.")

        # Get kernels for this stage
        if kernels is None:
            available = KernelRegistry.list_stage(stage)
            kernels = [k for k in available if self.manifest.is_kernel_enabled(k, stage)]

        if not kernels:
            logger.warning(f"No kernels to run for stage {stage}")
            return []

        # Resolve dependencies and get execution order
        ordered_kernels = KernelRegistry.resolve_dependencies(kernels)

        # Filter to only kernels in this stage (dependencies may be in earlier stages)
        stage_kernels = []
        for name in ordered_kernels:
            kernel_class = KernelRegistry.get(name)
            if kernel_class.stage == stage:
                stage_kernels.append(name)
            elif kernel_class.stage < stage:
                # Dependency from earlier stage - check if output exists
                output_path = self.workspace / f"stage{kernel_class.stage}" / f"{name}.json"
                if not output_path.exists():
                    # Need to run this dependency first
                    logger.info(f"Running dependency {name} from stage {kernel_class.stage}")
                    stage_kernels.insert(0, name)

        # Create execution context
        context = ExecutionContext(
            workspace=self.workspace,
            manifest=self.manifest,
        )

        # Load existing outputs from previous stages
        for s in range(1, stage):
            stage_dir = self.workspace / f"stage{s}"
            if stage_dir.exists():
                for json_file in stage_dir.glob("*.json"):
                    kernel_name = json_file.stem
                    # Create minimal KernelOutput for reference
                    context.outputs[kernel_name] = KernelOutput(
                        success=True,
                        data={},
                        summary="",
                        output_file=json_file,
                        kernel_name=kernel_name,
                        kernel_version="",
                        execution_time_ms=0,
                        input_hash="",
                        dependencies_used=[],
                    )

        total = len(stage_kernels)
        logger.info(f"[Stage {stage}] Executing {total} kernels: {stage_kernels}")

        if parallel and total > 1:
            results = self._run_stage_parallel(
                stage_kernels, context, progress_callback, max_workers
            )
        else:
            results = self._run_stage_sequential(
                stage_kernels, context, progress_callback
            )

        # Save audit trail
        context.save_audit_trail()

        # Generate stage summary
        self._generate_stage_summary(stage, results)

        return results

    def _run_stage_sequential(
        self,
        stage_kernels: List[str],
        context: "ExecutionContext",
        progress_callback: Optional[Callable],
    ) -> List[KernelOutput]:
        """Execute kernels sequentially."""
        results = []
        total = len(stage_kernels)

        for i, kernel_name in enumerate(stage_kernels):
            if progress_callback:
                progress_callback(kernel_name, "starting", (i + 1) / total)

            try:
                output = self._run_kernel(kernel_name, context)
                results.append(output)
                context.outputs[kernel_name] = output
                context.add_to_audit_trail(kernel_name, output)

                status = "completed" if output.success else "failed"
                if progress_callback:
                    progress_callback(kernel_name, status, (i + 1) / total)

            except Exception as e:
                logger.error(f"[{kernel_name}] Failed: {e}")
                if progress_callback:
                    progress_callback(kernel_name, "error", (i + 1) / total)
                raise

        return results

    def _run_stage_parallel(
        self,
        stage_kernels: List[str],
        context: "ExecutionContext",
        progress_callback: Optional[Callable],
        max_workers: int,
    ) -> List[KernelOutput]:
        """
        Execute kernels in parallel where possible.

        Groups kernels by dependency level and executes each level in parallel.
        """
        results = []
        completed = set()
        pending = set(stage_kernels)
        context_lock = Lock()
        total = len(stage_kernels)
        completed_count = 0

        def get_ready_kernels() -> List[str]:
            """Get kernels whose dependencies are all complete."""
            ready = []
            for name in pending:
                kernel_class = KernelRegistry.get(name)
                deps = set(kernel_class.requires)
                # Check if all dependencies are either complete or not in our list
                deps_in_stage = deps & set(stage_kernels)
                if deps_in_stage <= completed:
                    ready.append(name)
            return ready

        def run_kernel_safe(kernel_name: str) -> Tuple[str, KernelOutput]:
            """Run kernel with thread-safe context access."""
            output = self._run_kernel(kernel_name, context)

            # Thread-safe update of context
            with context_lock:
                context.outputs[kernel_name] = output
                context.add_to_audit_trail(kernel_name, output)

            return kernel_name, output

        logger.info(f"[Parallel] Running {total} kernels with max {max_workers} workers")

        while pending:
            ready = get_ready_kernels()

            if not ready:
                if pending:
                    logger.error(f"[Parallel] Deadlock detected! Remaining: {pending}")
                    raise RuntimeError(f"Dependency deadlock for kernels: {pending}")
                break

            logger.info(f"[Parallel] Batch: {ready}")

            with ThreadPoolExecutor(max_workers=min(max_workers, len(ready))) as executor:
                futures = {
                    executor.submit(run_kernel_safe, name): name
                    for name in ready
                }

                for future in as_completed(futures):
                    kernel_name = futures[future]
                    try:
                        name, output = future.result()
                        results.append(output)
                        completed.add(name)
                        pending.discard(name)

                        completed_count += 1
                        status = "completed" if output.success else "failed"
                        if progress_callback:
                            progress_callback(name, status, completed_count / total)

                    except Exception as e:
                        logger.error(f"[{kernel_name}] Parallel execution failed: {e}")
                        if progress_callback:
                            progress_callback(kernel_name, "error", completed_count / total)
                        raise

        return results

    def run_kernel(self, kernel_name: str) -> KernelOutput:
        """
        Execute a single kernel.

        Args:
            kernel_name: Name of kernel to execute

        Returns:
            KernelOutput from kernel execution
        """
        if not self.manifest:
            raise RuntimeError("No manifest loaded. Run init first.")

        # Create minimal context
        kernel_class = KernelRegistry.get(kernel_name)
        context = ExecutionContext(
            workspace=self.workspace,
            manifest=self.manifest,
        )

        # Load existing outputs for dependencies
        for dep_name in kernel_class.requires:
            dep_class = KernelRegistry.get(dep_name)
            dep_path = self.workspace / f"stage{dep_class.stage}" / f"{dep_name}.json"
            if dep_path.exists():
                context.outputs[dep_name] = KernelOutput(
                    success=True,
                    data={},
                    summary="",
                    output_file=dep_path,
                    kernel_name=dep_name,
                    kernel_version="",
                    execution_time_ms=0,
                    input_hash="",
                    dependencies_used=[],
                )

        return self._run_kernel(kernel_name, context)

    def _run_kernel(self, kernel_name: str, context: ExecutionContext) -> KernelOutput:
        """Internal kernel execution with context."""
        kernel_class = KernelRegistry.get(kernel_name)
        kernel = kernel_class()

        logger.info(f"[{kernel_name}] Starting (stage={kernel.stage})")

        # Build input
        kernel_input = KernelInput(
            workspace=context.workspace,
            config=context.manifest.get_kernel_config(kernel_name, kernel.stage),
            dependencies=context.get_dependency_paths(kernel_name),
        )

        # Execute
        output = kernel.run(kernel_input)

        if output.success:
            logger.info(
                f"[{kernel_name}] Completed in {output.execution_time_ms}ms"
            )
        else:
            logger.error(f"[{kernel_name}] Failed: {output.errors}")

        return output

    def _generate_stage_summary(self, stage: int, results: List[KernelOutput]):
        """Generate combined summary for a stage."""
        stage_dir = self.workspace / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        for output in results:
            summaries.append(f"## {output.kernel_name}\n{output.summary}\n")

        summary_text = f"# Stage {stage} Summary\n\n" + "\n".join(summaries)

        summary_file = stage_dir / "_stage_summary.txt"
        summary_file.write_text(summary_text)

        # Also create JSON summary for programmatic access
        summary_json = {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "kernels": [
                {
                    "name": o.kernel_name,
                    "success": o.success,
                    "summary": o.summary,
                    "execution_time_ms": o.execution_time_ms,
                }
                for o in results
            ],
        }
        summary_json_file = stage_dir / "_stage_summary.json"
        summary_json_file.write_text(json.dumps(summary_json, indent=2))

        logger.info(f"[Stage {stage}] Summary saved to {summary_file}")

    def get_summaries(self, stage: Optional[int] = None) -> Dict[str, str]:
        """
        Get summaries from completed kernels.

        Args:
            stage: Specific stage (default: all stages)

        Returns:
            Dictionary of kernel_name -> summary
        """
        summaries = {}
        stages = [stage] if stage else [1, 2, 3]

        for s in stages:
            stage_dir = self.workspace / f"stage{s}"
            if not stage_dir.exists():
                continue

            for summary_file in stage_dir.glob("*.summary.txt"):
                kernel_name = summary_file.stem.replace(".summary", "")
                summaries[kernel_name] = summary_file.read_text()

        return summaries

    def get_combined_summary(self) -> str:
        """Get all summaries combined for LLM consumption."""
        lines = ["# KOAS Audit Summary", ""]

        for stage in [1, 2, 3]:
            stage_summary = self.workspace / f"stage{stage}" / "_stage_summary.txt"
            if stage_summary.exists():
                lines.append(f"\n{stage_summary.read_text()}")

        return "\n".join(lines)


# =============================================================================
# Workspace Initialization
# =============================================================================

def init_workspace(
    workspace: Path,
    project_path: Path,
    project_name: Optional[str] = None,
    language: str = "java",
    author: str = "Unknown",
    init_rag: bool = False,
    rag_profile: str = "mixed_docs_code",
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """
    Initialize a new audit workspace.

    Args:
        workspace: Directory to create workspace in
        project_path: Path to project to audit
        project_name: Name of project (default: derived from path)
        language: Programming language (default: java)
        author: Author name for manifest
        init_rag: Whether to initialize RAG indexing for the project
        rag_profile: RAG indexing profile (docs_only, mixed_docs_code, code_only)

    Returns:
        Tuple of (Path to manifest.yaml, RAG status dict or None)
    """
    workspace = Path(workspace)
    project_path = Path(project_path)

    if project_name is None:
        project_name = project_path.name

    # Create directory structure
    workspace.mkdir(parents=True, exist_ok=True)
    for subdir in ["stage1", "stage2", "stage3", "report", "logs", "assets", "kb"]:
        (workspace / subdir).mkdir(exist_ok=True)

    # Generate manifest
    manifest_content = f"""# KOAS Audit Manifest
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

audit:
  name: "{project_name} Technical Audit"
  version: "1.0"
  date: "{datetime.now().strftime('%Y-%m-%d')}"
  author: "{author}"

project:
  name: "{project_name}"
  path: "{project_path}"
  language: "{language}"
  exclude:
    - "**/test/**"
    - "**/generated/**"

# Stage 1: Data Collection
stage1:
  ast_scan:
    enabled: true
    options:
      include_tests: false
      parse_javadoc: true

  metrics:
    enabled: true
    options:
      complexity_threshold: 10
      loc_threshold: 300

  dependency:
    enabled: true
    options:
      detect_cycles: true

  partition:
    enabled: true
    options:
      confidence_threshold: 0.7

  services:
    enabled: true

  timeline:
    enabled: true

# Stage 2: Analysis
stage2:
  stats_summary:
    enabled: true

  hotspots:
    enabled: true
    options:
      top_n: 50
      threshold_cc: 15

  dead_code:
    enabled: true

  coupling:
    enabled: true

  entropy:
    enabled: true

  risk:
    enabled: true

# Stage 3: Report Generation
stage3:
  sections:
    - id: "executive"
      title: "Executive Summary"
      kernel: "section_executive"

    - id: "overview"
      title: "Codebase Overview"
      kernel: "section_overview"

    - id: "risk"
      title: "Risk Assessment"
      kernel: "section_risk"

    - id: "recommendations"
      title: "Recommendations"
      kernel: "section_recommendations"

# Output Configuration
output:
  format: "markdown"
  template: "default"

# Sovereignty Settings
sovereignty:
  mode: "local"
  audit_trail: "required"
"""

    manifest_path = workspace / "manifest.yaml"
    manifest_path.write_text(manifest_content)

    logger.info(f"Initialized workspace at {workspace}")
    logger.info(f"Manifest created: {manifest_path}")

    # Initialize RAG indexing if requested
    rag_status = None
    if init_rag:
        rag_status = _init_project_rag(project_path, project_name, rag_profile)

    return manifest_path, rag_status


def _init_project_rag(
    project_path: Path,
    project_name: str,
    profile: str = "mixed_docs_code",
) -> Dict[str, Any]:
    """
    Initialize RAG indexing for a project.

    Args:
        project_path: Path to project to index
        project_name: Name of the project
        profile: RAG indexing profile

    Returns:
        Dictionary with RAG initialization status
    """
    try:
        from ragix_core.rag_project import RAGProject, ProfileType

        profile_map = {
            "docs_only": ProfileType.DOCS_ONLY,
            "mixed_docs_code": ProfileType.MIXED_DOCS_CODE,
            "code_only": ProfileType.CODE_ONLY,
        }
        profile_type = profile_map.get(profile, ProfileType.MIXED_DOCS_CODE)

        project = RAGProject(project_path)

        # Check if already initialized
        if project.is_initialized():
            logger.info(f"[RAG] Project RAG already initialized at {project_path}")
            status = project.get_status()
            return {
                "status": "exists",
                "rag_dir": str(project.rag_dir),
                "initialized": True,
                "exists": True,
                **status,
            }

        # Initialize
        logger.info(f"[RAG] Initializing Project RAG for {project_name} with profile {profile}")
        config = project.initialize(profile=profile_type, project_name=project_name)

        return {
            "status": "initialized",
            "rag_dir": str(project.rag_dir),
            "profile": profile,
            "project_name": config.project_name,
            "initialized": True,
            "exists": True,
        }

    except ImportError:
        logger.warning("[RAG] ragix_core.rag_project not available")
        return {
            "status": "unavailable",
            "error": "RAG module not installed",
            "initialized": False,
        }
    except Exception as e:
        logger.error(f"[RAG] Initialization failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "initialized": False,
        }


def start_rag_indexing(project_path: Path, full_reindex: bool = False) -> Dict[str, Any]:
    """
    Start RAG indexing for a project (background task).

    Args:
        project_path: Path to project to index
        full_reindex: Whether to clear and rebuild index

    Returns:
        Dictionary with indexing status
    """
    try:
        from ragix_core.rag_project import RAGProject

        project = RAGProject(project_path)

        if not project.is_initialized():
            return {"status": "error", "error": "Project RAG not initialized"}

        if project.is_indexing():
            return {"status": "already_running"}

        project.start_indexing(full_reindex=full_reindex)

        return {
            "status": "started",
            "project_path": str(project_path),
            "full_reindex": full_reindex,
        }

    except ImportError:
        return {"status": "error", "error": "RAG module not installed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# CLI Interface
# =============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def progress_reporter(kernel_name: str, status: str, progress: float):
    """Default progress callback for CLI."""
    bar_width = 30
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r[{bar}] {progress*100:5.1f}% | {kernel_name}: {status}", end="", flush=True)
    if status in ("completed", "failed", "error"):
        print()  # Newline after completion


def cmd_init(args):
    """Handle init command."""
    manifest_path, rag_status = init_workspace(
        workspace=Path(args.workspace),
        project_path=Path(args.project),
        project_name=args.name,
        language=args.language,
        author=args.author,
        init_rag=args.rag,
        rag_profile=args.rag_profile,
    )
    print(f"✓ Workspace initialized: {args.workspace}")
    print(f"✓ Manifest created: {manifest_path}")

    if rag_status:
        if rag_status.get("initialized"):
            print(f"✓ RAG initialized: {rag_status.get('rag_dir')}")
        elif rag_status.get("status") == "unavailable":
            print(f"⚠ RAG not available: {rag_status.get('error')}")
        elif rag_status.get("status") == "error":
            print(f"✗ RAG initialization failed: {rag_status.get('error')}")

    print(f"\nNext: Edit manifest.yaml to customize, then run:")
    print(f"  python -m ragix_kernels.orchestrator run --workspace {args.workspace} --stage 1")

    if args.rag:
        print(f"\nTo start RAG indexing:")
        print(f"  python -m ragix_kernels.orchestrator rag --project {args.project} --index")


def cmd_run(args):
    """Handle run command."""
    workspace = Path(args.workspace)

    if not workspace.exists():
        print(f"Error: Workspace does not exist: {workspace}")
        print("Run 'init' first to create a workspace.")
        sys.exit(1)

    orchestrator = Orchestrator(workspace)

    if not orchestrator.manifest:
        print(f"Error: No manifest.yaml found in {workspace}")
        sys.exit(1)

    # Validate manifest
    errors = validate_manifest(orchestrator.manifest)
    if errors:
        print("Manifest validation errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # Determine which stages to run
    if args.stage:
        stages = [args.stage]
    elif args.all:
        stages = [1, 2, 3]
    else:
        stages = [1]  # Default to stage 1

    # Parallel execution settings
    parallel = getattr(args, 'parallel', False)
    max_workers = getattr(args, 'workers', 4)

    if parallel:
        print(f"[Parallel mode] Max workers: {max_workers}")

    # Run stages
    for stage in stages:
        print(f"\n{'='*60}")
        stage_name = {1: 'Data Collection', 2: 'Analysis', 3: 'Reporting'}.get(stage, 'Unknown')
        print(f"Stage {stage}: {stage_name}")
        print(f"{'='*60}\n")

        try:
            results = orchestrator.run_stage(
                stage,
                kernels=args.kernels.split(",") if args.kernels else None,
                progress_callback=progress_reporter if not args.quiet else None,
                parallel=parallel,
                max_workers=max_workers,
            )

            # Summary
            succeeded = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            total_time = sum(r.execution_time_ms for r in results)

            mode = "parallel" if parallel else "sequential"
            print(f"\nStage {stage} complete ({mode}): {succeeded} succeeded, {failed} failed ({total_time}ms)")

        except Exception as e:
            print(f"\nError in stage {stage}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    print(f"\n✓ Audit complete. Results in: {workspace}")


def cmd_summary(args):
    """Handle summary command."""
    workspace = Path(args.workspace)
    orchestrator = Orchestrator(workspace)

    if args.stage:
        summaries = orchestrator.get_summaries(args.stage)
    else:
        print(orchestrator.get_combined_summary())
        return

    for kernel_name, summary in summaries.items():
        print(f"\n## {kernel_name}")
        print(summary)


def cmd_list(args):
    """Handle list command."""
    KernelRegistry.discover()

    if args.stage:
        kernels = KernelRegistry.list_stage(args.stage)
        print(f"Stage {args.stage} kernels:")
    elif args.category:
        kernels = KernelRegistry.list_category(args.category)
        print(f"Category '{args.category}' kernels:")
    else:
        kernels = KernelRegistry.list_all()
        print("All available kernels:")

    for name in kernels:
        info = KernelRegistry.get_info(name)
        print(f"  {name:20s} | stage={info['stage']} | {info['description']}")


def cmd_status(args):
    """Handle status command — show system and workspace status."""
    print("=" * 60)
    print("KOAS System Status")
    print("=" * 60)

    # Get system info
    info = get_system_info()

    # System section
    print("\n[System]")
    print(f"  Python:     {info.get('python_version', 'N/A')}")
    print(f"  Platform:   {info.get('platform', 'N/A')} {info.get('platform_version', '')}")
    print(f"  CPU cores:  {info.get('cpu_count', 'N/A')}")
    if info.get('memory_total_gb'):
        print(f"  Memory:     {info['memory_total_gb']} GB")
    if info.get('disk_free_gb'):
        print(f"  Disk free:  {info['disk_free_gb']} GB")

    # GPU section
    print("\n[GPU]")
    if info.get('gpu_available'):
        gpu_info = info.get('gpu_info', {})
        print(f"  ✓ {info.get('gpu_description', 'GPU detected')}")
        if gpu_info.get('cuda_version'):
            print(f"    CUDA:     {gpu_info['cuda_version']}")
        if gpu_info.get('driver_version'):
            print(f"    Driver:   {gpu_info['driver_version']}")
    else:
        print(f"  ✗ {info.get('gpu_description', 'No GPU detected')}")

    # Ollama section
    print("\n[Ollama LLM]")
    if info.get('ollama_available'):
        models = info.get('ollama_models', [])
        print(f"  ✓ Running with {len(models)} model(s)")
        for model in models[:5]:
            print(f"    - {model}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")
    else:
        print("  ✗ Not running (start with 'ollama serve')")

    # RAG section
    print("\n[RAG Module]")
    if info.get('rag_available'):
        print("  ✓ Available (ragix_core.rag_project)")
    else:
        print("  ✗ Not installed")

    # Workspace status (if provided)
    if hasattr(args, 'workspace') and args.workspace:
        workspace = Path(args.workspace)
        print(f"\n[Workspace: {workspace}]")
        if workspace.exists():
            manifest = workspace / "manifest.yaml"
            if manifest.exists():
                print("  ✓ Manifest found")
                orchestrator = Orchestrator(workspace)
                if orchestrator.manifest:
                    print(f"    Project: {orchestrator.manifest.project_name}")
                    print(f"    Path:    {orchestrator.manifest.project_path}")

                # Check stage outputs
                for stage in [1, 2, 3]:
                    stage_dir = workspace / f"stage{stage}"
                    if stage_dir.exists():
                        files = list(stage_dir.glob("*.json"))
                        stage_files = [f.stem for f in files if not f.stem.startswith("_")]
                        if stage_files:
                            print(f"  Stage {stage}: {len(stage_files)} kernel(s) completed")
            else:
                print("  ✗ No manifest.yaml found")
        else:
            print("  ✗ Workspace does not exist")

    # Kernel registry
    print("\n[Kernels]")
    KernelRegistry.discover()
    all_kernels = KernelRegistry.list_all()
    by_stage = {1: [], 2: [], 3: []}
    for name in all_kernels:
        info_k = KernelRegistry.get_info(name)
        stage = info_k.get('stage', 0)
        if stage in by_stage:
            by_stage[stage].append(name)
    print(f"  Stage 1: {len(by_stage[1])} kernels (data collection)")
    print(f"  Stage 2: {len(by_stage[2])} kernels (analysis)")
    print(f"  Stage 3: {len(by_stage[3])} kernels (reporting)")

    print("\n" + "=" * 60)


def cmd_rag(args):
    """Handle RAG command — manage project RAG indexing."""
    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)

    if args.init:
        # Initialize RAG
        print(f"Initializing RAG for: {project_path}")
        status = _init_project_rag(project_path, project_path.name, args.profile)

        if status.get("initialized"):
            print(f"✓ RAG initialized: {status.get('rag_dir')}")
        elif status.get("status") == "exists":
            print(f"✓ RAG already initialized: {status.get('rag_dir')}")
        else:
            print(f"✗ RAG initialization failed: {status.get('error')}")
            sys.exit(1)

    elif args.index:
        # Start indexing
        print(f"Starting RAG indexing for: {project_path}")
        status = start_rag_indexing(project_path, full_reindex=args.full)

        if status.get("status") == "started":
            print("✓ Indexing started in background")
            print("  Use 'ragix-web' or check .RAG/state/ for progress")
        elif status.get("status") == "already_running":
            print("⚠ Indexing already in progress")
        else:
            print(f"✗ Failed to start indexing: {status.get('error')}")
            sys.exit(1)

    elif args.status:
        # Show RAG status
        try:
            from ragix_core.rag_project import RAGProject

            project = RAGProject(project_path)
            print(f"\n[RAG Status: {project_path}]")

            if project.exists():
                print(f"  ✓ RAG directory: {project.rag_dir}")
                if project.is_initialized():
                    print("  ✓ Initialized")
                    status = project.get_status()
                    if status.get("is_indexing"):
                        print(f"  ⟳ Indexing in progress...")
                        progress = project.get_indexing_progress()
                        if progress:
                            print(f"    Progress: {progress.progress_percent:.1f}%")
                            print(f"    Files: {progress.files_processed}/{progress.files_total}")
                    else:
                        stats = project.get_stats()
                        print(f"  Files indexed:  {stats.get('files_indexed', 'N/A')}")
                        print(f"  Chunks:         {stats.get('chunks_total', 'N/A')}")
                        print(f"  Collections:    {stats.get('collections', [])}")
                else:
                    print("  ✗ Not initialized (run with --init)")
            else:
                print("  ✗ No .RAG directory found")
                print("    Run with --init to create")

        except ImportError:
            print("✗ RAG module not available")
            sys.exit(1)
    else:
        print("Specify --init, --index, or --status")
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KOAS Orchestrator — Kernel execution coordination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize a new audit workspace
  python -m ragix_kernels.orchestrator init --workspace ./audit/myproject --project /path/to/code

  # Initialize with RAG indexing
  python -m ragix_kernels.orchestrator init --workspace ./audit/myproject --project /path/to/code --rag

  # Run stage 1 (data collection)
  python -m ragix_kernels.orchestrator run --workspace ./audit/myproject --stage 1

  # Run all stages with parallel execution
  python -m ragix_kernels.orchestrator run --workspace ./audit/myproject --all --parallel

  # Run specific kernels
  python -m ragix_kernels.orchestrator run --workspace ./audit/myproject --kernels ast_scan,metrics

  # Show system and workspace status
  python -m ragix_kernels.orchestrator status --workspace ./audit/myproject

  # Manage RAG indexing
  python -m ragix_kernels.orchestrator rag --project /path/to/code --init
  python -m ragix_kernels.orchestrator rag --project /path/to/code --index

  # Get summaries for LLM
  python -m ragix_kernels.orchestrator summary --workspace ./audit/myproject

  # List available kernels
  python -m ragix_kernels.orchestrator list --stage 1
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize audit workspace")
    init_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    init_parser.add_argument("--project", "-p", required=True, help="Project path to audit")
    init_parser.add_argument("--name", "-n", help="Project name (default: from path)")
    init_parser.add_argument("--language", "-l", default="java", help="Language (default: java)")
    init_parser.add_argument("--author", "-a", default="Unknown", help="Author name")
    init_parser.add_argument("--rag", action="store_true", help="Initialize RAG indexing for project")
    init_parser.add_argument("--rag-profile", default="mixed_docs_code",
                            choices=["docs_only", "mixed_docs_code", "code_only"],
                            help="RAG indexing profile (default: mixed_docs_code)")
    init_parser.set_defaults(func=cmd_init)

    # run command
    run_parser = subparsers.add_parser("run", help="Run kernel execution")
    run_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    run_parser.add_argument("--stage", "-s", type=int, choices=[1, 2, 3], help="Stage to run")
    run_parser.add_argument("--all", action="store_true", help="Run all stages")
    run_parser.add_argument("--kernels", "-k", help="Specific kernels (comma-separated)")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    run_parser.add_argument("--parallel", "-P", action="store_true",
                           help="Enable parallel execution for independent kernels")
    run_parser.add_argument("--workers", "-W", type=int, default=4,
                           help="Max parallel workers (default: 4)")
    run_parser.set_defaults(func=cmd_run)

    # status command
    status_parser = subparsers.add_parser("status", help="Show system and workspace status")
    status_parser.add_argument("--workspace", "-w", help="Workspace directory (optional)")
    status_parser.set_defaults(func=cmd_status)

    # rag command
    rag_parser = subparsers.add_parser("rag", help="Manage project RAG indexing")
    rag_parser.add_argument("--project", "-p", required=True, help="Project path")
    rag_parser.add_argument("--init", action="store_true", help="Initialize RAG for project")
    rag_parser.add_argument("--index", action="store_true", help="Start RAG indexing")
    rag_parser.add_argument("--status", action="store_true", help="Show RAG status")
    rag_parser.add_argument("--full", action="store_true", help="Full reindex (with --index)")
    rag_parser.add_argument("--profile", default="mixed_docs_code",
                           choices=["docs_only", "mixed_docs_code", "code_only"],
                           help="RAG profile (default: mixed_docs_code)")
    rag_parser.set_defaults(func=cmd_rag)

    # summary command
    summary_parser = subparsers.add_parser("summary", help="Get kernel summaries")
    summary_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    summary_parser.add_argument("--stage", "-s", type=int, choices=[1, 2, 3], help="Specific stage")
    summary_parser.set_defaults(func=cmd_summary)

    # list command
    list_parser = subparsers.add_parser("list", help="List available kernels")
    list_parser.add_argument("--stage", "-s", type=int, choices=[1, 2, 3], help="Filter by stage")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    setup_logging(args.verbose)

    args.func(args)


if __name__ == "__main__":
    main()
