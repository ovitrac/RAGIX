"""
KOAS Pre-flight Validation — Catch configuration errors before expensive LLM runs.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18

Validates:
- Manifest configuration and paths
- Ollama availability and required models
- RAG project status and indexing
- Cache configuration
- Kernel dependencies

Usage:
    from ragix_kernels.preflight import run_preflight_checks, PreflightResult

    result = run_preflight_checks(workspace, worker_model="granite3.1-moe:3b")
    if not result.passed:
        for error in result.errors:
            print(f"ERROR: {error}")
        sys.exit(1)
"""

import json
import logging
import os
import socket
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of pre-flight validation checks."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, key: str, value: Any):
        self.info[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


@dataclass
class RunConfig:
    """Configuration for a KOAS run."""
    run_id: str
    workspace: Path
    project_path: Path
    run_dir: Path
    cache_dir: Path
    worker_model: str
    tutor_model: str
    endpoint: str
    timestamp: str

    @classmethod
    def create(
        cls,
        workspace: Path,
        project_path: Path,
        worker_model: str = "granite3.1-moe:3b",
        tutor_model: str = "mistral:7b-instruct",
        endpoint: str = "http://127.0.0.1:11434",
    ) -> "RunConfig":
        """Create a new run configuration with unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{secrets.token_hex(3)}"

        # Run outputs go to .KOAS/runs/<run_id>/
        run_dir = project_path / ".KOAS" / "runs" / run_id

        # Cache is shared at project level
        cache_dir = project_path / ".KOAS" / "cache"

        return cls(
            run_id=run_id,
            workspace=workspace,
            project_path=project_path,
            run_dir=run_dir,
            cache_dir=cache_dir,
            worker_model=worker_model,
            tutor_model=tutor_model,
            endpoint=endpoint,
            timestamp=timestamp,
        )

    def create_directories(self):
        """Create all required directories for the run."""
        # Create run directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["stage1", "stage2", "stage3", "appendices", "assets", "logs"]:
            (self.run_dir / subdir).mkdir(exist_ok=True)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create/update 'latest' symlink
        latest_link = self.project_path / ".KOAS" / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            latest_link.rename(latest_link.with_suffix(".bak"))
        latest_link.symlink_to(f"runs/{self.run_id}")

    def save_manifest_copy(self, manifest_path: Path):
        """Copy manifest to run directory for reproducibility."""
        if manifest_path.exists():
            import shutil
            shutil.copy(manifest_path, self.run_dir / "manifest.yaml")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "project_path": str(self.project_path),
            "run_dir": str(self.run_dir),
            "cache_dir": str(self.cache_dir),
            "worker_model": self.worker_model,
            "tutor_model": self.tutor_model,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
        }


def check_manifest(workspace: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate manifest.yaml exists and is well-formed.

    Returns:
        Tuple of (valid, errors, config_dict)
    """
    errors = []
    config = {}

    manifest_path = workspace / "manifest.yaml"

    if not manifest_path.exists():
        errors.append(f"manifest.yaml not found at {manifest_path}")
        return False, errors, config

    try:
        with open(manifest_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in manifest: {e}")
        return False, errors, config

    # Check required sections
    if "project" not in config:
        errors.append("Missing 'project' section in manifest")
    else:
        project = config["project"]
        if "path" not in project:
            errors.append("Missing 'project.path' in manifest")
        else:
            # Resolve project path
            project_path = Path(project["path"])
            if not project_path.is_absolute():
                project_path = workspace / project_path

            if not project_path.exists():
                errors.append(f"Project path does not exist: {project_path}")
            else:
                config["_resolved_project_path"] = str(project_path)

    return len(errors) == 0, errors, config


def check_ollama(
    endpoint: str = "http://127.0.0.1:11434",
    required_models: Optional[List[str]] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check Ollama availability and required models.

    Returns:
        Tuple of (available, errors, info)
    """
    errors = []
    info = {"endpoint": endpoint, "available": False, "models": []}

    try:
        import httpx
        response = httpx.get(f"{endpoint}/api/tags", timeout=5)

        if response.status_code != 200:
            errors.append(f"Ollama returned status {response.status_code}")
            return False, errors, info

        data = response.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        info["available"] = True
        info["models"] = models

        # Check required models
        if required_models:
            for model in required_models:
                # Check exact match or base name match
                model_base = model.split(":")[0]
                found = any(m == model or m.startswith(model_base + ":") for m in models)
                if not found:
                    errors.append(f"Required model not found: {model}")
                    errors.append(f"  Available models: {models}")
                    errors.append(f"  To install: ollama pull {model}")

    except ImportError:
        errors.append("httpx not installed (pip install httpx)")
    except Exception as e:
        errors.append(f"Cannot connect to Ollama at {endpoint}: {e}")
        errors.append("  Start Ollama with: ollama serve")

    return len(errors) == 0, errors, info


def check_rag_project(project_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check RAG project status.

    Returns:
        Tuple of (ready, errors, info)
    """
    errors = []
    info = {"initialized": False, "indexed": False, "chunks": 0, "files": 0}

    rag_dir = project_path / ".RAG"

    if not rag_dir.exists():
        errors.append(f"RAG not initialized for project: {project_path}")
        errors.append("  Initialize with: ragix-cli --workspace <path> init")
        return False, errors, info

    info["initialized"] = True

    try:
        from ragix_core.rag_project import RAGProject

        project = RAGProject(project_path)

        if not project.is_initialized():
            errors.append("RAG directory exists but not properly initialized")
            return False, errors, info

        status = project.get_status()

        # Extract state from the nested structure
        state = status.get("state", {})
        info["indexed"] = state.get("status") == "completed"
        info["chunks"] = state.get("chunks_indexed", 0)
        info["files"] = state.get("files_indexed", 0)
        info["status"] = state.get("status", "unknown")

        if not info["indexed"] or info["chunks"] == 0:
            errors.append("RAG index is empty or incomplete")
            errors.append("  Run indexing with: ragix-cli --workspace <path> index")
            return False, errors, info

    except ImportError:
        # RAG module not available, but directory exists - assume OK
        logger.warning("ragix_core not available, skipping detailed RAG check")

    return len(errors) == 0, errors, info


def check_cache_config(
    project_path: Path,
    cache_enabled: bool = True
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check cache configuration.

    Returns:
        Tuple of (valid, warnings, info)
    """
    warnings = []
    info = {"enabled": cache_enabled, "path": None, "entries": 0}

    if not cache_enabled:
        warnings.append("LLM caching is disabled - all requests will be sent to LLM")
        return True, warnings, info

    cache_dir = project_path / ".KOAS" / "cache"
    info["path"] = str(cache_dir)

    if cache_dir.exists():
        # Count existing entries
        llm_responses = cache_dir / "llm_responses"
        if llm_responses.exists():
            entries = sum(1 for _ in llm_responses.rglob("*.json"))
            info["entries"] = entries
            if entries > 0:
                info["has_cache"] = True

    return True, warnings, info


def check_kernels(stages: List[int] = [1, 2, 3]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Check kernel availability.

    Returns:
        Tuple of (valid, errors, info)
    """
    errors = []
    info = {"available": {}, "missing": []}

    try:
        from ragix_kernels.registry import KernelRegistry
        KernelRegistry.discover()

        for stage in stages:
            stage_kernels = KernelRegistry.list_stage(stage)
            info["available"][stage] = stage_kernels

        # Check doc kernels specifically
        doc_kernels = [
            "doc_metadata", "doc_concepts", "doc_structure",
            "doc_cluster", "doc_cluster_leiden", "doc_extract",
            "doc_pyramid", "doc_summarize_tutored", "doc_compare", "doc_final_report"
        ]

        for kernel in doc_kernels:
            if kernel not in KernelRegistry.list_all():
                errors.append(f"Missing required kernel: {kernel}")
                info["missing"].append(kernel)

    except ImportError as e:
        errors.append(f"Cannot import kernel registry: {e}")

    return len(errors) == 0, errors, info


def run_preflight_checks(
    workspace: Path,
    worker_model: str = "granite3.1-moe:3b",
    tutor_model: str = "mistral:7b-instruct",
    endpoint: str = "http://127.0.0.1:11434",
    enable_cache: bool = True,
    check_models: bool = True,
) -> PreflightResult:
    """
    Run all pre-flight validation checks.

    Args:
        workspace: Path to KOAS workspace
        worker_model: Worker LLM model name
        tutor_model: Tutor LLM model name
        endpoint: Ollama endpoint
        enable_cache: Whether caching is enabled
        check_models: Whether to verify LLM models are available

    Returns:
        PreflightResult with all validation results
    """
    result = PreflightResult(passed=True)

    logger.info("[preflight] Starting pre-flight checks...")

    # 1. Check manifest
    logger.info("[preflight] Checking manifest...")
    valid, errors, config = check_manifest(workspace)
    if not valid:
        for e in errors:
            result.add_error(f"[manifest] {e}")
    else:
        result.add_info("manifest", config)
        result.add_info("project_path", config.get("_resolved_project_path"))

    # 2. Check Ollama and models
    if check_models:
        logger.info("[preflight] Checking Ollama...")
        valid, errors, info = check_ollama(
            endpoint,
            required_models=[worker_model, tutor_model]
        )
        if not valid:
            for e in errors:
                result.add_error(f"[ollama] {e}")
        else:
            result.add_info("ollama", info)

    # 3. Check RAG project
    if config.get("_resolved_project_path"):
        logger.info("[preflight] Checking RAG project...")
        project_path = Path(config["_resolved_project_path"])
        valid, errors, info = check_rag_project(project_path)
        if not valid:
            for e in errors:
                result.add_error(f"[rag] {e}")
        else:
            result.add_info("rag", info)

        # 4. Check cache
        logger.info("[preflight] Checking cache configuration...")
        valid, warnings, info = check_cache_config(project_path, enable_cache)
        for w in warnings:
            result.add_warning(f"[cache] {w}")
        result.add_info("cache", info)

    # 5. Check kernels
    logger.info("[preflight] Checking kernels...")
    valid, errors, info = check_kernels()
    if not valid:
        for e in errors:
            result.add_error(f"[kernels] {e}")
    else:
        result.add_info("kernels", info)

    # Final status
    if result.passed:
        logger.info("[preflight] All checks passed!")
    else:
        logger.error(f"[preflight] {len(result.errors)} error(s) found")

    return result


def print_preflight_report(result: PreflightResult, verbose: bool = False):
    """Print formatted pre-flight report."""
    print("\n" + "=" * 60)
    print("KOAS Pre-flight Check Report")
    print("=" * 60)

    # Status
    if result.passed:
        print("\n[STATUS] All checks PASSED")
    else:
        print(f"\n[STATUS] FAILED - {len(result.errors)} error(s)")

    # Errors
    if result.errors:
        print("\n[ERRORS]")
        for e in result.errors:
            print(f"  ERROR: {e}")

    # Warnings
    if result.warnings:
        print("\n[WARNINGS]")
        for w in result.warnings:
            print(f"  WARNING: {w}")

    # Info (verbose)
    if verbose and result.info:
        print("\n[INFO]")
        for key, value in result.info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)


# =============================================================================
# LLM Statistics Tracking
# =============================================================================

@dataclass
class LLMStats:
    """Statistics for LLM usage tracking."""
    model: str
    role: str  # "worker" or "tutor"
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    time_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    def record_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        duration_s: float,
        cached: bool = False
    ):
        """Record a single LLM request."""
        self.requests += 1
        self.tokens_in += prompt_tokens
        self.tokens_out += completion_tokens
        self.time_s += duration_s
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_error(self):
        """Record an LLM error."""
        self.errors += 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "role": self.role,
            "requests": self.requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "time_s": round(self.time_s, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "errors": self.errors,
        }


class LLMStatsCollector:
    """Collect LLM statistics across a KOAS run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.stats: Dict[str, LLMStats] = {}
        self.start_time = datetime.now()

    def get_or_create(self, model: str, role: str) -> LLMStats:
        """Get or create stats for a model/role combination."""
        key = f"{role}:{model}"
        if key not in self.stats:
            self.stats[key] = LLMStats(model=model, role=role)
        return self.stats[key]

    def record_worker_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_s: float,
        cached: bool = False
    ):
        """Record a worker LLM request."""
        stats = self.get_or_create(model, "worker")
        stats.record_request(prompt_tokens, completion_tokens, duration_s, cached)

    def record_tutor_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_s: float,
        cached: bool = False
    ):
        """Record a tutor LLM request."""
        stats = self.get_or_create(model, "tutor")
        stats.record_request(prompt_tokens, completion_tokens, duration_s, cached)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_requests = sum(s.requests for s in self.stats.values())
        total_tokens_in = sum(s.tokens_in for s in self.stats.values())
        total_tokens_out = sum(s.tokens_out for s in self.stats.values())
        total_time = sum(s.time_s for s in self.stats.values())
        total_cache_hits = sum(s.cache_hits for s in self.stats.values())
        total_cache_misses = sum(s.cache_misses for s in self.stats.values())

        return {
            "run_id": self.run_id,
            "duration_s": (datetime.now() - self.start_time).total_seconds(),
            "by_model": {k: v.to_dict() for k, v in self.stats.items()},
            "totals": {
                "requests": total_requests,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "tokens_total": total_tokens_in + total_tokens_out,
                "time_s": round(total_time, 2),
                "cache_hits": total_cache_hits,
                "cache_misses": total_cache_misses,
                "cache_hit_rate": round(
                    total_cache_hits / (total_cache_hits + total_cache_misses)
                    if (total_cache_hits + total_cache_misses) > 0 else 0.0,
                    3
                ),
            },
        }

    def save(self, output_dir: Path):
        """Save statistics to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_file = output_dir / "llm_stats.json"
        stats_file.write_text(json.dumps(self.get_summary(), indent=2))
        logger.info(f"LLM statistics saved to {stats_file}")
