#!/usr/bin/env python3
"""
KOAS Document Summarization Pipeline — Run document analysis with dual clustering.

This script provides a dedicated launcher for document summarization audits,
with support for:
- Dual clustering (Hierarchical + Leiden in parallel)
- Tutor-verified LLM summaries (Worker: Granite 3B, Tutor: Mistral 7B)
- LLM response caching with sovereignty attestation
- Final report generation at run root
- Pre-flight validation to catch errors early
- Proper artifact management (.KOAS/runs/<run_id>/)

Usage:
    # Initialize workspace for document analysis
    python -m ragix_kernels.run_doc_koas init --workspace ./audit/vdp --project /path/to/docs

    # Run all stages with caching
    python -m ragix_kernels.run_doc_koas run --workspace ./audit/vdp --all --cache

    # Run with custom models
    python -m ragix_kernels.run_doc_koas run --workspace ./audit/vdp --all \
        --worker-model granite3.1-moe:3b --tutor-model mistral:7b-instruct

    # Clear cache and re-run
    python -m ragix_kernels.run_doc_koas run --workspace ./audit/vdp --all --no-cache

    # Run pre-flight checks only
    python -m ragix_kernels.run_doc_koas preflight --workspace ./audit/vdp

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

import argparse
import json
import logging
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import KOAS components
from ragix_kernels.orchestrator import (
    Orchestrator,
    ManifestConfig,
    validate_manifest,
    get_system_info,
    progress_reporter,
)
from ragix_kernels.registry import KernelRegistry
from ragix_kernels.cache import LLMCache, get_model_digest
from ragix_kernels.preflight import (
    run_preflight_checks,
    print_preflight_report,
    RunConfig,
    LLMStatsCollector,
)

logger = logging.getLogger(__name__)

# Global LLM stats collector (initialized per run)
_llm_stats: Optional[LLMStatsCollector] = None

# =============================================================================
# Constants
# =============================================================================

# Default LLM models
DEFAULT_WORKER_MODEL = "granite3.1-moe:3b"
DEFAULT_TUTOR_MODEL = "mistral:7b-instruct"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"

# Document kernels by stage
DOC_STAGE1_KERNELS = ["doc_metadata", "doc_concepts", "doc_structure"]
DOC_STAGE2_KERNELS = [
    "doc_cluster",           # Hierarchical clustering
    "doc_cluster_leiden",    # Leiden clustering (parallel)
    "doc_cluster_reconcile", # Reconciliation
    "doc_extract",
    "doc_coverage",
    "doc_func_extract",
    "doc_quality",           # 5-dimension quality scorecard + MRI/SRI
]
DOC_STAGE3_KERNELS = [
    "doc_pyramid",
    "doc_summarize_tutored",  # Tutored summaries (replaces doc_summarize)
    "doc_compare",
    "doc_final_report",       # Final report at run root
]


# =============================================================================
# Document Manifest Generation
# =============================================================================

def generate_doc_manifest(
    workspace: Path,
    project_path: Path,
    project_name: str,
    author: str,
    worker_model: str = DEFAULT_WORKER_MODEL,
    tutor_model: str = DEFAULT_TUTOR_MODEL,
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    enable_cache: bool = True,
    language: str = "en",
) -> Path:
    """
    Generate a document-focused KOAS manifest.

    Args:
        workspace: Audit workspace directory
        project_path: Path to document project
        project_name: Project name
        author: Author name
        worker_model: Worker LLM model (fast, generates initial summaries)
        tutor_model: Tutor LLM model (verifies and corrects summaries)
        ollama_endpoint: Ollama API endpoint
        enable_cache: Enable LLM response caching
        language: Output language (en/fr)

    Returns:
        Path to generated manifest.yaml
    """
    manifest_content = f"""# KOAS Document Summarization Manifest
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

audit:
  name: "{project_name} Document Analysis"
  version: "1.2"
  date: "{datetime.now().strftime('%Y-%m-%d')}"
  author: "{author}"

project:
  name: "{project_name}"
  path: "{project_path}"
  language: "docs"
  exclude:
    - "**/node_modules/**"
    - "**/.git/**"
    - "**/build/**"

# Stage 1: Document Collection
stage1:
  doc_metadata:
    enabled: true
    options:
      include_statistics: true

  doc_concepts:
    enabled: true
    options:
      min_frequency: 2
      max_concepts: 150

  doc_structure:
    enabled: true
    options:
      detect_headings: true
      extract_toc: true

# Stage 2: Document Analysis (Dual Clustering)
stage2:
  doc_cluster:
    enabled: true
    options:
      method: "hierarchical"
      n_clusters: auto
      min_cluster_size: 2

  doc_cluster_leiden:
    enabled: true
    options:
      # Note: resolution > 1.0 can cause exponential computation on dense graphs
      resolutions: [0.1, 0.5, 1.0]
      min_community_size: 2
      seed: 42

  doc_cluster_reconcile:
    enabled: true
    options:
      high_agreement_threshold: 0.8
      low_agreement_threshold: 0.5

  doc_extract:
    enabled: true
    options:
      sentences_per_concept: 3
      sentences_per_file: 5
      quality_threshold: 0.4

  doc_coverage:
    enabled: true

  doc_func_extract:
    enabled: true
    options:
      extract_functionalities: true
      include_spd: true

# Stage 3: Synthesis (with Tutor Verification)
stage3:
  doc_pyramid:
    enabled: true
    options:
      levels: 4
      include_markdown: true

  doc_summarize_tutored:
    enabled: true
    options:
      worker_model: "{worker_model}"
      tutor_model: "{tutor_model}"
      temperature: 0.3
      max_tokens: 512
      verify_all: true

  doc_compare:
    enabled: true
    options:
      similarity_threshold: 0.8
      detect_contradictions: true

  doc_visualize:
    enabled: true
    options:
      output_dir: "assets"
      formats: ["svg", "png", "pdf"]
      dpi: 150
      style: "publication"

  doc_final_report:
    enabled: true
    options:
      include_appendices: true
      include_sovereignty: true
      output_formats: ["markdown", "json"]

# Output Configuration
output:
  format: "markdown"
  template: "document_audit"
  language: "{language}"
  frontmatter:
    title: "{project_name} Document Analysis Report"
    author: "{author}"
    generated_by: "KOAS v1.2"

# LLM Configuration
llm:
  worker:
    model: "{worker_model}"
    endpoint: "{ollama_endpoint}"
    temperature: 0.3
    max_tokens: 512

  tutor:
    model: "{tutor_model}"
    endpoint: "{ollama_endpoint}"
    temperature: 0.2
    max_tokens: 768

  cache:
    enabled: {str(enable_cache).lower()}
    directory: ".KOAS/cache"

# Sovereignty Settings
sovereignty:
  mode: "local"
  audit_trail: "required"
  attestation:
    include_hostname: true
    include_user: true
    include_timestamps: true
    include_model_digests: true
"""

    manifest_path = workspace / "manifest.yaml"
    manifest_path.write_text(manifest_content)
    return manifest_path


# =============================================================================
# Cache Management
# =============================================================================

def setup_cache(workspace: Path, endpoint: str = DEFAULT_OLLAMA_ENDPOINT) -> LLMCache:
    """Initialize LLM cache for the workspace."""
    cache_dir = workspace / ".KOAS" / "cache"
    return LLMCache(cache_dir, endpoint=endpoint)


def get_cache_stats(workspace: Path) -> Dict[str, Any]:
    """Get cache statistics for a workspace."""
    cache_dir = workspace / ".KOAS" / "cache"
    if not cache_dir.exists():
        return {"status": "no_cache"}

    cache = LLMCache(cache_dir)
    return {
        "stats": cache.get_stats(),
        "sovereignty": cache.get_sovereignty_summary(),
    }


def clear_cache(workspace: Path, model: Optional[str] = None):
    """Clear LLM cache for a workspace."""
    cache_dir = workspace / ".KOAS" / "cache"
    if cache_dir.exists():
        cache = LLMCache(cache_dir)
        cache.clear(model)
        logger.info(f"Cache cleared for workspace {workspace}")


# =============================================================================
# Sovereignty Attestation
# =============================================================================

def capture_sovereignty_attestation(
    workspace: Path,
    worker_model: str,
    tutor_model: str,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
) -> Dict[str, Any]:
    """
    Capture full sovereignty attestation for the run.

    Returns:
        Attestation dictionary with hostname, user, timestamps, model digests
    """
    worker_digest = get_model_digest(worker_model, endpoint)
    tutor_digest = get_model_digest(tutor_model, endpoint)

    attestation = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "endpoint": endpoint,
        "local": "127.0.0.1" in endpoint or "localhost" in endpoint,
        "models": {
            "worker": {
                "name": worker_model,
                "digest": worker_digest,
            },
            "tutor": {
                "name": tutor_model,
                "digest": tutor_digest,
            },
        },
        "workspace": str(workspace),
        "koas_version": "1.2",
    }

    return attestation


def save_sovereignty_attestation(workspace: Path, attestation: Dict[str, Any]):
    """Save sovereignty attestation to workspace."""
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    attestation_file = logs_dir / "sovereignty_attestation.json"
    attestation_file.write_text(json.dumps(attestation, indent=2))

    logger.info(f"Sovereignty attestation saved to {attestation_file}")


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_doc_pipeline(
    workspace: Path,
    stages: List[int],
    parallel: bool = True,
    max_workers: int = 4,
    enable_cache: bool = True,
    worker_model: str = DEFAULT_WORKER_MODEL,
    tutor_model: str = DEFAULT_TUTOR_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    quiet: bool = False,
    verbose: bool = False,
    skip_preflight: bool = False,
) -> Dict[str, Any]:
    """
    Run the document summarization pipeline.

    Args:
        workspace: Audit workspace directory
        stages: List of stages to run [1, 2, 3]
        parallel: Enable parallel execution
        max_workers: Maximum parallel workers
        enable_cache: Enable LLM caching
        worker_model: Worker model name
        tutor_model: Tutor model name
        endpoint: Ollama endpoint
        quiet: Suppress progress output
        verbose: Verbose logging
        skip_preflight: Skip pre-flight validation

    Returns:
        Dictionary with execution results
    """
    global _llm_stats

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # =========================================================================
    # Step 1: Pre-flight validation
    # =========================================================================
    if not skip_preflight:
        print("\n[Pre-flight] Running validation checks...")
        preflight_result = run_preflight_checks(
            workspace=workspace,
            worker_model=worker_model,
            tutor_model=tutor_model,
            endpoint=endpoint,
            enable_cache=enable_cache,
        )

        if not preflight_result.passed:
            print_preflight_report(preflight_result, verbose=verbose)
            raise RuntimeError(
                f"Pre-flight validation failed with {len(preflight_result.errors)} error(s). "
                "Fix the errors above before running the pipeline."
            )

        if preflight_result.warnings and not quiet:
            for w in preflight_result.warnings:
                print(f"  WARNING: {w}")

        print("[Pre-flight] All checks passed!")

    # =========================================================================
    # Step 2: Create run configuration
    # =========================================================================
    # Load manifest to get project path
    orchestrator = Orchestrator(workspace)
    if not orchestrator.manifest:
        raise RuntimeError(f"No manifest.yaml found in {workspace}")

    project_path = orchestrator.manifest.project_path

    # Create run configuration with unique run ID
    run_config = RunConfig.create(
        workspace=workspace,
        project_path=project_path,
        worker_model=worker_model,
        tutor_model=tutor_model,
        endpoint=endpoint,
    )

    # Create run directory structure
    run_config.create_directories()
    run_config.save_manifest_copy(workspace / "manifest.yaml")

    print(f"\n[Run ID] {run_config.run_id}")
    print(f"[Output] {run_config.run_dir}")

    # Initialize LLM statistics collector
    _llm_stats = LLMStatsCollector(run_config.run_id)

    # =========================================================================
    # Step 3: Capture sovereignty attestation
    # =========================================================================
    attestation = capture_sovereignty_attestation(
        workspace, worker_model, tutor_model, endpoint
    )
    # Save to run directory
    save_sovereignty_attestation(run_config.run_dir, attestation)

    # Setup cache if enabled (at project level)
    if enable_cache:
        cache = LLMCache(run_config.cache_dir, endpoint=endpoint)
        logger.info(f"LLM cache enabled: {cache.cache_dir}")

    # Validate manifest
    errors = validate_manifest(orchestrator.manifest)
    if errors:
        raise RuntimeError(f"Manifest validation errors: {errors}")

    results = {
        "run_id": run_config.run_id,
        "run_dir": str(run_config.run_dir),
        "workspace": str(workspace),
        "stages_run": stages,
        "attestation": attestation,
        "stage_results": {},
    }

    # =========================================================================
    # Step 4: Run stages
    # =========================================================================
    for stage in stages:
        stage_name = {
            1: "Document Collection",
            2: "Document Analysis (Dual Clustering)",
            3: "Synthesis & Final Report",
        }.get(stage, f"Stage {stage}")

        print(f"\n{'='*60}")
        print(f"Stage {stage}: {stage_name}")
        print(f"{'='*60}\n")

        try:
            # For stage 2, ensure parallel execution of clustering
            stage_parallel = parallel
            if stage == 2:
                # Force parallel for dual clustering
                stage_parallel = True
                logger.info("[Stage 2] Running hierarchical and Leiden clustering in parallel")

            stage_results = orchestrator.run_stage(
                stage,
                progress_callback=progress_reporter if not quiet else None,
                parallel=stage_parallel,
                max_workers=max_workers,
            )

            succeeded = sum(1 for r in stage_results if r.success)
            failed = sum(1 for r in stage_results if not r.success)
            total_time = sum(r.execution_time_ms for r in stage_results)

            results["stage_results"][stage] = {
                "succeeded": succeeded,
                "failed": failed,
                "total_time_ms": total_time,
                "kernels": [r.kernel_name for r in stage_results],
            }

            mode = "parallel" if stage_parallel else "sequential"
            print(f"\nStage {stage} complete ({mode}): {succeeded} succeeded, {failed} failed ({total_time}ms)")

        except Exception as e:
            logger.error(f"Error in stage {stage}: {e}")
            results["stage_results"][stage] = {"error": str(e)}
            raise

    # =========================================================================
    # Step 5: Copy artifacts to run directory
    # =========================================================================
    _copy_artifacts_to_run_dir(workspace, run_config.run_dir, project_path)

    # Save LLM statistics
    if _llm_stats:
        _llm_stats.save(run_config.run_dir / "logs")
        results["llm_stats"] = _llm_stats.get_summary()

    # Add cache stats if enabled
    if enable_cache:
        results["cache_stats"] = get_cache_stats(workspace)

    # Save execution summary
    summary_file = run_config.run_dir / "logs" / "execution_summary.json"
    summary_file.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Run ID: {run_config.run_id}")
    print(f"  Output: {run_config.run_dir}")
    print(f"  Final report: {run_config.run_dir / 'final_report.md'}")
    print(f"  Latest symlink: {project_path / '.KOAS' / 'latest'}")
    print(f"{'='*60}")

    return results


def _copy_artifacts_to_run_dir(workspace: Path, run_dir: Path, project_path: Path = None):
    """Copy stage outputs to run directory with proper organization.

    Args:
        workspace: Workspace directory with stage outputs
        run_dir: Target run directory (.KOAS/runs/<run_id>/)
        project_path: Project directory where doc_final_report writes outputs
    """
    # Copy stage outputs
    for stage in [1, 2, 3]:
        src_dir = workspace / f"stage{stage}"
        dst_dir = run_dir / f"stage{stage}"

        if src_dir.exists():
            # Copy JSON files
            for json_file in src_dir.glob("*.json"):
                shutil.copy(json_file, dst_dir / json_file.name)

            # Copy summary files
            for txt_file in src_dir.glob("*.txt"):
                shutil.copy(txt_file, dst_dir / txt_file.name)

            # Copy markdown files (doc_report.md, audit_report.md, etc.)
            for md_file in src_dir.glob("*.md"):
                shutil.copy(md_file, dst_dir / md_file.name)

    # Copy assets (visualizations) from workspace
    assets_src = workspace / "assets"
    assets_dst = run_dir / "assets"
    if assets_src.exists():
        # Copy all files in assets/
        for asset_file in assets_src.glob("*"):
            if asset_file.is_file():
                shutil.copy(asset_file, assets_dst / asset_file.name)
            elif asset_file.is_dir():
                # Copy subdirectories (e.g., domain_clouds/)
                subdir_dst = assets_dst / asset_file.name
                subdir_dst.mkdir(parents=True, exist_ok=True)
                for subfile in asset_file.glob("*"):
                    if subfile.is_file():
                        shutil.copy(subfile, subdir_dst / subfile.name)

    # Copy final report and appendices from project path
    # (doc_final_report kernel writes to project_path, not workspace)
    if project_path:
        # Final report
        project_report = project_path / "final_report.md"
        if project_report.exists():
            shutil.copy(project_report, run_dir / "final_report.md")

        # Appendices
        project_appendices = project_path / "appendices"
        appendices_dst = run_dir / "appendices"
        if project_appendices.exists():
            for md_file in project_appendices.glob("*.md"):
                shutil.copy(md_file, appendices_dst / md_file.name)

    # Fallback: check workspace paths (for backwards compatibility)
    if not (run_dir / "final_report.md").exists():
        stage3_report = workspace / "stage3" / "doc_final_report.md"
        if stage3_report.exists():
            shutil.copy(stage3_report, run_dir / "final_report.md")

    if not list((run_dir / "appendices").glob("*.md")):
        appendices_src = workspace / "stage3" / "appendices"
        appendices_dst = run_dir / "appendices"
        if appendices_src.exists():
            for md_file in appendices_src.glob("*.md"):
                shutil.copy(md_file, appendices_dst / md_file.name)

    logger.info(f"Artifacts copied to {run_dir}")


def _copy_final_report_to_root(workspace: Path):
    """Copy final report from stage3 to workspace root."""
    stage3_report = workspace / "stage3" / "doc_final_report.md"
    root_report = workspace / "FINAL_REPORT.md"

    if stage3_report.exists():
        shutil.copy(stage3_report, root_report)
        logger.info(f"Final report copied to {root_report}")

    # Also copy JSON version
    stage3_json = workspace / "stage3" / "doc_final_report.json"
    root_json = workspace / "FINAL_REPORT.json"

    if stage3_json.exists():
        shutil.copy(stage3_json, root_json)


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(args):
    """Handle init command."""
    workspace = Path(args.workspace)
    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)

    project_name = args.name or project_path.name

    # Create directory structure
    workspace.mkdir(parents=True, exist_ok=True)
    for subdir in ["stage1", "stage2", "stage3", "logs", "assets", ".KOAS/cache"]:
        (workspace / subdir).mkdir(parents=True, exist_ok=True)

    # Generate manifest
    manifest_path = generate_doc_manifest(
        workspace=workspace,
        project_path=project_path,
        project_name=project_name,
        author=args.author,
        worker_model=args.worker_model,
        tutor_model=args.tutor_model,
        ollama_endpoint=args.endpoint,
        enable_cache=not args.no_cache,
        language=args.language,
    )

    print(f"Initialized document analysis workspace: {workspace}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Worker model: {args.worker_model}")
    print(f"  Tutor model: {args.tutor_model}")
    print(f"  Caching: {'enabled' if not args.no_cache else 'disabled'}")
    print(f"\nNext: Run the pipeline with:")
    print(f"  python -m ragix_kernels.run_doc_koas run --workspace {workspace} --all")


def cmd_run(args):
    """Handle run command."""
    workspace = Path(args.workspace)

    if not workspace.exists():
        print(f"Error: Workspace does not exist: {workspace}")
        print("Run 'init' first to create a workspace.")
        sys.exit(1)

    # Determine stages to run
    if args.all:
        stages = [1, 2, 3]
    elif args.stage:
        stages = [args.stage]
    else:
        stages = [1]

    try:
        results = run_doc_pipeline(
            workspace=workspace,
            stages=stages,
            parallel=not args.sequential,
            max_workers=args.workers,
            enable_cache=not args.no_cache,
            worker_model=args.worker_model,
            tutor_model=args.tutor_model,
            endpoint=args.endpoint,
            quiet=args.quiet,
            verbose=args.verbose,
            skip_preflight=getattr(args, 'skip_preflight', False),
        )

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_preflight(args):
    """Handle preflight command - run validation checks only."""
    workspace = Path(args.workspace)

    if not workspace.exists():
        print(f"Error: Workspace does not exist: {workspace}")
        sys.exit(1)

    result = run_preflight_checks(
        workspace=workspace,
        worker_model=args.worker_model,
        tutor_model=args.tutor_model,
        endpoint=args.endpoint,
        enable_cache=not args.no_cache,
        check_models=not args.skip_models,
    )

    print_preflight_report(result, verbose=args.verbose)

    if not result.passed:
        sys.exit(1)


def cmd_status(args):
    """Handle status command."""
    print("="*60)
    print("KOAS Document Analysis Status")
    print("="*60)

    # System info
    info = get_system_info()

    print("\n[System]")
    print(f"  Python: {info.get('python_version', 'N/A')}")
    print(f"  Platform: {info.get('platform', 'N/A')}")

    # Ollama
    print("\n[Ollama LLM]")
    if info.get('ollama_available'):
        models = info.get('ollama_models', [])
        print(f"  Running with {len(models)} model(s)")

        # Check for recommended models
        has_worker = any(DEFAULT_WORKER_MODEL in m for m in models)
        has_tutor = any(DEFAULT_TUTOR_MODEL.split(':')[0] in m for m in models)

        if has_worker:
            print(f"    Worker model ready: {DEFAULT_WORKER_MODEL}")
        else:
            print(f"    Worker model missing: {DEFAULT_WORKER_MODEL}")

        if has_tutor:
            print(f"    Tutor model ready: {DEFAULT_TUTOR_MODEL}")
        else:
            print(f"    Tutor model missing: {DEFAULT_TUTOR_MODEL}")
    else:
        print("  Not running (start with 'ollama serve')")

    # Workspace status
    if args.workspace:
        workspace = Path(args.workspace)
        print(f"\n[Workspace: {workspace}]")

        if workspace.exists():
            manifest = workspace / "manifest.yaml"
            if manifest.exists():
                print("  Manifest: found")
                with open(manifest) as f:
                    config = yaml.safe_load(f)
                print(f"  Project: {config.get('project', {}).get('name', 'N/A')}")

            # Cache stats
            cache_stats = get_cache_stats(workspace)
            if cache_stats.get("stats"):
                stats = cache_stats["stats"]
                print(f"\n[LLM Cache]")
                print(f"  Hits: {stats.get('hits', 0)}")
                print(f"  Misses: {stats.get('misses', 0)}")
                print(f"  Hit rate: {stats.get('hit_rate', 0):.1%}")

            # Stage completion
            for stage in [1, 2, 3]:
                stage_dir = workspace / f"stage{stage}"
                if stage_dir.exists():
                    files = list(stage_dir.glob("*.json"))
                    kernel_files = [f.stem for f in files if not f.stem.startswith("_")]
                    if kernel_files:
                        print(f"\n  Stage {stage}: {len(kernel_files)} kernel(s)")
                        for k in kernel_files[:5]:
                            print(f"    - {k}")
                        if len(kernel_files) > 5:
                            print(f"    ... and {len(kernel_files) - 5} more")

            # Final report
            final_report = workspace / "FINAL_REPORT.md"
            if final_report.exists():
                print(f"\n  Final report: {final_report}")
        else:
            print("  Workspace does not exist")

    # Available doc kernels
    print("\n[Document Kernels]")
    KernelRegistry.discover()
    for stage, kernels in [(1, DOC_STAGE1_KERNELS), (2, DOC_STAGE2_KERNELS), (3, DOC_STAGE3_KERNELS)]:
        available = [k for k in kernels if k in KernelRegistry.list_all()]
        print(f"  Stage {stage}: {len(available)}/{len(kernels)} available")

    print("\n" + "="*60)


def cmd_cache(args):
    """Handle cache command."""
    workspace = Path(args.workspace)

    if args.clear:
        clear_cache(workspace, args.model)
        print(f"Cache cleared for {workspace}")
    else:
        stats = get_cache_stats(workspace)
        if stats.get("status") == "no_cache":
            print("No cache found for this workspace")
        else:
            print(f"Cache statistics for {workspace}:")
            print(json.dumps(stats, indent=2))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KOAS Document Summarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize workspace
  python -m ragix_kernels.run_doc_koas init \\
    --workspace ./audit/vdp \\
    --project /path/to/docs

  # Run all stages with caching
  python -m ragix_kernels.run_doc_koas run \\
    --workspace ./audit/vdp --all

  # Run with custom models
  python -m ragix_kernels.run_doc_koas run \\
    --workspace ./audit/vdp --all \\
    --worker-model granite3.1-moe:3b \\
    --tutor-model mistral:7b-instruct

  # Show status
  python -m ragix_kernels.run_doc_koas status --workspace ./audit/vdp

  # Clear cache
  python -m ragix_kernels.run_doc_koas cache --workspace ./audit/vdp --clear
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize document analysis workspace")
    init_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    init_parser.add_argument("--project", "-p", required=True, help="Document project path")
    init_parser.add_argument("--name", "-n", help="Project name")
    init_parser.add_argument("--author", "-a", default="Unknown", help="Author name")
    init_parser.add_argument("--worker-model", default=DEFAULT_WORKER_MODEL, help="Worker LLM model")
    init_parser.add_argument("--tutor-model", default=DEFAULT_TUTOR_MODEL, help="Tutor LLM model")
    init_parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT, help="Ollama endpoint")
    init_parser.add_argument("--no-cache", action="store_true", help="Disable LLM caching")
    init_parser.add_argument("--language", "-l", default="en", choices=["en", "fr"], help="Output language")
    init_parser.set_defaults(func=cmd_init)

    # run command
    run_parser = subparsers.add_parser("run", help="Run document analysis pipeline")
    run_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    run_parser.add_argument("--stage", "-s", type=int, choices=[1, 2, 3], help="Run specific stage")
    run_parser.add_argument("--all", action="store_true", help="Run all stages")
    run_parser.add_argument("--sequential", action="store_true", help="Disable parallel execution")
    run_parser.add_argument("--workers", "-W", type=int, default=4, help="Max parallel workers")
    run_parser.add_argument("--worker-model", default=DEFAULT_WORKER_MODEL, help="Worker LLM model")
    run_parser.add_argument("--tutor-model", default=DEFAULT_TUTOR_MODEL, help="Tutor LLM model")
    run_parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT, help="Ollama endpoint")
    run_parser.add_argument("--no-cache", action="store_true", help="Disable LLM caching")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    run_parser.add_argument("--skip-preflight", action="store_true", help="Skip pre-flight checks")
    run_parser.set_defaults(func=cmd_run)

    # preflight command
    preflight_parser = subparsers.add_parser("preflight", help="Run pre-flight validation checks only")
    preflight_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    preflight_parser.add_argument("--worker-model", default=DEFAULT_WORKER_MODEL, help="Worker LLM model")
    preflight_parser.add_argument("--tutor-model", default=DEFAULT_TUTOR_MODEL, help="Tutor LLM model")
    preflight_parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT, help="Ollama endpoint")
    preflight_parser.add_argument("--no-cache", action="store_true", help="Disable LLM caching")
    preflight_parser.add_argument("--skip-models", action="store_true", help="Skip LLM model checks")
    preflight_parser.set_defaults(func=cmd_preflight)

    # status command
    status_parser = subparsers.add_parser("status", help="Show system and workspace status")
    status_parser.add_argument("--workspace", "-w", help="Workspace directory")
    status_parser.set_defaults(func=cmd_status)

    # cache command
    cache_parser = subparsers.add_parser("cache", help="Manage LLM cache")
    cache_parser.add_argument("--workspace", "-w", required=True, help="Workspace directory")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
    cache_parser.add_argument("--model", "-m", help="Clear cache for specific model only")
    cache_parser.set_defaults(func=cmd_cache)

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
