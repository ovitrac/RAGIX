#!/usr/bin/env python3
"""
Regression Test Runner for Interpreter-Tutor Benchmarks
========================================================

Runs benchmarks and compares against stored baselines to detect regressions.

Features:
- Fast mode with granite3.1-moe:3b (~2GB, fastest)
- Full mode with deepseek-r1:14b (gold standard)
- Baseline storage and comparison
- CI-friendly exit codes and output
- Retestable: any historical result can be validated

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default models for different test modes
FAST_MODEL = "granite3.1-moe:3b"      # ~2GB, fastest inference
STANDARD_MODEL = "qwen2.5-coder:7b"   # ~5GB, good balance
GOLD_MODEL = "deepseek-r1:14b"        # ~9GB, highest accuracy

# Baseline directory
BASELINE_DIR = Path(__file__).parent / "baselines"

# Tolerance for score comparisons (percentage)
SCORE_TOLERANCE_PCT = 10  # Allow 10% variance in scores
WIN_TOLERANCE = 0         # Wins must match exactly (no tolerance)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    benchmark_id: str
    benchmark_name: str
    model: str
    success: bool
    total_turns: int
    final_score: int
    own_solutions: int
    card_solutions: int
    syntax_errors: int
    repeated_actions: int
    latency_ms: int
    timestamp: str


@dataclass
class RegressionBaseline:
    """Stored baseline for regression comparison."""
    model: str
    created: str
    git_commit: Optional[str]
    benchmarks: Dict[str, BenchmarkResult]
    metadata: Dict = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Result of regression comparison."""
    benchmark_id: str
    passed: bool
    baseline_score: int
    actual_score: int
    baseline_success: bool
    actual_success: bool
    score_delta: int
    score_delta_pct: float
    message: str


# ============================================================================
# BASELINE MANAGEMENT
# ============================================================================

def get_baseline_path(model: str) -> Path:
    """Get path to baseline file for a model."""
    safe_name = model.replace("/", "_").replace(":", "_")
    return BASELINE_DIR / f"baseline_{safe_name}.json"


def load_baseline(model: str) -> Optional[RegressionBaseline]:
    """Load baseline for a model if it exists."""
    path = get_baseline_path(model)
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruct BenchmarkResult objects
    benchmarks = {}
    for bid, bdata in data.get("benchmarks", {}).items():
        benchmarks[bid] = BenchmarkResult(**bdata)

    return RegressionBaseline(
        model=data["model"],
        created=data["created"],
        git_commit=data.get("git_commit"),
        benchmarks=benchmarks,
        metadata=data.get("metadata", {})
    )


def save_baseline(baseline: RegressionBaseline) -> Path:
    """Save baseline to file."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = get_baseline_path(baseline.model)

    # Convert to serializable dict
    data = {
        "model": baseline.model,
        "created": baseline.created,
        "git_commit": baseline.git_commit,
        "metadata": baseline.metadata,
        "benchmarks": {
            bid: asdict(br) for bid, br in baseline.benchmarks.items()
        }
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_benchmarks(
    model: str,
    benchmarks: Optional[List[str]] = None,
    max_turns: int = 10,
    quiet: bool = True,
    timeout: int = 600
) -> Dict[str, BenchmarkResult]:
    """
    Run benchmarks and return results.

    Args:
        model: Ollama model name
        benchmarks: List of benchmark IDs (01-06), or None for all
        max_turns: Maximum turns per benchmark
        quiet: Suppress verbose output
        timeout: Timeout in seconds

    Returns:
        Dict mapping benchmark_id to BenchmarkResult
    """
    # Build command
    cmd = [
        sys.executable, "-m",
        "ragix_core.reasoning_tutor.benchmarks.scored_mode",
        "--models", model,
        "--max-turns", str(max_turns),
    ]

    if benchmarks:
        for b in benchmarks:
            cmd.extend(["--benchmark", b])

    if quiet:
        cmd.append("--quiet")

    # Run from project root
    project_root = Path(__file__).parent.parent.parent.parent

    print(f"Running benchmarks with {model}...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark run timed out after {timeout}s")
        return {}

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    # Parse output for results
    results = {}
    output = result.stdout + result.stderr

    # Look for game_end events in any JSONL output or structured output
    for line in output.split("\n"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("type") == "game_end":
                game_id = record.get("game_id", "")
                # Extract benchmark ID from game_id (e.g., "g183444_01" -> "01")
                bid = game_id.split("_")[-1] if "_" in game_id else ""
                if not bid:
                    continue

                results[bid] = BenchmarkResult(
                    benchmark_id=bid,
                    benchmark_name=record.get("benchmark", f"B{bid}"),
                    model=model,
                    success=record.get("success", False),
                    total_turns=record.get("total_turns", 0),
                    final_score=record.get("final_score", 0),
                    own_solutions=record.get("own_solutions", 0),
                    card_solutions=record.get("card_solutions", 0),
                    syntax_errors=record.get("syntax_errors", 0),
                    repeated_actions=record.get("repeated_actions", 0),
                    latency_ms=record.get("total_latency_ms", 0),
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
        except json.JSONDecodeError:
            continue

    return results


def parse_existing_results(jsonl_path: Path) -> Dict[str, BenchmarkResult]:
    """Parse existing JSONL results file into BenchmarkResults."""
    results = {}
    model = None

    with open(jsonl_path, "r") as f:
        current_game = {}
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)

                if record.get("type") == "game_start":
                    current_game = {
                        "game_id": record.get("game_id"),
                        "model": record.get("model"),
                        "benchmark": record.get("benchmark"),
                    }
                    model = record.get("model")

                elif record.get("type") == "game_end":
                    game_id = record.get("game_id", "")
                    # Extract benchmark ID from game_id (e.g., "g183444_01" -> "01")
                    bid = game_id.split("_")[-1] if "_" in game_id else game_id[:2]

                    results[bid] = BenchmarkResult(
                        benchmark_id=bid,
                        benchmark_name=current_game.get("benchmark", ""),
                        model=current_game.get("model", model or "unknown"),
                        success=record.get("success", False),
                        total_turns=record.get("total_turns", 0),
                        final_score=record.get("final_score", 0),
                        own_solutions=record.get("own_solutions", 0),
                        card_solutions=record.get("card_solutions", 0),
                        syntax_errors=record.get("syntax_errors", 0),
                        repeated_actions=record.get("repeated_actions", 0),
                        latency_ms=record.get("total_latency_ms", 0),
                        timestamp=record.get("ts", datetime.utcnow().isoformat() + "Z")
                    )
            except json.JSONDecodeError:
                continue

    return results


# ============================================================================
# REGRESSION COMPARISON
# ============================================================================

def compare_results(
    baseline: RegressionBaseline,
    actual: Dict[str, BenchmarkResult],
    score_tolerance_pct: float = SCORE_TOLERANCE_PCT,
    win_tolerance: int = WIN_TOLERANCE
) -> Tuple[List[RegressionResult], bool]:
    """
    Compare actual results against baseline.

    Returns:
        Tuple of (list of RegressionResult, overall_passed)
    """
    comparisons = []
    all_passed = True

    for bid, baseline_result in baseline.benchmarks.items():
        actual_result = actual.get(bid)

        if actual_result is None:
            comparisons.append(RegressionResult(
                benchmark_id=bid,
                passed=False,
                baseline_score=baseline_result.final_score,
                actual_score=0,
                baseline_success=baseline_result.success,
                actual_success=False,
                score_delta=-baseline_result.final_score,
                score_delta_pct=-100.0,
                message=f"MISSING: Benchmark {bid} not run"
            ))
            all_passed = False
            continue

        # Compare success/win
        success_match = baseline_result.success == actual_result.success

        # Compare scores with tolerance
        score_delta = actual_result.final_score - baseline_result.final_score
        if baseline_result.final_score != 0:
            score_delta_pct = (score_delta / abs(baseline_result.final_score)) * 100
        else:
            score_delta_pct = 100.0 if score_delta > 0 else (-100.0 if score_delta < 0 else 0.0)

        score_ok = abs(score_delta_pct) <= score_tolerance_pct

        # Determine pass/fail
        passed = success_match and score_ok

        # Build message
        if passed:
            msg = f"OK: score {actual_result.final_score} (baseline {baseline_result.final_score}, {score_delta_pct:+.1f}%)"
        elif not success_match:
            msg = f"REGRESSION: win={actual_result.success} (baseline win={baseline_result.success})"
            all_passed = False
        else:
            msg = f"SCORE DRIFT: {actual_result.final_score} vs baseline {baseline_result.final_score} ({score_delta_pct:+.1f}%)"
            all_passed = False

        comparisons.append(RegressionResult(
            benchmark_id=bid,
            passed=passed,
            baseline_score=baseline_result.final_score,
            actual_score=actual_result.final_score,
            baseline_success=baseline_result.success,
            actual_success=actual_result.success,
            score_delta=score_delta,
            score_delta_pct=score_delta_pct,
            message=msg
        ))

    return comparisons, all_passed


# ============================================================================
# REPORTING
# ============================================================================

def print_comparison_report(
    comparisons: List[RegressionResult],
    model: str,
    baseline_date: str
) -> None:
    """Print human-readable comparison report."""
    print("\n" + "=" * 70)
    print(f"REGRESSION TEST REPORT - {model}")
    print(f"Baseline: {baseline_date}")
    print("=" * 70)

    passed = sum(1 for c in comparisons if c.passed)
    total = len(comparisons)

    print(f"\nSummary: {passed}/{total} benchmarks passed\n")

    print(f"{'Bench':<6} {'Status':<10} {'Win':<12} {'Score':<18} {'Delta':<12}")
    print("-" * 70)

    for c in comparisons:
        status = "PASS" if c.passed else "FAIL"
        win_actual = "Y" if c.actual_success else "N"
        win_base = "Y" if c.baseline_success else "N"
        win_str = f"{win_actual} ({win_base})"
        score_str = f"{c.actual_score:>6} ({c.baseline_score:>6})"
        delta_str = f"{c.score_delta:>+5} ({c.score_delta_pct:>+5.1f}%)"

        print(f"B{c.benchmark_id:<5} {status:<10} {win_str:<12} {score_str:<18} {delta_str:<12}")

    print("-" * 70)

    if passed == total:
        print("\n[PASS] All regression tests PASSED")
    else:
        print(f"\n[FAIL] {total - passed} regression test(s) FAILED")


def generate_ci_output(comparisons: List[RegressionResult], model: str) -> str:
    """Generate CI-friendly output (GitHub Actions format)."""
    lines = []

    passed = sum(1 for c in comparisons if c.passed)
    total = len(comparisons)

    lines.append(f"## Regression Test Results - {model}")
    lines.append(f"**{passed}/{total}** benchmarks passed")
    lines.append("")
    lines.append("| Benchmark | Status | Win | Score | Delta |")
    lines.append("|-----------|--------|-----|-------|-------|")

    for c in comparisons:
        status = "PASS" if c.passed else "FAIL"
        win = "Y" if c.actual_success else "N"
        baseline_win = "Y" if c.baseline_success else "N"
        lines.append(
            f"| B{c.benchmark_id} | {status} | {win} ({baseline_win}) | "
            f"{c.actual_score} ({c.baseline_score}) | {c.score_delta:+d} ({c.score_delta_pct:+.1f}%) |"
        )

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Regression test runner for Interpreter-Tutor benchmarks"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run regression tests")
    run_parser.add_argument(
        "--model", "-m",
        default=FAST_MODEL,
        help=f"Model to test (default: {FAST_MODEL})"
    )
    run_parser.add_argument(
        "--fast", action="store_true",
        help=f"Use fast model ({FAST_MODEL})"
    )
    run_parser.add_argument(
        "--gold", action="store_true",
        help=f"Use gold standard model ({GOLD_MODEL})"
    )
    run_parser.add_argument(
        "--benchmarks", "-b",
        help="Comma-separated benchmark IDs (e.g., 01,02,03)"
    )
    run_parser.add_argument(
        "--max-turns", type=int, default=10,
        help="Maximum turns per benchmark"
    )
    run_parser.add_argument(
        "--tolerance", type=float, default=SCORE_TOLERANCE_PCT,
        help=f"Score tolerance percentage (default: {SCORE_TOLERANCE_PCT})"
    )
    run_parser.add_argument(
        "--ci", action="store_true",
        help="Output in CI-friendly format"
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output during benchmark run"
    )

    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Create or update baseline")
    baseline_parser.add_argument(
        "--model", "-m",
        default=FAST_MODEL,
        help=f"Model to baseline (default: {FAST_MODEL})"
    )
    baseline_parser.add_argument(
        "--from-file", "-f",
        help="Create baseline from existing JSONL results file"
    )
    baseline_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing baseline"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available baselines")

    # Compare command (for retesting historical results)
    compare_parser = subparsers.add_parser("compare", help="Compare JSONL file against baseline")
    compare_parser.add_argument(
        "jsonl_file",
        help="Path to JSONL results file"
    )
    compare_parser.add_argument(
        "--model", "-m",
        help="Model name (auto-detected from file if not specified)"
    )
    compare_parser.add_argument(
        "--tolerance", type=float, default=SCORE_TOLERANCE_PCT,
        help=f"Score tolerance percentage (default: {SCORE_TOLERANCE_PCT})"
    )

    args = parser.parse_args()

    # Handle commands
    if args.command == "run":
        model = GOLD_MODEL if args.gold else (FAST_MODEL if args.fast else args.model)
        benchmarks = args.benchmarks.split(",") if args.benchmarks else None

        # Load baseline
        baseline = load_baseline(model)
        if baseline is None:
            print(f"ERROR: No baseline found for {model}")
            print(f"Create one with: python -m ragix_core.reasoning_tutor.tests.regression_runner baseline --model {model}")
            sys.exit(1)

        # Run benchmarks
        results = run_benchmarks(
            model=model,
            benchmarks=benchmarks,
            max_turns=args.max_turns,
            quiet=not args.verbose
        )

        if not results:
            print("ERROR: No benchmark results obtained")
            sys.exit(1)

        # Compare
        comparisons, passed = compare_results(
            baseline, results, score_tolerance_pct=args.tolerance
        )

        # Report
        if args.ci:
            print(generate_ci_output(comparisons, model))
        else:
            print_comparison_report(comparisons, model, baseline.created)

        sys.exit(0 if passed else 1)

    elif args.command == "baseline":
        model = args.model

        # Check existing
        existing = load_baseline(model)
        if existing and not args.force:
            print(f"Baseline already exists for {model} (created {existing.created})")
            print("Use --force to overwrite")
            sys.exit(1)

        if args.from_file:
            # Create from existing file
            path = Path(args.from_file)
            if not path.exists():
                print(f"ERROR: File not found: {path}")
                sys.exit(1)

            results = parse_existing_results(path)
            if not results:
                print(f"ERROR: No results parsed from {path}")
                sys.exit(1)

            # Detect model from results if not specified
            first_result = next(iter(results.values()))
            if first_result.model and first_result.model != "unknown":
                model = first_result.model
        else:
            # Run fresh benchmarks
            print(f"Running fresh benchmarks with {model}...")
            results = run_benchmarks(model=model, quiet=True)
            if not results:
                print("ERROR: No benchmark results obtained")
                sys.exit(1)

        # Create baseline
        baseline = RegressionBaseline(
            model=model,
            created=datetime.utcnow().isoformat() + "Z",
            git_commit=get_git_commit(),
            benchmarks=results,
            metadata={
                "source": args.from_file if args.from_file else "fresh_run"
            }
        )

        path = save_baseline(baseline)
        print(f"Baseline saved: {path}")

        # Print summary
        total_score = sum(r.final_score for r in results.values())
        wins = sum(1 for r in results.values() if r.success)
        print(f"\nBaseline summary for {model}:")
        print(f"  Benchmarks: {len(results)}")
        print(f"  Wins: {wins}/{len(results)}")
        print(f"  Total score: {total_score}")

    elif args.command == "list":
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baselines = list(BASELINE_DIR.glob("baseline_*.json"))

        if not baselines:
            print("No baselines found")
            print(f"Create one with: python -m ragix_core.reasoning_tutor.tests.regression_runner baseline --model {FAST_MODEL}")
            sys.exit(0)

        print("Available baselines:")
        print("-" * 60)

        for path in sorted(baselines):
            with open(path, "r") as f:
                data = json.load(f)

            model = data.get("model", "unknown")
            created = data.get("created", "unknown")
            benchmarks = data.get("benchmarks", {})
            total_score = sum(b.get("final_score", 0) for b in benchmarks.values())
            wins = sum(1 for b in benchmarks.values() if b.get("success"))

            print(f"  {model:<30} {wins}/{len(benchmarks)} wins  {total_score:>+5} pts  ({created[:10]})")

    elif args.command == "compare":
        path = Path(args.jsonl_file)
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

        results = parse_existing_results(path)
        if not results:
            print(f"ERROR: No results parsed from {path}")
            sys.exit(1)

        # Detect model
        first_result = next(iter(results.values()))
        model = args.model or first_result.model

        # Load baseline
        baseline = load_baseline(model)
        if baseline is None:
            print(f"ERROR: No baseline found for {model}")
            sys.exit(1)

        # Compare
        comparisons, passed = compare_results(
            baseline, results, score_tolerance_pct=args.tolerance
        )

        print_comparison_report(comparisons, model, baseline.created)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
