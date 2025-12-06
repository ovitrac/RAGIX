#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Experiment Runner for ContractiveReasoner
===============================================

Runs systematic experiments for academic paper evaluation.
Supports scenario matrix × task set with baseline collection.

Usage:
    python run_paper_experiments.py \
        --scenarios configs/paper_scenarios.yaml \
        --tasks configs/task_set.yaml \
        --output-dir results/paper_run_001

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    yaml = None

from ContractiveReasoner import ContractiveReasoner


@dataclass
class TaskResult:
    """Result of a single task run."""
    task_id: str
    scenario: str
    model: str
    peer_model: Optional[str]

    # Timing
    start_time: str
    duration_sec: float

    # Answer
    final_answer: str
    keywords_found: int
    keywords_total: int
    keyword_score: float

    # Tree metrics
    total_nodes: int
    solved_nodes: int
    failed_nodes: int
    pruned_nodes: int
    max_depth: int

    # Entropy metrics
    entropy_model_mean: float
    entropy_model_min: float
    entropy_model_max: float
    relevance_mean: float

    # Token metrics
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

    # Peer metrics
    peer_enabled: bool = False
    peer_calls: int = 0
    peer_approved: int = 0
    peer_rejected: int = 0
    peer_revision: int = 0
    peer_tokens: int = 0
    peer_time_sec: float = 0.0

    # Decomposition
    decomposed: bool = False
    expected_decomposition: bool = False
    decomposition_correct: bool = True

    # Quality score (composite)
    total_score: float = 0.0
    passed: bool = False

    # Error tracking
    error: Optional[str] = None


@dataclass
class ExperimentRun:
    """Full experiment run metadata."""
    run_id: str
    start_time: str
    end_time: str
    total_duration_sec: float

    scenarios_file: str
    tasks_file: str
    output_dir: str

    scenarios_run: List[str]
    tasks_run: int
    total_runs: int
    successful_runs: int
    failed_runs: int

    results: List[TaskResult] = field(default_factory=list)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file."""
    if yaml is None:
        raise RuntimeError("PyYAML required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def count_keywords(answer: str, keywords: List[str]) -> int:
    """Count how many keywords appear in the answer."""
    answer_lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in answer_lower)


async def run_brute_force(
    question: str,
    model: str,
    base_url: str,
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    """Run direct LLM call without contractive reasoning (baseline)."""
    t0 = time.time()

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer the question directly and completely."},
                {"role": "user", "content": question},
            ],
            "stream": False,
        }

        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {
                "answer": "",
                "error": str(e),
                "duration_sec": time.time() - t0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

    msg = data.get("message", {})
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)

    return {
        "answer": content,
        "error": None,
        "duration_sec": time.time() - t0,
        "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
        "completion_tokens": data.get("eval_count", 0) or 0,
    }


async def run_single_task(
    task: Dict[str, Any],
    scenario_name: str,
    scenario_config: Dict[str, Any],
    defaults: Dict[str, Any],
    output_dir: Path,
) -> TaskResult:
    """Run a single task with a specific scenario."""
    task_id = task["id"]
    question = task["question"]
    keywords = task.get("keywords", [])
    expected_decomp = task.get("expected_decomposition", False)

    # Merge defaults with scenario config
    config = {**defaults, **scenario_config}
    peer_config = config.pop("peer", None)

    model = config.get("model", "granite3.1-moe:3b")
    base_url = config.get("base_url", "http://localhost:11434")

    start_time = datetime.now().isoformat()
    t0 = time.time()

    # Check if brute force mode
    if config.get("brute_force", False):
        bf_result = await run_brute_force(
            question=question,
            model=model,
            base_url=base_url,
            timeout_sec=config.get("timeout_sec", 60),
        )

        kw_found = count_keywords(bf_result["answer"], keywords)
        kw_score = kw_found / len(keywords) if keywords else 1.0

        return TaskResult(
            task_id=task_id,
            scenario=scenario_name,
            model=model,
            peer_model=None,
            start_time=start_time,
            duration_sec=bf_result["duration_sec"],
            final_answer=bf_result["answer"],
            keywords_found=kw_found,
            keywords_total=len(keywords),
            keyword_score=kw_score,
            total_nodes=1,
            solved_nodes=1,
            failed_nodes=0,
            pruned_nodes=0,
            max_depth=0,
            entropy_model_mean=0.0,
            entropy_model_min=0.0,
            entropy_model_max=0.0,
            relevance_mean=1.0,
            total_prompt_tokens=bf_result["prompt_tokens"],
            total_completion_tokens=bf_result["completion_tokens"],
            total_tokens=bf_result["prompt_tokens"] + bf_result["completion_tokens"],
            peer_enabled=False,
            decomposed=False,
            expected_decomposition=expected_decomp,
            decomposition_correct=not expected_decomp,
            total_score=kw_score,
            passed=kw_score >= 0.5,
            error=bf_result["error"],
        )

    # Create engine with scenario config
    engine = ContractiveReasoner(
        base_url=base_url,
        model=model,
        max_depth=config.get("max_depth", 3),
        max_loops=config.get("max_loops", 6),
        max_global_tokens=config.get("max_global_tokens", 64000),
        max_branch_tokens=config.get("max_branch_tokens", 16000),
        max_concurrent_branches=config.get("max_concurrent_branches", 4),
        entropy_decompose_threshold=config.get("entropy_decompose_threshold", 0.9),
        entropy_collapse_threshold=config.get("entropy_collapse_threshold", 0.4),
        entropy_gamma_min_reduction=config.get("entropy_gamma_min_reduction", 0.05),
        k_entropy_samples=config.get("entropy_samples", 4),
        min_relevance_threshold=config.get("min_relevance_threshold", 0.15),
        max_rebranch_attempts=config.get("max_rebranch_attempts", 2),
        timeout_sec=config.get("timeout_sec", 120),
    )

    # Setup peer reviewer if configured
    peer_reviewer = None
    peer_model = None
    if peer_config and peer_config.get("model"):
        try:
            from peer_review import PeerConfig, PeerReviewer, PeerTiming

            timing_map = {
                "before_collapse": PeerTiming.BEFORE_COLLAPSE,
                "end_of_reasoning": PeerTiming.END_OF_REASONING,
                "on_demand": PeerTiming.ON_DEMAND,
            }

            peer_model = peer_config["model"]
            pc = PeerConfig(
                enabled=True,
                peer_model=peer_model,
                peer_base_url=peer_config.get("base_url", base_url),
                peer_timeout_sec=peer_config.get("timeout_sec", 180),
                timing=timing_map.get(peer_config.get("timing", "before_collapse"), PeerTiming.BEFORE_COLLAPSE),
                min_branch_depth=peer_config.get("min_depth", 2),
                min_branch_nodes=peer_config.get("min_nodes", 3),
                max_peer_calls_per_branch=peer_config.get("max_per_branch", 1),
                max_peer_calls_total=peer_config.get("max_calls", 10),
                peer_token_budget=peer_config.get("token_budget", 50000),
                approval_threshold=peer_config.get("approval_threshold", 0.6),
                rejection_threshold=peer_config.get("rejection_threshold", 0.3),
                kill_rejected_branches=peer_config.get("kill_rejected", True),
                force_peer_on_root=peer_config.get("force_root", False),
                log_peer_prompts=peer_config.get("log_prompts", False),
            )
            peer_reviewer = PeerReviewer(pc)
            engine.peer_reviewer = peer_reviewer
        except ImportError:
            pass

    # Run reasoning
    error = None
    try:
        result = await engine.solve(
            question,
            max_depth=config.get("max_depth", 3),
            max_loops=config.get("max_loops", 6),
        )
    except Exception as e:
        error = str(e)
        result = None

    duration = time.time() - t0

    # Extract metrics
    if result:
        summary = result.summarize()
        final_answer = result.final_answer or ""

        # Count node states
        nodes = result.tree
        total_nodes = len(nodes)
        solved_nodes = sum(1 for n in nodes.values() if n.state == "solved")
        failed_nodes = sum(1 for n in nodes.values() if n.state == "failed")
        pruned_nodes = sum(1 for n in nodes.values() if n.state == "pruned")

        # Entropy stats
        entropy_values = [n.metrics.entropy_model for n in nodes.values() if n.metrics]
        entropy_mean = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
        entropy_min = min(entropy_values) if entropy_values else 0.0
        entropy_max = max(entropy_values) if entropy_values else 0.0

        # Relevance stats
        relevance_values = [n.metrics.relevance_root for n in nodes.values() if n.metrics and n.metrics.relevance_root is not None]
        relevance_mean = sum(relevance_values) / len(relevance_values) if relevance_values else 0.0

        # Token stats
        prompt_tokens = sum(n.metrics.prompt_tokens for n in nodes.values() if n.metrics) if nodes else 0
        completion_tokens = sum(n.metrics.completion_tokens for n in nodes.values() if n.metrics) if nodes else 0

        # Check decomposition
        decomposed = total_nodes > 1
    else:
        final_answer = ""
        total_nodes = 0
        solved_nodes = 0
        failed_nodes = 0
        pruned_nodes = 0
        entropy_mean = entropy_min = entropy_max = 0.0
        relevance_mean = 0.0
        prompt_tokens = completion_tokens = 0
        decomposed = False

    # Keyword scoring
    kw_found = count_keywords(final_answer, keywords)
    kw_score = kw_found / len(keywords) if keywords else 1.0

    # Decomposition correctness
    decomposition_correct = (decomposed == expected_decomp) or (decomposed and expected_decomp)

    # Peer metrics
    peer_enabled = peer_reviewer is not None
    peer_calls = 0
    peer_approved = 0
    peer_rejected = 0
    peer_revision = 0
    peer_tokens = 0
    peer_time = 0.0

    if peer_reviewer:
        pm = peer_reviewer.metrics
        peer_calls = pm.total_calls
        peer_approved = pm.approved_count
        peer_rejected = pm.rejected_count
        peer_revision = pm.revision_count
        peer_tokens = pm.total_prompt_tokens + pm.total_completion_tokens
        peer_time = pm.total_elapsed_sec

    # Compute total score
    # Weighted: 60% keywords, 20% decomposition correctness, 20% completion
    completion_score = 1.0 if solved_nodes > 0 else 0.0
    decomp_score = 1.0 if decomposition_correct else 0.5
    total_score = 0.6 * kw_score + 0.2 * decomp_score + 0.2 * completion_score

    passed = total_score >= 0.5 and error is None

    # Save detailed trace
    trace_file = output_dir / f"{task_id}__{scenario_name}_trace.json"
    if result:
        trace = result.export_trace()
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, default=str)

    # Close peer reviewer
    if peer_reviewer:
        await peer_reviewer.close()

    return TaskResult(
        task_id=task_id,
        scenario=scenario_name,
        model=model,
        peer_model=peer_model,
        start_time=start_time,
        duration_sec=duration,
        final_answer=final_answer[:2000],  # Truncate for storage
        keywords_found=kw_found,
        keywords_total=len(keywords),
        keyword_score=kw_score,
        total_nodes=total_nodes,
        solved_nodes=solved_nodes,
        failed_nodes=failed_nodes,
        pruned_nodes=pruned_nodes,
        max_depth=result.max_depth if result else 0,
        entropy_model_mean=entropy_mean,
        entropy_model_min=entropy_min,
        entropy_model_max=entropy_max,
        relevance_mean=relevance_mean,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        peer_enabled=peer_enabled,
        peer_calls=peer_calls,
        peer_approved=peer_approved,
        peer_rejected=peer_rejected,
        peer_revision=peer_revision,
        peer_tokens=peer_tokens,
        peer_time_sec=peer_time,
        decomposed=decomposed,
        expected_decomposition=expected_decomp,
        decomposition_correct=decomposition_correct,
        total_score=total_score,
        passed=passed,
        error=error,
    )


async def run_experiments(
    scenarios_file: str,
    tasks_file: str,
    output_dir: str,
    include_scenarios: Optional[List[str]] = None,
    include_categories: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> ExperimentRun:
    """Run full experiment matrix."""
    # Load configs
    scenarios_config = load_yaml(scenarios_file)
    tasks_config = load_yaml(tasks_file)

    scenarios = scenarios_config.get("scenarios", {})
    defaults = scenarios_config.get("defaults", {})
    tasks = tasks_config.get("tasks", [])

    # Filter scenarios
    if include_scenarios:
        scenarios = {k: v for k, v in scenarios.items() if k in include_scenarios}

    # Filter tasks by category
    if include_categories:
        tasks = [t for t in tasks if t.get("category") in include_categories]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now().isoformat()
    t0 = time.time()

    results: List[TaskResult] = []
    successful = 0
    failed = 0
    total_runs = len(scenarios) * len(tasks)

    if limit:
        total_runs = min(total_runs, limit)

    print(f"Starting experiment run: {run_id}")
    print(f"Scenarios: {len(scenarios)}, Tasks: {len(tasks)}, Total runs: {total_runs}")
    print(f"Output directory: {output_path}")
    print("-" * 60)

    run_count = 0
    for scenario_name, scenario_config in scenarios.items():
        for task in tasks:
            if limit and run_count >= limit:
                break

            task_id = task["id"]
            print(f"[{run_count + 1}/{total_runs}] {scenario_name} × {task_id}...", end=" ", flush=True)

            try:
                result = await run_single_task(
                    task=task,
                    scenario_name=scenario_name,
                    scenario_config=scenario_config,
                    defaults=defaults,
                    output_dir=output_path,
                )
                results.append(result)

                if result.error:
                    print(f"ERROR: {result.error[:50]}")
                    failed += 1
                else:
                    status = "PASS" if result.passed else "FAIL"
                    print(f"{status} (score={result.total_score:.2f}, {result.duration_sec:.1f}s)")
                    if result.passed:
                        successful += 1
                    else:
                        failed += 1

            except Exception as e:
                print(f"EXCEPTION: {e}")
                failed += 1

            run_count += 1

        if limit and run_count >= limit:
            break

    end_time = datetime.now().isoformat()
    total_duration = time.time() - t0

    # Create experiment run summary
    experiment = ExperimentRun(
        run_id=run_id,
        start_time=start_time,
        end_time=end_time,
        total_duration_sec=total_duration,
        scenarios_file=scenarios_file,
        tasks_file=tasks_file,
        output_dir=output_dir,
        scenarios_run=list(scenarios.keys()),
        tasks_run=len(tasks),
        total_runs=run_count,
        successful_runs=successful,
        failed_runs=failed,
        results=results,
    )

    # Save results
    results_file = output_path / "experiment_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(asdict(experiment), f, indent=2, default=str)

    # Save CSV summary
    csv_file = output_path / "summary.csv"
    save_csv_summary(results, csv_file)

    print("-" * 60)
    print(f"Experiment completed in {total_duration:.1f}s")
    print(f"Results: {successful} passed, {failed} failed out of {run_count} runs")
    print(f"Output: {results_file}")

    return experiment


def save_csv_summary(results: List[TaskResult], output_file: Path):
    """Save results as CSV."""
    if not results:
        return

    headers = [
        "task_id", "scenario", "model", "peer_model",
        "duration_sec", "total_score", "passed",
        "keyword_score", "keywords_found", "keywords_total",
        "total_nodes", "solved_nodes", "max_depth",
        "entropy_mean", "relevance_mean",
        "total_tokens", "peer_enabled", "peer_calls",
        "peer_approved", "peer_rejected", "peer_tokens",
        "decomposed", "expected_decomposition", "decomposition_correct",
        "error"
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r.task_id, r.scenario, r.model, r.peer_model or "",
                f"{r.duration_sec:.2f}", f"{r.total_score:.3f}", str(r.passed),
                f"{r.keyword_score:.3f}", str(r.keywords_found), str(r.keywords_total),
                str(r.total_nodes), str(r.solved_nodes), str(r.max_depth),
                f"{r.entropy_model_mean:.3f}", f"{r.relevance_mean:.3f}",
                str(r.total_tokens), str(r.peer_enabled), str(r.peer_calls),
                str(r.peer_approved), str(r.peer_rejected), str(r.peer_tokens),
                str(r.decomposed), str(r.expected_decomposition), str(r.decomposition_correct),
                (r.error or "")[:50].replace(",", ";")
            ]
            f.write(",".join(row) + "\n")

    print(f"CSV summary saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run paper experiments for ContractiveReasoner evaluation"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="benchmarks/configs/paper_scenarios.yaml",
        help="Path to scenarios YAML file",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="benchmarks/configs/task_set.yaml",
        help="Path to tasks YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/paper_run",
        help="Output directory for results",
    )
    parser.add_argument(
        "--include-scenarios",
        type=str,
        nargs="+",
        help="Only run these scenarios (e.g., S1_baseline_fast S3_peer_light)",
    )
    parser.add_argument(
        "--include-categories",
        type=str,
        nargs="+",
        help="Only run tasks from these categories (e.g., math reasoning)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of runs",
    )

    args = parser.parse_args()

    asyncio.run(run_experiments(
        scenarios_file=args.scenarios,
        tasks_file=args.tasks,
        output_dir=args.output_dir,
        include_scenarios=args.include_scenarios,
        include_categories=args.include_categories,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
