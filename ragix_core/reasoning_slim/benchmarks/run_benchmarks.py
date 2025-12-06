#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark runner for ContractiveReasoner
=========================================

Runs a matrix of questions × scenarios and saves traces, events, and summaries.

Usage:
    python benchmarks/run_benchmarks.py --config benchmarks/configs/sample_benchmarks.yaml
    python benchmarks/run_benchmarks.py --config benchmarks/configs/sample_benchmarks.yaml --limit 2
    python benchmarks/run_benchmarks.py --config benchmarks/configs/sample_benchmarks.yaml --output-dir /tmp/bench

Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    yaml = None

from ContractiveReasoner import (
    ContractiveReasoner,
    ReasoningNode,
    NodeMetrics,
    format_summary,
)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config file."""
    lower = path.lower()
    if lower.endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_params(defaults: Dict[str, Any], scenario_params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge default params with scenario-specific overrides."""
    result = dict(defaults)
    result.update(scenario_params)
    return result


async def run_single_benchmark(
    question_id: str,
    question_text: str,
    scenario_id: str,
    params: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a single benchmark (one question × one scenario).
    
    Saves:
      - {qid}__{scenario}_trace.json
      - {qid}__{scenario}_events.ndjson
      - {qid}__{scenario}_summary.json
    
    Returns summary dict.
    """
    prefix = f"{question_id}__{scenario_id}"
    trace_path = output_dir / f"{prefix}_trace.json"
    events_path = output_dir / f"{prefix}_events.ndjson"
    summary_path = output_dir / f"{prefix}_summary.json"

    # Build engine with merged params
    engine = ContractiveReasoner(
        base_url=params.get("base_url", "http://localhost:11434"),
        model=params.get("model", "granite3.1-moe:3b"),
        max_depth=params.get("max_depth", 3),
        max_loops=params.get("max_loops", 6),
        max_global_tokens=params.get("max_global_tokens", 64000),
        max_branch_tokens=params.get("max_branch_tokens", 16000),
        max_concurrent_branches=params.get("max_concurrent_branches", 4),
        entropy_decompose_threshold=params.get("entropy_decompose_threshold", 0.9),
        entropy_collapse_threshold=params.get("entropy_collapse_threshold", 0.4),
        entropy_gamma_min_reduction=params.get("entropy_gamma_min_reduction", 0.05),
        k_entropy_samples=params.get("entropy_samples", 4),
        timeout_sec=params.get("timeout_sec", 120),
    )

    # Collect events
    events: List[Dict[str, Any]] = []

    def event_callback(eng: ContractiveReasoner, node: ReasoningNode, metrics: NodeMetrics):
        events.append({
            "step": metrics.step_index,
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "role": node.role,
            "state": node.state,
            "depth": metrics.depth,
            "entropy_model": metrics.entropy_model,
            "entropy_struct": metrics.entropy_struct,
            "entropy_consistency": metrics.entropy_consistency,
            "relevance_root": metrics.relevance_root,
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "timestamp": metrics.timestamp,
        })

    engine.event_callback = event_callback

    # Run
    t0 = time.time()
    try:
        result = await engine.solve(question_text)
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        result = None
    elapsed = time.time() - t0

    # Save trace
    if result:
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(result.export_trace(), f, indent=2, ensure_ascii=False)

    # Save events
    with open(events_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # Build summary
    summary_data = {
        "question_id": question_id,
        "scenario_id": scenario_id,
        "question": question_text,
        "params": params,
        "success": success,
        "error": error_msg,
        "duration_sec": elapsed,
        "final_answer": result.final_answer if result else None,
        "summary": result.summarize() if result else None,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return summary_data


async def run_benchmark_matrix(
    config: Dict[str, Any],
    output_dir: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full benchmark matrix.
    
    Returns list of all summary dicts.
    """
    questions = config.get("questions", [])
    scenarios = config.get("scenarios", [])
    defaults = config.get("defaults", {})

    results: List[Dict[str, Any]] = []
    total_runs = len(questions) * len(scenarios)
    run_count = 0

    for q in questions:
        qid = q.get("id", "unknown")
        qtext = q.get("text", "")

        for s in scenarios:
            if limit and run_count >= limit:
                break

            sid = s.get("id", "unknown")
            sparams = s.get("params", {})
            merged = merge_params(defaults, sparams)

            print(f"\n[{run_count + 1}/{total_runs}] Running: {qid} × {sid}")
            print(f"  Question: {qtext[:60]}...")

            summary = await run_single_benchmark(qid, qtext, sid, merged, output_dir)
            results.append(summary)

            status = "OK" if summary["success"] else f"FAIL: {summary['error']}"
            print(f"  Result: {status} ({summary['duration_sec']:.1f}s)")

            run_count += 1

        if limit and run_count >= limit:
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark runner for ContractiveReasoner")
    parser.add_argument("--config", required=True, help="Path to benchmark config YAML/JSON")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit total number of runs")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config: {config_path}")
    config = load_config(str(config_path))

    print(f"Output directory: {output_dir}")
    print(f"Questions: {len(config.get('questions', []))}")
    print(f"Scenarios: {len(config.get('scenarios', []))}")

    t0 = time.time()
    results = asyncio.run(run_benchmark_matrix(config, output_dir, args.limit))
    total_time = time.time() - t0

    # Save aggregate results
    aggregate = {
        "total_runs": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_duration_sec": total_time,
        "runs": results,
    }
    agg_path = output_dir / "aggregate_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Benchmark complete: {aggregate['successful']}/{aggregate['total_runs']} successful")
    print(f"Total time: {total_time:.1f}s")
    print(f"Aggregate results: {agg_path}")


if __name__ == "__main__":
    main()
