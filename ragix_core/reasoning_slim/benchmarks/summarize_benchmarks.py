#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark summarizer for ContractiveReasoner
=============================================

Aggregates benchmark results into a CSV summary.

Usage:
    python benchmarks/summarize_benchmarks.py benchmarks/results --output benchmarks/results/summary.csv

Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_summary_files(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all *_summary.json files from results directory."""
    summaries = []
    for path in sorted(results_dir.glob("*_summary.json")):
        if path.name == "aggregate_results.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    return summaries


def extract_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract CSV row data from a summary dict."""
    s = summary.get("summary") or {}
    tokens = s.get("tokens", {})
    states = s.get("states", {})
    depth = s.get("depth", {})
    entropies = s.get("entropies", {})
    ent_model = entropies.get("model") or {}
    ent_struct = entropies.get("struct") or {}
    ent_consist = entropies.get("consistency") or {}
    relevance = s.get("relevance_root") or {}
    params = summary.get("params", {})

    return {
        "question_id": summary.get("question_id", ""),
        "scenario_id": summary.get("scenario_id", ""),
        "success": summary.get("success", False),
        "duration_sec": round(summary.get("duration_sec", 0), 2),
        "model": params.get("model", ""),
        "max_depth": params.get("max_depth", ""),
        "max_loops": params.get("max_loops", ""),
        "entropy_samples": params.get("entropy_samples", ""),
        "entropy_decompose_threshold": params.get("entropy_decompose_threshold", ""),
        "total_nodes": s.get("total_nodes", 0),
        "nodes_solved": states.get("solved", 0),
        "nodes_failed": states.get("failed", 0),
        "nodes_partial": states.get("partially_solved", 0),
        "tree_depth": depth.get("max", 0),
        "steps": s.get("steps", 0),
        "prompt_tokens": tokens.get("prompt", 0),
        "completion_tokens": tokens.get("completion", 0),
        "total_tokens": tokens.get("prompt", 0) + tokens.get("completion", 0),
        "entropy_model_min": round(ent_model.get("min", 0), 3) if ent_model else "",
        "entropy_model_mean": round(ent_model.get("mean", 0), 3) if ent_model else "",
        "entropy_model_max": round(ent_model.get("max", 0), 3) if ent_model else "",
        "entropy_struct_mean": round(ent_struct.get("mean", 0), 3) if ent_struct else "",
        "entropy_consist_mean": round(ent_consist.get("mean", 0), 3) if ent_consist else "",
        "relevance_mean": round(relevance.get("mean", 0), 3) if relevance else "",
        "ctx_window": s.get("ctx_window", ""),
        "error": summary.get("error", "") or "",
    }


def summarize_to_csv(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """Write summaries to CSV file."""
    if not summaries:
        print("No summaries found.")
        return

    rows = [extract_row(s) for s in summaries]
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    """Print a quick summary table to stdout."""
    print(f"\n{'Question':<15} {'Scenario':<15} {'OK':<5} {'Time':<8} {'Nodes':<6} {'Tokens':<10}")
    print("-" * 70)
    for s in summaries:
        row = extract_row(s)
        ok = "Y" if row["success"] else "N"
        print(
            f"{row['question_id']:<15} {row['scenario_id']:<15} {ok:<5} "
            f"{row['duration_sec']:<8.1f} {row['total_nodes']:<6} {row['total_tokens']:<10}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize ContractiveReasoner benchmarks to CSV")
    parser.add_argument("results_dir", help="Directory containing *_summary.json files")
    parser.add_argument("--output", default=None, help="Output CSV path (default: results_dir/summary.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return

    output_path = Path(args.output) if args.output else results_dir / "summary.csv"

    print(f"Loading summaries from: {results_dir}")
    summaries = load_summary_files(results_dir)
    print(f"Found {len(summaries)} benchmark results")

    if summaries:
        print_summary_table(summaries)
        summarize_to_csv(summaries, output_path)


if __name__ == "__main__":
    main()
