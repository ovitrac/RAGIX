#!/usr/bin/env python3
"""
Feature Extraction for PCA/PCoA Analysis
=========================================

Extracts features from JSON-Lines benchmark logs for multivariate analysis.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import csv

# Model metadata (parameters, context size)
MODEL_METADATA = {
    "granite3.1-moe:3b": {"params": 3e9, "context": 8192, "type": "moe"},
    "llama3.2:3b": {"params": 3e9, "context": 8192, "type": "base"},
    "phi3:latest": {"params": 3.8e9, "context": 4096, "type": "base"},
    "dolphin-mistral:7b-v2.6-dpo-laser": {"params": 7e9, "context": 8192, "type": "finetuned"},
    "mistral:7b-instruct": {"params": 7e9, "context": 8192, "type": "instruct"},
    "mistral:latest": {"params": 7e9, "context": 8192, "type": "base"},
    "qwen2.5:7b": {"params": 7e9, "context": 32768, "type": "base"},
    "qwen2.5-coder:7b": {"params": 7e9, "context": 32768, "type": "coder"},
    "llama3:latest": {"params": 8e9, "context": 8192, "type": "base"},
    "deepseek-r1:14b": {"params": 14e9, "context": 32768, "type": "reasoning"},
    "qwen2.5-coder:14b": {"params": 14e9, "context": 32768, "type": "coder"},
}


@dataclass
class ModelFeatures:
    """Extracted features for a single model."""
    model: str
    # Metadata
    params_b: float = 0.0        # Parameters in billions
    context_size: int = 8192    # Context window
    model_type: str = "base"    # Type category
    # Performance
    total_score: int = 0
    success_rate: float = 0.0
    avg_turns: float = 0.0
    path_efficiency: float = 0.0
    # Behaviors
    own_solutions: int = 0
    card_solutions: int = 0
    syntax_errors: int = 0
    repeated_actions: int = 0
    card_dependency: float = 0.0  # card_solutions / (own + card)
    # Latency
    latency_mean: float = 0.0
    latency_total: int = 0
    # Failure Detection Metrics (NEW)
    failures_total: int = 0
    failures_repetition: int = 0      # REPETITION_LOOP count
    failures_circular: int = 0        # CIRCULAR_PATTERN count
    failures_explicit_error: int = 0  # EXPLICIT_ERROR count
    failures_progress_stall: int = 0  # PROGRESS_STALL count
    failure_rate: float = 0.0         # failures / total_turns
    # Per-benchmark
    benchmark_scores: Dict[str, int] = field(default_factory=dict)


def parse_logs(log_file: Path) -> Dict[str, ModelFeatures]:
    """Parse JSON-Lines log file and extract features."""

    # Temporary storage
    games = {}  # game_id -> {model, benchmark, turns, ...}
    turns_data = defaultdict(list)  # game_id -> [turn events]
    failures_data = defaultdict(list)  # game_id -> [failure events]

    with open(log_file) as f:
        for line in f:
            event = json.loads(line.strip())
            event_type = event.get("type")

            if event_type == "game_start":
                games[event["game_id"]] = {
                    "model": event["model"],
                    "benchmark": event["benchmark"],
                    "optimal_turns": event.get("optimal_turns", 5)
                }

            elif event_type == "turn":
                turns_data[event["game_id"]].append(event)

            elif event_type == "failure_detected":
                failures_data[event["game_id"]].append(event)

            elif event_type == "game_end":
                gid = event["game_id"]
                if gid in games:
                    games[gid].update({
                        "success": event["success"],
                        "total_turns": event["total_turns"],
                        "final_score": event["final_score"],
                        "own_solutions": event["own_solutions"],
                        "card_solutions": event["card_solutions"],
                        "syntax_errors": event["syntax_errors"],
                        "repeated_actions": event["repeated_actions"],
                        "total_latency_ms": event["total_latency_ms"],
                        "path_efficiency": event.get("path_efficiency", 0.0)
                    })
                    # Add failure counts
                    game_failures = failures_data.get(gid, [])
                    games[gid]["failures"] = game_failures

    # Aggregate by model
    model_features = {}

    for gid, game in games.items():
        if "final_score" not in game:
            continue

        model = game["model"]
        if model not in model_features:
            meta = MODEL_METADATA.get(model, {})
            model_features[model] = ModelFeatures(
                model=model,
                params_b=meta.get("params", 7e9) / 1e9,
                context_size=meta.get("context", 8192),
                model_type=meta.get("type", "unknown")
            )

        mf = model_features[model]
        mf.total_score += game["final_score"]
        mf.own_solutions += game["own_solutions"]
        mf.card_solutions += game["card_solutions"]
        mf.syntax_errors += game["syntax_errors"]
        mf.repeated_actions += game["repeated_actions"]
        mf.latency_total += game["total_latency_ms"]
        mf.benchmark_scores[game["benchmark"]] = game["final_score"]

        # Aggregate failure counts
        for failure in game.get("failures", []):
            ftype = failure.get("failure_type", "")
            mf.failures_total += 1
            if ftype == "repetition_loop":
                mf.failures_repetition += 1
            elif ftype == "circular_pattern":
                mf.failures_circular += 1
            elif ftype == "explicit_error":
                mf.failures_explicit_error += 1
            elif ftype == "progress_stall":
                mf.failures_progress_stall += 1

    # Calculate derived features
    for model, mf in model_features.items():
        n_benchmarks = len(mf.benchmark_scores)
        if n_benchmarks > 0:
            mf.success_rate = sum(
                1 for gid, g in games.items()
                if g["model"] == model and g.get("success", False)
            ) / n_benchmarks

            total_turns = sum(
                g["total_turns"] for gid, g in games.items()
                if g["model"] == model
            )
            mf.avg_turns = total_turns / n_benchmarks

            total_pe = sum(
                g.get("path_efficiency", 0) for gid, g in games.items()
                if g["model"] == model
            )
            mf.path_efficiency = total_pe / n_benchmarks

            mf.latency_mean = mf.latency_total / max(1, total_turns)

            total_solutions = mf.own_solutions + mf.card_solutions
            mf.card_dependency = mf.card_solutions / max(1, total_solutions)

            # Failure rate = failures per turn
            mf.failure_rate = mf.failures_total / max(1, total_turns)

    return model_features


def export_csv(features: Dict[str, ModelFeatures], output_file: Path):
    """Export features to CSV for external analysis."""

    # Get all benchmark names
    all_benchmarks = set()
    for mf in features.values():
        all_benchmarks.update(mf.benchmark_scores.keys())
    benchmarks = sorted(all_benchmarks)

    # CSV header with failure metrics
    header = [
        "model", "params_b", "context_size", "model_type",
        "total_score", "success_rate", "avg_turns", "path_efficiency",
        "own_solutions", "card_solutions", "card_dependency",
        "syntax_errors", "repeated_actions",
        "latency_mean", "latency_total",
        # Failure detection metrics
        "failures_total", "failures_repetition", "failures_circular",
        "failures_explicit_error", "failures_progress_stall", "failure_rate"
    ] + [f"score_{b}" for b in benchmarks]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for model, mf in sorted(features.items()):
            row = [
                mf.model, mf.params_b, mf.context_size, mf.model_type,
                mf.total_score, f"{mf.success_rate:.3f}",
                f"{mf.avg_turns:.2f}", f"{mf.path_efficiency:.3f}",
                mf.own_solutions, mf.card_solutions, f"{mf.card_dependency:.3f}",
                mf.syntax_errors, mf.repeated_actions,
                f"{mf.latency_mean:.1f}", mf.latency_total,
                # Failure metrics
                mf.failures_total, mf.failures_repetition, mf.failures_circular,
                mf.failures_explicit_error, mf.failures_progress_stall,
                f"{mf.failure_rate:.3f}"
            ] + [mf.benchmark_scores.get(b, 0) for b in benchmarks]

            writer.writerow(row)

    print(f"Exported to: {output_file}")


def print_summary(features: Dict[str, ModelFeatures]):
    """Print human-readable summary."""

    print("\n" + "="*90)
    print("FEATURE EXTRACTION SUMMARY (with Failure Detection Metrics)")
    print("="*90)

    print(f"\n{'Model':<30} {'Params':<7} {'Score':<7} {'Win%':<6} {'Own':<4} {'Card':<4} "
          f"{'Fail':<5} {'⟳':<3} {'↻':<3} {'⚠':<3}")
    print("-"*90)

    for model, mf in sorted(features.items(), key=lambda x: -x[1].success_rate):
        print(f"{model:<30} {mf.params_b:.1f}B   {mf.total_score:<7} "
              f"{mf.success_rate:.0%}    {mf.own_solutions:<4} {mf.card_solutions:<4} "
              f"{mf.failures_total:<5} {mf.failures_repetition:<3} "
              f"{mf.failures_circular:<3} {mf.failures_explicit_error:<3}")

    print("\n" + "-"*90)
    print("Legend: ⟳=Repetition, ↻=Circular, ⚠=ExplicitError")
    print("Features extracted for PCA/PCoA analysis.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from benchmark logs")
    parser.add_argument("log_file", type=Path, help="JSON-Lines log file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output CSV file (default: feature_matrix.csv)")

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)

    output_file = args.output or args.log_file.parent / "feature_matrix.csv"

    print(f"Parsing: {args.log_file}")
    features = parse_logs(args.log_file)

    print_summary(features)
    export_csv(features, output_file)


if __name__ == "__main__":
    main()
