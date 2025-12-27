#!/usr/bin/env python3
"""
Compare Olympics Rounds - Round 1 vs Round 2 Analysis
======================================================

Generates comparative visualizations and analysis between Olympics rounds.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import json

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import pearsonr
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy pandas matplotlib scipy")
    sys.exit(1)


def load_features_csv(csv_path: Path) -> pd.DataFrame:
    """Load feature matrix from CSV."""
    df = pd.read_csv(csv_path)
    df.set_index("model", inplace=True)
    return df


def load_jsonl_summary(jsonl_path: Path) -> Dict[str, dict]:
    """Extract model summaries from JSONL log."""
    models = {}
    games = {}

    with open(jsonl_path) as f:
        for line in f:
            event = json.loads(line.strip())
            etype = event.get("type")

            if etype == "game_start":
                games[event["game_id"]] = {
                    "model": event["model"],
                    "benchmark": event["benchmark"],
                    "optimal_turns": event.get("optimal_turns", 5)
                }

            elif etype == "game_end":
                gid = event["game_id"]
                if gid in games:
                    model = games[gid]["model"]
                    benchmark = games[gid]["benchmark"]

                    if model not in models:
                        models[model] = {
                            "wins": 0,
                            "total_games": 0,
                            "total_score": 0,
                            "total_turns": 0,
                            "failures": 0,
                            "benchmarks": {}
                        }

                    models[model]["total_games"] += 1
                    models[model]["total_score"] += event["final_score"]
                    models[model]["total_turns"] += event["total_turns"]
                    if event["success"]:
                        models[model]["wins"] += 1

                    models[model]["benchmarks"][benchmark] = {
                        "success": event["success"],
                        "score": event["final_score"],
                        "turns": event["total_turns"]
                    }

            elif etype == "failure_detected":
                gid = event["game_id"]
                if gid in games:
                    model = games[gid]["model"]
                    if model in models:
                        models[model]["failures"] += 1

    # Calculate derived metrics
    for model, data in models.items():
        data["success_rate"] = data["wins"] / max(1, data["total_games"])
        data["avg_turns"] = data["total_turns"] / max(1, data["total_games"])
        data["failure_rate"] = data["failures"] / max(1, data["total_turns"])

    return models


def create_comparison_table(r1_data: Dict, r2_data: Dict) -> pd.DataFrame:
    """Create comparison DataFrame between rounds."""
    # Get common models
    r1_models = set(r1_data.keys()) if isinstance(r1_data, dict) else set(r1_data.index)
    r2_models = set(r2_data.keys())
    common = r1_models & r2_models

    rows = []
    for model in sorted(common):
        if isinstance(r1_data, pd.DataFrame):
            r1_wins = r1_data.loc[model, "success_rate"] * 6  # 6 benchmarks
            r1_score = r1_data.loc[model, "total_score"]
            r1_failures = r1_data.loc[model, "failures_total"]
        else:
            r1_wins = r1_data[model]["wins"]
            r1_score = r1_data[model]["total_score"]
            r1_failures = r1_data[model]["failures"]

        r2_wins = r2_data[model]["wins"]
        r2_score = r2_data[model]["total_score"]
        r2_failures = r2_data[model]["failures"]

        rows.append({
            "model": model,
            "r1_wins": r1_wins,
            "r2_wins": r2_wins,
            "delta_wins": r2_wins - r1_wins,
            "r1_score": r1_score,
            "r2_score": r2_score,
            "delta_score": r2_score - r1_score,
            "r1_failures": r1_failures,
            "r2_failures": r2_failures,
            "delta_failures": r2_failures - r1_failures,
            "r1_rate": r1_wins / 6,
            "r2_rate": r2_wins / 6
        })

    return pd.DataFrame(rows).set_index("model")


def plot_win_rate_comparison(df: pd.DataFrame, output_dir: Path, r1_name: str = "Round 1", r2_name: str = "Round 2"):
    """Bar chart comparing win rates between rounds."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df["r1_rate"], width, label=r1_name, color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, df["r2_rate"], width, label=r2_name, color="#e74c3c", alpha=0.8)

    # Add delta annotations
    for i, (idx, row) in enumerate(df.iterrows()):
        delta = row["delta_wins"]
        color = "green" if delta > 0 else ("red" if delta < 0 else "gray")
        symbol = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        ax.annotate(f"{symbol}{abs(delta):.0f}",
                    xy=(x[i], max(row["r1_rate"], row["r2_rate"]) + 0.05),
                    ha="center", fontsize=10, color=color, fontweight="bold")

    ax.set_xlabel("Model")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"LLM Olympics: {r1_name} vs {r2_name} Win Rates")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(":")[0] for m in df.index], rotation=45, ha="right")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add reference lines
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_win_rates.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'comparison_win_rates.png'}")


def plot_score_evolution(df: pd.DataFrame, output_dir: Path, r1_name: str = "Round 1", r2_name: str = "Round 2"):
    """Slope chart showing score changes."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (model, row) in enumerate(df.iterrows()):
        color = "green" if row["delta_score"] > 0 else ("red" if row["delta_score"] < 0 else "gray")
        lw = 2 if abs(row["delta_score"]) > 200 else 1

        ax.plot([0, 1], [row["r1_score"], row["r2_score"]],
                color=color, linewidth=lw, alpha=0.7, marker="o", markersize=8)

        # Labels
        ax.text(-0.05, row["r1_score"], model.split(":")[0],
                ha="right", va="center", fontsize=9)
        ax.text(1.05, row["r2_score"], f"{row['delta_score']:+.0f}",
                ha="left", va="center", fontsize=9, color=color)

    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r1_name, r2_name], fontsize=12)
    ax.set_ylabel("Total Score")
    ax.set_title(f"Score Evolution: {r1_name} → {r2_name}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_score_evolution.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'comparison_score_evolution.png'}")


def plot_failure_comparison(df: pd.DataFrame, output_dir: Path, r1_name: str = "Round 1", r2_name: str = "Round 2"):
    """Compare failure counts between rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width/2, df["r1_failures"], width, label=r1_name, color="#e74c3c", alpha=0.7)
    ax1.bar(x + width/2, df["r2_failures"], width, label=r2_name, color="#9b59b6", alpha=0.7)

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Total Failures Detected")
    ax1.set_title(f"Failure Detection: {r1_name} vs {r2_name}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split(":")[0] for m in df.index], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: Delta failures vs Delta wins scatter
    ax2 = axes[1]
    colors = ["green" if d > 0 else "red" for d in df["delta_wins"]]
    sc = ax2.scatter(df["delta_failures"], df["delta_wins"],
                     c=colors, s=100, alpha=0.7, edgecolors="black")

    for model, row in df.iterrows():
        ax2.annotate(model.split(":")[0],
                     (row["delta_failures"], row["delta_wins"]),
                     fontsize=8, ha="center", va="bottom")

    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(f"Δ Failures ({r2_name} - {r1_name})")
    ax2.set_ylabel(f"Δ Wins ({r2_name} - {r1_name})")
    ax2.set_title("Improvement Analysis: Failures vs Wins")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_failures.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'comparison_failures.png'}")


def plot_benchmark_heatmap(r1_data: Dict, r2_data: Dict, output_dir: Path, r2_name: str = "Round 2"):
    """Heatmap showing benchmark-level changes."""
    # Get common models and benchmarks
    common_models = sorted(set(r1_data.keys() if isinstance(r1_data, dict) else r1_data.index) &
                           set(r2_data.keys()))

    if not common_models:
        print("No common models for heatmap")
        return

    # Get benchmarks from r2_data
    sample_model = list(r2_data.keys())[0]
    benchmarks = list(r2_data[sample_model].get("benchmarks", {}).keys())

    if not benchmarks:
        print("No benchmark data for heatmap")
        return

    # Build delta matrix
    delta_matrix = []
    for model in common_models:
        row = []
        for bench in benchmarks:
            r2_success = r2_data[model]["benchmarks"].get(bench, {}).get("success", False)
            # For R1, we need to check if the model won that benchmark
            # This is approximate - we'd need the original JSONL for exact comparison
            r2_val = 1 if r2_success else 0
            row.append(r2_val)
        delta_matrix.append(row)

    delta_df = pd.DataFrame(delta_matrix, index=common_models, columns=benchmarks)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(delta_df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([b[:20] for b in benchmarks], rotation=45, ha="right")
    ax.set_yticks(range(len(common_models)))
    ax.set_yticklabels([m.split(":")[0] for m in common_models])

    # Annotate cells
    for i in range(len(common_models)):
        for j in range(len(benchmarks)):
            val = delta_df.iloc[i, j]
            symbol = "✓" if val == 1 else "✗"
            ax.text(j, i, symbol, ha="center", va="center", fontsize=12,
                    color="white" if val == 0 else "black")

    ax.set_title(f"{r2_name} Benchmark Results (✓=Win, ✗=Loss)")
    plt.colorbar(im, ax=ax, label="Success")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_benchmark_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'comparison_benchmark_heatmap.png'}")


def generate_comparison_report(df: pd.DataFrame, output_dir: Path, r1_name: str = "Round 1", r2_name: str = "Round 2"):
    """Generate markdown comparison report."""

    # Extract short labels (e.g., "Round 2" -> "R2")
    r1_short = r1_name.replace("Round ", "R")
    r2_short = r2_name.replace("Round ", "R")

    report = f"""# LLM Olympics: {r1_name} vs {r2_name} Comparison

## Overview

This report compares model performance between Olympics {r1_name} and {r2_name}.

## Win Rate Changes

| Model | {r1_short} Wins | {r2_short} Wins | Δ Wins | {r1_short} Rate | {r2_short} Rate | Change |
|-------|---------|---------|--------|---------|---------|--------|
"""

    for model, row in df.sort_values("delta_wins", ascending=False).iterrows():
        change = "↑" if row["delta_wins"] > 0 else ("↓" if row["delta_wins"] < 0 else "=")
        report += f"| {model} | {row['r1_wins']:.0f}/6 | {row['r2_wins']:.0f}/6 | "
        report += f"{row['delta_wins']:+.0f} | {row['r1_rate']:.0%} | {row['r2_rate']:.0%} | {change} |\n"

    report += f"""

## Score Changes

| Model | {r1_short} Score | {r2_short} Score | Δ Score |
|-------|----------|----------|---------|
"""

    for model, row in df.sort_values("delta_score", ascending=False).iterrows():
        report += f"| {model} | {row['r1_score']:.0f} | {row['r2_score']:.0f} | {row['delta_score']:+.0f} |\n"

    report += f"""

## Failure Analysis

| Model | {r1_short} Failures | {r2_short} Failures | Δ Failures |
|-------|-------------|-------------|------------|
"""

    for model, row in df.sort_values("delta_failures").iterrows():
        report += f"| {model} | {row['r1_failures']:.0f} | {row['r2_failures']:.0f} | {row['delta_failures']:+.0f} |\n"

    report += """

## Key Findings

### Improvements
"""
    improved = df[df["delta_wins"] > 0]
    for model, row in improved.iterrows():
        report += f"- **{model}**: +{row['delta_wins']:.0f} wins ({row['r1_rate']:.0%} → {row['r2_rate']:.0%})\n"

    report += """

### Regressions
"""
    regressed = df[df["delta_wins"] < 0]
    for model, row in regressed.iterrows():
        report += f"- **{model}**: {row['delta_wins']:.0f} wins ({row['r1_rate']:.0%} → {row['r2_rate']:.0%})\n"

    report += """

## Visualizations

- `comparison_win_rates.png` - Win rate comparison bar chart
- `comparison_score_evolution.png` - Score slope chart
- `comparison_failures.png` - Failure analysis
- `comparison_benchmark_heatmap.png` - Per-benchmark results

---

*Generated by RAGIX Interpreter-Tutor Analysis Suite*
"""

    report_file = output_dir / "comparison_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Saved: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare Olympics rounds")
    parser.add_argument("--r1", type=Path, required=True,
                        help="First round features CSV or JSONL")
    parser.add_argument("--r2", type=Path, required=True,
                        help="Second round JSONL file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: results/)")
    parser.add_argument("--r1-name", type=str, default="Round 2",
                        help="Name for first round (default: Round 2)")
    parser.add_argument("--r2-name", type=str, default="Round 3",
                        help="Name for second round (default: Round 3)")

    args = parser.parse_args()

    output_dir = args.output or args.r2.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_name = args.r1_name
    r2_name = args.r2_name

    print(f"Loading {r1_name}: {args.r1}")
    if args.r1.suffix == ".csv":
        r1_data = load_features_csv(args.r1)
    else:
        r1_data = load_jsonl_summary(args.r1)

    print(f"Loading {r2_name}: {args.r2}")
    r2_data = load_jsonl_summary(args.r2)

    print(f"\n{r1_name} models: {list(r1_data.keys()) if isinstance(r1_data, dict) else list(r1_data.index)}")
    print(f"{r2_name} models: {list(r2_data.keys())}")

    # Create comparison table
    comparison_df = create_comparison_table(r1_data, r2_data)
    print(f"\nCommon models: {list(comparison_df.index)}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_win_rate_comparison(comparison_df, output_dir, r1_name, r2_name)
    plot_score_evolution(comparison_df, output_dir, r1_name, r2_name)
    plot_failure_comparison(comparison_df, output_dir, r1_name, r2_name)
    plot_benchmark_heatmap(r1_data, r2_data, output_dir, r2_name)

    # Generate report
    generate_comparison_report(comparison_df, output_dir, r1_name, r2_name)

    # Save comparison CSV
    csv_path = output_dir / f"comparison_{r1_name.lower().replace(' ', '_')}_{r2_name.lower().replace(' ', '_')}.csv"
    comparison_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print(f"\n✓ All comparison files saved to: {output_dir}")


if __name__ == "__main__":
    main()
