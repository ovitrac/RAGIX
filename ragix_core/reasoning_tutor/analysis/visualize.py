#!/usr/bin/env python3
"""
Visualization Suite for Benchmark Analysis
============================================

Generates PCA plots, dendrograms, and heatmaps from feature matrices.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import sys
from pathlib import Path
from typing import Optional
import csv

# Check dependencies
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy pandas matplotlib scipy scikit-learn")
    sys.exit(1)


def load_features(csv_file: Path) -> pd.DataFrame:
    """Load feature matrix from CSV."""
    df = pd.read_csv(csv_file)
    df.set_index("model", inplace=True)
    return df


def run_pca(df: pd.DataFrame, output_dir: Path):
    """Run PCA and generate 2D visualization."""

    # Select numeric columns for PCA (including failure metrics)
    feature_cols = [
        "total_score", "success_rate", "avg_turns", "path_efficiency",
        "own_solutions", "card_solutions", "card_dependency",
        "syntax_errors", "repeated_actions", "latency_mean",
        # Failure detection metrics
        "failures_total", "failures_repetition", "failures_circular",
        "failures_explicit_error", "failure_rate"
    ]

    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(len(available), len(df)))
    X_pca = pca.fit_transform(X_scaled)

    # 2D Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PCA scatter
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for i, model in enumerate(df.index):
        ax1.scatter(X_pca[i, 0], X_pca[i, 1], c=[colors[i]], s=100, label=model[:20])
        ax1.annotate(model.split(":")[0], (X_pca[i, 0], X_pca[i, 1]),
                     fontsize=8, ha='center', va='bottom')

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax1.set_title("PCA: Model Positioning")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)

    # Right: Loadings
    ax2 = axes[1]
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    for i, feature in enumerate(available):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.05, head_length=0.02, fc='blue', ec='blue')
        ax2.annotate(feature, (loadings[i, 0], loadings[i, 1]),
                     fontsize=8, ha='center', va='bottom')

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("PCA Loadings: Feature Contributions")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_2d.png", dpi=150)
    plt.close()

    print(f"PCA plot saved: {output_dir / 'pca_2d.png'}")
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

    return X_pca, pca


def create_dendrogram(df: pd.DataFrame, output_dir: Path):
    """Create hierarchical clustering dendrogram."""

    # Select features for clustering (including failure metrics)
    feature_cols = [
        "total_score", "success_rate", "avg_turns",
        "own_solutions", "card_solutions",
        "syntax_errors", "repeated_actions",
        "failures_total", "failures_repetition", "failures_circular",
        "failures_explicit_error", "failure_rate"
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute linkage
    Z = linkage(X_scaled, method='ward')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    dendrogram(
        Z,
        labels=[m.split(":")[0] for m in df.index],
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax
    )

    ax.set_title("Hierarchical Clustering of LLM Behaviors")
    ax.set_ylabel("Distance (Ward)")

    plt.tight_layout()
    plt.savefig(output_dir / "dendrogram.png", dpi=150)
    plt.close()

    print(f"Dendrogram saved: {output_dir / 'dendrogram.png'}")


def create_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create score heatmap (model × benchmark)."""

    # Get benchmark score columns
    score_cols = [c for c in df.columns if c.startswith("score_")]

    if not score_cols:
        print("No benchmark scores found for heatmap")
        return

    scores = df[score_cols].copy()
    scores.columns = [c.replace("score_", "").replace("_", " ")[:15] for c in score_cols]

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(scores.values, cmap="RdYlGn", aspect="auto")

    # Labels
    ax.set_xticks(range(len(scores.columns)))
    ax.set_xticklabels(scores.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(scores.index)))
    ax.set_yticklabels([m.split(":")[0] for m in scores.index])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score")

    # Annotate cells
    for i in range(len(scores.index)):
        for j in range(len(scores.columns)):
            val = scores.iloc[i, j]
            color = "white" if val < scores.values.mean() else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_title("Benchmark Scores: Model × Benchmark")

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_scores.png", dpi=150)
    plt.close()

    print(f"Heatmap saved: {output_dir / 'heatmap_scores.png'}")


def create_params_vs_score(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Model Parameters vs Total Score."""

    if "params_b" not in df.columns:
        print("No params_b column found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by model type if available
    if "model_type" in df.columns:
        types = df["model_type"].unique()
        colors = {t: plt.cm.tab10(i) for i, t in enumerate(types)}
        for t in types:
            mask = df["model_type"] == t
            ax.scatter(df.loc[mask, "params_b"], df.loc[mask, "total_score"],
                       c=[colors[t]], s=100, label=t, alpha=0.7)
    else:
        ax.scatter(df["params_b"], df["total_score"], s=100, alpha=0.7)

    # Labels
    for model in df.index:
        ax.annotate(model.split(":")[0],
                    (df.loc[model, "params_b"], df.loc[model, "total_score"]),
                    fontsize=8, ha='left', va='bottom')

    ax.set_xlabel("Model Parameters (Billions)")
    ax.set_ylabel("Total Score")
    ax.set_title("Does Model Size Predict Performance?")
    ax.legend(title="Model Type")
    ax.grid(True, alpha=0.3)

    # Add correlation
    corr = df["params_b"].corr(df["total_score"])
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "params_vs_score.png", dpi=150)
    plt.close()

    print(f"Params vs Score saved: {output_dir / 'params_vs_score.png'}")
    print(f"Correlation (params vs score): r = {corr:.3f}")


def create_failure_analysis(df: pd.DataFrame, output_dir: Path):
    """Create failure analysis visualization."""

    failure_cols = ["failures_repetition", "failures_circular", "failures_explicit_error"]
    available = [c for c in failure_cols if c in df.columns]

    if not available:
        print("No failure columns found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar of failure types
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.8

    bottom = np.zeros(len(df))
    colors = {'failures_repetition': '#ff7f0e', 'failures_circular': '#2ca02c',
              'failures_explicit_error': '#d62728'}
    labels = {'failures_repetition': 'Repetition (⟳)', 'failures_circular': 'Circular (↻)',
              'failures_explicit_error': 'Explicit Error (⚠)'}

    for col in available:
        values = df[col].values
        ax1.bar(x, values, width, bottom=bottom, label=labels.get(col, col),
                color=colors.get(col, 'gray'))
        bottom += values

    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split(":")[0] for m in df.index], rotation=45, ha='right')
    ax1.set_ylabel("Number of Failures")
    ax1.set_title("Failure Types by Model")
    ax1.legend()

    # Right: Failure rate vs Success rate
    ax2 = axes[1]
    if "failure_rate" in df.columns and "success_rate" in df.columns:
        sc = ax2.scatter(df["failure_rate"], df["success_rate"], s=100,
                         c=df["total_score"], cmap="RdYlGn", alpha=0.7)
        for model in df.index:
            ax2.annotate(model.split(":")[0],
                        (df.loc[model, "failure_rate"], df.loc[model, "success_rate"]),
                        fontsize=8, ha='left', va='bottom')

        plt.colorbar(sc, ax=ax2, label="Total Score")
        ax2.set_xlabel("Failure Rate (failures/turn)")
        ax2.set_ylabel("Success Rate (wins/benchmarks)")
        ax2.set_title("Failure Rate vs Success Rate")
        ax2.grid(True, alpha=0.3)

        # Add correlation
        corr = df["failure_rate"].corr(df["success_rate"])
        ax2.text(0.95, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes,
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "failure_analysis.png", dpi=150)
    plt.close()

    print(f"Failure analysis saved: {output_dir / 'failure_analysis.png'}")


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate markdown summary report."""

    report = f"""# Benchmark Analysis Report — LLM Reasoning Olympics 2025-12-23

## Overview

- **Models analyzed:** {len(df)}
- **Benchmarks:** {len([c for c in df.columns if c.startswith('score_')])}
- **Total games:** {len(df) * len([c for c in df.columns if c.startswith('score_')])}

## Medal Standings

### By Success Rate (Win %)
| Rank | Model | Win Rate | Score | Failures |
|------|-------|----------|-------|----------|
"""

    for i, (model, row) in enumerate(df.sort_values("success_rate", ascending=False).iterrows(), 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else ""))
        report += f"| {i} {medal} | {model} | {row.get('success_rate', 0):.0%} | {row['total_score']:.0f} | {row.get('failures_total', 0):.0f} |\n"

    report += f"""

## Failure Analysis

### By Failure Type
| Model | Total | ⟳ Rep | ↻ Circ | ⚠ Error | Rate |
|-------|-------|-------|--------|---------|------|
"""

    for model, row in df.sort_values("failures_total", ascending=True).iterrows():
        report += f"| {model} | {row.get('failures_total', 0):.0f} | "
        report += f"{row.get('failures_repetition', 0):.0f} | "
        report += f"{row.get('failures_circular', 0):.0f} | "
        report += f"{row.get('failures_explicit_error', 0):.0f} | "
        report += f"{row.get('failure_rate', 0):.2f} |\n"

    report += """

## Key Findings

1. **Failure Rate predicts Success:** See `failure_analysis.png`
2. **Behavioral Clusters:** See `dendrogram.png` for model groupings
3. **Feature Space:** See `pca_2d.png` for dimensionality reduction
4. **Size ≠ Performance:** See `params_vs_score.png`

## Files Generated

- `pca_2d.png` - PCA visualization with failure metrics
- `dendrogram.png` - Hierarchical clustering
- `heatmap_scores.png` - Model × Benchmark scores
- `params_vs_score.png` - Size vs performance correlation
- `failure_analysis.png` - Failure type breakdown
- `report.md` - This report

## Clinical Diagnoses

- **⟳ Repetition Loop (Perseveration):** Syntactic aphasia — model repeats same command
- **↻ Circular Pattern (Disorientation):** Strategic confusion — model cycles without progress
- **⚠ Explicit Error (Agnosia):** Error recognition failure — cannot recover from mistakes

*Generated by RAGIX Interpreter-Tutor Analysis Suite*
"""

    report_file = output_dir / "report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Report saved: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations from features")
    parser.add_argument("csv_file", type=Path, help="Feature matrix CSV file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: same as input)")

    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    output_dir = args.output or args.csv_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.csv_file}")
    df = load_features(args.csv_file)

    print(f"\nGenerating visualizations...")
    run_pca(df, output_dir)
    create_dendrogram(df, output_dir)
    create_heatmap(df, output_dir)
    create_params_vs_score(df, output_dir)
    create_failure_analysis(df, output_dir)
    generate_report(df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
