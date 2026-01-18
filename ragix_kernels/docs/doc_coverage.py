"""
Kernel: Document Coverage
Stage: 2 (Analysis)

Analyzes concept coverage across documents and clusters.
Identifies gaps, overlaps, and dominant themes.

Useful for:
- Finding documents that cover specific topics
- Identifying coverage gaps (concepts with few documents)
- Detecting redundancy (same concept across many files)
- Comparing clusters by topic focus

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import logging
import json
import math

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocCoverageKernel(Kernel):
    """
    Analyze concept coverage across documents.

    This kernel builds a coverage matrix and identifies:
    - Which documents cover which concepts (coverage matrix)
    - Concept density per document
    - Coverage gaps (underrepresented concepts)
    - Cluster specialization (concepts dominant in specific clusters)

    Configuration options:
        reference_concepts: List of expected concepts to check for (optional)
        gap_threshold: Min file count to consider "covered" (default: 2)
        overlap_threshold: File count to consider "overrepresented" (default: 50%)

    Dependencies:
        doc_metadata: File inventory
        doc_concepts: Concept-file mappings
        doc_cluster: Document clusters

    Output:
        coverage_matrix: File × Concept matrix (sparse)
        file_coverage: Per-file coverage statistics
        concept_coverage: Per-concept coverage statistics
        cluster_coverage: Per-cluster concept profile
        gaps: Concepts with insufficient coverage
        overlaps: Concepts with high redundancy
        statistics: Coverage statistics
    """

    name = "doc_coverage"
    version = "1.0.0"
    category = "docs"
    stage = 2
    description = "Analyze concept coverage across documents"

    requires = ["doc_metadata", "doc_concepts", "doc_cluster"]
    provides = ["doc_coverage", "coverage_matrix", "coverage_gaps", "cluster_profiles"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Analyze concept coverage."""

        # Get configuration
        reference_concepts = input.config.get("reference_concepts", [])
        gap_threshold = input.config.get("gap_threshold", 2)
        overlap_threshold = input.config.get("overlap_threshold", 0.5)  # 50% of files

        logger.info("[doc_coverage] Analyzing concept coverage")

        # Load dependencies
        metadata_path = input.dependencies.get("doc_metadata")
        concepts_path = input.dependencies.get("doc_concepts")
        cluster_path = input.dependencies.get("doc_cluster")

        if not all(p and p.exists() for p in [metadata_path, concepts_path, cluster_path]):
            raise RuntimeError("Missing required dependencies")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})
        with open(concepts_path) as f:
            concepts_data = json.load(f).get("data", {})
        with open(cluster_path) as f:
            cluster_data = json.load(f).get("data", {})

        # Build lookups
        files = metadata_data.get("files", [])
        file_ids = [f["file_id"] for f in files]
        file_paths = {f["file_id"]: f["path"] for f in files}
        total_files = len(file_ids)

        file_concepts = concepts_data.get("file_concepts", {})
        concept_files = concepts_data.get("concept_files", {})
        concepts_list = concepts_data.get("concepts", [])
        concept_labels = {c["concept_id"]: c["label"] for c in concepts_list}

        clusters = cluster_data.get("clusters", [])
        file_to_cluster = {}
        cluster_files: Dict[str, List[str]] = {}
        for cluster in clusters:
            cid = cluster["cluster_id"]
            cluster_files[cid] = cluster.get("file_ids", [])
            for fid in cluster.get("file_ids", []):
                file_to_cluster[fid] = cid

        # Build coverage matrix (sparse representation)
        # coverage_matrix[file_id] = list of concept_ids
        coverage_matrix: Dict[str, List[str]] = {}
        for fid in file_ids:
            coverage_matrix[fid] = file_concepts.get(fid, [])

        # Compute per-file coverage statistics
        file_coverage: Dict[str, Dict[str, Any]] = {}
        for fid in file_ids:
            concepts = coverage_matrix.get(fid, [])
            file_coverage[fid] = {
                "file_id": fid,
                "path": file_paths.get(fid, fid),
                "concept_count": len(concepts),
                "concepts": concepts,
                "cluster_id": file_to_cluster.get(fid),
            }

        # Compute per-concept coverage statistics
        concept_coverage: Dict[str, Dict[str, Any]] = {}
        for concept in concepts_list:
            cid = concept["concept_id"]
            files = concept_files.get(cid, [])
            concept_coverage[cid] = {
                "concept_id": cid,
                "label": concept["label"],
                "file_count": len(files),
                "file_percentage": round(len(files) / total_files * 100, 1) if total_files else 0,
                "files": files,
            }

        # Identify gaps (concepts with low coverage)
        gaps: List[Dict[str, Any]] = []
        for cid, cov in concept_coverage.items():
            if cov["file_count"] < gap_threshold:
                gaps.append({
                    "concept_id": cid,
                    "label": cov["label"],
                    "file_count": cov["file_count"],
                    "reason": "below_threshold",
                })

        # Check reference concepts if provided
        if reference_concepts:
            existing_labels = {c["label"].lower() for c in concepts_list}
            for ref in reference_concepts:
                if ref.lower() not in existing_labels:
                    gaps.append({
                        "concept_id": None,
                        "label": ref,
                        "file_count": 0,
                        "reason": "missing_reference",
                    })

        # Identify overlaps (concepts in many files)
        overlap_count = int(total_files * overlap_threshold)
        overlaps: List[Dict[str, Any]] = []
        for cid, cov in concept_coverage.items():
            if cov["file_count"] >= overlap_count:
                overlaps.append({
                    "concept_id": cid,
                    "label": cov["label"],
                    "file_count": cov["file_count"],
                    "file_percentage": cov["file_percentage"],
                })

        # Compute cluster coverage profiles
        cluster_coverage: Dict[str, Dict[str, Any]] = {}
        for cluster in clusters:
            cid = cluster["cluster_id"]
            c_files = cluster_files.get(cid, [])

            # Count concepts in this cluster
            cluster_concept_counts: Dict[str, int] = defaultdict(int)
            for fid in c_files:
                for concept_id in coverage_matrix.get(fid, []):
                    cluster_concept_counts[concept_id] += 1

            # Identify dominant concepts (appear in >50% of cluster files)
            dominant = []
            cluster_size = len(c_files)
            for concept_id, count in cluster_concept_counts.items():
                if cluster_size > 0 and count / cluster_size >= 0.5:
                    dominant.append({
                        "concept_id": concept_id,
                        "label": concept_labels.get(concept_id, concept_id),
                        "count": count,
                        "percentage": round(count / cluster_size * 100, 1),
                    })

            dominant.sort(key=lambda x: -x["count"])

            cluster_coverage[cid] = {
                "cluster_id": cid,
                "label": cluster.get("label", cid),
                "file_count": cluster_size,
                "unique_concepts": len(cluster_concept_counts),
                "dominant_concepts": dominant[:10],
                "concept_density": (
                    round(sum(cluster_concept_counts.values()) / cluster_size, 1)
                    if cluster_size else 0
                ),
            }

        # Compute inter-cluster similarity (Jaccard on concepts)
        cluster_similarity: Dict[str, Dict[str, float]] = {}
        cluster_ids = list(cluster_coverage.keys())
        for i, c1 in enumerate(cluster_ids):
            cluster_similarity[c1] = {}
            c1_concepts = set(
                d["concept_id"] for d in cluster_coverage[c1]["dominant_concepts"]
            )
            for j, c2 in enumerate(cluster_ids):
                if i == j:
                    cluster_similarity[c1][c2] = 1.0
                elif j < i:
                    cluster_similarity[c1][c2] = cluster_similarity[c2][c1]
                else:
                    c2_concepts = set(
                        d["concept_id"] for d in cluster_coverage[c2]["dominant_concepts"]
                    )
                    if c1_concepts or c2_concepts:
                        jaccard = len(c1_concepts & c2_concepts) / len(c1_concepts | c2_concepts)
                    else:
                        jaccard = 0.0
                    cluster_similarity[c1][c2] = round(jaccard, 3)

        # Statistics
        concept_counts = [cov["concept_count"] for cov in file_coverage.values()]
        statistics = {
            "total_files": total_files,
            "total_concepts": len(concepts_list),
            "total_coverage_pairs": sum(len(v) for v in coverage_matrix.values()),
            "avg_concepts_per_file": round(sum(concept_counts) / len(concept_counts), 1) if concept_counts else 0,
            "min_concepts_per_file": min(concept_counts) if concept_counts else 0,
            "max_concepts_per_file": max(concept_counts) if concept_counts else 0,
            "gaps_count": len(gaps),
            "overlaps_count": len(overlaps),
            "gap_threshold": gap_threshold,
            "overlap_threshold_percent": int(overlap_threshold * 100),
        }

        logger.info(
            f"[doc_coverage] Coverage: {statistics['total_coverage_pairs']} pairs, "
            f"{statistics['gaps_count']} gaps, {statistics['overlaps_count']} overlaps"
        )

        return {
            "coverage_matrix": coverage_matrix,
            "file_coverage": file_coverage,
            "concept_coverage": concept_coverage,
            "cluster_coverage": cluster_coverage,
            "cluster_similarity": cluster_similarity,
            "gaps": gaps,
            "overlaps": overlaps,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total_files = stats.get("total_files", 0)
        total_concepts = stats.get("total_concepts", 0)
        avg_per_file = stats.get("avg_concepts_per_file", 0)
        gaps = stats.get("gaps_count", 0)
        overlaps = stats.get("overlaps_count", 0)

        return (
            f"Coverage: {total_concepts} concepts across {total_files} files "
            f"(avg {avg_per_file}/file). {gaps} gaps, {overlaps} overlapping concepts."
        )
