"""
Document Cluster Reconciliation Kernel — Reconcile hierarchical and Leiden clustering.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-18

This kernel reconciles results from two clustering methods:
1. Hierarchical clustering (doc_cluster) — based on path structure and concepts
2. Leiden community detection (doc_cluster_leiden) — based on content similarity

Decision logic:
- If cluster overlap > 80%: produce unified view
- If overlap 50-80%: attempt reconciliation with divergence notes
- If overlap < 50%: keep BOTH views (dual perspective)

The dual-view approach is preferred for limited-context LLMs, as it avoids
hallucination from forced reconciliation.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of cluster reconciliation."""
    mode: str  # "unified", "reconciled", "dual_view"
    agreement_score: float
    hierarchical_clusters: List[Dict]
    leiden_clusters: List[Dict]
    unified_clusters: Optional[List[Dict]]
    divergences: List[Dict]
    methodology: str


class DocClusterReconcileKernel(Kernel):
    """
    Reconcile hierarchical and Leiden clustering results.

    This kernel does NOT force reconciliation. Instead, it:
    1. Computes overlap between clustering methods
    2. Documents divergences explicitly
    3. ALWAYS preserves both views for transparency

    For limited-context LLMs, dual-view is safer than forced reconciliation.
    """

    name = "doc_cluster_reconcile"
    version = "1.0.0"
    category = "docs"
    stage = 2

    requires = ["doc_cluster", "doc_cluster_leiden"]
    provides = ["reconciled_clusters", "clustering_comparison"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Execute cluster reconciliation."""
        # Load dependencies
        hier_path = input.dependencies.get("doc_cluster")
        leiden_path = input.dependencies.get("doc_cluster_leiden")

        if not hier_path or not hier_path.exists():
            raise RuntimeError("Missing required dependency: doc_cluster")
        if not leiden_path or not leiden_path.exists():
            raise RuntimeError("Missing required dependency: doc_cluster_leiden")

        with open(hier_path) as f:
            hier_data = json.load(f).get("data", {})
        with open(leiden_path) as f:
            leiden_data = json.load(f).get("data", {})

        hier_clusters = hier_data.get("clusters", [])
        leiden_clusters = leiden_data.get("optimal_clusters", [])

        # Compute agreement metrics
        agreement = self._compute_agreement(hier_clusters, leiden_clusters)

        # Analyze divergences
        divergences = self._analyze_divergences(hier_clusters, leiden_clusters)

        # Determine mode based on agreement
        mode, methodology = self._determine_mode(agreement["overall_score"])

        # Build result based on mode
        result = {
            "mode": mode,
            "agreement": agreement,
            "methodology": methodology,
            "hierarchical": {
                "clusters": hier_clusters,
                "n_clusters": len(hier_clusters),
                "method": hier_data.get("method", "hierarchical")
            },
            "leiden": {
                "clusters": leiden_clusters,
                "n_clusters": len(leiden_clusters),
                "optimal_resolution": leiden_data.get("optimal_resolution"),
                "method": leiden_data.get("method", "leiden")
            },
            "divergences": divergences,
            "summary": self._build_summary(mode, agreement, divergences)
        }

        # If high agreement, also produce unified view
        if mode == "unified":
            result["unified_clusters"] = self._merge_clusters(hier_clusters, leiden_clusters)
        elif mode == "reconciled":
            result["unified_clusters"] = self._reconcile_clusters(hier_clusters, leiden_clusters)

        return result

    def _compute_agreement(
        self,
        hier_clusters: List[Dict],
        leiden_clusters: List[Dict]
    ) -> Dict[str, Any]:
        """Compute agreement metrics between clustering methods."""
        # Build file->cluster mappings
        hier_map = {}
        for cluster in hier_clusters:
            cluster_id = cluster.get("id", "")
            for file_id in cluster.get("file_ids", []):
                hier_map[file_id] = cluster_id

        leiden_map = {}
        for cluster in leiden_clusters:
            cluster_id = cluster.get("id", "")
            for file_id in cluster.get("file_ids", []):
                leiden_map[file_id] = cluster_id

        # Compute pairwise agreement
        all_files = set(hier_map.keys()) | set(leiden_map.keys())
        if len(all_files) < 2:
            return {"overall_score": 1.0, "pairs_compared": 0}

        agree_count = 0
        total_pairs = 0

        files_list = list(all_files)
        for i in range(len(files_list)):
            for j in range(i + 1, len(files_list)):
                f1, f2 = files_list[i], files_list[j]

                hier_same = hier_map.get(f1) == hier_map.get(f2)
                leiden_same = leiden_map.get(f1) == leiden_map.get(f2)

                if hier_same == leiden_same:
                    agree_count += 1
                total_pairs += 1

        overall_score = agree_count / total_pairs if total_pairs > 0 else 1.0

        # Compute cluster-level Jaccard similarity
        cluster_jaccard = self._compute_cluster_jaccard(hier_clusters, leiden_clusters)

        return {
            "overall_score": overall_score,
            "pairs_compared": total_pairs,
            "pairs_agreed": agree_count,
            "cluster_jaccard": cluster_jaccard
        }

    def _compute_cluster_jaccard(
        self,
        hier_clusters: List[Dict],
        leiden_clusters: List[Dict]
    ) -> float:
        """Compute average best-match Jaccard between clusters."""
        if not hier_clusters or not leiden_clusters:
            return 0.0

        jaccard_scores = []

        for h_cluster in hier_clusters:
            h_files = set(h_cluster.get("file_ids", []))
            best_jaccard = 0.0

            for l_cluster in leiden_clusters:
                l_files = set(l_cluster.get("file_ids", []))
                intersection = len(h_files & l_files)
                union = len(h_files | l_files)
                jaccard = intersection / union if union > 0 else 0.0
                best_jaccard = max(best_jaccard, jaccard)

            jaccard_scores.append(best_jaccard)

        return sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    def _analyze_divergences(
        self,
        hier_clusters: List[Dict],
        leiden_clusters: List[Dict]
    ) -> List[Dict]:
        """Identify specific divergences between clustering methods."""
        divergences = []

        # Build file->cluster mappings with labels
        hier_map = {}
        for cluster in hier_clusters:
            cluster_id = cluster.get("id", "")
            cluster_label = cluster.get("label", cluster_id)
            for file_id in cluster.get("file_ids", []):
                hier_map[file_id] = {"id": cluster_id, "label": cluster_label}

        leiden_map = {}
        for cluster in leiden_clusters:
            cluster_id = cluster.get("id", "")
            cluster_label = cluster.get("label", cluster_id)
            for file_id in cluster.get("file_ids", []):
                leiden_map[file_id] = {"id": cluster_id, "label": cluster_label}

        # Find files assigned to different semantic groups
        all_files = set(hier_map.keys()) & set(leiden_map.keys())

        # Group files by their cluster assignments
        assignments = defaultdict(list)
        for file_id in all_files:
            h_label = hier_map[file_id]["label"]
            l_label = leiden_map[file_id]["label"]
            assignments[(h_label, l_label)].append(file_id)

        # Identify divergent assignments (different labels)
        for (h_label, l_label), files in assignments.items():
            if h_label != l_label:
                divergences.append({
                    "type": "assignment_divergence",
                    "hierarchical_cluster": h_label,
                    "leiden_cluster": l_label,
                    "files": files[:10],  # Limit for readability
                    "file_count": len(files),
                    "description": (
                        f"{len(files)} file(s) assigned to '{h_label}' by hierarchical "
                        f"but to '{l_label}' by Leiden"
                    )
                })

        # Check for clusters with no good match
        for h_cluster in hier_clusters:
            h_files = set(h_cluster.get("file_ids", []))
            h_label = h_cluster.get("label", h_cluster.get("id"))

            best_overlap = 0
            for l_cluster in leiden_clusters:
                l_files = set(l_cluster.get("file_ids", []))
                overlap = len(h_files & l_files) / len(h_files) if h_files else 0
                best_overlap = max(best_overlap, overlap)

            if best_overlap < 0.5:
                divergences.append({
                    "type": "no_match",
                    "method": "hierarchical",
                    "cluster": h_label,
                    "best_overlap": best_overlap,
                    "description": (
                        f"Hierarchical cluster '{h_label}' has no strong match "
                        f"in Leiden (best overlap: {best_overlap:.0%})"
                    )
                })

        return divergences

    def _determine_mode(self, agreement_score: float) -> Tuple[str, str]:
        """Determine reconciliation mode based on agreement."""
        if agreement_score > 0.8:
            return "unified", (
                "Both clustering methods produced highly similar results "
                f"(agreement: {agreement_score:.0%}). A unified view is presented, "
                "using hierarchical labels validated by Leiden community detection."
            )
        elif agreement_score > 0.5:
            return "reconciled", (
                "Clustering methods showed moderate agreement "
                f"({agreement_score:.0%}). A reconciled view is presented with "
                "divergences noted. Both original views are preserved for reference."
            )
        else:
            return "dual_view", (
                "Clustering methods produced different perspectives "
                f"(agreement: {agreement_score:.0%}). BOTH views are preserved. "
                "Hierarchical clustering groups by path structure and explicit concepts. "
                "Leiden clustering groups by content similarity patterns. "
                "Consult both perspectives for comprehensive understanding. "
                "This dual-view approach prevents hallucination from forced reconciliation."
            )

    def _merge_clusters(
        self,
        hier_clusters: List[Dict],
        leiden_clusters: List[Dict]
    ) -> List[Dict]:
        """Merge clusters for unified view (high agreement case)."""
        # Use hierarchical as base, enrich with Leiden concepts
        merged = []

        for h_cluster in hier_clusters:
            h_files = set(h_cluster.get("file_ids", []))

            # Find best matching Leiden cluster
            best_match = None
            best_overlap = 0
            for l_cluster in leiden_clusters:
                l_files = set(l_cluster.get("file_ids", []))
                overlap = len(h_files & l_files)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = l_cluster

            # Merge concepts
            h_concepts = set(h_cluster.get("centroid_concepts", []))
            if best_match:
                l_concepts = set(best_match.get("centroid_concepts", []))
                merged_concepts = list(h_concepts | l_concepts)[:7]
            else:
                merged_concepts = list(h_concepts)

            merged.append({
                "id": h_cluster.get("id"),
                "label": h_cluster.get("label"),
                "file_ids": h_cluster.get("file_ids", []),
                "centroid_concepts": merged_concepts,
                "validation": "unified",
                "leiden_match": best_match.get("id") if best_match else None
            })

        return merged

    def _reconcile_clusters(
        self,
        hier_clusters: List[Dict],
        leiden_clusters: List[Dict]
    ) -> List[Dict]:
        """Reconcile clusters for moderate agreement case."""
        # Similar to merge but note divergences
        reconciled = []

        for h_cluster in hier_clusters:
            h_files = set(h_cluster.get("file_ids", []))

            # Find matching Leiden clusters
            matches = []
            for l_cluster in leiden_clusters:
                l_files = set(l_cluster.get("file_ids", []))
                overlap = len(h_files & l_files) / len(h_files) if h_files else 0
                if overlap > 0.3:
                    matches.append({
                        "id": l_cluster.get("id"),
                        "label": l_cluster.get("label"),
                        "overlap": overlap
                    })

            reconciled.append({
                "id": h_cluster.get("id"),
                "label": h_cluster.get("label"),
                "file_ids": h_cluster.get("file_ids", []),
                "centroid_concepts": h_cluster.get("centroid_concepts", []),
                "validation": "reconciled",
                "leiden_matches": matches,
                "split": len(matches) > 1
            })

        return reconciled

    def _build_summary(
        self,
        mode: str,
        agreement: Dict,
        divergences: List[Dict]
    ) -> str:
        """Build human-readable summary."""
        score = agreement.get("overall_score", 0)
        n_divergences = len(divergences)

        if mode == "unified":
            return (
                f"Clustering methods agree ({score:.0%}). Unified view provided."
            )
        elif mode == "reconciled":
            return (
                f"Moderate agreement ({score:.0%}). Reconciled view with "
                f"{n_divergences} divergence(s) noted."
            )
        else:
            return (
                f"Methods differ ({score:.0%}). Dual view preserved with "
                f"{n_divergences} divergence(s). Both perspectives valid."
            )

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        mode = data.get("mode", "unknown")
        agreement = data.get("agreement", {})
        score = agreement.get("overall_score", 0)
        n_hier = data.get("hierarchical", {}).get("n_clusters", 0)
        n_leiden = data.get("leiden", {}).get("n_clusters", 0)

        return (
            f"Cluster reconciliation: {mode} mode (agreement: {score:.0%}). "
            f"Hierarchical: {n_hier} clusters, Leiden: {n_leiden} clusters. "
            f"{data.get('summary', '')}"
        )
