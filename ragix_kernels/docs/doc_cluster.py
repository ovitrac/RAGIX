"""
Kernel: Document Cluster
Stage: 2 (Analysis)

Groups documents by topic similarity using concept-based vectors.
Supports two clustering methods:
- Hierarchical (agglomerative) clustering — default, no external dependencies
- Leiden community detection — GraphRAG-compatible, requires python-igraph

The output provides clusters at multiple resolutions for hierarchical
summarization (document → group → domain).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
import logging
import json
import math

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocClusterKernel(Kernel):
    """
    Cluster documents by topic similarity.

    This kernel groups documents based on their concept overlap:
    1. Builds file-concept vectors from doc_concepts output
    2. Computes pairwise similarity (Jaccard or cosine)
    3. Applies clustering algorithm (hierarchical or Leiden)
    4. Returns clusters at multiple granularity levels

    Configuration options:
        method: Clustering method ("hierarchical" or "leiden", default: "hierarchical")
        n_clusters: Target number of clusters ("auto" or integer, default: "auto")
        min_cluster_size: Minimum files per cluster (default: 2)
        similarity_metric: "jaccard" or "cosine" (default: "jaccard")
        linkage: For hierarchical: "average", "complete", "single" (default: "average")
        resolutions: For Leiden: list of resolution parameters (default: [0.5, 1.0, 2.0])

    Dependencies:
        doc_metadata: File inventory
        doc_concepts: File-concept mappings

    Output:
        clusters: List of clusters with file_ids, concepts, labels
        hierarchy: Multi-level cluster hierarchy
        file_vectors: File-concept vector representation
        similarity_matrix: Pairwise file similarities (optional)
        statistics: Clustering statistics
    """

    name = "doc_cluster"
    version = "1.0.0"
    category = "docs"
    stage = 2
    description = "Cluster documents by topic similarity"

    requires = ["doc_metadata", "doc_concepts"]
    provides = ["doc_clusters", "cluster_hierarchy", "file_vectors"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Cluster documents based on concept vectors."""

        # Get configuration
        method = input.config.get("method", "hierarchical")
        n_clusters = input.config.get("n_clusters", "auto")
        min_cluster_size = input.config.get("min_cluster_size", 2)
        similarity_metric = input.config.get("similarity_metric", "jaccard")
        linkage = input.config.get("linkage", "average")
        resolutions = input.config.get("resolutions", [0.5, 1.0, 2.0])

        logger.info(f"[doc_cluster] Clustering with method={method}")

        # Load dependencies
        metadata_path = input.dependencies.get("doc_metadata")
        concepts_path = input.dependencies.get("doc_concepts")

        if not metadata_path or not metadata_path.exists():
            raise RuntimeError("Missing required dependency: doc_metadata")
        if not concepts_path or not concepts_path.exists():
            raise RuntimeError("Missing required dependency: doc_concepts")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})
        with open(concepts_path) as f:
            concepts_data = json.load(f).get("data", {})

        # Get file-concept mappings
        file_concepts = concepts_data.get("file_concepts", {})
        concept_files = concepts_data.get("concept_files", {})
        concepts_list = concepts_data.get("concepts", [])

        # Build concept index for labeling
        concept_labels = {c["concept_id"]: c["label"] for c in concepts_list}

        # Get file info for cluster labeling
        files = metadata_data.get("files", [])
        file_paths = {f["file_id"]: f["path"] for f in files}

        # Filter to files that have concepts
        valid_file_ids = [fid for fid in file_paths.keys() if fid in file_concepts]
        logger.info(f"[doc_cluster] {len(valid_file_ids)} files with concept coverage")

        if len(valid_file_ids) < 2:
            logger.warning("[doc_cluster] Not enough files for clustering")
            return self._empty_result(valid_file_ids, file_concepts)

        # Build file vectors (file_id → set of concept_ids)
        file_vectors: Dict[str, Set[str]] = {
            fid: set(file_concepts.get(fid, []))
            for fid in valid_file_ids
        }

        # Compute similarity matrix
        logger.info(f"[doc_cluster] Computing {similarity_metric} similarity matrix")
        similarity_matrix = self._compute_similarity_matrix(
            valid_file_ids, file_vectors, similarity_metric
        )

        # Determine number of clusters
        if n_clusters == "auto":
            # Heuristic: sqrt(n) clusters, but at least 2 and at most 20
            n = len(valid_file_ids)
            n_clusters = max(2, min(20, int(math.sqrt(n))))
        else:
            n_clusters = int(n_clusters)

        logger.info(f"[doc_cluster] Target clusters: {n_clusters}")

        # Apply clustering method
        if method == "leiden":
            clusters, hierarchy = self._leiden_cluster(
                valid_file_ids, similarity_matrix, resolutions, min_cluster_size
            )
        else:
            # Default: hierarchical clustering
            clusters, hierarchy = self._hierarchical_cluster(
                valid_file_ids, similarity_matrix, n_clusters, linkage, min_cluster_size
            )

        # Enrich clusters with concept labels and generate cluster labels
        enriched_clusters = []
        for i, cluster in enumerate(clusters):
            file_ids = cluster["file_ids"]

            # Get all concepts in this cluster
            cluster_concepts: Dict[str, int] = defaultdict(int)
            for fid in file_ids:
                for cid in file_vectors.get(fid, set()):
                    cluster_concepts[cid] += 1

            # Sort concepts by frequency within cluster
            sorted_concepts = sorted(
                cluster_concepts.items(), key=lambda x: -x[1]
            )
            top_concept_ids = [c[0] for c in sorted_concepts[:5]]
            top_concept_labels = [concept_labels.get(c, c) for c in top_concept_ids]

            # Generate cluster label from top concepts
            label = " / ".join(top_concept_labels[:3]) if top_concept_labels else f"Cluster {i+1}"

            enriched_clusters.append({
                "cluster_id": f"CL{i+1:03d}",
                "label": label,
                "file_ids": file_ids,
                "file_count": len(file_ids),
                "file_paths": [file_paths.get(fid, fid) for fid in file_ids],
                "centroid_concepts": top_concept_ids,
                "centroid_labels": top_concept_labels,
                "concept_coverage": len(cluster_concepts),
            })

        # Statistics
        statistics = {
            "method": method,
            "total_files": len(valid_file_ids),
            "total_clusters": len(enriched_clusters),
            "avg_cluster_size": (
                round(sum(c["file_count"] for c in enriched_clusters) / len(enriched_clusters), 1)
                if enriched_clusters else 0
            ),
            "min_cluster_size": min(c["file_count"] for c in enriched_clusters) if enriched_clusters else 0,
            "max_cluster_size": max(c["file_count"] for c in enriched_clusters) if enriched_clusters else 0,
            "similarity_metric": similarity_metric,
            "n_clusters_requested": n_clusters if isinstance(n_clusters, int) else "auto",
        }

        logger.info(
            f"[doc_cluster] Created {statistics['total_clusters']} clusters "
            f"(avg size: {statistics['avg_cluster_size']})"
        )

        return {
            "clusters": enriched_clusters,
            "hierarchy": hierarchy,
            "file_vectors": {fid: list(v) for fid, v in file_vectors.items()},
            "statistics": statistics,
        }

    def _compute_similarity_matrix(
        self,
        file_ids: List[str],
        file_vectors: Dict[str, Set[str]],
        metric: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise similarity matrix."""
        matrix: Dict[str, Dict[str, float]] = {}

        for i, fid1 in enumerate(file_ids):
            matrix[fid1] = {}
            v1 = file_vectors[fid1]

            for j, fid2 in enumerate(file_ids):
                if i == j:
                    matrix[fid1][fid2] = 1.0
                elif j < i:
                    # Use symmetric value
                    matrix[fid1][fid2] = matrix[fid2][fid1]
                else:
                    v2 = file_vectors[fid2]
                    if metric == "jaccard":
                        sim = self._jaccard_similarity(v1, v2)
                    else:
                        sim = self._cosine_similarity(v1, v2)
                    matrix[fid1][fid2] = sim

        return matrix

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute cosine similarity (binary vectors)."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        magnitude = math.sqrt(len(set1)) * math.sqrt(len(set2))
        return intersection / magnitude if magnitude > 0 else 0.0

    def _hierarchical_cluster(
        self,
        file_ids: List[str],
        similarity_matrix: Dict[str, Dict[str, float]],
        n_clusters: int,
        linkage: str,
        min_cluster_size: int,
    ) -> Tuple[List[Dict], Dict]:
        """
        Agglomerative hierarchical clustering.

        This is a pure Python implementation that doesn't require external
        dependencies. Uses distance = 1 - similarity.
        """
        n = len(file_ids)

        # Convert similarity to distance matrix
        dist_matrix = {
            fid1: {fid2: 1.0 - sim for fid2, sim in row.items()}
            for fid1, row in similarity_matrix.items()
        }

        # Initialize: each file is its own cluster
        clusters: List[Set[str]] = [set([fid]) for fid in file_ids]
        merge_history: List[Dict] = []

        # Agglomerative merging
        while len(clusters) > n_clusters:
            # Find closest pair of clusters
            min_dist = float("inf")
            merge_i, merge_j = 0, 1

            for i, c1 in enumerate(clusters):
                for j, c2 in enumerate(clusters):
                    if i >= j:
                        continue
                    dist = self._cluster_distance(c1, c2, dist_matrix, linkage)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Merge clusters
            merged = clusters[merge_i] | clusters[merge_j]
            merge_history.append({
                "step": len(merge_history) + 1,
                "merged": [list(clusters[merge_i]), list(clusters[merge_j])],
                "distance": min_dist,
                "result_size": len(merged),
            })

            # Remove old clusters, add merged
            clusters = [
                c for idx, c in enumerate(clusters)
                if idx not in (merge_i, merge_j)
            ]
            clusters.append(merged)

        # Convert to output format
        result_clusters = [
            {"file_ids": list(c)} for c in clusters
            if len(c) >= min_cluster_size
        ]

        # Handle small clusters: merge into nearest larger cluster or create "other"
        small_files = [
            fid for c in clusters
            if len(c) < min_cluster_size
            for fid in c
        ]
        if small_files:
            result_clusters.append({"file_ids": small_files})

        hierarchy = {
            "method": "hierarchical",
            "linkage": linkage,
            "merge_history": merge_history[-10:],  # Keep last 10 merges
        }

        return result_clusters, hierarchy

    def _cluster_distance(
        self,
        c1: Set[str],
        c2: Set[str],
        dist_matrix: Dict[str, Dict[str, float]],
        linkage: str,
    ) -> float:
        """Compute distance between two clusters."""
        distances = []
        for fid1 in c1:
            for fid2 in c2:
                d = dist_matrix.get(fid1, {}).get(fid2, 1.0)
                distances.append(d)

        if not distances:
            return 1.0

        if linkage == "single":
            return min(distances)
        elif linkage == "complete":
            return max(distances)
        else:  # average
            return sum(distances) / len(distances)

    def _leiden_cluster(
        self,
        file_ids: List[str],
        similarity_matrix: Dict[str, Dict[str, float]],
        resolutions: List[float],
        min_cluster_size: int,
    ) -> Tuple[List[Dict], Dict]:
        """
        Leiden community detection (GraphRAG-compatible).

        Requires: python-igraph, leidenalg
        Falls back to hierarchical if not available.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            logger.warning(
                "[doc_cluster] Leiden requires igraph+leidenalg, "
                "falling back to hierarchical"
            )
            return self._hierarchical_cluster(
                file_ids, similarity_matrix, len(file_ids) // 5 + 2, "average", min_cluster_size
            )

        # Build igraph from similarity matrix
        n = len(file_ids)
        fid_to_idx = {fid: i for i, fid in enumerate(file_ids)}

        edges = []
        weights = []
        threshold = 0.1  # Minimum similarity to create edge

        for fid1, row in similarity_matrix.items():
            for fid2, sim in row.items():
                if fid1 < fid2 and sim > threshold:
                    edges.append((fid_to_idx[fid1], fid_to_idx[fid2]))
                    weights.append(sim)

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights
        g.vs["file_id"] = file_ids

        # Run Leiden at multiple resolutions
        partitions_by_resolution = {}
        best_partition = None
        best_modularity = -1

        for resolution in resolutions:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                weights="weight",
            )
            modularity = partition.modularity
            partitions_by_resolution[resolution] = {
                "communities": [list(p) for p in partition],
                "modularity": modularity,
                "n_communities": len(partition),
            }

            if modularity > best_modularity:
                best_modularity = modularity
                best_partition = partition

        # Use best partition
        clusters = []
        for community_indices in best_partition:
            community_file_ids = [file_ids[i] for i in community_indices]
            if len(community_file_ids) >= min_cluster_size:
                clusters.append({"file_ids": community_file_ids})

        hierarchy = {
            "method": "leiden",
            "resolutions": partitions_by_resolution,
            "best_resolution": [
                r for r, p in partitions_by_resolution.items()
                if p["modularity"] == best_modularity
            ][0],
            "modularity": best_modularity,
        }

        return clusters, hierarchy

    def _empty_result(
        self,
        file_ids: List[str],
        file_concepts: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Return empty result when clustering not possible."""
        return {
            "clusters": [{
                "cluster_id": "CL001",
                "label": "All Documents",
                "file_ids": file_ids,
                "file_count": len(file_ids),
                "file_paths": [],
                "centroid_concepts": [],
                "centroid_labels": [],
                "concept_coverage": 0,
            }] if file_ids else [],
            "hierarchy": {"method": "none", "reason": "insufficient_files"},
            "file_vectors": {fid: list(file_concepts.get(fid, [])) for fid in file_ids},
            "statistics": {
                "method": "none",
                "total_files": len(file_ids),
                "total_clusters": 1 if file_ids else 0,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        method = stats.get("method", "unknown")
        total_files = stats.get("total_files", 0)
        total_clusters = stats.get("total_clusters", 0)
        avg_size = stats.get("avg_cluster_size", 0)

        clusters = data.get("clusters", [])
        top_labels = [c["label"][:30] for c in clusters[:3]]
        labels_str = ", ".join(top_labels) if top_labels else "none"

        return (
            f"Clustering ({method}): {total_files} files → {total_clusters} clusters "
            f"(avg {avg_size} files). Top: {labels_str}."
        )
