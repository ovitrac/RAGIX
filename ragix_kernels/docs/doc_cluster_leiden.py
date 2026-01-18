"""
Document Leiden Clustering Kernel — Community detection using Leiden algorithm.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-18

This kernel provides an alternative clustering approach to hierarchical clustering,
using the Leiden algorithm for community detection. It runs in parallel with
doc_cluster (hierarchical) to provide dual perspectives on document organization.

The Leiden algorithm:
- Discovers communities in graphs based on modularity optimization
- Supports multi-resolution analysis (different granularity levels)
- Produces hierarchical communities naturally

References:
- Traag, V.A., Waltman, L. & van Eck, N.J. (2019). "From Louvain to Leiden:
  guaranteeing well-connected communities". Scientific Reports 9, 5233.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    logger.warning("python-igraph not available, Leiden clustering will use fallback")

try:
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logger.warning("leidenalg not available, Leiden clustering will use fallback")


@dataclass
class LeidenConfig:
    """Configuration for Leiden clustering."""
    resolutions: List[float] = None  # Multi-resolution analysis
    min_community_size: int = 2
    seed: int = 42  # For reproducibility

    def __post_init__(self):
        if self.resolutions is None:
            # Note: res=2.0 can cause exponential computation on dense graphs
            # Keeping max at 1.0 for reasonable performance
            self.resolutions = [0.1, 0.5, 1.0]


class DocClusterLeidenKernel(Kernel):
    """
    Leiden community detection on document similarity graph.

    Uses python-igraph + leidenalg for multi-resolution clustering.
    Produces communities at multiple resolution levels for comparison.

    If leidenalg is not available, falls back to igraph's built-in
    community detection (multilevel/Louvain).
    """

    name = "doc_cluster_leiden"
    version = "1.0.0"
    category = "docs"
    stage = 2

    requires = ["doc_metadata", "doc_concepts"]
    provides = ["leiden_communities", "resolution_levels"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Execute Leiden community detection."""
        import time

        # Load dependencies
        logger.info("[doc_cluster_leiden] Loading dependencies...")
        t0 = time.time()

        metadata_path = input.dependencies.get("doc_metadata")
        concepts_path = input.dependencies.get("doc_concepts")

        if not metadata_path or not metadata_path.exists():
            raise RuntimeError("Missing required dependency: doc_metadata")
        if not concepts_path or not concepts_path.exists():
            raise RuntimeError("Missing required dependency: doc_concepts")

        with open(metadata_path) as f:
            metadata = json.load(f).get("data", {})
        with open(concepts_path) as f:
            concepts = json.load(f).get("data", {})
        logger.info(f"[doc_cluster_leiden] Dependencies loaded in {time.time()-t0:.2f}s")

        # Get configuration
        # Note: res=2.0 removed from defaults due to exponential computation on dense graphs
        config = LeidenConfig(
            resolutions=input.config.get("leiden_resolutions", [0.1, 0.5, 1.0]),
            min_community_size=input.config.get("min_community_size", 2),
            seed=input.config.get("seed", 42)
        )

        # Build file-concept vectors
        logger.info("[doc_cluster_leiden] Building file-concept vectors...")
        t1 = time.time()
        file_concepts = self._build_file_concept_vectors(concepts)
        logger.info(f"[doc_cluster_leiden] Vectors built in {time.time()-t1:.2f}s ({len(file_concepts)} files)")

        if not file_concepts:
            return {
                "communities": {},
                "optimal_resolution": None,
                "optimal_clusters": [],
                "method": "none",
                "error": "No file-concept data available"
            }

        # Build similarity graph
        file_ids = list(file_concepts.keys())
        logger.info(f"[doc_cluster_leiden] Computing similarity matrix for {len(file_ids)} files...")
        t2 = time.time()
        similarity_matrix = self._compute_similarity_matrix(file_concepts, file_ids)
        logger.info(f"[doc_cluster_leiden] Similarity matrix computed in {time.time()-t2:.2f}s")

        # Check if we can use Leiden
        if IGRAPH_AVAILABLE:
            logger.info("[doc_cluster_leiden] Building igraph...")
            t3 = time.time()
            graph = self._build_igraph(file_ids, similarity_matrix)
            logger.info(f"[doc_cluster_leiden] Graph built in {time.time()-t3:.2f}s ({graph.vcount()} vertices, {graph.ecount()} edges)")

            logger.info(f"[doc_cluster_leiden] Detecting communities at {len(config.resolutions)} resolutions...")
            t4 = time.time()
            communities = self._detect_communities(graph, file_ids, config)
            logger.info(f"[doc_cluster_leiden] Communities detected in {time.time()-t4:.2f}s")
        else:
            # Fallback: use simple threshold-based clustering
            logger.info("[doc_cluster_leiden] Using fallback clustering (no igraph)")
            communities = self._fallback_clustering(file_ids, similarity_matrix, config)

        # Select optimal resolution
        logger.info("[doc_cluster_leiden] Selecting optimal resolution...")
        t5 = time.time()
        optimal_res, optimal_clusters = self._select_optimal(communities, file_ids)
        logger.info(f"[doc_cluster_leiden] Optimal: resolution={optimal_res}, {len(optimal_clusters)} clusters in {time.time()-t5:.2f}s")

        # Enrich clusters with metadata
        logger.info("[doc_cluster_leiden] Enriching clusters...")
        t6 = time.time()
        enriched_clusters = self._enrich_clusters(optimal_clusters, metadata, concepts)
        logger.info(f"[doc_cluster_leiden] Enrichment done in {time.time()-t6:.2f}s")

        return {
            "communities": communities,
            "optimal_resolution": optimal_res,
            "optimal_clusters": enriched_clusters,
            "file_count": len(file_ids),
            "method": "leiden" if (IGRAPH_AVAILABLE and LEIDEN_AVAILABLE) else
                      "louvain" if IGRAPH_AVAILABLE else "threshold",
            "resolutions_analyzed": list(communities.keys()) if communities else []
        }

    def _build_file_concept_vectors(self, concepts: Dict) -> Dict[str, Dict[str, float]]:
        """Build file-concept vectors from concept data."""
        file_vectors = defaultdict(dict)

        for concept_entry in concepts.get("concepts", []):
            # Support both old format (name, files) and new format (label, file_ids)
            concept_name = concept_entry.get("label", concept_entry.get("name", ""))
            frequency = concept_entry.get("file_count", concept_entry.get("frequency", 1))
            file_ids = concept_entry.get("file_ids", concept_entry.get("files", []))

            for file_id in file_ids:
                # TF-IDF-like weighting
                file_vectors[file_id][concept_name] = 1.0 / (1.0 + frequency * 0.1)

        return dict(file_vectors)

    def _compute_similarity_matrix(
        self,
        file_concepts: Dict[str, Dict[str, float]],
        file_ids: List[str]
    ) -> List[List[float]]:
        """Compute Jaccard similarity matrix between files."""
        n = len(file_ids)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            concepts_i = set(file_concepts.get(file_ids[i], {}).keys())
            for j in range(i + 1, n):
                concepts_j = set(file_concepts.get(file_ids[j], {}).keys())

                if concepts_i or concepts_j:
                    intersection = len(concepts_i & concepts_j)
                    union = len(concepts_i | concepts_j)
                    similarity = intersection / union if union > 0 else 0.0
                else:
                    similarity = 0.0

                matrix[i][j] = similarity
                matrix[j][i] = similarity

        return matrix

    def _build_igraph(
        self,
        file_ids: List[str],
        similarity_matrix: List[List[float]],
        threshold: float = 0.1
    ) -> "ig.Graph":
        """Build igraph Graph from similarity matrix."""
        n = len(file_ids)
        edges = []
        weights = []

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > threshold:
                    edges.append((i, j))
                    weights.append(similarity_matrix[i][j])

        graph = ig.Graph(n=n, edges=edges, directed=False)
        graph.vs["name"] = file_ids
        graph.es["weight"] = weights

        return graph

    def _detect_communities(
        self,
        graph: "ig.Graph",
        file_ids: List[str],
        config: LeidenConfig
    ) -> Dict[float, List[Dict]]:
        """Detect communities at multiple resolutions."""
        import time
        communities = {}

        for resolution in config.resolutions:
            t_res = time.time()
            if LEIDEN_AVAILABLE:
                # Use Leiden algorithm
                partition = leidenalg.find_partition(
                    graph,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,
                    seed=config.seed,
                    weights="weight"
                )
                membership = partition.membership
                modularity = partition.modularity
            else:
                # Fallback to igraph's multilevel (Louvain)
                partition = graph.community_multilevel(weights="weight")
                membership = partition.membership
                modularity = partition.modularity

            # Convert to cluster structure
            clusters_dict = defaultdict(list)
            for idx, comm_id in enumerate(membership):
                clusters_dict[comm_id].append(file_ids[idx])

            # Filter by minimum size
            clusters = []
            for comm_id, files in sorted(clusters_dict.items()):
                if len(files) >= config.min_community_size:
                    clusters.append({
                        "id": f"L{resolution:.1f}_C{comm_id:02d}",
                        "resolution": resolution,
                        "file_ids": files,
                        "size": len(files)
                    })

            communities[resolution] = {
                "clusters": clusters,
                "modularity": modularity,
                "n_clusters": len(clusters)
            }
            logger.info(f"[doc_cluster_leiden] res={resolution}: {len(clusters)} communities (mod={modularity:.3f}) in {time.time()-t_res:.2f}s")

        return communities

    def _fallback_clustering(
        self,
        file_ids: List[str],
        similarity_matrix: List[List[float]],
        config: LeidenConfig
    ) -> Dict[float, List[Dict]]:
        """Fallback clustering using threshold-based grouping."""
        communities = {}

        for resolution in config.resolutions:
            # Higher resolution = more clusters = higher threshold
            threshold = 0.3 + (resolution * 0.1)

            # Simple connected components at threshold
            clusters = self._threshold_clustering(file_ids, similarity_matrix, threshold)

            communities[resolution] = {
                "clusters": clusters,
                "modularity": 0.0,  # Not computed
                "n_clusters": len(clusters)
            }

        return communities

    def _threshold_clustering(
        self,
        file_ids: List[str],
        similarity_matrix: List[List[float]],
        threshold: float
    ) -> List[Dict]:
        """Simple threshold-based clustering."""
        n = len(file_ids)
        visited = [False] * n
        clusters = []
        cluster_id = 0

        def dfs(start: int, cluster_files: List[str]):
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                cluster_files.append(file_ids[node])
                for j in range(n):
                    if not visited[j] and similarity_matrix[node][j] > threshold:
                        stack.append(j)

        for i in range(n):
            if not visited[i]:
                cluster_files = []
                dfs(i, cluster_files)
                if len(cluster_files) >= 2:
                    clusters.append({
                        "id": f"T{threshold:.1f}_C{cluster_id:02d}",
                        "resolution": threshold,
                        "file_ids": cluster_files,
                        "size": len(cluster_files)
                    })
                    cluster_id += 1

        return clusters

    def _select_optimal(
        self,
        communities: Dict[float, Dict],
        file_ids: List[str]
    ) -> Tuple[Optional[float], List[Dict]]:
        """Select optimal resolution based on modularity and cluster distribution."""
        if not communities:
            return None, []

        best_res = None
        best_score = -1

        target_clusters = max(3, int(len(file_ids) ** 0.5))  # sqrt(n) heuristic

        for resolution, data in communities.items():
            modularity = data.get("modularity", 0)
            n_clusters = data.get("n_clusters", 0)

            # Score: balance modularity and target cluster count
            cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
            score = modularity - (0.2 * cluster_penalty)

            if score > best_score:
                best_score = score
                best_res = resolution

        optimal_clusters = communities.get(best_res, {}).get("clusters", [])
        return best_res, optimal_clusters

    def _enrich_clusters(
        self,
        clusters: List[Dict],
        metadata: Dict,
        concepts: Dict
    ) -> List[Dict]:
        """Enrich clusters with metadata and concept information."""
        # Build file metadata lookup
        file_meta = {f["file_id"]: f for f in metadata.get("files", [])}

        # Build file-concept lookup (support both old and new format)
        file_concepts = defaultdict(set)
        for concept_entry in concepts.get("concepts", []):
            concept_name = concept_entry.get("label", concept_entry.get("name", ""))
            file_ids = concept_entry.get("file_ids", concept_entry.get("files", []))
            for file_id in file_ids:
                file_concepts[file_id].add(concept_name)

        enriched = []
        for cluster in clusters:
            file_ids = cluster.get("file_ids", [])

            # Collect concepts across cluster
            cluster_concepts = defaultdict(int)
            paths = []
            for fid in file_ids:
                for concept in file_concepts.get(fid, []):
                    cluster_concepts[concept] += 1
                if fid in file_meta:
                    paths.append(file_meta[fid].get("path", ""))

            # Find centroid concepts (most common)
            sorted_concepts = sorted(cluster_concepts.items(), key=lambda x: -x[1])
            centroid = [c[0] for c in sorted_concepts[:5]]

            # Generate label from centroid or common path prefix
            label = self._generate_label(centroid, paths)

            enriched.append({
                **cluster,
                "label": label,
                "centroid_concepts": centroid,
                "paths": paths[:10]  # Limit for readability
            })

        return enriched

    def _generate_label(self, concepts: List[str], paths: List[str]) -> str:
        """Generate a human-readable label for a cluster."""
        # Try to find common path prefix
        if paths:
            common = os.path.commonpath(paths) if len(paths) > 1 else paths[0]
            parts = common.split("/")
            for part in reversed(parts):
                if part and part not in [".", "..", "src", "docs"]:
                    return part[:50]

        # Fall back to top concept
        if concepts:
            return concepts[0][:50]

        return "Unknown"

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        method = data.get("method", "unknown")
        optimal_res = data.get("optimal_resolution")
        n_clusters = len(data.get("optimal_clusters", []))
        n_files = data.get("file_count", 0)

        return (
            f"Leiden clustering ({method}): {n_files} files → {n_clusters} communities "
            f"at resolution {optimal_res}. "
            f"Resolutions analyzed: {data.get('resolutions_analyzed', [])}."
        )


# Import for path operations
import os
