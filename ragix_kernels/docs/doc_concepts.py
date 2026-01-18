"""
Kernel: Document Concepts
Stage: 1 (Collection)

Extracts and aggregates concepts from the RAG knowledge graph.
Concepts are the semantic backbone for document clustering and
hierarchical summarization.

Provides:
- Concept inventory from graph
- Concept frequencies across documents
- File-concept relationships
- Concept co-occurrence matrix

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocConceptsKernel(Kernel):
    """
    Extract and aggregate concepts from RAG knowledge graph.

    This kernel traverses the knowledge graph to collect:
    - All concept nodes
    - Concept → file mappings (via chunks)
    - Concept frequencies
    - Co-occurrence between concepts

    Configuration options:
        project.path: Path to the indexed project (required)
        min_concept_frequency: Minimum frequency to include (default: 1)
        max_concepts: Maximum concepts to return (default: 500)
        include_cooccurrence: Build co-occurrence matrix (default: true)

    Dependencies:
        doc_metadata: File inventory

    Output:
        concepts: List of concept details
        concept_files: Mapping concept → file_ids
        file_concepts: Mapping file_id → concept_ids
        cooccurrence: Concept co-occurrence matrix (if enabled)
        statistics: Aggregated statistics
    """

    name = "doc_concepts"
    version = "1.0.0"
    category = "docs"
    stage = 1
    description = "Extract concepts from RAG knowledge graph"

    requires = ["doc_metadata"]
    provides = ["doc_concepts", "concept_files", "file_concepts", "cooccurrence"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Extract concepts from RAG knowledge graph."""
        from ragix_core.rag_project import (
            RAGProject, KnowledgeGraph, NodeType, EdgeType
        )

        # Get project path from config
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        project_path = Path(project_path_str)

        # Configuration
        min_freq = input.config.get("min_concept_frequency", 1)
        max_concepts = input.config.get("max_concepts", 500)
        include_cooccurrence = input.config.get("include_cooccurrence", True)

        logger.info(f"[doc_concepts] Loading knowledge graph from {project_path}")

        # Load doc_metadata dependency to get file list
        metadata_path = input.dependencies.get("doc_metadata")
        doc_files: Set[str] = set()
        if metadata_path and metadata_path.exists():
            with open(metadata_path) as f:
                metadata_data = json.load(f).get("data", {})
            doc_files = {f["file_id"] for f in metadata_data.get("files", [])}
        logger.info(f"[doc_concepts] Filtering to {len(doc_files)} document files")

        # Initialize knowledge graph
        rag = RAGProject(project_path)
        graph = rag.graph
        graph.load()

        # Get all concept nodes
        concept_nodes = graph.get_nodes_by_type(NodeType.CONCEPT)
        logger.info(f"[doc_concepts] Found {len(concept_nodes)} concept nodes")

        # Build concept → chunks → files mapping
        concept_files: Dict[str, List[str]] = defaultdict(list)
        concept_chunks: Dict[str, List[str]] = defaultdict(list)
        file_concepts: Dict[str, List[str]] = defaultdict(list)

        # Get concept frequencies via MENTIONS edges
        concept_mention_scores: Dict[str, float] = defaultdict(float)

        for concept in concept_nodes:
            concept_id = concept.id
            concept_label = concept.label

            # Find all chunks that mention this concept
            for edge in graph.get_edges_to(concept_id):
                if edge.type == EdgeType.MENTIONS.value:
                    chunk_id = edge.source
                    score = edge.data.get("score", 1.0)
                    concept_mention_scores[concept_id] += score
                    concept_chunks[concept_id].append(chunk_id)

                    # Get file from chunk
                    chunk_node = graph.get_node(chunk_id)
                    if chunk_node:
                        file_id = chunk_node.data.get("file_id")
                        if file_id:
                            # Filter to document files only
                            if not doc_files or file_id in doc_files:
                                if file_id not in concept_files[concept_id]:
                                    concept_files[concept_id].append(file_id)
                                if concept_id not in file_concepts[file_id]:
                                    file_concepts[file_id].append(concept_id)

        # Build concept details list
        concepts: List[Dict[str, Any]] = []
        for concept in concept_nodes:
            concept_id = concept.id
            files = concept_files.get(concept_id, [])
            file_count = len(files)

            # Apply minimum frequency filter
            if file_count < min_freq:
                continue

            concepts.append({
                "concept_id": concept_id,
                "label": concept.label,
                "origin": concept.data.get("origin", "extracted"),
                "description": concept.data.get("description", ""),
                "file_count": file_count,
                "chunk_count": len(concept_chunks.get(concept_id, [])),
                "total_score": round(concept_mention_scores.get(concept_id, 0), 2),
                "file_ids": files,
            })

        # Sort by file_count (most referenced first) and limit
        concepts.sort(key=lambda c: -c["file_count"])
        if len(concepts) > max_concepts:
            concepts = concepts[:max_concepts]
            logger.info(f"[doc_concepts] Limited to top {max_concepts} concepts")

        # Build co-occurrence matrix if requested
        cooccurrence: Dict[str, Dict[str, int]] = {}
        if include_cooccurrence and concepts:
            concept_ids = {c["concept_id"] for c in concepts}
            logger.info("[doc_concepts] Building co-occurrence matrix")

            # For each file, count which concepts co-occur
            for file_id, file_concept_list in file_concepts.items():
                # Filter to included concepts
                relevant = [c for c in file_concept_list if c in concept_ids]
                # Count pairs
                for i, c1 in enumerate(relevant):
                    for c2 in relevant[i + 1:]:
                        # Ensure consistent key ordering
                        if c1 > c2:
                            c1, c2 = c2, c1
                        if c1 not in cooccurrence:
                            cooccurrence[c1] = {}
                        cooccurrence[c1][c2] = cooccurrence[c1].get(c2, 0) + 1

        # Statistics
        statistics = {
            "total_concepts": len(concepts),
            "total_concept_nodes": len(concept_nodes),
            "filtered_out": len(concept_nodes) - len(concepts),
            "files_with_concepts": len(file_concepts),
            "avg_concepts_per_file": (
                round(sum(len(v) for v in file_concepts.values()) / len(file_concepts), 1)
                if file_concepts else 0
            ),
            "avg_files_per_concept": (
                round(sum(c["file_count"] for c in concepts) / len(concepts), 1)
                if concepts else 0
            ),
            "cooccurrence_pairs": sum(len(v) for v in cooccurrence.values()),
        }

        logger.info(
            f"[doc_concepts] Extracted {statistics['total_concepts']} concepts "
            f"across {statistics['files_with_concepts']} files"
        )

        return {
            "concepts": concepts,
            "concept_files": dict(concept_files),
            "file_concepts": dict(file_concepts),
            "cooccurrence": cooccurrence,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total = stats.get("total_concepts", 0)
        files_with = stats.get("files_with_concepts", 0)
        avg_per_file = stats.get("avg_concepts_per_file", 0)

        concepts = data.get("concepts", [])
        top_concepts = [c["label"] for c in concepts[:5]]
        top_str = ", ".join(top_concepts) if top_concepts else "none"

        return (
            f"Concepts: {total} extracted from {files_with} files "
            f"(avg {avg_per_file}/file). Top: {top_str}."
        )
