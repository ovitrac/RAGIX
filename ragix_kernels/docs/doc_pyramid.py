"""
Kernel: Document Pyramid
Stage: 3 (Synthesis)

Builds a hierarchical/pyramidal summary structure from all previous
stages. This is the capstone kernel that synthesizes:

Level 4: CORPUS SUMMARY (1 summary for entire corpus)
    └── Level 3: DOMAIN SUMMARIES (N summaries by topic cluster)
        └── Level 2: GROUP SUMMARIES (per cluster)
            └── Level 1: DOCUMENT SUMMARIES (per file)

The output is structured JSON + optional Markdown, ready for:
- Direct human consumption
- LLM-based prose generation (post-kernel)
- Visualization (D3.js hierarchical views)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from datetime import datetime
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocPyramidKernel(Kernel):
    """
    Build hierarchical document summary structure.

    This kernel aggregates data from all previous stages to build
    a 4-level pyramid:

    1. Level 1 (Documents): Per-file summaries with key sentences
    2. Level 2 (Groups): Per-cluster summaries
    3. Level 3 (Domains): Topic-based domain summaries
    4. Level 4 (Corpus): Overall corpus summary

    Configuration options:
        levels: Number of levels to generate (1-4, default: 4)
        include_markdown: Generate Markdown output (default: true)
        language: Output language ("en" or "fr", default: "en")
        max_sentences_per_level: Limit sentences per level (default: 20)
        include_structure: Include document structure in summaries (default: true)

    Dependencies:
        doc_metadata: File inventory
        doc_concepts: Concept mappings
        doc_structure: Document outlines
        doc_cluster: Document clusters
        doc_extract: Key sentences
        doc_coverage: Coverage analysis

    Output:
        pyramid: Hierarchical structure with all levels
        markdown: Formatted Markdown representation
        statistics: Generation statistics
    """

    name = "doc_pyramid"
    version = "1.0.0"
    category = "docs"
    stage = 3
    description = "Build hierarchical document summary pyramid"

    requires = [
        "doc_metadata", "doc_concepts", "doc_structure",
        "doc_cluster", "doc_extract", "doc_coverage"
    ]
    provides = ["hierarchical_summary", "pyramid_markdown", "corpus_overview"]

    # i18n strings
    I18N = {
        "en": {
            "corpus_title": "Corpus Summary",
            "domain_title": "Domain",
            "group_title": "Document Group",
            "document_title": "Document",
            "files": "files",
            "chunks": "chunks",
            "concepts": "concepts",
            "key_concepts": "Key Concepts",
            "coverage_gaps": "Coverage Gaps",
            "representative_sentences": "Representative Sentences",
            "documents": "Documents",
            "structure": "Structure",
        },
        "fr": {
            "corpus_title": "Résumé du Corpus",
            "domain_title": "Domaine",
            "group_title": "Groupe de Documents",
            "document_title": "Document",
            "files": "fichiers",
            "chunks": "segments",
            "concepts": "concepts",
            "key_concepts": "Concepts Clés",
            "coverage_gaps": "Lacunes de Couverture",
            "representative_sentences": "Phrases Représentatives",
            "documents": "Documents",
            "structure": "Structure",
        },
    }

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Build hierarchical pyramid from all dependencies."""
        start_time = datetime.now()

        # Get configuration
        levels = input.config.get("levels", 4)
        include_markdown = input.config.get("include_markdown", True)
        language = input.config.get("language", "en")
        max_sentences = input.config.get("max_sentences_per_level", 20)
        include_structure = input.config.get("include_structure", True)

        i18n = self.I18N.get(language, self.I18N["en"])

        logger.info(f"[doc_pyramid] Building {levels}-level pyramid")

        # Load all dependencies
        deps = self._load_all_dependencies(input)

        # Build pyramid bottom-up
        pyramid: Dict[str, Any] = {
            "level_1_documents": [],
            "level_2_groups": [],
            "level_3_domains": [],
            "level_4_corpus": None,
        }

        # Level 1: Document summaries
        if levels >= 1:
            pyramid["level_1_documents"] = self._build_level_1(
                deps, max_sentences, include_structure
            )
            logger.info(f"[doc_pyramid] Level 1: {len(pyramid['level_1_documents'])} documents")

        # Level 2: Group/Cluster summaries
        if levels >= 2:
            pyramid["level_2_groups"] = self._build_level_2(
                deps, pyramid["level_1_documents"], max_sentences
            )
            logger.info(f"[doc_pyramid] Level 2: {len(pyramid['level_2_groups'])} groups")

        # Level 3: Domain summaries (from cluster concepts)
        if levels >= 3:
            pyramid["level_3_domains"] = self._build_level_3(
                deps, pyramid["level_2_groups"], max_sentences
            )
            logger.info(f"[doc_pyramid] Level 3: {len(pyramid['level_3_domains'])} domains")

        # Level 4: Corpus summary
        if levels >= 4:
            pyramid["level_4_corpus"] = self._build_level_4(
                deps, pyramid, i18n
            )
            logger.info("[doc_pyramid] Level 4: Corpus summary generated")

        # Generate Markdown if requested
        markdown = ""
        if include_markdown:
            markdown = self._generate_markdown(pyramid, i18n, levels)

        # Statistics
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        statistics = {
            "levels_generated": levels,
            "level_1_count": len(pyramid["level_1_documents"]),
            "level_2_count": len(pyramid["level_2_groups"]),
            "level_3_count": len(pyramid["level_3_domains"]),
            "level_4_generated": pyramid["level_4_corpus"] is not None,
            "processing_time_ms": processing_time,
            "language": language,
            "include_markdown": include_markdown,
        }

        return {
            "pyramid": pyramid,
            "markdown": markdown,
            "statistics": statistics,
        }

    def _load_all_dependencies(self, input: KernelInput) -> Dict[str, Any]:
        """Load all dependency data."""
        deps = {}
        for name in ["doc_metadata", "doc_concepts", "doc_structure",
                     "doc_cluster", "doc_extract", "doc_coverage"]:
            path = input.dependencies.get(name)
            if path and path.exists():
                with open(path) as f:
                    deps[name] = json.load(f).get("data", {})
            else:
                deps[name] = {}
        return deps

    def _build_level_1(
        self,
        deps: Dict[str, Any],
        max_sentences: int,
        include_structure: bool,
    ) -> List[Dict[str, Any]]:
        """Build Level 1: Document summaries."""
        documents = []

        files = deps["doc_metadata"].get("files", [])
        extracts = deps["doc_extract"].get("by_file", {})
        structures = deps["doc_structure"].get("documents", {})
        file_concepts = deps["doc_concepts"].get("file_concepts", {})
        concepts_list = deps["doc_concepts"].get("concepts", [])
        concept_labels = {c["concept_id"]: c["label"] for c in concepts_list}

        for file_info in files:
            file_id = file_info["file_id"]
            file_path = file_info["path"]

            # Get extracts for this file
            file_extract = extracts.get(file_id, {})
            sentences = file_extract.get("sentences", [])[:max_sentences]

            # Get structure
            structure = None
            if include_structure:
                doc_struct = structures.get(file_id, {})
                sections = doc_struct.get("sections", [])
                if sections:
                    structure = {
                        "section_count": len(sections),
                        "sections": [s.get("title") for s in sections[:5]],
                    }

            # Get concepts with labels
            concept_ids = file_concepts.get(file_id, [])
            concepts_with_labels = [
                {"id": cid, "label": concept_labels.get(cid, cid)}
                for cid in concept_ids[:10]
            ]

            documents.append({
                "file_id": file_id,
                "path": file_path,
                "kind": file_info.get("kind"),
                "chunk_count": file_info.get("chunk_count", 0),
                "concepts": concepts_with_labels,
                "concept_count": len(concept_ids),
                "key_sentences": [s.get("text") for s in sentences],
                "sentence_count": len(sentences),
                "structure": structure,
            })

        return documents

    def _build_level_2(
        self,
        deps: Dict[str, Any],
        level_1_docs: List[Dict],
        max_sentences: int,
    ) -> List[Dict[str, Any]]:
        """Build Level 2: Cluster/Group summaries."""
        groups = []

        clusters = deps["doc_cluster"].get("clusters", [])
        cluster_extracts = deps["doc_extract"].get("by_cluster", {})
        cluster_coverage = deps["doc_coverage"].get("cluster_coverage", {})

        # Build file lookup
        file_lookup = {d["file_id"]: d for d in level_1_docs}

        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            file_ids = cluster.get("file_ids", [])

            # Aggregate from level 1
            all_concepts = []
            all_sentences = []
            for fid in file_ids:
                doc = file_lookup.get(fid, {})
                all_concepts.extend(doc.get("concepts", []))
                all_sentences.extend(doc.get("key_sentences", []))

            # Deduplicate concepts by label
            seen_labels = set()
            unique_concepts = []
            for c in all_concepts:
                label = c.get("label", "").lower()
                if label not in seen_labels:
                    seen_labels.add(label)
                    unique_concepts.append(c)

            # Get cluster extracts
            cluster_ext = cluster_extracts.get(cluster_id, {})
            representative_sentences = [
                s.get("text") for s in cluster_ext.get("sentences", [])
            ][:max_sentences]

            # Get coverage info
            cov = cluster_coverage.get(cluster_id, {})

            groups.append({
                "cluster_id": cluster_id,
                "label": cluster.get("label", cluster_id),
                "file_count": len(file_ids),
                "file_ids": file_ids,
                "centroid_concepts": cluster.get("centroid_labels", [])[:5],
                "all_concepts": unique_concepts[:15],
                "concept_count": len(unique_concepts),
                "representative_sentences": representative_sentences,
                "concept_density": cov.get("concept_density", 0),
                "dominant_concepts": [
                    d.get("label") for d in cov.get("dominant_concepts", [])[:5]
                ],
            })

        return groups

    def _build_level_3(
        self,
        deps: Dict[str, Any],
        level_2_groups: List[Dict],
        max_sentences: int,
    ) -> List[Dict[str, Any]]:
        """Build Level 3: Domain summaries (from dominant concepts)."""
        # Group clusters by their dominant concept
        concept_to_clusters: Dict[str, List[Dict]] = defaultdict(list)

        for group in level_2_groups:
            dominant = group.get("dominant_concepts", [])
            if dominant:
                # Use first dominant concept as domain
                primary_concept = dominant[0]
                concept_to_clusters[primary_concept].append(group)

        # Handle clusters without dominant concepts
        orphan_clusters = [
            g for g in level_2_groups
            if not g.get("dominant_concepts")
        ]
        if orphan_clusters:
            concept_to_clusters["Other"].extend(orphan_clusters)

        # Build domain summaries
        domains = []
        for i, (concept_label, cluster_groups) in enumerate(concept_to_clusters.items()):
            # Aggregate from clusters
            all_files = []
            all_sentences = []
            all_concepts = set()

            for group in cluster_groups:
                all_files.extend(group.get("file_ids", []))
                all_sentences.extend(group.get("representative_sentences", []))
                for c in group.get("all_concepts", []):
                    all_concepts.add(c.get("label", ""))

            domains.append({
                "domain_id": f"D{i+1:02d}",
                "label": concept_label,
                "cluster_ids": [g["cluster_id"] for g in cluster_groups],
                "cluster_count": len(cluster_groups),
                "file_count": len(set(all_files)),
                "file_ids": list(set(all_files)),
                "related_concepts": list(all_concepts)[:10],
                "concept_count": len(all_concepts),
                "representative_sentences": all_sentences[:max_sentences],
            })

        # Sort by file count (most important domains first)
        domains.sort(key=lambda d: -d["file_count"])

        return domains

    def _build_level_4(
        self,
        deps: Dict[str, Any],
        pyramid: Dict[str, Any],
        i18n: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build Level 4: Corpus summary."""
        metadata_stats = deps["doc_metadata"].get("statistics", {})
        concept_stats = deps["doc_concepts"].get("statistics", {})
        coverage_data = deps["doc_coverage"]
        gaps = coverage_data.get("gaps", [])
        overlaps = coverage_data.get("overlaps", [])

        # Get top concepts across corpus
        concepts_list = deps["doc_concepts"].get("concepts", [])
        top_concepts = [c["label"] for c in concepts_list[:10]]

        # Get coverage gaps
        coverage_gaps = [g.get("label") for g in gaps[:5]]

        # Build corpus summary
        corpus = {
            "title": i18n["corpus_title"],
            "file_count": metadata_stats.get("total_files", 0),
            "chunk_count": metadata_stats.get("total_chunks", 0),
            "total_size_mb": metadata_stats.get("total_size_mb", 0),
            "domain_count": len(pyramid.get("level_3_domains", [])),
            "cluster_count": len(pyramid.get("level_2_groups", [])),
            "concept_count": concept_stats.get("total_concepts", 0),
            "key_concepts": top_concepts,
            "coverage_gaps": coverage_gaps,
            "overlapping_concepts": [o.get("label") for o in overlaps[:5]],
            "file_types": metadata_stats.get("kind_counts", {}),
            "by_extension": metadata_stats.get("extension_counts", {}),
        }

        return corpus

    def _generate_markdown(
        self,
        pyramid: Dict[str, Any],
        i18n: Dict[str, str],
        levels: int,
    ) -> str:
        """Generate Markdown representation of pyramid."""
        lines = []

        # Level 4: Corpus
        if levels >= 4 and pyramid.get("level_4_corpus"):
            corpus = pyramid["level_4_corpus"]
            lines.append(f"# {corpus.get('title', i18n['corpus_title'])}\n")
            lines.append(f"**{corpus['file_count']}** {i18n['files']} | "
                        f"**{corpus['chunk_count']}** {i18n['chunks']} | "
                        f"**{corpus['concept_count']}** {i18n['concepts']}\n")

            if corpus.get("key_concepts"):
                lines.append(f"\n## {i18n['key_concepts']}\n")
                for c in corpus["key_concepts"]:
                    lines.append(f"- {c}")
                lines.append("")

            if corpus.get("coverage_gaps"):
                lines.append(f"\n## {i18n['coverage_gaps']}\n")
                for g in corpus["coverage_gaps"]:
                    lines.append(f"- {g}")
                lines.append("")

        # Level 3: Domains
        if levels >= 3 and pyramid.get("level_3_domains"):
            lines.append(f"\n---\n\n# {i18n['domain_title']}s\n")
            for domain in pyramid["level_3_domains"]:
                lines.append(f"\n## {domain['label']}\n")
                lines.append(f"**{domain['file_count']}** {i18n['files']} | "
                            f"**{domain['cluster_count']}** clusters | "
                            f"**{domain['concept_count']}** {i18n['concepts']}\n")

                if domain.get("related_concepts"):
                    concepts_str = ", ".join(domain["related_concepts"][:5])
                    lines.append(f"\n**Related:** {concepts_str}\n")

                if domain.get("representative_sentences"):
                    lines.append(f"\n### {i18n['representative_sentences']}\n")
                    for sent in domain["representative_sentences"][:3]:
                        lines.append(f"> {sent}\n")

        # Level 2: Groups
        if levels >= 2 and pyramid.get("level_2_groups"):
            lines.append(f"\n---\n\n# {i18n['group_title']}s\n")
            for group in pyramid["level_2_groups"]:
                lines.append(f"\n## {group['label']}\n")
                lines.append(f"**{group['file_count']}** {i18n['files']}\n")

                if group.get("centroid_concepts"):
                    concepts_str = ", ".join(group["centroid_concepts"])
                    lines.append(f"\n**Focus:** {concepts_str}\n")

        # Level 1: Documents (abbreviated)
        if levels >= 1 and pyramid.get("level_1_documents"):
            lines.append(f"\n---\n\n# {i18n['documents']}\n")
            lines.append(f"*{len(pyramid['level_1_documents'])} documents indexed*\n")

            # Show first 10 documents
            for doc in pyramid["level_1_documents"][:10]:
                path = Path(doc["path"]).name
                concepts = ", ".join(
                    c.get("label", "") for c in doc.get("concepts", [])[:3]
                )
                lines.append(f"\n### {path}\n")
                lines.append(f"**Concepts:** {concepts}\n")

                if doc.get("key_sentences"):
                    lines.append(f"> {doc['key_sentences'][0]}\n")

            if len(pyramid["level_1_documents"]) > 10:
                remaining = len(pyramid["level_1_documents"]) - 10
                lines.append(f"\n*... and {remaining} more documents*\n")

        return "\n".join(lines)

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        levels = stats.get("levels_generated", 0)
        l1 = stats.get("level_1_count", 0)
        l2 = stats.get("level_2_count", 0)
        l3 = stats.get("level_3_count", 0)
        time_ms = stats.get("processing_time_ms", 0)

        pyramid = data.get("pyramid", {})
        corpus = pyramid.get("level_4_corpus", {})
        top_concepts = corpus.get("key_concepts", [])[:3]
        concepts_str = ", ".join(top_concepts) if top_concepts else "none"

        return (
            f"Pyramid ({levels} levels): {l1} docs → {l2} groups → {l3} domains. "
            f"Top concepts: {concepts_str}. Generated in {time_ms}ms."
        )
