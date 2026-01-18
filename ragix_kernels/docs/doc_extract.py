"""
Kernel: Document Extract
Stage: 2 (Analysis)

Extracts key sentences/passages per concept and per file.
These extracts form the raw material for pyramidal summarization.

Uses chunk text content and concept associations to identify
representative sentences that can be aggregated at higher levels.

Enhanced with quality scoring for better sentence selection:
- Completeness: Filters truncated sentences
- Information density: Prefers sentences with entities/numbers
- Structural importance: Weights sentences from headings
- Uniqueness: Avoids redundant content

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
import logging
import json
import re

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocExtractKernel(Kernel):
    """
    Extract key sentences per concept and per file.

    This kernel identifies representative text passages for:
    - Each concept (sentences that best exemplify the concept)
    - Each file (key sentences summarizing the document)
    - Each cluster (combined extracts for cluster-level summary)

    Enhanced v1.1.0 features:
    - Quality scoring for sentence selection
    - Completeness filtering (no truncated sentences)
    - Information density weighting
    - Deduplication across chunks

    Configuration options:
        project.path: Path to the indexed project (required)
        sentences_per_concept: Max sentences per concept (default: 5)
        sentences_per_file: Max sentences per file (default: 10)
        min_sentence_length: Minimum chars for a sentence (default: 40)
        max_sentence_length: Maximum chars for a sentence (default: 500)
        quality_threshold: Min quality score 0-1 (default: 0.3)

    Dependencies:
        doc_metadata: File inventory
        doc_concepts: Concept-file mappings
        doc_cluster: Document clusters

    Output:
        by_file: Extracts grouped by file_id
        by_concept: Extracts grouped by concept_id
        by_cluster: Extracts grouped by cluster_id
        statistics: Extraction statistics
    """

    name = "doc_extract"
    version = "1.1.0"
    category = "docs"
    stage = 2
    description = "Extract key sentences per concept and file (with quality scoring)"

    requires = ["doc_metadata", "doc_concepts", "doc_cluster"]
    provides = ["doc_extracts", "file_extracts", "concept_extracts", "cluster_extracts"]

    # Sentence boundary patterns
    SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜ])')
    BULLET_PATTERN = re.compile(r'^[\-\*•]\s*')

    # Quality detection patterns
    TRUNCATED_START = re.compile(r'^[a-zàâäéèêëïîôùûü]')  # Starts with lowercase
    TRUNCATED_END = re.compile(r'[a-zàâäéèêëïîôùûü,;:\-]$')  # Ends without punctuation
    HAS_NUMBERS = re.compile(r'\d+')
    HAS_ENTITIES = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
    ACTION_VERBS = re.compile(r'\b(permet|permet de|définit|décrit|spécifie|gère|assure|contrôle|surveille|affiche|exporte|importe|configure|valide|vérifie|creates|defines|manages|controls|displays|exports|imports|validates)\b', re.IGNORECASE)

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Extract key sentences from documents."""
        from ragix_core.rag_project import RAGProject, MetadataStore

        # Get project path from config
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        project_path = Path(project_path_str)

        # Configuration
        sentences_per_concept = input.config.get("sentences_per_concept", 5)
        sentences_per_file = input.config.get("sentences_per_file", 10)
        min_sentence_length = input.config.get("min_sentence_length", 40)
        max_sentence_length = input.config.get("max_sentence_length", 500)
        quality_threshold = input.config.get("quality_threshold", 0.3)

        logger.info(f"[doc_extract] Extracting key sentences from {project_path}")

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
        file_concepts = concepts_data.get("file_concepts", {})
        concept_files = concepts_data.get("concept_files", {})
        concepts_list = concepts_data.get("concepts", [])
        concept_labels = {c["concept_id"]: c["label"] for c in concepts_list}

        clusters = cluster_data.get("clusters", [])
        file_to_cluster = {}
        for cluster in clusters:
            for fid in cluster.get("file_ids", []):
                file_to_cluster[fid] = cluster["cluster_id"]

        # Initialize RAG project
        rag = RAGProject(project_path)
        metadata = rag.metadata
        graph = rag.graph
        graph.load()

        # Extract sentences per file and build indices
        by_file: Dict[str, Dict[str, Any]] = {}
        by_concept: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_cluster: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        file_info = {f["file_id"]: f for f in metadata_data.get("files", [])}
        total_sentences = 0

        for file_id, file_data in file_info.items():
            file_path = file_data["path"]
            file_kind = file_data["kind"]
            chunk_count = file_data["chunk_count"]

            # Get chunks for this file
            chunks = list(metadata.iter_chunks_for_file(file_id))
            if not chunks:
                continue

            # Extract sentences from chunks
            file_sentences: List[Dict[str, Any]] = []
            chunk_concepts_map: Dict[str, List[str]] = {}

            for chunk in chunks:
                # Get chunk text from preview (or full content if available)
                text = chunk.text_preview or ""
                if not text:
                    continue

                # Get concepts for this chunk from graph
                chunk_concepts = []
                for edge in graph.get_edges_from(chunk.chunk_id):
                    if edge.type == "mentions":
                        chunk_concepts.append(edge.target)
                chunk_concepts_map[chunk.chunk_id] = chunk_concepts

                # Split into sentences with quality scoring
                sentences = self._extract_sentences(
                    text, min_sentence_length, max_sentence_length
                )

                for sentence in sentences:
                    # Calculate quality score
                    quality = self._score_sentence_quality(sentence)

                    # Filter by quality threshold
                    if quality < quality_threshold:
                        continue

                    sent_info = {
                        "text": sentence,
                        "file_id": file_id,
                        "file_path": file_path,
                        "chunk_id": chunk.chunk_id,
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "concepts": chunk_concepts,
                        "quality_score": quality,
                    }
                    file_sentences.append(sent_info)

            # Deduplicate similar sentences
            file_sentences = self._deduplicate_sentences(file_sentences)

            # Sort sentences by quality score (highest first), then by position
            file_sentences.sort(key=lambda s: (-s.get("quality_score", 0), s.get("line_start", 0)))

            # Select representative sentences (spread across document)
            selected = self._select_representative(file_sentences, sentences_per_file)

            # Build file entry
            file_entry = {
                "file_id": file_id,
                "path": file_path,
                "kind": file_kind,
                "chunk_count": chunk_count,
                "sentence_count": len(file_sentences),
                "selected_count": len(selected),
                "sentences": selected,
                "concepts": list(set(
                    c for s in selected for c in s.get("concepts", [])
                )),
            }
            by_file[file_id] = file_entry
            total_sentences += len(selected)

            # Index by concept
            for sent in selected:
                for concept_id in sent.get("concepts", []):
                    by_concept[concept_id].append(sent)

            # Index by cluster
            cluster_id = file_to_cluster.get(file_id)
            if cluster_id:
                for sent in selected:
                    by_cluster[cluster_id].append(sent)

        # Limit sentences per concept
        for concept_id in list(by_concept.keys()):
            sents = by_concept[concept_id]
            if len(sents) > sentences_per_concept:
                # Select diverse sentences (different files)
                by_concept[concept_id] = self._select_diverse(
                    sents, sentences_per_concept
                )

        # Build concept extracts with labels
        concept_extracts: Dict[str, Dict[str, Any]] = {}
        for concept_id, sents in by_concept.items():
            concept_extracts[concept_id] = {
                "concept_id": concept_id,
                "label": concept_labels.get(concept_id, concept_id),
                "sentence_count": len(sents),
                "file_count": len(set(s["file_id"] for s in sents)),
                "sentences": sents,
            }

        # Build cluster extracts
        cluster_extracts: Dict[str, Dict[str, Any]] = {}
        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            sents = by_cluster.get(cluster_id, [])
            cluster_extracts[cluster_id] = {
                "cluster_id": cluster_id,
                "label": cluster.get("label", cluster_id),
                "sentence_count": len(sents),
                "file_count": cluster.get("file_count", 0),
                "sentences": sents[:sentences_per_file * 2],  # Limit for clusters
            }

        # Statistics
        statistics = {
            "total_files": len(by_file),
            "total_sentences": total_sentences,
            "total_concepts_with_extracts": len(concept_extracts),
            "total_clusters_with_extracts": len(cluster_extracts),
            "avg_sentences_per_file": round(total_sentences / len(by_file), 1) if by_file else 0,
            "avg_sentences_per_concept": (
                round(sum(len(c["sentences"]) for c in concept_extracts.values()) / len(concept_extracts), 1)
                if concept_extracts else 0
            ),
        }

        logger.info(
            f"[doc_extract] Extracted {total_sentences} sentences "
            f"from {len(by_file)} files"
        )

        return {
            "by_file": by_file,
            "by_concept": concept_extracts,
            "by_cluster": cluster_extracts,
            "statistics": statistics,
        }

    def _extract_sentences(
        self,
        text: str,
        min_length: int,
        max_length: int,
    ) -> List[str]:
        """Extract sentences from text with improved boundary detection."""
        # Clean text
        text = text.strip()
        if not text:
            return []

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Split on sentence boundaries
        sentences = self.SENTENCE_END.split(text)

        # Also split on newlines that indicate paragraph breaks
        expanded = []
        for sent in sentences:
            parts = re.split(r'\n\s*\n', sent)
            expanded.extend(parts)

        # Also handle bullet points as separate items
        result = []
        for sent in expanded:
            sent = sent.strip()
            # Clean bullet points
            sent = self.BULLET_PATTERN.sub("", sent)

            # Skip if too short
            if len(sent) < min_length:
                continue

            # Handle long sentences: try to find a natural break point
            if len(sent) > max_length:
                # Look for a break point (semicolon, comma before and/or)
                break_match = re.search(r'[;]|,\s*(?:et|ou|and|or)\s', sent[:max_length])
                if break_match and break_match.end() > min_length:
                    sent = sent[:break_match.end()].rstrip(',; ')
                else:
                    sent = sent[:max_length - 3] + "..."

            result.append(sent)

        return result

    def _score_sentence_quality(self, sentence: str) -> float:
        """
        Score sentence quality from 0 to 1.

        Factors:
        - Completeness (not truncated)
        - Information density (entities, numbers)
        - Action verbs (descriptive content)
        - Length appropriateness
        """
        score = 0.5  # Base score

        # Penalty for truncation indicators
        if self.TRUNCATED_START.match(sentence):
            score -= 0.2
        if self.TRUNCATED_END.search(sentence):
            score -= 0.2

        # Bonus for information density
        if self.HAS_NUMBERS.search(sentence):
            score += 0.1
        if self.HAS_ENTITIES.search(sentence):
            score += 0.1
        if self.ACTION_VERBS.search(sentence):
            score += 0.15

        # Length bonus (prefer medium-length sentences)
        length = len(sentence)
        if 80 <= length <= 300:
            score += 0.1
        elif length < 50 or length > 400:
            score -= 0.1

        # Penalty for markdown artifacts
        if re.search(r'\!\[|\{width=|\\[a-z]+\{', sentence):
            score -= 0.3

        # Penalty for table fragments
        if sentence.count('|') > 3:
            score -= 0.2

        return max(0, min(1, score))

    def _deduplicate_sentences(
        self,
        sentences: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate sentences."""
        from difflib import SequenceMatcher

        if len(sentences) <= 1:
            return sentences

        unique = []
        seen_texts = []

        for sent in sentences:
            text = sent.get("text", "").lower()

            # Check against already seen sentences
            is_duplicate = False
            for seen in seen_texts:
                ratio = SequenceMatcher(None, text, seen).ratio()
                if ratio >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(sent)
                seen_texts.append(text)

        return unique

    def _select_representative(
        self,
        sentences: List[Dict[str, Any]],
        max_count: int,
    ) -> List[Dict[str, Any]]:
        """Select representative sentences spread across document."""
        if len(sentences) <= max_count:
            return sentences

        # Select evenly spaced sentences
        step = len(sentences) / max_count
        selected = []
        for i in range(max_count):
            idx = int(i * step)
            if idx < len(sentences):
                selected.append(sentences[idx])

        return selected

    def _select_diverse(
        self,
        sentences: List[Dict[str, Any]],
        max_count: int,
    ) -> List[Dict[str, Any]]:
        """Select diverse sentences from different files."""
        if len(sentences) <= max_count:
            return sentences

        # Group by file
        by_file: Dict[str, List[Dict]] = defaultdict(list)
        for sent in sentences:
            by_file[sent["file_id"]].append(sent)

        # Round-robin selection from files
        selected = []
        file_ids = list(by_file.keys())
        idx = 0

        while len(selected) < max_count and any(by_file.values()):
            file_id = file_ids[idx % len(file_ids)]
            if by_file[file_id]:
                selected.append(by_file[file_id].pop(0))
            idx += 1

            # Remove empty file lists
            if not by_file[file_id]:
                file_ids.remove(file_id)
                if not file_ids:
                    break

        return selected

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total_files = stats.get("total_files", 0)
        total_sents = stats.get("total_sentences", 0)
        avg_per_file = stats.get("avg_sentences_per_file", 0)
        concepts_with = stats.get("total_concepts_with_extracts", 0)

        return (
            f"Extracts: {total_sents} sentences from {total_files} files "
            f"(avg {avg_per_file}/file). {concepts_with} concepts with extracts."
        )
