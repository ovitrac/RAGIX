"""
Kernel: Document Quality Scorecard
Stage: 2 (Analysis)

Implements 5-dimension quality assessment for RAG optimization:
- D1: Linguistic Quality (LQ) - deterministic
- D2: Structural Quality (SQ) - deterministic
- D3: Semantic Coherence (SC) - deterministic
- D4: Intent Clarity (IR) - optional LLM
- D5: Exploitability (EFU) - task-weighted

Plus readiness indices:
- MRI: Minutes Readiness Index
- SRI: Summarization Readiness Index

This kernel follows KOAS principles:
- No LLM inside computation (except optional intent classification)
- Deterministic outputs
- Full auditability

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-01-20
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging
import json
import re
import math

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.docs.config import (
    QualityConfig,
    DocKernelConfig,
    get_doc_kernel_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Intent Classification Patterns
# =============================================================================

INTENT_PATTERNS = {
    "prescriptive": [
        r"\b(must|shall|should|required|mandatory)\b",
        r"\b(follow|implement|ensure|comply|adhere)\b",
        r"\b(doit|devra|devrait|obligatoire|nécessaire)\b",
    ],
    "decisional": [
        r"\b(decision|decided|approve[ds]?|reject[eds]?|agreed?)\b",
        r"\b(vote[ds]?|resolution|consensus|ratifi[edy])\b",
        r"\b(décision|décidé|approuvé|rejeté|accord)\b",
    ],
    "descriptive": [
        r"\b(describes?|explains?|defines?|presents?|represents?)\b",
        r"\b(consists? of|contains?|includes?|comprises?)\b",
        r"\b(décrit|explique|définit|présente|comprend)\b",
    ],
    "exploratory": [
        r"\b(might|could|possibly|potentially|consider)\b",
        r"\b(alternative|option|hypothesis|maybe|perhaps)\b",
        r"\b(pourrait|possible|potentiel|alternative|hypothèse)\b",
    ],
}


@dataclass
class QualityScores:
    """Container for quality dimension scores."""
    lq: float = 0.0  # Linguistic Quality
    sq: float = 0.0  # Structural Quality
    sc: float = 0.0  # Semantic Coherence
    ir: float = 0.0  # Intent Clarity
    ir_entropy: float = 0.0  # Intent entropy (lower is better)
    efu: Dict[str, float] = None  # Task-specific exploitability

    def __post_init__(self):
        if self.efu is None:
            self.efu = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "LQ": round(self.lq, 3),
            "SQ": round(self.sq, 3),
            "SC": round(self.sc, 3),
            "IR": round(self.ir, 3),
            "IR_entropy": round(self.ir_entropy, 3),
            "EFU": {k: round(v, 3) for k, v in self.efu.items()},
        }


@dataclass
class ReadinessIndices:
    """Container for readiness indices."""
    mri: float = 0.0  # Minutes Readiness Index
    sri: float = 0.0  # Summarization Readiness Index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "MRI": round(self.mri, 3),
            "SRI": round(self.sri, 3),
        }


class DocQualityKernel(Kernel):
    """
    Document Quality Scorecard Kernel.

    Implements 5-dimension quality assessment:
    - D1: Linguistic Quality (LQ) - lexical richness, sentence regularity
    - D2: Structural Quality (SQ) - paragraph variance, heading depth
    - D3: Semantic Coherence (SC) - concept reuse, clustering agreement
    - D4: Intent Clarity (IR) - pattern-based or LLM classification
    - D5: Exploitability (EFU) - task-dependent fitness

    Configuration options:
        project.path: Path to the indexed project (required)
        quality_config: QualityConfig dict (optional, loads from ragix.yaml)

    Dependencies:
        doc_metadata: File inventory with chunks
        doc_concepts: Concept-file mappings
        doc_structure: Document structure (headings, sections)
        doc_cluster: Hierarchical clustering results
        doc_cluster_leiden: Leiden clustering results (optional)

    Output:
        quality_scores: Per-document quality scores
        readiness_indices: Per-document MRI and SRI
        quality_tags: Tags based on thresholds
        corpus_summary: Corpus-level quality statistics
        warnings: Consistency check warnings
    """

    name = "doc_quality"
    version = "1.0.0"
    category = "docs"
    stage = 2
    description = "Document quality scorecard with 5-dimension assessment"

    requires = ["doc_metadata", "doc_concepts", "doc_structure"]
    provides = ["quality_scores", "readiness_indices", "quality_tags"]

    # Precompiled patterns
    _sentence_end = re.compile(r'[.!?]')
    _paragraph_break = re.compile(r'\n\s*\n')
    _heading_h1 = re.compile(r'^#\s+', re.MULTILINE)
    _heading_h2 = re.compile(r'^##\s+', re.MULTILINE)
    _heading_h3 = re.compile(r'^###\s+', re.MULTILINE)
    _bullet_list = re.compile(r'^[\s]*[-*•]\s+', re.MULTILINE)
    _numbered_list = re.compile(r'^[\s]*\d+[.)]\s+', re.MULTILINE)
    _has_numbers = re.compile(r'\d+')
    _has_entities = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
    _action_verbs = re.compile(
        r'\b(permet|définit|décrit|gère|assure|contrôle|surveille|affiche|'
        r'creates|defines|manages|controls|displays|exports|validates)\b',
        re.IGNORECASE
    )

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Compute quality scores for all documents."""
        from ragix_core.rag_project import RAGProject

        # Get project path from config
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        project_path = Path(project_path_str)

        # Load quality configuration
        quality_config_dict = input.config.get("quality_config", {})
        if quality_config_dict:
            config = QualityConfig.from_dict(quality_config_dict)
        else:
            config = get_doc_kernel_config().quality

        logger.info(f"[doc_quality] Computing quality scores for {project_path}")
        logger.info(f"[doc_quality] Quality threshold: {config.quality_threshold}")
        logger.info(f"[doc_quality] LLM intent: {'enabled' if config.enable_llm_intent else 'disabled'}")

        # Load dependencies
        metadata_path = input.dependencies.get("doc_metadata")
        concepts_path = input.dependencies.get("doc_concepts")
        structure_path = input.dependencies.get("doc_structure")
        cluster_path = input.dependencies.get("doc_cluster")
        leiden_path = input.dependencies.get("doc_cluster_leiden")

        if not all(p and p.exists() for p in [metadata_path, concepts_path]):
            raise RuntimeError("Missing required dependencies: doc_metadata, doc_concepts")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})
        with open(concepts_path) as f:
            concepts_data = json.load(f).get("data", {})

        structure_data = {}
        if structure_path and structure_path.exists():
            with open(structure_path) as f:
                structure_data = json.load(f).get("data", {})

        cluster_data = {}
        if cluster_path and cluster_path.exists():
            with open(cluster_path) as f:
                cluster_data = json.load(f).get("data", {})

        leiden_data = {}
        if leiden_path and leiden_path.exists():
            with open(leiden_path) as f:
                leiden_data = json.load(f).get("data", {})

        # Initialize RAG project for chunk access
        rag = RAGProject(project_path)
        metadata = rag.metadata

        # Build lookups
        file_concepts = concepts_data.get("file_concepts", {})
        concept_files = concepts_data.get("concept_files", {})
        file_structures = structure_data.get("file_structures", {})
        file_info = {f["file_id"]: f for f in metadata_data.get("files", [])}

        # Process each document
        quality_results: Dict[str, Dict[str, Any]] = {}
        all_warnings = []

        for file_id, file_data in file_info.items():
            file_path = file_data["path"]

            # Get chunks for this file
            chunks = list(metadata.iter_chunks_for_file(file_id))
            chunk_texts = [c.text_preview or "" for c in chunks if c.text_preview]

            if not chunk_texts:
                logger.debug(f"[doc_quality] Skipping {file_path}: no chunks")
                continue

            # Compute quality scores
            scores = QualityScores()

            # D1: Linguistic Quality
            scores.lq = self._score_linguistic_quality(chunk_texts, config)

            # D2: Structural Quality
            structure = file_structures.get(file_id, {})
            scores.sq = self._score_structural_quality(chunk_texts, structure, config)

            # D3: Semantic Coherence
            file_concept_ids = file_concepts.get(file_id, [])
            scores.sc = self._score_semantic_coherence(
                file_id, file_concept_ids, cluster_data, leiden_data, concept_files
            )

            # D4: Intent Clarity
            scores.ir, scores.ir_entropy, intent_dist = self._score_intent_clarity(
                chunk_texts, config
            )

            # D5: Exploitability (task-specific)
            action_density = self._compute_action_density(chunk_texts)
            scores.efu = self._compute_exploitability(scores, action_density, config)

            # Compute readiness indices
            readiness = self._compute_readiness_indices(scores, action_density, config)

            # Generate quality tags
            tags = self._generate_quality_tags(scores, readiness, config)

            # Run consistency checks
            warnings = self._run_consistency_checks(
                scores, readiness, chunk_texts, file_path
            )
            all_warnings.extend(warnings)

            quality_results[file_id] = {
                "file_id": file_id,
                "path": file_path,
                "chunk_count": len(chunk_texts),
                "scores": scores.to_dict(),
                "readiness_indices": readiness.to_dict(),
                "tags": tags,
                "intent_distribution": intent_dist,
                "warnings": warnings,
            }

        # Corpus-level summary
        corpus_summary = self._compute_corpus_summary(quality_results, config)

        logger.info(
            f"[doc_quality] Processed {len(quality_results)} documents. "
            f"Minutes-ready: {corpus_summary['minutes_ready_count']}, "
            f"Summary-ready: {corpus_summary['summary_ready_count']}"
        )

        return {
            "quality_scores": quality_results,
            "corpus_summary": corpus_summary,
            "warnings": all_warnings,
            "config_used": config.to_dict(),
        }

    def _score_linguistic_quality(
        self,
        chunk_texts: List[str],
        config: QualityConfig
    ) -> float:
        """
        Score linguistic quality (D1: LQ).

        Factors:
        - Lexical richness (type-token ratio)
        - Sentence regularity (variance in sentence length)
        - Completeness (no truncation)
        """
        if not chunk_texts:
            return 0.0

        full_text = " ".join(chunk_texts)
        words = re.findall(r'\b\w+\b', full_text.lower())

        if len(words) < 10:
            return 0.3  # Too short to evaluate

        # Lexical richness (TTR normalized)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        # Normalize TTR to 0-1 range (typical TTR is 0.3-0.7)
        lexical_score = min(1.0, ttr / 0.6)

        # Sentence regularity
        sentences = self._sentence_end.split(full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) > 1:
            lengths = [len(s) for s in sentences]
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            std = variance ** 0.5
            # Lower variance is better (normalized)
            cv = std / mean_len if mean_len > 0 else 1
            regularity_score = max(0, 1 - cv * 0.5)  # CV of 2 = score of 0
        else:
            regularity_score = 0.5

        # Completeness check (truncation penalties)
        completeness_score = 1.0
        for chunk in chunk_texts:
            chunk = chunk.strip()
            if not chunk:
                continue
            # Starts with lowercase
            if chunk and chunk[0].islower():
                completeness_score -= config.truncation_penalty / len(chunk_texts)
            # Ends without punctuation
            if chunk and chunk[-1] not in '.!?:;"\'':
                completeness_score -= config.truncation_penalty / len(chunk_texts)

        completeness_score = max(0, completeness_score)

        # Weighted combination
        lq = (0.3 * lexical_score + 0.25 * regularity_score + 0.45 * completeness_score)
        return max(0, min(1, lq))

    def _score_structural_quality(
        self,
        chunk_texts: List[str],
        structure: Dict[str, Any],
        config: QualityConfig
    ) -> float:
        """
        Score structural quality (D2: SQ).

        Factors:
        - Paragraph variance (consistency)
        - Heading depth (hierarchical structure)
        - Prose/bullets ratio
        """
        if not chunk_texts:
            return 0.0

        full_text = "\n\n".join(chunk_texts)

        # Paragraph analysis
        paragraphs = self._paragraph_break.split(full_text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]

        if len(paragraphs) > 1:
            lengths = [len(p) for p in paragraphs]
            mean_len = sum(lengths) / len(lengths)
            # Target: paragraphs between 100-500 chars
            in_range = sum(1 for l in lengths if 100 <= l <= 500)
            para_score = in_range / len(paragraphs)
        else:
            para_score = 0.5

        # Heading depth from structure or regex
        heading_score = 0.0
        if structure:
            headings = structure.get("headings", [])
            h1_count = sum(1 for h in headings if h.get("level") == 1)
            h2_count = sum(1 for h in headings if h.get("level") == 2)
            h3_count = sum(1 for h in headings if h.get("level") == 3)
        else:
            h1_count = len(self._heading_h1.findall(full_text))
            h2_count = len(self._heading_h2.findall(full_text))
            h3_count = len(self._heading_h3.findall(full_text))

        if h1_count > 0:
            heading_score += 0.3
        if h2_count > 0:
            heading_score += 0.3
        if h2_count > 0 and h1_count > 0:
            heading_score += 0.2  # Proper hierarchy bonus
        if h3_count > 0 and h2_count > 0:
            heading_score += 0.2

        heading_score = min(1.0, heading_score)

        # Prose vs bullets ratio
        bullet_matches = (
            len(self._bullet_list.findall(full_text)) +
            len(self._numbered_list.findall(full_text))
        )
        total_lines = len(full_text.split('\n'))
        if total_lines > 0:
            bullet_ratio = bullet_matches / total_lines
            # Optimal: 10-40% bullets (60-90% prose)
            if 0.1 <= bullet_ratio <= 0.4:
                ratio_score = 1.0
            elif bullet_ratio < 0.1:
                ratio_score = 0.7  # All prose is OK
            else:
                ratio_score = max(0, 1 - (bullet_ratio - 0.4) * 2)
        else:
            ratio_score = 0.5

        # Weighted combination
        sq = (0.25 * para_score + 0.35 * heading_score + 0.4 * ratio_score)
        return max(0, min(1, sq))

    def _score_semantic_coherence(
        self,
        file_id: str,
        file_concept_ids: List[str],
        cluster_data: Dict[str, Any],
        leiden_data: Dict[str, Any],
        concept_files: Dict[str, List[str]]
    ) -> float:
        """
        Score semantic coherence (D3: SC).

        Factors:
        - Concept reuse (how often concepts are shared)
        - Clustering agreement (hierarchical vs Leiden)
        """
        if not file_concept_ids:
            return 0.5  # Neutral if no concepts

        # Concept reuse score
        concept_frequencies = []
        for concept_id in file_concept_ids:
            files_with_concept = concept_files.get(concept_id, [])
            concept_frequencies.append(len(files_with_concept))

        if concept_frequencies:
            # Higher frequency = more cohesive corpus
            avg_freq = sum(concept_frequencies) / len(concept_frequencies)
            # Normalize (assuming corpus of ~50 files is typical)
            reuse_score = min(1.0, avg_freq / 10)
        else:
            reuse_score = 0.3

        # Clustering agreement
        agreement_score = 0.5  # Default if only one clustering available

        if cluster_data and leiden_data:
            # Find file's cluster in both methods
            hierarchical_cluster = None
            leiden_cluster = None

            for cluster in cluster_data.get("clusters", []):
                if file_id in cluster.get("file_ids", []):
                    hierarchical_cluster = set(cluster.get("file_ids", []))
                    break

            for cluster in leiden_data.get("optimal_clusters", []):
                if file_id in cluster.get("file_ids", []):
                    leiden_cluster = set(cluster.get("file_ids", []))
                    break

            if hierarchical_cluster and leiden_cluster:
                # Jaccard similarity between clusters
                intersection = len(hierarchical_cluster & leiden_cluster)
                union = len(hierarchical_cluster | leiden_cluster)
                agreement_score = intersection / union if union > 0 else 0

        # Weighted combination
        sc = (0.5 * reuse_score + 0.5 * agreement_score)
        return max(0, min(1, sc))

    def _score_intent_clarity(
        self,
        chunk_texts: List[str],
        config: QualityConfig
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Score intent clarity (D4: IR).

        Uses pattern-based classification by default.
        Optional LLM classification if enabled.

        Returns:
            (ir_score, ir_entropy, intent_distribution)
        """
        full_text = " ".join(chunk_texts)

        # Pattern-based classification
        intent_counts = {intent: 0 for intent in INTENT_PATTERNS}

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, full_text, re.IGNORECASE))
                intent_counts[intent] += matches

        total_matches = sum(intent_counts.values())

        if total_matches == 0:
            # No clear intent markers
            return 0.5, 1.0, {"unknown": 1.0}

        # Compute distribution
        intent_dist = {
            intent: count / total_matches
            for intent, count in intent_counts.items()
        }

        # Compute entropy (lower = clearer intent)
        entropy = 0.0
        for prob in intent_dist.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)

        # Normalize entropy (max entropy for 4 categories = 2.0)
        max_entropy = math.log2(len(INTENT_PATTERNS))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # IR score: higher for clearer (lower entropy) intent
        ir_score = 1.0 - normalized_entropy

        # Bonus for dominant intent
        max_prob = max(intent_dist.values())
        if max_prob >= 0.8:
            ir_score = min(1.0, ir_score + 0.1)

        return ir_score, normalized_entropy, intent_dist

    def _compute_action_density(self, chunk_texts: List[str]) -> float:
        """Compute action verb density for minutes readiness."""
        full_text = " ".join(chunk_texts)
        action_matches = len(self._action_verbs.findall(full_text))
        words = len(full_text.split())

        if words == 0:
            return 0.0

        density = action_matches / words * 100  # Per 100 words
        # Normalize (10 action verbs per 100 words is high)
        return min(1.0, density / 5)

    def _compute_exploitability(
        self,
        scores: QualityScores,
        action_density: float,
        config: QualityConfig
    ) -> Dict[str, float]:
        """
        Compute task-specific exploitability (D5: EFU).

        Different tasks weight quality dimensions differently.
        """
        return {
            "summarization": (
                0.30 * scores.lq +
                0.30 * scores.sq +
                0.20 * scores.sc +
                0.20 * scores.ir
            ),
            "minutes_extraction": (
                0.35 * scores.lq +
                0.25 * scores.sq +
                0.25 * (1 - scores.ir_entropy) +
                0.15 * action_density
            ),
            "qa_retrieval": (
                0.25 * scores.lq +
                0.20 * scores.sq +
                0.35 * scores.sc +
                0.20 * scores.ir
            ),
        }

    def _compute_readiness_indices(
        self,
        scores: QualityScores,
        action_density: float,
        config: QualityConfig
    ) -> ReadinessIndices:
        """
        Compute readiness indices (MRI, SRI).

        MRI = Minutes Readiness Index
        SRI = Summarization Readiness Index
        """
        # MRI formula: 0.35*LQ + 0.25*SQ + 0.25*(1-entropy) + 0.15*action_density
        mri = (
            0.35 * scores.lq +
            0.25 * scores.sq +
            0.25 * (1 - scores.ir_entropy) +
            0.15 * action_density
        )

        # SRI formula: 0.30*LQ + 0.30*SQ + 0.20*SC + 0.20*IR
        sri = (
            0.30 * scores.lq +
            0.30 * scores.sq +
            0.20 * scores.sc +
            0.20 * scores.ir
        )

        return ReadinessIndices(mri=max(0, min(1, mri)), sri=max(0, min(1, sri)))

    def _generate_quality_tags(
        self,
        scores: QualityScores,
        readiness: ReadinessIndices,
        config: QualityConfig
    ) -> List[str]:
        """Generate quality tags based on scores and thresholds."""
        tags = []

        # Structural tags
        if scores.sq >= 0.7:
            tags.append("well_structured")
        elif scores.sq < 0.4:
            tags.append("needs_structure")

        # Content tags
        if scores.lq >= 0.7:
            tags.append("high_linguistic_quality")
        elif scores.lq < 0.4:
            tags.append("low_linguistic_quality")

        # Intent tags
        if scores.ir_entropy < 0.3:
            tags.append("clear_intent")
        elif scores.ir_entropy >= 0.5:
            tags.append("mixed_intent")

        # Readiness tags
        if readiness.mri >= config.mri_auto_threshold:
            tags.append("minutes_ready")
        elif readiness.mri < config.mri_assisted_threshold:
            tags.append("minutes_needs_rewrite")

        if readiness.sri >= config.sri_auto_threshold:
            tags.append("summary_ready")
        elif readiness.sri < config.sri_assisted_threshold:
            tags.append("summary_unsuitable")

        return tags

    def _run_consistency_checks(
        self,
        scores: QualityScores,
        readiness: ReadinessIndices,
        chunk_texts: List[str],
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Run consistency checks and generate warnings."""
        warnings = []

        # LQ vs SQ check
        if scores.lq > 0.7 and scores.sq < 0.4:
            warnings.append({
                "check": "LQ_vs_SQ",
                "file": file_path,
                "message": "High linguistic quality but poor structure",
                "lq": round(scores.lq, 3),
                "sq": round(scores.sq, 3),
            })

        # Minutes ready but long chunks
        avg_chunk_len = sum(len(c) for c in chunk_texts) / len(chunk_texts) if chunk_texts else 0
        if "minutes_ready" in self._generate_quality_tags(scores, readiness, QualityConfig()) and avg_chunk_len > 1000:
            warnings.append({
                "check": "minutes_long_chunks",
                "file": file_path,
                "message": "Minutes-ready document has long chunks",
                "avg_chunk_length": round(avg_chunk_len),
            })

        # Well-structured but high entropy
        if scores.sq >= 0.7 and scores.ir_entropy > 0.5:
            warnings.append({
                "check": "specs_high_entropy",
                "file": file_path,
                "message": "Well-structured document with mixed intent",
                "sq": round(scores.sq, 3),
                "ir_entropy": round(scores.ir_entropy, 3),
            })

        return warnings

    def _compute_corpus_summary(
        self,
        quality_results: Dict[str, Dict[str, Any]],
        config: QualityConfig
    ) -> Dict[str, Any]:
        """Compute corpus-level quality statistics."""
        if not quality_results:
            return {
                "total_documents": 0,
                "minutes_ready_count": 0,
                "summary_ready_count": 0,
                "needs_attention_count": 0,
            }

        # Aggregate scores
        lq_scores = [r["scores"]["LQ"] for r in quality_results.values()]
        sq_scores = [r["scores"]["SQ"] for r in quality_results.values()]
        sc_scores = [r["scores"]["SC"] for r in quality_results.values()]
        ir_scores = [r["scores"]["IR"] for r in quality_results.values()]
        mri_scores = [r["readiness_indices"]["MRI"] for r in quality_results.values()]
        sri_scores = [r["readiness_indices"]["SRI"] for r in quality_results.values()]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        # Count by readiness
        minutes_ready = sum(1 for mri in mri_scores if mri >= config.mri_auto_threshold)
        summary_ready = sum(1 for sri in sri_scores if sri >= config.sri_auto_threshold)
        needs_attention = sum(
            1 for mri, sri in zip(mri_scores, sri_scores)
            if mri < config.mri_assisted_threshold and sri < config.sri_assisted_threshold
        )

        # Quality distribution
        quality_buckets = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        for mri in mri_scores:
            if mri >= 0.75:
                quality_buckets["excellent"] += 1
            elif mri >= 0.6:
                quality_buckets["good"] += 1
            elif mri >= 0.45:
                quality_buckets["acceptable"] += 1
            else:
                quality_buckets["poor"] += 1

        return {
            "total_documents": len(quality_results),
            "avg_scores": {
                "LQ": round(avg(lq_scores), 3),
                "SQ": round(avg(sq_scores), 3),
                "SC": round(avg(sc_scores), 3),
                "IR": round(avg(ir_scores), 3),
                "MRI": round(avg(mri_scores), 3),
                "SRI": round(avg(sri_scores), 3),
            },
            "minutes_ready_count": minutes_ready,
            "summary_ready_count": summary_ready,
            "needs_attention_count": needs_attention,
            "quality_distribution": quality_buckets,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        summary = data.get("corpus_summary", {})
        total = summary.get("total_documents", 0)
        minutes_ready = summary.get("minutes_ready_count", 0)
        summary_ready = summary.get("summary_ready_count", 0)
        avg_mri = summary.get("avg_scores", {}).get("MRI", 0)
        warnings = len(data.get("warnings", []))

        return (
            f"Quality scorecard: {total} docs analyzed. "
            f"Minutes-ready: {minutes_ready}, Summary-ready: {summary_ready}. "
            f"Avg MRI: {avg_mri:.2f}. Warnings: {warnings}."
        )
