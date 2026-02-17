"""
Unified Configuration for Document Kernels
===========================================

Provides dataclasses for configuring document processing kernels:
- ChunkingConfig: Text chunking parameters
- QualityConfig: Quality scoring thresholds and weights
- DocKernelConfig: Main configuration container

This module centralizes previously hardcoded parameters to enable:
- Unified configuration via ragix.yaml
- Runtime modification via API
- Profile-based presets

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-01-20
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Chunking Configuration
# =============================================================================

@dataclass
class ChunkingConfig:
    """
    Configuration for text chunking strategies.

    Attributes:
        strategy: Chunking approach ("semantic", "fixed", "hybrid")
        target_tokens: Target chunk size in tokens (200-300 recommended for quality)
        overlap: Overlap strategy ("paragraph", "percentage", "fixed")
        overlap_tokens: Number of overlap tokens when using "fixed" overlap
        overlap_percentage: Overlap percentage when using "percentage" (0-0.5)
        preserve_bullets: Keep bullet lists intact within chunks
        preserve_tables: Keep tables intact within chunks
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
    """
    strategy: str = "semantic"
    target_tokens: int = 250
    overlap: str = "paragraph"
    overlap_tokens: int = 50
    overlap_percentage: float = 0.2
    preserve_bullets: bool = True
    preserve_tables: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    def __post_init__(self):
        """Validate configuration values."""
        valid_strategies = ("semantic", "fixed", "hybrid")
        if self.strategy not in valid_strategies:
            logger.warning(f"Unknown chunking strategy '{self.strategy}', defaulting to 'semantic'")
            self.strategy = "semantic"

        valid_overlaps = ("paragraph", "percentage", "fixed")
        if self.overlap not in valid_overlaps:
            logger.warning(f"Unknown overlap strategy '{self.overlap}', defaulting to 'paragraph'")
            self.overlap = "paragraph"

        if self.target_tokens < 50 or self.target_tokens > 2000:
            logger.warning(f"target_tokens={self.target_tokens} outside recommended range [50,2000]")

        if self.overlap_percentage < 0 or self.overlap_percentage > 0.5:
            logger.warning(f"overlap_percentage={self.overlap_percentage} should be in [0,0.5]")
            self.overlap_percentage = max(0, min(0.5, self.overlap_percentage))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "target_tokens": self.target_tokens,
            "overlap": self.overlap,
            "overlap_tokens": self.overlap_tokens,
            "overlap_percentage": self.overlap_percentage,
            "preserve_bullets": self.preserve_bullets,
            "preserve_tables": self.preserve_tables,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        """Create from dictionary."""
        return cls(
            strategy=data.get("strategy", "semantic"),
            target_tokens=data.get("target_tokens", 250),
            overlap=data.get("overlap", "paragraph"),
            overlap_tokens=data.get("overlap_tokens", 50),
            overlap_percentage=data.get("overlap_percentage", 0.2),
            preserve_bullets=data.get("preserve_bullets", True),
            preserve_tables=data.get("preserve_tables", True),
            min_chunk_size=data.get("min_chunk_size", 100),
            max_chunk_size=data.get("max_chunk_size", 2000),
        )


# =============================================================================
# Quality Scoring Configuration
# =============================================================================

@dataclass
class QualityConfig:
    """
    Configuration for document quality scoring.

    These parameters were previously hardcoded in doc_extract.py.
    Moving them here enables centralized configuration.

    Attributes:
        quality_threshold: Minimum quality score to accept a sentence (0-1)
        base_score: Starting score for quality computation
        truncation_penalty: Penalty for truncated sentences
        artifact_penalty: Penalty for markdown/table artifacts
        entity_bonus: Bonus for sentences with named entities
        numbers_bonus: Bonus for sentences with numbers/metrics
        action_verb_bonus: Bonus for sentences with action verbs
        length_bonus_min: Minimum sentence length for length bonus
        length_bonus_max: Maximum sentence length for length bonus
        enable_llm_intent: Enable LLM-based intent classification
        llm_intent_model: Model to use for intent classification
        llm_intent_confidence: Minimum confidence for LLM classification
        boilerplate_penalty: Penalty for document control boilerplate
        formatting_penalty: Penalty for formatting-heavy content
        dashed_line_min_length: Minimum consecutive dashes to trigger penalty
        boilerplate_vocab_fr: French boilerplate vocabulary patterns
        boilerplate_vocab_en: English boilerplate vocabulary patterns
    """
    # Thresholds
    quality_threshold: float = 0.4
    base_score: float = 0.5

    # Penalties (subtracted from score)
    truncation_penalty: float = 0.2
    artifact_penalty: float = 0.3
    table_fragment_penalty: float = 0.2

    # Boilerplate penalties (v0.64.1 - DOCSET fix)
    boilerplate_penalty: float = 0.4  # Heavy penalty for document control content
    formatting_penalty: float = 0.25  # Penalty for formatting-heavy lines
    dashed_line_min_length: int = 10  # Minimum dashes to trigger penalty

    # Boilerplate vocabulary (multilingual, configurable)
    boilerplate_vocab_fr: List[str] = field(default_factory=lambda: [
        r"PANNEAU DE CONTR[OÔ]LE",
        r"Historique des r[eé]visions",
        r"Contr[oô]le du document",
        r"Table des mati[eè]res",
        r"R[eé]f[eé]rence\s*:",
        r"Auteur\s*:",
        r"Date\s*:",
        r"Version\s*:",
        r"Statut\s*:",
        r"Approbation\s*:",
        r"Diffusion\s*:",
    ])
    boilerplate_vocab_en: List[str] = field(default_factory=lambda: [
        r"DOCUMENT CONTROL",
        r"Revision History",
        r"Table of Contents",
        r"Reference\s*:",
        r"Author\s*:",
        r"Date\s*:",
        r"Version\s*:",
        r"Status\s*:",
        r"Approval\s*:",
        r"Distribution\s*:",
    ])
    # Release changelog / version tracking patterns (v0.64.2 - DOCSET fix)
    boilerplate_changelog: List[str] = field(default_factory=lambda: [
        r"\d+\.\d+\.\d+\.\d+\s*.*?(?:MERGED|RELEASED|issues)",  # X.X.X.X ... MERGED/RELEASED
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d+\s+issues",  # Month YYYY N issues
        r"\d+\s+issues\s+(?:MERGED|RELEASED|CLOSED)",  # N issues MERGED/RELEASED
        r"(?:NOT\s+)?DELIVERED",  # NOT DELIVERED / DELIVERED
        r"(?:VM\d+\s*)+",  # VM1 VM2 VM3 (infrastructure notation)
        r"Sous-r[eé]seau\s+priv[eé]",  # Network diagram text
        r"Fibre\s+noire",  # Infrastructure terminology
        r"Datacenter\s+\w+",  # Datacenter references
        r"Lien\s+WAN",  # Network link references
    ])

    # Bonuses (added to score)
    entity_bonus: float = 0.1
    numbers_bonus: float = 0.1
    action_verb_bonus: float = 0.15
    length_bonus: float = 0.1

    # Length thresholds for bonus
    length_bonus_min: int = 80
    length_bonus_max: int = 300
    length_penalty_short: int = 50
    length_penalty_long: int = 400

    # LLM intent classification (optional)
    enable_llm_intent: bool = False
    llm_intent_model: str = "granite3.1-moe:3b"
    llm_intent_confidence: float = 0.7
    llm_intent_max_chars: int = 500

    # Readiness index thresholds
    mri_auto_threshold: float = 0.75
    mri_assisted_threshold: float = 0.45
    sri_auto_threshold: float = 0.70
    sri_assisted_threshold: float = 0.50

    def __post_init__(self):
        """Validate configuration values."""
        # Ensure thresholds are in valid range
        self.quality_threshold = max(0.0, min(1.0, self.quality_threshold))
        self.base_score = max(0.0, min(1.0, self.base_score))
        self.llm_intent_confidence = max(0.0, min(1.0, self.llm_intent_confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "quality_threshold": self.quality_threshold,
            "base_score": self.base_score,
            "truncation_penalty": self.truncation_penalty,
            "artifact_penalty": self.artifact_penalty,
            "table_fragment_penalty": self.table_fragment_penalty,
            "boilerplate_penalty": self.boilerplate_penalty,
            "formatting_penalty": self.formatting_penalty,
            "dashed_line_min_length": self.dashed_line_min_length,
            "boilerplate_vocab_fr": self.boilerplate_vocab_fr,
            "boilerplate_vocab_en": self.boilerplate_vocab_en,
            "boilerplate_changelog": self.boilerplate_changelog,
            "entity_bonus": self.entity_bonus,
            "numbers_bonus": self.numbers_bonus,
            "action_verb_bonus": self.action_verb_bonus,
            "length_bonus": self.length_bonus,
            "length_bonus_min": self.length_bonus_min,
            "length_bonus_max": self.length_bonus_max,
            "length_penalty_short": self.length_penalty_short,
            "length_penalty_long": self.length_penalty_long,
            "enable_llm_intent": self.enable_llm_intent,
            "llm_intent_model": self.llm_intent_model,
            "llm_intent_confidence": self.llm_intent_confidence,
            "llm_intent_max_chars": self.llm_intent_max_chars,
            "mri_auto_threshold": self.mri_auto_threshold,
            "mri_assisted_threshold": self.mri_assisted_threshold,
            "sri_auto_threshold": self.sri_auto_threshold,
            "sri_assisted_threshold": self.sri_assisted_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityConfig":
        """Create from dictionary."""
        # Default boilerplate vocabulary (French)
        default_vocab_fr = [
            r"PANNEAU DE CONTR[OÔ]LE",
            r"Historique des r[eé]visions",
            r"Contr[oô]le du document",
            r"Table des mati[eè]res",
            r"R[eé]f[eé]rence\s*:",
            r"Auteur\s*:",
            r"Date\s*:",
            r"Version\s*:",
            r"Statut\s*:",
            r"Approbation\s*:",
            r"Diffusion\s*:",
        ]
        # Default boilerplate vocabulary (English)
        default_vocab_en = [
            r"DOCUMENT CONTROL",
            r"Revision History",
            r"Table of Contents",
            r"Reference\s*:",
            r"Author\s*:",
            r"Date\s*:",
            r"Version\s*:",
            r"Status\s*:",
            r"Approval\s*:",
            r"Distribution\s*:",
        ]
        # Default changelog/release tracking patterns
        default_changelog = [
            r"\d+\.\d+\.\d+\.\d+\s*.*?(?:MERGED|RELEASED|issues)",
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d+\s+issues",
            r"\d+\s+issues\s+(?:MERGED|RELEASED|CLOSED)",
            r"(?:NOT\s+)?DELIVERED",
            r"(?:VM\d+\s*)+",
            r"Sous-r[eé]seau\s+priv[eé]",
            r"Fibre\s+noire",
            r"Datacenter\s+\w+",
            r"Lien\s+WAN",
        ]

        return cls(
            quality_threshold=data.get("quality_threshold", 0.4),
            base_score=data.get("base_score", 0.5),
            truncation_penalty=data.get("truncation_penalty", 0.2),
            artifact_penalty=data.get("artifact_penalty", 0.3),
            table_fragment_penalty=data.get("table_fragment_penalty", 0.2),
            boilerplate_penalty=data.get("boilerplate_penalty", 0.4),
            formatting_penalty=data.get("formatting_penalty", 0.25),
            dashed_line_min_length=data.get("dashed_line_min_length", 10),
            boilerplate_vocab_fr=data.get("boilerplate_vocab_fr", default_vocab_fr),
            boilerplate_vocab_en=data.get("boilerplate_vocab_en", default_vocab_en),
            boilerplate_changelog=data.get("boilerplate_changelog", default_changelog),
            entity_bonus=data.get("entity_bonus", 0.1),
            numbers_bonus=data.get("numbers_bonus", 0.1),
            action_verb_bonus=data.get("action_verb_bonus", 0.15),
            length_bonus=data.get("length_bonus", 0.1),
            length_bonus_min=data.get("length_bonus_min", 80),
            length_bonus_max=data.get("length_bonus_max", 300),
            length_penalty_short=data.get("length_penalty_short", 50),
            length_penalty_long=data.get("length_penalty_long", 400),
            enable_llm_intent=data.get("enable_llm_intent", False),
            llm_intent_model=data.get("llm_intent_model", "granite3.1-moe:3b"),
            llm_intent_confidence=data.get("llm_intent_confidence", 0.7),
            llm_intent_max_chars=data.get("llm_intent_max_chars", 500),
            mri_auto_threshold=data.get("mri_auto_threshold", 0.75),
            mri_assisted_threshold=data.get("mri_assisted_threshold", 0.45),
            sri_auto_threshold=data.get("sri_auto_threshold", 0.70),
            sri_assisted_threshold=data.get("sri_assisted_threshold", 0.50),
        )


# =============================================================================
# Clustering Configuration
# =============================================================================

@dataclass
class ClusteringConfig:
    """
    Configuration for document clustering.

    Unified parameters for both hierarchical and Leiden clustering.
    Previously, doc_cluster.py used 'resolutions' and doc_cluster_leiden.py
    used 'leiden_resolutions' - this unifies them.

    Attributes:
        method: Clustering method ("hierarchical", "leiden", "both")
        min_cluster_size: Minimum documents per cluster
        resolutions: Multi-resolution analysis levels (for Leiden)
        similarity_metric: Similarity metric ("jaccard", "cosine")
        linkage: Linkage method for hierarchical clustering
        seed: Random seed for reproducibility
    """
    method: str = "both"
    min_cluster_size: int = 2
    resolutions: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    similarity_metric: str = "jaccard"
    linkage: str = "average"
    seed: int = 42
    similarity_threshold: float = 0.1

    def __post_init__(self):
        """Validate configuration values."""
        valid_methods = ("hierarchical", "leiden", "both")
        if self.method not in valid_methods:
            logger.warning(f"Unknown clustering method '{self.method}', defaulting to 'both'")
            self.method = "both"

        valid_metrics = ("jaccard", "cosine")
        if self.similarity_metric not in valid_metrics:
            logger.warning(f"Unknown similarity metric '{self.similarity_metric}', defaulting to 'jaccard'")
            self.similarity_metric = "jaccard"

        valid_linkages = ("single", "complete", "average", "ward")
        if self.linkage not in valid_linkages:
            logger.warning(f"Unknown linkage method '{self.linkage}', defaulting to 'average'")
            self.linkage = "average"

        # Ensure resolutions are valid
        self.resolutions = [r for r in self.resolutions if 0 < r <= 2.0]
        if not self.resolutions:
            self.resolutions = [0.1, 0.5, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "min_cluster_size": self.min_cluster_size,
            "resolutions": self.resolutions,
            "similarity_metric": self.similarity_metric,
            "linkage": self.linkage,
            "seed": self.seed,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusteringConfig":
        """Create from dictionary."""
        return cls(
            method=data.get("method", "both"),
            min_cluster_size=data.get("min_cluster_size", 2),
            resolutions=data.get("resolutions", [0.1, 0.5, 1.0]),
            similarity_metric=data.get("similarity_metric", "jaccard"),
            linkage=data.get("linkage", "average"),
            seed=data.get("seed", 42),
            similarity_threshold=data.get("similarity_threshold", 0.1),
        )


# =============================================================================
# Main Document Kernel Configuration
# =============================================================================

@dataclass
class DocKernelConfig:
    """
    Main configuration container for document processing kernels.

    Aggregates all sub-configurations into a single config object
    that can be loaded from ragix.yaml or modified at runtime.

    Usage:
        config = DocKernelConfig.load()
        config.quality.quality_threshold = 0.5
        config.chunking.target_tokens = 300
    """
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    # Extraction parameters (from doc_extract.py)
    sentences_per_concept: int = 5
    sentences_per_file: int = 10
    min_sentence_length: int = 40
    max_sentence_length: int = 500
    similarity_threshold: float = 0.8  # For deduplication

    def __post_init__(self):
        """Validate configuration."""
        if self.sentences_per_concept < 1:
            self.sentences_per_concept = 5
        if self.sentences_per_file < 1:
            self.sentences_per_file = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunking": self.chunking.to_dict(),
            "quality": self.quality.to_dict(),
            "clustering": self.clustering.to_dict(),
            "sentences_per_concept": self.sentences_per_concept,
            "sentences_per_file": self.sentences_per_file,
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocKernelConfig":
        """Create from dictionary."""
        chunking_data = data.get("chunking", {})
        quality_data = data.get("quality", {})
        clustering_data = data.get("clustering", {})

        return cls(
            chunking=ChunkingConfig.from_dict(chunking_data) if chunking_data else ChunkingConfig(),
            quality=QualityConfig.from_dict(quality_data) if quality_data else QualityConfig(),
            clustering=ClusteringConfig.from_dict(clustering_data) if clustering_data else ClusteringConfig(),
            sentences_per_concept=data.get("sentences_per_concept", 5),
            sentences_per_file=data.get("sentences_per_file", 10),
            min_sentence_length=data.get("min_sentence_length", 40),
            max_sentence_length=data.get("max_sentence_length", 500),
            similarity_threshold=data.get("similarity_threshold", 0.8),
        )

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "DocKernelConfig":
        """
        Load configuration from ragix.yaml.

        Looks for a 'doc_kernels' section in the main ragix.yaml config.

        Args:
            config_path: Optional path to ragix.yaml

        Returns:
            DocKernelConfig instance
        """
        try:
            from ragix_core.config import get_config, find_config_file
            import yaml

            # Find config file
            if config_path is None:
                config_path = find_config_file()

            if config_path and config_path.exists():
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f) or {}

                # Look for doc_kernels section
                doc_config = data.get("doc_kernels", {})
                if doc_config:
                    return cls.from_dict(doc_config)

            logger.info("No doc_kernels config found, using defaults")
            return cls()

        except ImportError:
            logger.warning("ragix_core not available, using default config")
            return cls()
        except Exception as e:
            logger.warning(f"Error loading doc kernel config: {e}, using defaults")
            return cls()


# =============================================================================
# Configuration Profiles
# =============================================================================

# Predefined profiles for common use cases
PROFILE_DOCUMENTS_ONLY = DocKernelConfig(
    chunking=ChunkingConfig(
        strategy="semantic",
        target_tokens=300,
        overlap="paragraph",
    ),
    quality=QualityConfig(
        quality_threshold=0.4,
        enable_llm_intent=False,
    ),
    clustering=ClusteringConfig(
        method="both",
        min_cluster_size=2,
    ),
)

PROFILE_MINUTES_EXTRACTION = DocKernelConfig(
    chunking=ChunkingConfig(
        strategy="semantic",
        target_tokens=200,  # Smaller chunks for action items
        overlap="paragraph",
    ),
    quality=QualityConfig(
        quality_threshold=0.3,  # Lower threshold to capture more content
        action_verb_bonus=0.2,  # Higher bonus for action verbs
        enable_llm_intent=True,  # Enable intent classification
    ),
    clustering=ClusteringConfig(
        method="leiden",  # Leiden better for topic discovery
        min_cluster_size=2,
    ),
)

PROFILE_COMPLIANCE_AUDIT = DocKernelConfig(
    chunking=ChunkingConfig(
        strategy="fixed",  # Consistent chunk sizes for audit
        target_tokens=250,
        overlap="fixed",
        overlap_tokens=50,
    ),
    quality=QualityConfig(
        quality_threshold=0.5,  # Higher threshold for quality
        enable_llm_intent=True,
    ),
    clustering=ClusteringConfig(
        method="both",
        min_cluster_size=3,  # Larger clusters for audit
    ),
)

PROFILES = {
    "documents_only": PROFILE_DOCUMENTS_ONLY,
    "minutes_extraction": PROFILE_MINUTES_EXTRACTION,
    "compliance_audit": PROFILE_COMPLIANCE_AUDIT,
}


def get_profile(name: str) -> DocKernelConfig:
    """
    Get a predefined configuration profile.

    Args:
        name: Profile name (documents_only, minutes_extraction, compliance_audit)

    Returns:
        DocKernelConfig for the profile

    Raises:
        ValueError: If profile name is unknown
    """
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


# =============================================================================
# Global Configuration Instance
# =============================================================================

_global_doc_config: Optional[DocKernelConfig] = None


def get_doc_kernel_config() -> DocKernelConfig:
    """Get the global document kernel configuration (lazy-loaded)."""
    global _global_doc_config
    if _global_doc_config is None:
        _global_doc_config = DocKernelConfig.load()
    return _global_doc_config


def reload_doc_kernel_config(config_path: Optional[Path] = None) -> DocKernelConfig:
    """Reload document kernel configuration from file."""
    global _global_doc_config
    _global_doc_config = DocKernelConfig.load(config_path)
    return _global_doc_config


def set_doc_kernel_config(config: DocKernelConfig) -> None:
    """Set the global document kernel configuration."""
    global _global_doc_config
    _global_doc_config = config
