#!/usr/bin/env python3
"""
Global Configuration for RAGIX Reasoning Tutor v0.3
=====================================================

Centralized configuration for all meta-cognitive components:
- R1: FailureDetector
- R2: MetaCards
- R3: JustificationProtocol
- Semantic Tools (Phase 1+)

Configuration can be:
1. Modified directly in this file (defaults)
2. Overridden via environment variables (RAGIX_*)
3. Loaded from YAML/JSON config file

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class FailureDetectorConfig:
    """Configuration for FailureDetector (R1)."""

    # Detection thresholds
    repetition_threshold: int = 3      # Same action N times → REPETITION_LOOP
    circular_threshold: int = 2        # A→B→A→B cycles → CIRCULAR_PATTERN
    error_threshold: int = 3           # Consecutive errors → EXPLICIT_ERROR
    stall_threshold: int = 4           # No PCG growth for N turns → PROGRESS_STALL
    exhaustion_threshold: int = 3      # Cards tried without progress → EXHAUSTION

    # Behavior flags
    enabled: bool = True               # Master switch for failure detection
    log_detections: bool = True        # Log detected failures
    auto_intervention: bool = True     # Automatically trigger cards on failure


@dataclass
class MetaCardsConfig:
    """Configuration for MetaCards (R2) and Strategic Advisor (R4)."""

    # Card behavior
    enabled: bool = True               # Master switch for meta-cards
    max_cards_per_game: int = 5        # Maximum cards that can be issued
    cooldown_turns: int = 2            # Minimum turns between cards

    # Card effectiveness tracking
    track_effectiveness: bool = True   # Track card success rates
    min_success_rate: float = 0.3      # Retire cards below this rate

    # R3: Reactive card types enabled
    enable_escape_loop: bool = True
    enable_compass: bool = True
    enable_error_analysis: bool = True
    enable_progress_boost: bool = True
    enable_strategic_reset: bool = True

    # R4: TRIZ Strategic Cards
    enable_triz_cards: bool = True     # Master switch for TRIZ cards
    enable_segment_task: bool = True   # TRIZ #1: Segmentation
    enable_define_criteria: bool = True # TRIZ #10: Prior Action
    enable_list_instead: bool = True   # TRIZ #13: Inversion

    # R4: Kanban WIP Management
    enable_wip_cards: bool = True      # Master switch for WIP cards
    wip_limit_default: int = 2         # Default WIP limit
    wip_limit_mistral: int = 1         # Aggressive constraint for Mistral
    wip_limit_deepseek: int = 3        # Relaxed for champion
    wip_limit_granite: int = 3         # Relaxed for stable specialist

    # R4: Focus View (context optimization for 3B models)
    focus_view_enabled: bool = True
    max_done_items_shown: int = 3      # Compress DONE column

    # R4: Deepseek special handling
    deepseek_triz_turn_threshold: int = 15  # Only offer TRIZ after Turn N


@dataclass
class JustificationConfig:
    """Configuration for JustificationProtocol (R3)."""

    # Core settings
    enabled: bool = True               # Master switch for justification scoring
    require_justification: bool = True # Require justification for points

    # Quality thresholds
    excellent_threshold: float = 0.7   # Evidence match ratio for EXCELLENT
    good_threshold: float = 0.4        # Evidence match ratio for GOOD
    acceptable_threshold: float = 0.2  # Evidence match ratio for ACCEPTABLE

    # Scoring multipliers
    none_multiplier: float = 0.0       # Score multiplier for no justification
    weak_multiplier: float = 0.25      # Score multiplier for weak justification
    acceptable_multiplier: float = 0.5 # Score multiplier for acceptable
    good_multiplier: float = 0.75      # Score multiplier for good
    excellent_multiplier: float = 1.0  # Score multiplier for excellent

    # Goal proximity weights
    direct_weight: float = 1.0
    indirect_weight: float = 0.7
    exploratory_weight: float = 0.4
    tangential_weight: float = 0.1
    irrelevant_weight: float = 0.0


@dataclass
class SemanticConfig:
    """Configuration for Semantic Tools (Phase 1+)."""

    # Master switches
    enabled: bool = True               # Master switch for ALL semantic tools

    # === Phase 1: Semantic Intent Tracker ===
    intent_tracker_enabled: bool = True

    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    embedding_dim: int = 384                    # Embedding dimension
    use_ollama_embeddings: bool = False         # Use Ollama instead of sentence-transformers
    ollama_embedding_model: str = "nomic-embed-text"  # Ollama embedding model

    # Intent classification
    convergence_window: int = 3        # Number of recent actions for trajectory
    convergence_threshold: float = 0.1 # Cosine distance decrease to count as converging
    divergence_threshold: float = 0.1  # Cosine distance increase to count as diverging

    # Intent categories and their score multipliers
    converging_multiplier: float = 1.0   # Full score for converging intent
    stable_multiplier: float = 0.7       # Partial score for stable (not diverging)
    wandering_multiplier: float = 0.25   # Low score for wandering
    diverging_multiplier: float = 0.0    # Zero score for actively diverging

    # === Phase 2: Semantic Error Comprehension ===
    # Round 3: ENABLED to address phi3 "amnesia" (ErrorComprehensionLevel.NONE)
    error_comprehension_enabled: bool = True   # ENABLED for Round 3
    comprehension_threshold: float = 0.3       # Below this, inject ERROR_ANALYSIS card

    # === Phase 3: Semantic Card Relevance ===
    # Round 3: ENABLED to optimize card selection for granite/mistral
    card_relevance_enabled: bool = True        # ENABLED for Round 3
    card_relevance_threshold: float = 0.5      # Minimum relevance score to offer card

    # Caching
    cache_embeddings: bool = True      # Cache goal/action embeddings
    cache_size: int = 1000             # Max cached embeddings


@dataclass
class GameConfig:
    """Configuration for game mechanics."""

    # Turn limits
    max_turns: int = 10                # Maximum turns per game
    warning_turns: int = 8             # Warn model at this turn

    # Scoring
    base_action_score: int = 50        # Base score for valid action
    goal_bonus: int = 100              # Bonus for achieving goal
    error_penalty: int = -25           # Penalty for errors
    timeout_penalty: int = -50         # Penalty for timeout

    # Optimal path
    optimal_turn_bonus: int = 20       # Bonus per turn under optimal


@dataclass
class LoggingConfig:
    """Configuration for logging and output."""

    # Log levels
    log_level: str = "INFO"            # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True           # Write logs to file
    log_dir: str = ".agent_logs"       # Log directory

    # Output format
    json_logs: bool = True             # Use JSON-Lines format
    include_timestamps: bool = True    # Include timestamps
    include_embeddings: bool = False   # Include embeddings in logs (large!)

    # Verbosity
    verbose_failures: bool = True      # Detailed failure information
    verbose_cards: bool = True         # Detailed card information
    verbose_scores: bool = True        # Detailed scoring breakdown


@dataclass
class TutorConfig:
    """Master configuration aggregating all components."""

    # Component configs
    failure_detector: FailureDetectorConfig = field(default_factory=FailureDetectorConfig)
    meta_cards: MetaCardsConfig = field(default_factory=MetaCardsConfig)
    justification: JustificationConfig = field(default_factory=JustificationConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    game: GameConfig = field(default_factory=GameConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global version
    version: str = "0.3.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            "version": self.version,
            "failure_detector": self._dataclass_to_dict(self.failure_detector),
            "meta_cards": self._dataclass_to_dict(self.meta_cards),
            "justification": self._dataclass_to_dict(self.justification),
            "semantic": self._dataclass_to_dict(self.semantic),
            "game": self._dataclass_to_dict(self.game),
            "logging": self._dataclass_to_dict(self.logging),
        }

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dict."""
        return {k: v for k, v in obj.__dict__.items()}

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TutorConfig':
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = cls()

        if "failure_detector" in data:
            for k, v in data["failure_detector"].items():
                if hasattr(config.failure_detector, k):
                    setattr(config.failure_detector, k, v)

        if "meta_cards" in data:
            for k, v in data["meta_cards"].items():
                if hasattr(config.meta_cards, k):
                    setattr(config.meta_cards, k, v)

        if "justification" in data:
            for k, v in data["justification"].items():
                if hasattr(config.justification, k):
                    setattr(config.justification, k, v)

        if "semantic" in data:
            for k, v in data["semantic"].items():
                if hasattr(config.semantic, k):
                    setattr(config.semantic, k, v)

        if "game" in data:
            for k, v in data["game"].items():
                if hasattr(config.game, k):
                    setattr(config.game, k, v)

        if "logging" in data:
            for k, v in data["logging"].items():
                if hasattr(config.logging, k):
                    setattr(config.logging, k, v)

        return config


# =============================================================================
# ENVIRONMENT VARIABLE OVERRIDES
# =============================================================================

def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(key, default)


def load_config_with_env_overrides() -> TutorConfig:
    """
    Load configuration with environment variable overrides.

    Environment variables follow pattern: RAGIX_SECTION_PARAMETER
    Examples:
        RAGIX_SEMANTIC_ENABLED=false
        RAGIX_FAILURE_DETECTOR_REPETITION_THRESHOLD=5
        RAGIX_JUSTIFICATION_REQUIRE_JUSTIFICATION=true
    """
    config = TutorConfig()

    # === Semantic overrides ===
    config.semantic.enabled = _get_env_bool(
        "RAGIX_SEMANTIC_ENABLED", config.semantic.enabled)
    config.semantic.intent_tracker_enabled = _get_env_bool(
        "RAGIX_SEMANTIC_INTENT_TRACKER_ENABLED", config.semantic.intent_tracker_enabled)
    config.semantic.embedding_model = _get_env_str(
        "RAGIX_SEMANTIC_EMBEDDING_MODEL", config.semantic.embedding_model)
    config.semantic.use_ollama_embeddings = _get_env_bool(
        "RAGIX_SEMANTIC_USE_OLLAMA", config.semantic.use_ollama_embeddings)

    # === FailureDetector overrides ===
    config.failure_detector.enabled = _get_env_bool(
        "RAGIX_FAILURE_DETECTOR_ENABLED", config.failure_detector.enabled)
    config.failure_detector.repetition_threshold = _get_env_int(
        "RAGIX_FAILURE_DETECTOR_REPETITION_THRESHOLD", config.failure_detector.repetition_threshold)
    config.failure_detector.stall_threshold = _get_env_int(
        "RAGIX_FAILURE_DETECTOR_STALL_THRESHOLD", config.failure_detector.stall_threshold)

    # === Justification overrides ===
    config.justification.enabled = _get_env_bool(
        "RAGIX_JUSTIFICATION_ENABLED", config.justification.enabled)
    config.justification.require_justification = _get_env_bool(
        "RAGIX_JUSTIFICATION_REQUIRE_JUSTIFICATION", config.justification.require_justification)

    # === MetaCards overrides ===
    config.meta_cards.enabled = _get_env_bool(
        "RAGIX_META_CARDS_ENABLED", config.meta_cards.enabled)
    config.meta_cards.max_cards_per_game = _get_env_int(
        "RAGIX_META_CARDS_MAX_PER_GAME", config.meta_cards.max_cards_per_game)

    # === Game overrides ===
    config.game.max_turns = _get_env_int(
        "RAGIX_GAME_MAX_TURNS", config.game.max_turns)

    # === Logging overrides ===
    config.logging.log_level = _get_env_str(
        "RAGIX_LOG_LEVEL", config.logging.log_level)
    config.logging.json_logs = _get_env_bool(
        "RAGIX_LOG_JSON", config.logging.json_logs)

    return config


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

# Default global config instance
_global_config: Optional[TutorConfig] = None


def get_config() -> TutorConfig:
    """
    Get the global configuration instance.

    Lazy-loads with environment overrides on first access.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config_with_env_overrides()
    return _global_config


def set_config(config: TutorConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_config():
    """Reset to default configuration."""
    global _global_config
    _global_config = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_semantic_enabled() -> bool:
    """Check if semantic tools are enabled."""
    config = get_config()
    return config.semantic.enabled


def is_intent_tracker_enabled() -> bool:
    """Check if intent tracker is enabled."""
    config = get_config()
    return config.semantic.enabled and config.semantic.intent_tracker_enabled


def is_justification_enabled() -> bool:
    """Check if justification scoring is enabled."""
    config = get_config()
    return config.justification.enabled


def is_failure_detection_enabled() -> bool:
    """Check if failure detection is enabled."""
    config = get_config()
    return config.failure_detector.enabled


def is_meta_cards_enabled() -> bool:
    """Check if meta-cards are enabled."""
    config = get_config()
    return config.meta_cards.enabled


def is_error_comprehension_enabled() -> bool:
    """Check if semantic error comprehension (Phase 2) is enabled."""
    config = get_config()
    return config.semantic.enabled and config.semantic.error_comprehension_enabled


def is_card_relevance_enabled() -> bool:
    """Check if semantic card relevance (Phase 3) is enabled."""
    config = get_config()
    return config.semantic.enabled and config.semantic.card_relevance_enabled


# =============================================================================
# CONFIG SUMMARY
# =============================================================================

def print_config_summary():
    """Print a summary of current configuration."""
    config = get_config()

    print("=" * 70)
    print("RAGIX REASONING TUTOR v{} — CONFIGURATION".format(config.version))
    print("=" * 70)

    print("\n📊 COMPONENT STATUS:")
    print(f"  • FailureDetector (R1): {'✓ ENABLED' if config.failure_detector.enabled else '✗ DISABLED'}")
    print(f"  • MetaCards (R2):       {'✓ ENABLED' if config.meta_cards.enabled else '✗ DISABLED'}")
    print(f"  • Justification (R3):   {'✓ ENABLED' if config.justification.enabled else '✗ DISABLED'}")
    print(f"  • Semantic Tools:       {'✓ ENABLED' if config.semantic.enabled else '✗ DISABLED'}")
    print(f"    ├─ Intent Tracker (P1):     {'✓ ENABLED' if config.semantic.intent_tracker_enabled else '✗ DISABLED'}")
    print(f"    ├─ Error Comprehension (P2):{'✓ ENABLED' if config.semantic.error_comprehension_enabled else '✗ DISABLED'}")
    print(f"    └─ Card Relevance (P3):     {'✓ ENABLED' if config.semantic.card_relevance_enabled else '✗ DISABLED'}")

    print("\n⚙️  KEY PARAMETERS:")
    print(f"  • Repetition threshold: {config.failure_detector.repetition_threshold}")
    print(f"  • Stall threshold:      {config.failure_detector.stall_threshold}")
    print(f"  • Max turns:            {config.game.max_turns}")
    print(f"  • Embedding model:      {config.semantic.embedding_model}")

    print("\n📁 LOGGING:")
    print(f"  • Level: {config.logging.log_level}")
    print(f"  • JSON logs: {config.logging.json_logs}")
    print(f"  • Log dir: {config.logging.log_dir}")

    print("=" * 70)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo: show current config
    print_config_summary()

    # Demo: save to file
    config = get_config()
    demo_path = Path("tutor_config.json")
    config.save(demo_path)
    print(f"\nConfig saved to: {demo_path}")

    # Demo: environment override
    print("\n📝 Environment override example:")
    print("  export RAGIX_SEMANTIC_ENABLED=false")
    print("  export RAGIX_FAILURE_DETECTOR_REPETITION_THRESHOLD=5")
