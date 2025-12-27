#!/usr/bin/env python3
"""
Semantic Intent Tracker — Phase 1 Semantic Tool
=================================================

Tracks semantic intent trajectory to distinguish:
- CONVERGING: Actions semantically closer to goal each turn (deepseek pattern)
- STABLE: Maintaining reasonable distance to goal
- WANDERING: Random walk in semantic space (llama3.2 pattern)
- DIVERGING: Actions moving away from goal

This directly addresses the "Metric Bias" problem revealed by Olympics:
- llama3.2:3b: +4490 pts, 1/6 wins (high activity, aimless)
- deepseek-r1:14b: +1075 pts, 6/6 wins (focused, converging)

Integration:
    JustificationProtocol + SemanticIntentTracker
        ↓
    ADJUSTED_SCORE = BASE × JUSTIFICATION × INTENT_MULTIPLIER

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import warnings

# Import config
from config import get_config, is_intent_tracker_enabled


class IntentCategory(Enum):
    """Semantic intent categories based on trajectory analysis."""
    CONVERGING = "converging"    # Moving toward goal (deepseek pattern)
    STABLE = "stable"            # Maintaining position (acceptable)
    WANDERING = "wandering"      # Random walk (llama3.2 pattern)
    DIVERGING = "diverging"      # Moving away from goal


@dataclass
class IntentAnalysis:
    """Result of semantic intent analysis for an action."""
    category: IntentCategory
    confidence: float              # 0.0-1.0 confidence in classification
    distance_to_goal: float        # Current cosine distance to goal
    distance_delta: float          # Change from previous (negative = converging)
    trajectory_trend: str          # "decreasing", "stable", "increasing", "erratic"
    multiplier: float              # Score multiplier based on intent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "distance_to_goal": self.distance_to_goal,
            "distance_delta": self.distance_delta,
            "trajectory_trend": self.trajectory_trend,
            "multiplier": self.multiplier
        }


class EmbeddingProvider:
    """
    Abstract embedding provider with multiple backend support.

    Supports:
    - sentence-transformers (default, local)
    - Ollama embeddings API
    - Fallback to simple TF-IDF-like hashing
    """

    def __init__(self):
        self.config = get_config().semantic
        self._model = None
        self._backend = None
        self._cache: Dict[str, np.ndarray] = {}

    def _init_backend(self):
        """Lazy initialization of embedding backend."""
        if self._backend is not None:
            return

        # Try sentence-transformers first (unless Ollama explicitly requested)
        if not self.config.use_ollama_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.embedding_model)
                self._backend = "sentence-transformers"
                return
            except ImportError:
                warnings.warn(
                    "sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )

        # Try Ollama embeddings
        if self.config.use_ollama_embeddings or self._backend is None:
            try:
                import requests
                # Test Ollama connection
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    self._backend = "ollama"
                    return
            except Exception:
                pass

        # Fallback to simple hash-based embeddings
        warnings.warn(
            "No embedding backend available. Using fallback hash embeddings. "
            "For better results, install sentence-transformers or run Ollama."
        )
        self._backend = "fallback"

    def embed(self, text: str) -> np.ndarray:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (normalized)
        """
        self._init_backend()

        # Check cache
        if self.config.cache_embeddings and text in self._cache:
            return self._cache[text]

        # Generate embedding based on backend
        if self._backend == "sentence-transformers":
            embedding = self._embed_sentence_transformers(text)
        elif self._backend == "ollama":
            embedding = self._embed_ollama(text)
        else:
            embedding = self._embed_fallback(text)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache
        if self.config.cache_embeddings:
            if len(self._cache) >= self.config.cache_size:
                # Simple LRU: remove oldest entries
                keys = list(self._cache.keys())
                for k in keys[:len(keys)//2]:
                    del self._cache[k]
            self._cache[text] = embedding

        return embedding

    def _embed_sentence_transformers(self, text: str) -> np.ndarray:
        """Embed using sentence-transformers."""
        return self._model.encode(text, convert_to_numpy=True)

    def _embed_ollama(self, text: str) -> np.ndarray:
        """Embed using Ollama API."""
        import requests

        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": self.config.ollama_embedding_model,
                "prompt": text
            },
            timeout=30
        )

        if resp.status_code == 200:
            return np.array(resp.json()["embedding"])
        else:
            # Fallback if Ollama fails
            return self._embed_fallback(text)

    def _embed_fallback(self, text: str) -> np.ndarray:
        """
        Simple fallback embedding using character n-gram hashing.

        Not as good as neural embeddings but works without dependencies.
        """
        dim = self.config.embedding_dim
        embedding = np.zeros(dim)

        # Character 3-grams
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            ngram = text_lower[i:i+3]
            # Hash to embedding dimension
            idx = hash(ngram) % dim
            embedding[idx] += 1.0

        # Word-level features
        words = text_lower.split()
        for word in words:
            idx = hash(word) % dim
            embedding[idx] += 2.0  # Words weighted more

        return embedding

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine distance between two embeddings.

        Returns: 0.0 (identical) to 2.0 (opposite)
        """
        similarity = np.dot(a, b)
        return 1.0 - similarity  # Convert similarity to distance

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        self._init_backend()
        return self._backend


class SemanticIntentTracker:
    """
    Track semantic intent trajectory to detect convergence/divergence.

    This is the core Phase 1 semantic tool that addresses the llama3.2 anomaly.

    Usage:
        tracker = SemanticIntentTracker(goal="Find NEEDLE_VALUE in project")

        for action in game_actions:
            analysis = tracker.analyze(action)
            score_multiplier = analysis.multiplier

            # Integrate with JustificationProtocol
            final_score = base_score * justification_quality * score_multiplier
    """

    def __init__(self, goal: str):
        """
        Initialize tracker with goal.

        Args:
            goal: The goal description to track convergence toward
        """
        self.config = get_config().semantic
        self.goal = goal

        # Embedding provider
        self._embedder = EmbeddingProvider()

        # Goal embedding (computed once)
        self._goal_embedding: Optional[np.ndarray] = None

        # Trajectory history
        self.action_history: List[str] = []
        self.embedding_history: List[np.ndarray] = []
        self.distance_history: List[float] = []

    @property
    def goal_embedding(self) -> np.ndarray:
        """Lazy-load goal embedding."""
        if self._goal_embedding is None:
            self._goal_embedding = self._embedder.embed(self.goal)
        return self._goal_embedding

    def analyze(self, action: str) -> IntentAnalysis:
        """
        Analyze the semantic intent of an action.

        Args:
            action: The action/command to analyze

        Returns:
            IntentAnalysis with category, confidence, and multiplier
        """
        # Check if disabled
        if not is_intent_tracker_enabled():
            return IntentAnalysis(
                category=IntentCategory.STABLE,
                confidence=0.0,
                distance_to_goal=0.0,
                distance_delta=0.0,
                trajectory_trend="disabled",
                multiplier=1.0  # No penalty when disabled
            )

        # Get action embedding
        action_embedding = self._embedder.embed(action)

        # Compute distance to goal
        distance = self._embedder.cosine_distance(action_embedding, self.goal_embedding)

        # Store in history
        self.action_history.append(action)
        self.embedding_history.append(action_embedding)
        self.distance_history.append(distance)

        # Compute delta (negative = getting closer)
        if len(self.distance_history) > 1:
            distance_delta = distance - self.distance_history[-2]
        else:
            distance_delta = 0.0

        # Analyze trajectory trend
        trajectory_trend = self._analyze_trajectory()

        # Classify intent
        category, confidence = self._classify_intent(distance_delta, trajectory_trend)

        # Get multiplier from config
        multiplier = self._get_multiplier(category)

        return IntentAnalysis(
            category=category,
            confidence=confidence,
            distance_to_goal=distance,
            distance_delta=distance_delta,
            trajectory_trend=trajectory_trend,
            multiplier=multiplier
        )

    def _analyze_trajectory(self) -> str:
        """
        Analyze the trajectory over recent actions.

        Returns: "decreasing", "stable", "increasing", or "erratic"
        """
        window = self.config.convergence_window

        if len(self.distance_history) < 2:
            return "stable"

        # Get recent distances
        recent = self.distance_history[-window:] if len(self.distance_history) >= window else self.distance_history

        if len(recent) < 2:
            return "stable"

        # Compute deltas
        deltas = [recent[i+1] - recent[i] for i in range(len(recent)-1)]

        # Classify based on deltas
        decreasing = sum(1 for d in deltas if d < -self.config.convergence_threshold)
        increasing = sum(1 for d in deltas if d > self.config.divergence_threshold)
        stable = len(deltas) - decreasing - increasing

        if decreasing > increasing and decreasing > stable:
            return "decreasing"
        elif increasing > decreasing and increasing > stable:
            return "increasing"
        elif stable >= decreasing and stable >= increasing:
            return "stable"
        else:
            return "erratic"

    def _classify_intent(self, delta: float, trend: str) -> Tuple[IntentCategory, float]:
        """
        Classify intent based on delta, trend, AND absolute distance.

        Key insight from Olympics:
        - llama3.2 stayed at high distance (~0.9) consistently = WANDERING
        - deepseek decreased distance (0.9 → 0.3) = CONVERGING

        We must penalize "stable at high distance" (no progress).

        Returns: (category, confidence)
        """
        # Get current distance for absolute check
        current_distance = self.distance_history[-1] if self.distance_history else 1.0

        # Define "far from goal" threshold (>0.7 cosine distance is quite far)
        far_threshold = 0.7

        # Primary classification based on trend
        if trend == "decreasing":
            # Clearly converging
            return IntentCategory.CONVERGING, 0.9

        elif trend == "increasing":
            # Clearly diverging
            return IntentCategory.DIVERGING, 0.9

        elif trend == "stable":
            # CRITICAL: "Stable" at HIGH distance = WANDERING (llama3.2 pattern)
            # "Stable" at LOW distance = actually STABLE (near goal)
            if current_distance > far_threshold:
                # Stable but far = wandering without progress
                return IntentCategory.WANDERING, 0.85
            else:
                # Stable and close = maintaining good position
                # Check single-action delta for fine-tuning
                if delta < -self.config.convergence_threshold:
                    return IntentCategory.CONVERGING, 0.7
                elif delta > self.config.divergence_threshold:
                    return IntentCategory.DIVERGING, 0.7
                else:
                    return IntentCategory.STABLE, 0.8

        else:  # erratic
            return IntentCategory.WANDERING, 0.8

    def _get_multiplier(self, category: IntentCategory) -> float:
        """Get score multiplier for intent category."""
        if category == IntentCategory.CONVERGING:
            return self.config.converging_multiplier
        elif category == IntentCategory.STABLE:
            return self.config.stable_multiplier
        elif category == IntentCategory.WANDERING:
            return self.config.wandering_multiplier
        else:  # DIVERGING
            return self.config.diverging_multiplier

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of the trajectory so far."""
        if not self.distance_history:
            return {"actions": 0, "status": "no_data"}

        return {
            "actions": len(self.action_history),
            "initial_distance": self.distance_history[0],
            "current_distance": self.distance_history[-1],
            "total_change": self.distance_history[-1] - self.distance_history[0],
            "min_distance": min(self.distance_history),
            "max_distance": max(self.distance_history),
            "trajectory": self._analyze_trajectory(),
            "backend": self._embedder.backend_name
        }

    def reset(self):
        """Reset tracker for new game."""
        self.action_history.clear()
        self.embedding_history.clear()
        self.distance_history.clear()


# =============================================================================
# INTEGRATION WITH JUSTIFICATION PROTOCOL
# =============================================================================

def compute_semantic_score(
    base_score: int,
    justification_multiplier: float,
    intent_analysis: IntentAnalysis
) -> int:
    """
    Compute final score with semantic intent adjustment.

    Formula:
        FINAL_SCORE = BASE × JUSTIFICATION × INTENT_MULTIPLIER

    Args:
        base_score: Raw score from action
        justification_multiplier: Multiplier from JustificationProtocol (0.0-1.0)
        intent_analysis: Analysis from SemanticIntentTracker

    Returns:
        Adjusted score (integer)
    """
    return int(base_score * justification_multiplier * intent_analysis.multiplier)


# =============================================================================
# PHASE 2: SEMANTIC ERROR COMPREHENSION
# =============================================================================

class ErrorComprehensionLevel(Enum):
    """Levels of error comprehension by the model."""
    FULL = "full"            # Model understands error and fixes it
    PARTIAL = "partial"      # Model partially understands, attempts fix
    MINIMAL = "minimal"      # Model acknowledges error but doesn't understand
    NONE = "none"            # Model ignores or repeats the error


@dataclass
class ErrorComprehensionAnalysis:
    """Result of semantic error comprehension analysis."""
    comprehension_level: ErrorComprehensionLevel
    confidence: float
    error_embedding_similarity: float   # How similar action is to error fix
    action_addresses_error: bool        # Does action attempt to address error?
    multiplier: float                   # Score multiplier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comprehension_level": self.comprehension_level.value,
            "confidence": self.confidence,
            "error_embedding_similarity": self.error_embedding_similarity,
            "action_addresses_error": self.action_addresses_error,
            "multiplier": self.multiplier
        }


class SemanticErrorComprehension:
    """
    Phase 2: Analyze if the model comprehends errors semantically.

    Problem observed in Olympics:
    - phi3 had 26 failures, repeated same errors
    - Some models ignore error messages completely

    This tool measures:
    1. Semantic similarity between error message and follow-up action
    2. Whether the follow-up action addresses the error
    3. Triggers card intervention when comprehension is low

    Usage:
        comprehension = SemanticErrorComprehension()

        # After an error occurs
        analysis = comprehension.analyze_response(
            error_message="file not found: config.txt",
            follow_up_action="cat config.txt"  # Repeating same error!
        )

        if analysis.comprehension_level == ErrorComprehensionLevel.NONE:
            # Trigger ERROR_ANALYSIS card
    """

    def __init__(self):
        self.config = get_config().semantic
        self._embedder = EmbeddingProvider()

        # Error history for pattern detection
        self.error_history: List[str] = []
        self.response_history: List[str] = []
        self.comprehension_history: List[ErrorComprehensionLevel] = []

    def analyze_response(
        self,
        error_message: str,
        follow_up_action: str
    ) -> ErrorComprehensionAnalysis:
        """
        Analyze if the follow-up action shows comprehension of the error.

        Args:
            error_message: The error that occurred
            follow_up_action: The model's next action

        Returns:
            ErrorComprehensionAnalysis with comprehension level and multiplier
        """
        # Check if disabled
        if not self.config.enabled or not self.config.error_comprehension_enabled:
            return ErrorComprehensionAnalysis(
                comprehension_level=ErrorComprehensionLevel.FULL,
                confidence=0.0,
                error_embedding_similarity=0.0,
                action_addresses_error=True,
                multiplier=1.0
            )

        # Store history
        self.error_history.append(error_message)
        self.response_history.append(follow_up_action)

        # Get embeddings
        error_embedding = self._embedder.embed(error_message)
        action_embedding = self._embedder.embed(follow_up_action)

        # Compute semantic similarity
        similarity = 1.0 - self._embedder.cosine_distance(error_embedding, action_embedding)

        # Generate ideal fix description from error
        fix_description = self._infer_fix_from_error(error_message)
        fix_embedding = self._embedder.embed(fix_description)
        fix_similarity = 1.0 - self._embedder.cosine_distance(action_embedding, fix_embedding)

        # Detect if action is exact repeat of what caused error
        is_exact_repeat = self._is_repeat_error(error_message, follow_up_action)

        # Classify comprehension level
        level, confidence = self._classify_comprehension(
            error_similarity=similarity,
            fix_similarity=fix_similarity,
            is_repeat=is_exact_repeat
        )

        self.comprehension_history.append(level)

        # Get multiplier
        multiplier = self._get_multiplier(level)

        return ErrorComprehensionAnalysis(
            comprehension_level=level,
            confidence=confidence,
            error_embedding_similarity=fix_similarity,
            action_addresses_error=fix_similarity > 0.3,
            multiplier=multiplier
        )

    def _infer_fix_from_error(self, error_message: str) -> str:
        """Infer what a good fix would look like from error message."""
        error_lower = error_message.lower()

        # Common error patterns → fix descriptions
        if "no such file" in error_lower or "not found" in error_lower:
            return "check file exists, list directory, use correct path"
        elif "permission denied" in error_lower:
            return "check permissions, use sudo, change file permissions"
        elif "syntax error" in error_lower:
            return "fix command syntax, check quotes and parentheses"
        elif "command not found" in error_lower:
            return "use correct command name, check available commands"
        elif "timeout" in error_lower:
            return "reduce operation scope, try simpler command"
        else:
            return f"fix error: {error_message[:50]}"

    def _is_repeat_error(self, error_message: str, follow_up: str) -> bool:
        """Check if follow-up is likely to cause the same error."""
        # Extract key elements from error
        error_lower = error_message.lower()
        follow_lower = follow_up.lower()

        # If error mentioned a file and action uses same file, likely repeat
        # Simple heuristic: check for common substrings
        words_in_error = set(error_lower.split())
        words_in_follow = set(follow_lower.split())

        # If >50% of action words are in error, might be repeating
        if len(words_in_follow) > 0:
            overlap = len(words_in_error & words_in_follow) / len(words_in_follow)
            return overlap > 0.5

        return False

    def _classify_comprehension(
        self,
        error_similarity: float,
        fix_similarity: float,
        is_repeat: bool
    ) -> Tuple[ErrorComprehensionLevel, float]:
        """Classify comprehension level based on similarities."""
        # If exact repeat, no comprehension
        if is_repeat:
            return ErrorComprehensionLevel.NONE, 0.95

        # High fix similarity = good comprehension
        if fix_similarity > 0.5:
            return ErrorComprehensionLevel.FULL, 0.85
        elif fix_similarity > 0.3:
            return ErrorComprehensionLevel.PARTIAL, 0.75
        elif fix_similarity > 0.1:
            return ErrorComprehensionLevel.MINIMAL, 0.7
        else:
            return ErrorComprehensionLevel.NONE, 0.6

    def _get_multiplier(self, level: ErrorComprehensionLevel) -> float:
        """Get score multiplier for comprehension level."""
        multipliers = {
            ErrorComprehensionLevel.FULL: 1.0,
            ErrorComprehensionLevel.PARTIAL: 0.7,
            ErrorComprehensionLevel.MINIMAL: 0.4,
            ErrorComprehensionLevel.NONE: 0.1
        }
        return multipliers.get(level, 1.0)

    def should_trigger_card(self) -> bool:
        """Check if error comprehension is low enough to trigger card."""
        if not self.comprehension_history:
            return False

        # Trigger if last 2 compressions were NONE or MINIMAL
        recent = self.comprehension_history[-2:] if len(self.comprehension_history) >= 2 else self.comprehension_history
        low_comprehension = [
            c for c in recent
            if c in [ErrorComprehensionLevel.NONE, ErrorComprehensionLevel.MINIMAL]
        ]
        return len(low_comprehension) >= 2

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehension summary."""
        if not self.comprehension_history:
            return {"errors_analyzed": 0, "status": "no_data"}

        level_counts = {}
        for level in self.comprehension_history:
            level_counts[level.value] = level_counts.get(level.value, 0) + 1

        return {
            "errors_analyzed": len(self.comprehension_history),
            "comprehension_distribution": level_counts,
            "low_comprehension_ratio": (
                level_counts.get("none", 0) + level_counts.get("minimal", 0)
            ) / len(self.comprehension_history)
        }

    def reset(self):
        """Reset for new game."""
        self.error_history.clear()
        self.response_history.clear()
        self.comprehension_history.clear()


# =============================================================================
# PHASE 3: SEMANTIC CARD RELEVANCE
# =============================================================================

@dataclass
class CardRelevanceScore:
    """Relevance score for a card given the current context."""
    card_id: str
    card_type: str
    relevance_score: float      # 0.0-1.0 semantic relevance
    context_match: float        # How well card matches failure context
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "card_type": self.card_type,
            "relevance_score": self.relevance_score,
            "context_match": self.context_match,
            "confidence": self.confidence
        }


class SemanticCardRelevance:
    """
    Phase 3: Semantically match cards to failure contexts.

    Problem observed in Olympics:
    - Generic cards don't always help
    - Card selection based on failure type alone is coarse

    This tool:
    1. Embeds card instructions and failure contexts
    2. Computes semantic similarity
    3. Ranks cards by relevance to specific failure
    4. Enables more targeted interventions

    Usage:
        selector = SemanticCardRelevance()
        selector.register_card("escape_loop", "Try a completely different approach...")
        selector.register_card("compass", "You seem to be going in circles...")

        # When failure detected
        rankings = selector.rank_cards(
            failure_context="Model repeated 'ls -la' 3 times without progress",
            goal="Find NEEDLE_VALUE in source code"
        )

        best_card = rankings[0]  # Most relevant card
    """

    def __init__(self):
        self.config = get_config().semantic
        self._embedder = EmbeddingProvider()

        # Registered cards with their instruction embeddings
        self.cards: Dict[str, Dict[str, Any]] = {}
        self._card_embeddings: Dict[str, np.ndarray] = {}

        # Pre-register default cards
        self._register_default_cards()

    def _register_default_cards(self):
        """Register the standard meta-card templates."""
        default_cards = {
            "escape_loop": {
                "type": "ESCAPE_LOOP",
                "instruction": "Break free from repetition. The same action keeps failing. "
                              "Try a completely different approach. List alternative methods. "
                              "What else could work?"
            },
            "compass": {
                "type": "COMPASS",
                "instruction": "You're cycling between actions without progress. "
                              "Step back and consider: Where are you? What's the goal? "
                              "What's the shortest path from here to there?"
            },
            "error_analysis": {
                "type": "ERROR_ANALYSIS",
                "instruction": "An error keeps occurring. Read the error message carefully. "
                              "What exactly is failing? What's the root cause? "
                              "How can you avoid triggering this error?"
            },
            "progress_boost": {
                "type": "PROGRESS_BOOST",
                "instruction": "Progress has stalled. You're doing actions but not advancing. "
                              "What evidence have you gathered? What's still unknown? "
                              "Focus on the gap between current knowledge and goal."
            },
            "strategic_reset": {
                "type": "STRATEGIC_RESET",
                "instruction": "Multiple approaches have failed. Time for a strategic reset. "
                              "Forget previous attempts. Start fresh with new assumptions. "
                              "What's the simplest thing that could possibly work?"
            }
        }

        for card_id, card_info in default_cards.items():
            self.register_card(card_id, card_info["instruction"], card_info["type"])

    def register_card(self, card_id: str, instruction: str, card_type: str):
        """
        Register a card for semantic matching.

        Args:
            card_id: Unique identifier for the card
            instruction: The card's instruction text
            card_type: Type of card (ESCAPE_LOOP, COMPASS, etc.)
        """
        self.cards[card_id] = {
            "instruction": instruction,
            "type": card_type
        }
        # Cache embedding
        self._card_embeddings[card_id] = self._embedder.embed(instruction)

    def rank_cards(
        self,
        failure_context: str,
        goal: str,
        top_k: int = 3
    ) -> List[CardRelevanceScore]:
        """
        Rank cards by semantic relevance to the failure context.

        Args:
            failure_context: Description of the failure/stuck state
            goal: The current goal
            top_k: Number of top cards to return

        Returns:
            List of CardRelevanceScore, sorted by relevance (highest first)
        """
        # Check if disabled
        if not self.config.enabled or not self.config.card_relevance_enabled:
            # Return default ranking by card type
            return [
                CardRelevanceScore(
                    card_id=cid,
                    card_type=self.cards[cid]["type"],
                    relevance_score=1.0,
                    context_match=1.0,
                    confidence=0.0
                )
                for cid in list(self.cards.keys())[:top_k]
            ]

        # Embed the failure context
        context_embedding = self._embedder.embed(
            f"Problem: {failure_context}. Goal: {goal}"
        )

        # Score each card
        scores = []
        for card_id, card_embedding in self._card_embeddings.items():
            # Compute semantic similarity
            similarity = 1.0 - self._embedder.cosine_distance(
                context_embedding, card_embedding
            )

            # Also check goal relevance
            goal_embedding = self._embedder.embed(goal)
            goal_similarity = 1.0 - self._embedder.cosine_distance(
                goal_embedding, card_embedding
            )

            # Combined score (weighted)
            combined = 0.7 * similarity + 0.3 * goal_similarity

            scores.append(CardRelevanceScore(
                card_id=card_id,
                card_type=self.cards[card_id]["type"],
                relevance_score=combined,
                context_match=similarity,
                confidence=0.8 if similarity > 0.3 else 0.5
            ))

        # Sort by relevance
        scores.sort(key=lambda x: x.relevance_score, reverse=True)

        return scores[:top_k]

    def get_best_card(
        self,
        failure_context: str,
        goal: str
    ) -> Optional[CardRelevanceScore]:
        """Get the single best card for the context."""
        rankings = self.rank_cards(failure_context, goal, top_k=1)
        return rankings[0] if rankings else None

    def get_card_instruction(self, card_id: str) -> Optional[str]:
        """Get the instruction text for a card."""
        if card_id in self.cards:
            return self.cards[card_id]["instruction"]
        return None


# =============================================================================
# UNIFIED SEMANTIC ANALYZER
# =============================================================================

class SemanticAnalyzer:
    """
    Unified semantic analysis combining all three phases.

    This is the main entry point for semantic analysis in the game loop.

    Phases:
    - Phase 1: Intent tracking (converging vs wandering)
    - Phase 2: Error comprehension (understanding errors)
    - Phase 3: Card relevance (selecting best card)
    """

    def __init__(self, goal: str):
        self.goal = goal
        self.config = get_config().semantic

        # Phase 1: Intent Tracker
        self.intent_tracker = SemanticIntentTracker(goal)

        # Phase 2: Error Comprehension
        self.error_comprehension = SemanticErrorComprehension()

        # Phase 3: Card Relevance
        self.card_relevance = SemanticCardRelevance()

    def analyze_action(self, action: str) -> IntentAnalysis:
        """Analyze the semantic intent of an action (Phase 1)."""
        return self.intent_tracker.analyze(action)

    def analyze_error_response(
        self,
        error_message: str,
        follow_up_action: str
    ) -> ErrorComprehensionAnalysis:
        """Analyze error comprehension (Phase 2)."""
        return self.error_comprehension.analyze_response(error_message, follow_up_action)

    def select_best_card(
        self,
        failure_context: str
    ) -> Optional[CardRelevanceScore]:
        """Select the most relevant card for a failure (Phase 3)."""
        return self.card_relevance.get_best_card(failure_context, self.goal)

    def get_combined_multiplier(
        self,
        intent_analysis: Optional[IntentAnalysis] = None,
        error_analysis: Optional[ErrorComprehensionAnalysis] = None
    ) -> float:
        """
        Get combined score multiplier from all active phases.

        Multipliers are combined multiplicatively:
            FINAL = INTENT × ERROR_COMPREHENSION
        """
        multiplier = 1.0

        if intent_analysis:
            multiplier *= intent_analysis.multiplier

        if error_analysis:
            multiplier *= error_analysis.multiplier

        return multiplier

    def get_summary(self) -> Dict[str, Any]:
        """Get unified summary of all semantic analyses."""
        return {
            "phase1_intent": self.intent_tracker.get_trajectory_summary(),
            "phase2_errors": self.error_comprehension.get_summary(),
            "phase3_enabled": self.config.card_relevance_enabled,
            "phases_active": {
                "intent_tracker": self.config.intent_tracker_enabled,
                "error_comprehension": self.config.error_comprehension_enabled,
                "card_relevance": self.config.card_relevance_enabled
            }
        }

    def reset(self):
        """Reset all analyzers for new game."""
        self.intent_tracker.reset()
        self.error_comprehension.reset()


# =============================================================================
# DEMO
# =============================================================================

def demo_semantic_intent():
    """Demonstrate semantic intent tracking."""
    print("=" * 70)
    print("SEMANTIC INTENT TRACKER DEMO")
    print("Simulating llama3.2:3b (wandering) vs deepseek-r1:14b (converging)")
    print("=" * 70)

    goal = "Find NEEDLE_VALUE in the project source code"

    base_score = 50
    justification = 0.5  # Same justification quality for both

    # === Deepseek pattern: Converging ===
    print("\n📊 DEEPSEEK PATTERN (Converging toward goal):")
    print("-" * 60)

    tracker_deepseek = SemanticIntentTracker(goal)

    deepseek_actions = [
        "ls -la",                              # Explore structure
        "find . -name '*.py'",                 # Narrow to Python files
        "grep -r 'NEEDLE' .",                  # Search for keyword
        "grep -r 'NEEDLE_VALUE' src/",         # Refine search
        "cat src/config.py | grep NEEDLE",    # Read and extract
    ]

    deepseek_total = 0
    for action in deepseek_actions:
        analysis = tracker_deepseek.analyze(action)
        score = compute_semantic_score(base_score, justification, analysis)
        deepseek_total += score
        print(f"  {action[:35]:<35} → {analysis.category.value:<10} "
              f"dist={analysis.distance_to_goal:.2f} "
              f"mult={analysis.multiplier:.2f} "
              f"pts={score:+3d}")

    summary = tracker_deepseek.get_trajectory_summary()
    print(f"\n  📈 Trajectory: {summary['trajectory']}")
    print(f"  📉 Distance: {summary['initial_distance']:.2f} → {summary['current_distance']:.2f} "
          f"(Δ={summary['total_change']:+.2f})")
    print(f"  💰 TOTAL SCORE: {deepseek_total}")

    # === Llama3.2 pattern: Wandering ===
    print("\n📊 LLAMA3.2 PATTERN (Wandering at high distance):")
    print("-" * 60)

    tracker_llama = SemanticIntentTracker(goal)

    llama_actions = [
        "ls -la",                              # Explore
        "cat README.md",                       # Read unrelated
        "ls src/",                             # Back to listing
        "cat requirements.txt",                # Read unrelated
        "ls -la tests/",                       # Different directory
    ]

    llama_total = 0
    for action in llama_actions:
        analysis = tracker_llama.analyze(action)
        score = compute_semantic_score(base_score, justification, analysis)
        llama_total += score
        print(f"  {action[:35]:<35} → {analysis.category.value:<10} "
              f"dist={analysis.distance_to_goal:.2f} "
              f"mult={analysis.multiplier:.2f} "
              f"pts={score:+3d}")

    summary = tracker_llama.get_trajectory_summary()
    print(f"\n  📈 Trajectory: {summary['trajectory']}")
    print(f"  📉 Distance: {summary['initial_distance']:.2f} → {summary['current_distance']:.2f} "
          f"(Δ={summary['total_change']:+.2f})")
    print(f"  💰 TOTAL SCORE: {llama_total}")

    # === Comparison ===
    print("\n" + "=" * 60)
    print("📊 CUMULATIVE SCORE COMPARISON")
    print("=" * 60)
    print(f"  Goal: \"{goal}\"")
    print(f"  Base score per action: {base_score}")
    print(f"  Justification multiplier: {justification}")
    print()
    print(f"  Deepseek (focused):   {deepseek_total:>4} points ({len(deepseek_actions)} actions)")
    print(f"  Llama3.2 (wandering): {llama_total:>4} points ({len(llama_actions)} actions)")
    print()

    if deepseek_total > llama_total:
        diff = deepseek_total - llama_total
        pct = (deepseek_total / max(llama_total, 1) - 1) * 100
        print(f"  ✅ Deepseek earns {diff} more points ({pct:+.0f}%)")
        print(f"     → Focused convergence rewards strategic actions")
    else:
        print(f"  ⚠️  Scores are similar — both patterns penalized")

    print(f"\n  Backend: {tracker_deepseek._embedder.backend_name}")
    print()
    print("💡 KEY INSIGHT:")
    print("   The llama3.2 'Point Farmer' pattern (+4490 pts, 1/6 wins)")
    print("   would be detected and penalized by semantic intent tracking.")
    print("   Actions that don't converge toward goal get reduced scores.")


if __name__ == "__main__":
    demo_semantic_intent()
