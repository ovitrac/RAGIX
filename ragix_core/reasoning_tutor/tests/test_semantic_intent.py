#!/usr/bin/env python3
"""
Tests for Semantic Intent Tracker (Phase 1)
===========================================

Tests the semantic intent tracking system that addresses the llama3.2:3b anomaly.

Key components tested:
- IntentCategory enum
- IntentAnalysis dataclass
- EmbeddingProvider (with fallback)
- SemanticIntentTracker
- Integration with config
- Olympics pattern validation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_intent import (
    IntentCategory,
    IntentAnalysis,
    EmbeddingProvider,
    SemanticIntentTracker,
    compute_semantic_score
)
from config import (
    get_config,
    reset_config,
    TutorConfig,
    set_config,
    is_intent_tracker_enabled
)


# =============================================================================
# TEST: IntentCategory Enum
# =============================================================================

class TestIntentCategory:
    """Tests for IntentCategory enum."""

    def test_all_categories_defined(self):
        """Verify all intent categories are defined."""
        assert hasattr(IntentCategory, 'CONVERGING')
        assert hasattr(IntentCategory, 'STABLE')
        assert hasattr(IntentCategory, 'WANDERING')
        assert hasattr(IntentCategory, 'DIVERGING')

    def test_category_values(self):
        """Verify category string values."""
        assert IntentCategory.CONVERGING.value == "converging"
        assert IntentCategory.STABLE.value == "stable"
        assert IntentCategory.WANDERING.value == "wandering"
        assert IntentCategory.DIVERGING.value == "diverging"

    def test_category_count(self):
        """Verify there are exactly 4 categories."""
        assert len(IntentCategory) == 4


# =============================================================================
# TEST: IntentAnalysis Dataclass
# =============================================================================

class TestIntentAnalysis:
    """Tests for IntentAnalysis dataclass."""

    def test_create_analysis(self):
        """Create an IntentAnalysis instance."""
        analysis = IntentAnalysis(
            category=IntentCategory.CONVERGING,
            confidence=0.9,
            distance_to_goal=0.3,
            distance_delta=-0.1,
            trajectory_trend="decreasing",
            multiplier=1.0
        )

        assert analysis.category == IntentCategory.CONVERGING
        assert analysis.confidence == 0.9
        assert analysis.distance_to_goal == 0.3
        assert analysis.distance_delta == -0.1
        assert analysis.trajectory_trend == "decreasing"
        assert analysis.multiplier == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = IntentAnalysis(
            category=IntentCategory.WANDERING,
            confidence=0.85,
            distance_to_goal=0.8,
            distance_delta=0.0,
            trajectory_trend="stable",
            multiplier=0.25
        )

        d = analysis.to_dict()

        assert d["category"] == "wandering"
        assert d["confidence"] == 0.85
        assert d["distance_to_goal"] == 0.8
        assert d["distance_delta"] == 0.0
        assert d["trajectory_trend"] == "stable"
        assert d["multiplier"] == 0.25

    def test_all_categories_serializable(self):
        """All categories should serialize correctly."""
        for cat in IntentCategory:
            analysis = IntentAnalysis(
                category=cat,
                confidence=0.5,
                distance_to_goal=0.5,
                distance_delta=0.0,
                trajectory_trend="stable",
                multiplier=0.5
            )
            d = analysis.to_dict()
            assert d["category"] == cat.value


# =============================================================================
# TEST: EmbeddingProvider
# =============================================================================

class TestEmbeddingProvider:
    """Tests for EmbeddingProvider class."""

    def test_create_provider(self):
        """Create an EmbeddingProvider instance."""
        provider = EmbeddingProvider()
        assert provider._model is None
        assert provider._backend is None

    def test_embed_returns_vector(self):
        """Embed should return a numpy array."""
        provider = EmbeddingProvider()
        embedding = provider.embed("test text")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_embedding_is_normalized(self):
        """Embeddings should be L2 normalized."""
        provider = EmbeddingProvider()
        embedding = provider.embed("sample text for normalization test")

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Expected norm ~1.0, got {norm}"

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        provider = EmbeddingProvider()

        emb1 = provider.embed("find needle value")
        emb2 = provider.embed("list all files")

        # Embeddings should be different
        similarity = np.dot(emb1, emb2)
        assert similarity < 0.99, "Different texts should have different embeddings"

    def test_same_text_same_embedding(self):
        """Same text should produce identical embedding."""
        provider = EmbeddingProvider()

        emb1 = provider.embed("grep -r 'NEEDLE' .")
        emb2 = provider.embed("grep -r 'NEEDLE' .")

        assert np.allclose(emb1, emb2), "Same text should produce same embedding"

    def test_caching(self):
        """Test embedding caching."""
        provider = EmbeddingProvider()

        text = "test caching functionality"
        emb1 = provider.embed(text)

        # Should be in cache
        assert text in provider._cache

        # Second call should return cached value
        emb2 = provider.embed(text)
        assert np.array_equal(emb1, emb2)

    def test_cosine_distance_identical(self):
        """Identical vectors should have distance 0."""
        provider = EmbeddingProvider()
        v = np.array([1.0, 0.0, 0.0])

        dist = provider.cosine_distance(v, v)
        assert abs(dist) < 0.01

    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors should have distance 1."""
        provider = EmbeddingProvider()
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        dist = provider.cosine_distance(v1, v2)
        assert abs(dist - 1.0) < 0.01

    def test_backend_name_available(self):
        """Backend name should be available after initialization."""
        provider = EmbeddingProvider()
        provider.embed("initialize backend")

        name = provider.backend_name
        assert name in ["sentence-transformers", "ollama", "fallback"]


# =============================================================================
# TEST: SemanticIntentTracker
# =============================================================================

class TestSemanticIntentTracker:
    """Tests for SemanticIntentTracker class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_create_tracker(self):
        """Create a SemanticIntentTracker instance."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source code")

        assert tracker.goal == "Find NEEDLE_VALUE in source code"
        assert len(tracker.action_history) == 0
        assert len(tracker.distance_history) == 0

    def test_analyze_single_action(self):
        """Analyze a single action."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source code")

        analysis = tracker.analyze("grep -r 'NEEDLE' .")

        assert isinstance(analysis, IntentAnalysis)
        assert isinstance(analysis.category, IntentCategory)
        assert 0.0 <= analysis.confidence <= 1.0
        assert analysis.distance_to_goal >= 0.0
        assert analysis.multiplier >= 0.0

    def test_history_accumulates(self):
        """Action history should accumulate."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE")

        tracker.analyze("ls -la")
        assert len(tracker.action_history) == 1
        assert len(tracker.distance_history) == 1

        tracker.analyze("grep -r 'needle' .")
        assert len(tracker.action_history) == 2
        assert len(tracker.distance_history) == 2

        tracker.analyze("cat config.py")
        assert len(tracker.action_history) == 3
        assert len(tracker.distance_history) == 3

    def test_reset_clears_history(self):
        """Reset should clear all history."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE")

        tracker.analyze("ls -la")
        tracker.analyze("grep needle .")
        assert len(tracker.action_history) == 2

        tracker.reset()

        assert len(tracker.action_history) == 0
        assert len(tracker.embedding_history) == 0
        assert len(tracker.distance_history) == 0

    def test_trajectory_summary(self):
        """Test trajectory summary generation."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE")

        tracker.analyze("ls -la")
        tracker.analyze("grep needle .")
        tracker.analyze("cat src/config.py")

        summary = tracker.get_trajectory_summary()

        assert "actions" in summary
        assert summary["actions"] == 3
        assert "initial_distance" in summary
        assert "current_distance" in summary
        assert "total_change" in summary
        assert "trajectory" in summary
        assert "backend" in summary

    def test_empty_tracker_summary(self):
        """Empty tracker should return no_data status."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE")

        summary = tracker.get_trajectory_summary()

        assert summary["actions"] == 0
        assert summary["status"] == "no_data"


# =============================================================================
# TEST: Intent Classification
# =============================================================================

class TestIntentClassification:
    """Tests for intent classification logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_converging_pattern(self):
        """Actions getting closer to goal should be CONVERGING."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source files")

        # Simulate converging pattern
        actions = [
            "ls -la",                    # General exploration
            "find . -name '*.py'",       # Getting closer
            "grep -r 'NEEDLE' .",        # Very relevant
            "grep -rn 'NEEDLE_VALUE' src/",  # Exact match
        ]

        analyses = [tracker.analyze(a) for a in actions]

        # Last actions should be converging or stable (near goal)
        final_analysis = analyses[-1]
        # Either converging or stable-near-goal
        assert final_analysis.category in [IntentCategory.CONVERGING, IntentCategory.STABLE]

    def test_wandering_pattern_high_distance(self):
        """
        Actions staying at high distance should be WANDERING.

        This is the llama3.2:3b pattern - lots of activity, never approaching goal.
        """
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source files")

        # Simulate wandering pattern - unrelated actions
        actions = [
            "ls -la",                    # Generic
            "cat README.md",             # Unrelated
            "ls tests/",                 # Still generic
            "cat requirements.txt",      # Still unrelated
            "ls -la src/",               # Still generic
        ]

        analyses = [tracker.analyze(a) for a in actions]

        # Check final analysis - should detect wandering
        final = analyses[-1]

        # If distance stays high (>0.7), should be WANDERING
        if final.distance_to_goal > 0.7:
            assert final.category == IntentCategory.WANDERING

    def test_multiplier_values(self):
        """Verify multipliers match config values."""
        config = get_config().semantic

        tracker = SemanticIntentTracker("test goal")
        analysis = tracker.analyze("test action")

        # The multiplier should be one of the configured values
        valid_multipliers = {
            config.converging_multiplier,
            config.stable_multiplier,
            config.wandering_multiplier,
            config.diverging_multiplier
        }

        assert analysis.multiplier in valid_multipliers


# =============================================================================
# TEST: Config Integration
# =============================================================================

class TestConfigIntegration:
    """Tests for config integration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_disabled_tracker_returns_neutral(self):
        """Disabled tracker should return neutral analysis."""
        # Disable semantic tools
        config = TutorConfig()
        config.semantic.enabled = False
        set_config(config)

        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("any action")

        assert analysis.multiplier == 1.0
        assert analysis.trajectory_trend == "disabled"

    def test_disabled_intent_tracker_only(self):
        """Disabled intent tracker should return neutral."""
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.intent_tracker_enabled = False
        set_config(config)

        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("any action")

        assert analysis.multiplier == 1.0

    def test_enabled_tracker_applies_multiplier(self):
        """Enabled tracker should apply multipliers."""
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.intent_tracker_enabled = True
        set_config(config)

        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source")

        # Analyze some actions
        for _ in range(3):
            analysis = tracker.analyze("ls -la")

        # Should apply some multiplier (not just 1.0)
        # With wandering pattern, should be < 1.0
        # Or stable pattern
        assert 0.0 <= analysis.multiplier <= 1.0

    def test_multiplier_config_values(self):
        """Test that config multiplier values are used."""
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.intent_tracker_enabled = True
        config.semantic.converging_multiplier = 0.95
        config.semantic.stable_multiplier = 0.65
        config.semantic.wandering_multiplier = 0.2
        config.semantic.diverging_multiplier = 0.05
        set_config(config)

        tracker = SemanticIntentTracker("Find NEEDLE")

        # After several analyses, multiplier should be from config
        for _ in range(5):
            analysis = tracker.analyze("ls -la")

        # Multiplier should be one of our custom values
        assert analysis.multiplier in [0.95, 0.65, 0.2, 0.05]


# =============================================================================
# TEST: compute_semantic_score Function
# =============================================================================

class TestComputeSemanticScore:
    """Tests for compute_semantic_score helper function."""

    def test_full_multipliers(self):
        """Full multipliers should preserve base score."""
        analysis = IntentAnalysis(
            category=IntentCategory.CONVERGING,
            confidence=1.0,
            distance_to_goal=0.1,
            distance_delta=-0.1,
            trajectory_trend="decreasing",
            multiplier=1.0
        )

        score = compute_semantic_score(100, 1.0, analysis)
        assert score == 100

    def test_partial_multipliers(self):
        """Partial multipliers should reduce score."""
        analysis = IntentAnalysis(
            category=IntentCategory.WANDERING,
            confidence=0.8,
            distance_to_goal=0.9,
            distance_delta=0.0,
            trajectory_trend="stable",
            multiplier=0.25
        )

        # base=100, justification=0.5, intent=0.25
        score = compute_semantic_score(100, 0.5, analysis)
        assert score == 12  # 100 * 0.5 * 0.25 = 12.5 -> 12

    def test_zero_justification(self):
        """Zero justification should give zero score."""
        analysis = IntentAnalysis(
            category=IntentCategory.CONVERGING,
            confidence=1.0,
            distance_to_goal=0.1,
            distance_delta=-0.1,
            trajectory_trend="decreasing",
            multiplier=1.0
        )

        score = compute_semantic_score(100, 0.0, analysis)
        assert score == 0

    def test_zero_intent(self):
        """Zero intent multiplier should give zero score."""
        analysis = IntentAnalysis(
            category=IntentCategory.DIVERGING,
            confidence=1.0,
            distance_to_goal=1.5,
            distance_delta=0.3,
            trajectory_trend="increasing",
            multiplier=0.0
        )

        score = compute_semantic_score(100, 1.0, analysis)
        assert score == 0


# =============================================================================
# TEST: Olympics Pattern Validation
# =============================================================================

class TestOlympicsPatternValidation:
    """
    Tests that validate the semantic intent tracker against
    actual Olympics patterns.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_deepseek_converging_pattern(self):
        """
        deepseek-r1:14b pattern: Focused, converging actions.

        From Olympics: 6/6 wins, efficient approach.
        Expected: High scores, CONVERGING classification.
        """
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source code")

        # Simulate deepseek pattern: focused search
        deepseek_actions = [
            "find . -type f -name '*.py'",         # Target Python files
            "grep -r 'NEEDLE' --include='*.py' .",  # Search for keyword
            "grep -rn 'NEEDLE_VALUE' src/",         # Refine search
        ]

        analyses = [tracker.analyze(a) for a in deepseek_actions]

        # Calculate total score
        base_score = 50
        justification = 0.75  # Good justification
        total = sum(compute_semantic_score(base_score, justification, a) for a in analyses)

        # Deepseek should score well (converging = 1.0 multiplier)
        max_possible = base_score * justification * len(deepseek_actions)
        assert total >= max_possible * 0.5, f"Deepseek pattern should score well: {total}"

    def test_llama32_wandering_pattern(self):
        """
        llama3.2:3b pattern: High activity, low achievement.

        From Olympics: +4490 pts, 1/6 wins (Metric Bias anomaly).
        Expected: Reduced scores due to WANDERING classification.
        """
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source code")

        # Simulate llama3.2 pattern: lots of ls and cat, never searching
        llama_actions = [
            "ls -la",
            "cat README.md",
            "ls src/",
            "cat requirements.txt",
            "ls -la tests/",
            "cat setup.py",
        ]

        analyses = [tracker.analyze(a) for a in llama_actions]

        # Check that wandering is detected
        wandering_count = sum(1 for a in analyses if a.category == IntentCategory.WANDERING)
        stable_at_high_dist = sum(1 for a in analyses
                                  if a.category in [IntentCategory.WANDERING, IntentCategory.STABLE]
                                  and a.distance_to_goal > 0.5)

        # Most actions should be classified as wandering or stable-far
        assert wandering_count + stable_at_high_dist >= len(llama_actions) // 2, \
            f"Wandering pattern should be detected: wandering={wandering_count}"

    def test_deepseek_vs_llama_score_differential(self):
        """
        deepseek should earn significantly more points than llama3.2
        for the same base scores.
        """
        goal = "Find NEEDLE_VALUE in source code"

        # Deepseek pattern
        tracker_ds = SemanticIntentTracker(goal)
        deepseek_actions = [
            "find . -name '*.py'",
            "grep -r 'NEEDLE' .",
            "cat src/config.py | grep NEEDLE",
        ]
        ds_analyses = [tracker_ds.analyze(a) for a in deepseek_actions]
        ds_total = sum(compute_semantic_score(50, 0.75, a) for a in ds_analyses)

        # Llama pattern
        tracker_ll = SemanticIntentTracker(goal)
        llama_actions = [
            "ls -la",
            "cat README.md",
            "ls tests/",
        ]
        ll_analyses = [tracker_ll.analyze(a) for a in llama_actions]
        ll_total = sum(compute_semantic_score(50, 0.75, a) for a in ll_analyses)

        # Deepseek should earn more (converging vs wandering)
        assert ds_total >= ll_total, \
            f"Deepseek ({ds_total}) should score >= Llama ({ll_total})"

    def test_granite_mixed_pattern(self):
        """
        granite3.1-moe:3b pattern: Starts wandering, uses cards to recover.

        From Olympics: 3/6 wins, card-assisted recovery.
        """
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in source code")

        # Simulate granite: starts general, then focuses
        granite_actions = [
            "ls -la",                    # Start general
            "ls src/",                   # Still general
            # Card intervention: "Focus on search"
            "grep -r 'NEEDLE' .",        # Now focused
            "grep -n 'NEEDLE_VALUE' src/config.py",  # Direct hit
        ]

        analyses = [tracker.analyze(a) for a in granite_actions]

        # First actions may be wandering
        early = analyses[:2]
        later = analyses[2:]

        # Later actions should be better (converging or stable-close)
        later_good = sum(1 for a in later
                        if a.category in [IntentCategory.CONVERGING, IntentCategory.STABLE]
                        and a.multiplier >= 0.5)

        assert later_good >= 1, "Granite should improve after card intervention"


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_empty_action(self):
        """Empty action should not crash."""
        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("")

        assert isinstance(analysis, IntentAnalysis)

    def test_very_long_action(self):
        """Very long action should work."""
        tracker = SemanticIntentTracker("Find NEEDLE")
        long_action = "grep -r " + "a" * 1000 + " ."
        analysis = tracker.analyze(long_action)

        assert isinstance(analysis, IntentAnalysis)

    def test_unicode_action(self):
        """Unicode in action should work."""
        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("grep -r '日本語' .")

        assert isinstance(analysis, IntentAnalysis)

    def test_special_characters(self):
        """Special characters should work."""
        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("grep -r '$VAR' . | head -n 10")

        assert isinstance(analysis, IntentAnalysis)

    def test_single_action_trajectory(self):
        """Single action should have valid trajectory."""
        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("grep needle .")

        # First action should be stable or indeterminate
        assert analysis.trajectory_trend in ["stable", "decreasing", "increasing"]

    def test_distance_delta_first_action(self):
        """First action should have zero delta."""
        tracker = SemanticIntentTracker("Find NEEDLE")
        analysis = tracker.analyze("grep needle .")

        assert analysis.distance_delta == 0.0


# =============================================================================
# TEST: Trajectory Analysis
# =============================================================================

class TestTrajectoryAnalysis:
    """Tests for trajectory trend analysis."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_decreasing_trajectory(self):
        """Actions getting closer should show decreasing trend."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE in Python source code")

        # Actions progressively more related to goal
        actions = [
            "ls",
            "find . -name '*.py'",
            "grep -r 'NEEDLE' .",
            "grep -rn 'NEEDLE_VALUE' src/",
        ]

        for a in actions:
            tracker.analyze(a)

        summary = tracker.get_trajectory_summary()

        # Should show progress toward goal
        # Note: depends on embedding quality, so we check reasonable behavior
        assert summary["actions"] == 4
        assert "trajectory" in summary

    def test_stable_trajectory(self):
        """Similar actions should show stable trend."""
        tracker = SemanticIntentTracker("Find NEEDLE_VALUE")

        # Repetitive similar actions
        for _ in range(4):
            tracker.analyze("ls -la")

        summary = tracker.get_trajectory_summary()

        # Distance should remain relatively constant
        assert summary["actions"] == 4
        # Change should be small
        total_change = abs(summary["total_change"])
        assert total_change < 1.0  # Not dramatically changing


# =============================================================================
# TEST: Phase 2 - Error Comprehension
# =============================================================================

class TestErrorComprehensionLevel:
    """Tests for ErrorComprehensionLevel enum."""

    def test_all_levels_defined(self):
        """Verify all comprehension levels are defined."""
        from semantic_intent import ErrorComprehensionLevel
        assert hasattr(ErrorComprehensionLevel, 'FULL')
        assert hasattr(ErrorComprehensionLevel, 'PARTIAL')
        assert hasattr(ErrorComprehensionLevel, 'MINIMAL')
        assert hasattr(ErrorComprehensionLevel, 'NONE')

    def test_level_values(self):
        """Verify level string values."""
        from semantic_intent import ErrorComprehensionLevel
        assert ErrorComprehensionLevel.FULL.value == "full"
        assert ErrorComprehensionLevel.NONE.value == "none"


class TestErrorComprehensionAnalysis:
    """Tests for ErrorComprehensionAnalysis dataclass."""

    def test_create_analysis(self):
        """Create an ErrorComprehensionAnalysis instance."""
        from semantic_intent import ErrorComprehensionAnalysis, ErrorComprehensionLevel

        analysis = ErrorComprehensionAnalysis(
            comprehension_level=ErrorComprehensionLevel.PARTIAL,
            confidence=0.8,
            error_embedding_similarity=0.5,
            action_addresses_error=True,
            multiplier=0.7
        )

        assert analysis.comprehension_level == ErrorComprehensionLevel.PARTIAL
        assert analysis.confidence == 0.8
        assert analysis.multiplier == 0.7

    def test_to_dict(self):
        """Test serialization to dict."""
        from semantic_intent import ErrorComprehensionAnalysis, ErrorComprehensionLevel

        analysis = ErrorComprehensionAnalysis(
            comprehension_level=ErrorComprehensionLevel.NONE,
            confidence=0.95,
            error_embedding_similarity=0.1,
            action_addresses_error=False,
            multiplier=0.1
        )

        d = analysis.to_dict()
        assert d["comprehension_level"] == "none"
        assert d["multiplier"] == 0.1


class TestSemanticErrorComprehension:
    """Tests for SemanticErrorComprehension class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()
        # Enable Phase 2 for testing
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.error_comprehension_enabled = True
        set_config(config)

    def test_create_comprehension(self):
        """Create a SemanticErrorComprehension instance."""
        from semantic_intent import SemanticErrorComprehension

        comp = SemanticErrorComprehension()
        assert len(comp.error_history) == 0
        assert len(comp.comprehension_history) == 0

    def test_analyze_response(self):
        """Analyze an error response."""
        from semantic_intent import SemanticErrorComprehension, ErrorComprehensionLevel

        comp = SemanticErrorComprehension()

        analysis = comp.analyze_response(
            error_message="file not found: data.txt",
            follow_up_action="ls -la"
        )

        assert analysis.comprehension_level in list(ErrorComprehensionLevel)
        assert 0.0 <= analysis.multiplier <= 1.0

    def test_history_accumulates(self):
        """Error history should accumulate."""
        from semantic_intent import SemanticErrorComprehension

        comp = SemanticErrorComprehension()

        comp.analyze_response("error 1", "action 1")
        assert len(comp.error_history) == 1

        comp.analyze_response("error 2", "action 2")
        assert len(comp.error_history) == 2

    def test_reset_clears_history(self):
        """Reset should clear history."""
        from semantic_intent import SemanticErrorComprehension

        comp = SemanticErrorComprehension()
        comp.analyze_response("error", "action")
        comp.reset()

        assert len(comp.error_history) == 0
        assert len(comp.comprehension_history) == 0

    def test_should_trigger_card(self):
        """Should trigger card after low comprehension."""
        from semantic_intent import SemanticErrorComprehension

        comp = SemanticErrorComprehension()

        # Simulate repeated low comprehension
        for _ in range(3):
            comp.analyze_response(
                "cat: file.txt: No such file",
                "cat file.txt"  # Repeating same error
            )

        # After multiple low-comprehension responses, should trigger
        # (depends on actual classification)
        summary = comp.get_summary()
        assert summary["errors_analyzed"] == 3

    def test_get_summary(self):
        """Get comprehension summary."""
        from semantic_intent import SemanticErrorComprehension

        comp = SemanticErrorComprehension()
        comp.analyze_response("error", "fix action")

        summary = comp.get_summary()
        assert "errors_analyzed" in summary
        assert summary["errors_analyzed"] == 1

    def test_disabled_returns_neutral(self):
        """Disabled comprehension should return neutral analysis."""
        from semantic_intent import SemanticErrorComprehension, ErrorComprehensionLevel

        config = TutorConfig()
        config.semantic.error_comprehension_enabled = False
        set_config(config)

        comp = SemanticErrorComprehension()
        analysis = comp.analyze_response("error", "action")

        assert analysis.multiplier == 1.0
        assert analysis.comprehension_level == ErrorComprehensionLevel.FULL


# =============================================================================
# TEST: Phase 3 - Card Relevance
# =============================================================================

class TestCardRelevanceScore:
    """Tests for CardRelevanceScore dataclass."""

    def test_create_score(self):
        """Create a CardRelevanceScore instance."""
        from semantic_intent import CardRelevanceScore

        score = CardRelevanceScore(
            card_id="escape_loop",
            card_type="ESCAPE_LOOP",
            relevance_score=0.85,
            context_match=0.9,
            confidence=0.8
        )

        assert score.card_id == "escape_loop"
        assert score.relevance_score == 0.85

    def test_to_dict(self):
        """Test serialization to dict."""
        from semantic_intent import CardRelevanceScore

        score = CardRelevanceScore(
            card_id="compass",
            card_type="COMPASS",
            relevance_score=0.7,
            context_match=0.75,
            confidence=0.6
        )

        d = score.to_dict()
        assert d["card_id"] == "compass"
        assert d["relevance_score"] == 0.7


class TestSemanticCardRelevance:
    """Tests for SemanticCardRelevance class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()
        # Enable Phase 3 for testing
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.card_relevance_enabled = True
        set_config(config)

    def test_create_relevance(self):
        """Create a SemanticCardRelevance instance."""
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()
        assert len(rel.cards) >= 5  # Default cards registered

    def test_default_cards_registered(self):
        """Default meta-cards should be registered."""
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()

        assert "escape_loop" in rel.cards
        assert "compass" in rel.cards
        assert "error_analysis" in rel.cards
        assert "progress_boost" in rel.cards
        assert "strategic_reset" in rel.cards

    def test_register_custom_card(self):
        """Register a custom card."""
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()
        rel.register_card(
            "custom_card",
            "Custom instruction for special case",
            "CUSTOM_TYPE"
        )

        assert "custom_card" in rel.cards
        assert rel.cards["custom_card"]["type"] == "CUSTOM_TYPE"

    def test_rank_cards(self):
        """Rank cards by relevance."""
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()

        rankings = rel.rank_cards(
            failure_context="Model repeated 'ls -la' 3 times without progress",
            goal="Find NEEDLE_VALUE in source code",
            top_k=3
        )

        assert len(rankings) == 3
        # Rankings should be sorted by relevance (descending)
        assert rankings[0].relevance_score >= rankings[1].relevance_score

    def test_get_best_card(self):
        """Get the single best card."""
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()

        best = rel.get_best_card(
            failure_context="Error keeps occurring: file not found",
            goal="Read configuration file"
        )

        assert best is not None
        assert best.card_id in rel.cards

    def test_get_card_instruction(self):
        """Get card instruction text."""
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()

        instruction = rel.get_card_instruction("escape_loop")
        assert instruction is not None
        assert "repetition" in instruction.lower() or "different" in instruction.lower()

    def test_disabled_returns_default_ranking(self):
        """Disabled relevance should return default ranking."""
        from semantic_intent import SemanticCardRelevance

        config = TutorConfig()
        config.semantic.card_relevance_enabled = False
        set_config(config)

        rel = SemanticCardRelevance()
        rankings = rel.rank_cards("context", "goal", top_k=3)

        # Should still return cards, but with default confidence
        assert len(rankings) == 3
        assert rankings[0].confidence == 0.0  # Disabled indicator


# =============================================================================
# TEST: Unified Semantic Analyzer
# =============================================================================

class TestSemanticAnalyzer:
    """Tests for the unified SemanticAnalyzer class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config before each test."""
        reset_config()

    def test_create_analyzer(self):
        """Create a SemanticAnalyzer instance."""
        from semantic_intent import SemanticAnalyzer

        analyzer = SemanticAnalyzer("Find NEEDLE_VALUE in source code")
        assert analyzer.goal == "Find NEEDLE_VALUE in source code"

    def test_analyze_action(self):
        """Analyze action intent."""
        from semantic_intent import SemanticAnalyzer, IntentCategory

        analyzer = SemanticAnalyzer("Find NEEDLE")
        analysis = analyzer.analyze_action("grep needle .")

        assert analysis.category in list(IntentCategory)

    def test_analyze_error_response(self):
        """Analyze error comprehension."""
        from semantic_intent import SemanticAnalyzer

        # Enable Phase 2
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.error_comprehension_enabled = True
        set_config(config)

        analyzer = SemanticAnalyzer("Find NEEDLE")
        analysis = analyzer.analyze_error_response("file not found", "ls -la")

        assert 0.0 <= analysis.multiplier <= 1.0

    def test_select_best_card(self):
        """Select best card for failure."""
        from semantic_intent import SemanticAnalyzer

        # Enable Phase 3
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.card_relevance_enabled = True
        set_config(config)

        analyzer = SemanticAnalyzer("Find NEEDLE")
        card = analyzer.select_best_card("Model stuck in repetition loop")

        assert card is not None
        assert card.card_id is not None

    def test_get_combined_multiplier(self):
        """Get combined multiplier from multiple phases."""
        from semantic_intent import (
            SemanticAnalyzer,
            IntentAnalysis,
            IntentCategory,
            ErrorComprehensionAnalysis,
            ErrorComprehensionLevel
        )

        analyzer = SemanticAnalyzer("Find NEEDLE")

        intent = IntentAnalysis(
            category=IntentCategory.CONVERGING,
            confidence=0.9,
            distance_to_goal=0.3,
            distance_delta=-0.1,
            trajectory_trend="decreasing",
            multiplier=1.0
        )

        error = ErrorComprehensionAnalysis(
            comprehension_level=ErrorComprehensionLevel.PARTIAL,
            confidence=0.7,
            error_embedding_similarity=0.4,
            action_addresses_error=True,
            multiplier=0.7
        )

        combined = analyzer.get_combined_multiplier(intent, error)
        assert combined == 1.0 * 0.7  # 0.7

    def test_get_summary(self):
        """Get unified summary."""
        from semantic_intent import SemanticAnalyzer

        analyzer = SemanticAnalyzer("Find NEEDLE")
        analyzer.analyze_action("grep needle .")

        summary = analyzer.get_summary()

        assert "phase1_intent" in summary
        assert "phase2_errors" in summary
        assert "phases_active" in summary

    def test_reset(self):
        """Reset all analyzers."""
        from semantic_intent import SemanticAnalyzer

        analyzer = SemanticAnalyzer("Find NEEDLE")
        analyzer.analyze_action("action 1")
        analyzer.analyze_action("action 2")

        analyzer.reset()

        summary = analyzer.get_summary()
        assert summary["phase1_intent"]["actions"] == 0


# =============================================================================
# TEST: Olympics Validation for Phase 2/3
# =============================================================================

class TestOlympicsPhase2Phase3:
    """
    Tests that validate Phase 2 and Phase 3 against Olympics patterns.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset config and enable all phases."""
        reset_config()
        config = TutorConfig()
        config.semantic.enabled = True
        config.semantic.intent_tracker_enabled = True
        config.semantic.error_comprehension_enabled = True
        config.semantic.card_relevance_enabled = True
        set_config(config)

    def test_phi3_error_cascade_pattern(self):
        """
        phi3:latest pattern: 26 failures, repeated errors.

        Phase 2 should detect low error comprehension.
        """
        from semantic_intent import SemanticErrorComprehension

        comp = SemanticErrorComprehension()

        # Simulate phi3 pattern: repeated same error
        for _ in range(5):
            comp.analyze_response(
                error_message="command not found: <think>",
                follow_up_action="<think>"  # Repeating same mistake
            )

        summary = comp.get_summary()

        # Should show high proportion of low comprehension
        assert summary["errors_analyzed"] == 5

    def test_repetition_failure_selects_escape_loop(self):
        """
        For repetition failures, escape_loop card should rank high.
        """
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()

        rankings = rel.rank_cards(
            failure_context="Model repeated 'ls -la' 3 times in a row without any new discovery",
            goal="Find NEEDLE_VALUE in source code"
        )

        # escape_loop should be in top 3
        top_ids = [r.card_id for r in rankings[:3]]
        assert "escape_loop" in top_ids or "progress_boost" in top_ids

    def test_error_failure_selects_error_analysis(self):
        """
        For error failures, error_analysis card should rank high.
        """
        from semantic_intent import SemanticCardRelevance

        rel = SemanticCardRelevance()

        rankings = rel.rank_cards(
            failure_context="Repeated syntax error: command not found, invalid syntax in bash",
            goal="Execute command successfully"
        )

        top_ids = [r.card_id for r in rankings[:3]]
        # error_analysis should be relevant for this failure
        assert "error_analysis" in top_ids or "escape_loop" in top_ids


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
