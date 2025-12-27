#!/usr/bin/env python3
"""
Integration Tests for Meta-Cognitive Architecture v0.3
========================================================

Tests the full flow:
    FailureDetector (R1) → MetaCardSelector (R2) → JustificationEvaluator (R3)

Key integration scenarios:
1. Failure detection triggers card selection
2. Cards help break failure patterns
3. Justification scoring reflects improvement

Validated against LLM Olympics 2025-12-23 results.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from failure_detector import FailureDetector, FailureType, FailureContext
from meta_cards import MetaCardSelector, MetaCardType
from justification_protocol import (
    JustifiedAction,
    JustificationEvaluator,
    JustificationQuality,
    GoalProximity
)


class TestFailureToCardFlow:
    """Test integration: FailureDetector → MetaCardSelector."""

    def test_repetition_triggers_escape_card(self):
        """Repetition loop detection triggers ESCAPE_LOOP card."""
        # Setup
        detector = FailureDetector(repetition_threshold=3)
        selector = MetaCardSelector()

        # Simulate repetition
        for _ in range(3):
            detector.record_action("cat file.txt", "content", is_error=False)

        # Detect failure
        failure = detector.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.REPETITION_LOOP

        # Select card based on failure
        card = selector.select_card(
            failure.failure_type.value,
            {
                "repeated_action": failure.repeated_action,
                "turn_detected": failure.turn_detected
            }
        )

        assert card is not None
        assert card.card_type == MetaCardType.ESCAPE_LOOP
        assert failure.repeated_action in (card.pcg_hint or "")

    def test_circular_triggers_compass_card(self):
        """Circular pattern detection triggers COMPASS card."""
        detector = FailureDetector()
        selector = MetaCardSelector()

        # Simulate circular pattern
        for _ in range(2):
            detector.record_action("ls", "files", is_error=False)
            detector.record_action("cat README.md", "readme", is_error=False)

        failure = detector.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.CIRCULAR_PATTERN

        card = selector.select_card(
            failure.failure_type.value,
            {"pattern_sequence": failure.pattern_sequence}
        )

        assert card is not None
        assert card.card_type == MetaCardType.COMPASS

    def test_errors_trigger_analysis_card(self):
        """Consecutive errors trigger ERROR_ANALYSIS card."""
        detector = FailureDetector(error_threshold=3)
        selector = MetaCardSelector()

        # Simulate errors
        for i in range(3):
            detector.record_action(
                f"cat missing{i}.txt",
                "No such file",
                is_error=True
            )

        failure = detector.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.EXPLICIT_ERROR

        card = selector.select_card(
            failure.failure_type.value,
            {"last_error": "No such file", "error_count": 3}
        )

        assert card is not None
        assert card.card_type == MetaCardType.ERROR_ANALYSIS

    def test_stall_triggers_boost_card(self):
        """Progress stall triggers PROGRESS_BOOST card."""
        detector = FailureDetector(stall_threshold=4)
        selector = MetaCardSelector()

        # Initial progress
        detector.record_pcg_state(2)

        # Stall - actions without PCG growth
        for i in range(5):
            detector.record_action(f"cat file{i}.txt", "content", is_error=False)
            detector.record_pcg_state(2)

        failure = detector.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.PROGRESS_STALL

        card = selector.select_card(
            failure.failure_type.value,
            {"stall_turns": failure.stall_turns}
        )

        assert card is not None
        assert card.card_type == MetaCardType.PROGRESS_BOOST


class TestCardImpactOnJustification:
    """Test integration: MetaCards → JustificationProtocol."""

    def test_card_hint_improves_justification(self):
        """Card hints should help produce better justifications."""
        selector = MetaCardSelector()
        evaluator = JustificationEvaluator(
            goal_description="Find NEEDLE_VALUE",
            goal_keywords=["needle", "value", "find", "grep"]
        )

        # Get a card for repetition
        card = selector.select_card(
            "repetition_loop",
            {"repeated_action": "find . -name '*.txt'"}
        )

        # Action WITHOUT card guidance (unjustified)
        action_before = JustifiedAction(
            action="find . -name '*.txt'",  # Same repetitive action
            hypothesis="",
            expected_evidence="",
            goal_connection=""
        )

        # Action WITH card guidance (justified)
        action_after = JustifiedAction(
            action="grep -r 'NEEDLE' .",  # Different approach
            hypothesis="I expect grep to find NEEDLE_VALUE directly",
            expected_evidence="File path containing NEEDLE_VALUE",
            goal_connection="Direct search is more effective than listing files"
        )

        # Evaluate both
        eval_before = evaluator.evaluate_justification(
            action_before, "file1.txt\nfile2.txt", False
        )
        eval_after = evaluator.evaluate_justification(
            action_after, "./config.py:NEEDLE_VALUE='x'", True
        )

        # Card-guided action should score higher
        assert eval_after.score_multiplier() > eval_before.score_multiplier()


class TestFullPipelineScenarios:
    """Test complete scenarios from Olympics."""

    def test_granite_recovery_scenario(self):
        """
        Simulate granite3.1-moe:3b recovery pattern.

        B01: Repeated find commands → Card rescue → Success
        """
        detector = FailureDetector(repetition_threshold=3)
        selector = MetaCardSelector()
        evaluator = JustificationEvaluator(
            goal_description="Find needle in haystack",
            goal_keywords=["needle", "find", "haystack", "file"]
        )

        # Phase 1: Repetition (granite's B01 pattern)
        for _ in range(3):
            detector.record_action("find . -name '*.py'", "files...", is_error=False)

        failure = detector.detect_failure()
        assert failure is not None

        # Phase 2: Card intervention
        card = selector.select_card(failure.failure_type.value, {
            "repeated_action": "find . -name '*.py'"
        })
        assert card is not None

        # Phase 3: Model tries different approach (simulated card effect)
        detector.reset()  # Fresh start after card

        action = JustifiedAction(
            action="grep -rn 'needle' .",
            hypothesis="Grep will find needle directly",
            expected_evidence="Line numbers with needle",
            goal_connection="Direct content search"
        )

        evaluated = evaluator.evaluate_justification(
            action,
            "./haystack.txt:42:needle=found",
            goal_achieved=True
        )

        # Should be high quality after card guidance
        assert evaluated.justification_quality.value >= JustificationQuality.ACCEPTABLE.value
        assert evaluated.goal_proximity == GoalProximity.DIRECT

        # Record card success
        selector.record_outcome(card.card_id, success=True)
        assert card.success_rate == 1.0

    def test_phi3_failure_cascade(self):
        """
        Simulate phi3:latest failure cascade.

        26 failures, mostly EXPLICIT_ERROR from T3-T10.
        Shows that even cards can't save severe dysfunction.
        """
        detector = FailureDetector(error_threshold=3)
        selector = MetaCardSelector()

        failures_detected = 0
        cards_issued = 0

        # Simulate phi3's error cascade
        for turn in range(10):
            detector.record_action(
                f"invalid_{turn}",
                "Error: syntax error",
                is_error=True
            )
            detector.record_pcg_state(1)  # No progress

            failure = detector.detect_failure()
            if failure:
                failures_detected += 1
                card = selector.select_card(
                    failure.failure_type.value,
                    {"last_error": "syntax error"}
                )
                if card:
                    cards_issued += 1
                    # Simulate card failure (phi3 doesn't recover)
                    selector.record_outcome(card.card_id, success=False)

        # Should detect multiple failures
        assert failures_detected > 0

        # Cards issued but unsuccessful
        stats = selector.get_statistics()
        assert stats["overall_success_rate"] == 0.0

    def test_llama32_confabulation_detection(self):
        """
        Simulate llama3.2:3b confabulation pattern.

        High activity, low achievement → Justification protocol penalizes.
        """
        detector = FailureDetector(stall_threshold=4)
        evaluator = JustificationEvaluator(
            goal_description="Verify chain of trust",
            goal_keywords=["verify", "chain", "trust", "signature"]
        )

        # Phase 1: Lots of activity but no PCG progress
        detector.record_pcg_state(1)

        old_scores = []
        new_scores = []

        for i in range(7):
            detector.record_action(f"cat file{i}.txt", f"content {i}", is_error=False)
            detector.record_pcg_state(1)  # PCG never grows

            # Unjustified action (llama3.2 style)
            action = JustifiedAction(
                action=f"cat file{i}.txt",
                hypothesis="",  # No justification
                expected_evidence="",
                goal_connection=""
            )

            base_score = 50 + i * 10  # Increasing base scores
            old_scores.append(base_score)

            evaluated = evaluator.evaluate_justification(action, f"content {i}", False)
            adjusted = evaluator.compute_adjusted_score(base_score, evaluated)
            new_scores.append(adjusted)

        # Progress stall should be detected
        failure = detector.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.PROGRESS_STALL

        # Scores should be massively reduced
        total_old = sum(old_scores)
        total_new = sum(new_scores)

        reduction = (total_old - total_new) / total_old
        assert reduction > 0.8, f"Expected >80% reduction, got {reduction:.0%}"


class TestDeepseekPerfectRun:
    """
    Simulate deepseek-r1:14b perfect run pattern.

    0 failures, 6/6 wins, clean methodical approach.
    """

    def test_no_failures_detected(self):
        """Perfect run should have no failure detection."""
        detector = FailureDetector()

        # Methodical, non-repeating actions with progress
        actions = [
            ("ls -la", 1),
            ("cat requirements.txt", 2),
            ("grep -r 'config' .", 3),
            ("python -m pytest", 4),
            ("git status", 5),
        ]

        for action, pcg_size in actions:
            detector.record_action(action, "success", is_error=False)
            detector.record_pcg_state(pcg_size)

            failure = detector.detect_failure()
            assert failure is None, f"Unexpected failure at {action}"

    def test_justified_actions_score_well(self):
        """Well-justified methodical actions should score highly."""
        evaluator = JustificationEvaluator(
            goal_description="Run tests and verify code quality",
            goal_keywords=["test", "pytest", "verify", "quality", "check", "tests", "pass"]
        )

        justified_actions = [
            JustifiedAction(
                action="ls -la",
                hypothesis="List files to understand project structure",
                expected_evidence="Directory listing with test files",
                goal_connection="Need to find test files to verify quality"
            ),
            JustifiedAction(
                action="pytest -v",
                hypothesis="Run tests to verify code quality",
                expected_evidence="Test results showing pass/fail",
                goal_connection="Directly addresses the goal"
            ),
        ]

        outcomes = [
            "drwxr-xr-x tests/",
            "===== 5 passed ====="
        ]

        scores = []
        for action, outcome in zip(justified_actions, outcomes):
            evaluated = evaluator.evaluate_justification(
                action,
                outcome,
                goal_achieved=(action.action == "pytest -v")
            )
            scores.append(evaluated.score_multiplier())

        # First action is exploratory (lower score ok), second should be good
        # Relaxed threshold: at least one action > 0.3, and goal-achieving action >= 0.5
        assert scores[1] >= 0.5, f"Goal-achieving action should score >= 0.5, got {scores[1]}"
        assert sum(scores) > 0.5, f"Total scores should be > 0.5, got {sum(scores)}"


class TestCardEffectivenessTracking:
    """Test effectiveness tracking across the pipeline."""

    def test_card_success_rate_updates(self):
        """Card success rates should reflect actual outcomes."""
        selector = MetaCardSelector()

        # Issue multiple cards
        cards = []
        for failure_type in ["repetition_loop", "circular_pattern", "explicit_error"]:
            card = selector.select_card(failure_type, {})
            cards.append(card)

        # Record mixed outcomes
        selector.record_outcome(cards[0].card_id, True)   # Success
        selector.record_outcome(cards[1].card_id, False)  # Failure
        selector.record_outcome(cards[2].card_id, True)   # Success

        stats = selector.get_statistics()

        assert stats["total_cards_issued"] == 3
        assert stats["overall_success_rate"] == pytest.approx(2/3)

    def test_per_type_statistics(self):
        """Statistics should be tracked per card type."""
        selector = MetaCardSelector()

        # Issue 3 escape_loop cards, 2 succeed
        for i in range(3):
            card = selector.select_card("repetition_loop", {})
            selector.record_outcome(card.card_id, i < 2)

        # Issue 2 compass cards, 1 succeeds
        for i in range(2):
            card = selector.select_card("circular_pattern", {})
            selector.record_outcome(card.card_id, i < 1)

        stats = selector.get_statistics()

        assert stats["by_type"]["escape_loop"]["count"] == 3
        assert stats["by_type"]["escape_loop"]["rate"] == pytest.approx(2/3)
        assert stats["by_type"]["compass"]["count"] == 2
        assert stats["by_type"]["compass"]["rate"] == pytest.approx(1/2)


class TestJustificationSummary:
    """Test justification summary across sessions."""

    def test_summary_reflects_quality(self):
        """Summary should reflect overall justification quality."""
        evaluator = JustificationEvaluator(
            goal_description="Complete the task",
            goal_keywords=["complete", "task", "finish"]
        )

        # Mix of justified and unjustified actions
        actions = [
            JustifiedAction(action="ls", hypothesis="", expected_evidence="", goal_connection=""),  # None
            JustifiedAction(action="cat", hypothesis="Read file", expected_evidence="content", goal_connection="Info"),  # Some
            JustifiedAction(action="grep", hypothesis="Find target", expected_evidence="Match", goal_connection="Direct"),  # Good
        ]

        outcomes = ["files", "content here", "target found"]

        for action, outcome in zip(actions, outcomes):
            evaluator.evaluate_justification(action, outcome, False)

        summary = evaluator.get_summary()

        assert summary["total_actions"] == 3
        assert summary["none_count"] >= 1  # At least one unjustified
        assert summary["avg_quality"] < 1.0  # Not all excellent


# === End-to-End Scenario Tests ===

class TestEndToEndScenarios:
    """Complete end-to-end scenarios."""

    def test_recovery_after_card_intervention(self):
        """
        Full scenario: Failure → Card → Recovery → Success
        """
        # Components
        detector = FailureDetector(repetition_threshold=3)
        selector = MetaCardSelector()
        evaluator = JustificationEvaluator(
            goal_description="Find the secret key",
            goal_keywords=["secret", "key", "find", "config"]
        )

        # --- Turn 1-3: Repetition (failure) ---
        for turn in range(3):
            detector.record_action("ls -la", "files...", is_error=False)
            detector.record_pcg_state(1)

        failure = detector.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.REPETITION_LOOP

        # --- Card Intervention ---
        card = selector.select_card(
            failure.failure_type.value,
            {"repeated_action": "ls -la"}
        )
        assert card is not None

        # --- Turn 4: Changed approach (recovery) ---
        detector.reset()

        # Justified action based on card guidance
        recovery_action = JustifiedAction(
            action="grep -r 'SECRET_KEY' .",
            hypothesis="Direct search for secret key",
            expected_evidence="File path with SECRET_KEY",
            goal_connection="This directly searches for the target"
        )

        # Simulate success
        detector.record_action(recovery_action.action, "config.py:SECRET_KEY='abc'", is_error=False)
        detector.record_pcg_state(2)  # Progress!

        # No failure after recovery
        new_failure = detector.detect_failure()
        assert new_failure is None

        # Good justification score
        evaluated = evaluator.evaluate_justification(
            recovery_action,
            "config.py:SECRET_KEY='abc'",
            goal_achieved=True
        )

        assert evaluated.goal_proximity == GoalProximity.DIRECT
        assert evaluated.score_multiplier() >= 0.5

        # Record card success
        selector.record_outcome(card.card_id, success=True)

        # Verify card effectiveness
        stats = selector.get_statistics()
        assert stats["overall_success_rate"] == 1.0


# === Fixtures ===

@pytest.fixture
def full_pipeline():
    """Provide all three components for integration testing."""
    return {
        "detector": FailureDetector(),
        "selector": MetaCardSelector(),
        "evaluator": JustificationEvaluator(
            goal_description="Complete the benchmark task",
            goal_keywords=["complete", "task", "goal", "find", "verify"]
        )
    }


# === Run Tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
