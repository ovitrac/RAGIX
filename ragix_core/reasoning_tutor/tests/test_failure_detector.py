#!/usr/bin/env python3
"""
Test Suite for FailureDetector (R1)
====================================

Tests meta-cognitive detection of stuck states:
- REPETITION_LOOP (perseveration)
- CIRCULAR_PATTERN (disorientation)
- EXPLICIT_ERROR (agnosia)
- PROGRESS_STALL (confabulation)
- EXHAUSTION

Validated against LLM Olympics 2025-12-23 results.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from failure_detector import (
    FailureDetector,
    FailureType,
    FailureContext,
    get_fat_llm_instruction
)


class TestFailureDetectorInit:
    """Test FailureDetector initialization."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        fd = FailureDetector()
        assert fd.repetition_threshold == 3
        assert fd.stall_threshold == 4
        assert fd.error_threshold == 3

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        fd = FailureDetector(
            repetition_threshold=5,
            stall_threshold=6,
            error_threshold=4
        )
        assert fd.repetition_threshold == 5
        assert fd.stall_threshold == 6
        assert fd.error_threshold == 4

    def test_initial_state(self):
        """Test initial state is clean."""
        fd = FailureDetector()
        assert fd.current_turn == 0
        assert len(fd.action_history) == 0
        assert len(fd.error_history) == 0
        assert len(fd.pcg_node_counts) == 0


class TestRepetitionLoopDetection:
    """Test REPETITION_LOOP detection (perseveration / syntactic aphasia)."""

    def test_no_repetition(self):
        """No failure when actions are different."""
        fd = FailureDetector(repetition_threshold=3)

        fd.record_action("ls -la", "output1", is_error=False)
        fd.record_action("cat file.txt", "output2", is_error=False)
        fd.record_action("grep pattern .", "output3", is_error=False)

        failure = fd.detect_failure()
        assert failure is None

    def test_detect_repetition_exact(self):
        """Detect exact repetition of same command."""
        fd = FailureDetector(repetition_threshold=3)

        fd.record_action("cat file.txt", "content", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.REPETITION_LOOP
        assert failure.repeated_action == "cat file.txt"

    def test_repetition_threshold_respected(self):
        """No detection below threshold."""
        fd = FailureDetector(repetition_threshold=4)

        fd.record_action("cat file.txt", "content", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)

        failure = fd.detect_failure()
        assert failure is None  # Below threshold of 4

        fd.record_action("cat file.txt", "content", is_error=False)
        failure = fd.detect_failure()
        assert failure is not None  # Now at threshold

    def test_repetition_normalized(self):
        """Test that commands are normalized for comparison."""
        fd = FailureDetector(repetition_threshold=3)

        fd.record_action("cat   file.txt", "content", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)
        fd.record_action("  cat file.txt  ", "content", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.REPETITION_LOOP

    def test_olympics_granite_b01_pattern(self):
        """
        Reproduce granite3.1-moe:3b B01 failure pattern.
        Model repeated 'find -name' commands until turn 10.
        """
        fd = FailureDetector(repetition_threshold=3)

        # Simulate granite's B01 behavior
        commands = [
            "find . -name '*.py'",
            "find . -name '*.txt'",
            "find . -name '*.py'",  # Repeat starts
            "find . -name '*.py'",
            "find . -name '*.py'",
        ]

        for cmd in commands[:2]:
            fd.record_action(cmd, "files...", is_error=False)
            assert fd.detect_failure() is None

        for cmd in commands[2:]:
            fd.record_action(cmd, "files...", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.REPETITION_LOOP


class TestCircularPatternDetection:
    """Test CIRCULAR_PATTERN detection (strategic disorientation)."""

    def test_no_circular_pattern(self):
        """No failure when no cycle exists."""
        fd = FailureDetector()

        fd.record_action("ls", "files", is_error=False)
        fd.record_action("cat a.txt", "content a", is_error=False)
        fd.record_action("cat b.txt", "content b", is_error=False)
        fd.record_action("grep x .", "matches", is_error=False)

        failure = fd.detect_failure()
        assert failure is None

    def test_detect_ab_ab_cycle(self):
        """Detect A→B→A→B cycling pattern."""
        fd = FailureDetector()

        fd.record_action("ls", "files", is_error=False)
        fd.record_action("cat README.md", "readme", is_error=False)
        fd.record_action("ls", "files", is_error=False)
        fd.record_action("cat README.md", "readme", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.CIRCULAR_PATTERN
        assert failure.pattern_sequence is not None
        assert len(failure.pattern_sequence) >= 2

    def test_detect_abc_abc_cycle(self):
        """Detect A→B→C→A→B→C cycling pattern."""
        fd = FailureDetector()

        actions = ["ls", "cat file.txt", "grep pattern ."] * 2

        for action in actions:
            fd.record_action(action, "output", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.CIRCULAR_PATTERN

    def test_olympics_dolphin_pattern(self):
        """
        Reproduce dolphin-mistral failure pattern.
        Model cycled through ls → cat → ls → cat without progress.
        """
        fd = FailureDetector()

        # Simulate dolphin's cycling behavior
        for _ in range(3):
            fd.record_action("ls -la", "drwxr files...", is_error=False)
            fd.record_action("cat main.py", "import ...", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.CIRCULAR_PATTERN


class TestExplicitErrorDetection:
    """Test EXPLICIT_ERROR detection (agnosia / error recognition failure)."""

    def test_no_error_detection_on_success(self):
        """No failure when commands succeed."""
        fd = FailureDetector(error_threshold=3)

        fd.record_action("ls", "files", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)

        failure = fd.detect_failure()
        assert failure is None

    def test_detect_consecutive_errors(self):
        """Detect consecutive command failures."""
        fd = FailureDetector(error_threshold=3)

        fd.record_action("cat missing.txt", "No such file", is_error=True)
        fd.record_action("cat missing2.txt", "No such file", is_error=True)
        fd.record_action("cat missing3.txt", "No such file", is_error=True)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.EXPLICIT_ERROR

    def test_error_streak_broken_by_success(self):
        """Error streak is broken by successful command."""
        fd = FailureDetector(error_threshold=3)

        fd.record_action("cat missing.txt", "No such file", is_error=True)
        fd.record_action("cat missing2.txt", "No such file", is_error=True)
        fd.record_action("ls", "files", is_error=False)  # Success breaks streak
        fd.record_action("cat missing3.txt", "No such file", is_error=True)

        failure = fd.detect_failure()
        assert failure is None  # Streak was broken

    def test_olympics_phi3_cascade(self):
        """
        Reproduce phi3:latest failure cascade.
        Model had 26 failures, mostly EXPLICIT_ERROR from T3-T10.
        """
        fd = FailureDetector(error_threshold=3)

        # Simulate phi3's error cascade
        for i in range(8):
            fd.record_action(
                f"invalid_command_{i}",
                "Error: command not found",
                is_error=True
            )

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.EXPLICIT_ERROR


class TestProgressStallDetection:
    """Test PROGRESS_STALL detection (verbose confabulation)."""

    def test_no_stall_with_progress(self):
        """No failure when PCG grows."""
        fd = FailureDetector(stall_threshold=4)

        fd.record_action("ls", "files", is_error=False)
        fd.record_pcg_state(1)
        fd.record_action("cat a.txt", "content", is_error=False)
        fd.record_pcg_state(2)
        fd.record_action("cat b.txt", "content", is_error=False)
        fd.record_pcg_state(3)

        failure = fd.detect_failure()
        assert failure is None

    def test_detect_stall(self):
        """Detect when PCG stops growing."""
        fd = FailureDetector(stall_threshold=4)

        # Initial progress
        fd.record_action("ls", "files", is_error=False)
        fd.record_pcg_state(2)

        # Now stall - lots of actions but no PCG growth
        for i in range(5):
            fd.record_action(f"cat file{i}.txt", "content", is_error=False)
            fd.record_pcg_state(2)  # PCG stays at 2

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.PROGRESS_STALL
        assert failure.stall_turns >= 4

    def test_olympics_llama32_confabulation(self):
        """
        Reproduce llama3.2:3b 'verbose confabulation' pattern.
        Model had +4490 points but only 1/6 wins - lots of activity, no progress.
        """
        fd = FailureDetector(stall_threshold=4)

        # Simulate llama3.2:3b behavior: many actions, PCG stagnant
        fd.record_pcg_state(1)

        for i in range(10):
            fd.record_action(
                f"cat file{i}.txt",
                f"Content of file {i}...",
                is_error=False
            )
            fd.record_pcg_state(1)  # PCG never grows

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.failure_type == FailureType.PROGRESS_STALL


class TestExhaustionDetection:
    """Test EXHAUSTION detection."""

    def test_detect_exhaustion(self):
        """Detect when all cards tried without progress."""
        fd = FailureDetector()

        # Record card usage
        fd.record_card_usage("hint_1")
        fd.record_card_usage("hint_2")
        fd.record_card_usage("hint_3")

        # Set available cards (simulating limited deck)
        fd.set_available_cards({"hint_1", "hint_2", "hint_3"})

        # Many turns without progress after cards
        for i in range(5):
            fd.record_action(f"action_{i}", "output", is_error=False)
            fd.record_pcg_state(1)  # No growth

        failure = fd.detect_failure()
        # Should detect either EXHAUSTION or PROGRESS_STALL
        assert failure is not None


class TestTurnTracking:
    """Test turn counting and history."""

    def test_turn_increment(self):
        """Turn counter increments on each action."""
        fd = FailureDetector()

        assert fd.current_turn == 0

        fd.record_action("ls", "files", is_error=False)
        assert fd.current_turn == 1

        fd.record_action("cat file.txt", "content", is_error=False)
        assert fd.current_turn == 2

    def test_action_history(self):
        """Action history is maintained."""
        fd = FailureDetector()

        fd.record_action("ls", "files", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)

        assert len(fd.action_history) == 2
        assert fd.action_history[0] == "ls"
        assert fd.action_history[1] == "cat file.txt"


class TestFailureContext:
    """Test FailureContext data structure."""

    def test_context_contains_turn(self):
        """FailureContext includes turn number."""
        fd = FailureDetector(repetition_threshold=2)

        fd.record_action("ls", "files", is_error=False)
        fd.record_action("ls", "files", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.turn_detected == 2

    def test_context_contains_details(self):
        """FailureContext includes relevant details."""
        fd = FailureDetector(repetition_threshold=2)

        fd.record_action("cat file.txt", "content", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=False)

        failure = fd.detect_failure()
        assert failure is not None
        assert failure.repeated_action == "cat file.txt"


class TestFatLLMInstruction:
    """Test Fat-LLM instruction generation."""

    def test_instruction_for_repetition(self):
        """Generate instruction for repetition failure."""
        context = FailureContext(
            failure_type=FailureType.REPETITION_LOOP,
            turn_detected=5,
            repeated_action="cat file.txt"
        )

        instruction = get_fat_llm_instruction(context)

        assert instruction is not None
        assert "repeat" in instruction.lower() or "loop" in instruction.lower()
        assert "cat file.txt" in instruction

    def test_instruction_for_circular(self):
        """Generate instruction for circular pattern failure."""
        context = FailureContext(
            failure_type=FailureType.CIRCULAR_PATTERN,
            turn_detected=6,
            pattern_sequence=["ls", "cat", "ls", "cat"]
        )

        instruction = get_fat_llm_instruction(context)

        assert instruction is not None
        assert "cycle" in instruction.lower() or "circular" in instruction.lower()

    def test_instruction_for_error(self):
        """Generate instruction for explicit error failure."""
        context = FailureContext(
            failure_type=FailureType.EXPLICIT_ERROR,
            turn_detected=4,
            details={"error_count": 3}
        )

        instruction = get_fat_llm_instruction(context)

        assert instruction is not None
        assert "error" in instruction.lower()


class TestReset:
    """Test detector reset functionality."""

    def test_reset_clears_state(self):
        """Reset clears all accumulated state."""
        fd = FailureDetector()

        # Accumulate some state
        fd.record_action("ls", "files", is_error=False)
        fd.record_action("cat file.txt", "content", is_error=True)
        fd.record_pcg_state(5)

        # Reset
        fd.reset()

        assert fd.current_turn == 0
        assert len(fd.action_history) == 0
        assert len(fd.error_history) == 0
        assert len(fd.pcg_node_counts) == 0


class TestOlympicsValidation:
    """
    Validation tests based on LLM Olympics 2025-12-23 results.

    Key findings to validate:
    - deepseek-r1:14b: 0 failures, 6/6 wins
    - granite3.1-moe:3b: 4 failures, 3/6 wins
    - phi3:latest: 26 failures, 1/6 wins
    - Inverse correlation: failure_rate ↔ success_rate
    """

    def test_clean_run_pattern(self):
        """
        Simulate deepseek-r1:14b 'clean run' pattern.
        No repetitions, no cycles, no errors, steady progress.
        """
        fd = FailureDetector()

        # Methodical, non-repeating actions with steady progress
        actions = [
            ("ls -la", 1),
            ("cat requirements.txt", 2),
            ("grep -r 'pattern' .", 3),
            ("head -20 main.py", 4),
            ("python main.py", 5),
        ]

        for action, pcg_size in actions:
            fd.record_action(action, "success output", is_error=False)
            fd.record_pcg_state(pcg_size)
            failure = fd.detect_failure()
            assert failure is None, f"Unexpected failure at {action}"

    def test_high_failure_pattern(self):
        """
        Simulate phi3:latest high-failure pattern.
        26 failures detected, mostly EXPLICIT_ERROR.
        """
        fd = FailureDetector(error_threshold=3)

        failure_count = 0

        # Simulate error cascade
        for i in range(10):
            fd.record_action(
                f"broken_command_{i}",
                "Error: invalid syntax",
                is_error=True
            )
            fd.record_pcg_state(1)

            failure = fd.detect_failure()
            if failure:
                failure_count += 1

        # Should have detected failures
        assert failure_count > 0, "Should detect failures in error cascade"


# === Fixtures ===

@pytest.fixture
def fresh_detector():
    """Provide a fresh FailureDetector instance."""
    return FailureDetector()


@pytest.fixture
def strict_detector():
    """Provide a FailureDetector with strict thresholds."""
    return FailureDetector(
        repetition_threshold=2,
        stall_threshold=2,
        error_threshold=2
    )


# === Run Tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
