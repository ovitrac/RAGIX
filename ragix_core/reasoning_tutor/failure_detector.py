#!/usr/bin/env python3
"""
Failure Detector - Meta-Cognitive Detection of Stuck States
============================================================

**ROADMAP STATUS: R1 COMPLETE** — Validated in LLM Olympics 2025-12-23

Detects when slim LLMs are stuck, enabling targeted interventions.
This transforms the system from an execution loop (watching bash return codes)
into a meta-cognitive system (watching game dynamics).

Key Detection Types:
- EXPLICIT_ERROR: Command fails repeatedly
- REPETITION_LOOP: Same action repeated (syntactic aphasia / perseveration)
- CIRCULAR_PATTERN: A→B→A→B cycling (strategic disorientation)
- PROGRESS_STALL: No new PCG nodes (verbose confabulation)
- EXHAUSTION: All cards tried without progress

Olympic Validation Results (2025-12-23):
- 11 models × 6 benchmarks = 66 games
- Strong inverse correlation: failure_rate ↔ success_rate
- Zero failures → 100% wins (deepseek-r1:14b)
- High failures → Low wins (phi3: 26 failures, 1/6 wins)
- Conclusion: FailureDetector is a VALID quality signal

Clinical Diagnoses:
- ⟳ Repetition Loop = Perseveration (Syntactic Aphasia)
- ↻ Circular Pattern = Disorientation (Strategic Confusion)
- ⚠ Explicit Error = Agnosia (Error Recognition Failure)

See: ROADMAP_METACOGNITION.md, OLYMPICS_2025-12-23.md, README_GAME_NOTATION.md

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from collections import Counter

# Import config for default thresholds
from config import get_config, is_failure_detection_enabled


class FailureType(Enum):
    """Types of failure patterns detected."""
    EXPLICIT_ERROR = "explicit_error"       # Command returns error 3+ times
    REPETITION_LOOP = "repetition_loop"     # cat A, cat A, cat A (syntactic stupidity)
    CIRCULAR_PATTERN = "circular_pattern"   # ls → cat → ls → cat (strategic disorientation)
    PROGRESS_STALL = "progress_stall"       # No new PCG nodes (verbose but unproductive)
    EXHAUSTION = "exhaustion"               # All cards tried, no progress


@dataclass
class FailureContext:
    """Context information about the detected failure."""
    failure_type: FailureType
    turn_detected: int
    details: Dict[str, Any] = field(default_factory=dict)

    # Specific context for different failure types
    repeated_action: Optional[str] = None       # For REPETITION_LOOP
    pattern_sequence: Optional[List[str]] = None  # For CIRCULAR_PATTERN
    stall_turns: int = 0                        # For PROGRESS_STALL
    cards_tried: Optional[Set[str]] = None      # For EXHAUSTION


class FailureDetector:
    """
    Detect when to trigger card generation - not just on explicit errors.

    This class monitors game dynamics to detect stuck states that require
    intervention (e.g., Fat LLM card generation).

    Usage:
        detector = FailureDetector()

        for turn in game:
            detector.record_action(action, output, is_error)
            detector.record_pcg_state(pcg)

            failure = detector.detect_failure()
            if failure:
                # Trigger intervention
                instruction = get_fat_llm_instruction(failure)
    """

    def __init__(
        self,
        repetition_threshold: Optional[int] = None,
        stall_threshold: Optional[int] = None,
        error_threshold: Optional[int] = None
    ):
        """
        Initialize the failure detector.

        Args:
            repetition_threshold: Same action N times triggers REPETITION_LOOP
                                 (default: from config)
            stall_threshold: No PCG progress for N turns triggers PROGRESS_STALL
                            (default: from config)
            error_threshold: Explicit errors N times triggers EXPLICIT_ERROR
                            (default: from config)
        """
        # Load defaults from config
        config = get_config().failure_detector

        self.repetition_threshold = repetition_threshold if repetition_threshold is not None else config.repetition_threshold
        self.stall_threshold = stall_threshold if stall_threshold is not None else config.stall_threshold
        self.error_threshold = error_threshold if error_threshold is not None else config.error_threshold

        # Store config reference for enabled check
        self._config = config

        # Action history
        self.action_history: List[str] = []
        self.output_history: List[str] = []
        self.error_history: List[bool] = []

        # PCG tracking
        self.pcg_node_counts: List[int] = []

        # Card tracking
        self.cards_used: Set[str] = set()
        self.available_cards: Set[str] = set()

        # Current turn
        self.current_turn: int = 0

    def record_action(self, action: str, output: str, is_error: bool):
        """Record an action and its result."""
        # Normalize action for comparison
        normalized = self._normalize_action(action)

        self.action_history.append(normalized)
        self.output_history.append(output)
        self.error_history.append(is_error)
        self.current_turn += 1

    def record_pcg_state(self, pcg_node_count: int):
        """
        Record PCG state at end of turn.

        Args:
            pcg_node_count: Number of nodes in PCG (len(pcg.nodes))
        """
        self.pcg_node_counts.append(pcg_node_count)

    def record_card_usage(self, card_name: str):
        """Record that a card was used."""
        self.cards_used.add(card_name)

    def set_available_cards(self, cards: Set[str]):
        """Set the available card deck."""
        self.available_cards = cards

    def detect_failure(self) -> Optional[FailureContext]:
        """
        Detect if the model is in a failure state.

        Returns:
            FailureContext if failure detected, None otherwise.
            Returns None if failure detection is disabled in config.

        Detection priority (first match wins):
        1. EXPLICIT_ERROR - Command fails repeatedly
        2. REPETITION_LOOP - Same action repeated
        3. CIRCULAR_PATTERN - Cycling between actions
        4. PROGRESS_STALL - No new PCG nodes
        5. EXHAUSTION - All cards tried
        """
        # Check if failure detection is enabled
        if not is_failure_detection_enabled():
            return None

        # 1. Check for explicit errors
        if self._detect_explicit_error():
            return FailureContext(
                failure_type=FailureType.EXPLICIT_ERROR,
                turn_detected=self.current_turn,
                repeated_action=self.action_history[-1] if self.action_history else None,
                details={"error_count": self._count_recent_errors()}
            )

        # 2. Check for repetition loop
        repeated = self._detect_repetition_loop()
        if repeated:
            return FailureContext(
                failure_type=FailureType.REPETITION_LOOP,
                turn_detected=self.current_turn,
                repeated_action=repeated,
                details={"repeat_count": self._count_repetitions(repeated)}
            )

        # 3. Check for circular pattern
        pattern = self._detect_circular_pattern()
        if pattern:
            return FailureContext(
                failure_type=FailureType.CIRCULAR_PATTERN,
                turn_detected=self.current_turn,
                pattern_sequence=pattern,
                details={"pattern_length": len(pattern)}
            )

        # 4. Check for progress stall
        if self._detect_progress_stall():
            return FailureContext(
                failure_type=FailureType.PROGRESS_STALL,
                turn_detected=self.current_turn,
                stall_turns=self.stall_threshold,
                details={"pcg_nodes": self.pcg_node_counts[-1] if self.pcg_node_counts else 0}
            )

        # 5. Check for exhaustion
        if self._detect_exhaustion():
            return FailureContext(
                failure_type=FailureType.EXHAUSTION,
                turn_detected=self.current_turn,
                cards_tried=self.cards_used.copy(),
                details={"cards_remaining": len(self.available_cards - self.cards_used)}
            )

        return None

    def _normalize_action(self, action: str) -> str:
        """Normalize action string for comparison."""
        # Remove whitespace variations
        normalized = " ".join(action.strip().split())
        # Lowercase for comparison
        return normalized.lower()

    def _detect_explicit_error(self) -> bool:
        """Check if last N actions all resulted in errors."""
        if len(self.error_history) < self.error_threshold:
            return False

        recent_errors = self.error_history[-self.error_threshold:]
        return all(recent_errors)

    def _count_recent_errors(self) -> int:
        """Count consecutive recent errors."""
        count = 0
        for is_error in reversed(self.error_history):
            if is_error:
                count += 1
            else:
                break
        return count

    def _detect_repetition_loop(self) -> Optional[str]:
        """
        Detect if the same action is repeated N+ times.

        Returns:
            The repeated action string, or None if no repetition detected.

        DIAGNOSIS: Syntactic stupidity or hallucination.
        The model thinks the command didn't work.
        """
        if len(self.action_history) < self.repetition_threshold:
            return None

        recent = self.action_history[-self.repetition_threshold:]

        # Check if all recent actions are the same
        if len(set(recent)) == 1:
            return recent[0]

        return None

    def _count_repetitions(self, action: str) -> int:
        """Count how many times an action was repeated consecutively."""
        count = 0
        for a in reversed(self.action_history):
            if a == action:
                count += 1
            else:
                break
        return count

    def _detect_circular_pattern(self) -> Optional[List[str]]:
        """
        Detect A→B→A→B or A→B→C→A→B→C patterns.

        Returns:
            The repeating pattern sequence, or None if no pattern detected.

        DIAGNOSIS: Strategic disorientation.
        The model doesn't know what to do next, searching randomly.
        """
        if len(self.action_history) < 4:
            return None

        # Check for period-2 cycle (A→B→A→B)
        recent_4 = self.action_history[-4:]
        if recent_4[0] == recent_4[2] and recent_4[1] == recent_4[3]:
            return [recent_4[0], recent_4[1]]

        # Check for period-3 cycle (A→B→C→A→B→C)
        if len(self.action_history) >= 6:
            recent_6 = self.action_history[-6:]
            if recent_6[0:3] == recent_6[3:6]:
                return recent_6[0:3]

        return None

    def _detect_progress_stall(self) -> bool:
        """
        No new PCG nodes added in N turns.

        REFINED: "Empty outputs" is too weak a signal. A model can "babble"
        or do useless ls commands that produce text but no actual progress.

        STRICT DEFINITION: No new Truth (T), Observation (Obs), or Entity (Ent)
        nodes added to the graph in N turns.

        WHY: If the model spins in circles, it generates no new proofs.
        This is the most reliable progress metric - agnostic to output verbosity.
        """
        if len(self.pcg_node_counts) < self.stall_threshold:
            return False

        # Compare PCG node count at turn T vs turn T-N
        current_nodes = self.pcg_node_counts[-1]
        nodes_n_turns_ago = self.pcg_node_counts[-self.stall_threshold]

        # No new nodes = no progress (even if model produces verbose output)
        return current_nodes == nodes_n_turns_ago

    def _detect_exhaustion(self) -> bool:
        """
        All cards tried but still no progress toward goal.

        This indicates the card deck is insufficient for the task.
        """
        if not self.available_cards:
            return False

        # All cards used
        if not self.cards_used >= self.available_cards:
            return False

        # But still no recent progress (combine with stall check)
        if len(self.pcg_node_counts) >= 2:
            return self.pcg_node_counts[-1] == self.pcg_node_counts[-2]

        return False

    def get_diagnosis(self, failure: FailureContext) -> str:
        """Get human-readable diagnosis for a failure."""
        diagnoses = {
            FailureType.EXPLICIT_ERROR:
                f"Command '{failure.repeated_action}' failing repeatedly ({failure.details.get('error_count', 0)} times)",
            FailureType.REPETITION_LOOP:
                f"Repeating same action '{failure.repeated_action}' {failure.details.get('repeat_count', 0)} times (syntactic stupidity)",
            FailureType.CIRCULAR_PATTERN:
                f"Cycling between actions {failure.pattern_sequence} (strategic disorientation)",
            FailureType.PROGRESS_STALL:
                f"No new evidence added to proof graph in {failure.stall_turns} turns (verbose but unproductive)",
            FailureType.EXHAUSTION:
                f"All {len(failure.cards_tried or [])} cards tried without achieving goal",
        }
        return diagnoses.get(failure.failure_type, "Unknown failure")

    def reset(self):
        """Reset the detector for a new game."""
        self.action_history.clear()
        self.output_history.clear()
        self.error_history.clear()
        self.pcg_node_counts.clear()
        self.cards_used.clear()
        self.current_turn = 0


def get_fat_llm_instruction(failure: FailureContext, goal: str = "") -> str:
    """
    Generate appropriate Fat LLM prompt based on failure type.

    Different failure modes require different interventions:
    - REPETITION_LOOP → Alternative syntax card
    - CIRCULAR_PATTERN → Strategic/tactical card
    - PROGRESS_STALL → Synthesis card

    Args:
        failure: The detected failure context
        goal: The current game goal (for context)

    Returns:
        Instruction prompt for the Fat LLM to generate a helpful card.
    """

    if failure.failure_type == FailureType.REPETITION_LOOP:
        # DIAGNOSIS: Syntactic stupidity or hallucination
        # The model thinks the command didn't work
        # ACTION: Propose alternative syntax
        return f"""CARD GENERATION REQUEST: Alternative Syntax

The player is repeating the same command: `{failure.repeated_action}`
Repeated {failure.details.get('repeat_count', 0)} times without success.

DIAGNOSIS: Syntactic confusion or belief that the command failed.

TASK: Propose an ALTERNATIVE SYNTAX card that achieves the same goal differently.

Examples:
- If repeating 'cat file.txt' → propose 'head -50 file.txt' or grep-based extraction
- If repeating 'find -name X' → propose 'grep -r X .' or 'locate X'
- If repeating 'ls dir/' → propose 'tree dir/' or 'find dir/ -type f'

Goal context: {goal}

Generate a YAML card definition with:
- id: ALTERNATIVE_<original_command_type>
- name: Clear descriptive name
- description: What it does differently
- template: The bash command template
"""

    elif failure.failure_type == FailureType.CIRCULAR_PATTERN:
        # DIAGNOSIS: Strategic disorientation (searching randomly)
        # The model doesn't know what to do next
        # ACTION: Propose a strategic/tactical card
        pattern_str = " → ".join(failure.pattern_sequence or [])
        return f"""CARD GENERATION REQUEST: Strategic Direction

The player is circling between actions: {pattern_str}
This pattern has repeated, indicating strategic disorientation.

DIAGNOSIS: The model doesn't know how to proceed toward the goal.

TASK: Propose a STRATEGIC CARD that provides clear direction, not just syntax.

Good strategic cards:
- SCAN_DIRECTORY_RECURSIVE(path) - systematic exploration
- FIND_ENTRY_POINT(project) - locate main/index files
- TRACE_DEPENDENCY_CHAIN(file) - follow imports/includes
- SUMMARIZE_PROJECT_STRUCTURE() - bird's eye view
- SEARCH_FOR_GOAL_KEYWORDS(goal) - targeted search based on goal

Goal context: {goal}

Generate a YAML card definition that BREAKS THE CYCLE by providing a clear next step.
The card should guide the model toward the goal, not just offer another command variant.
"""

    elif failure.failure_type == FailureType.PROGRESS_STALL:
        # DIAGNOSIS: Verbose but unproductive (babbling)
        # The model produces output but no new proofs
        # ACTION: Propose a synthesis card
        return f"""CARD GENERATION REQUEST: Synthesis/Consolidation

The player has not added new evidence to the proof graph in {failure.stall_turns} turns.
They are producing output but not making structural progress.

DIAGNOSIS: Reading/exploring but not learning or synthesizing.

TASK: Propose a SYNTHESIS CARD that helps consolidate findings.

Good synthesis cards:
- SUMMARIZE_FINDINGS() - compile evidence collected so far
- IDENTIFY_GAPS() - what's missing to achieve the goal?
- FORMULATE_HYPOTHESIS() - explicit next step based on evidence
- EXTRACT_KEY_FACTS(files_read) - distill important information
- CHECK_GOAL_PROGRESS(goal, evidence) - how close are we?

Goal context: {goal}
PCG nodes: {failure.details.get('pcg_nodes', 0)}

Generate a YAML card that helps the model CONSOLIDATE what it has learned
and identify the NEXT MEANINGFUL STEP toward the goal.
"""

    elif failure.failure_type == FailureType.EXPLICIT_ERROR:
        return f"""CARD GENERATION REQUEST: Error Recovery

The command `{failure.repeated_action}` is failing repeatedly.
Error count: {failure.details.get('error_count', 0)}

DIAGNOSIS: Syntax error, missing file, or permission issue.

TASK: Propose an ERROR RECOVERY card that:
1. Diagnoses the error type
2. Offers a corrected or alternative approach

Goal context: {goal}

Generate a YAML card for robust error handling.
"""

    elif failure.failure_type == FailureType.EXHAUSTION:
        cards_tried = failure.cards_tried or set()
        return f"""CARD GENERATION REQUEST: Deck Extension

All available cards have been tried: {cards_tried}
The current deck is insufficient for this task.

DIAGNOSIS: The task requires capabilities not covered by existing cards.

TASK: Propose a NEW CAPABILITY card that addresses the gap.

Analyze the goal and cards tried to identify what's missing:
- Is it a search capability?
- Is it a transformation capability?
- Is it a verification capability?

Goal context: {goal}

Generate a YAML card that EXTENDS the deck with the missing capability.
"""

    return f"Unknown failure type: {failure.failure_type}"


# Convenience function for quick detection
def detect_stuck_state(
    action_history: List[str],
    pcg_node_counts: List[int],
    error_history: List[bool] = None
) -> Optional[FailureContext]:
    """
    Quick detection of stuck states from action history.

    Args:
        action_history: List of actions taken
        pcg_node_counts: List of PCG node counts per turn
        error_history: List of error flags per action (optional)

    Returns:
        FailureContext if stuck, None otherwise.
    """
    detector = FailureDetector()

    for i, action in enumerate(action_history):
        is_error = error_history[i] if error_history and i < len(error_history) else False
        detector.record_action(action, "", is_error)

        if i < len(pcg_node_counts):
            detector.record_pcg_state(pcg_node_counts[i])

    return detector.detect_failure()


if __name__ == "__main__":
    # Demo: Test the failure detector
    print("=" * 60)
    print("FailureDetector Demo")
    print("=" * 60)

    # Test 1: Repetition Loop
    print("\n--- Test 1: Repetition Loop ---")
    detector = FailureDetector(repetition_threshold=3)
    for i in range(4):
        detector.record_action("cat data/file.txt", "file contents...", False)
        detector.record_pcg_state(5 + i)  # PCG growing (not stalled)

    failure = detector.detect_failure()
    if failure:
        print(f"Detected: {failure.failure_type.value}")
        print(f"Diagnosis: {detector.get_diagnosis(failure)}")

    # Test 2: Circular Pattern
    print("\n--- Test 2: Circular Pattern ---")
    detector = FailureDetector()
    actions = ["ls data/", "cat data/file1.txt", "ls data/", "cat data/file2.txt"]
    for i, action in enumerate(actions):
        detector.record_action(action, "output...", False)
        detector.record_pcg_state(5)  # PCG not growing

    failure = detector.detect_failure()
    if failure:
        print(f"Detected: {failure.failure_type.value}")
        print(f"Diagnosis: {detector.get_diagnosis(failure)}")
        print(f"\nFat LLM Instruction:\n{get_fat_llm_instruction(failure, 'Find the secret code')[:500]}...")

    # Test 3: Progress Stall
    print("\n--- Test 3: Progress Stall ---")
    detector = FailureDetector(stall_threshold=3)
    for i in range(5):
        detector.record_action(f"grep pattern file{i}.txt", "some output", False)
        detector.record_pcg_state(10)  # PCG stuck at 10 nodes

    failure = detector.detect_failure()
    if failure:
        print(f"Detected: {failure.failure_type.value}")
        print(f"Diagnosis: {detector.get_diagnosis(failure)}")

    print("\n" + "=" * 60)
    print("Demo complete")
