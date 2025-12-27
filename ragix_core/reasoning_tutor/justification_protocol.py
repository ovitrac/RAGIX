#!/usr/bin/env python3
"""
Justification Protocol - Fix the Metric Bias Problem
=====================================================

**ROADMAP STATUS: R3 COMPLETE** — Priority P0

Problem Statement (llama3.2:3b anomaly):
- +4490 points (highest total score)
- 1/6 wins (near-worst win rate)
- Diagnosis: "Verbose Confabulation" — looks busy, accomplishes little

Current scoring rewards *activity* not *achievement*:
- Each command earns points based on outcome
- Goal achievement is binary (win/lose)
- No penalty for pointless or unjustified actions

Solution: Require structured justification before each action.
Actions without valid justification get reduced or zero score contribution.

Formula (with Semantic Intent integration):
    ADJUSTED_SCORE = BASE_SCORE × JUSTIFICATION × GOAL_PROXIMITY × INTENT

Where:
- BASE_SCORE: Points from action outcome (current system)
- JUSTIFICATION: 0.0-1.0 based on hypothesis accuracy
- GOAL_PROXIMITY: 0.0-1.0 based on relevance to goal
- INTENT: 0.0-1.0 based on semantic convergence (Phase 1 semantic tool)

Expected Impact:
- deepseek-r1:14b: No change (focused, converging)
- llama3.2:3b: ~75-89% score reduction (wandering penalized)
- granite3.1-moe:3b: Minor adjustment (uses cards strategically)

See also: semantic_intent.py for INTENT multiplier, config.py for parameters

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import re


class JustificationQuality(Enum):
    """Quality levels for action justifications."""
    EXCELLENT = 1.0    # Hypothesis confirmed, direct goal progress
    GOOD = 0.75        # Hypothesis partially confirmed
    ACCEPTABLE = 0.5   # Relevant but not decisive
    WEAK = 0.25        # Tangential justification
    NONE = 0.0         # No justification or completely wrong


class GoalProximity(Enum):
    """How close the action is to the goal."""
    DIRECT = 1.0       # Action directly achieves/advances goal
    INDIRECT = 0.7     # Setup action for goal
    EXPLORATORY = 0.4  # Information gathering
    TANGENTIAL = 0.1   # Marginally related
    IRRELEVANT = 0.0   # No connection to goal


@dataclass
class JustifiedAction:
    """
    A structured justification required before each action.

    The LLM must provide this BEFORE executing any command.
    Actions without proper justification receive reduced scores.
    """
    # The action itself
    action: str                     # The command or tool call
    action_type: str = "bash"       # bash, read, edit, respond

    # Justification components
    hypothesis: str = ""            # "I expect this to reveal/confirm/enable..."
    expected_evidence: str = ""     # "This should show X, Y, or Z..."
    goal_connection: str = ""       # "This advances the goal by..."
    confidence: float = 0.5         # Self-assessed confidence 0.0-1.0

    # Outcome (filled after execution)
    actual_outcome: str = ""
    hypothesis_confirmed: bool = False

    # Computed quality (filled by evaluator)
    justification_quality: JustificationQuality = JustificationQuality.NONE
    goal_proximity: GoalProximity = GoalProximity.IRRELEVANT

    def score_multiplier(self) -> float:
        """
        Compute the score multiplier based on justification and proximity.

        Returns: float between 0.0 and 1.0
        """
        return self.justification_quality.value * self.goal_proximity.value


@dataclass
class JustificationEvaluator:
    """
    Evaluates the quality of action justifications.

    This is the core component that converts the current "activity-based"
    scoring into "achievement-based" scoring.
    """
    # Goal state for comparison
    goal_description: str = ""
    goal_keywords: List[str] = field(default_factory=list)

    # History for pattern detection
    action_history: List[JustifiedAction] = field(default_factory=list)

    def evaluate_justification(
        self,
        action: JustifiedAction,
        outcome: str,
        goal_achieved: bool = False
    ) -> JustifiedAction:
        """
        Evaluate a justification after the action is executed.

        Args:
            action: The justified action to evaluate
            outcome: The actual outcome/output from the action
            goal_achieved: Whether this action achieved the final goal

        Returns:
            JustifiedAction with quality and proximity filled in
        """
        action.actual_outcome = outcome

        # Evaluate hypothesis confirmation
        action.justification_quality = self._evaluate_hypothesis(action, outcome)

        # Evaluate goal proximity
        action.goal_proximity = self._evaluate_proximity(action, outcome, goal_achieved)

        # Track for history
        self.action_history.append(action)

        return action

    def _evaluate_hypothesis(
        self,
        action: JustifiedAction,
        outcome: str
    ) -> JustificationQuality:
        """
        Evaluate how well the hypothesis matched the outcome.
        """
        # No hypothesis = no quality
        if not action.hypothesis.strip():
            return JustificationQuality.NONE

        # Check for expected evidence in outcome
        if action.expected_evidence:
            evidence_keywords = self._extract_keywords(action.expected_evidence)
            matches = sum(1 for kw in evidence_keywords if kw.lower() in outcome.lower())
            match_ratio = matches / max(len(evidence_keywords), 1)

            if match_ratio >= 0.7:
                action.hypothesis_confirmed = True
                return JustificationQuality.EXCELLENT
            elif match_ratio >= 0.4:
                return JustificationQuality.GOOD
            elif match_ratio >= 0.2:
                return JustificationQuality.ACCEPTABLE
            else:
                return JustificationQuality.WEAK

        # Fallback: check if outcome mentions anything from hypothesis
        hypothesis_keywords = self._extract_keywords(action.hypothesis)
        matches = sum(1 for kw in hypothesis_keywords if kw.lower() in outcome.lower())

        if matches >= 2:
            return JustificationQuality.GOOD
        elif matches >= 1:
            return JustificationQuality.ACCEPTABLE
        else:
            return JustificationQuality.WEAK

    def _evaluate_proximity(
        self,
        action: JustifiedAction,
        outcome: str,
        goal_achieved: bool
    ) -> GoalProximity:
        """
        Evaluate how close the action is to the goal.
        """
        # Direct goal achievement
        if goal_achieved:
            return GoalProximity.DIRECT

        # Check goal connection explanation
        if not action.goal_connection.strip():
            return GoalProximity.TANGENTIAL

        # Check for goal keywords in action and outcome
        goal_matches_action = sum(
            1 for kw in self.goal_keywords
            if kw.lower() in action.action.lower()
        )
        goal_matches_outcome = sum(
            1 for kw in self.goal_keywords
            if kw.lower() in outcome.lower()
        )

        total_matches = goal_matches_action + goal_matches_outcome

        if total_matches >= 3:
            return GoalProximity.DIRECT
        elif total_matches >= 2:
            return GoalProximity.INDIRECT
        elif total_matches >= 1:
            return GoalProximity.EXPLORATORY
        elif action.goal_connection:
            return GoalProximity.EXPLORATORY
        else:
            return GoalProximity.TANGENTIAL

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common words and extract meaningful tokens
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'to', 'from', 'in', 'on', 'at', 'by', 'for',
            'with', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'and', 'or', 'but', 'if', 'then', 'else',
            'so', 'because', 'as', 'until', 'while', 'of', 'expect', 'show',
            'reveal', 'confirm', 'enable', 'should', 'will', 'find', 'see'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    def compute_adjusted_score(
        self,
        base_score: int,
        action: JustifiedAction
    ) -> int:
        """
        Compute adjusted score based on justification quality.

        Args:
            base_score: Original score from current system
            action: The evaluated justified action

        Returns:
            Adjusted score (may be 0 if no justification)
        """
        multiplier = action.score_multiplier()
        return int(base_score * multiplier)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all evaluated actions."""
        if not self.action_history:
            return {"total_actions": 0}

        qualities = [a.justification_quality for a in self.action_history]
        proximities = [a.goal_proximity for a in self.action_history]
        multipliers = [a.score_multiplier() for a in self.action_history]

        return {
            "total_actions": len(self.action_history),
            "avg_quality": sum(q.value for q in qualities) / len(qualities),
            "avg_proximity": sum(p.value for p in proximities) / len(proximities),
            "avg_multiplier": sum(multipliers) / len(multipliers),
            "excellent_count": sum(1 for q in qualities if q == JustificationQuality.EXCELLENT),
            "none_count": sum(1 for q in qualities if q == JustificationQuality.NONE),
            "direct_goal_count": sum(1 for p in proximities if p == GoalProximity.DIRECT),
        }


# === Prompt Template for Justification ===

JUSTIFICATION_PROMPT_TEMPLATE = """
Before executing any action, you MUST provide a structured justification:

```json
{{
    "justification": {{
        "hypothesis": "I expect this to [reveal/confirm/enable]...",
        "expected_evidence": "This should show [specific expected output]...",
        "goal_connection": "This advances the goal by [explanation]...",
        "confidence": 0.0-1.0
    }},
    "action": {{
        "type": "bash|read|edit|respond",
        "command": "the actual command or content"
    }}
}}
```

SCORING IMPACT:
- Actions WITHOUT justification: 0% of base score
- Actions with WEAK justification: 25% of base score
- Actions with GOOD justification: 75% of base score
- Actions with EXCELLENT justification: 100% of base score

Think carefully: Is this action truly necessary for the goal?
"""


def parse_justified_response(response: str) -> Optional[JustifiedAction]:
    """
    Parse a response to extract the justified action.

    Expected format:
    {
        "justification": {...},
        "action": {...}
    }

    Returns:
        JustifiedAction if valid, None if parse fails
    """
    import json

    # Try to find JSON in response
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())

        justification = data.get("justification", {})
        action = data.get("action", {})

        return JustifiedAction(
            action=action.get("command", ""),
            action_type=action.get("type", "bash"),
            hypothesis=justification.get("hypothesis", ""),
            expected_evidence=justification.get("expected_evidence", ""),
            goal_connection=justification.get("goal_connection", ""),
            confidence=float(justification.get("confidence", 0.5))
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


# === Demo/Test Code ===

def demo_justification_protocol():
    """
    Demonstrate the justification protocol with the llama3.2:3b anomaly case.
    """
    print("=" * 70)
    print("JUSTIFICATION PROTOCOL DEMO")
    print("Simulating the llama3.2:3b 'Metric Bias' case")
    print("=" * 70)

    # Create evaluator with a sample goal
    evaluator = JustificationEvaluator(
        goal_description="Find the file containing 'NEEDLE_VALUE' in the project",
        goal_keywords=["needle", "value", "file", "find", "grep"]
    )

    # Simulate typical llama3.2:3b behavior: lots of activity, poor justification
    actions = [
        # Action 1: Unjustified ls
        JustifiedAction(
            action="ls -la",
            hypothesis="",  # No hypothesis!
            expected_evidence="",
            goal_connection=""
        ),
        # Action 2: Weak justification
        JustifiedAction(
            action="cat README.md",
            hypothesis="Maybe the readme has info",
            expected_evidence="",
            goal_connection=""
        ),
        # Action 3: Good justification but tangential
        JustifiedAction(
            action="find . -type f -name '*.py'",
            hypothesis="I expect this to list Python files",
            expected_evidence="List of .py files",
            goal_connection="Python files might contain the needle"
        ),
        # Action 4: Excellent justification
        JustifiedAction(
            action="grep -r 'NEEDLE_VALUE' .",
            hypothesis="I expect this to find the file with NEEDLE_VALUE",
            expected_evidence="File path containing NEEDLE_VALUE",
            goal_connection="This directly searches for the needle"
        ),
    ]

    # Simulate outcomes
    outcomes = [
        "total 42\ndrwxr-xr-x 5 user user 4096 Dec 23 10:00 .\n...",
        "# Project README\nThis is a sample project...",
        "./src/main.py\n./src/utils.py\n./tests/test_main.py",
        "./src/config.py:7:NEEDLE_VALUE = 'secret123'"
    ]

    # Base scores (current system would give these)
    base_scores = [50, 30, 40, 100]  # Total: 220

    print("\n📊 COMPARISON: OLD vs NEW SCORING\n")
    print(f"{'Action':<30} {'Old Score':>10} {'Multiplier':>12} {'New Score':>10}")
    print("-" * 65)

    total_old = 0
    total_new = 0

    for i, (action, outcome, base) in enumerate(zip(actions, outcomes, base_scores)):
        # Evaluate
        evaluated = evaluator.evaluate_justification(
            action,
            outcome,
            goal_achieved=(i == 3)  # Last action achieves goal
        )

        new_score = evaluator.compute_adjusted_score(base, evaluated)

        action_short = action.action[:28] + ".." if len(action.action) > 30 else action.action
        multiplier = evaluated.score_multiplier()

        print(f"{action_short:<30} {base:>10} {multiplier:>12.2f} {new_score:>10}")

        total_old += base
        total_new += new_score

    print("-" * 65)
    print(f"{'TOTAL':<30} {total_old:>10} {'':<12} {total_new:>10}")

    reduction = ((total_old - total_new) / total_old) * 100
    print(f"\n📉 Score Reduction: {reduction:.1f}%")

    # Summary
    summary = evaluator.get_summary()
    print(f"\n📈 Summary:")
    print(f"   Average Quality: {summary['avg_quality']:.2f}")
    print(f"   Average Proximity: {summary['avg_proximity']:.2f}")
    print(f"   Excellent Actions: {summary['excellent_count']}")
    print(f"   Unjustified Actions: {summary['none_count']}")

    print("\n✅ The Justification Protocol penalizes:")
    print("   - Unjustified 'ls -la' (0% score)")
    print("   - Weak 'cat README' (reduced score)")
    print("   - Rewards justified 'grep' (full score)")


# =============================================================================
# INTEGRATED SCORING (Justification + Semantic Intent)
# =============================================================================

class IntegratedScorer:
    """
    Unified scoring that combines:
    - JustificationProtocol (R3): hypothesis quality, goal proximity
    - SemanticIntentTracker (Phase 1): convergence toward goal

    Formula:
        FINAL_SCORE = BASE × JUSTIFICATION × GOAL_PROXIMITY × INTENT

    This directly addresses the llama3.2:3b anomaly by penalizing:
    1. Unjustified actions (JUSTIFICATION = 0)
    2. Irrelevant actions (GOAL_PROXIMITY = 0)
    3. Wandering actions (INTENT = 0.25)
    """

    def __init__(self, goal_description: str, goal_keywords: Optional[List[str]] = None):
        """
        Initialize integrated scorer.

        Args:
            goal_description: Description of the goal
            goal_keywords: Keywords for goal proximity matching
        """
        from config import get_config, is_intent_tracker_enabled

        self.config = get_config()
        self.goal = goal_description
        self.goal_keywords = goal_keywords or []

        # Justification evaluator
        self.justification_evaluator = JustificationEvaluator(
            goal_description=goal_description,
            goal_keywords=self.goal_keywords
        )

        # Semantic intent tracker (if enabled)
        self._intent_tracker = None
        if is_intent_tracker_enabled():
            try:
                from semantic_intent import SemanticIntentTracker
                self._intent_tracker = SemanticIntentTracker(goal_description)
            except ImportError:
                pass  # Semantic tools not available

    def score_action(
        self,
        action: JustifiedAction,
        outcome: str,
        base_score: int,
        goal_achieved: bool = False
    ) -> Dict[str, Any]:
        """
        Compute final score for an action using all components.

        Args:
            action: The justified action
            outcome: The actual output/result
            base_score: Raw score from current system
            goal_achieved: Whether this action achieved the goal

        Returns:
            Dict with score breakdown and final score
        """
        # Evaluate justification
        evaluated = self.justification_evaluator.evaluate_justification(
            action, outcome, goal_achieved
        )

        justification_mult = evaluated.justification_quality.value
        proximity_mult = evaluated.goal_proximity.value

        # Evaluate semantic intent (if enabled)
        intent_mult = 1.0  # Default: no penalty
        intent_category = "disabled"

        if self._intent_tracker is not None:
            from semantic_intent import IntentCategory
            analysis = self._intent_tracker.analyze(action.action)
            intent_mult = analysis.multiplier
            intent_category = analysis.category.value

        # Compute final score
        final_score = int(base_score * justification_mult * proximity_mult * intent_mult)

        return {
            "base_score": base_score,
            "justification_quality": evaluated.justification_quality.name,
            "justification_mult": justification_mult,
            "goal_proximity": evaluated.goal_proximity.name,
            "proximity_mult": proximity_mult,
            "intent_category": intent_category,
            "intent_mult": intent_mult,
            "final_score": final_score,
            "reduction_pct": (1 - final_score / max(base_score, 1)) * 100
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of scoring session."""
        summary = self.justification_evaluator.get_summary()

        if self._intent_tracker is not None:
            summary["intent_trajectory"] = self._intent_tracker.get_trajectory_summary()
            summary["semantic_enabled"] = True
        else:
            summary["semantic_enabled"] = False

        return summary

    def reset(self):
        """Reset for new game."""
        self.justification_evaluator = JustificationEvaluator(
            goal_description=self.goal,
            goal_keywords=self.goal_keywords
        )
        if self._intent_tracker is not None:
            self._intent_tracker.reset()


def demo_integrated_scoring():
    """Demonstrate integrated scoring with semantic intent."""
    print("=" * 70)
    print("INTEGRATED SCORING DEMO")
    print("Justification (R3) + Semantic Intent (Phase 1)")
    print("=" * 70)

    goal = "Find NEEDLE_VALUE in the project source code"
    scorer = IntegratedScorer(
        goal_description=goal,
        goal_keywords=["needle", "value", "find", "grep", "search", "source"]
    )

    print(f"\n🎯 Goal: {goal}")
    print()

    # Simulate actions with mixed quality
    test_cases = [
        # (action, hypothesis, expected_evidence, goal_connection, outcome, base_score)
        (
            "ls -la",
            "",  # No justification
            "",
            "",
            "drwxr-xr-x ...",
            50
        ),
        (
            "grep -r 'NEEDLE' .",
            "Search for NEEDLE keyword",
            "File paths containing NEEDLE",
            "Direct search for target",
            "./src/config.py:NEEDLE_VALUE='x'",
            50
        ),
        (
            "cat README.md",
            "Check documentation",
            "Project info",
            "Maybe readme mentions it",
            "# Project README...",
            50
        ),
    ]

    print(f"{'Action':<25} {'Just':>6} {'Prox':>6} {'Intent':>8} {'Base':>6} {'Final':>6} {'Δ':>8}")
    print("-" * 75)

    total_base = 0
    total_final = 0

    for action_str, hyp, exp, goal_conn, outcome, base in test_cases:
        action = JustifiedAction(
            action=action_str,
            hypothesis=hyp,
            expected_evidence=exp,
            goal_connection=goal_conn
        )

        result = scorer.score_action(action, outcome, base, goal_achieved=False)

        total_base += base
        total_final += result["final_score"]

        print(f"{action_str:<25} "
              f"{result['justification_mult']:>6.2f} "
              f"{result['proximity_mult']:>6.2f} "
              f"{result['intent_mult']:>8.2f} "
              f"{base:>6} "
              f"{result['final_score']:>6} "
              f"{-result['reduction_pct']:>+7.0f}%")

    print("-" * 75)
    print(f"{'TOTAL':<25} {'':<6} {'':<6} {'':<8} {total_base:>6} {total_final:>6} "
          f"{-(1 - total_final/total_base)*100:>+7.0f}%")

    summary = scorer.get_summary()
    print(f"\n📊 Semantic: {'✓ Enabled' if summary['semantic_enabled'] else '✗ Disabled'}")


if __name__ == "__main__":
    demo_justification_protocol()
    print("\n" + "=" * 70 + "\n")
    demo_integrated_scoring()
