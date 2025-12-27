#!/usr/bin/env python3
"""
Test Suite for JustificationProtocol (R3)
==========================================

Tests the Metric Bias fix:
- JustifiedAction structure
- JustificationQuality evaluation
- GoalProximity evaluation
- Score multiplier calculation
- Response parsing

Key validation: llama3.2:3b anomaly fix
- Old: +4490 points, 1/6 wins
- New: ~75% score reduction for unjustified actions

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from justification_protocol import (
    JustifiedAction,
    JustificationQuality,
    GoalProximity,
    JustificationEvaluator,
    parse_justified_response,
    JUSTIFICATION_PROMPT_TEMPLATE
)


class TestJustifiedAction:
    """Test JustifiedAction dataclass."""

    def test_default_values(self):
        """Test default field values."""
        action = JustifiedAction(action="ls -la")

        assert action.action == "ls -la"
        assert action.action_type == "bash"
        assert action.hypothesis == ""
        assert action.expected_evidence == ""
        assert action.goal_connection == ""
        assert action.confidence == 0.5
        assert action.justification_quality == JustificationQuality.NONE
        assert action.goal_proximity == GoalProximity.IRRELEVANT

    def test_score_multiplier_none(self):
        """Score multiplier is 0 for unjustified actions."""
        action = JustifiedAction(action="ls -la")
        # Default: NONE quality × IRRELEVANT proximity = 0.0 × 0.0 = 0.0
        assert action.score_multiplier() == 0.0

    def test_score_multiplier_excellent_direct(self):
        """Score multiplier is 1.0 for excellent + direct."""
        action = JustifiedAction(
            action="grep -r 'NEEDLE' .",
            justification_quality=JustificationQuality.EXCELLENT,
            goal_proximity=GoalProximity.DIRECT
        )
        assert action.score_multiplier() == 1.0

    def test_score_multiplier_partial(self):
        """Score multiplier scales with quality × proximity."""
        action = JustifiedAction(
            action="cat README.md",
            justification_quality=JustificationQuality.ACCEPTABLE,  # 0.5
            goal_proximity=GoalProximity.EXPLORATORY  # 0.4
        )
        assert action.score_multiplier() == pytest.approx(0.2)


class TestJustificationQuality:
    """Test JustificationQuality enum values."""

    def test_quality_values(self):
        """Verify quality enum values."""
        assert JustificationQuality.EXCELLENT.value == 1.0
        assert JustificationQuality.GOOD.value == 0.75
        assert JustificationQuality.ACCEPTABLE.value == 0.5
        assert JustificationQuality.WEAK.value == 0.25
        assert JustificationQuality.NONE.value == 0.0

    def test_quality_ordering(self):
        """Quality values should be ordered."""
        qualities = [
            JustificationQuality.EXCELLENT,
            JustificationQuality.GOOD,
            JustificationQuality.ACCEPTABLE,
            JustificationQuality.WEAK,
            JustificationQuality.NONE
        ]
        values = [q.value for q in qualities]
        assert values == sorted(values, reverse=True)


class TestGoalProximity:
    """Test GoalProximity enum values."""

    def test_proximity_values(self):
        """Verify proximity enum values."""
        assert GoalProximity.DIRECT.value == 1.0
        assert GoalProximity.INDIRECT.value == 0.7
        assert GoalProximity.EXPLORATORY.value == 0.4
        assert GoalProximity.TANGENTIAL.value == 0.1
        assert GoalProximity.IRRELEVANT.value == 0.0

    def test_proximity_ordering(self):
        """Proximity values should be ordered."""
        proximities = [
            GoalProximity.DIRECT,
            GoalProximity.INDIRECT,
            GoalProximity.EXPLORATORY,
            GoalProximity.TANGENTIAL,
            GoalProximity.IRRELEVANT
        ]
        values = [p.value for p in proximities]
        assert values == sorted(values, reverse=True)


class TestJustificationEvaluator:
    """Test JustificationEvaluator class."""

    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = JustificationEvaluator(
            goal_description="Find the needle",
            goal_keywords=["needle", "find", "search"]
        )
        assert evaluator.goal_description == "Find the needle"
        assert "needle" in evaluator.goal_keywords
        assert len(evaluator.action_history) == 0

    def test_evaluate_no_justification(self):
        """Unjustified actions get NONE quality."""
        evaluator = JustificationEvaluator(
            goal_description="Find the config file",
            goal_keywords=["config", "file", "find"]
        )

        action = JustifiedAction(
            action="ls -la",
            hypothesis="",  # No hypothesis
            expected_evidence="",
            goal_connection=""
        )

        evaluated = evaluator.evaluate_justification(
            action,
            outcome="drwxr-xr-x files...",
            goal_achieved=False
        )

        assert evaluated.justification_quality == JustificationQuality.NONE

    def test_evaluate_good_justification(self):
        """Well-justified actions with matching evidence get higher quality."""
        evaluator = JustificationEvaluator(
            goal_description="Find the config file",
            goal_keywords=["config", "file", "find", "yaml"]
        )

        action = JustifiedAction(
            action="find . -name '*.yaml'",
            hypothesis="I expect this to find YAML config files",
            expected_evidence="List of .yaml files including config",
            goal_connection="YAML files are common config formats"
        )

        evaluated = evaluator.evaluate_justification(
            action,
            outcome="./config.yaml\n./settings.yaml\n./docker-compose.yaml",
            goal_achieved=False
        )

        # Should be at least ACCEPTABLE since evidence mentions yaml/config
        assert evaluated.justification_quality.value >= JustificationQuality.ACCEPTABLE.value

    def test_evaluate_goal_achieved(self):
        """Goal achievement gives DIRECT proximity."""
        evaluator = JustificationEvaluator(
            goal_description="Find NEEDLE_VALUE",
            goal_keywords=["needle", "value", "find"]
        )

        action = JustifiedAction(
            action="grep -r 'NEEDLE_VALUE' .",
            hypothesis="This should find the needle",
            expected_evidence="File containing NEEDLE_VALUE",
            goal_connection="Direct search for the needle"
        )

        evaluated = evaluator.evaluate_justification(
            action,
            outcome="./config.py:7:NEEDLE_VALUE='secret'",
            goal_achieved=True
        )

        assert evaluated.goal_proximity == GoalProximity.DIRECT

    def test_compute_adjusted_score(self):
        """Test score adjustment calculation."""
        evaluator = JustificationEvaluator()

        # Action with 50% quality × 40% proximity = 20% multiplier
        action = JustifiedAction(
            action="cat file.txt",
            justification_quality=JustificationQuality.ACCEPTABLE,  # 0.5
            goal_proximity=GoalProximity.EXPLORATORY  # 0.4
        )

        base_score = 100
        adjusted = evaluator.compute_adjusted_score(base_score, action)

        assert adjusted == 20  # 100 × 0.5 × 0.4 = 20


class TestMetricBiasFix:
    """
    Test the Metric Bias fix for llama3.2:3b anomaly.

    Original problem:
    - +4490 points (highest)
    - 1/6 wins (near-worst)

    Expected fix:
    - ~75% score reduction for unjustified actions
    """

    def test_confabulation_penalty(self):
        """Verbose confabulation should be heavily penalized."""
        evaluator = JustificationEvaluator(
            goal_description="Find NEEDLE in project",
            goal_keywords=["needle", "find", "search"]
        )

        # Simulate llama3.2:3b pattern: many unjustified actions
        actions = [
            JustifiedAction(action="ls -la", hypothesis="", expected_evidence="", goal_connection=""),
            JustifiedAction(action="cat README.md", hypothesis="", expected_evidence="", goal_connection=""),
            JustifiedAction(action="cat setup.py", hypothesis="", expected_evidence="", goal_connection=""),
            JustifiedAction(action="ls src/", hypothesis="", expected_evidence="", goal_connection=""),
        ]

        base_scores = [50, 30, 40, 35]  # Total: 155

        total_old = sum(base_scores)
        total_new = 0

        for action, base in zip(actions, base_scores):
            evaluated = evaluator.evaluate_justification(action, "output", False)
            adjusted = evaluator.compute_adjusted_score(base, evaluated)
            total_new += adjusted

        # Should see significant reduction (>50%)
        reduction = (total_old - total_new) / total_old
        assert reduction >= 0.5, f"Expected >50% reduction, got {reduction:.0%}"

    def test_justified_actions_preserved(self):
        """Well-justified actions should keep most of their score."""
        evaluator = JustificationEvaluator(
            goal_description="Find NEEDLE_VALUE in project",
            goal_keywords=["needle", "value", "find", "grep", "search"]
        )

        action = JustifiedAction(
            action="grep -r 'NEEDLE_VALUE' .",
            hypothesis="I expect this to find the file containing NEEDLE_VALUE",
            expected_evidence="File path and line with NEEDLE_VALUE",
            goal_connection="This directly searches for the target value"
        )

        evaluated = evaluator.evaluate_justification(
            action,
            outcome="./src/config.py:7:NEEDLE_VALUE = 'secret123'",
            goal_achieved=True
        )

        base_score = 100
        adjusted = evaluator.compute_adjusted_score(base_score, evaluated)

        # Should keep at least 50% of score
        assert adjusted >= 50, f"Expected >=50% preserved, got {adjusted}%"


class TestResponseParsing:
    """Test parsing of justified responses from LLMs."""

    def test_parse_valid_response(self):
        """Parse well-formed JSON response."""
        response = '''
        Let me search for the file.
        ```json
        {
            "justification": {
                "hypothesis": "I expect to find Python files",
                "expected_evidence": "List of .py files",
                "goal_connection": "Python files may contain the target",
                "confidence": 0.8
            },
            "action": {
                "type": "bash",
                "command": "find . -name '*.py'"
            }
        }
        ```
        '''

        action = parse_justified_response(response)

        assert action is not None
        assert action.action == "find . -name '*.py'"
        assert action.action_type == "bash"
        assert action.hypothesis == "I expect to find Python files"
        assert action.confidence == 0.8

    def test_parse_invalid_response(self):
        """Return None for invalid responses."""
        response = "Just run ls -la"

        action = parse_justified_response(response)

        assert action is None

    def test_parse_partial_response(self):
        """Handle partially complete JSON."""
        response = '''
        {
            "justification": {
                "hypothesis": "Check the directory"
            },
            "action": {
                "command": "ls"
            }
        }
        '''

        action = parse_justified_response(response)

        assert action is not None
        assert action.action == "ls"
        assert action.hypothesis == "Check the directory"
        # Missing fields should have defaults
        assert action.expected_evidence == ""
        assert action.confidence == 0.5


class TestEvaluatorSummary:
    """Test evaluator summary statistics."""

    def test_empty_summary(self):
        """Summary of empty evaluator."""
        evaluator = JustificationEvaluator()
        summary = evaluator.get_summary()

        assert summary["total_actions"] == 0

    def test_summary_with_actions(self):
        """Summary after evaluating actions."""
        evaluator = JustificationEvaluator(
            goal_description="Find the config",
            goal_keywords=["config"]
        )

        # Add some actions
        for i in range(5):
            action = JustifiedAction(
                action=f"command_{i}",
                hypothesis="test hypothesis" if i % 2 == 0 else ""
            )
            evaluator.evaluate_justification(action, "output", i == 4)

        summary = evaluator.get_summary()

        assert summary["total_actions"] == 5
        assert "avg_quality" in summary
        assert "avg_proximity" in summary
        assert "excellent_count" in summary
        assert "none_count" in summary


class TestPromptTemplate:
    """Test justification prompt template."""

    def test_template_exists(self):
        """Prompt template should be defined."""
        assert JUSTIFICATION_PROMPT_TEMPLATE is not None
        assert len(JUSTIFICATION_PROMPT_TEMPLATE) > 0

    def test_template_contains_structure(self):
        """Template should explain expected structure."""
        assert "hypothesis" in JUSTIFICATION_PROMPT_TEMPLATE.lower()
        assert "expected_evidence" in JUSTIFICATION_PROMPT_TEMPLATE.lower()
        assert "goal_connection" in JUSTIFICATION_PROMPT_TEMPLATE.lower()
        assert "confidence" in JUSTIFICATION_PROMPT_TEMPLATE.lower()

    def test_template_contains_scoring_info(self):
        """Template should explain scoring impact."""
        assert "0%" in JUSTIFICATION_PROMPT_TEMPLATE or "without" in JUSTIFICATION_PROMPT_TEMPLATE.lower()
        assert "100%" in JUSTIFICATION_PROMPT_TEMPLATE or "excellent" in JUSTIFICATION_PROMPT_TEMPLATE.lower()


class TestKeywordExtraction:
    """Test keyword extraction helper."""

    def test_stopword_removal(self):
        """Common stopwords should be removed."""
        evaluator = JustificationEvaluator()

        text = "I expect this to reveal the file content"
        keywords = evaluator._extract_keywords(text)

        assert "i" not in keywords
        assert "this" not in keywords
        assert "to" not in keywords
        assert "the" not in keywords

    def test_meaningful_words_kept(self):
        """Meaningful words should be preserved."""
        evaluator = JustificationEvaluator()

        text = "Find the configuration file in the project directory"
        keywords = evaluator._extract_keywords(text)

        assert "configuration" in keywords or "config" in keywords
        assert "file" in keywords
        assert "project" in keywords
        assert "directory" in keywords


# === Fixtures ===

@pytest.fixture
def needle_evaluator():
    """Evaluator for 'find needle' goal."""
    return JustificationEvaluator(
        goal_description="Find NEEDLE_VALUE in the project",
        goal_keywords=["needle", "value", "find", "grep", "search", "file"]
    )


@pytest.fixture
def config_evaluator():
    """Evaluator for 'find config' goal."""
    return JustificationEvaluator(
        goal_description="Locate the configuration file",
        goal_keywords=["config", "configuration", "yaml", "json", "settings"]
    )


# === Run Tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
