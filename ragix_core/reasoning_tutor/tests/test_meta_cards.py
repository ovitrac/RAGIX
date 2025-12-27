#!/usr/bin/env python3
"""
Test Suite for MetaCards (R2)
==============================

Tests strategic intervention cards:
- Card template structure
- Card selection based on failure type
- Context enrichment
- Prompt injection formatting
- Effectiveness tracking

Key validation: Break the Complexity Wall (B03-B05)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_cards import (
    MetaCard,
    MetaCardType,
    MetaCardSelector,
    CARD_TEMPLATES
)


class TestMetaCardType:
    """Test MetaCardType enum."""

    def test_all_types_defined(self):
        """All expected card types should exist."""
        assert MetaCardType.ESCAPE_LOOP
        assert MetaCardType.COMPASS
        assert MetaCardType.ERROR_ANALYSIS
        assert MetaCardType.PROGRESS_BOOST
        assert MetaCardType.STRATEGIC_RESET

    def test_type_values(self):
        """Card type values should be strings."""
        for card_type in MetaCardType:
            assert isinstance(card_type.value, str)
            assert len(card_type.value) > 0


class TestCardTemplates:
    """Test pre-defined card templates."""

    def test_all_types_have_templates(self):
        """Every card type should have a template."""
        for card_type in MetaCardType:
            assert card_type in CARD_TEMPLATES, f"Missing template for {card_type}"

    def test_template_required_fields(self):
        """Templates should have all required fields."""
        required_fields = ["title", "instruction", "rationale", "example", "anti_pattern"]

        for card_type, template in CARD_TEMPLATES.items():
            for field in required_fields:
                assert field in template, f"Missing {field} in {card_type} template"
                assert len(template[field]) > 0, f"Empty {field} in {card_type} template"

    def test_escape_loop_template(self):
        """ESCAPE_LOOP template should address repetition."""
        template = CARD_TEMPLATES[MetaCardType.ESCAPE_LOOP]

        assert "repeat" in template["instruction"].lower() or "loop" in template["title"].lower()
        assert "different" in template["instruction"].lower()

    def test_compass_template(self):
        """COMPASS template should address cycling."""
        template = CARD_TEMPLATES[MetaCardType.COMPASS]

        assert "cycle" in template["rationale"].lower() or "circular" in template["rationale"].lower()
        assert "enumerate" in template["instruction"].lower() or "list" in template["example"].lower()

    def test_error_analysis_template(self):
        """ERROR_ANALYSIS template should address errors."""
        template = CARD_TEMPLATES[MetaCardType.ERROR_ANALYSIS]

        assert "error" in template["instruction"].lower()
        assert "analyze" in template["instruction"].lower() or "parse" in template["title"].lower()

    def test_progress_boost_template(self):
        """PROGRESS_BOOST template should address stalls."""
        template = CARD_TEMPLATES[MetaCardType.PROGRESS_BOOST]

        assert "break" in template["instruction"].lower() or "decompose" in template["instruction"].lower()
        assert "sub-goal" in template["example"].lower() or "smaller" in template["instruction"].lower()

    def test_strategic_reset_template(self):
        """STRATEGIC_RESET template should address exhaustion."""
        template = CARD_TEMPLATES[MetaCardType.STRATEGIC_RESET]

        assert "summarize" in template["instruction"].lower() or "pivot" in template["title"].lower()
        assert "new" in template["instruction"].lower() or "fresh" in template["rationale"].lower()


class TestMetaCard:
    """Test MetaCard dataclass."""

    def test_card_creation(self):
        """Create a card with required fields."""
        card = MetaCard(
            card_type=MetaCardType.ESCAPE_LOOP,
            card_id="escape_1",
            title="Escape the Loop",
            instruction="Try something different",
            rationale="Repetition won't help",
            example="Instead of X, try Y",
            anti_pattern="Don't repeat"
        )

        assert card.card_type == MetaCardType.ESCAPE_LOOP
        assert card.card_id == "escape_1"
        assert card.title == "Escape the Loop"

    def test_card_defaults(self):
        """Card should have sensible defaults."""
        card = MetaCard(
            card_type=MetaCardType.COMPASS,
            card_id="compass_1",
            title="Find Your Compass",
            instruction="Enumerate options",
            rationale="Cycles indicate confusion",
            example="List all possibilities",
            anti_pattern="Don't cycle"
        )

        assert card.trigger_context == {}
        assert card.pcg_hint is None
        assert card.times_used == 0
        assert card.success_rate == 0.0

    def test_to_prompt_injection(self):
        """Card should format correctly for prompt injection."""
        card = MetaCard(
            card_type=MetaCardType.ERROR_ANALYSIS,
            card_id="error_1",
            title="Parse the Error",
            instruction="Read the error message",
            rationale="Errors contain clues",
            example="FileNotFoundError means...",
            anti_pattern="Don't ignore errors",
            pcg_hint="Last error: No such file"
        )

        prompt = card.to_prompt_injection()

        assert "Parse the Error" in prompt
        assert "Read the error message" in prompt
        assert "Errors contain clues" in prompt
        assert "FileNotFoundError" in prompt
        assert "Don't ignore errors" in prompt
        assert "Last error: No such file" in prompt

    def test_to_dict(self):
        """Card should serialize to dictionary."""
        card = MetaCard(
            card_type=MetaCardType.PROGRESS_BOOST,
            card_id="boost_1",
            title="Decompose",
            instruction="Break it down",
            rationale="Smaller steps",
            example="Try sub-goals",
            anti_pattern="Don't wander",
            trigger_context={"stall_turns": 5},
            pcg_hint="No progress for 5 turns"
        )

        d = card.to_dict()

        assert d["card_type"] == "progress_boost"
        assert d["card_id"] == "boost_1"
        assert d["title"] == "Decompose"
        assert d["trigger_context"]["stall_turns"] == 5
        assert d["pcg_hint"] == "No progress for 5 turns"


class TestMetaCardSelector:
    """Test MetaCardSelector class."""

    def test_selector_init(self):
        """Selector should initialize empty."""
        selector = MetaCardSelector()

        assert len(selector.cards_issued) == 0
        assert len(selector.effectiveness) == 0

    def test_select_escape_loop_card(self):
        """Select ESCAPE_LOOP card for repetition failures."""
        selector = MetaCardSelector()

        card = selector.select_card(
            failure_type="repetition_loop",
            context={"repeated_action": "cat file.txt", "repetitions": 4}
        )

        assert card is not None
        assert card.card_type == MetaCardType.ESCAPE_LOOP
        assert "cat file.txt" in (card.pcg_hint or "")

    def test_select_compass_card(self):
        """Select COMPASS card for circular pattern failures."""
        selector = MetaCardSelector()

        card = selector.select_card(
            failure_type="circular_pattern",
            context={"pattern_sequence": ["ls", "cat", "ls", "cat"]}
        )

        assert card is not None
        assert card.card_type == MetaCardType.COMPASS
        assert "ls" in (card.pcg_hint or "") or "cat" in (card.pcg_hint or "")

    def test_select_error_analysis_card(self):
        """Select ERROR_ANALYSIS card for explicit error failures."""
        selector = MetaCardSelector()

        card = selector.select_card(
            failure_type="explicit_error",
            context={"last_error": "FileNotFoundError: /path/to/file"}
        )

        assert card is not None
        assert card.card_type == MetaCardType.ERROR_ANALYSIS
        assert "FileNotFoundError" in (card.pcg_hint or "")

    def test_select_progress_boost_card(self):
        """Select PROGRESS_BOOST card for progress stall failures."""
        selector = MetaCardSelector()

        card = selector.select_card(
            failure_type="progress_stall",
            context={"stall_turns": 5}
        )

        assert card is not None
        assert card.card_type == MetaCardType.PROGRESS_BOOST
        assert "5" in (card.pcg_hint or "")

    def test_select_strategic_reset_card(self):
        """Select STRATEGIC_RESET card for exhaustion failures."""
        selector = MetaCardSelector()

        card = selector.select_card(
            failure_type="exhaustion",
            context={"approaches_tried": ["grep", "find", "cat"]}
        )

        assert card is not None
        assert card.card_type == MetaCardType.STRATEGIC_RESET

    def test_unknown_failure_type(self):
        """Return None for unknown failure types."""
        selector = MetaCardSelector()

        card = selector.select_card(
            failure_type="unknown_type",
            context={}
        )

        assert card is None

    def test_card_id_uniqueness(self):
        """Each issued card should have unique ID."""
        selector = MetaCardSelector()

        card1 = selector.select_card("repetition_loop", {})
        card2 = selector.select_card("repetition_loop", {})
        card3 = selector.select_card("circular_pattern", {})

        ids = {card1.card_id, card2.card_id, card3.card_id}
        assert len(ids) == 3  # All unique


class TestCardEffectivenessTracking:
    """Test card effectiveness tracking."""

    def test_record_outcome_success(self):
        """Record successful card outcome."""
        selector = MetaCardSelector()

        card = selector.select_card("repetition_loop", {})
        selector.record_outcome(card.card_id, success=True)

        assert card.card_id in selector.effectiveness
        assert selector.effectiveness[card.card_id] == [True]
        assert card.success_rate == 1.0
        assert card.times_used == 1

    def test_record_outcome_failure(self):
        """Record failed card outcome."""
        selector = MetaCardSelector()

        card = selector.select_card("circular_pattern", {})
        selector.record_outcome(card.card_id, success=False)

        assert selector.effectiveness[card.card_id] == [False]
        assert card.success_rate == 0.0
        assert card.times_used == 1

    def test_record_multiple_outcomes(self):
        """Track multiple outcomes for same card type."""
        selector = MetaCardSelector()

        card = selector.select_card("explicit_error", {})  # Use correct failure type name
        assert card is not None, "Card should be selected for explicit_error"

        selector.record_outcome(card.card_id, success=True)
        selector.record_outcome(card.card_id, success=True)
        selector.record_outcome(card.card_id, success=False)

        assert card.times_used == 3
        assert card.success_rate == pytest.approx(2/3)


class TestSelectorStatistics:
    """Test selector statistics."""

    def test_empty_statistics(self):
        """Statistics for empty selector."""
        selector = MetaCardSelector()
        stats = selector.get_statistics()

        assert stats["total_cards_issued"] == 0
        assert stats["overall_success_rate"] == 0.0

    def test_statistics_with_cards(self):
        """Statistics after issuing cards."""
        selector = MetaCardSelector()

        # Issue cards of different types
        card1 = selector.select_card("repetition_loop", {})
        card2 = selector.select_card("circular_pattern", {})
        card3 = selector.select_card("repetition_loop", {})

        # Record outcomes
        selector.record_outcome(card1.card_id, True)
        selector.record_outcome(card2.card_id, False)
        selector.record_outcome(card3.card_id, True)

        stats = selector.get_statistics()

        assert stats["total_cards_issued"] == 3
        assert stats["overall_success_rate"] == pytest.approx(2/3)
        assert "escape_loop" in stats["by_type"]
        assert stats["by_type"]["escape_loop"]["count"] == 2


class TestContextEnrichment:
    """Test context-specific card enrichment."""

    def test_enrich_repetition_context(self):
        """ESCAPE_LOOP cards include repeated action."""
        selector = MetaCardSelector()

        card = selector.select_card(
            "repetition_loop",
            {"repeated_action": "find . -name '*.py'"}
        )

        assert card.pcg_hint is not None
        assert "find" in card.pcg_hint

    def test_enrich_circular_context(self):
        """COMPASS cards include cycle sequence."""
        selector = MetaCardSelector()

        card = selector.select_card(
            "circular_pattern",
            {"pattern_sequence": ["ls", "cat README", "ls"]}
        )

        assert card.pcg_hint is not None
        assert "→" in card.pcg_hint  # Arrow notation for cycle

    def test_enrich_error_context(self):
        """ERROR_ANALYSIS cards include error message."""
        selector = MetaCardSelector()

        card = selector.select_card(
            "explicit_error",
            {"last_error": "PermissionError: access denied"}
        )

        assert card.pcg_hint is not None
        assert "PermissionError" in card.pcg_hint

    def test_enrich_stall_context(self):
        """PROGRESS_BOOST cards include stall duration."""
        selector = MetaCardSelector()

        card = selector.select_card(
            "progress_stall",
            {"stall_turns": 7}
        )

        assert card.pcg_hint is not None
        assert "7" in card.pcg_hint

    def test_enrich_exhaustion_context(self):
        """STRATEGIC_RESET cards include tried approaches."""
        selector = MetaCardSelector()

        card = selector.select_card(
            "exhaustion",
            {"approaches_tried": ["grep", "find", "awk"]}
        )

        assert card.pcg_hint is not None
        assert "grep" in card.pcg_hint or "Tried" in card.pcg_hint


class TestFailureTypeMapping:
    """Test mapping from failure types to card types."""

    @pytest.mark.parametrize("failure_type,expected_card_type", [
        ("repetition_loop", MetaCardType.ESCAPE_LOOP),
        ("circular_pattern", MetaCardType.COMPASS),
        ("explicit_error", MetaCardType.ERROR_ANALYSIS),
        ("progress_stall", MetaCardType.PROGRESS_BOOST),
        ("exhaustion", MetaCardType.STRATEGIC_RESET),
    ])
    def test_failure_to_card_mapping(self, failure_type, expected_card_type):
        """Each failure type maps to correct card type."""
        selector = MetaCardSelector()

        card = selector.select_card(failure_type, {})

        assert card is not None
        assert card.card_type == expected_card_type

    def test_case_insensitive_mapping(self):
        """Failure type mapping should be case-insensitive."""
        selector = MetaCardSelector()

        card1 = selector.select_card("REPETITION_LOOP", {})
        card2 = selector.select_card("Repetition_Loop", {})
        card3 = selector.select_card("repetition_loop", {})

        assert card1 is not None
        assert card2 is not None
        assert card3 is not None
        assert card1.card_type == card2.card_type == card3.card_type


class TestOlympicsValidation:
    """
    Validate cards against Olympics failure patterns.

    Key patterns to address:
    - granite B01: repetition of find commands
    - dolphin: circular ls → cat → ls → cat
    - phi3: error cascade from T3-T10
    - llama3.2: progress stall (high activity, no PCG growth)
    """

    def test_granite_b01_card(self):
        """Card for granite's B01 repetition pattern."""
        selector = MetaCardSelector()

        # Simulate granite's B01 failure
        card = selector.select_card(
            "repetition_loop",
            {
                "repeated_action": "find . -name '*.py' -exec cat {} \\;",
                "repetitions": 4,
                "turn_detected": 6
            }
        )

        assert card is not None
        assert card.card_type == MetaCardType.ESCAPE_LOOP
        # Card should suggest trying different approach
        assert "different" in card.instruction.lower()

    def test_dolphin_circular_card(self):
        """Card for dolphin's circular pattern."""
        selector = MetaCardSelector()

        # Simulate dolphin's cycling
        card = selector.select_card(
            "circular_pattern",
            {
                "pattern_sequence": ["ls -la", "cat main.py", "ls -la", "cat main.py"],
                "cycle_length": 2
            }
        )

        assert card is not None
        assert card.card_type == MetaCardType.COMPASS
        # Card should suggest enumeration
        assert "enumerate" in card.instruction.lower() or "list" in card.example.lower()

    def test_phi3_error_cascade_card(self):
        """Card for phi3's error cascade."""
        selector = MetaCardSelector()

        # Simulate phi3's errors
        card = selector.select_card(
            "explicit_error",
            {
                "last_error": "Error: invalid syntax",
                "error_count": 8,
                "turn_detected": 5
            }
        )

        assert card is not None
        assert card.card_type == MetaCardType.ERROR_ANALYSIS
        # Card should suggest parsing error
        assert "analyze" in card.instruction.lower() or "error" in card.instruction.lower()

    def test_llama32_stall_card(self):
        """Card for llama3.2's progress stall (confabulation)."""
        selector = MetaCardSelector()

        # Simulate llama3.2's stall
        card = selector.select_card(
            "progress_stall",
            {
                "stall_turns": 7,
                "last_pcg_size": 1,
                "actions_since_progress": 7
            }
        )

        assert card is not None
        assert card.card_type == MetaCardType.PROGRESS_BOOST
        # Card should suggest decomposition
        assert "break" in card.instruction.lower() or "sub-goal" in card.example.lower()


# === Fixtures ===

@pytest.fixture
def fresh_selector():
    """Provide a fresh MetaCardSelector."""
    return MetaCardSelector()


# === Run Tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
