#!/usr/bin/env python3
"""
Meta-Cards - Strategic Interventions for Stuck LLMs
====================================================

**ROADMAP STATUS: R2 DESIGN** — Priority P1

Problem Statement:
- The "Complexity Wall" (B03-B05) blocks most models
- Current help cards are generic, static, and reactive
- Models that "know when to ask for help" (granite) outperform stubborn ones

Solution: Context-aware meta-cards triggered by FailureDetector.
Each card type matches a specific failure pattern and provides
targeted strategic guidance.

Integration Flow:
    FailureDetector → MetaCardSelector → GameLoop → LLM
         ↓                  ↓               ↓         ↓
    Detect stuck      Select card      Inject hint   Resume

Card Types:
- ESCAPE_LOOP: Break repetition loops (perseveration)
- COMPASS: Navigate circular patterns (disorientation)
- ERROR_ANALYSIS: Parse and learn from errors (agnosia)
- PROGRESS_BOOST: Decompose stalled progress
- STRATEGIC_RESET: Pivot when exhausted

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json

# Import config for defaults
from config import get_config, is_meta_cards_enabled


class MetaCardType(Enum):
    """Types of meta-cards, matched to failure types."""
    # === R3: Reactive Cards (triggered by FailureDetector) ===
    ESCAPE_LOOP = "escape_loop"           # For REPETITION_LOOP
    COMPASS = "compass"                   # For CIRCULAR_PATTERN
    ERROR_ANALYSIS = "error_analysis"     # For EXPLICIT_ERROR
    PROGRESS_BOOST = "progress_boost"     # For PROGRESS_STALL
    STRATEGIC_RESET = "strategic_reset"   # For EXHAUSTION

    # === R4: TRIZ Strategic Cards (triggered by ContradictionDetector) ===
    SEGMENT_TASK = "segment_task"         # TRIZ #1: Segmentation (Divide & Conquer)
    DEFINE_CRITERIA = "define_criteria"   # TRIZ #10: Prior Action (Pre-computation)
    LIST_INSTEAD = "list_instead"         # TRIZ #13: Inversion (Try opposite)

    # === R4: Kanban Constraint Cards ===
    WIP_OVERFLOW = "wip_overflow"         # Kanban: WIP limit exceeded


@dataclass
class MetaCard:
    """
    A strategic intervention card for stuck LLMs.

    Cards are context-aware hints that help models break out of
    failure patterns without giving away the solution.
    """
    card_type: MetaCardType
    card_id: str                  # Unique identifier

    # Content
    title: str                    # Human-readable title
    instruction: str              # What to do (imperative)
    rationale: str                # Why this helps
    example: str                  # Concrete example
    anti_pattern: str             # What NOT to do

    # Context (filled by selector)
    trigger_context: Dict[str, Any] = field(default_factory=dict)
    pcg_hint: Optional[str] = None  # Relevant PCG evidence

    # Tracking
    times_used: int = 0
    success_rate: float = 0.0     # Historical success rate

    def to_prompt_injection(self) -> str:
        """Format the card for injection into LLM prompt."""
        return f"""
📋 STRATEGIC HINT ({self.title})
{'='*50}

🎯 INSTRUCTION: {self.instruction}

💡 WHY: {self.rationale}

✅ EXAMPLE: {self.example}

❌ AVOID: {self.anti_pattern}

{f"🔍 HINT: {self.pcg_hint}" if self.pcg_hint else ""}
{'='*50}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "card_type": self.card_type.value,
            "card_id": self.card_id,
            "title": self.title,
            "instruction": self.instruction,
            "trigger_context": self.trigger_context,
            "pcg_hint": self.pcg_hint
        }


# === Pre-defined Card Templates ===

CARD_TEMPLATES: Dict[MetaCardType, Dict[str, Any]] = {
    MetaCardType.ESCAPE_LOOP: {
        "title": "Escape the Loop",
        "instruction": "You are repeating the same action. Try a DIFFERENT approach.",
        "rationale": "Repeating the same command won't yield new results. "
                     "Change your strategy, not just parameters.",
        "example": "Instead of 'cat file.txt' again, try 'grep pattern file.txt' "
                   "or 'head -20 file.txt' for a different view.",
        "anti_pattern": "Do NOT run the same command with minor variations. "
                        "Step back and reconsider your approach entirely."
    },
    MetaCardType.COMPASS: {
        "title": "Find Your Compass",
        "instruction": "You are cycling through actions without progress. "
                       "ENUMERATE all options before choosing.",
        "rationale": "Circular patterns indicate strategic confusion. "
                     "A systematic enumeration prevents endless loops.",
        "example": "Before your next action, list: (1) Files not yet checked, "
                   "(2) Commands not yet tried, (3) Patterns not yet searched.",
        "anti_pattern": "Do NOT immediately try another action. "
                        "First create a checklist of what you've tried and what remains."
    },
    MetaCardType.ERROR_ANALYSIS: {
        "title": "Parse the Error",
        "instruction": "ANALYZE the error message carefully before retrying.",
        "rationale": "Errors contain diagnostic information. "
                     "Understanding the error prevents repeated failures.",
        "example": "'FileNotFoundError: /path/to/file' means: "
                   "(1) Check if path exists, (2) Check spelling, (3) Check permissions.",
        "anti_pattern": "Do NOT ignore error details. "
                        "Do NOT assume you know the cause without reading the message."
    },
    MetaCardType.PROGRESS_BOOST: {
        "title": "Decompose the Problem",
        "instruction": "BREAK DOWN the goal into smaller, verifiable sub-goals.",
        "rationale": "You're reading without learning. "
                     "Each action should add to your verified knowledge.",
        "example": "Instead of 'find the answer', try: "
                   "(1) Verify file exists, (2) Find relevant section, "
                   "(3) Extract specific value, (4) Confirm it matches criteria.",
        "anti_pattern": "Do NOT continue exploring without a clear sub-goal. "
                        "Each action should have a specific, verifiable outcome."
    },
    MetaCardType.STRATEGIC_RESET: {
        "title": "Strategic Pivot",
        "instruction": "SUMMARIZE your progress so far and CHOOSE a new direction.",
        "rationale": "You've tried many approaches. "
                     "A fresh perspective may reveal missed opportunities.",
        "example": "Progress so far: I found [X], ruled out [Y], but haven't tried [Z]. "
                   "New strategy: Focus on [Z] because [reason].",
        "anti_pattern": "Do NOT give up. Do NOT repeat failed strategies. "
                        "Use what you've learned to inform a new approach."
    },

    # === R4: TRIZ Strategic Cards ===

    MetaCardType.SEGMENT_TASK: {
        "title": "Divide & Conquer (TRIZ #1)",
        "instruction": "SEGMENT the task. Process ONE file at a time, "
                       "recording facts to memory before moving to the next.",
        "rationale": "You're losing context by processing too much at once. "
                     "Break the problem into atomic steps with checkpoints.",
        "example": "Instead of reading all files: "
                   "(1) Read file1 -> Extract fact -> Record T1 -> "
                   "(2) Read file2 -> Extract fact -> Record T2 -> etc.",
        "anti_pattern": "Do NOT read multiple files in one command. "
                        "Do NOT proceed without recording each fact explicitly.",
        "triz_principle": 1,
        "target_benchmark": "B06_Memory"
    },
    MetaCardType.DEFINE_CRITERIA: {
        "title": "Define Before Search (TRIZ #10)",
        "instruction": "ASSERT your success criteria BEFORE reading. "
                       "What exactly will prove or disprove your hypothesis?",
        "rationale": "You're reading without knowing what to look for. "
                     "Pre-computation prevents wasted exploration.",
        "example": "Before 'cat config.yaml', state: "
                   "'I am looking for key=X where value matches pattern Y. "
                   "If found -> T(valid). If not found -> T(missing).'",
        "anti_pattern": "Do NOT read files 'to see what's there'. "
                        "Do NOT proceed without explicit success criteria.",
        "triz_principle": 10,
        "target_benchmark": "B04_Verification"
    },
    MetaCardType.LIST_INSTEAD: {
        "title": "Invert Your Approach (TRIZ #13)",
        "instruction": "If you cannot find X by guessing, ENUMERATE the space. "
                       "List the directory to see what actually exists.",
        "rationale": "You're guessing file names that don't exist. "
                     "Inversion: discover what IS there, then select.",
        "example": "Instead of 'cat config.yaml' (FileNotFoundError), try: "
                   "'ls -la' to see actual files, then read the correct one.",
        "anti_pattern": "Do NOT keep guessing file names. "
                        "Do NOT assume files exist without verification.",
        "triz_principle": 13,
        "target_benchmark": "B01_FileNotFound"
    },

    # === R4: Kanban Constraint Cards ===

    MetaCardType.WIP_OVERFLOW: {
        "title": "Close Before Opening (Kanban WIP)",
        "instruction": "You have too many open threads. FINISH one investigation "
                       "before starting another.",
        "rationale": "Parallel exploration causes context loss. "
                     "Complete current work before opening new questions.",
        "example": "You have Q1 (config location) and Q2 (version check) open. "
                   "Complete Q1 with a definitive answer before exploring Q2.",
        "anti_pattern": "Do NOT open a third question/action. "
                        "Do NOT switch context without recording findings.",
        "kanban_rule": "wip_limit"
    }
}


class MetaCardSelector:
    """
    Selects appropriate meta-cards based on failure context.

    This is the brain that connects FailureDetector to helpful interventions.
    """

    def __init__(self):
        # Load config
        self._config = get_config().meta_cards

        # Card usage tracking
        self.cards_issued: List[MetaCard] = []
        self.effectiveness: Dict[str, List[bool]] = {}  # card_id -> [success, ...]

    def select_card(
        self,
        failure_type: str,
        context: Dict[str, Any]
    ) -> Optional[MetaCard]:
        """
        Select the most appropriate card for the failure.

        Args:
            failure_type: Type from FailureDetector (e.g., "repetition_loop")
            context: Context from FailureContext.details

        Returns:
            MetaCard ready for prompt injection, or None if:
            - Meta-cards are disabled in config
            - Max cards per game reached
            - Specific card type is disabled
            - No matching card type
        """
        # Check if meta-cards are enabled
        if not is_meta_cards_enabled():
            return None

        # Check if we've reached max cards per game
        if len(self.cards_issued) >= self._config.max_cards_per_game:
            return None

        # Map failure type to card type
        type_mapping = {
            # R3: Reactive cards (FailureDetector triggers)
            "repetition_loop": MetaCardType.ESCAPE_LOOP,
            "circular_pattern": MetaCardType.COMPASS,
            "explicit_error": MetaCardType.ERROR_ANALYSIS,
            "progress_stall": MetaCardType.PROGRESS_BOOST,
            "exhaustion": MetaCardType.STRATEGIC_RESET,
            # R4: TRIZ strategic cards (ContradictionDetector triggers)
            "segment_task": MetaCardType.SEGMENT_TASK,        # TRIZ #1
            "define_criteria": MetaCardType.DEFINE_CRITERIA,  # TRIZ #10
            "list_instead": MetaCardType.LIST_INSTEAD,        # TRIZ #13
            "resource_depletion": MetaCardType.SEGMENT_TASK,  # Alias
            "blind_reading": MetaCardType.DEFINE_CRITERIA,    # Alias
            "repeated_failure": MetaCardType.LIST_INSTEAD,    # Alias
            # R4: Kanban constraint cards
            "wip_overflow": MetaCardType.WIP_OVERFLOW
        }

        card_type = type_mapping.get(failure_type.lower())
        if not card_type:
            return None

        # Check if this specific card type is enabled
        if not self._is_card_type_enabled(card_type):
            return None

        # Get template
        template = CARD_TEMPLATES.get(card_type)
        if not template:
            return None

        # Create card instance
        card = MetaCard(
            card_type=card_type,
            card_id=f"{card_type.value}_{len(self.cards_issued) + 1}",
            title=template["title"],
            instruction=template["instruction"],
            rationale=template["rationale"],
            example=template["example"],
            anti_pattern=template["anti_pattern"],
            trigger_context=context
        )

        # Add context-specific hints
        card = self._enrich_with_context(card, context)

        self.cards_issued.append(card)
        return card

    def _enrich_with_context(
        self,
        card: MetaCard,
        context: Dict[str, Any]
    ) -> MetaCard:
        """Add context-specific hints to the card."""

        # For repetition loops, mention the repeated action
        if card.card_type == MetaCardType.ESCAPE_LOOP:
            repeated = context.get("repeated_action", "")
            if repeated:
                card.pcg_hint = f"You've repeated '{repeated[:50]}...' multiple times."

        # For circular patterns, mention the cycle
        elif card.card_type == MetaCardType.COMPASS:
            pattern = context.get("pattern_sequence", [])
            if pattern:
                card.pcg_hint = f"Cycle detected: {' → '.join(pattern[:4])} → ..."

        # For errors, include error message
        elif card.card_type == MetaCardType.ERROR_ANALYSIS:
            error_msg = context.get("last_error", "")
            if error_msg:
                card.pcg_hint = f"Last error: {error_msg[:100]}..."

        # For stalls, mention turns without progress
        elif card.card_type == MetaCardType.PROGRESS_BOOST:
            stall_turns = context.get("stall_turns", 0)
            if stall_turns:
                card.pcg_hint = f"No new evidence added for {stall_turns} turns."

        # For exhaustion, summarize tried approaches
        elif card.card_type == MetaCardType.STRATEGIC_RESET:
            tried = context.get("approaches_tried", [])
            if tried:
                card.pcg_hint = f"Tried: {', '.join(tried[:5])}"

        # === R4: TRIZ Card Enrichment ===

        # For segmentation, mention the files being juggled
        elif card.card_type == MetaCardType.SEGMENT_TASK:
            files = context.get("files_accessed", [])
            if files:
                card.pcg_hint = f"Files juggled: {', '.join(files[:3])}... Process ONE at a time."
            else:
                card.pcg_hint = "Break task into atomic steps: Read -> Extract -> Record -> Next."

        # For criteria definition, mention what was read without criteria
        elif card.card_type == MetaCardType.DEFINE_CRITERIA:
            reads = context.get("recent_reads", 0)
            truths = context.get("validated_truths", 0)
            card.pcg_hint = f"You've read {reads} times but only recorded {truths} facts. Define what you're looking for."

        # For inversion, mention the failed guesses
        elif card.card_type == MetaCardType.LIST_INSTEAD:
            failed_files = context.get("failed_file_names", [])
            if failed_files:
                card.pcg_hint = f"Failed to find: {', '.join(failed_files[:3])}. Try 'ls' first."
            else:
                card.pcg_hint = "Stop guessing. List the directory to see what exists."

        # For WIP overflow, show current open threads
        elif card.card_type == MetaCardType.WIP_OVERFLOW:
            wip_count = context.get("wip_count", 0)
            wip_limit = context.get("wip_limit", 2)
            open_questions = context.get("open_questions", [])
            card.pcg_hint = f"WIP: {wip_count}/{wip_limit}. Open: {', '.join(open_questions[:2])}. Close one first."

        return card

    def _is_card_type_enabled(self, card_type: MetaCardType) -> bool:
        """Check if a specific card type is enabled in config."""
        # R3: Reactive cards
        if card_type == MetaCardType.ESCAPE_LOOP:
            return self._config.enable_escape_loop
        elif card_type == MetaCardType.COMPASS:
            return self._config.enable_compass
        elif card_type == MetaCardType.ERROR_ANALYSIS:
            return self._config.enable_error_analysis
        elif card_type == MetaCardType.PROGRESS_BOOST:
            return self._config.enable_progress_boost
        elif card_type == MetaCardType.STRATEGIC_RESET:
            return self._config.enable_strategic_reset
        # R4: TRIZ cards - check for triz_enabled attribute or default True
        elif card_type in [MetaCardType.SEGMENT_TASK,
                           MetaCardType.DEFINE_CRITERIA,
                           MetaCardType.LIST_INSTEAD]:
            return getattr(self._config, 'enable_triz_cards', True)
        # R4: Kanban cards
        elif card_type == MetaCardType.WIP_OVERFLOW:
            return getattr(self._config, 'enable_wip_cards', True)
        return True  # Default: enabled

    def record_outcome(self, card_id: str, success: bool):
        """Record whether a card helped the model succeed."""
        if card_id not in self.effectiveness:
            self.effectiveness[card_id] = []
        self.effectiveness[card_id].append(success)

        # Update card success rate
        for card in self.cards_issued:
            if card.card_id == card_id:
                outcomes = self.effectiveness[card_id]
                card.success_rate = sum(outcomes) / len(outcomes)
                card.times_used = len(outcomes)
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all cards."""
        stats = {
            "total_cards_issued": len(self.cards_issued),
            "by_type": {},
            "overall_success_rate": 0.0
        }

        # Count by type
        type_counts: Dict[str, int] = {}
        type_successes: Dict[str, int] = {}

        for card in self.cards_issued:
            t = card.card_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

            outcomes = self.effectiveness.get(card.card_id, [])
            type_successes[t] = type_successes.get(t, 0) + sum(outcomes)

        for t, count in type_counts.items():
            stats["by_type"][t] = {
                "count": count,
                "successes": type_successes.get(t, 0),
                "rate": type_successes.get(t, 0) / count if count > 0 else 0.0
            }

        # Overall success rate
        all_outcomes = [o for outcomes in self.effectiveness.values() for o in outcomes]
        if all_outcomes:
            stats["overall_success_rate"] = sum(all_outcomes) / len(all_outcomes)

        return stats


# === Demo/Test Code ===

def demo_meta_cards():
    """Demonstrate meta-card selection and formatting."""
    print("=" * 70)
    print("META-CARDS DEMONSTRATION")
    print("Strategic interventions for stuck LLMs")
    print("=" * 70)

    selector = MetaCardSelector()

    # Simulate failure scenarios from Olympics
    failures = [
        {
            "type": "repetition_loop",
            "context": {
                "repeated_action": "find . -name '*.py' -exec cat {} \\;",
                "repetitions": 4
            }
        },
        {
            "type": "circular_pattern",
            "context": {
                "pattern_sequence": ["ls", "cat README", "ls", "cat README"],
                "cycle_length": 2
            }
        },
        {
            "type": "explicit_error",
            "context": {
                "last_error": "FileNotFoundError: No such file: /project/config.yaml",
                "error_count": 3
            }
        },
        {
            "type": "progress_stall",
            "context": {
                "stall_turns": 5,
                "last_pcg_size": 3
            }
        }
    ]

    print("\n📋 GENERATED CARDS:\n")

    for failure in failures:
        card = selector.select_card(failure["type"], failure["context"])
        if card:
            print(card.to_prompt_injection())
            print()

    # Simulate outcomes
    for i, card in enumerate(selector.cards_issued):
        # Simulate: escape_loop and error_analysis help more
        success = card.card_type in [MetaCardType.ESCAPE_LOOP, MetaCardType.ERROR_ANALYSIS]
        selector.record_outcome(card.card_id, success)

    # Statistics
    stats = selector.get_statistics()
    print("\n📊 CARD EFFECTIVENESS:\n")
    print(f"Total cards issued: {stats['total_cards_issued']}")
    print(f"Overall success rate: {stats['overall_success_rate']:.1%}")
    print("\nBy type:")
    for t, data in stats["by_type"].items():
        print(f"  {t}: {data['successes']}/{data['count']} ({data['rate']:.0%})")


if __name__ == "__main__":
    demo_meta_cards()
