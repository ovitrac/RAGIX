#!/usr/bin/env python3
"""
Strategic Advisor — Round 4 Meta-Cognitive Layer
=================================================

**ROADMAP STATUS: R4 IMPLEMENTATION**

Evolved from "Guided Dare" to "Strategic Advisor" with:
- TRIZ contradiction detection
- Kanban WIP management
- Focus View (compressed DONE column for 3B models)

Integration Flow:
    PCG State → Safety Check → WIP Check → TRIZ Check → Failure Check → Action Menu
                     ↓            ↓            ↓             ↓
              Token Leak    WIP_OVERFLOW   TRIZ Card    Reactive Card

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2025-12-23
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from pcg import PCG, Node, NodeType, Status

# Import from local modules
from meta_cards import MetaCard, MetaCardSelector, MetaCardType


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StrategicAdvisorConfig:
    """Configuration for Strategic Advisor (R4)."""

    # Kanban WIP limits
    wip_limit: int = 2                      # Max concurrent actions (default)
    wip_warning_threshold: int = 1          # Warn at this level

    # TRIZ card enablement
    enable_triz: bool = True
    enable_segment_task: bool = True        # TRIZ #1
    enable_define_criteria: bool = True     # TRIZ #10
    enable_list_instead: bool = True        # TRIZ #13

    # Kanban enablement
    enable_kanban: bool = True
    enable_wip_overflow: bool = True

    # Contradiction detection thresholds
    repeated_failure_threshold: int = 2     # Failures before INVERSION
    blind_reading_threshold: int = 2        # Reads without criteria
    resource_depletion_threshold: int = 2   # Multi-file commands

    # Focus View (context optimization for 3B models)
    focus_view_enabled: bool = True
    max_done_items_shown: int = 3           # Compress DONE column
    max_doing_items_shown: int = 5          # Full DOING column
    max_blocked_items_shown: int = 5        # Full BLOCKED column

    # Deepseek special handling
    deepseek_triz_turn_threshold: int = 15  # Only offer TRIZ if Turn > N


# Model-specific WIP overrides (based on Round 3 analysis)
MODEL_WIP_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "wip_limit": 2,
        "enable_triz": True,
        "enable_kanban": True
    },
    "deepseek-r1:14b": {
        # Champion: minimal intervention
        "wip_limit": 3,
        "enable_triz": False,  # Only if Turn > 15
        "enable_kanban": True
    },
    "llama3.2:3b": {
        # Rehabilitated: maintain strict oversight
        "wip_limit": 2,
        "enable_triz": True,
        "enable_kanban": True
    },
    "granite3.1-moe:3b": {
        # Stable specialist: give more freedom
        "wip_limit": 3,
        "enable_triz": True,
        "enable_kanban": True
    },
    "mistral:latest": {
        # Stubborn: aggressive constraints (straightjacket)
        "wip_limit": 1,           # Force single-thread
        "enable_triz": True,
        "enable_kanban": True
    },
    "ibm/granite4:32b-a9b-h": {
        # Round 5 diagnosis: Protocol Mismatch, not Policy Overfitting
        # Granite 4 uses structured tool calling — see tool_call_adapter.py
        # Standard WIP limit is fine with proper interface
        "wip_limit": 2,
        "enable_triz": True,
        "enable_kanban": True
    },
    "phi3:latest": {
        # Terminal agnosia: retired, but keep config for testing
        "wip_limit": 2,
        "enable_triz": True,
        "enable_kanban": True
    }
}


def get_model_wip_limit(model: str) -> int:
    """Get WIP limit for a specific model."""
    profile = MODEL_WIP_PROFILES.get(model, MODEL_WIP_PROFILES["default"])
    return profile.get("wip_limit", 2)


def is_triz_enabled_for_model(model: str, current_turn: int = 0) -> bool:
    """Check if TRIZ cards are enabled for this model."""
    profile = MODEL_WIP_PROFILES.get(model, MODEL_WIP_PROFILES["default"])

    # Special handling for Deepseek: only enable TRIZ after turn threshold
    if "deepseek" in model.lower():
        threshold = StrategicAdvisorConfig().deepseek_triz_turn_threshold
        return current_turn > threshold

    return profile.get("enable_triz", True)


# =============================================================================
# Kanban PCG View (with Focus View optimization)
# =============================================================================

class KanbanColumn(Enum):
    """Kanban board columns."""
    BACKLOG = "backlog"     # Questions, Goals (OPEN)
    DOING = "doing"         # Actions (PROPOSED/EXECUTING)
    BLOCKED = "blocked"     # Truths (UNDECIDABLE)
    DONE = "done"           # Truths (VALIDATED)


@dataclass
class KanbanItem:
    """A single item on the Kanban board."""
    node_id: str
    summary: str
    column: KanbanColumn
    priority: int = 0  # Lower = higher priority


class KanbanPCGView:
    """
    Present PCG state as a Kanban board for LLM context injection.

    Implements "Focus View" to compress DONE column for 3B context limits.
    """

    def __init__(self, wip_limit: int = 2, config: StrategicAdvisorConfig = None):
        self.wip_limit = wip_limit
        self.config = config or StrategicAdvisorConfig()
        self._columns: Dict[KanbanColumn, List[KanbanItem]] = {
            KanbanColumn.BACKLOG: [],
            KanbanColumn.DOING: [],
            KanbanColumn.BLOCKED: [],
            KanbanColumn.DONE: []
        }

    def update_from_pcg(self, pcg: "PCG"):
        """Update Kanban view from PCG state."""
        # Clear previous state
        for col in self._columns:
            self._columns[col] = []

        # Categorize nodes
        for node_id, node in pcg.nodes.items():
            item = self._node_to_kanban_item(node)
            if item:
                self._columns[item.column].append(item)

    def _node_to_kanban_item(self, node: "Node") -> Optional[KanbanItem]:
        """Convert PCG node to Kanban item."""
        from pcg import NodeType, Status

        node_type = node.node_type
        status = getattr(node, 'status', None)

        # Questions and Goals -> BACKLOG
        if node_type == NodeType.QUESTION:
            if status == Status.OPEN:
                return KanbanItem(
                    node_id=node.id,
                    summary=f"Q: {node.text[:40]}..." if len(node.text) > 40 else f"Q: {node.text}",
                    column=KanbanColumn.BACKLOG
                )

        elif node_type == NodeType.GOAL:
            if status == Status.OPEN:
                return KanbanItem(
                    node_id=node.id,
                    summary=f"G: {node.text[:40]}..." if len(node.text) > 40 else f"G: {node.text}",
                    column=KanbanColumn.BACKLOG
                )

        # Actions -> DOING
        elif node_type == NodeType.ACTION:
            if status in [Status.PROPOSED, Status.EXECUTING]:
                return KanbanItem(
                    node_id=node.id,
                    summary=f"Act: {node.intent[:35]}..." if len(node.intent) > 35 else f"Act: {node.intent}",
                    column=KanbanColumn.DOING
                )

        # Truths -> BLOCKED or DONE
        elif node_type == NodeType.TRUTH:
            if status == Status.UNDECIDABLE:
                return KanbanItem(
                    node_id=node.id,
                    summary=f"T?: {node.text[:35]}..." if len(node.text) > 35 else f"T?: {node.text}",
                    column=KanbanColumn.BLOCKED
                )
            elif status == Status.VALIDATED:
                return KanbanItem(
                    node_id=node.id,
                    summary=f"T: {node.text[:35]}..." if len(node.text) > 35 else f"T: {node.text}",
                    column=KanbanColumn.DONE
                )

        return None

    def render(self, focus_view: bool = True) -> str:
        """
        Render PCG as Kanban board for prompt injection.

        Args:
            focus_view: If True, compress DONE column (for 3B models)
        """
        doing = self._columns[KanbanColumn.DOING]
        blocked = self._columns[KanbanColumn.BLOCKED]
        done = self._columns[KanbanColumn.DONE]
        backlog = self._columns[KanbanColumn.BACKLOG]

        # Build board
        board = """
+------------------+------------------+------------------+------------------+
|     BACKLOG      |      DOING       |     BLOCKED      |       DONE       |
|   (Questions)    |    (Actions)     |   (Need Help)    |     (Proven)     |
+------------------+------------------+------------------+------------------+
"""
        # Render each column
        max_rows = max(
            len(backlog),
            len(doing),
            len(blocked),
            len(done) if not focus_view else min(len(done), self.config.max_done_items_shown)
        )

        for i in range(max(max_rows, 1)):
            row = "|"
            # BACKLOG
            if i < len(backlog):
                row += f" {backlog[i].summary[:16]:<16} |"
            else:
                row += " " * 17 + "|"

            # DOING
            if i < len(doing):
                row += f" {doing[i].summary[:16]:<16} |"
            else:
                row += " " * 17 + "|"

            # BLOCKED
            if i < len(blocked):
                row += f" {blocked[i].summary[:16]:<16} |"
            else:
                row += " " * 17 + "|"

            # DONE (with Focus View compression)
            if focus_view and len(done) > self.config.max_done_items_shown:
                if i < self.config.max_done_items_shown:
                    row += f" {done[i].summary[:16]:<16} |"
                elif i == self.config.max_done_items_shown:
                    remaining = len(done) - self.config.max_done_items_shown
                    row += f" (+{remaining} more)      |"
                else:
                    row += " " * 17 + "|"
            else:
                if i < len(done):
                    row += f" {done[i].summary[:16]:<16} |"
                else:
                    row += " " * 17 + "|"

            board += row + "\n"

        board += "+------------------+------------------+------------------+------------------+"

        # Add WIP status
        wip_count = len(doing)
        if wip_count >= self.wip_limit:
            board += f"\n!! WIP LIMIT REACHED ({wip_count}/{self.wip_limit}) - Close an action before opening new ones"
        elif wip_count > 0:
            board += f"\n   WIP: {wip_count}/{self.wip_limit}"

        return board

    def is_wip_exceeded(self) -> bool:
        """Check if WIP limit is exceeded."""
        return len(self._columns[KanbanColumn.DOING]) >= self.wip_limit

    def get_wip_count(self) -> int:
        """Get current WIP count."""
        return len(self._columns[KanbanColumn.DOING])

    def get_open_questions(self) -> List[str]:
        """Get list of open question summaries."""
        return [item.summary for item in self._columns[KanbanColumn.BACKLOG]]

    def get_done_count(self) -> int:
        """Get count of validated truths."""
        return len(self._columns[KanbanColumn.DONE])


# =============================================================================
# Contradiction Detector (TRIZ trigger)
# =============================================================================

@dataclass
class TurnRecord:
    """Record of a single turn for history analysis."""
    turn: int
    action: str
    output: str
    is_error: bool
    is_repeat: bool
    points: int = 0


class ContradictionDetector:
    """
    Detect TRIZ-style contradictions in model behavior.

    Triggers:
    - repeated_failure → LIST_INSTEAD (TRIZ #13)
    - blind_reading → DEFINE_CRITERIA (TRIZ #10)
    - resource_depletion → SEGMENT_TASK (TRIZ #1)
    """

    def __init__(self, config: StrategicAdvisorConfig = None):
        self.config = config or StrategicAdvisorConfig()

    def detect(self, history: List[TurnRecord], kanban: KanbanPCGView) -> Optional[Dict[str, Any]]:
        """
        Detect contradictions that trigger TRIZ cards.

        Returns:
            Dict with contradiction type and context, or None
        """
        # Pattern 1: Tried X twice, failed twice → INVERSION needed
        result = self._detect_repeated_failure(history)
        if result:
            return result

        # Pattern 2: Reading without criteria → PRIOR_ACTION needed
        result = self._detect_blind_reading(history, kanban)
        if result:
            return result

        # Pattern 3: Multiple files, context loss → SEGMENTATION needed
        result = self._detect_resource_depletion(history)
        if result:
            return result

        return None

    def _detect_repeated_failure(self, history: List[TurnRecord]) -> Optional[Dict[str, Any]]:
        """
        Detect: Tried same approach twice, failed twice (FileNotFound pattern).

        Returns TRIZ #13 (Inversion) trigger.
        """
        threshold = self.config.repeated_failure_threshold
        if len(history) < threshold:
            return None

        recent = history[-threshold:]

        # Check for FileNotFoundError pattern
        fnf_errors = []
        for t in recent:
            if t.is_error:
                output_lower = t.output.lower()
                if "not found" in output_lower or "no such file" in output_lower:
                    fnf_errors.append(t)

        if len(fnf_errors) >= threshold:
            # Extract failed file names
            failed_files = []
            for t in fnf_errors:
                # Try to extract file name from action
                action = t.action
                for keyword in ["cat ", "read ", "READ_FILE("]:
                    if keyword in action:
                        start = action.find(keyword) + len(keyword)
                        end = action.find(")", start) if ")" in action[start:] else len(action)
                        file_part = action[start:end].strip().strip('"').strip("'")
                        if file_part:
                            failed_files.append(file_part[:30])
                        break

            return {
                "contradiction_type": "repeated_failure",
                "card_type": "list_instead",
                "context": {
                    "failed_file_names": failed_files,
                    "failure_count": len(fnf_errors)
                }
            }

        return None

    def _detect_blind_reading(self, history: List[TurnRecord],
                               kanban: KanbanPCGView) -> Optional[Dict[str, Any]]:
        """
        Detect: Reading files without stated criteria.

        Returns TRIZ #10 (Prior Action) trigger.
        """
        threshold = self.config.blind_reading_threshold
        if len(history) < threshold:
            return None

        # Count recent reads
        recent_reads = 0
        for t in history[-5:]:
            action_upper = t.action.upper()
            if "READ" in action_upper or "CAT " in action_upper:
                recent_reads += 1

        if recent_reads < threshold:
            return None

        # Check if there are validated truths (facts recorded)
        validated_truths = kanban.get_done_count()

        # Reading a lot but not recording truths → blind reading
        if recent_reads > validated_truths + 1:  # Allow 1 read ahead
            return {
                "contradiction_type": "blind_reading",
                "card_type": "define_criteria",
                "context": {
                    "recent_reads": recent_reads,
                    "validated_truths": validated_truths
                }
            }

        return None

    def _detect_resource_depletion(self, history: List[TurnRecord]) -> Optional[Dict[str, Any]]:
        """
        Detect: Processing multiple files, losing context.

        Returns TRIZ #1 (Segmentation) trigger.
        """
        threshold = self.config.resource_depletion_threshold
        if len(history) < threshold:
            return None

        # Look for multi-file commands or rapid file switching
        multi_file_cmds = 0
        files_accessed = []

        for t in history[-5:]:
            action = t.action

            # Multi-file patterns
            if action.count("file") > 1 or "*.txt" in action or action.count(";") > 1:
                multi_file_cmds += 1

            # Track files accessed
            for keyword in ["cat ", "read ", "READ_FILE(", "head ", "tail "]:
                if keyword.lower() in action.lower():
                    # Extract file name
                    idx = action.lower().find(keyword.lower())
                    file_part = action[idx + len(keyword):].split()[0] if action[idx + len(keyword):] else ""
                    file_part = file_part.strip('"').strip("'").strip(")")
                    if file_part and file_part not in files_accessed:
                        files_accessed.append(file_part[:25])

        # Too many files in rapid succession
        if len(files_accessed) >= 3 or multi_file_cmds >= threshold:
            return {
                "contradiction_type": "resource_depletion",
                "card_type": "segment_task",
                "context": {
                    "files_accessed": files_accessed,
                    "multi_file_commands": multi_file_cmds
                }
            }

        return None


# =============================================================================
# Strategic Advisor (Main Class)
# =============================================================================

class StrategicAdvisor:
    """
    Strategic Advisor (evolved from Guided Dare).

    Implements the 4-layer check:
    1. Safety Check (token leak detection)
    2. Kanban WIP limit
    3. TRIZ contradiction detection
    4. Failure pattern detection (R3 cards)
    """

    def __init__(self, model: str = "default", config: StrategicAdvisorConfig = None):
        self.model = model
        self.config = config or StrategicAdvisorConfig()

        # Get model-specific WIP limit
        self.wip_limit = get_model_wip_limit(model)

        # Initialize components
        self.kanban = KanbanPCGView(wip_limit=self.wip_limit, config=self.config)
        self.contradiction_detector = ContradictionDetector(config=self.config)
        self.card_selector = MetaCardSelector()

        # Track current turn for Deepseek special handling
        self.current_turn = 0

    def update_pcg(self, pcg: "PCG"):
        """Update Kanban view from PCG."""
        self.kanban.update_from_pcg(pcg)

    def advise(self, history: List[TurnRecord], pcg: "PCG" = None) -> Optional[MetaCard]:
        """
        Get strategic advice based on current state.

        Priority:
        1. Safety Check (token leak - handled in scored_mode.py)
        2. WIP limit → WIP_OVERFLOW card
        3. TRIZ contradiction → TRIZ card
        4. Failure pattern → Reactive card (delegated to FailureDetector)

        Args:
            history: List of TurnRecord objects
            pcg: Optional PCG for state analysis

        Returns:
            MetaCard or None
        """
        # Update turn counter
        if history:
            self.current_turn = max(t.turn for t in history)

        # Update Kanban if PCG provided
        if pcg:
            self.update_pcg(pcg)

        # === Layer 1: Safety Check ===
        # Token leak detection is handled in scored_mode.py (strip_reasoning_tokens)
        # We skip it here as it's a pre-processing step

        # === Layer 2: Kanban WIP Check ===
        if self.config.enable_kanban and self.kanban.is_wip_exceeded():
            return self.card_selector.select_card(
                "wip_overflow",
                {
                    "wip_count": self.kanban.get_wip_count(),
                    "wip_limit": self.wip_limit,
                    "open_questions": self.kanban.get_open_questions()
                }
            )

        # === Layer 3: TRIZ Contradiction Check ===
        if self.config.enable_triz and is_triz_enabled_for_model(self.model, self.current_turn):
            contradiction = self.contradiction_detector.detect(history, self.kanban)
            if contradiction:
                card_type = contradiction["card_type"]
                context = contradiction["context"]
                context["contradiction_type"] = contradiction["contradiction_type"]
                return self.card_selector.select_card(card_type, context)

        # === Layer 4: Failure Pattern Check ===
        # This is delegated to FailureDetector in scored_mode.py
        # Strategic Advisor focuses on proactive (TRIZ/Kanban) interventions

        return None

    def get_context_injection(self, focus_view: bool = True) -> str:
        """
        Get full context to inject into LLM prompt.

        Args:
            focus_view: Compress DONE column (recommended for 3B models)

        Returns:
            Kanban board string for prompt injection
        """
        if not self.config.enable_kanban:
            return ""

        return self.kanban.render(focus_view=focus_view)

    def get_statistics(self) -> Dict[str, Any]:
        """Get advisor statistics."""
        return {
            "model": self.model,
            "wip_limit": self.wip_limit,
            "current_wip": self.kanban.get_wip_count(),
            "triz_enabled": is_triz_enabled_for_model(self.model, self.current_turn),
            "current_turn": self.current_turn,
            "cards_issued": self.card_selector.get_statistics()
        }


# =============================================================================
# Demo/Test Code
# =============================================================================

def demo_strategic_advisor():
    """Demonstrate Strategic Advisor functionality."""
    print("=" * 70)
    print("STRATEGIC ADVISOR DEMONSTRATION (Round 4)")
    print("TRIZ + Kanban Integration")
    print("=" * 70)

    # Test different model profiles
    models = ["mistral:latest", "llama3.2:3b", "deepseek-r1:14b", "granite3.1-moe:3b"]

    for model in models:
        print(f"\n--- Model: {model} ---")
        advisor = StrategicAdvisor(model=model)
        print(f"  WIP Limit: {advisor.wip_limit}")
        print(f"  TRIZ Enabled: {is_triz_enabled_for_model(model)}")

    # Simulate contradiction detection
    print("\n--- Contradiction Detection Demo ---")
    detector = ContradictionDetector()

    # Simulate FileNotFound loop (triggers LIST_INSTEAD)
    history = [
        TurnRecord(turn=1, action='READ_FILE("config.yaml")', output="FileNotFoundError: config.yaml", is_error=True, is_repeat=False),
        TurnRecord(turn=2, action='READ_FILE("settings.yaml")', output="FileNotFoundError: settings.yaml", is_error=True, is_repeat=False),
    ]

    kanban = KanbanPCGView(wip_limit=2)
    result = detector.detect(history, kanban)
    if result:
        print(f"\nContradiction Detected: {result['contradiction_type']}")
        print(f"Recommended Card: {result['card_type']}")
        print(f"Context: {result['context']}")

    # Render empty Kanban board
    print("\n--- Kanban Board (Empty) ---")
    print(kanban.render())


if __name__ == "__main__":
    demo_strategic_advisor()
