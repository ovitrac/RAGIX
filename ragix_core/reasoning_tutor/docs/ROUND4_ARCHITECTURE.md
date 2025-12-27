# Round 4 Architecture: TRIZ + Kanban Strategic Advisor

**Date:** 2025-12-23
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Status:** DESIGN PROPOSAL — Awaiting Approval

---

## Executive Summary

Round 4 introduces a paradigm shift from **reactive tool assistance** to **proactive strategic coaching**. The integration of TRIZ inventive principles and Kanban WIP management addresses the "Complexity Wall" identified in academic critique.

### Key Changes

| Component | Round 3 (Current) | Round 4 (Proposed) |
|-----------|-------------------|-------------------|
| Cards | Reactive tools (grep, wc) | **Strategic TRIZ Meta-Cards** |
| PCG View | Log/history | **Kanban Board** (TO DO/DOING/BLOCKED/DONE) |
| Recommender | Guided Dare | **Strategic Advisor** |
| Constraints | None | **WIP Limits** (max 2 concurrent actions) |

---

## 1. TRIZ Meta-Cards Architecture

### 1.1 New Card Types

Extend `MetaCardType` enum with TRIZ-inspired strategic cards:

```python
class MetaCardType(Enum):
    # === Existing (R3) ===
    ESCAPE_LOOP = "escape_loop"           # For REPETITION_LOOP
    COMPASS = "compass"                   # For CIRCULAR_PATTERN
    ERROR_ANALYSIS = "error_analysis"     # For EXPLICIT_ERROR
    PROGRESS_BOOST = "progress_boost"     # For PROGRESS_STALL
    STRATEGIC_RESET = "strategic_reset"   # For EXHAUSTION

    # === NEW: TRIZ Meta-Cards (R4) ===
    SEGMENT_TASK = "segment_task"         # TRIZ #1: Segmentation
    DEFINE_CRITERIA = "define_criteria"   # TRIZ #10: Prior Action
    LIST_INSTEAD = "list_instead"         # TRIZ #13: Inversion
    WIP_OVERFLOW = "wip_overflow"         # Kanban: WIP limit exceeded
```

### 1.2 TRIZ Card Templates

#### SEGMENT_TASK (Principle #1: Segmentation)

**Trigger:** Resource Depletion (model reads multiple files, loses context)

```python
MetaCardType.SEGMENT_TASK: {
    "title": "Divide & Conquer",
    "instruction": "SEGMENT the task. Process ONE file at a time, "
                   "recording facts to PCG before moving to the next.",
    "rationale": "You're losing context by processing too much at once. "
                 "Break the problem into atomic steps with checkpoints.",
    "example": "Instead of 'cat file1.txt file2.txt file3.txt', try: "
               "(1) Read file1 → Extract fact → Record T1 → "
               "(2) Read file2 → Extract fact → Record T2 → etc.",
    "anti_pattern": "Do NOT read multiple files in one command. "
                    "Do NOT proceed without recording each fact to PCG.",
    "triz_principle": 1,  # Segmentation
    "complexity_target": "B06_Memory"
}
```

#### DEFINE_CRITERIA (Principle #10: Prior Action)

**Trigger:** Undecidable CHECK (model doesn't know what to verify)

```python
MetaCardType.DEFINE_CRITERIA: {
    "title": "Define Before You Search",
    "instruction": "ASSERT your success criteria BEFORE reading. "
                   "What exactly will prove or disprove your hypothesis?",
    "rationale": "You're reading without knowing what to look for. "
                 "Pre-computation prevents wasted exploration.",
    "example": "Before 'cat config.yaml', state: "
               "'I am looking for key=X where value matches pattern Y. "
               "If found, T(config_valid). If not found, T(config_missing).'",
    "anti_pattern": "Do NOT read files 'to see what's there'. "
                    "Do NOT proceed without explicit success criteria.",
    "triz_principle": 10,  # Prior Action
    "complexity_target": "B04_Verification"
}
```

#### LIST_INSTEAD (Principle #13: Inversion)

**Trigger:** Perseveration on "File not found" or similar guessing failures

```python
MetaCardType.LIST_INSTEAD: {
    "title": "Invert Your Approach",
    "instruction": "If you cannot find X by guessing, ENUMERATE the space. "
                   "List the directory to see what actually exists.",
    "rationale": "You're guessing file names that don't exist. "
                 "Inversion: discover what IS there, then select.",
    "example": "Instead of 'cat config.yaml' (FileNotFoundError), try: "
               "'ls -la' to see actual files, then read the correct one.",
    "anti_pattern": "Do NOT keep guessing file names. "
                    "Do NOT assume the file exists without verification.",
    "triz_principle": 13,  # Inversion
    "complexity_target": "B01_FileNotFound"
}
```

#### WIP_OVERFLOW (Kanban Constraint)

**Trigger:** More than 2 Actions in DOING state

```python
MetaCardType.WIP_OVERFLOW: {
    "title": "Close Before Opening",
    "instruction": "You have too many open threads. FINISH one investigation "
                   "before starting another.",
    "rationale": "Parallel exploration causes context loss. "
                 "WIP limit: maximum 2 concurrent actions.",
    "example": "You have Q1 (config location) and Q2 (version check) open. "
               "Complete Q1 with T(answer) before exploring Q2.",
    "anti_pattern": "Do NOT open a third question/action. "
                    "Do NOT switch context without recording findings.",
    "kanban_rule": "wip_limit",
    "wip_max": 2
}
```

---

## 2. Kanban PCG Visualization

### 2.1 Column Mapping

Map PCG node types to Kanban columns:

| Kanban Column | PCG Node Type | Status Filter | Description |
|---------------|---------------|---------------|-------------|
| **BACKLOG** | Question (Q), Goal (G) | status=OPEN | What needs to be answered |
| **TO DO** | Action (Act) | status=PROPOSED | Planned but not executed |
| **DOING** | Action (Act) | status=EXECUTING | Currently running |
| **BLOCKED** | Truth (T) | status=UNDECIDABLE | Needs Dare/Evidence |
| **DONE** | Truth (T) | status=VALIDATED | Proven facts |

### 2.2 Kanban Board Renderer

```python
class KanbanPCGView:
    """
    Present PCG state as a Kanban board for LLM context injection.
    """

    def __init__(self, pcg: PCG, wip_limit: int = 2):
        self.pcg = pcg
        self.wip_limit = wip_limit

    def render(self) -> str:
        """Render PCG as Kanban board for prompt injection."""
        columns = self._categorize_nodes()

        board = """
╔══════════════════════════════════════════════════════════════════╗
║                    📋 KANBAN PROGRESS BOARD                       ║
╠════════════════╦════════════════╦════════════════╦═══════════════╣
║    BACKLOG     ║     DOING      ║    BLOCKED     ║     DONE      ║
║  (Questions)   ║   (Actions)    ║  (Need Help)   ║   (Proven)    ║
╠════════════════╬════════════════╬════════════════╬═══════════════╣
"""
        # Render columns side by side
        board += self._render_columns(columns)
        board += """
╚════════════════╩════════════════╩════════════════╩═══════════════╝
"""
        # Add WIP warning if needed
        if len(columns["DOING"]) >= self.wip_limit:
            board += f"\n⚠️  WIP LIMIT REACHED ({len(columns['DOING'])}/{self.wip_limit}) — Close an action before opening new ones\n"

        return board

    def _categorize_nodes(self) -> Dict[str, List[Node]]:
        """Categorize PCG nodes into Kanban columns."""
        columns = {
            "BACKLOG": [],  # Q, G with status OPEN
            "DOING": [],    # Act with status PROPOSED/EXECUTING
            "BLOCKED": [],  # T with status UNDECIDABLE
            "DONE": []      # T with status VALIDATED
        }

        for node in self.pcg.nodes.values():
            if node.node_type == NodeType.QUESTION:
                if node.status == Status.OPEN:
                    columns["BACKLOG"].append(node)
            elif node.node_type == NodeType.ACTION:
                if node.status in [Status.PROPOSED, Status.EXECUTING]:
                    columns["DOING"].append(node)
            elif node.node_type == NodeType.TRUTH:
                if node.status == Status.UNDECIDABLE:
                    columns["BLOCKED"].append(node)
                elif node.status == Status.VALIDATED:
                    columns["DONE"].append(node)

        return columns

    def is_wip_exceeded(self) -> bool:
        """Check if WIP limit is exceeded."""
        columns = self._categorize_nodes()
        return len(columns["DOING"]) >= self.wip_limit

    def get_wip_count(self) -> int:
        """Get current WIP count."""
        columns = self._categorize_nodes()
        return len(columns["DOING"])
```

---

## 3. Strategic Advisor (Renamed from Guided Dare)

### 3.1 Decision Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGIC ADVISOR WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                                              │
│   │ PCG State    │                                              │
│   │ Check        │                                              │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐     YES    ┌───────────────────┐            │
│   │ WIP Limit    │ ─────────► │ Issue WIP_OVERFLOW │            │
│   │ Exceeded?    │            │ Card               │            │
│   └──────┬───────┘            └───────────────────┘            │
│          │ NO                                                    │
│          ▼                                                       │
│   ┌──────────────┐     YES    ┌───────────────────┐            │
│   │ Contradiction│ ─────────► │ Issue TRIZ Card    │            │
│   │ Detected?    │            │ (SEGMENT/CRITERIA/ │            │
│   └──────┬───────┘            │  LIST_INSTEAD)     │            │
│          │ NO                 └───────────────────┘            │
│          ▼                                                       │
│   ┌──────────────┐     YES    ┌───────────────────┐            │
│   │ Failure      │ ─────────► │ Issue Reactive     │            │
│   │ Pattern?     │            │ Card (R3 cards)    │            │
│   └──────┬───────┘            └───────────────────┘            │
│          │ NO                                                    │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │ Proceed to   │                                              │
│   │ Action Menu  │                                              │
│   └──────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Contradiction Detection

```python
class ContradictionDetector:
    """
    Detect TRIZ-style contradictions in model behavior.
    """

    def detect(self, history: List[TurnRecord], pcg: PCG) -> Optional[str]:
        """
        Detect contradictions that trigger TRIZ cards.

        Returns:
            Contradiction type or None
        """
        # Pattern: Tried X twice, failed twice → INVERSION needed
        if self._detect_repeated_failure(history, min_failures=2):
            return "repeated_failure"  # → LIST_INSTEAD

        # Pattern: Reading without criteria → PRIOR_ACTION needed
        if self._detect_blind_reading(history, pcg):
            return "blind_reading"  # → DEFINE_CRITERIA

        # Pattern: Multiple files, context loss → SEGMENTATION needed
        if self._detect_resource_depletion(history, pcg):
            return "resource_depletion"  # → SEGMENT_TASK

        return None

    def _detect_repeated_failure(self, history: List[TurnRecord],
                                  min_failures: int = 2) -> bool:
        """Detect: Tried same approach twice, failed twice."""
        if len(history) < min_failures:
            return False

        recent = history[-min_failures:]
        # Check if all recent are errors with similar commands
        errors = [t for t in recent if t.is_error]
        if len(errors) < min_failures:
            return False

        # Check for FileNotFoundError pattern
        fnf_errors = [e for e in errors if "not found" in e.output.lower()
                      or "no such file" in e.output.lower()]
        return len(fnf_errors) >= min_failures

    def _detect_blind_reading(self, history: List[TurnRecord],
                               pcg: PCG) -> bool:
        """Detect: Reading files without stated criteria."""
        recent_reads = [t for t in history[-3:]
                        if "READ" in t.action or "cat" in t.action]
        if len(recent_reads) < 2:
            return False

        # Check if there are Questions without corresponding Truth assertions
        open_questions = [n for n in pcg.nodes.values()
                          if n.node_type == NodeType.QUESTION
                          and n.status == Status.OPEN]
        validated_truths = [n for n in pcg.nodes.values()
                           if n.node_type == NodeType.TRUTH
                           and n.status == Status.VALIDATED]

        # Reading a lot but not recording truths
        return len(recent_reads) > len(validated_truths)

    def _detect_resource_depletion(self, history: List[TurnRecord],
                                    pcg: PCG) -> bool:
        """Detect: Processing multiple files, losing context."""
        # Look for multi-file commands
        multi_file_cmds = [t for t in history[-5:]
                          if t.action.count("file") > 1
                          or "*.txt" in t.action
                          or t.action.count(";") > 1]
        return len(multi_file_cmds) >= 2
```

### 3.3 Strategic Advisor Integration

```python
class StrategicAdvisor:
    """
    Strategic Advisor (evolved from Guided Dare).

    Implements the 3-layer check:
    1. Kanban WIP limit
    2. TRIZ contradiction detection
    3. Failure pattern detection (R3 cards)
    """

    def __init__(self, pcg: PCG, config: TutorConfig):
        self.pcg = pcg
        self.config = config
        self.kanban = KanbanPCGView(pcg, wip_limit=config.wip_limit)
        self.contradiction_detector = ContradictionDetector()
        self.card_selector = MetaCardSelector()
        self.failure_detector = FailureDetector(
            repetition_threshold=config.repetition_threshold,
            error_threshold=config.error_threshold,
            stall_threshold=config.stall_threshold
        )

    def advise(self, history: List[TurnRecord]) -> Optional[MetaCard]:
        """
        Get strategic advice based on current state.

        Priority:
        1. WIP limit → WIP_OVERFLOW card
        2. TRIZ contradiction → TRIZ card
        3. Failure pattern → Reactive card
        """
        # === Layer 1: Kanban WIP Check ===
        if self.kanban.is_wip_exceeded():
            return self.card_selector.select_card(
                "wip_overflow",
                {"wip_count": self.kanban.get_wip_count(),
                 "wip_limit": self.config.wip_limit}
            )

        # === Layer 2: TRIZ Contradiction Check ===
        contradiction = self.contradiction_detector.detect(history, self.pcg)
        if contradiction:
            triz_mapping = {
                "repeated_failure": "list_instead",      # TRIZ #13
                "blind_reading": "define_criteria",      # TRIZ #10
                "resource_depletion": "segment_task"     # TRIZ #1
            }
            card_type = triz_mapping.get(contradiction)
            if card_type:
                return self.card_selector.select_card(
                    card_type,
                    {"contradiction_type": contradiction}
                )

        # === Layer 3: Failure Pattern Check (R3 logic) ===
        # Use existing FailureDetector
        for turn in history[-5:]:
            failure = self.failure_detector.check(turn)
            if failure:
                return self.card_selector.select_card(
                    failure.failure_type.value,
                    failure.details
                )

        return None

    def get_context_injection(self) -> str:
        """
        Get full context to inject into LLM prompt.

        Includes:
        - Kanban board visualization
        - Any active strategic advice
        """
        return self.kanban.render()
```

---

## 4. Configuration Updates

### 4.1 New Config Fields

```python
@dataclass
class StrategicAdvisorConfig:
    """Configuration for Strategic Advisor (R4)."""

    # Kanban WIP limits
    wip_limit: int = 2                      # Max concurrent actions
    wip_warning_threshold: int = 1          # Warn at this level

    # TRIZ card enablement
    enable_segment_task: bool = True        # TRIZ #1
    enable_define_criteria: bool = True     # TRIZ #10
    enable_list_instead: bool = True        # TRIZ #13
    enable_wip_overflow: bool = True        # Kanban constraint

    # Contradiction detection thresholds
    repeated_failure_threshold: int = 2     # Failures before INVERSION
    blind_reading_threshold: int = 2        # Reads without criteria
    resource_depletion_threshold: int = 2   # Multi-file commands

    # Kanban rendering
    show_kanban_board: bool = True          # Inject board into prompt
    kanban_max_items_per_column: int = 5    # Truncate for readability
```

### 4.2 Model-Specific Overrides

Based on Round 3 findings:

```python
MODEL_STRATEGIC_PROFILES = {
    "default": {
        "wip_limit": 2,
        "enable_triz": True,
        "enable_kanban": True
    },
    "deepseek-r1:14b": {
        # Champion: minimal intervention
        "wip_limit": 3,
        "enable_triz": False,  # Doesn't need it
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
        # Stubborn: aggressive constraints
        "wip_limit": 1,           # Force single-thread
        "enable_triz": True,
        "enable_kanban": True,
        "force_simple_bash": True  # Constraint injection
    }
}
```

---

## 5. Implementation Plan

### Phase 1: Core Components (Day 1)

1. **Extend MetaCardType enum** with TRIZ cards
2. **Add TRIZ card templates** to CARD_TEMPLATES
3. **Implement KanbanPCGView** class
4. **Update MetaCardSelector** with TRIZ mapping

### Phase 2: Strategic Advisor (Day 1-2)

1. **Implement ContradictionDetector** class
2. **Create StrategicAdvisor** (refactor from Guided Dare)
3. **Add configuration dataclasses**
4. **Integrate with scored_mode.py**

### Phase 3: Testing & Validation (Day 2)

1. **Unit tests** for contradiction detection
2. **Integration tests** for full workflow
3. **Run mini-Olympics** (2 games per model)
4. **Validate WIP limits** work correctly

### Phase 4: Round 4 Olympics (Day 3)

1. **Full 30-game benchmark**
2. **Generate comparison report** (R3 vs R4)
3. **Clinical analysis**

---

## 6. Expected Outcomes

### Success Criteria

| Model | R3 Win Rate | R4 Target | Key Mechanism |
|-------|-------------|-----------|---------------|
| deepseek-r1:14b | 100% | 100% | Maintain (no intervention) |
| llama3.2:3b | 67% | 83%+ | Kanban prevents wandering |
| granite3.1-moe:3b | 33% | 50%+ | TRIZ unlocks B06 Memory |
| mistral:latest | 33% | 50%+ | WIP=1 forces simplicity |

### Failure Modes to Watch

1. **Over-constraint**: WIP=1 may slow down efficient models
2. **Card fatigue**: Too many TRIZ cards may confuse models
3. **False positives**: Contradiction detection may trigger incorrectly

---

## 7. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `meta_cards.py` | Extend | Add TRIZ card types and templates |
| `kanban_view.py` | **New** | KanbanPCGView renderer |
| `strategic_advisor.py` | **New** | StrategicAdvisor + ContradictionDetector |
| `config.py` | Extend | StrategicAdvisorConfig |
| `scored_mode.py` | Modify | Integrate StrategicAdvisor |
| `tutor.py` | Modify | Use Kanban context injection |

---

## Approval Checklist

- [ ] TRIZ card design approved
- [ ] Kanban column mapping approved
- [ ] WIP limits (default=2) approved
- [ ] Model-specific profiles approved
- [ ] Implementation timeline approved

---

*Awaiting approval to proceed with implementation.*
