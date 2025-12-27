# =============================================================================
# Tutor - Deterministic Referee
# =============================================================================
#
# The Tutor enforces game rules, executes actions, and validates proofs.
# It NEVER reasons in natural language - all decisions are deterministic.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-21)
#
# =============================================================================

"""
Tutor (Referee) for the Interpreter-Tutor game.

Responsibilities:
- CHECK protocol: Validate claims as provable/refutable/undecidable/ill-typed
- Execute actions in sandbox
- Apply rules to derive truths from observations
- Enforce constraints (policy, process)
- Track game state and scoring

The Tutor is ALWAYS deterministic. Fat LLM involvement is only for
generating new rules, which are then executed deterministically.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from pathlib import Path
import subprocess
import shlex
import re

from .pcg import (
    PCG, Node, Edge, NodeType, EdgeType, Status, ClaimKind, Soundness,
    Truth, Observation, Entity, Action, Question, Assumption, Constraint, Goal,
)
from .rules import Rule, RuleLibrary, load_rules
from .moves import (
    Move, MoveType, AssertMove, AskMove, ProposeMove,
    ChallengeMove, ReformulateMove, RespondMove, parse_move,
)


class CheckVerdict(Enum):
    """Verdict from CHECK protocol."""
    PROVABLE = "provable"           # Has valid proof
    REFUTABLE = "refutable"         # Contradicted by evidence
    UNDECIDABLE = "undecidable"     # Missing information
    ILL_TYPED = "ill-typed"         # Invalid formulation


class MoveVerdict(Enum):
    """Verdict for a proposed move."""
    LEGAL = "legal"
    ILLEGAL = "illegal"
    BLOCKED = "blocked"             # Blocked by constraint


@dataclass
class CheckResult:
    """Result of CHECK protocol."""
    verdict: CheckVerdict
    proof: list[str] = field(default_factory=list)      # Proof evidence IDs
    missing: list[str] = field(default_factory=list)    # Missing prerequisites
    reason: str = ""


@dataclass
class MoveResult:
    """Result of move validation/execution."""
    verdict: MoveVerdict
    reason: str = ""
    observation: Optional[Observation] = None           # If action executed
    truths: list[Truth] = field(default_factory=list)   # Derived truths
    entities: list[Entity] = field(default_factory=list) # Extracted entities


@dataclass
class GameState:
    """Current state of the game."""
    turn: int = 0
    score: int = 0
    validated_truths: int = 0
    open_questions: int = 0
    pending_actions: int = 0
    goal_satisfied: bool = False


# =============================================================================
# Scoring Constants
# =============================================================================

SCORE_TRUTH_DIRECT = 3      # Validated truth (direct evidence)
SCORE_TRUTH_RULE = 2        # Validated truth (rule chain)
SCORE_QUESTION_RESOLVED = 2 # Question answered
SCORE_GOOD_REFORMULATION = 1
SCORE_ILLEGAL_MOVE = -3
SCORE_POLICY_VIOLATION = -5


# =============================================================================
# Shell Executor
# =============================================================================

class ShellExecutor:
    """
    Safe shell command executor.

    Respects sandbox constraints and denylists dangerous commands.
    """

    DENYLIST = [
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+\*",
        r"mkfs",
        r"dd\s+if=",
        r"shutdown",
        r"reboot",
        r":(){ :|:& };:",  # Fork bomb
        r">\s*/dev/sd",
        r"chmod\s+-R\s+777\s+/",
    ]

    def __init__(self, sandbox_root: str = ".", timeout: int = 30):
        self.sandbox_root = Path(sandbox_root).resolve()
        self.timeout = timeout

    def is_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute."""
        for pattern in self.DENYLIST:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches denylist pattern: {pattern}"
        return True, ""

    def execute(self, command: str, mode: str = "read") -> tuple[int, str, str]:
        """
        Execute a shell command.

        Returns (return_code, stdout, stderr).
        """
        # Safety check
        is_safe, reason = self.is_safe(command)
        if not is_safe:
            return -1, "", f"BLOCKED: {reason}"

        # Execute in sandbox
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.sandbox_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"TIMEOUT: Command exceeded {self.timeout}s"
        except Exception as e:
            return -1, "", f"ERROR: {str(e)}"


# =============================================================================
# Tutor (Referee)
# =============================================================================

class Tutor:
    """
    The Tutor (Referee) for the Interpreter-Tutor game.

    All decisions are deterministic. The Tutor:
    - Validates moves (LEGAL/ILLEGAL/BLOCKED)
    - Executes actions and creates observations
    - Applies rules to derive truths
    - Implements CHECK protocol for claims
    - Tracks scoring and game state
    """

    def __init__(
        self,
        game_id: str,
        sandbox_root: str = ".",
        rules_path: Optional[str] = None,
        fat_llm_callback: Optional[Callable] = None,
    ):
        self.game_id = game_id
        self.pcg = PCG(game_id)
        self.executor = ShellExecutor(sandbox_root)
        self.rules = RuleLibrary()
        self.state = GameState()
        self.fat_llm_callback = fat_llm_callback  # For generating ad-hoc rules

        # Load rules
        if rules_path:
            self.rules.load_directory(Path(rules_path))
        else:
            # Try default location
            default_rules = Path(__file__).parent / "rules"
            if default_rules.exists():
                self.rules.load_directory(default_rules)

    # =========================================================================
    # Game Setup
    # =========================================================================

    def setup_game(self, goal: str, constraints: list[str] = None) -> None:
        """Initialize the game with goal and constraints."""
        # Add goal
        self.pcg.add_goal(goal)

        # Add default constraints
        default_constraints = [
            ("Read-only actions first; write actions only after cause is localized.", "process"),
            ("No network destruction. No deletion without confirmation.", "policy"),
            ("All truths must be provable by evidence or sound rule chains.", "epistemic"),
        ]

        for text, kind in default_constraints:
            self.pcg.add_constraint(text, kind)

        # Add custom constraints
        if constraints:
            for c in constraints:
                self.pcg.add_constraint(c, "custom")

        self.pcg.turn = 0

    # =========================================================================
    # CHECK Protocol
    # =========================================================================

    def check(self, truth: Truth) -> CheckResult:
        """
        CHECK protocol - determine if a claim is provable.

        Decidability ladder:
        1. Type checking (is claim well-formed?)
        2. Direct evidence matching (does an observation support it?)
        3. Rule-based inference (can a rule derive it?)
        4. Conflict detection (does evidence refute it?)
        """
        # Step 1: Type checking
        if not truth.text or not truth.kind:
            return CheckResult(
                verdict=CheckVerdict.ILL_TYPED,
                reason="Claim must have text and kind",
            )

        # Step 2: Check for direct evidence support
        supporting_obs = self.pcg.get_supporting_observations(truth.id)
        if supporting_obs:
            return CheckResult(
                verdict=CheckVerdict.PROVABLE,
                proof=[obs.id for obs in supporting_obs],
                reason="Direct evidence support",
            )

        # Step 3: Check if any rule can derive this truth
        # (Look at all observations and see if any rule matches)
        for obs in self.pcg.get_observations():
            obs_dict = {
                "tool": obs.tool,
                "command": obs.command,
                "rc": obs.rc,
                "stdout": obs.stdout,
                "stderr": obs.stderr,
                "scope": obs.source_act,
            }
            matches = self.rules.apply_all(obs_dict, game_id=self.game_id)
            for rule, conclusions in matches:
                for c in conclusions:
                    # Check if conclusion matches this truth
                    if self._conclusion_matches_truth(c, truth):
                        return CheckResult(
                            verdict=CheckVerdict.PROVABLE,
                            proof=[obs.id, rule.id],
                            reason=f"Derived via rule {rule.id}",
                        )

        # Step 4: Check for refuting evidence
        # (Look for observations that contradict the claim)
        for edge in self.pcg.edges:
            if edge.edge_type == EdgeType.REFUTES and edge.dst == truth.id:
                return CheckResult(
                    verdict=CheckVerdict.REFUTABLE,
                    proof=[edge.src] if isinstance(edge.src, str) else edge.src,
                    reason="Contradicted by evidence",
                )

        # Step 5: Undecidable - need more evidence
        return CheckResult(
            verdict=CheckVerdict.UNDECIDABLE,
            missing=["Evidence observation"],
            reason="No evidence or rule to prove/refute",
        )

    def _conclusion_matches_truth(self, conclusion: dict, truth: Truth) -> bool:
        """Check if a rule conclusion matches a truth claim."""
        if conclusion["type"] != "truth":
            return False

        # Fuzzy text matching (could be improved)
        c_text = conclusion["text"].lower()
        t_text = truth.text.lower()

        # Check for key phrase overlap
        c_words = set(c_text.split())
        t_words = set(t_text.split())
        overlap = len(c_words & t_words) / max(len(c_words), len(t_words))

        return overlap > 0.5

    # =========================================================================
    # Move Validation & Execution
    # =========================================================================

    def validate_move(self, move: Move) -> MoveResult:
        """Validate a move before execution."""
        if isinstance(move, ProposeMove):
            return self._validate_propose(move)
        elif isinstance(move, AssertMove):
            return self._validate_assert(move)
        elif isinstance(move, AskMove):
            return MoveResult(verdict=MoveVerdict.LEGAL)
        elif isinstance(move, ChallengeMove):
            return MoveResult(verdict=MoveVerdict.LEGAL)
        elif isinstance(move, ReformulateMove):
            return MoveResult(verdict=MoveVerdict.LEGAL)
        elif isinstance(move, RespondMove):
            return MoveResult(verdict=MoveVerdict.LEGAL)
        else:
            return MoveResult(
                verdict=MoveVerdict.ILLEGAL,
                reason=f"Unknown move type: {type(move)}",
            )

    def _validate_propose(self, move: ProposeMove) -> MoveResult:
        """Validate a PROPOSE move."""
        # Check command safety
        is_safe, reason = self.executor.is_safe(move.command)
        if not is_safe:
            return MoveResult(
                verdict=MoveVerdict.BLOCKED,
                reason=reason,
            )

        # Check constraints
        for constraint in self.pcg.get_active_constraints():
            # Process constraint: read-only first
            if "read-only" in constraint.text.lower() and move.mode == "write":
                # Check if we have localized the cause
                validated = self.pcg.get_truths(Status.VALIDATED)
                if len(validated) < 2:  # Need some evidence first
                    return MoveResult(
                        verdict=MoveVerdict.BLOCKED,
                        reason="Write actions blocked until cause is localized (need more evidence)",
                    )

        return MoveResult(verdict=MoveVerdict.LEGAL)

    def _validate_assert(self, move: AssertMove) -> MoveResult:
        """Validate an ASSERT move."""
        if not move.text:
            return MoveResult(
                verdict=MoveVerdict.ILLEGAL,
                reason="ASSERT requires text",
            )
        return MoveResult(verdict=MoveVerdict.LEGAL)

    def execute_move(self, move: Move) -> MoveResult:
        """Execute a validated move."""
        # Validate first
        validation = self.validate_move(move)
        if validation.verdict != MoveVerdict.LEGAL:
            self.state.score += SCORE_ILLEGAL_MOVE
            self.pcg.log_event("move_rejected", verdict=validation.verdict.value, reason=validation.reason)
            return validation

        # Execute based on type
        if isinstance(move, ProposeMove):
            return self._execute_propose(move)
        elif isinstance(move, AssertMove):
            return self._execute_assert(move)
        elif isinstance(move, AskMove):
            return self._execute_ask(move)
        elif isinstance(move, ChallengeMove):
            return self._execute_challenge(move)
        elif isinstance(move, RespondMove):
            return MoveResult(verdict=MoveVerdict.LEGAL)
        else:
            return MoveResult(verdict=MoveVerdict.LEGAL)

    def _execute_propose(self, move: ProposeMove) -> MoveResult:
        """Execute a PROPOSE action."""
        # Create action node
        action = self.pcg.add_action(
            intent=move.intent,
            mode=move.mode,
            command=move.command,
            verify=move.verify,
        )

        # Log execution start
        self.pcg.log_execute(action.id, "started")

        # Execute command
        rc, stdout, stderr = self.executor.execute(move.command, move.mode)

        # Create observation
        obs = self.pcg.add_observation(
            source_act=action.id,
            tool="bash",
            command=move.command,
            rc=rc,
            stdout=stdout,
            stderr=stderr,
        )

        # Log execution complete
        self.pcg.log_execute(action.id, "finished", rc=rc)

        # Apply rules to derive truths
        obs_dict = {
            "tool": "bash",
            "command": move.command,
            "rc": rc,
            "stdout": stdout,
            "stderr": stderr,
            "scope": action.id,
        }

        derived_truths = []
        derived_entities = []

        for rule, conclusions in self.rules.apply_all(obs_dict, game_id=self.game_id):
            for c in conclusions:
                if c["type"] == "truth":
                    truth = self.pcg.add_truth(
                        text=c["text"],
                        kind=ClaimKind(c.get("kind", "property")),
                        domain=c.get("domain", "bash"),
                        scope=c.get("scope", action.id),
                        status=Status.VALIDATED,
                    )
                    self.pcg.add_support(obs.id, truth.id, transform=f"rule:{rule.id}")
                    derived_truths.append(truth)
                    self.state.validated_truths += 1
                    self.state.score += SCORE_TRUTH_RULE

                elif c["type"] == "entity":
                    entity = self.pcg.add_entity(
                        kind=c.get("kind", "unknown"),
                        value=c["text"],
                    )
                    self.pcg.add_mention(obs.id, [entity.id])
                    derived_entities.append(entity)

        return MoveResult(
            verdict=MoveVerdict.LEGAL,
            observation=obs,
            truths=derived_truths,
            entities=derived_entities,
        )

    def _execute_assert(self, move: AssertMove) -> MoveResult:
        """Execute an ASSERT claim."""
        # Create truth node (proposed, not validated)
        truth = self.pcg.add_truth(
            text=move.text,
            kind=ClaimKind(move.kind) if move.kind else ClaimKind.PROPERTY,
            domain=move.domain,
            scope=move.scope,
            status=Status.PROPOSED,
        )

        # Run CHECK protocol
        check_result = self.check(truth)
        self.pcg.log_check(
            truth.id,
            check_result.verdict.value,
            reason=check_result.reason,
            proof=check_result.proof,
            missing=check_result.missing,
        )

        if check_result.verdict == CheckVerdict.PROVABLE:
            # Promote to validated
            truth.status = Status.VALIDATED
            self.pcg.log_promote(truth.id, "validated")
            self.state.validated_truths += 1
            self.state.score += SCORE_TRUTH_DIRECT

        return MoveResult(
            verdict=MoveVerdict.LEGAL,
            truths=[truth],
            reason=f"CHECK verdict: {check_result.verdict.value}",
        )

    def _execute_ask(self, move: AskMove) -> MoveResult:
        """Execute an ASK question."""
        question = self.pcg.add_question(
            text=move.text,
            targets=move.targets,
        )
        self.state.open_questions += 1

        return MoveResult(
            verdict=MoveVerdict.LEGAL,
            reason=f"Question registered: {question.id}",
        )

    def _execute_challenge(self, move: ChallengeMove) -> MoveResult:
        """Execute a CHALLENGE assumption."""
        assumption = Assumption(
            id=self.pcg._next_id("A"),
            text=move.text,
            falsify_hint=move.falsify,
        )
        self.pcg.add_node(assumption)

        return MoveResult(
            verdict=MoveVerdict.LEGAL,
            reason=f"Assumption registered: {assumption.id}",
        )

    # =========================================================================
    # Truth/Dare Mechanism
    # =========================================================================

    def truth_or_dare(self, truth_id: str) -> dict:
        """
        When a claim is undecidable, offer exactly two options:
        - TRUTH: Reformulate into a decidable claim
        - DARE: Propose an evidence-producing action
        """
        truth = self.pcg.get_node(truth_id)
        if not isinstance(truth, Truth):
            return {"error": f"Node {truth_id} is not a Truth"}

        check_result = self.check(truth)
        if check_result.verdict != CheckVerdict.UNDECIDABLE:
            return {"error": f"Claim is not undecidable: {check_result.verdict.value}"}

        return {
            "claim_id": truth_id,
            "claim_text": truth.text,
            "options": {
                "TRUTH": "Reformulate this claim into something decidable",
                "DARE": "Propose an action that produces evidence for this claim",
            },
            "hint": f"Missing: {', '.join(check_result.missing)}",
        }

    # =========================================================================
    # Fat LLM Rule Generation
    # =========================================================================

    def request_rule_from_fat_llm(self, claim: Truth, obs: Observation) -> Optional[Rule]:
        """
        Request a new rule from the fat LLM to derive a claim.

        The fat LLM generates a rule (YAML), not a direct answer.
        The Tutor then applies the rule deterministically.
        """
        if not self.fat_llm_callback:
            return None

        from .rules import generate_rule_prompt, parse_rule_from_llm

        # Generate prompt
        obs_dict = {
            "tool": obs.tool,
            "rc": obs.rc,
            "stdout": obs.stdout[:500],
            "scope": obs.source_act,
        }
        prompt = generate_rule_prompt(claim.text, obs_dict)

        # Call fat LLM
        try:
            response = self.fat_llm_callback(prompt)
            rule = parse_rule_from_llm(
                response,
                game_id=self.game_id,
                turn=self.state.turn,
                model="fat_llm",
            )

            # Add as session rule
            self.rules.add_session_rule(self.game_id, rule)
            self.pcg.log_event(
                "rule_generated",
                target=rule.id,
                reason=f"Generated to derive {claim.id}",
                generated_by="fat_llm",
            )

            return rule

        except Exception as e:
            self.pcg.log_event("rule_generation_failed", reason=str(e))
            return None

    # =========================================================================
    # Game State
    # =========================================================================

    def next_turn(self) -> None:
        """Advance to next turn."""
        self.state.turn += 1
        self.pcg.turn = self.state.turn

    def check_goal_satisfaction(self) -> bool:
        """Check if the game goal is satisfied."""
        goals = self.pcg.get_active_goals()
        if not goals:
            return False

        # Simple heuristic: goal satisfied if we have validated truths
        # and no critical open questions
        validated = self.pcg.get_truths(Status.VALIDATED)
        open_qs = self.pcg.get_open_questions()

        if len(validated) >= 3 and len(open_qs) == 0:
            self.state.goal_satisfied = True
            self.pcg.log_event(
                "goal_check",
                goal=goals[0].id,
                verdict="satisfied",
                by=[t.id for t in validated[:5]],
            )
            return True

        return False

    def get_state_summary(self) -> dict:
        """Get current game state summary."""
        return {
            "game_id": self.game_id,
            "turn": self.state.turn,
            "score": self.state.score,
            "validated_truths": self.state.validated_truths,
            "open_questions": self.state.open_questions,
            "goal_satisfied": self.state.goal_satisfied,
            "pcg_summary": self.pcg.summary(),
            "rules_summary": self.rules.summary(),
        }

    def get_context_for_llm(self, max_observations: int = 5) -> str:
        """
        Generate context to inject into LLM prompt.

        Provides the LLM with:
        - Validated truths
        - Recent observations
        - Open questions
        - Active constraints
        """
        lines = ["=== GAME STATE ==="]
        lines.append(f"Turn: {self.state.turn}")
        lines.append(f"Score: {self.state.score}")
        lines.append("")

        # Validated truths
        validated = self.pcg.get_truths(Status.VALIDATED)
        if validated:
            lines.append("VALIDATED TRUTHS:")
            for t in validated[-10:]:  # Last 10
                lines.append(f"  [{t.id}] {t.text}")
            lines.append("")

        # Recent observations
        observations = self.pcg.get_observations()
        if observations:
            lines.append(f"RECENT OBSERVATIONS (last {max_observations}):")
            for obs in observations[-max_observations:]:
                lines.append(f"  [{obs.id}] {obs.tool}: {obs.command}")
                lines.append(f"      rc={obs.rc}, stdout={obs.stdout[:100]}...")
            lines.append("")

        # Open questions
        questions = self.pcg.get_open_questions()
        if questions:
            lines.append("OPEN QUESTIONS:")
            for q in questions:
                lines.append(f"  [{q.id}] {q.text}")
            lines.append("")

        # Constraints
        constraints = self.pcg.get_active_constraints()
        if constraints:
            lines.append("CONSTRAINTS:")
            for c in constraints:
                lines.append(f"  - {c.text}")

        return "\n".join(lines)
