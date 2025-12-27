# =============================================================================
# RAGIX Reasoning Tutor - Interpreter-Tutor Architecture
# =============================================================================
#
# A game-based reasoning system where:
# - LLM (Player): Proposes moves, claims, questions
# - Tutor (Referee): Validates, executes, proves - always deterministic
#
# Core principle: "The LLM does not interact with the real world. It plays a game."
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-21)
#
# =============================================================================

"""
Interpreter-Tutor Architecture for Rule-Based LLM Reasoning

This module implements a proof-carrying game where slim LLMs propose moves
and a deterministic Tutor validates them against evidence and rules.

Key Components:
- PCG (Proof-Carrying Graph): Append-only graph of truths, observations, entities
- Rules: YAML-based declarative rules with typed operators
- Tutor: Deterministic referee implementing CHECK protocol
- Moves: Structured LLM output parser (ASSERT, ASK, PROPOSE, etc.)

Design Philosophy:
- Hallucination suppression by structure (illegal moves rejected)
- Truth via evidence proofs, not LLM claims
- Minimal LLM requirements (only proposal generation)
- Full auditability (append-only log)
"""

from .pcg import (
    PCG,
    Node,
    Edge,
    NodeType,
    EdgeType,
    Truth,
    Observation,
    Entity,
    Action,
    Question,
    Assumption,
    Constraint,
    Goal,
    RuleNode,
)

from .rules import (
    Rule,
    Match,
    Extract,
    Conclusion,
    RuleLibrary,
    load_rules,
)

from .tutor import (
    Tutor,
    CheckVerdict,
    MoveVerdict,
    GameState,
)

from .moves import (
    Move,
    MoveType,
    AssertMove,
    AskMove,
    ProposeMove,
    ChallengeMove,
    ReformulateMove,
    parse_move,
)

__version__ = "0.1.0"
__all__ = [
    # PCG
    "PCG", "Node", "Edge", "NodeType", "EdgeType",
    "Truth", "Observation", "Entity", "Action", "Question",
    "Assumption", "Constraint", "Goal", "RuleNode",
    # Rules
    "Rule", "Match", "Extract", "Conclusion", "RuleLibrary", "load_rules",
    # Tutor
    "Tutor", "CheckVerdict", "MoveVerdict", "GameState",
    # Moves
    "Move", "MoveType", "AssertMove", "AskMove", "ProposeMove",
    "ChallengeMove", "ReformulateMove", "parse_move",
]
