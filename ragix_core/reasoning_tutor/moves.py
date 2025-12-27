# =============================================================================
# Move Parser - Structured LLM Output
# =============================================================================
#
# Parses LLM output into structured moves for the Interpreter-Tutor game.
# The LLM proposes moves; the Tutor validates them.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-21)
#
# =============================================================================

"""
Move Parser for LLM Output.

The LLM (player) can propose the following moves:
- ASSERT(T): Claim a truth
- ASK(Q): Ask a question
- PROPOSE(Act): Propose an action
- CHALLENGE(A): Make an assumption with falsification plan
- REFORMULATE(T → T'): Reframe a claim
- LINK(node → node): Propose an edge

The parser extracts structured moves from LLM text output.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union
import re
import json


class MoveType(Enum):
    """Types of moves the LLM can propose."""
    ASSERT = "ASSERT"           # Claim a truth
    ASK = "ASK"                 # Ask a question
    PROPOSE = "PROPOSE"         # Propose an action
    CHALLENGE = "CHALLENGE"     # Make an assumption
    REFORMULATE = "REFORMULATE" # Reframe a claim
    LINK = "LINK"               # Propose an edge
    RESPOND = "RESPOND"         # Just respond (no game move)


@dataclass
class Move:
    """Base class for all moves."""
    move_type: MoveType = MoveType.RESPOND
    raw_text: str = ""          # Original LLM output


@dataclass
class AssertMove(Move):
    """ASSERT(T) - Claim a truth."""
    text: str = ""              # The claim text
    kind: str = "property"      # Claim kind
    domain: str = "bash"        # Domain
    scope: str = ""             # Scope
    move_type: MoveType = field(default=MoveType.ASSERT)


@dataclass
class AskMove(Move):
    """ASK(Q) - Ask a question."""
    text: str = ""              # The question
    targets: list[str] = field(default_factory=list)  # Referenced entities
    move_type: MoveType = field(default=MoveType.ASK)


@dataclass
class ProposeMove(Move):
    """PROPOSE(Act) - Propose an action."""
    intent: str = ""            # What the action is for
    command: str = ""           # Shell command
    mode: str = "read"          # "read" or "write"
    verify: list[str] = field(default_factory=list)   # Verification conditions
    move_type: MoveType = field(default=MoveType.PROPOSE)


@dataclass
class ChallengeMove(Move):
    """CHALLENGE(A) - Make an assumption with falsification plan."""
    text: str = ""              # The assumption
    falsify: str = ""           # How to falsify it
    move_type: MoveType = field(default=MoveType.CHALLENGE)


@dataclass
class ReformulateMove(Move):
    """REFORMULATE(T → T') - Reframe a claim."""
    original_id: str = ""       # Original claim ID
    new_text: str = ""          # Reformulated claim
    move_type: MoveType = field(default=MoveType.REFORMULATE)


@dataclass
class RespondMove(Move):
    """RESPOND - Just respond, no game move."""
    message: str = ""
    move_type: MoveType = field(default=MoveType.RESPOND)


# =============================================================================
# Move Parser
# =============================================================================

class MoveParser:
    """
    Parse LLM output into structured moves.

    Supports multiple formats:
    1. JSON format: {"action": "ASSERT", "text": "..."}
    2. Tagged format: ASSERT: <text>
    3. Natural language (best effort)
    """

    # Patterns for tagged format
    PATTERNS = {
        MoveType.ASSERT: [
            r"ASSERT[:\s]+(.+?)(?=\n(?:ASSERT|ASK|PROPOSE|CHALLENGE|REFORMULATE)|$)",
            r"I claim[:\s]+(.+?)(?=\n|$)",
            r"I assert[:\s]+(.+?)(?=\n|$)",
        ],
        MoveType.ASK: [
            r"ASK[:\s]+(.+?)(?=\n(?:ASSERT|ASK|PROPOSE|CHALLENGE|REFORMULATE)|$)",
            r"I ask[:\s]+(.+?)(?=\n|$)",
            r"Question[:\s]+(.+?)(?=\n|$)",
        ],
        MoveType.PROPOSE: [
            r"PROPOSE[:\s]+(.+?)(?=\n(?:ASSERT|ASK|PROPOSE|CHALLENGE|REFORMULATE)|$)",
            r"I propose[:\s]+(.+?)(?=\n|$)",
            r"Let me run[:\s]+(.+?)(?=\n|$)",
            r"Run[:\s]+`([^`]+)`",
        ],
        MoveType.CHALLENGE: [
            r"CHALLENGE[:\s]+(.+?)(?=\n(?:ASSERT|ASK|PROPOSE|CHALLENGE|REFORMULATE)|$)",
            r"I assume[:\s]+(.+?)(?=\n|$)",
            r"Assumption[:\s]+(.+?)(?=\n|$)",
        ],
        MoveType.REFORMULATE: [
            r"REFORMULATE[:\s]+(.+?)(?=\n|$)",
            r"Let me rephrase[:\s]+(.+?)(?=\n|$)",
        ],
    }

    # Command extraction patterns
    COMMAND_PATTERNS = [
        r"`([^`]+)`",                    # Backtick code
        r"```(?:bash|sh)?\n([^`]+)```",  # Code block
        r"command[:\s]+(.+?)(?=\n|$)",   # Explicit command
        r"run[:\s]+(.+?)(?=\n|$)",       # Run command
    ]

    def parse(self, text: str) -> list[Move]:
        """Parse LLM output into moves."""
        moves = []

        # Try JSON format first
        json_moves = self._parse_json(text)
        if json_moves:
            return json_moves

        # Try tagged format
        for move_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                    content = match.group(1).strip()
                    move = self._create_move(move_type, content, text)
                    if move:
                        moves.append(move)

        # If no structured moves found, treat as PROPOSE with command extraction
        if not moves:
            commands = self._extract_commands(text)
            for cmd in commands:
                moves.append(ProposeMove(
                    raw_text=text,
                    intent="Execute command",
                    command=cmd,
                    mode="read",
                ))

        # If still nothing, treat as RESPOND
        if not moves:
            moves.append(RespondMove(raw_text=text, message=text))

        return moves

    def _parse_json(self, text: str) -> list[Move]:
        """Try to parse JSON-formatted moves."""
        moves = []

        # Look for JSON objects in the text
        json_pattern = r'\{[^{}]+\}'
        for match in re.finditer(json_pattern, text):
            try:
                data = json.loads(match.group())
                move = self._json_to_move(data, text)
                if move:
                    moves.append(move)
            except json.JSONDecodeError:
                continue

        return moves

    def _json_to_move(self, data: dict, raw_text: str) -> Optional[Move]:
        """Convert JSON dict to Move."""
        action = data.get("action", "").upper()

        if action == "ASSERT":
            return AssertMove(
                raw_text=raw_text,
                text=data.get("text", data.get("claim", "")),
                kind=data.get("kind", "property"),
                domain=data.get("domain", "bash"),
                scope=data.get("scope", ""),
            )
        elif action == "ASK":
            return AskMove(
                raw_text=raw_text,
                text=data.get("text", data.get("question", "")),
                targets=data.get("targets", []),
            )
        elif action in ("PROPOSE", "BASH", "BASH_AND_RESPOND"):
            return ProposeMove(
                raw_text=raw_text,
                intent=data.get("intent", "Execute command"),
                command=data.get("command", ""),
                mode=data.get("mode", "read"),
                verify=data.get("verify", []),
            )
        elif action == "CHALLENGE":
            return ChallengeMove(
                raw_text=raw_text,
                text=data.get("text", data.get("assumption", "")),
                falsify=data.get("falsify", data.get("falsification", "")),
            )
        elif action == "REFORMULATE":
            return ReformulateMove(
                raw_text=raw_text,
                original_id=data.get("original", data.get("original_id", "")),
                new_text=data.get("new_text", data.get("reformulation", "")),
            )
        elif action == "RESPOND":
            return RespondMove(
                raw_text=raw_text,
                message=data.get("message", data.get("text", "")),
            )

        return None

    def _create_move(self, move_type: MoveType, content: str, raw_text: str) -> Optional[Move]:
        """Create a move from parsed content."""
        if move_type == MoveType.ASSERT:
            return AssertMove(raw_text=raw_text, text=content)
        elif move_type == MoveType.ASK:
            return AskMove(raw_text=raw_text, text=content)
        elif move_type == MoveType.PROPOSE:
            # Extract command from content
            commands = self._extract_commands(content)
            cmd = commands[0] if commands else content
            return ProposeMove(raw_text=raw_text, intent=content, command=cmd)
        elif move_type == MoveType.CHALLENGE:
            return ChallengeMove(raw_text=raw_text, text=content)
        elif move_type == MoveType.REFORMULATE:
            return ReformulateMove(raw_text=raw_text, new_text=content)

        return None

    def _extract_commands(self, text: str) -> list[str]:
        """Extract shell commands from text."""
        commands = []

        for pattern in self.COMMAND_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                cmd = match.group(1).strip()
                # Filter out non-commands
                if cmd and not cmd.startswith("#") and len(cmd) > 2:
                    commands.append(cmd)

        return commands


def parse_move(text: str) -> list[Move]:
    """Parse LLM output into moves."""
    parser = MoveParser()
    return parser.parse(text)


# =============================================================================
# Move Prompt Generation
# =============================================================================

def generate_move_prompt(context: dict) -> str:
    """
    Generate a prompt that teaches the LLM the move format.

    This is injected into the system prompt to guide the LLM.
    """
    return """You are a PLAYER in a proof game. You propose MOVES; a TUTOR validates them.

LEGAL MOVES:

1. PROPOSE an action (most common):
   {"action": "PROPOSE", "intent": "<why>", "command": "<shell command>", "mode": "read"}

2. ASSERT a claim:
   {"action": "ASSERT", "text": "<claim>", "kind": "property", "domain": "bash"}

3. ASK a question:
   {"action": "ASK", "text": "<question>", "targets": ["<entity_id>"]}

4. CHALLENGE (assumption with falsification):
   {"action": "CHALLENGE", "text": "<assumption>", "falsify": "<how to disprove>"}

5. RESPOND (just answer, no game move):
   {"action": "RESPOND", "message": "<your message>"}

RULES:
- Output ONE JSON object per move
- For shell commands, use PROPOSE with mode="read" (safe) or mode="write" (changes files)
- Claims without evidence will be marked "undecidable" - you must PROPOSE an action to get evidence
- The TUTOR executes commands and validates claims - you never see raw output directly

EXAMPLE TURN:
User: "Check if config.yaml exists"
You: {"action": "PROPOSE", "intent": "Check file existence", "command": "test -f config.yaml && echo exists || echo missing", "mode": "read"}
"""
