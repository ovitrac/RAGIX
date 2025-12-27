# =============================================================================
# PCG - Proof-Carrying Graph
# =============================================================================
#
# Append-only graph structure for the Interpreter-Tutor game.
# All state changes are logged as events - nothing is mutated.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-21)
#
# =============================================================================

"""
Proof-Carrying Graph (PCG) - The game board.

Node Types:
- Obs: Immutable observation (tool output, file slice, metric)
- Ent: Entity (file, function, package, host, branch)
- T: Claim / Truth
- A: Assumption (claim with falsification plan)
- Q: Question / Unknown
- Act: Action (template or executed)
- R: Inference rule
- C: Constraint / Policy
- G: Goal

Edge Types:
- supports: Obs → T (evidence supports claim)
- derives: R + premises → T (rule derivation)
- refutes: Obs → T (evidence contradicts claim)
- mentions: Obs → Ent (observation references entity)
- requires: Act → {T,A,C} (action preconditions)
- produces: Act → Obs (action generates observation)
- blocks: C → Act (constraint prevents action)
- targets: Q → Ent|T (question about something)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union
from datetime import datetime
import hashlib
import json


class NodeType(Enum):
    """Node types in the Proof-Carrying Graph."""
    OBSERVATION = "Obs"      # Immutable tool output
    ENTITY = "Ent"           # File, function, variable, host...
    TRUTH = "T"              # Claim (proposed, validated, refuted)
    ASSUMPTION = "A"         # Claim with falsification plan
    QUESTION = "Q"           # Open unknown
    ACTION = "Act"           # Proposed or executed action
    RULE = "R"               # Inference rule
    CONSTRAINT = "C"         # Policy or process constraint
    GOAL = "G"               # Game goal


class EdgeType(Enum):
    """Edge types in the Proof-Carrying Graph."""
    SUPPORTS = "supports"       # Obs → T
    DERIVES = "derives"         # R + premises → T
    REFUTES = "refutes"         # Obs → T
    MENTIONS = "mentions"       # Obs → Ent
    REQUIRES = "requires"       # Act → {T,A,C}
    PRODUCES = "produces"       # Act → Obs
    BLOCKS = "blocks"           # C → Act
    TARGETS = "targets"         # Q → Ent|T


class Status(Enum):
    """Status for nodes."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REFUTED = "refuted"
    OPEN = "open"
    CLOSED = "closed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMMITTED = "committed"
    STARTED = "started"
    FINISHED = "finished"


class ClaimKind(Enum):
    """Kind of claim (for Truth nodes)."""
    EXISTENCE = "existence"
    EQUALITY = "equality"
    MEMBERSHIP = "membership"
    PROPERTY = "property"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"
    NORMATIVE = "normative"


class Soundness(Enum):
    """Soundness of inference rules."""
    SOUND = "sound"
    HEURISTIC = "heuristic"


@dataclass
class Node:
    """Base class for all PCG nodes."""
    id: str
    node_type: NodeType = field(default=NodeType.TRUTH)
    created_at: str = field(default="")

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": f"node.{self.node_type.value}",
            "id": self.id,
            "created_at": self.created_at,
        }


@dataclass
class Observation(Node):
    """Immutable observation from tool execution."""
    source_act: str = ""               # Action ID that produced this
    tool: str = ""                     # Tool used (bash, grep, sed...)
    command: str = ""                  # Command executed
    rc: int = 0                        # Return code
    stdout: str = ""                   # Standard output
    stderr: str = ""                   # Standard error
    hash: str = ""                     # Content hash
    status: Status = Status.COMMITTED

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.OBSERVATION
        if not self.hash:
            content = f"{self.tool}:{self.command}:{self.rc}:{self.stdout}:{self.stderr}"
            self.hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "source_act": self.source_act,
            "tool": self.tool,
            "command": self.command,
            "rc": self.rc,
            "stdout_snip": self.stdout[:500] if len(self.stdout) > 500 else self.stdout,
            "stderr_snip": self.stderr[:200] if len(self.stderr) > 200 else self.stderr,
            "hash": self.hash,
            "status": self.status.value,
        })
        return d


@dataclass
class Entity(Node):
    """Entity referenced in observations."""
    kind: str = ""         # file, function, variable, test_id, host...
    value: Any = None      # The entity value

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.ENTITY

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "kind": self.kind,
            "value": self.value,
        })
        return d


@dataclass
class Truth(Node):
    """Claim that can be proposed, validated, or refuted."""
    text: str = ""                         # The claim text
    kind: ClaimKind = ClaimKind.PROPERTY   # Type of claim
    domain: str = ""                       # fs, git, python, bash...
    scope: str = ""                        # repo, file path, function...
    status: Status = Status.PROPOSED

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.TRUTH

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "text": self.text,
            "kind": self.kind.value,
            "domain": self.domain,
            "scope": self.scope,
            "status": self.status.value,
        })
        return d


@dataclass
class Assumption(Node):
    """Claim with a falsification plan."""
    text: str = ""
    falsify_hint: str = ""     # How to falsify this assumption
    status: Status = Status.PROPOSED

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.ASSUMPTION

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "text": self.text,
            "falsify_hint": self.falsify_hint,
            "status": self.status.value,
        })
        return d


@dataclass
class Question(Node):
    """Open question requiring evidence."""
    text: str = ""
    targets: list[str] = field(default_factory=list)  # Entity/Truth IDs
    status: Status = Status.OPEN

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.QUESTION

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "text": self.text,
            "targets": self.targets,
            "status": self.status.value,
        })
        return d


@dataclass
class Action(Node):
    """Proposed or executed action."""
    intent: str = ""                      # Human-readable intent
    mode: str = "read"                    # "read" or "write"
    command: str = ""                     # Shell command
    pre: list[str] = field(default_factory=list)      # Precondition IDs
    verify: list[str] = field(default_factory=list)   # Verification checks
    status: Status = Status.PROPOSED

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.ACTION

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "intent": self.intent,
            "mode": self.mode,
            "command": self.command,
            "pre": self.pre,
            "verify": self.verify,
            "status": self.status.value,
        })
        return d


@dataclass
class Constraint(Node):
    """Policy or process constraint."""
    text: str = ""
    kind: str = ""         # "policy", "process", "epistemic"
    status: Status = Status.ACTIVE

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.CONSTRAINT

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "text": self.text,
            "kind": self.kind,
            "status": self.status.value,
        })
        return d


@dataclass
class Goal(Node):
    """Game goal."""
    text: str = ""
    status: Status = Status.ACTIVE

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.GOAL

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "text": self.text,
            "status": self.status.value,
        })
        return d


@dataclass
class RuleNode(Node):
    """Inference rule (stored as node for traceability)."""
    rule_id: str = ""          # Reference to rule in library
    soundness: Soundness = Soundness.SOUND
    premise_pattern: str = ""  # Human-readable premise pattern
    conclusion_pattern: str = "" # Human-readable conclusion pattern

    def __post_init__(self):
        super().__post_init__()
        self.node_type = NodeType.RULE

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "rule_id": self.rule_id,
            "soundness": self.soundness.value,
            "premise": self.premise_pattern,
            "conclusion": self.conclusion_pattern,
        })
        return d


@dataclass
class Edge:
    """Edge in the PCG."""
    edge_type: EdgeType
    src: Union[str, list[str]]     # Source node ID(s)
    dst: Union[str, list[str]]     # Destination node ID(s)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    locator: Optional[dict] = None  # For supports edges: where in observation
    transform: Optional[str] = None # How evidence maps to claim

    def to_dict(self) -> dict:
        d = {
            "type": f"edge.{self.edge_type.value}",
            "src": self.src,
            "dst": self.dst,
            "created_at": self.created_at,
        }
        if self.locator:
            d["locator"] = self.locator
        if self.transform:
            d["transform"] = self.transform
        return d


@dataclass
class Event:
    """Event in the append-only log."""
    event_type: str            # check, execute, accept, reject, promote, etc.
    target: Optional[str] = None
    verdict: Optional[str] = None
    reason: Optional[str] = None
    data: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = {
            "type": f"event.{self.event_type}",
            "created_at": self.created_at,
        }
        if self.target:
            d["target"] = self.target
        if self.verdict:
            d["verdict"] = self.verdict
        if self.reason:
            d["reason"] = self.reason
        d.update(self.data)
        return d


class PCG:
    """
    Proof-Carrying Graph - The game board.

    Append-only: nodes and edges are never modified, only added.
    Status changes are represented as events.
    """

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.events: list[Event] = []
        self.turn = 0
        self._id_counter = 0

    def _next_id(self, prefix: str) -> str:
        """Generate unique ID."""
        self._id_counter += 1
        return f"{prefix}{self._id_counter}"

    # =========================================================================
    # Node Operations (append-only)
    # =========================================================================

    def add_node(self, node: Node) -> str:
        """Add a node to the graph."""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node
        self.events.append(Event(
            event_type="node_added",
            target=node.id,
            data={"node_type": node.node_type.value}
        ))
        return node.id

    def add_observation(self, source_act: str, tool: str, command: str,
                        rc: int, stdout: str = "", stderr: str = "") -> Observation:
        """Add an observation from tool execution."""
        obs = Observation(
            id=self._next_id("Obs"),
            source_act=source_act,
            tool=tool,
            command=command,
            rc=rc,
            stdout=stdout,
            stderr=stderr,
        )
        self.add_node(obs)
        self.add_edge(EdgeType.PRODUCES, source_act, obs.id)
        return obs

    def add_truth(self, text: str, kind: ClaimKind, domain: str, scope: str,
                  status: Status = Status.PROPOSED) -> Truth:
        """Add a truth claim."""
        truth = Truth(
            id=self._next_id("T"),
            text=text,
            kind=kind,
            domain=domain,
            scope=scope,
            status=status,
        )
        self.add_node(truth)
        return truth

    def add_entity(self, kind: str, value: Any) -> Entity:
        """Add an entity."""
        entity = Entity(
            id=self._next_id("Ent"),
            kind=kind,
            value=value,
        )
        self.add_node(entity)
        return entity

    def add_question(self, text: str, targets: list[str] = None) -> Question:
        """Add an open question."""
        question = Question(
            id=self._next_id("Q"),
            text=text,
            targets=targets or [],
        )
        self.add_node(question)
        return question

    def add_action(self, intent: str, mode: str, command: str,
                   pre: list[str] = None, verify: list[str] = None) -> Action:
        """Add a proposed action."""
        action = Action(
            id=self._next_id("Act"),
            intent=intent,
            mode=mode,
            command=command,
            pre=pre or [],
            verify=verify or [],
        )
        self.add_node(action)
        return action

    def add_constraint(self, text: str, kind: str) -> Constraint:
        """Add a constraint."""
        constraint = Constraint(
            id=self._next_id("C"),
            text=text,
            kind=kind,
        )
        self.add_node(constraint)
        return constraint

    def add_goal(self, text: str) -> Goal:
        """Add a goal."""
        goal = Goal(
            id=self._next_id("G"),
            text=text,
        )
        self.add_node(goal)
        return goal

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(self, edge_type: EdgeType, src: Union[str, list[str]],
                 dst: Union[str, list[str]], locator: dict = None,
                 transform: str = None) -> Edge:
        """Add an edge to the graph."""
        edge = Edge(
            edge_type=edge_type,
            src=src,
            dst=dst,
            locator=locator,
            transform=transform,
        )
        self.edges.append(edge)
        return edge

    def add_support(self, obs_id: str, truth_id: str,
                    locator: dict = None, transform: str = None) -> Edge:
        """Add a supports edge (Obs → T)."""
        return self.add_edge(EdgeType.SUPPORTS, obs_id, truth_id, locator, transform)

    def add_derivation(self, rule_id: str, premises: list[str], truth_id: str) -> Edge:
        """Add a derives edge (R + premises → T)."""
        return self.add_edge(EdgeType.DERIVES, [rule_id] + premises, truth_id)

    def add_mention(self, obs_id: str, entity_ids: list[str]) -> Edge:
        """Add mentions edges (Obs → Ent)."""
        return self.add_edge(EdgeType.MENTIONS, obs_id, entity_ids)

    # =========================================================================
    # Event Operations
    # =========================================================================

    def log_event(self, event_type: str, target: str = None,
                  verdict: str = None, reason: str = None, **data) -> Event:
        """Log an event."""
        event = Event(
            event_type=event_type,
            target=target,
            verdict=verdict,
            reason=reason,
            data=data,
        )
        self.events.append(event)
        return event

    def log_check(self, target: str, verdict: str, reason: str = None,
                  proof: list[str] = None, missing: list[str] = None) -> Event:
        """Log a CHECK verdict."""
        return self.log_event(
            "check",
            target=target,
            verdict=verdict,
            reason=reason,
            proof=proof or [],
            missing=missing or [],
        )

    def log_execute(self, act_id: str, status: str, rc: int = None) -> Event:
        """Log action execution."""
        return self.log_event("execute", act=act_id, status=status, rc=rc)

    def log_promote(self, target: str, to_status: str) -> Event:
        """Log status promotion."""
        return self.log_event("promote", target=target, to_status=to_status)

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_truths(self, status: Status = None) -> list[Truth]:
        """Get all truth nodes, optionally filtered by status."""
        truths = [n for n in self.nodes.values() if isinstance(n, Truth)]
        if status:
            truths = [t for t in truths if t.status == status]
        return truths

    def get_observations(self) -> list[Observation]:
        """Get all observations."""
        return [n for n in self.nodes.values() if isinstance(n, Observation)]

    def get_supporting_observations(self, truth_id: str) -> list[Observation]:
        """Get observations that support a truth."""
        obs_ids = []
        for edge in self.edges:
            if edge.edge_type == EdgeType.SUPPORTS and edge.dst == truth_id:
                if isinstance(edge.src, str):
                    obs_ids.append(edge.src)
                else:
                    obs_ids.extend(edge.src)
        return [self.nodes[oid] for oid in obs_ids if oid in self.nodes]

    def get_open_questions(self) -> list[Question]:
        """Get open questions."""
        return [n for n in self.nodes.values()
                if isinstance(n, Question) and n.status == Status.OPEN]

    def get_active_constraints(self) -> list[Constraint]:
        """Get active constraints."""
        return [n for n in self.nodes.values()
                if isinstance(n, Constraint) and n.status == Status.ACTIVE]

    def get_active_goals(self) -> list[Goal]:
        """Get active goals."""
        return [n for n in self.nodes.values()
                if isinstance(n, Goal) and n.status == Status.ACTIVE]

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_jsonl(self) -> str:
        """Export as JSONL (one event per line)."""
        lines = []
        # Export nodes
        for node in self.nodes.values():
            lines.append(json.dumps(node.to_dict()))
        # Export edges
        for edge in self.edges:
            lines.append(json.dumps(edge.to_dict()))
        # Export events
        for event in self.events:
            lines.append(json.dumps(event.to_dict()))
        return "\n".join(lines)

    def summary(self) -> dict:
        """Get a summary of the graph state."""
        return {
            "game_id": self.game_id,
            "turn": self.turn,
            "nodes": {
                "total": len(self.nodes),
                "observations": len([n for n in self.nodes.values() if isinstance(n, Observation)]),
                "truths": len([n for n in self.nodes.values() if isinstance(n, Truth)]),
                "validated": len([n for n in self.nodes.values() if isinstance(n, Truth) and n.status == Status.VALIDATED]),
                "entities": len([n for n in self.nodes.values() if isinstance(n, Entity)]),
                "questions": len([n for n in self.nodes.values() if isinstance(n, Question)]),
                "actions": len([n for n in self.nodes.values() if isinstance(n, Action)]),
            },
            "edges": len(self.edges),
            "events": len(self.events),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return f"PCG(game={self.game_id}, turn={s['turn']}, nodes={s['nodes']['total']}, validated={s['nodes']['validated']})"
