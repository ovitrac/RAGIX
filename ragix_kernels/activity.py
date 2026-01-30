"""
Activity Logging — Centralized, auditable activity stream for KOAS.

This module provides governance-level logging for:
- Kernel execution (start/end, metrics, decisions)
- LLM calls (cache hits, model usage)
- Orchestration events (workflow start/end)
- Authentication events (when using broker)

Design principles:
- Activity logging ≠ content logging (log what happened, not what was processed)
- Single canonical event stream (append-only JSONL)
- Kernel-agnostic with common schema
- Sovereignty-first but activity-oriented

Schema version: koas.event/1.0

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-01-30
Reference: docs/developer/ROADMAP_ACTIVITY_LOGGING.md
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Schema version for forward compatibility
SCHEMA_VERSION = "koas.event/1.0"


# =============================================================================
# Event Schema (koas.event/1.0)
# =============================================================================

class ActorType(str, Enum):
    """Types of actors that can emit events."""
    SYSTEM = "system"
    OPERATOR = "operator"
    EXTERNAL_ORCHESTRATOR = "external_orchestrator"
    AUDITOR = "auditor"


class AuthMethod(str, Enum):
    """Authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    HMAC = "hmac"


@dataclass
class Actor:
    """
    Entity that initiated an action.

    Attributes:
        type: Actor type (system, operator, external_orchestrator, auditor)
        id: Unique identifier for the actor
        auth: Authentication method used
        session: Optional session identifier
    """
    type: str
    id: str
    auth: str = "none"
    session: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type, "id": self.id, "auth": self.auth}
        if self.session:
            d["session"] = self.session
        return d


@dataclass
class KernelInfo:
    """
    Information about a kernel.

    Attributes:
        name: Kernel name (e.g., "doc_extract")
        version: Kernel version (e.g., "1.2.0")
        stage: Pipeline stage (1, 2, or 3)
    """
    name: str
    version: str
    stage: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "stage": self.stage}


@dataclass
class NodeRef:
    """
    Reference to a node in the document hierarchy.

    Attributes:
        level: Hierarchy level (chunk, doc, group, domain, corpus)
        id: Node identifier
    """
    level: str
    id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "id": self.id}


@dataclass
class ActivityEvent:
    """
    A single activity event in the KOAS event stream.

    This is the core data structure for activity logging.
    Each event is a self-contained envelope with:
    - Identification (event_id, run_id, timestamp)
    - Actor information (who initiated)
    - Scope and phase (what category of action)
    - Optional kernel, node, I/O, decision, metrics information

    Attributes:
        scope: Event category (e.g., "docs.kernel", "docs.llm", "system.auth")
        actor: Entity that initiated the action
        phase: Specific phase within scope (e.g., "start", "end", "cache_hit")
        kernel: Kernel information (for kernel events)
        node_ref: Node reference (for document hierarchy events)
        io: Input/output hashes for provenance
        decision: Decision metadata (cache hit/miss, branch taken)
        metrics: Quantitative metrics (duration, counts, scores)
        refs: References to related entities (call_hash, parent_event)
        sovereignty: Sovereignty attestation (local_only, endpoint)
    """
    scope: str
    actor: Actor

    # Optional fields
    phase: str = ""
    kernel: Optional[KernelInfo] = None
    node_ref: Optional[NodeRef] = None
    io: Dict[str, str] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    refs: Dict[str, str] = field(default_factory=dict)
    sovereignty: Dict[str, Any] = field(default_factory=lambda: {"local_only": True})

    # Auto-generated fields
    v: str = field(default=SCHEMA_VERSION)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec='milliseconds'))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "v": self.v,
            "ts": self.ts,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "actor": self.actor.to_dict(),
            "scope": self.scope,
        }

        if self.phase:
            d["phase"] = self.phase
        if self.kernel:
            d["kernel"] = self.kernel.to_dict()
        if self.node_ref:
            d["node_ref"] = self.node_ref.to_dict()
        if self.io:
            d["io"] = self.io
        if self.decision:
            d["decision"] = self.decision
        if self.metrics:
            d["metrics"] = self.metrics
        if self.refs:
            d["refs"] = self.refs
        if self.sovereignty:
            d["sovereignty"] = self.sovereignty

        return d

    def to_json(self) -> str:
        """Serialize to compact JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'), ensure_ascii=False)


# =============================================================================
# Activity Writer
# =============================================================================

class ActivityWriter:
    """
    Append-only writer for activity events.

    Thread-safe, writes events to a JSONL file.

    Attributes:
        log_path: Path to the events.jsonl file
        run_id: Current run identifier (set for all events)
    """

    def __init__(self, workspace: Path, run_id: Optional[str] = None):
        """
        Initialize activity writer.

        Args:
            workspace: KOAS workspace directory
            run_id: Optional run identifier (auto-generated if not provided)
        """
        self.log_path = workspace / ".KOAS" / "activity" / "events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if run_id:
            self.run_id = run_id
        else:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self._lock = threading.Lock()
        self._event_count = 0

        logger.debug(f"[activity] Writer initialized: {self.log_path}, run_id={self.run_id}")

    def emit(self, event: ActivityEvent) -> None:
        """
        Emit an activity event to the stream.

        Thread-safe, appends event as JSON line.

        Args:
            event: The activity event to emit
        """
        event.run_id = self.run_id

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
            self._event_count += 1

        logger.debug(f"[activity] Event emitted: scope={event.scope}, phase={event.phase}")

    def emit_kernel_start(
        self,
        kernel_name: str,
        kernel_version: str,
        stage: int,
        input_hash: str = "",
    ) -> str:
        """
        Emit a kernel start event.

        Args:
            kernel_name: Name of the kernel
            kernel_version: Version of the kernel
            stage: Pipeline stage
            input_hash: Hash of kernel input

        Returns:
            Event ID for correlation
        """
        event = ActivityEvent(
            scope="docs.kernel",
            actor=Actor(type="system", id="koas"),
            phase="start",
            kernel=KernelInfo(name=kernel_name, version=kernel_version, stage=stage),
            io={"input_hash": input_hash} if input_hash else {},
        )
        self.emit(event)
        return event.event_id

    def emit_kernel_end(
        self,
        kernel_name: str,
        kernel_version: str,
        stage: int,
        success: bool,
        duration_ms: int,
        output_hash: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        cache_hit: bool = False,
        start_event_id: str = "",
    ) -> str:
        """
        Emit a kernel end event.

        Args:
            kernel_name: Name of the kernel
            kernel_version: Version of the kernel
            stage: Pipeline stage
            success: Whether kernel completed successfully
            duration_ms: Execution time in milliseconds
            output_hash: Hash of kernel output
            metrics: Additional metrics from kernel
            cache_hit: Whether result was from cache
            start_event_id: Event ID of corresponding start event

        Returns:
            Event ID
        """
        event = ActivityEvent(
            scope="docs.kernel",
            actor=Actor(type="system", id="koas"),
            phase="end",
            kernel=KernelInfo(name=kernel_name, version=kernel_version, stage=stage),
            io={"output_hash": output_hash} if output_hash else {},
            decision={"success": success, "cache_hit": cache_hit},
            metrics={"duration_ms": duration_ms, **(metrics or {})},
            refs={"start_event": start_event_id} if start_event_id else {},
        )
        self.emit(event)
        return event.event_id

    def emit_llm_call(
        self,
        model: str,
        cache_hit: bool,
        prompt_hash: str = "",
        response_hash: str = "",
        tokens: Optional[Dict[str, int]] = None,
        endpoint: str = "localhost:11434",
        duration_ms: int = 0,
    ) -> str:
        """
        Emit an LLM call event.

        Args:
            model: LLM model name
            cache_hit: Whether response was from cache
            prompt_hash: Hash of prompt
            response_hash: Hash of response
            tokens: Token usage metrics
            endpoint: LLM endpoint
            duration_ms: Call duration in milliseconds (0 for cache hits)

        Returns:
            Event ID
        """
        metrics = tokens.copy() if tokens else {}
        if duration_ms > 0:
            metrics["duration_ms"] = duration_ms

        event = ActivityEvent(
            scope="docs.llm",
            actor=Actor(type="system", id="koas"),
            phase="cache_hit" if cache_hit else "call",
            decision={"cache_hit": cache_hit, "model": model},
            refs={"prompt_hash": prompt_hash, "response_hash": response_hash},
            metrics=metrics,
            sovereignty={"local_only": True, "endpoint": endpoint},
        )
        self.emit(event)
        return event.event_id

    def emit_workflow_start(
        self,
        stages: List[int],
        actor: Optional[Actor] = None,
    ) -> str:
        """
        Emit a workflow start event.

        Args:
            stages: List of stages to execute
            actor: Actor initiating the workflow

        Returns:
            Event ID
        """
        event = ActivityEvent(
            scope="docs.workflow",
            actor=actor or Actor(type="system", id="koas"),
            phase="start",
            decision={"stages": stages},
        )
        self.emit(event)
        return event.event_id

    def emit_workflow_end(
        self,
        success: bool,
        duration_ms: int,
        kernels_run: int,
        actor: Optional[Actor] = None,
        start_event_id: str = "",
    ) -> str:
        """
        Emit a workflow end event.

        Args:
            success: Whether workflow completed successfully
            duration_ms: Total execution time
            kernels_run: Number of kernels executed
            actor: Actor that initiated the workflow
            start_event_id: Event ID of workflow start

        Returns:
            Event ID
        """
        event = ActivityEvent(
            scope="docs.workflow",
            actor=actor or Actor(type="system", id="koas"),
            phase="end",
            decision={"success": success},
            metrics={"duration_ms": duration_ms, "kernels_run": kernels_run},
            refs={"start_event": start_event_id} if start_event_id else {},
        )
        self.emit(event)
        return event.event_id

    @property
    def event_count(self) -> int:
        """Number of events emitted in this session."""
        return self._event_count


# =============================================================================
# Activity Reader
# =============================================================================

class ActivityReader:
    """
    Reader for activity event stream.

    Provides methods to read, filter, and analyze events.
    """

    def __init__(self, workspace: Path):
        """
        Initialize activity reader.

        Args:
            workspace: KOAS workspace directory
        """
        self.log_path = workspace / ".KOAS" / "activity" / "events.jsonl"

    def exists(self) -> bool:
        """Check if activity log exists."""
        return self.log_path.exists()

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all events from the stream.

        Returns:
            List of event dictionaries
        """
        if not self.exists():
            return []

        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"[activity] Skipping malformed line: {line[:50]}...")
        return events

    def iter_events(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over events lazily.

        Yields:
            Event dictionaries
        """
        if not self.exists():
            return

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def filter_by_scope(self, scope: str) -> List[Dict[str, Any]]:
        """
        Filter events by scope.

        Args:
            scope: Scope to filter by (e.g., "docs.kernel")

        Returns:
            List of matching events
        """
        return [e for e in self.read_all() if e.get("scope") == scope]

    def filter_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Filter events by run ID.

        Args:
            run_id: Run identifier

        Returns:
            List of matching events
        """
        return [e for e in self.read_all() if e.get("run_id") == run_id]

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a run.

        Args:
            run_id: Run identifier

        Returns:
            Summary dictionary with counts and metrics
        """
        events = self.filter_by_run(run_id)

        summary = {
            "run_id": run_id,
            "event_count": len(events),
            "scopes": {},
            "kernels": [],
            "llm_calls": 0,
            "cache_hits": 0,
            "duration_ms": 0,
        }

        for event in events:
            scope = event.get("scope", "unknown")
            summary["scopes"][scope] = summary["scopes"].get(scope, 0) + 1

            if scope == "docs.kernel" and event.get("phase") == "end":
                kernel = event.get("kernel", {})
                summary["kernels"].append(kernel.get("name", "unknown"))
                summary["duration_ms"] += event.get("metrics", {}).get("duration_ms", 0)

            if scope == "docs.llm":
                summary["llm_calls"] += 1
                if event.get("decision", {}).get("cache_hit"):
                    summary["cache_hits"] += 1

        return summary

    def count(self) -> int:
        """Count total events in stream."""
        if not self.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


# =============================================================================
# Global Activity Writer (Singleton Pattern)
# =============================================================================

_global_writer: Optional[ActivityWriter] = None
_global_lock = threading.Lock()


def init_activity_writer(workspace: Path, run_id: Optional[str] = None) -> ActivityWriter:
    """
    Initialize the global activity writer.

    Call this once at the start of a KOAS run.

    Args:
        workspace: KOAS workspace directory
        run_id: Optional run identifier

    Returns:
        ActivityWriter instance
    """
    global _global_writer
    with _global_lock:
        _global_writer = ActivityWriter(workspace, run_id)
    return _global_writer


def get_activity_writer() -> Optional[ActivityWriter]:
    """
    Get the global activity writer.

    Returns:
        ActivityWriter if initialized, None otherwise
    """
    return _global_writer


def emit_activity(event: ActivityEvent) -> None:
    """
    Emit an activity event using the global writer.

    No-op if writer is not initialized.

    Args:
        event: Event to emit
    """
    writer = get_activity_writer()
    if writer:
        writer.emit(event)


# =============================================================================
# Context Managers for Kernel/Workflow Tracking
# =============================================================================

@contextmanager
def track_kernel(
    kernel_name: str,
    kernel_version: str,
    stage: int,
    input_hash: str = "",
):
    """
    Context manager for tracking kernel execution.

    Automatically emits start/end events with timing.

    Usage:
        with track_kernel("doc_extract", "1.2.0", 2, input_hash) as tracker:
            # ... kernel execution ...
            tracker.add_metrics({"sentences": 42})

    Args:
        kernel_name: Name of the kernel
        kernel_version: Version of the kernel
        stage: Pipeline stage
        input_hash: Hash of kernel input
    """
    writer = get_activity_writer()
    start_time = datetime.now(timezone.utc)
    start_event_id = ""
    metrics = {}
    cache_hit = False
    success = True

    class Tracker:
        def add_metrics(self, m: Dict[str, Any]):
            metrics.update(m)

        def set_cache_hit(self, hit: bool):
            nonlocal cache_hit
            cache_hit = hit

        def set_success(self, s: bool):
            nonlocal success
            success = s

    tracker = Tracker()

    if writer:
        start_event_id = writer.emit_kernel_start(kernel_name, kernel_version, stage, input_hash)

    try:
        yield tracker
    except Exception:
        success = False
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        if writer:
            writer.emit_kernel_end(
                kernel_name=kernel_name,
                kernel_version=kernel_version,
                stage=stage,
                success=success,
                duration_ms=duration_ms,
                metrics=metrics,
                cache_hit=cache_hit,
                start_event_id=start_event_id,
            )


@contextmanager
def track_workflow(stages: List[int], actor: Optional[Actor] = None):
    """
    Context manager for tracking workflow execution.

    Automatically emits workflow start/end events.

    Usage:
        with track_workflow([1, 2, 3]) as tracker:
            # ... run stages ...
            tracker.add_kernel("doc_metadata")

    Args:
        stages: List of stages to execute
        actor: Actor initiating the workflow
    """
    writer = get_activity_writer()
    start_time = datetime.now(timezone.utc)
    start_event_id = ""
    kernels_run = 0
    success = True

    class WorkflowTracker:
        def add_kernel(self, name: str):
            nonlocal kernels_run
            kernels_run += 1

        def set_success(self, s: bool):
            nonlocal success
            success = s

    tracker = WorkflowTracker()

    if writer:
        start_event_id = writer.emit_workflow_start(stages, actor)

    try:
        yield tracker
    except Exception:
        success = False
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        if writer:
            writer.emit_workflow_end(
                success=success,
                duration_ms=duration_ms,
                kernels_run=kernels_run,
                actor=actor,
                start_event_id=start_event_id,
            )
