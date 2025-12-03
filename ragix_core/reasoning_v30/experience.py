"""
RAGIX v0.30 Experience Corpus

Hybrid experience storage for learning from past reasoning:
- ExperienceCorpus: Single corpus (global or project)
- HybridExperienceCorpus: Combines global (~/.ragix/) and project (.ragix/) corpora
- SessionTraceWriter: Per-session trace logging

Canonical layout:
    ~/.ragix/
      experience/
        events.jsonl        # Global experience
      traces/
        {session_id}.jsonl

    .ragix/
      experience/
        events.jsonl        # Project-specific
      reasoning_traces/
        {session_id}.jsonl

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any

from .types import ReasoningEvent

logger = logging.getLogger(__name__)


class ExperienceCorpus:
    """
    Single experience corpus with JSONL storage and TTL-based pruning.

    Events are stored in append-only JSONL format for simplicity and
    compatibility with streaming processing tools.
    """

    def __init__(self, root: Path, max_age_days: int = 90):
        """
        Initialize experience corpus.

        Args:
            root: Root directory for the corpus
            max_age_days: Maximum age of events to retain (TTL pruning)
        """
        self.root = Path(root).expanduser()
        self.events_path = self.root / "experience" / "events.jsonl"
        self.max_age = timedelta(days=max_age_days)

        # Ensure directory exists
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: ReasoningEvent) -> None:
        """
        Append an event to the corpus.

        Args:
            event: ReasoningEvent to store
        """
        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append event to corpus: {e}")

    def _load_recent(self) -> List[Dict[str, Any]]:
        """Load events within TTL window."""
        if not self.events_path.exists():
            return []

        cutoff = datetime.utcnow() - self.max_age
        events: List[Dict[str, Any]] = []

        try:
            with self.events_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ts_str = obj.get("timestamp", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts >= cutoff:
                                events.append(obj)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"Skipping malformed line {line_num}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to load experience corpus: {e}")

        return events

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Simple keyword-based search with recency weighting.

        Args:
            query: Search query (space-separated keywords)
            top_k: Maximum number of results

        Returns:
            Formatted text block suitable for LLM context
        """
        query_tokens = set(query.lower().split())
        candidates: List[tuple] = []

        for ev in self._load_recent():
            # Build searchable text blob
            blob = " ".join([
                ev.get("goal", ""),
                ev.get("step_description", "") or "",
                ev.get("error", "") or "",
                ev.get("llm_critique", "") or "",
            ]).lower()

            # Keyword score
            blob_tokens = set(blob.split())
            keyword_score = len(query_tokens & blob_tokens)

            if keyword_score > 0:
                # Recency score (0-1, higher = more recent)
                try:
                    ts_str = ev.get("timestamp", "")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    age_days = (datetime.utcnow() - ts).days
                    recency_score = max(0, 1 - age_days / self.max_age.days)
                except (ValueError, TypeError):
                    recency_score = 0

                # Success bonus (prefer successful recoveries)
                success_bonus = 1.5 if ev.get("outcome_status") == "success" else 1.0

                total_score = (keyword_score * 2 + recency_score) * success_bonus
                candidates.append((total_score, ev))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = [ev for _, ev in candidates[:top_k]]

        # Format for LLM
        if not selected:
            return ""

        lines = []
        for i, ev in enumerate(selected, 1):
            ts = ev.get("timestamp", "")[:10]  # Date only
            goal = ev.get("goal", "")[:80]
            lines.append(f"[{i}] {ts} — {goal}")

            if ev.get("error"):
                error = ev["error"][:100]
                lines.append(f"    error: {error}")

            if ev.get("llm_critique"):
                critique = ev["llm_critique"][:150]
                lines.append(f"    lesson: {critique}")

        return "\n".join(lines)

    def get_event_count(self) -> int:
        """Get total number of events in corpus."""
        return len(self._load_recent())

    def prune(self) -> int:
        """
        Remove events older than TTL.

        Returns:
            Number of events removed
        """
        if not self.events_path.exists():
            return 0

        recent = self._load_recent()
        original_count = 0

        # Count original events
        try:
            with self.events_path.open("r", encoding="utf-8") as f:
                original_count = sum(1 for line in f if line.strip())
        except Exception:
            pass

        # Rewrite with only recent events
        try:
            with self.events_path.open("w", encoding="utf-8") as f:
                for ev in recent:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to prune corpus: {e}")
            return 0

        removed = original_count - len(recent)
        if removed > 0:
            logger.info(f"Pruned {removed} old events from corpus")

        return removed


class HybridExperienceCorpus:
    """
    Combines global (~/.ragix/) and project (.ragix/) experience corpora.

    Similar to CLAUDE.md pattern:
    - Global: General patterns, cross-project learnings (90 day TTL)
    - Project: Project-specific history and context (30 day TTL)

    Project-specific events are prioritized in search results.
    """

    DEFAULT_GLOBAL_ROOT = Path.home() / ".ragix"

    def __init__(
        self,
        global_root: Optional[Path] = None,
        project_root: Optional[Path] = None,
        global_max_age_days: int = 90,
        project_max_age_days: int = 30
    ):
        """
        Initialize hybrid corpus.

        Args:
            global_root: Root for global corpus (default: ~/.ragix)
            project_root: Root for project corpus (default: .ragix in cwd)
            global_max_age_days: TTL for global events
            project_max_age_days: TTL for project events
        """
        global_root = global_root or self.DEFAULT_GLOBAL_ROOT
        project_root = project_root or Path(".ragix")

        self.global_corpus = ExperienceCorpus(
            global_root,
            max_age_days=global_max_age_days
        )
        self.project_corpus = ExperienceCorpus(
            project_root,
            max_age_days=project_max_age_days
        )

    def append(
        self,
        event: ReasoningEvent,
        to_project: bool = True,
        to_global: bool = True
    ) -> None:
        """
        Append event to one or both corpora.

        Args:
            event: ReasoningEvent to store
            to_project: Store in project corpus
            to_global: Store in global corpus
        """
        if to_project:
            self.project_corpus.append(event)
        if to_global:
            self.global_corpus.append(event)

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search both corpora with project results prioritized.

        Args:
            query: Search query
            top_k: Maximum results per corpus

        Returns:
            Formatted context string for LLM
        """
        project_results = self.project_corpus.search(query, top_k=top_k)
        global_results = self.global_corpus.search(query, top_k=top_k)

        sections = []

        if project_results:
            sections.append("### Project experience\n" + project_results)

        if global_results:
            sections.append("### Global experience\n" + global_results)

        if not sections:
            return "[No relevant past experiences found]"

        return "\n\n".join(sections)

    def prune_all(self) -> Dict[str, int]:
        """
        Prune both corpora.

        Returns:
            Dict with 'global' and 'project' counts of removed events
        """
        return {
            "global": self.global_corpus.prune(),
            "project": self.project_corpus.prune(),
        }


class SessionTraceWriter:
    """
    Per-session trace logging to JSONL files.

    Each session gets its own trace file for detailed debugging
    and replay capabilities.
    """

    def __init__(self, traces_dir: Path, session_id: str, max_events: int = 1000):
        """
        Initialize session trace writer.

        Args:
            traces_dir: Directory for trace files
            session_id: Unique session identifier
            max_events: Maximum events per session (prevents unbounded growth)
        """
        self.traces_dir = Path(traces_dir).expanduser()
        self.session_id = session_id
        self.max_events = max_events
        self.event_count = 0

        # Create traces directory
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Trace file path
        self.trace_path = self.traces_dir / f"{session_id}.jsonl"

    def write(self, event: ReasoningEvent) -> bool:
        """
        Write event to session trace.

        Args:
            event: ReasoningEvent to log

        Returns:
            True if written, False if max_events reached
        """
        if self.event_count >= self.max_events:
            logger.warning(f"Session {self.session_id} reached max_events ({self.max_events})")
            return False

        try:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self.event_count += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to write trace event: {e}")
            return False

    def write_dict(self, data: Dict[str, Any]) -> bool:
        """
        Write arbitrary dict to session trace.

        Useful for logging graph state, node transitions, etc.
        """
        if self.event_count >= self.max_events:
            return False

        try:
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.utcnow().isoformat()

            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.event_count += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to write trace dict: {e}")
            return False

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all events from session trace."""
        if not self.trace_path.exists():
            return []

        events = []
        try:
            with self.trace_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Failed to read trace: {e}")

        return events
