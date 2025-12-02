"""
RAGIX Experience Corpus - Hybrid RAG for Reasoning from Past Experiences

This module implements:
- ExperienceCorpus: Single corpus (global or project-level)
- HybridExperienceCorpus: Combines global (~/.ragix/) and project (.ragix/) corpora
- Experience-based retrieval for the REFLECT node

The corpus stores ReasoningEvents and enables keyword + recency search
to help the agent learn from past successes and failures.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-02
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from .reasoning_types import ReasoningEvent

logger = logging.getLogger(__name__)


class ExperienceCorpus:
    """
    Single experience corpus for storing and retrieving past reasoning events.

    Can be either global (~/.ragix/experience/) or project-level (.ragix/experience/).
    """

    def __init__(
        self,
        events_path: Path,
        max_age_days: int = 30,
        max_events: int = 1000,
    ):
        """
        Initialize an experience corpus.

        Args:
            events_path: Path to events.jsonl file
            max_age_days: Maximum age of events to load (older are ignored)
            max_events: Maximum number of events to keep in memory
        """
        self.events_path = Path(events_path)
        self.max_age_days = max_age_days
        self.max_events = max_events
        self._events: List[ReasoningEvent] = []
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load events from disk."""
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self):
        """Load events from the JSONL file."""
        self._events = []

        if not self.events_path.exists():
            logger.debug(f"Experience corpus not found: {self.events_path}")
            return

        cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)

        try:
            with open(self.events_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        timestamp_str = data.get("timestamp", "")

                        # Parse timestamp and filter by age
                        try:
                            event_time = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )
                            if event_time.replace(tzinfo=None) < cutoff:
                                continue  # Skip old events
                        except (ValueError, TypeError):
                            pass  # Keep events with unparseable timestamps

                        event = ReasoningEvent.from_dict(data)
                        self._events.append(event)

                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Skipping malformed line {line_num}: {e}")
                        continue

            # Prune to max_events (keep most recent)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

            logger.info(f"Loaded {len(self._events)} events from {self.events_path}")

        except Exception as e:
            logger.error(f"Failed to load experience corpus: {e}")
            self._events = []

    def reload(self):
        """Force reload from disk."""
        self._loaded = False
        self._ensure_loaded()

    def search(
        self,
        query: str,
        top_k: int = 5,
        outcome_filter: Optional[str] = None,
    ) -> List[Tuple[float, ReasoningEvent]]:
        """
        Search for relevant past events.

        Uses simple keyword + recency scoring. Can be upgraded to embeddings later.

        Args:
            query: Search query (goal, error, step description)
            top_k: Number of results to return
            outcome_filter: Optional filter ("success" or "failure")

        Returns:
            List of (score, event) tuples, sorted by relevance
        """
        self._ensure_loaded()

        if not self._events:
            return []

        query_terms = set(query.lower().split())

        scored: List[Tuple[float, ReasoningEvent]] = []

        for event in self._events:
            # Apply outcome filter
            if outcome_filter and event.outcome_status != outcome_filter:
                continue

            # Keyword score
            event_text = event.get_searchable_text()
            event_terms = set(event_text.split())
            keyword_overlap = len(query_terms & event_terms)

            if keyword_overlap == 0:
                continue  # Skip events with no overlap

            keyword_score = keyword_overlap / max(len(query_terms), 1)

            # Recency score (0-1, higher = more recent)
            try:
                event_time = datetime.fromisoformat(
                    event.timestamp.replace("Z", "+00:00")
                )
                age_days = (datetime.utcnow() - event_time.replace(tzinfo=None)).days
                recency_score = max(0, 1 - age_days / self.max_age_days)
            except (ValueError, TypeError, AttributeError):
                recency_score = 0.5  # Default for unparseable

            # Success bonus (prefer successful recoveries for learning)
            if event.outcome_status == "success":
                success_bonus = 1.5
            elif event.outcome_status == "failure" and event.llm_critique:
                success_bonus = 1.2  # Failures with critiques are also valuable
            else:
                success_bonus = 1.0

            # Combined score
            total_score = (keyword_score * 3 + recency_score) * success_bonus

            scored.append((total_score, event))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[:top_k]

    def append(self, event: ReasoningEvent):
        """
        Append a new event to the corpus.

        Writes to disk immediately and adds to in-memory cache.
        """
        self._ensure_loaded()

        # Ensure parent directory exists
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        try:
            with open(self.events_path, "a") as f:
                f.write(event.to_jsonl() + "\n")
        except Exception as e:
            logger.error(f"Failed to append event: {e}")
            return

        # Add to memory
        self._events.append(event)

        # Prune if needed
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]

    def get_stats(self) -> dict:
        """Get corpus statistics."""
        self._ensure_loaded()

        if not self._events:
            return {"count": 0, "path": str(self.events_path)}

        success_count = sum(1 for e in self._events if e.outcome_status == "success")
        failure_count = sum(1 for e in self._events if e.outcome_status == "failure")

        return {
            "count": len(self._events),
            "successes": success_count,
            "failures": failure_count,
            "path": str(self.events_path),
        }


class HybridExperienceCorpus:
    """
    Combines global (~/.ragix/experience/) and project (.ragix/experience/) corpora.

    Similar to CLAUDE.md pattern:
    - Global: General patterns, cross-project learnings (90 days retention)
    - Project: Project-specific history and context (30 days retention)

    Project results are prioritized in search results.
    """

    GLOBAL_PATH = Path.home() / ".ragix" / "experience" / "events.jsonl"

    def __init__(
        self,
        project_path: Optional[Path] = None,
        global_max_age_days: int = 90,
        project_max_age_days: int = 30,
    ):
        """
        Initialize hybrid corpus.

        Args:
            project_path: Root of the project (contains .ragix/)
            global_max_age_days: Retention for global corpus
            project_max_age_days: Retention for project corpus
        """
        self.global_corpus = ExperienceCorpus(
            self.GLOBAL_PATH,
            max_age_days=global_max_age_days,
            max_events=2000,
        )

        if project_path:
            project_events = Path(project_path) / ".ragix" / "experience" / "events.jsonl"
            self.project_corpus = ExperienceCorpus(
                project_events,
                max_age_days=project_max_age_days,
                max_events=500,
            )
        else:
            self.project_corpus = None

        self.project_path = project_path

    def search(
        self,
        query: str,
        top_k: int = 5,
        project_weight: float = 1.5,
    ) -> str:
        """
        Search both corpora with project results prioritized.

        Args:
            query: Search query
            top_k: Total number of results to return
            project_weight: Score multiplier for project results

        Returns:
            Formatted context string for LLM injection
        """
        results: List[Tuple[float, str, ReasoningEvent]] = []

        # Search project corpus (higher priority)
        if self.project_corpus:
            project_results = self.project_corpus.search(query, top_k=top_k)
            for score, event in project_results:
                results.append((score * project_weight, "PROJECT", event))

        # Search global corpus
        global_results = self.global_corpus.search(query, top_k=top_k)
        for score, event in global_results:
            results.append((score, "GLOBAL", event))

        # Sort by weighted score
        results.sort(key=lambda x: x[0], reverse=True)

        # Take top_k overall
        results = results[:top_k]

        if not results:
            return "[No relevant past experiences found]"

        # Format for LLM
        formatted_lines = []
        for score, source, event in results:
            lines = [f"[{source}] {event.timestamp[:10]} (relevance: {score:.2f})"]
            lines.append(f"  Goal: {event.goal[:100]}...")

            if event.step_description:
                lines.append(f"  Step: {event.step_description[:80]}...")

            lines.append(f"  Outcome: {event.outcome_status or 'unknown'}")

            if event.error:
                lines.append(f"  Error: {event.error[:100]}...")

            if event.llm_critique:
                lines.append(f"  Lesson: {event.llm_critique[:150]}...")

            formatted_lines.append("\n".join(lines))

        return "\n\n".join(formatted_lines)

    def append(self, event: ReasoningEvent, to_global: bool = False):
        """
        Append event to appropriate corpus.

        Args:
            event: The event to store
            to_global: If True, store in global corpus; otherwise project
        """
        if to_global or self.project_corpus is None:
            self.global_corpus.append(event)
        else:
            self.project_corpus.append(event)

    def append_to_both(self, event: ReasoningEvent):
        """Append event to both global and project corpora."""
        self.global_corpus.append(event)
        if self.project_corpus:
            self.project_corpus.append(event)

    def get_stats(self) -> dict:
        """Get combined statistics."""
        stats = {
            "global": self.global_corpus.get_stats(),
            "project": self.project_corpus.get_stats() if self.project_corpus else None,
        }

        total = stats["global"]["count"]
        if stats["project"]:
            total += stats["project"]["count"]

        stats["total_events"] = total
        return stats

    def search_failures(self, query: str, top_k: int = 3) -> str:
        """Search specifically for past failures with lessons learned."""
        results: List[Tuple[float, str, ReasoningEvent]] = []

        if self.project_corpus:
            project_results = self.project_corpus.search(
                query, top_k=top_k, outcome_filter="failure"
            )
            for score, event in project_results:
                if event.llm_critique:  # Only include failures with critiques
                    results.append((score * 1.5, "PROJECT", event))

        global_results = self.global_corpus.search(
            query, top_k=top_k, outcome_filter="failure"
        )
        for score, event in global_results:
            if event.llm_critique:
                results.append((score, "GLOBAL", event))

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:top_k]

        if not results:
            return "[No relevant past failures found]"

        formatted = []
        for _, source, event in results:
            formatted.append(
                f"[{source}] Failed: {event.step_description or event.goal[:50]}...\n"
                f"  Error: {event.error or 'unknown'}\n"
                f"  Lesson: {event.llm_critique}"
            )

        return "\n\n".join(formatted)

    def search_successes(self, query: str, top_k: int = 3) -> str:
        """Search specifically for past successes as templates."""
        results: List[Tuple[float, str, ReasoningEvent]] = []

        if self.project_corpus:
            project_results = self.project_corpus.search(
                query, top_k=top_k, outcome_filter="success"
            )
            for score, event in project_results:
                results.append((score * 1.5, "PROJECT", event))

        global_results = self.global_corpus.search(
            query, top_k=top_k, outcome_filter="success"
        )
        for score, event in global_results:
            results.append((score, "GLOBAL", event))

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:top_k]

        if not results:
            return "[No relevant past successes found]"

        formatted = []
        for _, source, event in results:
            formatted.append(
                f"[{source}] Succeeded: {event.goal[:80]}...\n"
                f"  Tool: {event.tool or 'unknown'}\n"
                f"  Command: {event.tool_input or 'N/A'}"
            )

        return "\n\n".join(formatted)


def get_hybrid_corpus(project_path: Optional[Path] = None) -> HybridExperienceCorpus:
    """
    Factory function to get a hybrid experience corpus.

    Args:
        project_path: Optional project root path

    Returns:
        Configured HybridExperienceCorpus instance
    """
    return HybridExperienceCorpus(project_path=project_path)
