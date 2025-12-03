"""
Tests for reasoning_v30 experience module.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta

from ragix_core.reasoning_v30.experience import (
    ExperienceCorpus,
    HybridExperienceCorpus,
    SessionTraceWriter,
)
from ragix_core.reasoning_v30.types import ReasoningEvent


class TestExperienceCorpus:
    """Tests for ExperienceCorpus class."""

    def test_append_and_search(self, tmp_path: Path):
        """Test appending events and searching."""
        corpus = ExperienceCorpus(tmp_path, max_age_days=30)

        # Add events
        event1 = ReasoningEvent.create_now(
            session_id="s1",
            event_type="execution",
            goal="Find Python files",
            outcome_status="success",
        )
        event2 = ReasoningEvent.create_now(
            session_id="s2",
            event_type="execution",
            goal="Search for TODO comments",
            error="File not found",
            outcome_status="failure",
        )
        corpus.append(event1)
        corpus.append(event2)

        # Search
        results = corpus.search("Python files")
        assert "Python" in results

        results = corpus.search("TODO comments")
        assert "TODO" in results

    def test_empty_search(self, tmp_path: Path):
        """Test search with no results."""
        corpus = ExperienceCorpus(tmp_path)
        results = corpus.search("nonexistent query")
        assert results == ""

    def test_event_count(self, tmp_path: Path):
        """Test getting event count."""
        corpus = ExperienceCorpus(tmp_path)

        for i in range(5):
            event = ReasoningEvent.create_now(
                session_id=f"s{i}",
                event_type="execution",
                goal=f"Task {i}",
            )
            corpus.append(event)

        assert corpus.get_event_count() == 5

    def test_prune_old_events(self, tmp_path: Path):
        """Test TTL pruning."""
        corpus = ExperienceCorpus(tmp_path, max_age_days=7)

        # Add old event (manually create with old timestamp)
        old_event = {
            "timestamp": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "session_id": "old",
            "event_type": "execution",
            "goal": "Old task",
        }
        with corpus.events_path.open("a") as f:
            f.write(json.dumps(old_event) + "\n")

        # Add recent event
        recent = ReasoningEvent.create_now(
            session_id="recent",
            event_type="execution",
            goal="Recent task",
        )
        corpus.append(recent)

        # Prune
        removed = corpus.prune()
        assert removed == 1
        assert corpus.get_event_count() == 1


class TestHybridExperienceCorpus:
    """Tests for HybridExperienceCorpus class."""

    def test_append_to_both(self, tmp_path: Path):
        """Test appending to both corpora."""
        global_root = tmp_path / "global"
        project_root = tmp_path / "project"

        corpus = HybridExperienceCorpus(
            global_root=global_root,
            project_root=project_root,
        )

        event = ReasoningEvent.create_now(
            session_id="test",
            event_type="execution",
            goal="Test task",
        )
        corpus.append(event, to_project=True, to_global=True)

        assert corpus.global_corpus.get_event_count() == 1
        assert corpus.project_corpus.get_event_count() == 1

    def test_append_project_only(self, tmp_path: Path):
        """Test appending to project only."""
        corpus = HybridExperienceCorpus(
            global_root=tmp_path / "global",
            project_root=tmp_path / "project",
        )

        event = ReasoningEvent.create_now(
            session_id="test",
            event_type="execution",
            goal="Project task",
        )
        corpus.append(event, to_project=True, to_global=False)

        assert corpus.global_corpus.get_event_count() == 0
        assert corpus.project_corpus.get_event_count() == 1

    def test_search_combined(self, tmp_path: Path):
        """Test searching both corpora."""
        corpus = HybridExperienceCorpus(
            global_root=tmp_path / "global",
            project_root=tmp_path / "project",
        )

        # Add to project
        project_event = ReasoningEvent.create_now(
            session_id="p1",
            event_type="execution",
            goal="Project Python task",
        )
        corpus.append(project_event, to_project=True, to_global=False)

        # Add to global
        global_event = ReasoningEvent.create_now(
            session_id="g1",
            event_type="execution",
            goal="Global Python task",
        )
        corpus.append(global_event, to_project=False, to_global=True)

        results = corpus.search("Python")
        assert "Project experience" in results
        assert "Global experience" in results

    def test_no_results(self, tmp_path: Path):
        """Test search with no matches."""
        corpus = HybridExperienceCorpus(
            global_root=tmp_path / "global",
            project_root=tmp_path / "project",
        )

        results = corpus.search("nonexistent")
        assert "No relevant past experiences" in results


class TestSessionTraceWriter:
    """Tests for SessionTraceWriter class."""

    def test_write_event(self, tmp_path: Path):
        """Test writing events."""
        writer = SessionTraceWriter(tmp_path, "test-session")

        event = ReasoningEvent.create_now(
            session_id="test-session",
            event_type="execution",
            goal="Test",
        )
        result = writer.write(event)

        assert result is True
        assert writer.event_count == 1
        assert writer.trace_path.exists()

    def test_write_dict(self, tmp_path: Path):
        """Test writing arbitrary dicts."""
        writer = SessionTraceWriter(tmp_path, "test-session")

        result = writer.write_dict({"type": "node_transition", "from": "A", "to": "B"})

        assert result is True
        events = writer.read_all()
        assert len(events) == 1
        assert events[0]["type"] == "node_transition"

    def test_max_events_limit(self, tmp_path: Path):
        """Test max events limit."""
        writer = SessionTraceWriter(tmp_path, "test-session", max_events=5)

        for i in range(10):
            event = ReasoningEvent.create_now(
                session_id="test",
                event_type="execution",
                goal=f"Task {i}",
            )
            writer.write(event)

        assert writer.event_count == 5
        events = writer.read_all()
        assert len(events) == 5

    def test_read_all(self, tmp_path: Path):
        """Test reading all events."""
        writer = SessionTraceWriter(tmp_path, "test-session")

        for i in range(3):
            event = ReasoningEvent.create_now(
                session_id="test",
                event_type="execution",
                goal=f"Task {i}",
            )
            writer.write(event)

        events = writer.read_all()
        assert len(events) == 3
        assert events[0]["goal"] == "Task 0"
        assert events[2]["goal"] == "Task 2"
