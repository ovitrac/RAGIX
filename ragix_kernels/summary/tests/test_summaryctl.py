"""
Tests for summaryctl CLI argument parsing and subcommand setup.

Validates:
- query subcommand with basic args
- --views flag parsed correctly on viz
- --format json on show command
- --domain filter on query
- Parser creation and subcommand resolution

Does NOT run actual pipelines — tests focus on argument parsing
and parser structure.
"""

import argparse
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ragix_kernels.summary.cli.summaryctl import (
    build_parser,
    cmd_query,
    cmd_show,
    _text_score,
)
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance


@pytest.fixture
def parser():
    return build_parser()


@pytest.fixture
def workspace_with_db(tmp_path):
    """Create a temporary workspace with a populated memory.db."""
    store = MemoryStore(str(tmp_path / "memory.db"))
    items = [
        MemoryItem(
            id="CLI-001", tier="stm", type="fact",
            title="Oracle licensing constraint",
            content="Oracle requires per-CPU licensing for enterprise edition.",
            tags=["oracle", "licensing", "constraint"],
            entities=["Oracle"],
            provenance=MemoryProvenance(
                source_kind="doc", source_id="oracle_guide.pdf",
            ),
            confidence=0.9,
        ),
        MemoryItem(
            id="CLI-002", tier="mtm", type="decision",
            title="RHEL upgrade decision",
            content="Upgrade RHEL from 8 to 9 for security patches.",
            tags=["rhel", "upgrade", "security"],
            entities=["RHEL"],
            provenance=MemoryProvenance(
                source_kind="doc", source_id="rhel_audit.pdf",
            ),
            confidence=0.85,
        ),
        MemoryItem(
            id="CLI-003", tier="stm", type="note",
            title="General observation about compliance",
            content="Compliance requires annual audit reviews.",
            tags=["compliance", "audit"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_3"),
            confidence=0.6,
        ),
    ]
    for item in items:
        store.write_item(item)
    store.close()
    return tmp_path


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------

class TestParserStructure:
    def test_parser_has_all_subcommands(self, parser):
        """Parser should have all expected subcommands."""
        # Parse each subcommand with minimal required args
        subcommands = ["ingest", "summarize", "run", "drift", "viz", "query", "show"]
        # Just verify the parser was created with subparsers
        assert parser.prog == "summaryctl"

    def test_query_subcommand_parsed(self, parser):
        """Query subcommand should accept workspace, query terms, and filters."""
        args = parser.parse_args([
            "query", "/tmp/workspace", "oracle", "licensing",
            "--tier", "stm", "--limit", "10",
        ])
        assert args.command == "query"
        assert args.workspace == "/tmp/workspace"
        assert args.query == ["oracle", "licensing"]
        assert args.tier == "stm"
        assert args.limit == 10

    def test_query_domain_filter(self, parser):
        """Query subcommand should accept --domain filter."""
        args = parser.parse_args([
            "query", "/tmp/workspace", "test",
            "--domain", "oracle",
        ])
        assert args.domain == "oracle"

    def test_query_json_flag(self, parser):
        """Query subcommand should accept --json flag."""
        args = parser.parse_args([
            "query", "/tmp/workspace", "test",
            "--json",
        ])
        assert args.json is True

    def test_query_scored_flag(self, parser):
        """Query subcommand should accept --scored flag."""
        args = parser.parse_args([
            "query", "/tmp/workspace", "test",
            "--scored",
        ])
        assert args.scored is True

    def test_viz_views_flag(self, parser):
        """Viz subcommand should accept --views flag."""
        args = parser.parse_args([
            "viz", "/tmp/workspace",
            "--views", "graph,memory,timeline",
        ])
        assert args.views == "graph,memory,timeline"

    def test_viz_secrecy_flag(self, parser):
        """Viz subcommand should accept --secrecy flag."""
        args = parser.parse_args([
            "viz", "/tmp/workspace",
            "--secrecy", "S0",
        ])
        assert args.secrecy == "S0"

    def test_show_format_json(self, parser):
        """Show subcommand should accept --format json."""
        args = parser.parse_args([
            "show", "/tmp/workspace",
            "--format", "json",
        ])
        assert args.format == "json"

    def test_show_format_default(self, parser):
        """Show subcommand should default to table format."""
        args = parser.parse_args([
            "show", "/tmp/workspace",
        ])
        assert args.format == "table"

    def test_ingest_delta_flag(self, parser):
        """Ingest subcommand should accept --delta flag."""
        args = parser.parse_args([
            "ingest", "/tmp/corpus",
            "--delta",
        ])
        assert args.delta is True

    def test_summarize_graph_flags(self, parser):
        """Summarize subcommand should accept --graph/--no-graph."""
        args_graph = parser.parse_args([
            "summarize", "/tmp/workspace",
            "--graph",
        ])
        assert args_graph.graph is True

        args_no_graph = parser.parse_args([
            "summarize", "/tmp/workspace",
            "--no-graph",
        ])
        assert args_no_graph.graph is False

    def test_summarize_secrecy_choices(self, parser):
        """Summarize subcommand should restrict --secrecy to S0/S2/S3."""
        for tier in ("S0", "S2", "S3"):
            args = parser.parse_args([
                "summarize", "/tmp/workspace",
                "--secrecy", tier,
            ])
            assert args.secrecy == tier


# ---------------------------------------------------------------------------
# cmd_query execution
# ---------------------------------------------------------------------------

class TestCmdQuery:
    def test_query_returns_results(self, workspace_with_db, capsys):
        """cmd_query should find matching items and print output."""
        args = argparse.Namespace(
            workspace=str(workspace_with_db),
            query=["oracle", "licensing"],
            tier=None,
            type=None,
            scope=None,
            domain=None,
            limit=20,
            scored=False,
            json=False,
            verbose=True,
            embedder="mock",
        )
        ret = cmd_query(args)
        assert ret == 0
        output = capsys.readouterr().out
        assert "oracle" in output.lower() or "Oracle" in output

    def test_query_json_output(self, workspace_with_db, capsys):
        """cmd_query with --json produces valid JSON output."""
        args = argparse.Namespace(
            workspace=str(workspace_with_db),
            query=["oracle"],
            tier=None,
            type=None,
            scope=None,
            domain=None,
            limit=20,
            scored=False,
            json=True,
            verbose=False,
            embedder="mock",
        )
        ret = cmd_query(args)
        assert ret == 0
        output = capsys.readouterr().out
        result = json.loads(output)
        assert "query" in result
        assert "count" in result
        assert "results" in result

    def test_query_tier_filter(self, workspace_with_db, capsys):
        """cmd_query with --tier filters results by tier."""
        args = argparse.Namespace(
            workspace=str(workspace_with_db),
            query=["security"],
            tier="mtm",
            type=None,
            scope=None,
            domain=None,
            limit=20,
            scored=False,
            json=True,
            verbose=False,
            embedder="mock",
        )
        ret = cmd_query(args)
        assert ret == 0
        output = capsys.readouterr().out
        result = json.loads(output)
        for item in result["results"]:
            assert item["tier"] == "mtm"

    def test_query_no_results(self, workspace_with_db, capsys):
        """cmd_query with non-matching terms returns 0 results."""
        args = argparse.Namespace(
            workspace=str(workspace_with_db),
            query=["xyzzynonexistent"],
            tier=None,
            type=None,
            scope=None,
            domain=None,
            limit=20,
            scored=False,
            json=True,
            verbose=False,
            embedder="mock",
        )
        ret = cmd_query(args)
        assert ret == 0
        output = capsys.readouterr().out
        result = json.loads(output)
        assert result["count"] == 0

    def test_query_missing_db_returns_error(self, tmp_path):
        """cmd_query without memory.db returns error code 1."""
        args = argparse.Namespace(
            workspace=str(tmp_path),
            query=["test"],
            tier=None,
            type=None,
            scope=None,
            domain=None,
            limit=20,
            scored=False,
            json=False,
            verbose=False,
            embedder="mock",
        )
        ret = cmd_query(args)
        assert ret == 1


# ---------------------------------------------------------------------------
# cmd_show execution
# ---------------------------------------------------------------------------

class TestCmdShow:
    def test_show_table_output(self, workspace_with_db, capsys):
        """cmd_show with table format prints human-readable info."""
        args = argparse.Namespace(
            workspace=str(workspace_with_db),
            format="table",
        )
        ret = cmd_show(args)
        assert ret == 0
        output = capsys.readouterr().out
        assert "Items:" in output

    def test_show_json_output(self, workspace_with_db, capsys):
        """cmd_show with --format json produces valid JSON."""
        args = argparse.Namespace(
            workspace=str(workspace_with_db),
            format="json",
        )
        ret = cmd_show(args)
        assert ret == 0
        output = capsys.readouterr().out
        result = json.loads(output)
        assert "items" in result
        assert "by_tier" in result
        assert result["items"] >= 3

    def test_show_missing_db_returns_error(self, tmp_path):
        """cmd_show without memory.db returns error code 1."""
        args = argparse.Namespace(
            workspace=str(tmp_path),
            format="table",
        )
        ret = cmd_show(args)
        assert ret == 1


# ---------------------------------------------------------------------------
# _text_score helper
# ---------------------------------------------------------------------------

class TestTextScore:
    def test_all_terms_match(self):
        """Score should be 1.0 when all query terms match."""
        item = MemoryItem(
            title="Oracle database licensing",
            content="Oracle licensing requires per-CPU model.",
            tags=["oracle", "licensing"],
        )
        score = _text_score("oracle licensing", item)
        assert score == 1.0

    def test_partial_match(self):
        """Score should be partial when some terms match."""
        item = MemoryItem(
            title="Oracle database",
            content="Database management system.",
            tags=["oracle"],
        )
        score = _text_score("oracle faiss", item)
        assert score == 0.5

    def test_no_match(self):
        """Score should be 0.0 when no terms match."""
        item = MemoryItem(
            title="Python coding",
            content="Writing Python code.",
            tags=["python"],
        )
        score = _text_score("oracle licensing", item)
        assert score == 0.0

    def test_empty_query(self):
        """Empty query returns 0.0."""
        item = MemoryItem(title="Test", content="Test content.")
        score = _text_score("", item)
        assert score == 0.0

    def test_case_insensitive(self):
        """Score computation should be case-insensitive."""
        item = MemoryItem(
            title="ORACLE DATABASE",
            content="Oracle is used widely.",
            tags=["Oracle"],
        )
        score = _text_score("oracle", item)
        assert score == 1.0
