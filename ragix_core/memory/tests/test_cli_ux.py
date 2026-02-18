"""
Tests for CLI UX improvements — env vars, init, push/pull, serve.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-18
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helper: run CLI as subprocess (isolates argparse / sys.exit)
# ---------------------------------------------------------------------------

def _run_cli(*args: str, input_text: str = "", env_extra: dict = None) -> subprocess.CompletedProcess:
    """Run ragix-memory CLI in a subprocess and capture output."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "ragix_core.memory.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# _env_int defensive parser
# ---------------------------------------------------------------------------

class TestEnvInt:
    """Test the _env_int helper."""

    def test_env_int_returns_default_when_unset(self):
        from ragix_core.memory.cli import _env_int
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _env_int("NONEXISTENT_VAR", 42) == 42

    def test_env_int_parses_valid_value(self):
        from ragix_core.memory.cli import _env_int
        with mock.patch.dict(os.environ, {"TEST_VAR": "100"}):
            assert _env_int("TEST_VAR", 42) == 100

    def test_env_int_returns_default_on_bad_value(self):
        from ragix_core.memory.cli import _env_int
        with mock.patch.dict(os.environ, {"TEST_VAR": "not_a_number"}):
            assert _env_int("TEST_VAR", 42) == 42

    def test_env_int_returns_default_on_empty(self):
        from ragix_core.memory.cli import _env_int
        with mock.patch.dict(os.environ, {"TEST_VAR": ""}):
            assert _env_int("TEST_VAR", 42) == 42


# ---------------------------------------------------------------------------
# RAGIX_MEMORY_DB env var
# ---------------------------------------------------------------------------

class TestEnvVarDb:
    """Test RAGIX_MEMORY_DB environment variable support."""

    def test_env_var_db_used_in_help(self, tmp_path):
        """When RAGIX_MEMORY_DB is set, it appears in the default."""
        db = str(tmp_path / "test.db")
        result = _run_cli("--help", env_extra={"RAGIX_MEMORY_DB": db})
        assert result.returncode == 0
        # Help text may wrap the path across lines; normalize whitespace
        normalized = " ".join(result.stdout.split())
        assert "test.db" in normalized

    def test_env_var_db_used_by_init(self, tmp_path):
        """init creates workspace, then stats can use the env var."""
        ws = tmp_path / "ws"
        result = _run_cli("init", str(ws))
        assert result.returncode == 0
        assert "Memory workspace initialized" in result.stdout

        db = str(ws / "memory.db")
        result = _run_cli("stats", env_extra={"RAGIX_MEMORY_DB": db})
        assert result.returncode == 0
        assert "Total items" in result.stdout


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------

class TestInit:
    """Test the init subcommand."""

    def test_init_creates_workspace(self, tmp_path):
        ws = tmp_path / "test-ws"
        result = _run_cli("init", str(ws))
        assert result.returncode == 0
        assert (ws / "memory.db").exists()
        assert (ws / "config.yaml").exists()
        assert (ws / ".gitignore").exists()

    def test_init_refuses_existing_without_force(self, tmp_path):
        ws = tmp_path / "test-ws"
        _run_cli("init", str(ws))
        result = _run_cli("init", str(ws))
        assert result.returncode != 0
        assert "--force" in result.stderr

    def test_init_force_reinitializes(self, tmp_path):
        ws = tmp_path / "test-ws"
        _run_cli("init", str(ws))
        result = _run_cli("init", str(ws), "--force")
        assert result.returncode == 0

    def test_init_default_path(self, tmp_path):
        """When no path given, defaults to .memory in CWD."""
        result = subprocess.run(
            [sys.executable, "-m", "ragix_core.memory.cli", "init"],
            capture_output=True, text=True,
            cwd=str(tmp_path), timeout=30,
        )
        assert result.returncode == 0
        assert (tmp_path / ".memory" / "memory.db").exists()


# ---------------------------------------------------------------------------
# push = pipe alias
# ---------------------------------------------------------------------------

class TestPushAlias:
    """Test that push and pipe are equivalent."""

    def test_push_shows_in_help(self):
        result = _run_cli("push", "--help")
        assert result.returncode == 0
        assert "query" in result.stdout

    def test_push_and_pipe_share_args(self):
        """Both parsers accept --budget, --source, etc."""
        for cmd in ["push", "pipe"]:
            result = _run_cli(cmd, "--help")
            assert "--budget" in result.stdout
            assert "--source" in result.stdout
            assert "--workspace" in result.stdout


# ---------------------------------------------------------------------------
# pull command
# ---------------------------------------------------------------------------

class TestPull:
    """Test the pull subcommand."""

    def test_pull_empty_stdin_exits_error(self, tmp_path):
        ws = tmp_path / "pull-empty"
        _run_cli("init", str(ws))
        db = str(ws / "memory.db")
        result = _run_cli("--db", db, "pull", input_text="")
        assert result.returncode != 0
        assert "No input" in result.stderr

    def test_pull_stores_single_note(self, tmp_path):
        ws = tmp_path / "pull-ws"
        _run_cli("init", str(ws))
        db = str(ws / "memory.db")

        result = _run_cli(
            "--db", db,
            "pull", "--tags", "test", "--title", "Test note",
            input_text="This is a test memory item from pull.",
        )
        assert result.returncode == 0
        assert "Stored" in result.stderr

        # Verify it's searchable
        result = _run_cli("--db", db, "search", "test memory")
        assert result.returncode == 0
        assert "Test note" in result.stdout

    def test_pull_auto_title_with_timestamp(self, tmp_path):
        ws = tmp_path / "pull-ts"
        _run_cli("init", str(ws))
        db = str(ws / "memory.db")

        result = _run_cli(
            "--db", db, "pull",
            input_text="Content without explicit title.",
        )
        assert result.returncode == 0
        assert "LLM capture" in result.stderr

    def test_pull_long_content_chunked(self, tmp_path):
        ws = tmp_path / "pull-chunk"
        _run_cli("init", str(ws))
        db = str(ws / "memory.db")

        # Generate content > 2000 chars
        long_text = "Important fact. " * 200  # ~3200 chars
        result = _run_cli(
            "--db", db,
            "pull", "--tags", "long", "--title", "Long input",
            input_text=long_text,
        )
        assert result.returncode == 0
        assert "note(s)" in result.stderr


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------

class TestServe:
    """Test the serve subcommand."""

    def test_serve_shows_in_help(self):
        result = _run_cli("serve", "--help")
        assert result.returncode == 0
        assert "--fts-tokenizer" in result.stdout


# ---------------------------------------------------------------------------
# Budget env var
# ---------------------------------------------------------------------------

class TestBudgetEnvVar:
    """Test RAGIX_MEMORY_BUDGET environment variable."""

    def test_budget_appears_in_pipe_help(self):
        result = _run_cli("pipe", "--help", env_extra={"RAGIX_MEMORY_BUDGET": "3000"})
        assert result.returncode == 0
        assert "3000" in result.stdout

    def test_budget_appears_in_push_help(self):
        result = _run_cli("push", "--help", env_extra={"RAGIX_MEMORY_BUDGET": "3000"})
        assert result.returncode == 0
        assert "3000" in result.stdout
