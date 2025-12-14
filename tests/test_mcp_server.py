#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for RAGIX MCP Server v0.8.0

Tests the MCP tools for functionality:
- Core tools (chat, scan, read, search)
- KOAS tools (init, run, status, summary, list_kernels, report)
- AST tools (scan, metrics)
- Model management tools (list, info)
- System info tools

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "MCP"))

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_sandbox():
    """Create a temporary sandbox directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_sandbox):
    """Create a sample Python file for testing."""
    code = '''
"""Sample module for testing."""

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


def helper_function(x):
    """A helper function."""
    return x * 2


def _private_function():
    """A private function."""
    pass
'''
    file_path = temp_sandbox / "calculator.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_project(temp_sandbox):
    """Create a sample project structure."""
    # Create directories
    src = temp_sandbox / "src"
    src.mkdir()
    tests = temp_sandbox / "tests"
    tests.mkdir()

    # Main module
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text('''
"""Main module."""

class Application:
    def run(self):
        print("Running")
''')

    # Utils module
    (src / "utils.py").write_text('''
"""Utility functions."""

def format_string(s: str) -> str:
    return s.strip().lower()

class StringHelper:
    @staticmethod
    def capitalize(s: str) -> str:
        return s.capitalize()
''')

    # Test file
    (tests / "__init__.py").write_text("")
    (tests / "test_main.py").write_text('''
"""Tests for main module."""
import pytest

def test_application():
    pass
''')

    return temp_sandbox


# =============================================================================
# Test: ragix_ast_scan
# =============================================================================

class TestRagixAstScan:
    """Test the ragix_ast_scan tool."""

    def test_scan_single_file(self, sample_python_file):
        """Test scanning a single Python file."""
        # Import after fixtures are set up
        from ragix_mcp_server import ragix_ast_scan

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(sample_python_file.parent)):
            result = ragix_ast_scan(str(sample_python_file))

        assert "error" not in result
        assert "symbols" in result
        assert "summary" in result

        # Check we found the class
        symbols = result["symbols"]
        class_names = [s["name"] for s in symbols if s["type"] == "class"]
        assert "Calculator" in class_names

        # Check summary counts
        summary = result["summary"]
        assert summary["classes"] >= 1
        assert summary["functions"] >= 1

    def test_scan_excludes_private_by_default(self, sample_python_file):
        """Test that private symbols are excluded by default."""
        from ragix_mcp_server import ragix_ast_scan

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(sample_python_file.parent)):
            result = ragix_ast_scan(str(sample_python_file), include_private=False)

        symbols = result.get("symbols", [])
        private_names = [s["name"] for s in symbols if s["name"].startswith("_")]
        assert "_private_function" not in private_names

    def test_scan_includes_private_when_requested(self, sample_python_file):
        """Test that private symbols can be included."""
        from ragix_mcp_server import ragix_ast_scan

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(sample_python_file.parent)):
            result = ragix_ast_scan(str(sample_python_file), include_private=True)

        symbols = result.get("symbols", [])
        names = [s["name"] for s in symbols]
        assert "_private_function" in names

    def test_scan_directory(self, sample_project):
        """Test scanning a directory."""
        from ragix_mcp_server import ragix_ast_scan

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(sample_project)):
            result = ragix_ast_scan(str(sample_project / "src"))

        assert "error" not in result
        summary = result.get("summary", {})
        assert summary.get("files_scanned", 0) >= 2

    def test_scan_nonexistent_path(self, temp_sandbox):
        """Test scanning a non-existent path."""
        from ragix_mcp_server import ragix_ast_scan

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(temp_sandbox)):
            result = ragix_ast_scan("/nonexistent/path")

        assert "error" in result


# =============================================================================
# Test: ragix_ast_metrics
# =============================================================================

class TestRagixAstMetrics:
    """Test the ragix_ast_metrics tool."""

    def test_metrics_single_file(self, sample_python_file):
        """Test computing metrics for a single file."""
        from ragix_mcp_server import ragix_ast_metrics

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(sample_python_file.parent)):
            result = ragix_ast_metrics(str(sample_python_file))

        assert "error" not in result
        assert "metrics" in result

        metrics = result["metrics"]
        assert metrics["total_files"] >= 1
        assert metrics["total_loc"] > 0

    def test_metrics_directory(self, sample_project):
        """Test computing metrics for a directory."""
        from ragix_mcp_server import ragix_ast_metrics

        with patch('ragix_mcp_server.SANDBOX_ROOT', str(sample_project)):
            result = ragix_ast_metrics(str(sample_project))

        assert "error" not in result
        assert "hotspots" in result

        hotspots = result.get("hotspots", [])
        assert len(hotspots) > 0


# =============================================================================
# Test: ragix_models_list
# =============================================================================

class TestRagixModelsList:
    """Test the ragix_models_list tool."""

    def test_list_models_with_ollama(self):
        """Test listing models when Ollama is available."""
        from ragix_mcp_server import ragix_models_list

        mock_output = """NAME                    ID              SIZE    MODIFIED
mistral:latest          abc123          4.1 GB  2 days ago
qwen2.5:14b             def456          8.5 GB  1 day ago
"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
            result = ragix_models_list()

        assert "error" not in result
        assert "models" in result
        assert result["total"] >= 2

        model_names = [m["name"] for m in result["models"]]
        assert "mistral:latest" in model_names

    def test_list_models_ollama_not_running(self):
        """Test behavior when Ollama is not running."""
        from ragix_mcp_server import ragix_models_list

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = ragix_models_list()

        assert "error" in result


# =============================================================================
# Test: ragix_model_info
# =============================================================================

class TestRagixModelInfo:
    """Test the ragix_model_info tool."""

    def test_model_info_success(self):
        """Test getting model info successfully."""
        from ragix_mcp_server import ragix_model_info

        mock_output = """Model info for mistral:latest
Parameters: 7B
Quantization: Q4_K_M
Context: 4096
"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_output)
            result = ragix_model_info("mistral")

        assert "error" not in result
        assert result["name"] == "mistral"
        assert "capabilities" in result
        assert "text_generation" in result["capabilities"]

    def test_model_info_not_found(self):
        """Test getting info for non-existent model."""
        from ragix_mcp_server import ragix_model_info

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
            result = ragix_model_info("nonexistent_model")

        assert "error" in result


# =============================================================================
# Test: ragix_system_info
# =============================================================================

class TestRagixSystemInfo:
    """Test the ragix_system_info tool."""

    def test_system_info_returns_structure(self):
        """Test that system info returns expected structure."""
        from ragix_mcp_server import ragix_system_info

        with patch('subprocess.run') as mock_run:
            # Mock free command
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Mem: 16000000000 8000000000 8000000000 0 0 8000000000"
            )
            result = ragix_system_info()

        assert "platform" in result
        assert "ragix" in result
        assert "cpu" in result

    def test_system_info_contains_platform(self):
        """Test that platform info is included."""
        from ragix_mcp_server import ragix_system_info

        result = ragix_system_info()

        assert "platform" in result
        assert "system" in result["platform"]
        assert "python" in result["platform"]


# =============================================================================
# Test: KOAS koas_run with parallel execution
# =============================================================================

class TestKoasRunParallel:
    """Test the koas_run tool with parallel execution."""

    def test_koas_run_returns_execution_mode(self):
        """Test that koas_run returns execution mode info."""
        from ragix_mcp_server import koas_run

        # Mock the orchestrator
        with patch('ragix_mcp_server.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            result = koas_run("/nonexistent", stage=1)

        # Should fail but include duration
        assert "duration_seconds" in result

    def test_koas_run_parallel_parameter(self):
        """Test that parallel parameter is accepted."""
        from ragix_mcp_server import koas_run
        import inspect

        sig = inspect.signature(koas_run)
        params = sig.parameters

        assert "parallel" in params
        assert "workers" in params
        assert params["parallel"].default is False
        assert params["workers"].default == 4


# =============================================================================
# Integration Test: Tool Availability
# =============================================================================

class TestToolAvailability:
    """Test that all expected tools are available."""

    def test_core_tools_exist(self):
        """Test that core tools are importable."""
        from ragix_mcp_server import (
            ragix_chat,
            ragix_scan_repo,
            ragix_read_file,
            ragix_search,
            ragix_health,
        )

        assert callable(ragix_chat)
        assert callable(ragix_scan_repo)
        assert callable(ragix_read_file)
        assert callable(ragix_search)
        assert callable(ragix_health)

    def test_koas_tools_exist(self):
        """Test that KOAS tools are importable."""
        from ragix_mcp_server import (
            koas_init,
            koas_run,
            koas_status,
            koas_summary,
            koas_list_kernels,
            koas_report,
        )

        assert callable(koas_init)
        assert callable(koas_run)
        assert callable(koas_status)
        assert callable(koas_summary)
        assert callable(koas_list_kernels)
        assert callable(koas_report)

    def test_v080_tools_exist(self):
        """Test that v0.8.0 tools are importable."""
        from ragix_mcp_server import (
            ragix_ast_scan,
            ragix_ast_metrics,
            ragix_models_list,
            ragix_model_info,
            ragix_system_info,
        )

        assert callable(ragix_ast_scan)
        assert callable(ragix_ast_metrics)
        assert callable(ragix_models_list)
        assert callable(ragix_model_info)
        assert callable(ragix_system_info)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
