"""
Pytest configuration for Interpreter-Tutor tests.

Registers custom markers and provides common fixtures.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "smoke: Quick tests that don't require Ollama"
    )
    config.addinivalue_line(
        "markers", "regression: Full regression tests requiring Ollama"
    )
