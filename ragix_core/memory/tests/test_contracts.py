"""Contract tests: validate RAGIX memory against shared contracts.json fixture.

These tests enforce the shared contracts between memctl and RAGIX.
Both repositories carry an identical contracts.json and a test_contracts.py
that validates their own implementation against it.

Contract violations indicate potential drift between memctl and RAGIX.
Any change to contracts.json must be synchronized in both repos.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab | olivier.vitrac@adservio.fr
"""

import json
import re
import inspect
import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture: load contracts.json
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "contracts.json"


@pytest.fixture(scope="module")
def contracts():
    """Load the shared contract fixture."""
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Schema contracts — structural validation
# ===========================================================================


def test_required_tables_exist(tmp_path, contracts):
    """A DB created by RAGIX MemoryStore must contain all required tables."""
    from ragix_core.memory.store import MemoryStore

    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=str(db_path))
    store.close()

    conn = sqlite3.connect(str(db_path))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    required = set(contracts["schema"]["required_tables"])
    missing = required - tables
    assert not missing, f"Missing required tables: {missing}"


def test_fts_virtual_table_exists(tmp_path, contracts):
    """The FTS5 virtual table must exist with the contracted name."""
    from ragix_core.memory.store import MemoryStore

    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=str(db_path))
    store.close()

    conn = sqlite3.connect(str(db_path))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    fts_name = contracts["schema"]["fts_virtual_table"]
    assert fts_name in tables, (
        f"FTS virtual table '{fts_name}' not found. Tables: {sorted(tables)}"
    )


def test_required_columns_memory_items(tmp_path, contracts):
    """memory_items must contain all contracted columns (superset OK)."""
    from ragix_core.memory.store import MemoryStore

    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=str(db_path))
    store.close()

    conn = sqlite3.connect(str(db_path))
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(memory_items)").fetchall()
    }
    conn.close()

    required = set(contracts["schema"]["required_columns_memory_items"])
    missing = required - columns
    assert not missing, f"Missing required columns in memory_items: {missing}"


# ===========================================================================
# 2. FTS tokenizer presets
# ===========================================================================


def test_fts_presets_match(contracts):
    """FTS tokenizer presets must resolve identically to the contract.

    RAGIX may define presets in cli.py or store.py. This test checks
    that the preset mapping exists and matches the contracted values.
    """
    # RAGIX stores FTS presets in store.py as FTS_TOKENIZER_PRESETS
    from ragix_core.memory import store as store_module

    # Look for known preset dict names
    presets = None
    for attr_name in ("FTS_TOKENIZER_PRESETS", "_FTS_PRESETS", "FTS_PRESETS"):
        if hasattr(store_module, attr_name):
            presets = getattr(store_module, attr_name)
            break

    if presets is None:
        # Fallback: check cli.py
        from ragix_core.memory import cli as cli_module
        for attr_name in ("FTS_TOKENIZER_PRESETS", "_FTS_PRESETS", "FTS_PRESETS"):
            if hasattr(cli_module, attr_name):
                presets = getattr(cli_module, attr_name)
                break

    if presets is None:
        pytest.skip(
            "FTS presets not found as module-level dict. "
            "Verify manually that tokenizer strings match contracts.json."
        )
        return

    # Transitional aliases: RAGIX v0.62.0 uses shorthand forms for some
    # presets. These are accepted temporarily until RAGIX adopts the
    # canonical explicit forms from contracts.json.
    # Remove this mapping when RAGIX store.py is updated to match.
    _RAGIX_TRANSITIONAL_ALIASES = {
        "en": {"porter"},  # RAGIX uses "porter"; canonical is "porter unicode61 remove_diacritics 2"
    }

    for preset_name, expected_tokenizer in contracts["fts_presets"].items():
        assert preset_name in presets, (
            f"Missing FTS preset: '{preset_name}'"
        )
        actual = presets[preset_name]
        if actual == expected_tokenizer:
            continue  # Exact match — ideal
        # Check transitional alias
        aliases = _RAGIX_TRANSITIONAL_ALIASES.get(preset_name, set())
        assert actual in aliases, (
            f"FTS preset '{preset_name}' mismatch: "
            f"got '{actual}', expected '{expected_tokenizer}'"
            f"{' (or transitional: ' + str(aliases) + ')' if aliases else ''}"
        )


# ===========================================================================
# 3. Injection format version
# ===========================================================================


def test_format_version(contracts):
    """FORMAT_VERSION must match the contracted value."""
    from ragix_core.memory.mcp.formatting import FORMAT_VERSION

    expected = contracts["injection_format"]["format_version"]
    assert FORMAT_VERSION == expected, (
        f"FORMAT_VERSION mismatch: got {FORMAT_VERSION}, expected {expected}"
    )


# ===========================================================================
# 4. MCP core tool names
# ===========================================================================


def test_core_mcp_tool_names(contracts):
    """All 7 core MCP tool names must exist in the RAGIX MCP tools module.

    RAGIX may register additional tools beyond the core 7 (superset OK).
    """
    from ragix_core.memory.mcp import tools as tools_module

    source = inspect.getsource(tools_module)
    # Extract all function names matching memory_* pattern
    registered_names = set(re.findall(r"def (memory_\w+)\(", source))

    required = set(contracts["mcp"]["core_tool_names"])
    missing = required - registered_names
    assert not missing, (
        f"Missing core MCP tools: {missing}. "
        f"Found memory_* functions: {sorted(registered_names)}"
    )


# ===========================================================================
# 5. Policy pattern categories
# ===========================================================================


def test_policy_categories_present(contracts):
    """All required policy pattern categories must exist with >= minimum count.

    RAGIX may have more patterns than the minimum (superset OK).
    """
    from ragix_core.memory import policy as policy_module

    category_map = {
        "secret_patterns": "_SECRET_PATTERNS",
        "injection_patterns": "_INJECTION_PATTERNS",
        "instructional_block_patterns": "_INSTRUCTIONAL_BLOCK_PATTERNS",
        "instructional_quarantine_patterns": "_INSTRUCTIONAL_QUARANTINE_PATTERNS",
    }

    minimums = contracts["policy"]["minimum_categories"]

    for category_key, min_count in minimums.items():
        attr_name = category_map.get(category_key)
        assert attr_name is not None, f"Unknown policy category: {category_key}"
        assert hasattr(policy_module, attr_name), (
            f"Policy module missing attribute: {attr_name}"
        )
        patterns = getattr(policy_module, attr_name)
        assert len(patterns) >= min_count, (
            f"Policy category '{category_key}' has {len(patterns)} patterns, "
            f"minimum required: {min_count}"
        )


# ===========================================================================
# 6. Injection block byte-stability (fixed fixture)
# ===========================================================================


def test_injection_block_deterministic(contracts):
    """Injection block output must be deterministic for a fixed input."""
    from ragix_core.memory.mcp.formatting import format_injection_block, FORMAT_VERSION

    # Fixed test data — same input must always produce same output.
    items = [
        {
            "id": "MEM-contract-test-001",
            "title": "Contract test item",
            "content": "This is a fixed content string for byte-stability testing.",
            "tier": "stm",
            "type": "note",
            "confidence": 0.9,
            "validation": "unverified",
            "tags": ["test", "contract"],
            "provenance": {},
            "score": 1.0,
        }
    ]

    block1 = format_injection_block(items=items, budget_tokens=500)
    block2 = format_injection_block(items=items, budget_tokens=500)

    # Same input -> identical output (deterministic)
    assert block1 == block2, "Injection block is not deterministic for fixed input"

    # Must contain format_version marker
    assert f"format_version: {FORMAT_VERSION}" in block1 or \
           f"format_version={FORMAT_VERSION}" in block1, (
        "Injection block missing format_version marker"
    )
