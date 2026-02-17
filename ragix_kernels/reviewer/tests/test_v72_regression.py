"""
Regression test for v7.2 production baseline.

Validates that a completed integration test run meets the frozen invariants
from the MSG-HUB 13-chunk benchmark.  Does NOT run the LLM — it reads the
status.jsonl and fingerprint artifacts produced by a prior run and checks
expected properties.

Usage:
    # After running the integration test:
    python -m pytest ragix_kernels/reviewer/tests/test_v72_regression.py \
        --status-jsonl /path/to/workspace/stage2/ops/status.jsonl \
        --fingerprint-json /path/to/workspace/stage2/md_fingerprint_chunk.json

    # Or with defaults (MSG-HUB workspace):
    python -m pytest ragix_kernels/reviewer/tests/test_v72_regression.py

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-07
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "msg_hub_v72_regression.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spec() -> Dict[str, Any]:
    """Load the frozen regression spec."""
    return json.loads(_FIXTURE.read_text())


@pytest.fixture(scope="session")
def status_entries(request) -> List[Dict[str, Any]]:
    """Load status.jsonl entries from the test run."""
    path = Path(request.config.getoption("--status-jsonl"))
    if not path.exists():
        pytest.skip(f"status.jsonl not found: {path}")
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


@pytest.fixture(scope="session")
def fingerprints(request) -> Dict[str, Dict[str, Any]]:
    """Load fingerprint data keyed by chunk_id."""
    path = Path(request.config.getoption("--fingerprint-json"))
    if not path.exists():
        pytest.skip(f"fingerprint JSON not found: {path}")
    data = json.loads(path.read_text())["data"]
    return {fp["chunk_id"]: fp for fp in data.get("fingerprints", [])}


@pytest.fixture(scope="session")
def recipe_artifacts(request) -> Dict[str, Dict[str, Any]]:
    """Load _recipe.json artifacts keyed by chunk_id."""
    masks_dir = Path(request.config.getoption("--masks-dir"))
    recipes: Dict[str, Dict[str, Any]] = {}
    if masks_dir.exists():
        for p in masks_dir.glob("*_recipe.json"):
            data = json.loads(p.read_text())
            recipes[data["chunk_id"]] = data
    return recipes


@pytest.fixture(scope="session")
def status_by_id(status_entries) -> Dict[str, Dict[str, Any]]:
    """Status entries keyed by chunk_id."""
    return {e["chunk_id"]: e for e in status_entries}


# ---------------------------------------------------------------------------
# Aggregate invariants
# ---------------------------------------------------------------------------

class TestAggregateInvariants:
    """v7.2 production baseline: aggregate constraints."""

    def test_total_chunks(self, spec, status_entries):
        expected = spec["aggregate_invariants"]["total_chunks"]
        assert len(status_entries) == expected, (
            f"Expected {expected} chunks, got {len(status_entries)}"
        )

    def test_parse_success_all(self, spec, status_entries):
        if not spec["aggregate_invariants"]["parse_success_all"]:
            pytest.skip("parse_success_all not required")
        for e in status_entries:
            assert e.get("parse") == "OK", (
                f"Chunk {e['chunk_id']} failed parse: {e.get('extraction_method')}"
            )

    def test_degenerate_max(self, spec, status_entries):
        max_degen = spec["aggregate_invariants"]["degenerate_max"]
        actual_degen = sum(
            1 for e in status_entries
            if "decoded_empty" in e.get("extraction_method", "")
        )
        assert actual_degen <= max_degen, (
            f"Expected at most {max_degen} degenerate chunks, got {actual_degen}"
        )

    def test_json_strict_min(self, spec, status_entries):
        floor = spec["aggregate_invariants"]["json_strict_min"]
        actual = sum(
            1 for e in status_entries
            if e.get("extraction_method") == "json_strict"
        )
        assert actual >= floor, (
            f"Expected at least {floor} json_strict extractions, got {actual}"
        )

    def test_ops_min(self, spec, status_entries):
        floor = spec["aggregate_invariants"]["ops_min"]
        actual = sum(e.get("ops", 0) for e in status_entries)
        assert actual >= floor, (
            f"Expected at least {floor} total ops, got {actual}"
        )


# ---------------------------------------------------------------------------
# Per-chunk invariants
# ---------------------------------------------------------------------------

class TestPerChunkInvariants:
    """v7.2 production baseline: per-chunk property checks."""

    def test_per_chunk_parse_success(self, spec, status_by_id):
        for chunk_spec in spec["per_chunk_invariants"]:
            cid = chunk_spec["chunk_id"]
            if cid not in status_by_id:
                continue  # chunk may have different ID suffix across runs
            entry = status_by_id[cid]
            if chunk_spec.get("parse_success"):
                assert entry.get("parse") == "OK", (
                    f"{cid}: expected parse=OK, got {entry.get('parse')}"
                )

    def test_per_chunk_no_degenerate(self, spec, status_by_id):
        for chunk_spec in spec["per_chunk_invariants"]:
            cid = chunk_spec["chunk_id"]
            if cid not in status_by_id:
                continue
            entry = status_by_id[cid]
            if not chunk_spec.get("degenerate", True):
                method = entry.get("extraction_method", "")
                assert "decoded_empty" not in method, (
                    f"{cid}: expected non-degenerate, got {method}"
                )


# ---------------------------------------------------------------------------
# Recipe trigger invariants
# ---------------------------------------------------------------------------

class TestRecipeTriggers:
    """v7.2 production baseline: recipe triggers fire on known chunks."""

    def test_recipe_triggers_fire(self, spec, status_by_id):
        for cid, trigger_spec in spec["recipe_triggers"].items():
            if cid not in status_by_id:
                continue
            entry = status_by_id[cid]
            if trigger_spec.get("must_trigger_recipe"):
                recipes = entry.get("content_recipe", [])
                assert len(recipes) > 0, (
                    f"{cid}: expected content recipe to fire, got none"
                )

    def test_recipe_includes_expected_transforms(self, spec, status_by_id):
        for cid, trigger_spec in spec["recipe_triggers"].items():
            if cid not in status_by_id:
                continue
            entry = status_by_id[cid]
            actual = set(entry.get("content_recipe", []))
            expected = set(trigger_spec.get("expected_recipes_superset", []))
            missing = expected - actual
            assert not missing, (
                f"{cid}: expected recipes {expected}, got {actual}, missing {missing}"
            )

    def test_fingerprint_triggers(self, spec, fingerprints):
        for cid, trigger_spec in spec["recipe_triggers"].items():
            if cid not in fingerprints:
                continue
            fp = fingerprints[cid]
            assert fp.get("triggers_recipe", False), (
                f"{cid}: fingerprint should trigger recipe but doesn't"
            )
            # Check specific thresholds
            if "fingerprint_blockquote_lines_min" in trigger_spec:
                assert fp["blockquote_lines"] >= trigger_spec["fingerprint_blockquote_lines_min"], (
                    f"{cid}: blockquote_lines={fp['blockquote_lines']} < "
                    f"{trigger_spec['fingerprint_blockquote_lines_min']}"
                )
            if "fingerprint_emoji_count_min" in trigger_spec:
                assert fp["emoji_count"] >= trigger_spec["fingerprint_emoji_count_min"]
            if "fingerprint_table_rows_min" in trigger_spec:
                assert fp["table_rows"] >= trigger_spec["fingerprint_table_rows_min"]
            if "fingerprint_safety_keyword_hits_min" in trigger_spec:
                assert fp["safety_keyword_hits"] >= trigger_spec["fingerprint_safety_keyword_hits_min"]


# ---------------------------------------------------------------------------
# Recipe effectiveness invariants
# ---------------------------------------------------------------------------

class TestRecipeEffectiveness:
    """v7.2 production baseline: recipe artifact quality."""

    def test_recipe_artifacts_have_metrics(self, spec, recipe_artifacts):
        for cid in spec["recipe_triggers"]:
            if cid not in recipe_artifacts:
                continue
            art = recipe_artifacts[cid]
            assert "tokens_before" in art, f"{cid}: missing tokens_before"
            assert "tokens_after" in art, f"{cid}: missing tokens_after"
            assert "reduction_pct" in art, f"{cid}: missing reduction_pct"
            assert "recovered" in art, f"{cid}: missing recovered"
            assert "ops_count" in art, f"{cid}: missing ops_count"
            assert "extraction_method" in art, f"{cid}: missing extraction_method"

    def test_recipe_token_reduction_positive(self, spec, recipe_artifacts):
        for cid in spec["recipe_triggers"]:
            if cid not in recipe_artifacts:
                continue
            art = recipe_artifacts[cid]
            assert art["tokens_after"] < art["tokens_before"], (
                f"{cid}: recipe did not reduce tokens "
                f"({art['tokens_before']} → {art['tokens_after']})"
            )

    def test_recipe_recovered(self, spec, recipe_artifacts):
        for cid in spec["recipe_triggers"]:
            if cid not in recipe_artifacts:
                continue
            art = recipe_artifacts[cid]
            assert art.get("recovered", False), (
                f"{cid}: recipe applied but chunk not recovered"
            )
