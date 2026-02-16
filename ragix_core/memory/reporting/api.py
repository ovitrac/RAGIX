"""
Public API — generate_report() entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_core.memory.reporting.engine import ReportEngine
from ragix_core.memory.reporting.io import apply_overrides, load_config, resolve_config_path
from ragix_core.memory.reporting.scenarios import get_scenario, list_scenario_ids


def generate_report(
    *,
    db_path: str,
    workspace: str,
    scenario: str = "summarize_content",
    config_path: Optional[str] = None,
    out_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    embedder: str = "mock",
    scope: str = "audit",
    corpus_id: Optional[str] = None,
) -> str:
    """
    Run a reporting scenario against a RAGIX Memory DB.

    Args:
        db_path: Path to SQLite memory database.
        workspace: Named workspace (must exist or will be registered).
        scenario: Scenario name (builtin) or dotted module path.
        config_path: YAML config file. If None, uses builtin default.
        out_path: Write report here. If None, return string only.
        overrides: Dict of dotted-path overrides.
        embedder: Embedding backend ("mock" for now).
        scope: Default scope for workspace registration.
        corpus_id: Corpus ID for workspace registration.

    Returns:
        Generated Markdown report as string.
    """
    # Resolve scenario
    sc = get_scenario(scenario)

    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config(sc.default_config_name)

    # Apply overrides
    config = apply_overrides(config, overrides)

    # Build engine
    engine = ReportEngine(
        db_path=db_path,
        workspace=workspace,
        embedder=embedder,
        scope=scope,
        corpus_id=corpus_id,
    )

    # Run scenario
    md = sc.run(engine, config)

    # Write output
    if out_path:
        Path(out_path).write_text(md, encoding="utf-8")

    return md


def list_scenarios() -> List[str]:
    """Return sorted list of available scenario names."""
    return list_scenario_ids()
