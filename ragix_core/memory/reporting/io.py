"""
YAML config loader, builtin path resolution, and --set override merging.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Builtin configs live alongside this module
_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def resolve_config_path(name_or_path: str) -> Path:
    """Resolve a config name (builtin) or explicit path to a Path object."""
    p = Path(name_or_path)
    if p.exists():
        return p
    # Try builtin
    builtin = _CONFIG_DIR / name_or_path
    if builtin.exists():
        return builtin
    # Try with .yml extension
    builtin_yml = _CONFIG_DIR / f"{name_or_path}.yml"
    if builtin_yml.exists():
        return builtin_yml
    raise FileNotFoundError(
        f"Config not found: '{name_or_path}' (tried path, "
        f"builtin '{builtin}', '{builtin_yml}')"
    )


def load_config(name_or_path: str) -> dict:
    """Load a YAML config file by name (builtin) or path."""
    path = resolve_config_path(name_or_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(config: dict, overrides: Optional[Dict[str, Any]]) -> dict:
    """
    Apply dotted-path overrides to a config dict.

    Example:
        apply_overrides(config, {"recall.budgets": [500, 1500]})
        → sets config["recall"]["budgets"] = [500, 1500]
    """
    if not overrides:
        return config
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return config


def parse_set_args(set_args: list) -> dict:
    """
    Parse CLI --set arguments into an overrides dict.

    Handles:
        "recall.budgets=500,1500,4000" → {"recall.budgets": [500, 1500, 4000]}
        "search.k=8" → {"search.k": 8}
        "meta.strict=true" → {"meta.strict": True}
    """
    overrides = {}
    for arg in set_args:
        if "=" not in arg:
            continue
        key, raw_value = arg.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        # Try list of ints
        if "," in raw_value:
            try:
                overrides[key] = [int(x.strip()) for x in raw_value.split(",")]
                continue
            except ValueError:
                overrides[key] = [x.strip() for x in raw_value.split(",")]
                continue

        # Try int
        try:
            overrides[key] = int(raw_value)
            continue
        except ValueError:
            pass

        # Try float
        try:
            overrides[key] = float(raw_value)
            continue
        except ValueError:
            pass

        # Try bool
        if raw_value.lower() in ("true", "false"):
            overrides[key] = raw_value.lower() == "true"
            continue

        # String
        overrides[key] = raw_value

    return overrides
