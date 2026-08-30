"""
saqqara.store.config — one shape, declared once, refused when it does not match.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.14.

The packaged `defaults.yaml` is the single source of truth for what a
configuration IS. A user file is a partial overlay naming only what it changes,
and every key it names must exist in the defaults.

**An unknown key is refused, with its path.** `embeder: {...}` is a typo, and the
alternative to refusing it is a run that silently uses the default embedder and
reports success — the user's instruction discarded without a word. The path is
reported because "unknown key" in a nested mapping is not actionable and
`embedder.privider` is.

**Secrets travel by reference.** `api_key_ref` holds `env:VAR` or
`file:/path#Label`, never a value. `resolve_secret` reads it at the moment of use
and fails closed. Nothing resolves during load, so a config object is always safe
to serialise, log or POST — which is the only way that stays true.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

__all__ = ["DEFAULTS_PATH", "StoreConfig", "load_config", "resolve_secret"]

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


def _load_yaml(text: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("a configuration is a mapping at the top level")
    return data


def _merge(defaults: dict[str, Any], overlay: dict[str, Any],
           path: str = "") -> dict[str, Any]:
    """Deep merge, refusing any key the defaults do not declare."""
    merged = dict(defaults)
    for key, value in overlay.items():
        where = f"{path}.{key}" if path else key
        if key not in defaults:
            known = ", ".join(sorted(defaults)) or "(nothing)"
            raise ValueError(f"unknown configuration key {where!r}; known here: {known}")
        if isinstance(defaults[key], dict) and isinstance(value, dict):
            merged[key] = _merge(defaults[key], value, where)
        elif isinstance(defaults[key], dict) != isinstance(value, dict):
            raise ValueError(
                f"configuration key {where!r} is a "
                f"{'mapping' if isinstance(defaults[key], dict) else 'value'} "
                f"in the defaults but a "
                f"{'mapping' if isinstance(value, dict) else 'value'} here"
            )
        else:
            merged[key] = value
    return merged


class StoreConfig:
    """A validated configuration. Its `data` is always safe to write down."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data.get(name, {}))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for step in dotted.split("."):
            if not isinstance(node, dict) or step not in node:
                return default
            node = node[step]
        return node

    def to_dict(self) -> dict[str, Any]:
        """The configuration as data. No secret has been resolved into it."""
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.data.items()}

    def api_key(self) -> Optional[str]:
        """Resolve the embedder's secret, at the moment of use and not before."""
        ref = self.get("embedder.api_key_ref", "")
        return resolve_secret(ref) if ref else None


def resolve_secret(reference: str) -> str:
    """Read a secret named by reference. Fails closed, and never returns the reference.

    Two forms, both local: `env:NAME` and `file:/path#Label`, where the optional
    `#Label` selects a `Label <value>` line. A reference that cannot be resolved
    raises: returning the reference itself would send the string "env:TOKEN" as a
    credential, which fails somewhere far away and unrecognisably.
    """
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError(
            f"a secret is named by reference, not by value: expected env:NAME or "
            f"file:/path#Label, got {reference[:12]!r}..."
        )
    scheme, _, rest = reference.partition(":")

    if scheme == "env":
        value = os.environ.get(rest)
        if not value:
            raise ValueError(f"environment variable {rest!r} is unset or empty")
        return value

    if scheme == "file":
        path_part, _, label = rest.partition("#")
        path = Path(path_part).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read secret file {path_part!r}: {exc}") from exc
        if not label:
            value = text.strip()
            if not value:
                raise ValueError(f"secret file {path_part!r} is empty")
            return value
        for line in text.splitlines():
            if line.startswith(label):
                value = line[len(label):].strip()
                if value:
                    return value
        raise ValueError(f"label {label!r} not found in {path_part!r}")

    raise ValueError(f"unknown secret scheme {scheme!r}; use env: or file:")


def load_config(user: Optional[dict[str, Any] | str | Path] = None,
                **overrides: Any) -> StoreConfig:
    """Packaged defaults, then the user file, then explicit overrides.

    Each layer is validated against the defaults, so an override typed on a command
    line is refused by the same rule as a key in a file.
    """
    data = _load_yaml(DEFAULTS_PATH.read_text(encoding="utf-8"))

    if user is not None:
        if isinstance(user, (str, Path)):
            path = Path(user)
            if not path.is_file():
                raise ValueError(f"no configuration file at {path}")
            user = _load_yaml(path.read_text(encoding="utf-8"))
        data = _merge(data, user)

    if overrides:
        nested: dict[str, Any] = {}
        for dotted, value in overrides.items():
            node = nested
            steps = dotted.split(".")
            for step in steps[:-1]:
                node = node.setdefault(step, {})
            node[steps[-1]] = value
        data = _merge(data, nested)

    provider = data["embedder"]["provider"]
    from .embed import PROVIDERS
    if provider not in PROVIDERS:
        raise ValueError(f"embedder.provider must be one of {PROVIDERS}, not {provider!r}")
    backend = data["index"]["backend"]
    if backend not in ("numpy", "faiss"):
        raise ValueError(f"index.backend must be 'numpy' or 'faiss', not {backend!r}")

    return StoreConfig(data)
