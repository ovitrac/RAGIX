"""
Configuration for the family: packaged defaults, one overlay, refused when unknown.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

The same shape and the same refusals as the store's configuration, deliberately:
two families whose configuration behaved differently would be two things to learn.
The merge and the secret resolver are imported rather than copied — one rule, one
place, and the copy that drifts is whichever nobody is watching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ragix_kernels.saqqara.store.config import (  # the same rules, not a second copy
    StoreConfig,
    _load_yaml,
    _merge,
    resolve_secret,
)

__all__ = ["DEFAULTS_PATH", "TenderConfig", "load_config", "resolve_secret"]

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")

#: A `StoreConfig` by another name: the accessors are identical and the shape is
#: what differs. Subclassing rather than aliasing so the type says which family's
#: configuration it is when one appears in a traceback.
class TenderConfig(StoreConfig):
    """A validated configuration for this family."""


def load_config(user: Optional[dict[str, Any] | str | Path] = None,
                **overrides: Any) -> TenderConfig:
    """Packaged defaults, then the overlay, then explicit overrides.

    Every layer is validated against the defaults, so a key typed on a command
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

    return TenderConfig(data)
