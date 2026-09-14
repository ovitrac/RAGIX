"""tender.vocabulary — the CLOSED coarse-axis vocabulary of the requirement lane.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The coarse scale of ``DESIGN_REQUIREMENT_LANE_20260821`` §4 is a closed
vocabulary: an extractor may only CHOOSE a theme, never split one. The
granularity IS the vocabulary — which is why the vocabulary is mined from the
corpus (``scripts/axis_vocabulary.py``, gate GR2'' material) and then frozen by
the scientific lead, never written by hand.

Status of the shipped file (``data/axes_coarse_v0-draft.json``):
**v0-draft, RED** — the freeze is the lead's signature (gate GR3-0a,
``NOTE_AXIS_VOCABULARY_COARSE_20260821``). This module reads whatever version
is on disk and exposes its ``version`` so every record and trace records which
vocabulary produced it (rule 12: no configuration change escapes the gates).

``unknown`` is a valid coordinate everywhere: abstention is a success.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

#: packaged default; overridable for tests and for a re-mined vocabulary
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "axes_coarse_v0-draft.json"

UNKNOWN = "unknown"

#: French articles/prepositions dropped by the alias fold. Must stay identical
#: to ``scripts/axis_vocabulary.STOP`` — the mining and the lookup share one
#: normalization or aliases silently stop matching.
_STOP = frozenset("""le la les l un une des du de d au aux a à et en pour par
sur dans avec ou ses son sa leur leurs ce cet cette nos notre""".split())


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def fold(name: str) -> str:
    """Alias key: accents, apostrophes, articles, plurals (mining-identical)."""
    s = _strip_accents(name.lower()).replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9']+", " ", s)
    out = []
    for tok in s.split():
        tok = tok.split("'")[-1] if "'" in tok else tok
        if not tok or tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return " ".join(out)


@lru_cache(maxsize=4)
def load_axes(path: Optional[str] = None) -> dict[str, Any]:
    """Load the vocabulary file. Cached; pass a path to load another version."""
    p = Path(path) if path else DEFAULT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("vocabulary") != "coarse-axes":
        raise ValueError(f"{p}: not a coarse-axis vocabulary")
    axes = data["axes"]
    by_id = {a["axis_id"]: a for a in axes}
    if len(by_id) != len(axes):
        raise ValueError(f"{p}: duplicate axis_id")
    lookup: dict[str, str] = {}
    for a in axes:                      # canonical + aliases, folded
        for name in [a["canonical"], *a.get("aliases", [])]:
            lookup.setdefault(fold(name), a["axis_id"])
    return {"version": data["version"], "path": str(p), "axes": axes,
            "by_id": by_id, "lookup": lookup,
            "status": data.get("status", "")}


def version(path: Optional[str] = None) -> str:
    return load_axes(path)["version"]


def axis_ids(path: Optional[str] = None) -> frozenset[str]:
    """Valid coarse themes, WITHOUT ``unknown`` (which is valid everywhere)."""
    return frozenset(load_axes(path)["by_id"])


def is_valid_theme(theme: str, path: Optional[str] = None) -> bool:
    return theme == UNKNOWN or theme in axis_ids(path)


def resolve(name: str, path: Optional[str] = None) -> Optional[str]:
    """Map a surface name (canonical, alias, or a spelling variant of either)
    to its ``axis_id``. Returns None when the name is outside the vocabulary —
    the caller abstains (``unknown``); it never invents an axis."""
    return load_axes(path)["lookup"].get(fold(name))


def canonical(axis_id: str, path: Optional[str] = None) -> str:
    return load_axes(path)["by_id"][axis_id]["canonical"]


def prompt_names(path: Optional[str] = None, *,
                 requirement_only: bool = False) -> list[str]:
    """Canonical names for a closed LLM enum, in vocabulary order.

    ``requirement_only`` drops the ``furniture`` axes (navigational headings).
    They are kept by default: they act as sinks that protect the requirement
    axes from contamination (NOTE_AXIS_VOCABULARY_COARSE §3.2).
    """
    return [a["canonical"] for a in load_axes(path)["axes"]
            if not (requirement_only and a.get("class") == "furniture")]
