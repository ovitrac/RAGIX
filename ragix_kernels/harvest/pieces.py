"""The piece map's parser: a piece name, then the document paths filed under it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The drivers (`core`, the registers) read the piece map through this one function, copied from the
analysis step that wrote the map, so that a map read by the harvest and a map read by the analysis
cannot disagree about which document a piece holds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_pieces(path: Path) -> dict[str, list[str]]:
    """piece -> every document path in it. The map is complete on its own (§2)."""
    pieces: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(RC|CCAP|CCTP|annexes):", stripped):
            current = stripped[:-1]
            pieces[current] = []
        elif stripped == "target_side:":
            current = "target_side"
            pieces[current] = []
        elif current and stripped.startswith('- "'):
            pieces[current].append(json.loads(stripped[2:]))
    return pieces
