"""
Glossary loader / prompt-formatter for KOAS-Translate.

The glossary is per-project data (``EN,FR,rule`` CSV) injected into the draft and
harmonize prompts as a deterministic bullet list. Ported from the translation
pipeline's ``glossary.py`` and decoupled from its global config — :func:`load`
takes an explicit path.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Entry:
    en: str
    fr: str
    rule: str


def load(path: Path) -> List[Entry]:
    """Load glossary entries from a ``EN,FR,rule`` CSV at *path*."""
    path = Path(path)
    entries: List[Entry] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"EN", "FR", "rule"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"glossary CSV missing columns: {sorted(missing)}")
        for i, row in enumerate(reader, start=2):  # header is line 1
            en = (row["EN"] or "").strip()
            fr = (row["FR"] or "").strip()
            rule = (row["rule"] or "").strip()
            if not en or not fr:
                raise ValueError(f"glossary CSV line {i}: EN and FR are required")
            entries.append(Entry(en=en, fr=fr, rule=rule))
    return entries


def format_for_prompt(entries: List[Entry]) -> str:
    """Render entries as a stable bullet list for prompt injection."""
    lines = []
    for e in entries:
        lines.append(f"- {e.en} → {e.fr}  ({e.rule})" if e.rule else f"- {e.en} → {e.fr}")
    return "\n".join(lines)
