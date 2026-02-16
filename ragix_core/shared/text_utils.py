"""
Shared Text Utilities

Deterministic text processing functions used across RAGIX subsystems.
No LLM dependency — pure Python.

Functions:
    normalize_whitespace  - Replace Unicode whitespace (U+202F etc.) with ASCII
    estimate_tokens       - Rough token count (chars / 4)
    stable_hash           - SHA-256 content hash with 'sha256:' prefix
    split_paragraphs      - Split text at paragraph boundaries
    normalize_for_search  - Lowercase + whitespace-normalized for keyword matching

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import List


def normalize_whitespace(text: str) -> str:
    """
    Replace all Unicode whitespace characters with ASCII space and collapse.

    Handles French PDF typography (U+202F narrow no-break space, U+00A0
    non-breaking space, etc.) that causes keyword matching failures.
    """
    normalized = []
    for ch in text:
        if unicodedata.category(ch).startswith("Z") or ch in ("\t", "\r"):
            normalized.append(" ")
        elif ch == "\n":
            normalized.append("\n")
        else:
            normalized.append(ch)
    # Collapse multiple spaces (preserve newlines)
    result = re.sub(r"[^\S\n]+", " ", "".join(normalized))
    return result.strip()


def normalize_for_search(text: str) -> str:
    """
    Normalize text for keyword matching: lowercase + whitespace normalization.

    Collapses all whitespace (including newlines) to single ASCII space.
    """
    normalized = []
    for ch in text:
        if unicodedata.category(ch).startswith("Z") or ch in ("\t", "\n", "\r"):
            normalized.append(" ")
        else:
            normalized.append(ch)
    return " ".join("".join(normalized).lower().split())


def estimate_tokens(text: str) -> int:
    """
    Rough token count approximation.

    Uses chars/4 heuristic — good enough for budget estimation.
    """
    return max(1, len(text) // 4)


def stable_hash(text: str) -> str:
    """
    SHA-256 content hash with 'sha256:' prefix.

    Deterministic and reproducible for any input.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def split_paragraphs(text: str, min_length: int = 20) -> List[str]:
    """
    Split text at paragraph boundaries (double newline or blank line).

    Args:
        text: Input text
        min_length: Minimum paragraph length in chars; shorter are merged with next

    Returns:
        List of paragraph strings
    """
    # Split on blank lines
    raw = re.split(r"\n\s*\n", text)
    paragraphs = []
    buffer = ""

    for part in raw:
        part = part.strip()
        if not part:
            continue
        if buffer:
            buffer += "\n\n" + part
        else:
            buffer = part
        if len(buffer) >= min_length:
            paragraphs.append(buffer)
            buffer = ""

    if buffer:
        if paragraphs:
            paragraphs[-1] += "\n\n" + buffer
        else:
            paragraphs.append(buffer)

    return paragraphs
