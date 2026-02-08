"""
Kernel: md_fingerprint_chunk
Stage: 2 (Analysis)

Pure deterministic kernel — no LLM.  Computes a structural fingerprint
for each chunk, used by md_edit_plan to decide whether content-level
masking recipes should be applied before the LLM call.

Features extracted per chunk:
  - table_rows / table_count  (Markdown pipe-tables)
  - math_count                (display + inline LaTeX)
  - emoji_count               (Unicode emoji code points)
  - digit_density             (numeric content ratio)
  - bullet_count              (list items)
  - blockquote_lines          (lines starting with >)
  - max_line_length
  - safety_keyword_hits       (terms that trigger model self-censoring)
  - content_tokens            (rough token estimate)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-07
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import (
    ChunkFingerprint,
    ReviewChunk,
    estimate_tokens,
)

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction — all pure functions
# ---------------------------------------------------------------------------

# Reuse the table separator pattern from md_parser / md_protected_regions
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s\-:]+\|[\s\-:|]+\|?\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|")

# Math patterns (same as md_edit_plan)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# Emoji ranges: Emoticons, Dingbats, Symbols, Supplemental, Flags, etc.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U00002600-\U000026FF"  # Misc symbols
    "\U0000200D"             # Zero-width joiner
    "\U00002B05-\U00002B07"  # Arrows
    "\U00002934-\U00002935"
    "\U00003030\U000025AA\U000025AB\U000025B6\U000025C0"
    "\U000025FB-\U000025FE"
    "\U00002614-\U00002615"
    "\U00002648-\U00002653"
    "\U0000267F\U00002693\U000026A1\U000026AA\U000026AB"
    "\U000026BD\U000026BE\U000026C4\U000026C5\U000026CE"
    "\U000026D4\U000026EA\U000026F2\U000026F3\U000026F5"
    "\U000026FA\U000026FD"
    "]+"
)

_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]) ")
_BLOCKQUOTE_RE = re.compile(r"^\s*>")

# Safety keywords that trigger model self-censoring (case-insensitive)
_DEFAULT_SAFETY_KEYWORDS = [
    "vulnérabilité", "vulnérabilités", "vulnerability", "vulnerabilities",
    "attaque", "attaques", "attack", "attacks",
    "exploit", "exploiter", "exploits",
    "faille", "failles",
    "injection", "injections",
    "backdoor", "backdoors",
    "malware", "ransomware",
    "phishing",
    "brute force", "brute-force",
    "denial of service", "dos", "ddos",
    "zero-day", "0-day",
    "privilege escalation",
    "remote code execution", "rce",
]


def _count_table_rows(lines: List[str], start: int, end: int) -> Tuple[int, int]:
    """
    Count total table rows and distinct tables in a line range.

    Args:
        lines: All document lines (0-indexed)
        start: Start index (0-based, inclusive)
        end: End index (0-based, exclusive)

    Returns:
        (total_rows, table_count)
    """
    total_rows = 0
    table_count = 0
    in_table = False

    for i in range(start, min(end, len(lines))):
        line = lines[i]
        is_table_line = bool(_TABLE_ROW_RE.match(line)) or bool(_TABLE_SEP_RE.match(line))

        if is_table_line:
            if not in_table:
                in_table = True
                table_count += 1
            if not _TABLE_SEP_RE.match(line):
                total_rows += 1
        else:
            in_table = False

    return total_rows, table_count


def _count_emoji(text: str) -> int:
    """Count emoji code points in text."""
    return sum(len(m.group()) for m in _EMOJI_RE.finditer(text))


def _count_math(text: str) -> int:
    """Count display + inline math expressions."""
    display = len(_DISPLAY_MATH_RE.findall(text))
    inline = len(_INLINE_MATH_RE.findall(text))
    return display + inline


def _digit_density(text: str) -> float:
    """Compute digit density: digits / total non-whitespace characters."""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    digit_count = sum(1 for c in non_ws if c.isdigit())
    return digit_count / len(non_ws)


def _count_blockquote_lines(lines: List[str], start: int, end: int) -> int:
    """Count lines starting with > in a range."""
    count = 0
    for i in range(start, min(end, len(lines))):
        if _BLOCKQUOTE_RE.match(lines[i]):
            count += 1
    return count


def _count_bullets(lines: List[str], start: int, end: int) -> int:
    """Count bullet/numbered list items in a range."""
    count = 0
    for i in range(start, min(end, len(lines))):
        if _BULLET_RE.match(lines[i]):
            count += 1
    return count


def _count_safety_keywords(text: str, keywords: List[str]) -> int:
    """Count case-insensitive word-boundary matches for safety keywords."""
    count = 0
    text_lower = text.lower()
    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        count += len(pattern.findall(text_lower))
    return count


def _max_line_length(lines: List[str], start: int, end: int) -> int:
    """Maximum line length in a range."""
    max_len = 0
    for i in range(start, min(end, len(lines))):
        max_len = max(max_len, len(lines[i]))
    return max_len


_CODE_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _count_code_fences(lines: List[str], start: int, end: int) -> int:
    """Count fenced code blocks (``` or ~~~ pairs) in a line range."""
    count = 0
    in_fence = False
    fence_char = ""
    for i in range(start, min(end, len(lines))):
        m = _CODE_FENCE_RE.match(lines[i])
        if m:
            marker = m.group(1)[0]  # '`' or '~'
            if not in_fence:
                in_fence = True
                fence_char = marker
            elif marker == fence_char:
                in_fence = False
                count += 1
    return count


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class MdFingerprintChunkKernel(Kernel):
    """
    Structural fingerprinting of review chunks for content-recipe triggering.

    Pure deterministic Stage 2 kernel — no LLM.  Reads md_chunk output and
    the raw document, computes a ChunkFingerprint per chunk describing its
    structural complexity (tables, math, emoji, blockquotes, safety keywords).

    Used by md_edit_plan to decide whether content-level masking recipes
    (table masking, emoji masking, synonym map) should be applied before
    the LLM call.
    """

    name = "md_fingerprint_chunk"
    version = "1.0.0"
    category = "reviewer"
    stage = 2
    description = "Deterministic structural fingerprint per chunk for content-recipe routing"

    requires: List[str] = ["md_chunk"]
    provides: List[str] = ["chunk_fingerprints"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load document
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Load chunks
        chunk_path = input.dependencies.get("md_chunk")
        chunks: List[ReviewChunk] = []
        if chunk_path and chunk_path.exists():
            chunk_data = json.loads(chunk_path.read_text())["data"]
            chunks = [ReviewChunk.from_dict(d) for d in chunk_data["chunks"]]

        # Safety keywords from config
        reviewer_cfg = input.config.get("reviewer", {})
        recipe_cfg = reviewer_cfg.get("content_recipes", {})
        safety_keywords = recipe_cfg.get(
            "safety_keywords", _DEFAULT_SAFETY_KEYWORDS,
        )

        # Compute fingerprints
        fingerprints: List[Dict[str, Any]] = []
        triggers_count = 0

        for chunk in chunks:
            start_0 = chunk.line_start - 1  # 0-based inclusive
            end_0 = chunk.line_end          # 0-based exclusive for slicing

            chunk_text = "\n".join(lines[start_0:end_0])

            table_rows, table_count = _count_table_rows(lines, start_0, end_0)
            emoji_count = _count_emoji(chunk_text)
            math_count = _count_math(chunk_text)

            fp = ChunkFingerprint(
                chunk_id=chunk.chunk_id,
                table_rows=table_rows,
                table_count=table_count,
                math_count=math_count,
                emoji_count=emoji_count,
                digit_density=_digit_density(chunk_text),
                bullet_count=_count_bullets(lines, start_0, end_0),
                blockquote_lines=_count_blockquote_lines(lines, start_0, end_0),
                max_line_length=_max_line_length(lines, start_0, end_0),
                safety_keyword_hits=_count_safety_keywords(chunk_text, safety_keywords),
                content_tokens=estimate_tokens(chunk_text),
                code_fence_count=_count_code_fences(lines, start_0, end_0),
            )

            fingerprints.append(fp.to_dict())
            if fp.triggers_recipe():
                triggers_count += 1
                logger.info(
                    f"[md_fingerprint_chunk] {chunk.chunk_id}: triggers recipe "
                    f"(table={table_rows}/{table_count}, emoji={emoji_count}, "
                    f"math={math_count}, bq={fp.blockquote_lines}, "
                    f"fence={fp.code_fence_count}, safety={fp.safety_keyword_hits})"
                )

        return {
            "fingerprints": fingerprints,
            "total_chunks": len(chunks),
            "triggers_count": triggers_count,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        total = data.get("total_chunks", 0)
        triggers = data.get("triggers_count", 0)
        return (
            f"Fingerprinted {total} chunks: {triggers} trigger content recipes "
            f"({triggers}/{total})"
        )
