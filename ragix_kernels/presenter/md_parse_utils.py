"""
Shared Markdown parsing utilities for KOAS Presenter kernels.

Compiled regex constants and pure functions reused by pres_content_extract
and pres_asset_catalog. No external dependencies beyond stdlib + PyYAML.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Compiled regex constants
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$")
CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})\s*([\w+\-.*]*)")
MATH_BLOCK_RE = re.compile(r"^\$\$\s*$")
TABLE_SEP_RE = re.compile(r"^\|?[\s:]*-[-:|\s]*\|")
IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
YAML_DELIM_RE = re.compile(r"^---\s*$")
ADMONITION_RE = re.compile(r"^>\s*\[!(\w+)\]")
BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)")
BULLET_RE = re.compile(r"^(\s*)[*\-+]\s+(.+)")
NUMBERED_RE = re.compile(r"^(\s*)\d+[.)]\s+(.+)")
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_in_code_fence(lines: List[str], idx: int) -> bool:
    """Check if line at *idx* (0-based) is inside a fenced code block."""
    in_fence = False
    fence_marker = ""
    for i, line in enumerate(lines):
        if i == idx:
            return in_fence
        stripped = line.strip()
        m = CODE_FENCE_RE.match(stripped)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0]  # ` or ~
            elif stripped.rstrip(fence_marker) == "" or stripped == fence_marker * len(marker):
                # Closing fence: same char, at least as many
                if marker[0] == fence_marker and len(marker) >= 3:
                    in_fence = False
                    fence_marker = ""
    return in_fence


def detect_front_matter(lines: List[str]) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    """
    Detect YAML front matter at the start of a Markdown document.

    Returns (start_line_0based, end_line_0based_inclusive, parsed_dict) or None.
    Falls back to empty dict if YAML parsing fails.
    """
    if not lines or lines[0].strip() != "---":
        return None

    for j in range(1, len(lines)):
        if lines[j].strip() == "---":
            yaml_text = "\n".join(lines[1:j])
            parsed: Dict[str, Any] = {}
            try:
                import yaml
                parsed = yaml.safe_load(yaml_text) or {}
                if not isinstance(parsed, dict):
                    parsed = {"_raw": str(parsed)}
            except Exception:
                parsed = {}
            return (0, j, parsed)

    return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for European languages)."""
    return max(1, len(text) // 4)


def content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content (sha256:hex format)."""
    import hashlib
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
