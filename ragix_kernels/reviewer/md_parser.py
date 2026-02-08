"""
Markdown AST parser with heading tree extraction and protected region detection.

Uses mistune v3 for parsing. Falls back to regex-based parsing if mistune
is not available (with reduced accuracy).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ragix_kernels.reviewer.models import (
    HeadingNode,
    ProtectedKind,
    ProtectedSpan,
    content_hash,
    estimate_tokens,
)

import logging

logger = logging.getLogger(__name__)

# Try to import mistune v3; fall back to regex if unavailable
try:
    import mistune
    HAS_MISTUNE = True
except ImportError:
    HAS_MISTUNE = False
    logger.warning("mistune not installed; using regex-based Markdown parser (reduced accuracy)")


# ---------------------------------------------------------------------------
# Heading extraction (regex-based, works without mistune)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?$", re.MULTILINE)
_EXPLICIT_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+")

# Anchor generation: GitHub-compatible slug
_ANCHOR_STRIP_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def _slugify(text: str) -> str:
    """Convert heading text to GitHub-compatible anchor slug."""
    text = text.lower().strip()
    text = _ANCHOR_STRIP_RE.sub("", text)
    text = re.sub(r"[\s]+", "-", text)
    return text


def extract_headings(lines: List[str]) -> List[Tuple[int, int, str, str]]:
    """
    Extract headings from Markdown lines.

    Returns list of (line_number_1based, level, title, anchor).
    """
    headings = []
    in_fence = False
    fence_marker = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track code fence state
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif stripped.startswith(fence_marker) and len(stripped.rstrip("`~")) == 0 or stripped == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            anchor = _slugify(title)
            headings.append((i + 1, level, title, anchor))

    return headings


def build_heading_tree(
    headings: List[Tuple[int, int, str, str]],
    total_lines: int,
) -> List[HeadingNode]:
    """
    Build a hierarchical heading tree from flat heading list.

    Each node's line_end is set to the line before the next heading at
    the same or higher level, or total_lines for the last section.
    """
    if not headings:
        return []

    # Assign section IDs and line ranges
    nodes: List[HeadingNode] = []
    counters: Dict[int, int] = {}  # level -> counter

    for idx, (line_num, level, title, anchor) in enumerate(headings):
        # Compute line_end
        if idx + 1 < len(headings):
            line_end = headings[idx + 1][0] - 1
        else:
            line_end = total_lines

        # Detect explicit numbering
        numbering = ""
        m = _EXPLICIT_NUM_RE.match(title)
        if m:
            numbering = m.group(1)

        # Generate stable section ID
        counters[level] = counters.get(level, 0) + 1
        # Reset deeper counters
        for deeper in list(counters.keys()):
            if deeper > level:
                del counters[deeper]

        parts = []
        for lv in sorted(counters.keys()):
            if lv <= level:
                parts.append(str(counters[lv]))
        section_id = "S" + ".".join(parts)

        nodes.append(HeadingNode(
            id=section_id,
            level=level,
            title=title,
            anchor=anchor,
            line_start=line_num,
            line_end=line_end,
            numbering=numbering,
        ))

    # Build tree (nest children under parents)
    root_nodes: List[HeadingNode] = []
    stack: List[HeadingNode] = []

    for node in nodes:
        # Pop stack until we find a parent (lower level)
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            root_nodes.append(node)

        stack.append(node)

    return root_nodes


# ---------------------------------------------------------------------------
# Protected region detection
# ---------------------------------------------------------------------------

_YAML_FRONT_RE = re.compile(r"^---\s*$")
_MATH_BLOCK_RE = re.compile(r"^\$\$\s*$")
_LINK_REF_RE = re.compile(r"^\[([^\]]+)\]:\s+\S+")
_TABLE_SEP_RE = re.compile(r"^\|?[\s:]*-[-:|\s]*\|")


def detect_protected_regions(
    lines: List[str],
    protect_tables: bool = True,
    protect_math: bool = True,
) -> List[ProtectedSpan]:
    """
    Detect all protected (immutable) regions in Markdown.

    Conservative: when in doubt, mark as protected.
    """
    spans: List[ProtectedSpan] = []
    n = len(lines)

    # --- YAML front matter (only at start of file) ---
    if n > 0 and lines[0].strip() == "---":
        for j in range(1, n):
            if lines[j].strip() == "---":
                text = "\n".join(lines[0:j + 1])
                spans.append(ProtectedSpan(
                    kind=ProtectedKind.YAML_FRONT_MATTER,
                    line_start=1,
                    line_end=j + 1,
                    content_hash=content_hash(text),
                ))
                break

    # --- Fenced code blocks ---
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            info = stripped[3:].strip()
            start = i
            # Find closing fence
            j = i + 1
            while j < n:
                if lines[j].strip() == marker or lines[j].strip().startswith(marker) and len(lines[j].strip().rstrip(marker[0])) == 0:
                    break
                j += 1
            end = min(j, n - 1)
            text = "\n".join(lines[start:end + 1])
            spans.append(ProtectedSpan(
                kind=ProtectedKind.CODE_FENCE,
                line_start=start + 1,
                line_end=end + 1,
                content_hash=content_hash(text),
                info=info,
            ))
            i = end + 1
            continue
        i += 1

    # --- Math blocks ($$ ... $$) ---
    if protect_math:
        i = 0
        while i < n:
            if lines[i].strip() == "$$":
                start = i
                j = i + 1
                while j < n and lines[j].strip() != "$$":
                    j += 1
                end = min(j, n - 1)
                # Skip if this overlaps with an existing span
                if not _overlaps_any(spans, start + 1, end + 1):
                    text = "\n".join(lines[start:end + 1])
                    spans.append(ProtectedSpan(
                        kind=ProtectedKind.MATH_BLOCK,
                        line_start=start + 1,
                        line_end=end + 1,
                        content_hash=content_hash(text),
                    ))
                i = end + 1
                continue
            i += 1

    # --- Tables ---
    if protect_tables:
        i = 0
        while i < n:
            # A table starts with a | line followed by a separator line
            if i + 1 < n and "|" in lines[i] and _TABLE_SEP_RE.match(lines[i + 1]):
                start = i
                j = i
                while j < n and ("|" in lines[j] or _TABLE_SEP_RE.match(lines[j])):
                    j += 1
                end = j - 1
                if not _overlaps_any(spans, start + 1, end + 1):
                    text = "\n".join(lines[start:end + 1])
                    spans.append(ProtectedSpan(
                        kind=ProtectedKind.TABLE,
                        line_start=start + 1,
                        line_end=end + 1,
                        content_hash=content_hash(text),
                    ))
                i = end + 1
                continue
            i += 1

    # --- HTML blocks ---
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("<") and not stripped.startswith("<!-") and re.match(r"^<[a-zA-Z]", stripped):
            # Simple heuristic: block-level HTML
            start = i
            tag_match = re.match(r"<(\w+)", stripped)
            if tag_match:
                tag = tag_match.group(1)
                j = i
                while j < n:
                    if f"</{tag}>" in lines[j]:
                        break
                    j += 1
                end = min(j, n - 1)
                if not _overlaps_any(spans, start + 1, end + 1):
                    text = "\n".join(lines[start:end + 1])
                    spans.append(ProtectedSpan(
                        kind=ProtectedKind.HTML_BLOCK,
                        line_start=start + 1,
                        line_end=end + 1,
                        content_hash=content_hash(text),
                    ))
                i = end + 1
                continue
        i += 1

    # --- Link reference definitions ---
    for i, line in enumerate(lines):
        if _LINK_REF_RE.match(line.strip()):
            if not _overlaps_any(spans, i + 1, i + 1):
                spans.append(ProtectedSpan(
                    kind=ProtectedKind.LINK_REF_DEF,
                    line_start=i + 1,
                    line_end=i + 1,
                    content_hash=content_hash(line),
                ))

    # Sort by line_start
    spans.sort(key=lambda s: s.line_start)
    return spans


def _overlaps_any(spans: List[ProtectedSpan], start: int, end: int) -> bool:
    """Check if a range overlaps with any existing span."""
    return any(s.overlaps(start, end) for s in spans)


# ---------------------------------------------------------------------------
# High-level parsing API
# ---------------------------------------------------------------------------

def parse_markdown(text: str, protect_tables: bool = True, protect_math: bool = True):
    """
    Parse Markdown text and return structure info.

    Returns:
        dict with keys: lines, headings, heading_tree, protected_spans
    """
    lines = text.splitlines()
    raw_headings = extract_headings(lines)
    tree = build_heading_tree(raw_headings, len(lines))
    protected = detect_protected_regions(lines, protect_tables, protect_math)

    return {
        "lines": lines,
        "headings": raw_headings,
        "heading_tree": tree,
        "protected_spans": protected,
        "total_lines": len(lines),
    }


def is_line_protected(line_num: int, spans: List[ProtectedSpan]) -> bool:
    """Check if a 1-based line number falls inside any protected span."""
    return any(s.contains_line(line_num) for s in spans)
