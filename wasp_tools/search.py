"""
WASP Search Tools - Text search and pattern matching

Deterministic search tools for text content.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import re
from typing import Any, Dict, List, Optional, Union


def search_pattern(
    pattern: str,
    content: str,
    flags: Optional[str] = None,
    max_matches: int = 100,
) -> Dict[str, Any]:
    """
    Search for regex pattern in content.

    Args:
        pattern: Regular expression pattern
        content: Text content to search
        flags: Optional regex flags (i=ignorecase, m=multiline, s=dotall)
        max_matches: Maximum number of matches to return

    Returns:
        dict with keys:
            - success: bool
            - matches: list of match objects with text, start, end, line, groups
            - count: total match count
            - error: error message (if invalid pattern)
    """
    result = {"success": True, "matches": []}

    # Parse flags
    re_flags = 0
    if flags:
        if 'i' in flags:
            re_flags |= re.IGNORECASE
        if 'm' in flags:
            re_flags |= re.MULTILINE
        if 's' in flags:
            re_flags |= re.DOTALL

    try:
        compiled = re.compile(pattern, re_flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "matches": [],
            "count": 0,
        }

    # Build line index for position-to-line mapping
    line_starts = _build_line_index(content)

    # Find matches
    for i, match in enumerate(compiled.finditer(content)):
        if i >= max_matches:
            result["truncated"] = True
            break

        line_num = _pos_to_line(match.start(), line_starts)
        line_col = match.start() - line_starts[line_num - 1] if line_num > 0 else match.start()

        match_info = {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "line": line_num,
            "column": line_col + 1,
        }

        # Include named groups if any
        if match.groupdict():
            match_info["named_groups"] = match.groupdict()
        elif match.groups():
            match_info["groups"] = list(match.groups())

        result["matches"].append(match_info)

    result["count"] = len(result["matches"])
    return result


def search_lines(
    pattern: str,
    content: str,
    flags: Optional[str] = None,
    context_before: int = 0,
    context_after: int = 0,
    max_matches: int = 100,
) -> Dict[str, Any]:
    """
    Search for pattern and return matching lines with context.

    Args:
        pattern: Regular expression pattern
        content: Text content to search
        flags: Optional regex flags
        context_before: Number of lines before match to include
        context_after: Number of lines after match to include
        max_matches: Maximum matches to return

    Returns:
        dict with keys:
            - success: bool
            - lines: list of matching lines with context
            - count: total matching line count
    """
    result = {"success": True, "lines": []}

    # Parse flags
    re_flags = 0
    if flags:
        if 'i' in flags:
            re_flags |= re.IGNORECASE
        if 'm' in flags:
            re_flags |= re.MULTILINE

    try:
        compiled = re.compile(pattern, re_flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "lines": [],
            "count": 0,
        }

    lines = content.splitlines()
    match_count = 0

    for i, line in enumerate(lines):
        if compiled.search(line):
            if match_count >= max_matches:
                result["truncated"] = True
                break

            match_info = {
                "line_number": i + 1,
                "text": line,
                "matches": [],
            }

            # Add match positions within line
            for m in compiled.finditer(line):
                match_info["matches"].append({
                    "text": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                })

            # Add context
            if context_before > 0:
                start = max(0, i - context_before)
                match_info["context_before"] = [
                    {"line_number": j + 1, "text": lines[j]}
                    for j in range(start, i)
                ]

            if context_after > 0:
                end = min(len(lines), i + context_after + 1)
                match_info["context_after"] = [
                    {"line_number": j + 1, "text": lines[j]}
                    for j in range(i + 1, end)
                ]

            result["lines"].append(match_info)
            match_count += 1

    result["count"] = len(result["lines"])
    return result


def count_matches(
    pattern: str,
    content: str,
    flags: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Count pattern matches in content.

    Args:
        pattern: Regular expression pattern
        content: Text content to search
        flags: Optional regex flags

    Returns:
        dict with keys:
            - success: bool
            - count: total match count
            - line_count: number of lines with matches
    """
    result = {"success": True}

    # Parse flags
    re_flags = 0
    if flags:
        if 'i' in flags:
            re_flags |= re.IGNORECASE
        if 'm' in flags:
            re_flags |= re.MULTILINE

    try:
        compiled = re.compile(pattern, re_flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "count": 0,
        }

    # Count total matches
    matches = compiled.findall(content)
    result["count"] = len(matches)

    # Count lines with matches
    lines_with_matches = sum(1 for line in content.splitlines() if compiled.search(line))
    result["line_count"] = lines_with_matches
    result["total_lines"] = len(content.splitlines())

    return result


def extract_matches(
    pattern: str,
    content: str,
    flags: Optional[str] = None,
    group: Optional[Union[int, str]] = None,
    unique: bool = False,
    max_matches: int = 1000,
) -> Dict[str, Any]:
    """
    Extract all pattern matches with optional group selection.

    Args:
        pattern: Regular expression pattern
        content: Text content to search
        flags: Optional regex flags
        group: Specific group to extract (index or name)
        unique: Return only unique matches
        max_matches: Maximum matches to return

    Returns:
        dict with keys:
            - success: bool
            - matches: list of extracted strings
            - count: total count
    """
    result = {"success": True, "matches": []}

    # Parse flags
    re_flags = 0
    if flags:
        if 'i' in flags:
            re_flags |= re.IGNORECASE
        if 'm' in flags:
            re_flags |= re.MULTILINE
        if 's' in flags:
            re_flags |= re.DOTALL

    try:
        compiled = re.compile(pattern, re_flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "matches": [],
            "count": 0,
        }

    seen = set()
    for i, match in enumerate(compiled.finditer(content)):
        if i >= max_matches:
            result["truncated"] = True
            break

        # Extract specified group or full match
        if group is not None:
            try:
                if isinstance(group, int):
                    extracted = match.group(group)
                else:
                    extracted = match.group(group)
            except (IndexError, KeyError):
                continue
        else:
            extracted = match.group(0)

        if extracted is None:
            continue

        if unique:
            if extracted in seen:
                continue
            seen.add(extracted)

        result["matches"].append(extracted)

    result["count"] = len(result["matches"])
    return result


def replace_pattern(
    pattern: str,
    replacement: str,
    content: str,
    flags: Optional[str] = None,
    count: int = 0,
) -> Dict[str, Any]:
    """
    Replace pattern matches in content.

    Args:
        pattern: Regular expression pattern
        replacement: Replacement string (supports backreferences)
        content: Text content to modify
        flags: Optional regex flags
        count: Maximum replacements (0 = unlimited)

    Returns:
        dict with keys:
            - success: bool
            - content: modified content
            - replacements: number of replacements made
            - error: error message (if invalid pattern)
    """
    result = {"success": True}

    # Parse flags
    re_flags = 0
    if flags:
        if 'i' in flags:
            re_flags |= re.IGNORECASE
        if 'm' in flags:
            re_flags |= re.MULTILINE
        if 's' in flags:
            re_flags |= re.DOTALL

    try:
        compiled = re.compile(pattern, re_flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "content": content,
            "replacements": 0,
        }

    # Perform replacement
    new_content, num_replacements = compiled.subn(replacement, content, count=count)

    result["content"] = new_content
    result["replacements"] = num_replacements
    result["changed"] = num_replacements > 0

    return result


# Additional utility functions

def find_all_occurrences(
    text: str,
    content: str,
    case_sensitive: bool = True,
) -> Dict[str, Any]:
    """
    Find all occurrences of a literal string (not regex).

    Args:
        text: Literal string to find
        content: Text content to search
        case_sensitive: Whether search is case-sensitive

    Returns:
        dict with positions and line numbers of occurrences
    """
    result = {"success": True, "occurrences": []}

    search_content = content if case_sensitive else content.lower()
    search_text = text if case_sensitive else text.lower()

    line_starts = _build_line_index(content)

    pos = 0
    while True:
        pos = search_content.find(search_text, pos)
        if pos == -1:
            break

        line_num = _pos_to_line(pos, line_starts)
        line_col = pos - line_starts[line_num - 1] if line_num > 0 else pos

        result["occurrences"].append({
            "position": pos,
            "line": line_num,
            "column": line_col + 1,
        })

        pos += 1

    result["count"] = len(result["occurrences"])
    return result


def diff_lines(
    content1: str,
    content2: str,
    context: int = 3,
) -> Dict[str, Any]:
    """
    Generate unified diff between two contents.

    Args:
        content1: Original content
        content2: Modified content
        context: Number of context lines

    Returns:
        dict with diff information
    """
    import difflib

    result = {"success": True}

    lines1 = content1.splitlines(keepends=True)
    lines2 = content2.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        lines1, lines2,
        fromfile='original',
        tofile='modified',
        n=context,
    ))

    result["diff"] = "".join(diff)
    result["has_changes"] = len(diff) > 0

    # Count changes
    additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

    result["additions"] = additions
    result["deletions"] = deletions

    return result


# Helper functions

def _build_line_index(content: str) -> List[int]:
    """Build index of line start positions."""
    line_starts = [0]
    for i, char in enumerate(content):
        if char == '\n':
            line_starts.append(i + 1)
    return line_starts


def _pos_to_line(pos: int, line_starts: List[int]) -> int:
    """Convert character position to line number (1-indexed)."""
    for i, start in enumerate(line_starts):
        if pos < start:
            return i
    return len(line_starts)
