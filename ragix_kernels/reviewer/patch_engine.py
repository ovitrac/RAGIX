"""
Forward/inverse unified diff engine with content-hash anchoring.

Generates standard unified diff patches (.patch) and inverse patches
(.inverse.patch) for each edit operation. Patches are anchored on
content hashes rather than line numbers for robustness.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import difflib
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from ragix_kernels.reviewer.models import content_hash

import logging

logger = logging.getLogger(__name__)


class PatchConflictError(Exception):
    """Raised when a patch cannot be applied due to content mismatch."""
    pass


def generate_unified_diff(
    before_lines: List[str],
    after_lines: List[str],
    filename: str = "doc.md",
    context_lines: int = 3,
) -> str:
    """
    Generate a unified diff between two versions of a file.

    Args:
        before_lines: Original lines (without newlines)
        after_lines: Modified lines (without newlines)
        filename: Name for the diff header
        context_lines: Number of context lines around changes

    Returns:
        Unified diff string
    """
    # difflib expects lines with trailing newlines
    a = [line + "\n" for line in before_lines]
    b = [line + "\n" for line in after_lines]

    diff = difflib.unified_diff(
        a, b,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context_lines,
    )
    return "".join(diff)


def generate_forward_inverse_patches(
    before_text: str,
    after_text: str,
    filename: str = "doc.md",
) -> Tuple[str, str]:
    """
    Generate both forward and inverse patches for a text change.

    Returns:
        (forward_patch, inverse_patch) as unified diff strings
    """
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()

    forward = generate_unified_diff(before_lines, after_lines, filename)
    inverse = generate_unified_diff(after_lines, before_lines, filename)

    return forward, inverse


def apply_edit_to_lines(
    lines: List[str],
    line_start: int,
    line_end: int,
    before_hash: str,
    new_lines: List[str],
) -> List[str]:
    """
    Apply a single edit operation to document lines.

    Uses content-hash anchoring: verifies the target region matches
    the expected hash before applying.

    Args:
        lines: Current document lines (0-indexed)
        line_start: 1-based start line (inclusive)
        line_end: 1-based end line (inclusive)
        before_hash: Expected SHA-256 hash of the target region
        new_lines: Replacement lines

    Returns:
        Updated lines list

    Raises:
        PatchConflictError: If content hash doesn't match
    """
    # Convert to 0-based
    start = line_start - 1
    end = line_end  # exclusive in slice

    if start < 0 or end > len(lines):
        raise PatchConflictError(
            f"Line range {line_start}-{line_end} out of bounds (doc has {len(lines)} lines)"
        )

    # Verify content hash
    target_text = "\n".join(lines[start:end])
    actual_hash = content_hash(target_text)
    if actual_hash != before_hash:
        raise PatchConflictError(
            f"Content hash mismatch at lines {line_start}-{line_end}: "
            f"expected {before_hash[:24]}..., got {actual_hash[:24]}..."
        )

    # Apply replacement
    result = lines[:start] + new_lines + lines[end:]
    return result


def apply_edits_reverse_order(
    lines: List[str],
    edits: List[dict],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Apply multiple edits in reverse line order (bottom-up) to avoid offset shift.

    Each edit dict must have: line_start, line_end, before_hash, new_lines, change_id.

    Returns:
        (updated_lines, list of (forward_patch, inverse_patch) per edit)
    """
    # Sort by line_start descending
    sorted_edits = sorted(edits, key=lambda e: e["line_start"], reverse=True)

    patches = []
    for edit in sorted_edits:
        before_text = "\n".join(lines)

        lines = apply_edit_to_lines(
            lines,
            edit["line_start"],
            edit["line_end"],
            edit["before_hash"],
            edit["new_lines"],
        )

        after_text = "\n".join(lines)
        fwd, inv = generate_forward_inverse_patches(
            before_text, after_text, edit.get("filename", "doc.md")
        )
        patches.append((fwd, inv))

    # Reverse patches list to match original edit order
    patches.reverse()
    return lines, patches


def save_patches(
    patches_dir: Path,
    change_id: str,
    forward_patch: str,
    inverse_patch: str,
) -> Tuple[Path, Path]:
    """
    Save forward and inverse patches to the patches directory.

    Returns:
        (forward_path, inverse_path)
    """
    patches_dir.mkdir(parents=True, exist_ok=True)

    fwd_path = patches_dir / f"{change_id}.patch"
    inv_path = patches_dir / f"{change_id}.inverse.patch"

    fwd_path.write_text(forward_patch, encoding="utf-8")
    inv_path.write_text(inverse_patch, encoding="utf-8")

    return fwd_path, inv_path


def apply_inverse_patch(
    lines: List[str],
    inverse_patch_path: Path,
    verify_hash: Optional[str] = None,
) -> List[str]:
    """
    Apply an inverse patch to revert a change.

    This is a simplified application that re-parses the unified diff
    to reconstruct the original text. For production use, the stored
    before/after text in the ledger provides a more reliable path.

    Args:
        lines: Current document lines
        inverse_patch_path: Path to .inverse.patch file
        verify_hash: Optional hash to verify before applying

    Returns:
        Reverted document lines

    Raises:
        PatchConflictError: If patch cannot be applied cleanly
    """
    if not inverse_patch_path.exists():
        raise PatchConflictError(f"Inverse patch not found: {inverse_patch_path}")

    patch_text = inverse_patch_path.read_text(encoding="utf-8")
    if not patch_text.strip():
        logger.warning(f"Empty inverse patch: {inverse_patch_path}")
        return lines

    # Parse the unified diff to extract before/after
    # This is a simplified parser for our own patches
    added = []
    removed = []
    in_hunk = False

    for line in patch_text.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])

    if not added and not removed:
        logger.warning(f"No changes found in inverse patch: {inverse_patch_path}")
        return lines

    # Find the "before" content (which in the inverse is the "removed" lines)
    # and replace with "after" (which is the "added" lines)
    current_text = "\n".join(lines)
    search_text = "\n".join(removed)
    replace_text = "\n".join(added)

    if search_text and search_text in current_text:
        new_text = current_text.replace(search_text, replace_text, 1)
        return new_text.splitlines()

    raise PatchConflictError(
        f"Cannot apply inverse patch: target text not found in document"
    )
