"""
Kernel: md_revert
Stage: 3 (Reporting)

Selective revert of individual changes via inverse patches.
Supports single-change and bulk (reverse-chronological) revert.
Records revert events in the ledger for full traceability.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import content_hash
from ragix_kernels.reviewer.patch_engine import (
    PatchConflictError,
    apply_inverse_patch,
    generate_forward_inverse_patches,
    save_patches,
)
from ragix_kernels.reviewer.ledger import Ledger, create_ledger_entry

import logging

logger = logging.getLogger(__name__)


def revert_single(
    doc_lines: List[str],
    change_id: str,
    ledger: Ledger,
    patches_dir: Path,
    doc_path: str = "doc.md",
) -> Dict[str, Any]:
    """
    Revert a single change by applying its inverse patch.

    Args:
        doc_lines: Current document lines
        change_id: The RVW-NNNN change ID to revert
        ledger: Active ledger instance
        patches_dir: Directory containing .patch/.inverse.patch files
        doc_path: Document path for ledger entry

    Returns:
        Dict with reverted_lines, success, and metadata

    Raises:
        PatchConflictError: If the inverse patch cannot be applied
    """
    # Verify the change exists and hasn't been reverted
    entry = ledger.get_by_id(change_id)
    if entry is None:
        return {
            "success": False,
            "error": f"Change {change_id} not found in ledger",
            "reverted_lines": doc_lines,
        }

    if entry.is_revert:
        return {
            "success": False,
            "error": f"{change_id} is itself a revert entry, cannot revert a revert",
            "reverted_lines": doc_lines,
        }

    # Check if already reverted
    existing_reverts = ledger.filter(is_revert=True)
    for r in existing_reverts:
        if r.reverted_id == change_id:
            return {
                "success": False,
                "error": f"Change {change_id} has already been reverted by {r.id}",
                "reverted_lines": doc_lines,
            }

    # Find inverse patch
    inv_path = patches_dir / f"{change_id}.inverse.patch"
    if not inv_path.exists():
        return {
            "success": False,
            "error": f"Inverse patch not found: {inv_path}",
            "reverted_lines": doc_lines,
        }

    # Apply inverse patch
    before_text = "\n".join(doc_lines)
    doc_hash_before = content_hash(before_text)

    try:
        reverted_lines = apply_inverse_patch(doc_lines, inv_path)
    except PatchConflictError as e:
        return {
            "success": False,
            "error": f"Patch conflict: {e}",
            "reverted_lines": doc_lines,
        }

    after_text = "\n".join(reverted_lines)
    doc_hash_after = content_hash(after_text)

    # Generate revert patches
    fwd_patch, inv_patch = generate_forward_inverse_patches(
        before_text, after_text, Path(doc_path).name
    )
    revert_id = f"REVERT-{change_id}"
    fwd_path, inv_path_new = save_patches(patches_dir, revert_id, fwd_patch, inv_patch)

    # Record revert in ledger
    revert_entry = create_ledger_entry(
        change_id=revert_id,
        doc_path=doc_path,
        doc_hash_before=doc_hash_before,
        doc_hash_after=doc_hash_after,
        kind="revert",
        severity="minor",
        silent=False,
        summary=f"Reverted change {change_id}",
        rationale=f"User-initiated revert of {change_id}: {entry.summary[:100]}",
        patch_forward=str(fwd_path),
        patch_inverse=str(inv_path_new),
        scope=entry.scope,
    )
    revert_entry.is_revert = True
    revert_entry.reverted_id = change_id
    ledger.append(revert_entry)

    logger.info(f"[md_revert] Reverted {change_id} -> {revert_id}")

    return {
        "success": True,
        "revert_id": revert_id,
        "reverted_change": change_id,
        "doc_hash_before": doc_hash_before,
        "doc_hash_after": doc_hash_after,
        "reverted_lines": reverted_lines,
    }


def revert_bulk(
    doc_lines: List[str],
    change_ids: List[str],
    ledger: Ledger,
    patches_dir: Path,
    doc_path: str = "doc.md",
) -> Dict[str, Any]:
    """
    Revert multiple changes in reverse chronological order.

    Changes are reverted most-recent-first to minimize patch conflicts.
    Stops at the first failure.
    """
    # Sort by sequence number descending (most recent first)
    from ragix_kernels.reviewer.models import ChangeID

    def _sort_key(cid: str) -> int:
        try:
            return -ChangeID.parse(cid).seq
        except ValueError:
            return 0

    sorted_ids = sorted(change_ids, key=_sort_key)

    results = []
    current_lines = doc_lines

    for cid in sorted_ids:
        result = revert_single(current_lines, cid, ledger, patches_dir, doc_path)
        results.append(result)
        if result["success"]:
            current_lines = result["reverted_lines"]
        else:
            logger.warning(f"[md_revert] Bulk revert stopped at {cid}: {result['error']}")
            break

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    return {
        "reverted_lines": current_lines,
        "total_reverted": len(successful),
        "total_failed": len(failed),
        "results": results,
    }


class MdRevertKernel(Kernel):
    """Selective revert of individual changes via inverse patches."""

    name = "md_revert"
    version = "1.0.0"
    category = "reviewer"
    stage = 3
    description = "Selective revert via inverse patches"

    requires: List[str] = ["md_apply_ops"]
    provides: List[str] = ["reverted_doc"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Which changes to revert
        revert_ids = input.config.get("revert_ids", [])
        if not revert_ids:
            logger.info("[md_revert] No change IDs specified for revert")
            return {
                "total_reverted": 0,
                "total_failed": 0,
                "results": [],
            }

        # Load current edited document
        edited_path = input.workspace / "stage3" / "doc.edited.md"
        if not edited_path.exists():
            raise RuntimeError("Missing edited document from md_apply_ops")

        text = edited_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        doc_path = input.config.get("doc_path", "doc.md")

        # Setup
        review_dir = input.workspace / "review"
        patches_dir = review_dir / "patches"
        ledger = Ledger(review_dir / "ledger.jsonl")

        # Perform revert
        if len(revert_ids) == 1:
            result = revert_single(lines, revert_ids[0], ledger, patches_dir, doc_path)
            if result["success"]:
                final_lines = result["reverted_lines"]
                results = [result]
                total_reverted = 1
                total_failed = 0
            else:
                final_lines = lines
                results = [result]
                total_reverted = 0
                total_failed = 1
        else:
            bulk_result = revert_bulk(lines, revert_ids, ledger, patches_dir, doc_path)
            final_lines = bulk_result["reverted_lines"]
            results = bulk_result["results"]
            total_reverted = bulk_result["total_reverted"]
            total_failed = bulk_result["total_failed"]

        # Save reverted document
        if total_reverted > 0:
            stage3_dir = input.workspace / "stage3"
            reverted_path = stage3_dir / "doc.edited.md"
            reverted_path.write_text("\n".join(final_lines), encoding="utf-8")

        logger.info(
            f"[md_revert] Reverted {total_reverted}/{len(revert_ids)} changes"
        )

        return {
            "total_reverted": total_reverted,
            "total_failed": total_failed,
            "results": [
                {k: v for k, v in r.items() if k != "reverted_lines"}
                for r in results
            ],
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        return (
            f"Reverted {data['total_reverted']} changes, "
            f"{data['total_failed']} failed."
        )
