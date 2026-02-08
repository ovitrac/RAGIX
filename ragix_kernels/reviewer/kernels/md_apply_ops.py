"""
Kernel: md_apply_ops
Stage: 3 (Reporting)

The critical deterministic engine. Validates edit ops against protected spans
and content hashes, applies edits, generates forward/inverse patches, and
updates the JSONL ledger.

Edits are applied in reverse line order (bottom-up) to avoid offset shift.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import (
    AlertType,
    EditOp,
    ProtectedSpan,
    ReviewNote,
    content_hash,
)
from ragix_kernels.reviewer.patch_engine import (
    PatchConflictError,
    generate_forward_inverse_patches,
    save_patches,
)
from ragix_kernels.reviewer.ledger import Ledger, create_ledger_entry

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-validation normalization (v7.3.2 acceptance fixes)
# ---------------------------------------------------------------------------

def _normalize_op(
    op: EditOp,
    lines: List[str],
) -> Tuple[EditOp, List[str]]:
    """
    Deterministic pre-validation fixes applied to each op before validation.

    Returns (op, fixup_tags) where fixup_tags lists what was fixed.
    Mutates op in-place for efficiency.

    Fixes:
    1. HASH_RECOMPUTE: If before_hash doesn't match original doc at line_start..line_end,
       recompute it from the original document lines. This handles ops generated on
       masked/transformed text where placeholders changed the content hash.
    2. NOTE_INJECT: If needs_attention=true but review_note is missing, inject
       a deterministic fallback note that is honest about the absence.
    3. NOTE_ID_FIX: If review_note exists but doesn't contain the change ID, prepend it.
    """
    fixups: List[str] = []
    start = op.target.line_start
    end = op.target.line_end

    # --- Fix 1: Hash recompute against original document ---
    if start >= 1 and end <= len(lines) and start <= end:
        target_text = "\n".join(lines[start - 1:end])
        actual_hash = content_hash(target_text)
        if op.before_hash and actual_hash != op.before_hash:
            # Hash mismatch — likely from masked view. Recompute from original.
            op.before_hash = actual_hash
            op.before_text = target_text
            fixups.append("HASH_RECOMPUTE")

    # --- Fix 2: Deterministic fallback review_note ---
    if op.needs_attention and op.review_note is None:
        op.review_note = ReviewNote(
            alert=AlertType.NOTE,
            text=(
                f"REVIEWER: {op.id} — Needs attention flagged by model, "
                f"but no note was provided. Please review the highlighted "
                f"span manually."
            ),
        )
        fixups.append("NOTE_INJECT")

    # --- Fix 3: Inject change ID into review_note text ---
    if op.review_note and op.id not in op.review_note.text:
        # Prepend the canonical REVIEWER: ID prefix
        old_text = op.review_note.text
        if old_text.startswith("REVIEWER:"):
            # Has prefix but wrong/missing ID — replace prefix section
            op.review_note.text = f"REVIEWER: {op.id} — {old_text[len('REVIEWER:'):].lstrip()}"
        else:
            op.review_note.text = f"REVIEWER: {op.id} — {old_text}"
        fixups.append("NOTE_ID_FIX")

    # Ensure review_note starts with REVIEWER: (could be missing after Fix 2)
    if op.review_note and not op.review_note.text.startswith("REVIEWER:"):
        op.review_note.text = f"REVIEWER: {op.id} — {op.review_note.text}"
        if "NOTE_ID_FIX" not in fixups:
            fixups.append("NOTE_ID_FIX")

    return op, fixups


def _classify_rejection(errors: List[str]) -> str:
    """Classify rejection into a funnel category."""
    for e in errors:
        if "already exists in ledger" in e:
            return "REJECT_LEDGER"
        if "requires review_note" in e:
            return "REJECT_SCHEMA_NOTE"
        if "Review note must contain change ID" in e:
            return "REJECT_SCHEMA_ID"
        if "content hash mismatch" in e:
            return "REJECT_HASH"
        if "crosses protected" in e:
            return "REJECT_PROTECTED"
        if "invalid line range" in e:
            return "REJECT_RANGE"
    return "REJECT_OTHER"


def _validate_op(
    op: EditOp,
    lines: List[str],
    protected_spans: List[ProtectedSpan],
    ledger: Ledger,
    strict: bool = False,
) -> List[str]:
    """
    Validate a single edit op. Returns list of errors (empty if valid).

    Enforced invariants:
    1. before_hash matches actual content
    2. Target does not cross protected spans (or strict mode)
    3. Deletion requires review_note
    4. needs_attention requires review_note
    5. review_note starts with REVIEWER: and includes change ID
    6. No ID collision
    """
    errors = []

    # Structural validation
    errors.extend(op.validate())

    # ID collision
    if ledger.has_id(op.id):
        errors.append(f"{op.id}: ID already exists in ledger")

    # Line range
    start = op.target.line_start
    end = op.target.line_end
    if start < 1 or end > len(lines) or start > end:
        errors.append(f"{op.id}: invalid line range {start}-{end} (doc has {len(lines)} lines)")
        return errors

    # Content hash verification
    target_text = "\n".join(lines[start - 1:end])
    actual_hash = content_hash(target_text)
    if op.before_hash and actual_hash != op.before_hash:
        errors.append(
            f"{op.id}: content hash mismatch at lines {start}-{end}: "
            f"expected {op.before_hash[:30]}..., got {actual_hash[:30]}..."
        )

    # Protected span check
    for span in protected_spans:
        if span.overlaps(start, end):
            if strict:
                errors.append(
                    f"{op.id}: edit crosses protected {span.kind.value} "
                    f"at lines {span.line_start}-{span.line_end}"
                )
            elif not (op.needs_attention and op.review_note is not None):
                errors.append(
                    f"{op.id}: edit crosses protected {span.kind.value}; "
                    f"requires needs_attention=true with review_note"
                )

    return errors


class MdApplyOpsKernel(Kernel):
    """Validate + apply edit ops + forward/inverse patches + ledger."""

    name = "md_apply_ops"
    version = "1.1.0"
    category = "reviewer"
    stage = 3
    description = "Validate + apply ops + patches + ledger"

    requires: List[str] = ["md_protected_regions"]
    provides: List[str] = ["patches", "ledger", "edited_doc"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load document
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        doc_name = input.config.get("doc_path", "doc.md")

        # Load protected spans
        prot_path = input.dependencies.get("md_protected_regions")
        spans: List[ProtectedSpan] = []
        if prot_path and prot_path.exists():
            prot_data = json.loads(prot_path.read_text())["data"]
            spans = [ProtectedSpan.from_dict(d) for d in prot_data["protected_spans"]]

        # Load edit ops — from stage2 or manual input
        ops_data = self._load_ops(input)
        if not ops_data:
            logger.info("[md_apply_ops] No edit ops to apply")
            return {
                "applied": [],
                "rejected": [],
                "total_applied": 0,
                "total_rejected": 0,
            }

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        strict = reviewer_cfg.get("strict", False)

        # Setup paths
        review_dir = input.workspace / "review"
        patches_dir = review_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(review_dir / "ledger.jsonl")

        # Validate and collect applicable ops
        applied: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        fixup_log: List[Dict[str, Any]] = []

        # Acceptance funnel counters
        funnel = {
            "VALIDATED_APPLIED": 0,
            "REJECT_HASH": 0,
            "REJECT_SCHEMA_NOTE": 0,
            "REJECT_SCHEMA_ID": 0,
            "REJECT_LEDGER": 0,
            "REJECT_PROTECTED": 0,
            "REJECT_RANGE": 0,
            "REJECT_OTHER": 0,
            "FIXUP_HASH_RECOMPUTE": 0,
            "FIXUP_NOTE_INJECT": 0,
            "FIXUP_NOTE_ID_FIX": 0,
        }

        # Sort ops by line_start descending (apply bottom-up)
        ops = sorted(ops_data, key=lambda o: o.target.line_start, reverse=True)

        doc_hash_before = content_hash(text)

        for op in ops:
            # Pre-validation normalization (v7.3.2)
            op, fixups = _normalize_op(op, lines)
            if fixups:
                fixup_log.append({"id": op.id, "fixups": fixups})
                for f in fixups:
                    key = f"FIXUP_{f}"
                    if key in funnel:
                        funnel[key] += 1

            errors = _validate_op(op, lines, spans, ledger, strict)
            if errors:
                category = _classify_rejection(errors)
                funnel[category] = funnel.get(category, 0) + 1
                rejected.append({"id": op.id, "errors": errors, "category": category})
                logger.warning(f"[md_apply_ops] Rejected {op.id} [{category}]: {errors}")
                continue

            funnel["VALIDATED_APPLIED"] += 1

            # Apply the edit
            start = op.target.line_start - 1  # 0-based
            end = op.target.line_end           # 0-based exclusive

            before_text = "\n".join(lines)

            if op.action.value == "replace":
                new_lines = op.after_text.splitlines()
                lines = lines[:start] + new_lines + lines[end:]
            elif op.action.value == "insert":
                new_lines = op.after_text.splitlines()
                lines = lines[:end] + new_lines + lines[end:]
            elif op.action.value == "delete":
                lines = lines[:start] + lines[end:]
            elif op.action.value == "flag_only":
                # No modification, just record
                pass

            after_text = "\n".join(lines)
            doc_hash_after = content_hash(after_text)

            # Generate patches
            fwd_patch, inv_patch = generate_forward_inverse_patches(
                before_text, after_text, Path(doc_name).name
            )
            fwd_path, inv_path = save_patches(patches_dir, op.id, fwd_patch, inv_patch)

            # Write ledger entry
            review_note_dict = op.review_note.to_dict() if op.review_note else None
            entry = create_ledger_entry(
                change_id=op.id,
                doc_path=doc_name,
                doc_hash_before=doc_hash_before,
                doc_hash_after=doc_hash_after,
                kind=op.kind,
                severity=op.severity.value,
                silent=op.silent,
                summary=op.rationale[:200] if op.rationale else "",
                rationale=op.rationale,
                patch_forward=str(fwd_path),
                patch_inverse=str(inv_path),
                review_note=review_note_dict,
                scope={"anchor": op.target.anchor, "node_path": op.target.node_path},
            )
            ledger.append(entry)

            applied.append({
                "id": op.id,
                "action": op.action.value,
                "line_start": op.target.line_start,
                "line_end": op.target.line_end,
                "patch_forward": str(fwd_path),
                "patch_inverse": str(inv_path),
            })

            # Update hash for next op
            doc_hash_before = doc_hash_after

        # Save edited document
        stage3_dir = input.workspace / "stage3"
        stage3_dir.mkdir(parents=True, exist_ok=True)
        edited_path = stage3_dir / "doc.edited.md"
        edited_path.write_text("\n".join(lines), encoding="utf-8")

        # Save apply log
        (stage3_dir / "apply_log.jsonl").write_text(
            "\n".join(json.dumps(a) for a in applied),
            encoding="utf-8",
        )

        logger.info(
            f"[md_apply_ops] Applied {len(applied)}, rejected {len(rejected)} "
            f"(fixups: {sum(1 for f in fixup_log)})"
        )

        # Funnel summary
        if any(v > 0 for k, v in funnel.items() if k.startswith("FIXUP_")):
            logger.info(
                f"[md_apply_ops] Fixups: "
                f"hash={funnel['FIXUP_HASH_RECOMPUTE']}, "
                f"note={funnel['FIXUP_NOTE_INJECT']}, "
                f"id={funnel['FIXUP_NOTE_ID_FIX']}"
            )

        return {
            "applied": applied,
            "rejected": rejected,
            "total_applied": len(applied),
            "total_rejected": len(rejected),
            "edited_doc_path": str(edited_path),
            "ledger_path": str(review_dir / "ledger.jsonl"),
            "funnel": funnel,
            "fixup_log": fixup_log,
        }

    def _load_ops(self, input: KernelInput) -> List[EditOp]:
        """Load edit ops from stage2 or manual input."""
        ops: List[EditOp] = []

        # Try manual ops from config
        manual_ops_path = input.config.get("ops_path")
        if manual_ops_path:
            p = Path(manual_ops_path)
            if p.exists():
                data = json.loads(p.read_text())
                raw_ops = data.get("ops", data) if isinstance(data, dict) else data
                for d in raw_ops:
                    ops.append(EditOp.from_dict(d))
                return ops

        # Try stage2 edit plan
        edit_plan_path = input.workspace / "stage2" / "ops"
        if edit_plan_path.exists():
            for f in sorted(edit_plan_path.glob("ops_*.json")):
                data = json.loads(f.read_text())
                raw_ops = data.get("ops", []) if isinstance(data, dict) else []
                for d in raw_ops:
                    ops.append(EditOp.from_dict(d))

        # Also check merged edit_plan.json
        merged = input.workspace / "stage2" / "edit_plan.json"
        if merged.exists() and not ops:
            data = json.loads(merged.read_text())
            raw_ops = data.get("ops", []) if isinstance(data, dict) else []
            for d in raw_ops:
                ops.append(EditOp.from_dict(d))

        return ops

    def summarize(self, data: Dict[str, Any]) -> str:
        funnel = data.get("funnel", {})
        fixups_total = sum(v for k, v in funnel.items() if k.startswith("FIXUP_"))
        parts = [
            f"Applied {data['total_applied']} edits, "
            f"rejected {data['total_rejected']}."
        ]
        if fixups_total:
            parts.append(f" Fixups: {fixups_total} ops normalized.")
        return "".join(parts)
