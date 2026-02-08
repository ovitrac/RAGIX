"""
Kernel: md_inline_notes_inject
Stage: 3 (Reporting)

For every needs_attention=true op, inject GitHub alert block immediately
after the modified text. Each alert starts with REVIEWER: RVW-NNNN.

Alert types mapped to severity:
    minor    -> [!NOTE]
    attention -> [!WARNING]
    deletion  -> [!CAUTION]
    critical  -> [!IMPORTANT]

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import LedgerEntry

import logging

logger = logging.getLogger(__name__)

DEFAULT_SEVERITY_MAP = {
    "minor": "NOTE",
    "attention": "WARNING",
    "deletion": "CAUTION",
    "critical": "IMPORTANT",
}


class MdInlineNotesInjectKernel(Kernel):
    """Inject GitHub alert blocks after attention-requiring edits."""

    name = "md_inline_notes_inject"
    version = "1.0.0"
    category = "reviewer"
    stage = 3
    description = "GitHub alert blocks (REVIEWER: prefix)"

    requires: List[str] = ["md_apply_ops"]
    provides: List[str] = ["annotated_doc"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load edited document
        edited_path = input.workspace / "stage3" / "doc.edited.md"
        if not edited_path.exists():
            raise RuntimeError("Missing edited document from md_apply_ops")

        text = edited_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Load ledger
        ledger_path = input.workspace / "review" / "ledger.jsonl"
        entries: List[LedgerEntry] = []
        if ledger_path.exists():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    entries.append(LedgerEntry.from_dict(json.loads(line)))

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        severity_map = reviewer_cfg.get("severity_alert_map", DEFAULT_SEVERITY_MAP)

        # Collect notes to inject (from entries with review_note)
        # Sort by line position descending to inject bottom-up
        notes_to_inject: List[Dict[str, Any]] = []
        for entry in entries:
            if entry.is_revert or not entry.review_note:
                continue
            note = entry.review_note
            alert_type = note.get("alert", severity_map.get(entry.severity, "NOTE"))
            note_text = note.get("text", f"REVIEWER: {entry.id}")

            # Find insertion point: after the edit scope
            scope = entry.scope or {}
            # Use the apply log to find the line
            apply_log_path = input.workspace / "stage3" / "apply_log.jsonl"
            insert_after = 0
            if apply_log_path.exists():
                for log_line in apply_log_path.read_text().splitlines():
                    log_entry = json.loads(log_line)
                    if log_entry.get("id") == entry.id:
                        insert_after = log_entry.get("line_end", 0)
                        break

            if insert_after > 0:
                notes_to_inject.append({
                    "change_id": entry.id,
                    "insert_after_line": insert_after,
                    "alert_type": alert_type,
                    "text": note_text,
                })

        # Sort descending by insertion point (inject bottom-up)
        notes_to_inject.sort(key=lambda n: n["insert_after_line"], reverse=True)

        injected_count = 0
        for note in notes_to_inject:
            insert_idx = note["insert_after_line"]  # 1-based
            if insert_idx > len(lines):
                insert_idx = len(lines)

            alert_block = [
                "",
                f"> [!{note['alert_type']}]",
                f"> {note['text']}",
                "",
            ]
            # Insert after the target line (0-based index = insert_idx)
            lines = lines[:insert_idx] + alert_block + lines[insert_idx:]
            injected_count += 1

        # Determine output path
        doc_name = input.config.get("doc_path", "doc.md")
        in_place = input.config.get("reviewer", {}).get("in_place", False)
        suffix = input.config.get("reviewer", {}).get("output_suffix", ".REVIEWED.md")

        if in_place:
            output_path = Path(doc_name)
        else:
            base = Path(doc_name)
            output_path = base.parent / (base.stem + suffix)

        # Save annotated document
        stage3_dir = input.workspace / "stage3"
        reviewed_path = stage3_dir / output_path.name
        reviewed_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info(
            f"[md_inline_notes_inject] Injected {injected_count} alert blocks"
        )

        return {
            "injected_count": injected_count,
            "reviewed_doc_path": str(reviewed_path),
            "notes": [n["change_id"] for n in notes_to_inject],
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        return (
            f"Injected {data['injected_count']} inline review notes. "
            f"Output: {data['reviewed_doc_path']}"
        )
