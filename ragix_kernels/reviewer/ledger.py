"""
Append-only JSONL ledger for change tracking and audit trail.

Thread-safe via threading.Lock. Reverts add new entries (never mutate).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ragix_kernels.reviewer.models import ChangeID, LedgerEntry

import logging

logger = logging.getLogger(__name__)


class Ledger:
    """
    Append-only JSONL ledger for tracking all review changes.

    Thread-safe. Supports filtered queries by ID, kind, severity, timestamp.
    Allocates sequential ChangeIDs.
    """

    def __init__(self, ledger_path: Path):
        self._path = ledger_path
        self._lock = threading.Lock()
        self._next_seq = 1
        self._entries: List[LedgerEntry] = []

        # Load existing entries if file exists
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        """Load existing ledger entries from JSONL file."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        entry = LedgerEntry.from_dict(d)
                        self._entries.append(entry)
                        # Track highest sequence number
                        try:
                            cid = ChangeID.parse(entry.id)
                            self._next_seq = max(self._next_seq, cid.seq + 1)
                        except ValueError:
                            pass
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed ledger line {line_num}: {e}")
        except OSError as e:
            logger.error(f"Cannot read ledger: {e}")

    def allocate_id(self, namespace: Optional[str] = None) -> ChangeID:
        """Allocate the next sequential ChangeID (thread-safe)."""
        with self._lock:
            cid = ChangeID(seq=self._next_seq, namespace=namespace)
            self._next_seq += 1
            return cid

    def append(self, entry: LedgerEntry) -> None:
        """Append a single entry (thread-safe, writes immediately to disk)."""
        with self._lock:
            self._entries.append(entry)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(entry.to_json_line() + "\n")

    def get_all(self) -> List[LedgerEntry]:
        """Return all entries in chronological order."""
        with self._lock:
            return list(self._entries)

    def get_by_id(self, change_id: str) -> Optional[LedgerEntry]:
        """Find a single entry by its change ID."""
        with self._lock:
            for entry in self._entries:
                if entry.id == change_id:
                    return entry
            return None

    def filter(
        self,
        kind: Optional[str] = None,
        severity: Optional[str] = None,
        silent: Optional[bool] = None,
        since: Optional[str] = None,
        is_revert: Optional[bool] = None,
    ) -> List[LedgerEntry]:
        """
        Filter entries by criteria.

        Args:
            kind: Filter by change kind (e.g. "typo", "logic_flow")
            severity: Filter by severity level
            silent: Filter by silent flag
            since: ISO 8601 timestamp; return entries after this time
            is_revert: Filter revert entries

        Returns:
            Matching entries in chronological order
        """
        with self._lock:
            result = list(self._entries)

        if kind is not None:
            result = [e for e in result if e.kind == kind]
        if severity is not None:
            result = [e for e in result if e.severity == severity]
        if silent is not None:
            result = [e for e in result if e.silent == silent]
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        if is_revert is not None:
            result = [e for e in result if e.is_revert == is_revert]

        return result

    def has_id(self, change_id: str) -> bool:
        """Check if a change ID already exists in the ledger."""
        with self._lock:
            return any(e.id == change_id for e in self._entries)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def next_seq(self) -> int:
        with self._lock:
            return self._next_seq

    def summary_counts(self) -> Dict[str, int]:
        """Return counts for report generation."""
        entries = self.filter(is_revert=False)
        attention = sum(1 for e in entries if not e.silent and e.severity != "minor")
        deletions = sum(1 for e in entries if e.kind == "deletion" or "delet" in e.summary.lower())
        silent = sum(1 for e in entries if e.silent)
        reverts = len(self.filter(is_revert=True))
        return {
            "total_changes": len(entries),
            "attention_changes": attention,
            "deletions": deletions,
            "silent_minor": silent,
            "reverts": reverts,
        }


def create_ledger_entry(
    change_id: str,
    doc_path: str,
    doc_hash_before: str,
    doc_hash_after: str = "",
    kind: str = "",
    severity: str = "minor",
    silent: bool = False,
    summary: str = "",
    rationale: str = "",
    patch_forward: str = "",
    patch_inverse: str = "",
    review_note: Optional[Dict[str, str]] = None,
    actor: Optional[Dict[str, str]] = None,
    scope: Optional[Dict[str, str]] = None,
) -> LedgerEntry:
    """Convenience factory for creating a LedgerEntry with current timestamp."""
    return LedgerEntry(
        id=change_id,
        doc=doc_path,
        doc_hash_before=doc_hash_before,
        doc_hash_after=doc_hash_after,
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor=actor or {"tool": "reviewctl", "operator": "koas_reviewer"},
        scope=scope or {},
        kind=kind,
        severity=severity,
        silent=silent,
        summary=summary,
        rationale=rationale,
        patch_forward=patch_forward,
        patch_inverse=patch_inverse,
        review_note=review_note,
    )
