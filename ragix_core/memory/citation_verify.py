"""
Citation Verification — Deterministic Post-Processor

Verifies that [MID: xxx] citations in generated summaries reference
valid memory items. Builds a citation map for audit trail.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_core.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Pattern to match [MID: xxx] citations in text
_MID_PATTERN = re.compile(r"\[MID:\s*(MEM-[a-f0-9]+)\]", re.IGNORECASE)
# Also match [MID: xxx, yyy] multi-citations
_MID_MULTI_PATTERN = re.compile(
    r"\[MID:\s*(MEM-[a-f0-9]+(?:\s*,\s*MEM-[a-f0-9]+)*)\]", re.IGNORECASE
)


@dataclass
class CitationEntry:
    """A single citation reference."""
    bullet_text: str
    line_number: int
    mids: List[str]
    valid: bool = True
    invalid_mids: List[str] = field(default_factory=list)


@dataclass
class CitationReport:
    """Full citation verification report."""
    total_bullets: int = 0
    bullets_with_citation: int = 0
    bullets_without_citation: int = 0
    total_mids: int = 0
    valid_mids: int = 0
    invalid_mids: int = 0
    unique_mids: int = 0
    domain_coverage: Dict[str, int] = field(default_factory=dict)
    entries: List[CitationEntry] = field(default_factory=list)
    uncited_bullets: List[Tuple[int, str]] = field(default_factory=list)
    invalid_mid_list: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize citation report to a plain dictionary."""
        return {
            "total_bullets": self.total_bullets,
            "bullets_with_citation": self.bullets_with_citation,
            "bullets_without_citation": self.bullets_without_citation,
            "citation_rate": (
                f"{self.bullets_with_citation / self.total_bullets:.1%}"
                if self.total_bullets > 0 else "N/A"
            ),
            "total_mids": self.total_mids,
            "valid_mids": self.valid_mids,
            "invalid_mids": self.invalid_mids,
            "unique_mids_referenced": self.unique_mids,
            "domain_coverage": self.domain_coverage,
            "uncited_bullets": [
                {"line": ln, "text": txt[:80]} for ln, txt in self.uncited_bullets
            ],
            "invalid_mid_list": self.invalid_mid_list,
        }


def verify_citations(
    text: str,
    store: MemoryStore,
    scope: Optional[str] = None,
) -> CitationReport:
    """
    Verify all [MID: xxx] citations in a summary text.

    Args:
        text: The generated summary text
        store: Memory store to validate MIDs against
        scope: Optional scope filter

    Returns:
        CitationReport with verification results
    """
    report = CitationReport()
    all_mids_seen: Set[str] = set()

    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Only check bullet lines (starting with - or *)
        if not (stripped.startswith("- ") or stripped.startswith("* ")
                or stripped.startswith("  - ") or stripped.startswith("  * ")):
            continue

        report.total_bullets += 1

        # Extract MIDs
        mids = _MID_PATTERN.findall(stripped)
        if not mids:
            # Try multi-pattern
            multi = _MID_MULTI_PATTERN.findall(stripped)
            if multi:
                for group in multi:
                    mids.extend(
                        m.strip() for m in group.split(",")
                    )

        if mids:
            report.bullets_with_citation += 1
            entry = CitationEntry(
                bullet_text=stripped[:120],
                line_number=i,
                mids=mids,
            )

            for mid in mids:
                report.total_mids += 1
                all_mids_seen.add(mid)
                # Verify MID exists in store
                item = store.read_item(mid)
                if item is None:
                    entry.valid = False
                    entry.invalid_mids.append(mid)
                    report.invalid_mids += 1
                    report.invalid_mid_list.append(mid)
                else:
                    report.valid_mids += 1

            report.entries.append(entry)
        else:
            report.bullets_without_citation += 1
            report.uncited_bullets.append((i, stripped[:80]))

    report.unique_mids = len(all_mids_seen)

    # Compute domain coverage from valid MIDs
    for mid in all_mids_seen:
        item = store.read_item(mid)
        if item:
            domain = _domain_from_item(item)
            report.domain_coverage[domain] = report.domain_coverage.get(domain, 0) + 1

    logger.info(
        f"Citation verification: {report.bullets_with_citation}/{report.total_bullets} "
        f"cited, {report.valid_mids}/{report.total_mids} valid MIDs"
    )
    return report


def build_citation_map(
    text: str,
    store: MemoryStore,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a citation map: bullet_text -> [MID details].

    Returns a dict suitable for JSON serialization as citation_map.json.
    """
    citation_map = {}
    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith("- ") or stripped.startswith("* ")):
            continue

        mids = _MID_PATTERN.findall(stripped)
        if not mids:
            continue

        details = []
        for mid in mids:
            item = store.read_item(mid)
            if item:
                details.append({
                    "mid": mid,
                    "title": item.title,
                    "type": item.type,
                    "tier": item.tier,
                    "provenance_source": item.provenance.source_id,
                    "pointer_ids": item.provenance.chunk_ids,
                    "rule_id": getattr(item, "rule_id", None),
                })
            else:
                details.append({"mid": mid, "valid": False})

        citation_map[stripped[:120]] = details

    return citation_map


def _domain_from_item(item) -> str:
    """Extract domain label from item for coverage stats."""
    source = item.provenance.source_id or ""
    if ":" in source:
        doc = source.split(":")[0].lower()
        for tech in ("rhel", "oracle", "postgresql", "weblogic",
                     "tomcat", "java", "kubernetes", "k8s"):
            if tech in doc:
                return tech
    for tag in item.tags:
        tag_l = tag.lower()
        for tech in ("rhel", "oracle", "postgresql", "weblogic",
                     "tomcat", "java", "kubernetes", "k8s"):
            if tech in tag_l:
                return tech
    return "unknown"
