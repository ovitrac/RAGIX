"""
Domain Extraction — Shared utility for extracting document-level domains.

Parses MemoryItem.provenance.source_id to extract the RIE document domain.
Falls back to tags[0] when source_id is not a recognizable filename.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import re
from typing import Optional


# Pattern: "RIE - <DOMAIN>.pdf:..." or "<DOMAIN>.pdf:..."
_RIE_PATTERN = re.compile(
    r"^(?:RIE\s*[-–—]\s*)?(.+?)\.pdf",
    re.IGNORECASE,
)


def extract_domain(item) -> str:
    """
    Extract document-level domain from a MemoryItem.

    Priority:
        1. Parse source_id filename (e.g. "RIE - RHEL.pdf:..." → "rhel")
        2. Fall back to tags[0] if available
        3. "untagged" as last resort
    """
    source_id = getattr(getattr(item, "provenance", None), "source_id", "") or ""

    # Try filename extraction
    domain = _extract_domain_from_source(source_id)
    if domain:
        return domain

    # Fallback to first tag
    tags = getattr(item, "tags", None) or []
    if tags:
        return tags[0].lower().replace(" ", "-")

    return "untagged"


def _extract_domain_from_source(source_id: str) -> Optional[str]:
    """
    Extract domain name from source_id string.

    Handles formats:
        "RIE - RHEL.pdf:RIE - RHEL.pdf:P024"  → "rhel"
        "RIE - Windows 2016.pdf:..."            → "windows-2016"
        "some-doc.pdf"                          → "some-doc"
    """
    if not source_id:
        return None

    m = _RIE_PATTERN.match(source_id)
    if m:
        raw = m.group(1).strip()
        # Normalize: lowercase, spaces/underscores → hyphens
        return re.sub(r"[\s_]+", "-", raw).lower()

    return None
