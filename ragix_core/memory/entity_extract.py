"""
Deterministic entity extraction — regex + controlled vocabulary.

Extracts entities from memory item title + content text.
Reuses the same vocabulary as graph_store._is_valid_entity()
but in FINDING mode (word boundaries) not MATCHING mode (anchored).

No LLM. Pure deterministic.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# FINDING-mode patterns (word boundaries, not anchored)
# ---------------------------------------------------------------------------

_CVE_FINDER = re.compile(r"\bCVE-\d{4}-\d+\b", re.I)
_COMPLIANCE_FINDER = re.compile(
    r"\b(?:MUST|SHALL|SHOULD|PROHIBITED|REQUIRED)\b"
)
_PATH_FINDER = re.compile(r"(?:/etc/|/var/|/opt/)[\w/.+-]+")
_PORT_FINDER = re.compile(r"\b\d{1,5}(?:/tcp|/udp)\b")

# Version tokens — guarded to avoid false positives (page numbers, etc.)
# Require: (a) 3+ component (x.y.z), OR (b) preceded by version/v/V, OR
# (c) preceded by a known product name (handled in product scan)
_VERSION_STRICT_FINDER = re.compile(
    r"\b\d+(?:\.\d+){2,3}[a-z]?\b"
)
_VERSION_PREFIXED_FINDER = re.compile(
    r"(?:(?:version|ver\.?|[vV])[\s.]?)(\d+(?:\.\d+){1,3}[a-z]?)\b"
)
# Product-adjacent version: "RHEL 8.4", "OpenSSH 9.1" etc.
# This is handled dynamically during product scanning.
_VERSION_2COMP = re.compile(r"\b(\d+\.\d+[a-z]?)\b")


def extract_entities(
    text: str,
    product_vocabulary: Optional[Set[str]] = None,
) -> List[str]:
    """
    Extract entities from text using FINDING-mode patterns + product vocabulary.

    Returns deduplicated, sorted list of entity strings.
    Each returned entity is guaranteed to pass graph_store._is_valid_entity().

    Args:
        text: Input text (typically item.title + " " + item.content).
        product_vocabulary: Override set of known products (lower-cased).
            Defaults to graph_store._PRODUCT_VOCABULARY.

    Returns:
        Sorted, deduplicated list of entity strings.
    """
    if product_vocabulary is None:
        from ragix_core.memory.graph_store import _PRODUCT_VOCABULARY
        product_vocabulary = _PRODUCT_VOCABULARY

    found: Set[str] = set()
    text_lower = text.lower()

    # 1. CVE IDs
    for m in _CVE_FINDER.finditer(text):
        found.add(m.group(0).upper())  # normalize to uppercase

    # 2. Compliance markers (uppercase only — avoid matching prose "should")
    for m in _COMPLIANCE_FINDER.finditer(text):
        found.add(m.group(0).upper())

    # 3. Config paths
    for m in _PATH_FINDER.finditer(text):
        found.add(m.group(0))

    # 4. Port numbers
    for m in _PORT_FINDER.finditer(text):
        found.add(m.group(0).lower())

    # 5. Version tokens (strict: 3+ components)
    for m in _VERSION_STRICT_FINDER.finditer(text):
        found.add(m.group(0))

    # 6. Version tokens (prefixed: "version 1.2", "v3.4")
    for m in _VERSION_PREFIXED_FINDER.finditer(text):
        found.add(m.group(1))

    # 7. Product vocabulary scan (case-insensitive word boundary match)
    for product in product_vocabulary:
        # Build word-boundary pattern for each product
        # Multi-word products (e.g. "sql server", "active directory") use \s+
        pat_str = r"\b" + re.escape(product).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat_str, text_lower):
            found.add(product)  # stored lower-cased (matches vocabulary)

            # Check for product-adjacent 2-component version (e.g. "RHEL 8.4")
            # Find the product match position, then look for version right after
            for pm in re.finditer(pat_str, text_lower):
                after = text[pm.end():pm.end() + 20]
                vm = _VERSION_2COMP.match(after.lstrip())
                if vm:
                    found.add(vm.group(1))

    return sorted(found)
