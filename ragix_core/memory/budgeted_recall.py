"""
Budgeted Recall — Per-Domain Quota Retrieval

Retrieves memory items with per-domain quota to prevent single-domain
domination. Addresses the CORP-ENERGY RIE benchmark failure where RHEL items
(97/272) dwarfed Java (5), Tomcat (10), etc.

Strategy:
  1. Group items by document source (doc_source tag or provenance source_id)
  2. Allocate per-domain budget proportionally (with min/max caps)
  3. Within each domain, rank by importance (type weight + confidence)
  4. Trim to global token budget

V2.3: Dynamic domain discovery from filenames (replaces hardcoded dict).
      Coverage entropy metric for retrieval quality measurement.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem
from ragix_core.shared.text_utils import estimate_tokens

logger = logging.getLogger(__name__)

# Type importance weights for ranking within a domain
_TYPE_WEIGHTS = {
    "constraint": 1.0,
    "decision": 0.9,
    "definition": 0.7,
    "pattern": 0.6,
    "fact": 0.5,
    "todo": 0.3,
    "note": 0.2,
    "pointer": 0.1,
}


# ── Domain aliases ─────────────────────────────────────────────────────────
# Canonical aliases: normalize variant names to canonical domain.
# V2.3: extended with regex-extracted variants from common document naming.

_DOMAIN_ALIASES: Dict[str, str] = {
    "k8s": "kubernetes",
    "rie-k8s": "kubernetes",
    "win": "windows",
    "windows-2016": "windows",
    "windows-serveur": "windows",
    "pg": "postgresql",
    "postgres": "postgresql",
    "wls": "weblogic",
    "ssh": "openssh",
    "faq-rhel": "rhel",
    # V2.3: corpus-driven aliases (from filename regex extraction)
    "ansible-tower": "ansible",
    "dns-applicatif": "dns",
    "vsphere7": "vsphere",
}

# Keyword patterns for tag-based fallback (secondary to provenance)
_TECH_KEYWORDS = (
    "rhel", "oracle", "postgresql", "weblogic", "tomcat", "java",
    "kubernetes", "k8s", "angular", "ansible", "crowdstrike", "openssh",
    "php", "proftpd", "vsphere", "sql-server", "windows", "argo",
    "active-directory", "dns", "nas", "san", "sauvegarde", "ordonnancement",
)


# ── V2.3: Legacy static map (kept as emergency fallback only) ─────────────
# _DOC_NAME_TO_DOMAIN is preserved but only used when dynamic extraction
# produces "unknown". New corpora do NOT need to update this table.

_DOC_NAME_TO_DOMAIN: Dict[str, str] = {
    "FAQ - RHEL.pdf": "rhel",
    "RIE - Active Directory.pdf": "active-directory",
    "RIE-ANGULAR.pdf": "angular",
    "RIE - Ansible Tower.pdf": "ansible",
    "RIE - Argo Workflows.pdf": "argo-workflows",
    "RIE - Crowdstrike.pdf": "crowdstrike",
    "RIE - DNS Applicatif.pdf": "dns",
    "RIE - Java 21.pdf": "java",
    "RIE - K8S.pdf": "kubernetes",
    "RIE - NAS.pdf": "nas",
    "RIE - OpenSSH.pdf": "openssh",
    "RIE - Oracle 19c.pdf": "oracle",
    "RIE - Ordonnancement.pdf": "ordonnancement",
    "RIE - PHP 8.pdf": "php",
    "RIE - PORT APPLICATIF.pdf": "port-applicatif",
    "RIE - PostgreSQL 13.pdf": "postgresql",
    "RIE - PROFTPD.pdf": "proftpd",
    "RIE - RHEL.pdf": "rhel",
    "RIE - SAN.pdf": "san",
    "RIE - Sauvegarde.pdf": "sauvegarde",
    "RIE - SQL Server 2019.pdf": "sql-server",
    "RIE - STK S3 EDGAR.pdf": "stk-s3-edgar",
    "RIE - TOMCAT.pdf": "tomcat",
    "RIE - VSPHERE7.pdf": "vsphere",
    "RIE - Weblogic.pdf": "weblogic",
    "RIE - Windows 2016.pdf": "windows",
    "RIE - WINDOWS SERVEUR.pdf": "windows",
}


def _canonicalize_domain(domain: str) -> str:
    """Apply alias mapping to normalize domain name."""
    return _DOMAIN_ALIASES.get(domain, domain)


# ── V2.3: Dynamic domain extraction from filename ─────────────────────────

# Regex: strip common document prefixes (RIE, FAQ, etc.)
_PREFIX_RE = re.compile(r"^(?:RIE|FAQ|POL|NRM|STD)\s*[-–]\s*", re.I)
# Regex: strip trailing version-like suffix preceded by space (e.g. " 19c", " 2019", " 7")
# Requires leading whitespace to avoid stripping digits within names like K8S
_TRAILING_VERSION_RE = re.compile(r"\s+\d[\w.]*$")


def _extract_domain_from_filename(filename: str) -> str:
    """
    V2.3: Extract domain label from document filename using patterns.

    Works for any corpus following naming conventions like:
        "RIE - Oracle 19c.pdf" → "oracle"
        "FAQ - RHEL.pdf" → "rhel"
        "RIE - K8S.pdf" → "kubernetes" (via alias)

    Returns "unknown" if filename cannot be parsed.
    """
    if not filename:
        return "unknown"

    # Strip extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Strip common prefixes
    name = _PREFIX_RE.sub("", name).strip()

    if not name:
        return "unknown"

    # Strip trailing version numbers ("Oracle 19c" → "Oracle")
    name = _TRAILING_VERSION_RE.sub("", name).strip()

    if not name:
        return "unknown"

    # Normalize: lowercase, replace spaces/underscores with hyphens
    domain = re.sub(r"[\s_]+", "-", name.lower()).strip("-")

    # Apply canonical aliases
    return _canonicalize_domain(domain)


def _extract_domain(item: MemoryItem) -> str:
    """
    Extract canonical domain label from a memory item.

    V2.3 priority chain (dynamic-first):
    1. Provenance source_id → dynamic filename extraction (regex-based)
    2. Provenance source_id → static fallback (_DOC_NAME_TO_DOMAIN)
    3. Tags containing known technology patterns → canonicalize
    4. Fallback: "unknown"
    """
    # ── Priority 1: provenance source_id → dynamic extraction ──
    source = item.provenance.source_id or ""
    if ":" in source:
        doc_name = source.split(":")[0]
    else:
        doc_name = source

    if doc_name:
        # V2.3: try dynamic extraction first (no hardcoded table needed)
        dynamic = _extract_domain_from_filename(doc_name)
        if dynamic != "unknown":
            return dynamic

        # Static fallback (legacy, for edge cases not covered by regex)
        if doc_name in _DOC_NAME_TO_DOMAIN:
            return _DOC_NAME_TO_DOMAIN[doc_name]
        doc_lower = doc_name.lower()
        for known_doc, domain in _DOC_NAME_TO_DOMAIN.items():
            if known_doc.lower() == doc_lower:
                return domain

    # ── Priority 2: tag-based extraction ──
    for tag in item.tags:
        tag_lower = tag.lower().strip()
        # Direct alias match
        if tag_lower in _DOMAIN_ALIASES:
            return _DOMAIN_ALIASES[tag_lower]
        # Substring match against tech keywords
        for tech in _TECH_KEYWORDS:
            if tech in tag_lower:
                return _canonicalize_domain(tech)

    # ── Fallback ──
    return "unknown"


def _item_importance(item: MemoryItem) -> float:
    """
    Score an item's importance for selection within its domain.

    Combines type weight, confidence, and usage.
    """
    type_w = _TYPE_WEIGHTS.get(item.type, 0.3)
    usage_bonus = min(0.2, item.usage_count * 0.05)
    return type_w * 0.5 + item.confidence * 0.3 + usage_bonus + 0.1


# ── V2.3: Coverage Entropy ────────────────────────────────────────────────

def coverage_entropy(domain_counts: Dict[str, int]) -> float:
    """
    Compute Shannon entropy of domain distribution (normalized to [0, 1]).

    Higher entropy = more balanced coverage across domains.
    Target: >= 0.85 (near-uniform).

    Returns 0.0 if fewer than 2 domains.
    """
    total = sum(domain_counts.values())
    if total == 0 or len(domain_counts) < 2:
        return 0.0

    max_entropy = math.log2(len(domain_counts))
    if max_entropy == 0:
        return 0.0

    entropy = 0.0
    for count in domain_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return round(entropy / max_entropy, 4)


def recall_budgeted(
    store: MemoryStore,
    scope: str,
    max_tokens: int = 12000,
    min_items_per_domain: int = 3,
    max_items_per_domain: int = 25,
    exclude_types: Optional[List[str]] = None,
    include_pointers: bool = False,
) -> Dict[str, List[MemoryItem]]:
    """
    Recall memory items with per-domain quota.

    Args:
        store: Memory store to query
        scope: Memory scope to filter
        max_tokens: Global token budget for all items
        min_items_per_domain: Minimum items to include per domain
        max_items_per_domain: Maximum items per domain
        exclude_types: Item types to exclude (default: ["pointer"])
        include_pointers: If True, include pointer items

    Returns:
        Dict mapping domain -> list of MemoryItems, sorted by importance.
        Special key "_meta" contains budget allocation metadata.
    """
    exclude = set(exclude_types or [])
    if not include_pointers:
        exclude.add("pointer")

    # Step 1: Collect all non-archived items for scope
    all_items = store.list_items(
        scope=scope, exclude_archived=True, limit=2000,
    )
    # Filter by type
    items = [it for it in all_items if it.type not in exclude]

    if not items:
        logger.warning(f"No items found for scope={scope}")
        return {"_meta": {"total_items": 0, "domains": 0}}

    # Step 2: Group by domain
    by_domain: Dict[str, List[MemoryItem]] = defaultdict(list)
    for item in items:
        domain = _extract_domain(item)
        by_domain[domain].append(item)

    # Sort each domain by importance (descending)
    for domain in by_domain:
        by_domain[domain].sort(
            key=lambda it: _item_importance(it), reverse=True
        )

    n_domains = len(by_domain)
    logger.info(
        f"Budgeted recall: {len(items)} items across {n_domains} domains "
        f"(budget={max_tokens} tokens)"
    )

    # Step 3: Allocate per-domain quota
    # Proportional allocation with min/max caps
    total_items = len(items)
    per_domain_budget: Dict[str, int] = {}
    for domain, domain_items in by_domain.items():
        proportion = len(domain_items) / total_items
        # Base allocation: proportional share of a rough item budget
        rough_items_budget = max_tokens // 80  # ~80 tokens per item
        allocated = max(
            min_items_per_domain,
            min(max_items_per_domain, int(proportion * rough_items_budget))
        )
        per_domain_budget[domain] = min(allocated, len(domain_items))

    # Step 4: Select items within budget
    result: Dict[str, List[MemoryItem]] = {}
    total_tokens_used = 0

    # First pass: ensure minimum per domain
    for domain, domain_items in sorted(
        by_domain.items(), key=lambda kv: len(kv[1])
    ):
        budget = per_domain_budget[domain]
        selected = []
        for item in domain_items[:budget]:
            item_tokens = estimate_tokens(
                f"{item.title} {item.content}"
            )
            if total_tokens_used + item_tokens > max_tokens:
                break
            selected.append(item)
            total_tokens_used += item_tokens
        if selected:
            result[domain] = selected

    # Second pass: fill remaining budget from largest domains
    remaining = max_tokens - total_tokens_used
    if remaining > 200:
        for domain, domain_items in sorted(
            by_domain.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            already = len(result.get(domain, []))
            for item in domain_items[already:]:
                item_tokens = estimate_tokens(
                    f"{item.title} {item.content}"
                )
                if total_tokens_used + item_tokens > max_tokens:
                    break
                result.setdefault(domain, []).append(item)
                total_tokens_used += item_tokens

    # Build metadata (includes domain extraction report)
    unclassified = len(by_domain.get("unknown", []))
    unclassified_pct = (unclassified / len(items) * 100) if items else 0.0

    # V2.3: Coverage entropy
    selected_domain_counts = {d: len(v) for d, v in result.items()}
    entropy = coverage_entropy(selected_domain_counts)

    meta = {
        "total_items": sum(len(v) for v in result.values()),
        "total_tokens": total_tokens_used,
        "max_tokens": max_tokens,
        "domains": len(result),
        "coverage_entropy": entropy,
        "per_domain": {
            d: {"selected": len(v), "available": len(by_domain.get(d, []))}
            for d, v in result.items()
        },
        # Domain extraction report (P1 requirement)
        "domain_extraction": {
            "total_classified": len(items) - unclassified,
            "total_unclassified": unclassified,
            "unclassified_pct": round(unclassified_pct, 2),
            "all_domains": sorted(
                {d for d in by_domain if d != "unknown"}
            ),
            "domain_counts": {
                d: len(v) for d, v in sorted(by_domain.items())
            },
        },
    }
    result["_meta"] = meta  # type: ignore[assignment]

    # Log domain extraction quality
    if unclassified_pct > 1.0:
        logger.warning(
            f"Domain extraction: {unclassified}/{len(items)} items "
            f"({unclassified_pct:.1f}%) unclassified — exceeds 1% threshold"
        )
    else:
        logger.info(
            f"Domain extraction: {unclassified}/{len(items)} unclassified "
            f"({unclassified_pct:.1f}%) — within threshold"
        )

    logger.info(
        f"Budgeted recall result: {meta['total_items']} items, "
        f"{meta['total_tokens']} tokens, {meta['domains']} domains, "
        f"entropy={entropy:.3f}"
    )
    return result


def format_budgeted_inject(
    budgeted: Dict[str, List[MemoryItem]],
    header: str = "# Memory Items by Domain\n\n",
) -> str:
    """
    Format budgeted recall results for context injection.

    Groups items by domain with headers for structured injection.
    """
    parts = [header]
    for domain, items in sorted(budgeted.items()):
        if domain == "_meta":
            continue
        parts.append(f"## {domain.upper()}\n")
        for item in items:
            parts.append(item.format_inject())
            parts.append("")
    return "\n".join(parts)
