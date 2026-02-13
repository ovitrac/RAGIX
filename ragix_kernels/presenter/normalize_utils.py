"""
Shared deterministic utilities for semantic normalization.

Pure functions used by both deterministic and LLM normalizer modes:
- Topic clustering by heading path
- Role assignment via keyword matching (FR + EN)
- Importance scoring with heading boost and depth decay
- Near-duplicate detection via Jaccard similarity
- Narrative arc detection from role distributions

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_kernels.presenter.models import (
    NarrativeArc,
    NormalizedUnit,
    SemanticUnit,
    TopicCluster,
    UnitRole,
    UnitType,
)
from ragix_kernels.presenter.config import ImportanceConfig


# ---------------------------------------------------------------------------
# Role keyword lexicons
# ---------------------------------------------------------------------------

ROLE_KEYWORDS_FR: Dict[UnitRole, List[str]] = {
    UnitRole.FINDING: [
        "résultat", "résultats", "chiffre", "chiffres", "métrique", "métriques",
        "performance", "taux", "mesure", "observation", "constat", "bilan",
        "indicateur", "score", "valeur",
    ],
    UnitRole.RECOMMENDATION: [
        "recommandation", "préconisation", "action", "il convient",
        "nous recommandons", "il est recommandé", "mesure corrective",
        "plan d'action", "amélioration", "proposition",
    ],
    UnitRole.PROBLEM: [
        "problème", "risque", "dette", "vulnérabilité", "alerte",
        "faiblesse", "défaut", "anomalie", "incident", "menace",
        "criticité", "non-conformité", "dysfonctionnement",
    ],
    UnitRole.METHOD: [
        "méthode", "approche", "processus", "architecture", "conception",
        "méthodologie", "démarche", "stratégie", "procédure", "technique",
        "algorithme", "modèle", "framework",
    ],
    UnitRole.CONTEXT: [
        "contexte", "introduction", "cadre", "périmètre", "historique",
        "préambule", "situation", "environnement", "objectif", "objectifs",
        "portée", "scope", "enjeu", "enjeux",
    ],
    UnitRole.CONCLUSION: [
        "conclusion", "synthèse", "résumé", "bilan", "récapitulatif",
        "en résumé", "pour conclure", "perspectives",
    ],
    UnitRole.REFERENCE: [
        "référence", "références", "bibliographie", "annexe", "annexes",
        "source", "sources", "glossaire", "index",
    ],
    UnitRole.ILLUSTRATION: [
        "figure", "diagramme", "schéma", "tableau", "graphique",
        "illustration", "image", "capture",
    ],
}

ROLE_KEYWORDS_EN: Dict[UnitRole, List[str]] = {
    UnitRole.FINDING: [
        "result", "results", "finding", "findings", "metric", "metrics",
        "performance", "rate", "measurement", "observation", "score",
        "indicator", "value", "outcome",
    ],
    UnitRole.RECOMMENDATION: [
        "recommendation", "action item", "we recommend", "should",
        "corrective action", "improvement", "suggestion", "proposal",
        "remediation", "mitigation",
    ],
    UnitRole.PROBLEM: [
        "problem", "risk", "debt", "vulnerability", "alert",
        "weakness", "defect", "anomaly", "incident", "threat",
        "issue", "concern", "gap", "non-compliance",
    ],
    UnitRole.METHOD: [
        "method", "approach", "process", "architecture", "design",
        "methodology", "strategy", "procedure", "technique",
        "algorithm", "model", "framework", "implementation",
    ],
    UnitRole.CONTEXT: [
        "context", "introduction", "background", "scope", "history",
        "overview", "objective", "objectives", "purpose", "preamble",
        "environment", "setting",
    ],
    UnitRole.CONCLUSION: [
        "conclusion", "summary", "recap", "takeaway", "takeaways",
        "in summary", "to conclude", "perspectives", "outlook",
    ],
    UnitRole.REFERENCE: [
        "reference", "references", "bibliography", "appendix", "appendices",
        "source", "sources", "glossary", "index",
    ],
    UnitRole.ILLUSTRATION: [
        "figure", "diagram", "chart", "table", "graph",
        "illustration", "image", "screenshot", "capture",
    ],
}

# Canonical narrative ordering for sections
_NARRATIVE_ORDER: List[UnitRole] = [
    UnitRole.CONTEXT,
    UnitRole.PROBLEM,
    UnitRole.METHOD,
    UnitRole.FINDING,
    UnitRole.RECOMMENDATION,
    UnitRole.CONCLUSION,
    UnitRole.REFERENCE,
]


# ---------------------------------------------------------------------------
# Topic clustering (deterministic)
# ---------------------------------------------------------------------------

def cluster_by_heading_path(
    units: List[SemanticUnit],
    max_levels: int = 2,
) -> Dict[str, List[str]]:
    """
    Group units by their heading_path prefix (first *max_levels* levels).

    Args:
        units: List of SemanticUnit objects.
        max_levels: How many heading levels to use for grouping (1 or 2).

    Returns:
        {"Architecture > Overview": ["id1", "id2"], ...}
        Units with no heading_path go into "Uncategorized".
    """
    clusters: Dict[str, List[str]] = {}

    for u in units:
        if u.heading_path:
            key_parts = u.heading_path[:max_levels]
            key = " > ".join(key_parts)
        else:
            key = "Uncategorized"

        clusters.setdefault(key, []).append(u.id)

    return clusters


def consolidate_clusters(
    raw_clusters: Dict[str, List[str]],
    units: List[SemanticUnit],
    max_clusters: int = 20,
) -> Dict[str, List[str]]:
    """
    Merge clusters when there are too many for a presentation.

    Strategy (applied in order until cluster count <= max_clusters):
    1. If already within budget, return as-is.
    2. Fall back to level-1 heading paths (coarser grouping).
    3. Merge singleton clusters (1 unit) into their parent or "Other".

    Args:
        raw_clusters: Cluster dict from cluster_by_heading_path().
        units: Original SemanticUnit list (needed for re-clustering).
        max_clusters: Maximum allowed clusters.

    Returns:
        Consolidated cluster dict with len <= max_clusters (best effort).
    """
    if len(raw_clusters) <= max_clusters:
        return raw_clusters

    # Phase 1: re-cluster at level 1 (coarser)
    coarse = cluster_by_heading_path(units, max_levels=1)
    if len(coarse) <= max_clusters:
        return coarse

    # Phase 2: merge smallest clusters into "Other" until within budget
    # Sort clusters by size (ascending) — merge smallest first
    sorted_labels = sorted(coarse.keys(), key=lambda k: len(coarse[k]))
    merged: Dict[str, List[str]] = {}
    overflow: List[str] = []

    for label in sorted_labels:
        if len(merged) < max_clusters - 1:  # reserve 1 slot for "Other"
            merged[label] = coarse[label]
        else:
            overflow.extend(coarse[label])

    if overflow:
        merged["Other"] = overflow

    return merged


# ---------------------------------------------------------------------------
# Role assignment (deterministic keyword matching)
# ---------------------------------------------------------------------------

def assign_role_by_keywords(
    unit: SemanticUnit,
    lang: str = "fr",
) -> UnitRole:
    """
    Assign a semantic role to a unit via keyword matching.

    Checks heading_path first (highest signal), then content.
    Uses bilingual lexicons selected by *lang* ("fr" or "en").

    Special cases:
    - FRONT_MATTER units → METADATA
    - IMAGE_REF / MERMAID units → ILLUSTRATION
    - EQUATION_BLOCK units → ILLUSTRATION

    Args:
        unit: SemanticUnit to classify.
        lang: Language code ("fr" or "en").

    Returns:
        Best-matching UnitRole, or UNKNOWN if no keywords match.
    """
    # Type-based shortcuts
    if unit.type == UnitType.FRONT_MATTER:
        return UnitRole.METADATA
    if unit.type in (UnitType.IMAGE_REF, UnitType.MERMAID):
        return UnitRole.ILLUSTRATION
    if unit.type == UnitType.EQUATION_BLOCK:
        return UnitRole.ILLUSTRATION

    lexicon = ROLE_KEYWORDS_FR if lang == "fr" else ROLE_KEYWORDS_EN

    # Build search text: heading path (weighted) + content
    heading_text = " ".join(unit.heading_path).lower()
    content_text = unit.content[:500].lower()
    combined = heading_text + " " + content_text

    best_role = UnitRole.UNKNOWN
    best_score = 0

    for role, keywords in lexicon.items():
        score = 0
        for kw in keywords:
            kw_lower = kw.lower()
            # Heading matches count double
            if kw_lower in heading_text:
                score += 2
            if kw_lower in content_text:
                score += 1
        if score > best_score:
            best_score = score
            best_role = role

    return best_role


# ---------------------------------------------------------------------------
# Importance scoring (deterministic)
# ---------------------------------------------------------------------------

def compute_importance(
    unit: SemanticUnit,
    role: UnitRole,
    config: Optional[ImportanceConfig] = None,
) -> float:
    """
    Compute 0.0-1.0 importance score for a semantic unit.

    Factors:
    - Heading level boost (H1=+0.3, H2=+0.2, H3=+0.1)
    - Role boost (FINDING/RECOMMENDATION=+0.2, PROBLEM=+0.15)
    - Content signals (numbers, percentages, lists)
    - Depth decay (deeper nesting reduces score)

    Args:
        unit: SemanticUnit to score.
        role: Assigned UnitRole.
        config: Optional ImportanceConfig for tuning.

    Returns:
        Score clamped to [0.0, 1.0].
    """
    if config is None:
        config = ImportanceConfig()

    base = 0.5

    # Heading level boost
    if config.boost_headings and unit.type == UnitType.HEADING:
        level = unit.metadata.get("level", 3)
        if level == 1:
            base += 0.3
        elif level == 2:
            base += 0.2
        elif level == 3:
            base += 0.1

    # Role boost
    role_boosts = {
        UnitRole.FINDING: 0.2,
        UnitRole.RECOMMENDATION: 0.2,
        UnitRole.PROBLEM: 0.15,
        UnitRole.ILLUSTRATION: 0.15,
        UnitRole.METHOD: 0.1,
        UnitRole.CONCLUSION: 0.1,
        UnitRole.CONTEXT: 0.05,
    }
    base += role_boosts.get(role, 0.0)

    # Image references are inherently high-value for slides
    if unit.type == UnitType.IMAGE_REF:
        base += 0.2

    # Content signal boost
    if config.boost_findings:
        text = unit.content
        # Numbers and percentages
        if re.search(r'\d+[.,]?\d*\s*%', text):
            base += 0.1
        elif re.search(r'\b\d{2,}\b', text):
            base += 0.05
        # Lists with items (implies structured data)
        if unit.type in (UnitType.BULLET_LIST, UnitType.NUMBERED_LIST):
            base += 0.05

    # Depth decay
    depth_penalty = unit.depth * config.decay_depth
    base -= depth_penalty

    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# Deduplication (deterministic Jaccard)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Set[str]:
    """Simple word-level tokenization for Jaccard."""
    return set(re.findall(r'\w+', text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """
    Word-level Jaccard similarity between two text strings.

    Args:
        a: First text.
        b: Second text.

    Returns:
        Jaccard index in [0.0, 1.0].
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def find_duplicates(
    units: List[NormalizedUnit],
    threshold: float = 0.70,
) -> Dict[str, str]:
    """
    Find near-duplicate units via pairwise Jaccard similarity.

    O(n^2) but n < 2000 units — acceptable for document-scale corpora.

    Args:
        units: List of NormalizedUnit objects.
        threshold: Minimum Jaccard similarity to flag as duplicate.

    Returns:
        {duplicate_id: canonical_id} for units exceeding threshold.
        The unit with lower index is the canonical.
    """
    duplicates: Dict[str, str] = {}
    seen_canonical: Set[str] = set()

    for i in range(len(units)):
        uid_i = units[i].unit.id
        if uid_i in duplicates:
            continue

        for j in range(i + 1, len(units)):
            uid_j = units[j].unit.id
            if uid_j in duplicates:
                continue
            # Only compare same-type units
            if units[i].unit.type != units[j].unit.type:
                continue
            # Skip headings (too short for meaningful Jaccard)
            if units[i].unit.type == UnitType.HEADING:
                continue

            sim = jaccard_similarity(units[i].unit.content, units[j].unit.content)
            if sim >= threshold:
                duplicates[uid_j] = uid_i

    return duplicates


def find_global_duplicates(
    units: List[NormalizedUnit],
    threshold: float = 0.80,
) -> Dict[str, str]:
    """
    Find near-duplicate units across *different* topic clusters.

    Same Jaccard logic as find_duplicates() but only compares units that
    belong to different clusters. Stricter default threshold (0.80) to
    avoid cross-topic false positives.

    Args:
        units: List of NormalizedUnit objects (only active, non-duplicate).
        threshold: Minimum Jaccard similarity to flag as duplicate.

    Returns:
        {duplicate_id: canonical_id} for cross-cluster duplicates.
    """
    duplicates: Dict[str, str] = {}

    for i in range(len(units)):
        uid_i = units[i].unit.id
        if uid_i in duplicates:
            continue
        cluster_i = units[i].topic_cluster or ""

        for j in range(i + 1, len(units)):
            uid_j = units[j].unit.id
            if uid_j in duplicates:
                continue
            cluster_j = units[j].topic_cluster or ""

            # Only compare across different clusters
            if cluster_i == cluster_j:
                continue
            # Only compare same-type units
            if units[i].unit.type != units[j].unit.type:
                continue
            # Skip headings (too short)
            if units[i].unit.type == UnitType.HEADING:
                continue

            sim = jaccard_similarity(units[i].unit.content, units[j].unit.content)
            if sim >= threshold:
                duplicates[uid_j] = uid_i

    return duplicates


# ---------------------------------------------------------------------------
# Narrative arc (deterministic)
# ---------------------------------------------------------------------------

def detect_narrative_arc(
    clusters: List[TopicCluster],
    units: List[NormalizedUnit],
) -> NarrativeArc:
    """
    Order clusters by their dominant role to form a narrative arc.

    Ordering follows the canonical sequence:
    CONTEXT -> PROBLEM -> METHOD -> FINDING -> RECOMMENDATION -> CONCLUSION -> REFERENCE

    Clusters with mixed roles are placed by their dominant (most frequent) role.
    Clusters with UNKNOWN/METADATA/ILLUSTRATION roles are placed at the end
    before REFERENCE.

    Args:
        clusters: List of TopicCluster objects.
        units: List of NormalizedUnit objects.

    Returns:
        NarrativeArc with ordered section labels.
    """
    if not clusters:
        return NarrativeArc(sections=["Content"])

    # Build unit lookup
    unit_map: Dict[str, NormalizedUnit] = {u.unit.id: u for u in units}

    # For each cluster, determine dominant role
    cluster_roles: List[Tuple[str, UnitRole]] = []
    for cluster in clusters:
        role_counts: Dict[UnitRole, int] = {}
        for uid in cluster.unit_ids:
            nu = unit_map.get(uid)
            if nu:
                role_counts[nu.role] = role_counts.get(nu.role, 0) + 1

        # Dominant role (most frequent, tie-break by _NARRATIVE_ORDER)
        if role_counts:
            dominant = max(
                role_counts.keys(),
                key=lambda r: (role_counts[r], -_role_order(r)),
            )
        else:
            dominant = UnitRole.UNKNOWN

        cluster_roles.append((cluster.label, dominant))

    # Sort by narrative order
    cluster_roles.sort(key=lambda pair: _role_order(pair[1]))

    sections = [label for label, _ in cluster_roles]
    return NarrativeArc(sections=sections)


def _role_order(role: UnitRole) -> int:
    """Return sort index for a role in narrative sequence."""
    try:
        return _NARRATIVE_ORDER.index(role)
    except ValueError:
        # UNKNOWN, METADATA, ILLUSTRATION → between RECOMMENDATION and CONCLUSION
        return len(_NARRATIVE_ORDER) - 2
