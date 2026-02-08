"""
Adaptive context assembly for edit-time LLM injection.

Priority-based budget allocation: always includes target chunk and
instructions, then adds pyramid context in descending priority order
until token budget is exhausted.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.reviewer.models import (
    PyramidNode,
    ReviewChunk,
    estimate_tokens,
)

import logging

logger = logging.getLogger(__name__)

# Approximate token cost for fixed-template instructions
INSTRUCTION_TOKENS = 300

# Tier → max priority level mapping
# Tier 0: chunk + skeleton only (priorities 1-2)
# Tier 1: + section summary (priority 4)
# Tier 2: + abstract + issues (priorities 3, 5)
# Tier 3: + siblings + subsections + glossary (priorities 6-8, full)
_TIER_MAX_PRIORITY = {0: 2, 1: 4, 2: 5, 3: 8}

# Approximate token cost per tier (context tokens, excluding chunk)
_TIER_COST_ESTIMATE = {0: 0, 1: 200, 2: 700, 3: 1500}


def compute_context_tier(
    chunk_tokens: int,
    prompt_budget: int,
    tier_default: int = 3,
) -> int:
    """
    Compute the highest viable context tier given chunk size and budget.

    Starts from tier_default and falls back to lower tiers if the budget
    cannot accommodate chunk + skeleton overhead + tier context cost.

    Args:
        chunk_tokens: Estimated tokens for the chunk text
        prompt_budget: Total prompt token budget
        tier_default: Starting tier (0-3), default 3 (full context)

    Returns:
        Tier level (0-3)
    """
    tier = min(tier_default, 3)
    available = prompt_budget - chunk_tokens - INSTRUCTION_TOKENS
    while tier > 0 and available < _TIER_COST_ESTIMATE.get(tier, 0):
        tier -= 1
    return tier


def assemble_edit_context(
    chunk_text: str,
    chunk: ReviewChunk,
    pyramid_nodes: Dict[str, PyramidNode],
    issues: List[Dict[str, Any]],
    style_rules: str,
    budget_tokens: int = 3500,
    root_node_id: str = "root",
    tier: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Priority-based adaptive context assembly.

    When budget is tight (small model), drops low-priority items.
    When budget is generous (128K+ model), includes full siblings.

    Args:
        chunk_text: The actual text of the chunk to review
        chunk: ReviewChunk metadata
        pyramid_nodes: Dict of node_id -> PyramidNode
        issues: Issues relevant to this chunk
        style_rules: Rendered style rules text
        budget_tokens: Total token budget for context
        root_node_id: ID of the root pyramid node
        tier: Optional context tier (0-3). When set, caps which
              priority levels are emitted. None = existing budget waterfall.

    Returns:
        List of (label, content) tuples in priority order
    """
    max_priority = _TIER_MAX_PRIORITY.get(tier, 8) if tier is not None else 8

    parts: List[Tuple[str, str]] = []
    remaining = budget_tokens

    # Priority 1 (ALWAYS): Target chunk text (~40%)
    parts.append(("chunk", chunk_text))
    remaining -= estimate_tokens(chunk_text)

    # Priority 2 (ALWAYS): Edit instructions (~8%)
    parts.append(("instructions", style_rules))
    remaining -= min(INSTRUCTION_TOKENS, estimate_tokens(style_rules))

    if remaining <= 0 or max_priority <= 2:
        return parts

    # Priority 3 (HIGH): Document abstract (L0) — tier >= 2
    root = pyramid_nodes.get(root_node_id)
    if max_priority >= 3 and root and remaining > 200:
        abstract = root.summary
        if abstract:
            parts.append(("abstract", abstract))
            remaining -= estimate_tokens(abstract)

    # Priority 4 (HIGH): Current section summary (L1) — tier >= 1
    section_id = chunk.section_id
    section_node = pyramid_nodes.get(section_id)
    if max_priority >= 4 and section_node and remaining > 150:
        parts.append(("section_summary", section_node.summary))
        remaining -= estimate_tokens(section_node.summary)

    # Priority 5 (HIGH): Issue findings for this chunk — tier >= 2
    if max_priority >= 5 and issues and remaining > 100:
        chunk_issues = [i for i in issues if _issue_matches_chunk(i, chunk)]
        if chunk_issues:
            issues_text = _format_issues(chunk_issues)
            parts.append(("issues", issues_text))
            remaining -= estimate_tokens(issues_text)

    if max_priority <= 5:
        return parts

    # Priority 6 (MEDIUM): Subsection summary (L2)
    # Walk up from section to find parent with summary
    if section_node and remaining > 100:
        for child_id in section_node.children:
            child = pyramid_nodes.get(child_id)
            if child and child.summary:
                parts.append(("subsection_summary", child.summary))
                remaining -= estimate_tokens(child.summary)
                break  # Only first subsection

    # Priority 7 (MEDIUM): Sibling section summaries (L1, 2 nearest)
    if remaining > 200:
        siblings = _find_siblings(section_id, pyramid_nodes, max_count=2)
        for sib in siblings:
            if remaining > 100 and sib.summary:
                parts.append(("sibling", f"[{sib.node_id}] {sib.summary}"))
                remaining -= estimate_tokens(sib.summary)

    # Priority 8 (LOW): Global glossary
    glossary_node = pyramid_nodes.get("glossary")
    if glossary_node and remaining > 80 and glossary_node.summary:
        parts.append(("glossary", glossary_node.summary))

    return parts


def _issue_matches_chunk(issue: Dict[str, Any], chunk: ReviewChunk) -> bool:
    """Check if an issue is relevant to a specific chunk."""
    line = issue.get("line", 0)
    return chunk.line_start <= line <= chunk.line_end


def _format_issues(issues: List[Dict[str, Any]]) -> str:
    """Format issues into a concise text block."""
    lines = []
    for issue in issues[:5]:  # Limit to top 5
        line_num = issue.get("line", "?")
        issue_type = issue.get("type", "unknown")
        evidence = issue.get("evidence", "")[:80]
        lines.append(f"- L{line_num} [{issue_type}]: {evidence}")
    return "\n".join(lines)


def _find_siblings(
    section_id: str,
    nodes: Dict[str, PyramidNode],
    max_count: int = 2,
) -> List[PyramidNode]:
    """Find sibling nodes at the same level."""
    target = nodes.get(section_id)
    if not target:
        return []

    siblings = [
        n for n in nodes.values()
        if n.level == target.level and n.node_id != section_id and n.summary
    ]

    # Sort by proximity (simple: by node_id)
    siblings.sort(key=lambda n: abs(hash(n.node_id) - hash(section_id)))
    return siblings[:max_count]


def render_context_prompt(parts: List[Tuple[str, str]]) -> str:
    """
    Render assembled context parts into a prompt string.

    Each part is labeled with a header for LLM clarity.
    """
    sections = []
    label_map = {
        "abstract": "DOCUMENT OVERVIEW",
        "section_summary": "CURRENT SECTION",
        "subsection_summary": "CURRENT SUBSECTION",
        "sibling": "NEIGHBORING SECTION",
        "issues": "KNOWN ISSUES IN THIS CHUNK",
        "glossary": "KEY TERMINOLOGY",
        "chunk": "TEXT TO REVIEW",
        "instructions": "REVIEW INSTRUCTIONS",
    }

    for label, content in parts:
        header = label_map.get(label, label.upper())
        sections.append(f"=== {header} ===\n{content}")

    return "\n\n".join(sections)
