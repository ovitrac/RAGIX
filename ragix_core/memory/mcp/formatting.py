"""
Memory Injection Formatting — Stable Contract (format_version=1)

Renders memory items into the canonical injection block format defined
in PLAN_MEMORY_MCP_SKILL.md §3.4.  This module is the single source of
truth for injection formatting; all MCP tools delegate here.

Breaking changes to the output format MUST increment FORMAT_VERSION.
Additive fields (new optional fields) do not require a version bump.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

FORMAT_VERSION = 1


def format_injection_block(
    items: List[Dict[str, Any]],
    budget_tokens: int = 1500,
    total_matched: Optional[int] = None,
    injection_type: str = "memory_recall",
) -> str:
    """
    Render a list of scored memory items into the stable injection format.

    Each item dict should contain at minimum:
        id, tier, validation, type, title, content, provenance, tags, confidence

    Args:
        items: Scored items ordered by relevance (best first).
        budget_tokens: Requested token budget.
        total_matched: Total items matching query (before budget truncation).
        injection_type: "memory_recall" or "session_inject".

    Returns:
        Formatted injection block string.
    """
    if not items:
        return ""

    matched = total_matched if total_matched is not None else len(items)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    lines.append("## Memory (Injected)")
    lines.append(f"format_version: {FORMAT_VERSION}")
    lines.append(f"injection_type: {injection_type}")
    lines.append(f"generated_at: {now_iso}")
    lines.append(f"budget_tokens: {budget_tokens}")
    lines.append(f"matched: {matched}")

    # Estimate tokens used (rough: 1 token ≈ 4 chars)
    char_budget = budget_tokens * 4
    total_chars = 0
    included: list[tuple[int, Dict[str, Any]]] = []

    for rank, item in enumerate(items, 1):
        entry = _format_single_item(rank, item)
        entry_chars = len(entry)
        if total_chars + entry_chars > char_budget and included:
            break
        included.append((rank, item))
        total_chars += entry_chars

    tokens_used = total_chars // 4
    lines.append(f"used: {tokens_used}")
    lines.append("")

    for rank, item in included:
        lines.append(_format_single_item(rank, item))

    lines.append(
        f"--- End Memory (format_version={FORMAT_VERSION}, "
        f"{len(included)} items, {tokens_used} tokens) ---"
    )

    return "\n".join(lines)


def _format_single_item(rank: int, item: Dict[str, Any]) -> str:
    """Format a single memory item for injection."""
    tier = item.get("tier", "stm").upper()
    validation = item.get("validation", "unverified")
    item_type = item.get("type", "note")
    title = item.get("title", "(untitled)")
    content = item.get("content", "")
    confidence = item.get("confidence", 0.5)

    # Provenance
    prov = item.get("provenance", {})
    if isinstance(prov, dict):
        source_id = prov.get("source_id", "")
        source_kind = prov.get("source_kind", "chat")
        content_hashes = prov.get("content_hashes", [])
        prov_str = f"{source_kind}:{source_id}" if source_id else source_kind
        if content_hashes:
            prov_str += f" | {content_hashes[0][:16]}…"
    else:
        prov_str = str(prov)

    # Tags
    tags = item.get("tags", [])
    tag_str = ", ".join(tags) if tags else "none"

    lines = [
        f"[{rank}] [{tier}:{validation}] {item_type} — {title}",
    ]
    # Content (indented, multi-line)
    for cline in content.strip().splitlines():
        lines.append(f"    {cline}")
    lines.append(f"    provenance: {prov_str}")
    lines.append(f"    tags: {tag_str}")
    lines.append(f"    confidence: {confidence:.2f}")

    # Optional: entities
    entities = item.get("entities", [])
    if entities:
        lines.append(f"    entities: {', '.join(entities)}")

    lines.append("")
    return "\n".join(lines)


def parse_injection_block(text: str) -> Dict[str, Any]:
    """
    Parse an injection block produced by :func:`format_injection_block`.

    Extracts the primary source file, chunk index, token usage, match count,
    and a short insight (first substantial content line after the provenance
    header).  This is the inverse of ``format_injection_block`` — useful for
    summarising what recall returned without showing the full block.

    Args:
        text: Raw text of an injection block (``## Memory (Injected) …``).

    Returns:
        Dict with ``source`` (basename), ``chunk``, ``tokens_used``,
        ``matched``, and ``insight`` (first meaningful content line,
        truncated to 120 chars).
    """
    # Source path + chunk id
    path_match = re.search(r"\[path:(\S+)\s+chunk:(\d+)", text)
    source = path_match.group(1) if path_match else "unknown"
    chunk = path_match.group(2) if path_match else "?"
    basename = os.path.basename(source)

    # Token usage
    used_match = re.search(r"used:\s*(\d+)", text)
    tokens_used = int(used_match.group(1)) if used_match else 0

    # Matched count
    matched_match = re.search(r"matched:\s*(\d+)", text)
    matched = int(matched_match.group(1)) if matched_match else 0

    # Extract first substantial content line (skip headers, metadata, blanks)
    content_lines: list[str] = []
    in_content = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[path:"):
            in_content = True
            continue
        if not in_content:
            continue
        if stripped.startswith(("provenance:", "tags:", "confidence:")):
            continue
        if stripped.startswith("--- End Memory"):
            break
        if not stripped or stripped.startswith(("---", "```", "#")) or len(stripped) < 20:
            continue
        content_lines.append(stripped)
        if len(content_lines) >= 3:
            break

    insight = content_lines[0] if content_lines else "(no extractable content)"
    if len(insight) > 120:
        insight = insight[:117] + "..."

    return {
        "source": basename,
        "chunk": chunk,
        "tokens_used": tokens_used,
        "matched": matched,
        "insight": insight,
    }


def format_search_results(
    items: List[Dict[str, Any]],
    query: str = "",
) -> List[Dict[str, Any]]:
    """
    Format search results for MCP response (structured, not injection block).

    Returns list of dicts suitable for JSON serialization.
    """
    results = []
    for item in items:
        results.append({
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "tier": item.get("tier", "stm"),
            "type": item.get("type", "note"),
            "tags": item.get("tags", []),
            "confidence": item.get("confidence", 0.5),
            "validation": item.get("validation", "unverified"),
            "content_preview": (item.get("content", ""))[:200],
        })
    return results
