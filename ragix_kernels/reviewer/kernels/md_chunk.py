"""
Kernel: md_chunk
Stage: 1 (Collection)

Produce chunk plan aligned to document structure and protected spans.
Chunk IDs are hash-stable: same content produces the same IDs.

Never splits inside protected regions. Large sections with no sub-headings
are sub-split at paragraph boundaries.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import (
    HeadingNode,
    ProtectedSpan,
    ReviewChunk,
    content_hash,
    estimate_tokens,
)

import logging

logger = logging.getLogger(__name__)


def _collect_leaf_sections(tree: List[HeadingNode]) -> List[HeadingNode]:
    """Flatten heading tree to list of leaf sections (sections with no children)."""
    leaves: List[HeadingNode] = []

    def _walk(nodes: List[HeadingNode]):
        for node in nodes:
            if not node.children:
                leaves.append(node)
            else:
                _walk(node.children)

    _walk(tree)
    return leaves


def _collect_all_sections(tree: List[HeadingNode]) -> List[HeadingNode]:
    """Flatten heading tree to list of all sections in document order."""
    result: List[HeadingNode] = []

    def _walk(nodes: List[HeadingNode]):
        for node in nodes:
            result.append(node)
            _walk(node.children)

    _walk(tree)
    return result


def _find_paragraph_breaks(lines: List[str], start: int, end: int) -> List[int]:
    """
    Find paragraph break points (blank lines) within a line range.

    Args:
        lines: All document lines (0-indexed)
        start: 0-based start index (inclusive)
        end: 0-based end index (exclusive)

    Returns:
        List of 0-based line indices where paragraphs start
    """
    breaks = [start]
    for i in range(start + 1, end):
        if i > 0 and lines[i - 1].strip() == "" and lines[i].strip() != "":
            breaks.append(i)
    return breaks


def chunk_section(
    lines: List[str],
    section: HeadingNode,
    protected_spans: List[ProtectedSpan],
    max_tokens: int = 1800,
    min_tokens: int = 50,
) -> List[ReviewChunk]:
    """
    Create chunks for a single section, respecting protected regions.

    If the section fits within max_tokens, it becomes a single chunk.
    Otherwise, split at paragraph boundaries.
    """
    start_0 = section.line_start - 1  # Convert to 0-based
    # For content lines, skip the heading line itself
    content_start = start_0 + 1
    end_0 = section.line_end  # 0-based exclusive

    if content_start >= end_0:
        # Empty section (heading only)
        return []

    section_text = "\n".join(lines[content_start:end_0])
    section_tokens = estimate_tokens(section_text)

    if section_tokens <= max_tokens:
        # Single chunk for this section
        h = content_hash(section_text)
        chunk_id = f"{section.anchor}_{h[7:15]}"  # Skip "sha256:" prefix
        return [ReviewChunk(
            chunk_id=chunk_id,
            section_id=section.id,
            line_start=content_start + 1,  # Back to 1-based
            line_end=end_0,
            token_estimate=section_tokens,
            content_hash=h,
        )]

    # Need to sub-split at paragraph boundaries
    para_breaks = _find_paragraph_breaks(lines, content_start, end_0)
    if len(para_breaks) <= 1:
        # No paragraph breaks found; treat as single chunk anyway
        h = content_hash(section_text)
        chunk_id = f"{section.anchor}_{h[7:15]}"
        return [ReviewChunk(
            chunk_id=chunk_id,
            section_id=section.id,
            line_start=content_start + 1,
            line_end=end_0,
            token_estimate=section_tokens,
            content_hash=h,
        )]

    # Group paragraph breaks into chunks that fit within budget
    chunks: List[ReviewChunk] = []
    current_start = para_breaks[0]

    for i in range(1, len(para_breaks) + 1):
        current_end = para_breaks[i] if i < len(para_breaks) else end_0
        chunk_text = "\n".join(lines[current_start:current_end])
        chunk_tokens = estimate_tokens(chunk_text)

        if chunk_tokens >= max_tokens or i == len(para_breaks):
            # Emit chunk
            if chunk_tokens >= min_tokens:
                h = content_hash(chunk_text)
                chunk_id = f"{section.anchor}_{h[7:15]}"

                # Check we're not splitting inside a protected region
                line_s = current_start + 1
                line_e = current_end
                for span in protected_spans:
                    if span.overlaps(line_s, line_e):
                        # Extend chunk to include the full protected span
                        line_e = max(line_e, span.line_end)

                chunks.append(ReviewChunk(
                    chunk_id=chunk_id,
                    section_id=section.id,
                    line_start=line_s,
                    line_end=line_e,
                    token_estimate=chunk_tokens,
                    content_hash=h,
                ))
            current_start = current_end

    return chunks


class MdChunkKernel(Kernel):
    """Structure-aligned chunk plan with hash-stable IDs."""

    name = "md_chunk"
    version = "1.0.0"
    category = "reviewer"
    stage = 1
    description = "Structure-aligned chunk plan with hash-stable IDs"

    requires: List[str] = ["md_structure", "md_protected_regions"]
    provides: List[str] = ["chunks"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load dependencies
        struct_path = input.dependencies.get("md_structure")
        prot_path = input.dependencies.get("md_protected_regions")
        if not struct_path or not struct_path.exists():
            raise RuntimeError("Missing dependency: md_structure")
        if not prot_path or not prot_path.exists():
            raise RuntimeError("Missing dependency: md_protected_regions")

        struct_data = json.loads(struct_path.read_text())["data"]
        prot_data = json.loads(prot_path.read_text())["data"]

        # Reconstruct heading tree
        tree = [HeadingNode.from_dict(d) for d in struct_data["heading_tree"]]
        spans = [ProtectedSpan.from_dict(d) for d in prot_data["protected_spans"]]

        # Load document lines
        # Find snapshot path from workspace
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        chunk_cfg = reviewer_cfg.get("chunk", {})
        max_tokens = chunk_cfg.get("max_chunk_tokens", 1800)
        min_tokens = chunk_cfg.get("min_chunk_tokens", 50)

        # Generate chunks for each section
        all_sections = _collect_all_sections(tree)
        # Only chunk leaf sections to avoid overlap
        leaf_sections = _collect_leaf_sections(tree)

        # If no headings, treat entire document as one section
        if not leaf_sections:
            full_text = "\n".join(lines)
            h = content_hash(full_text)
            all_chunks = [ReviewChunk(
                chunk_id=f"root_{h[7:15]}",
                section_id="root",
                line_start=1,
                line_end=len(lines),
                token_estimate=estimate_tokens(full_text),
                content_hash=h,
            )]
        else:
            all_chunks = []
            for section in leaf_sections:
                section_chunks = chunk_section(
                    lines, section, spans, max_tokens, min_tokens
                )
                all_chunks.extend(section_chunks)

        # Sort by line_start
        all_chunks.sort(key=lambda c: c.line_start)

        # Save chunks.json
        stage_dir = input.workspace / "stage1"
        (stage_dir / "chunks.json").write_text(
            json.dumps({
                "chunks": [c.to_dict() for c in all_chunks],
                "total_chunks": len(all_chunks),
                "total_tokens": sum(c.token_estimate for c in all_chunks),
            }, indent=2),
            encoding="utf-8",
        )

        total_tokens = sum(c.token_estimate for c in all_chunks)
        logger.info(
            f"[md_chunk] {len(all_chunks)} chunks, "
            f"~{total_tokens} tokens total"
        )

        return {
            "chunks": [c.to_dict() for c in all_chunks],
            "total_chunks": len(all_chunks),
            "total_tokens": total_tokens,
            "max_chunk_tokens_config": max_tokens,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        return (
            f"Chunking: {data['total_chunks']} chunks, "
            f"~{data['total_tokens']} tokens total "
            f"(max {data['max_chunk_tokens_config']} per chunk)"
        )
