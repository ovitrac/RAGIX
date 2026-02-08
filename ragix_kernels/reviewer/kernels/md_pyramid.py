"""
Kernel: md_pyramid
Stage: 2 (Analysis)

Bottom-up single-document hierarchical summary construction.
4 levels: paragraph-group -> subsection -> section -> document abstract.

Hash-stable nodes for incremental updates: unchanged sections skip LLM calls.
Level 3 summaries are parallelizable (independent leaf nodes).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import (
    HeadingNode,
    PyramidNode,
    ReviewChunk,
    content_hash,
    estimate_tokens,
)
from ragix_kernels.reviewer.llm_backend import LLMBackend, OllamaBackend, get_backend

import logging

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_TEMPLATE = """Summarize the following section of a Markdown document.

Requirements:
- Extract key claims and arguments
- Identify important terminology
- Note cross-references to other sections
- Keep the summary under {max_tokens} tokens
- Be factual and precise

Section heading: {heading}

Text:
{text}

Provide a concise summary:"""


def _summarize_text(
    backend: LLMBackend,
    text: str,
    heading: str = "",
    max_tokens: int = 200,
) -> str:
    """Summarize text using LLM backend."""
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        max_tokens=max_tokens,
        heading=heading,
        text=text[:6000],  # Limit input size
    )
    try:
        response = backend.call(
            prompt=prompt,
            temperature=0.1,
            num_predict=max_tokens * 2,
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        # Fallback: extract first sentences
        sentences = text.split(". ")
        return ". ".join(sentences[:3]) + "."


def _build_leaf_summaries(
    lines: List[str],
    leaves: List[HeadingNode],
    chunks: List[ReviewChunk],
    backend: LLMBackend,
    max_tokens: int,
    existing_nodes: Dict[str, PyramidNode],
    parallel: bool = True,
) -> Dict[str, PyramidNode]:
    """Build Level 3 (leaf) pyramid nodes. Parallelizable."""
    nodes: Dict[str, PyramidNode] = {}

    def _process_leaf(leaf: HeadingNode) -> Optional[PyramidNode]:
        start = leaf.line_start - 1
        end = leaf.line_end
        text = "\n".join(lines[start:end])
        h = content_hash(text)

        # Check cache: skip if unchanged
        if leaf.id in existing_nodes and existing_nodes[leaf.id].content_hash == h:
            return existing_nodes[leaf.id]

        summary = _summarize_text(backend, text, leaf.title, max_tokens)
        return PyramidNode(
            node_id=leaf.id,
            level=3,
            heading=leaf.title,
            anchor=leaf.anchor,
            content_hash=h,
            summary=summary,
            token_count=estimate_tokens(text),
        )

    if parallel and len(leaves) > 1:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_process_leaf, leaf): leaf for leaf in leaves}
            for future in as_completed(futures):
                node = future.result()
                if node:
                    nodes[node.node_id] = node
    else:
        for leaf in leaves:
            node = _process_leaf(leaf)
            if node:
                nodes[node.node_id] = node

    return nodes


def _build_parent_summaries(
    tree: List[HeadingNode],
    leaf_nodes: Dict[str, PyramidNode],
    backend: LLMBackend,
    max_tokens: int,
    existing_nodes: Dict[str, PyramidNode],
    level: int = 2,
) -> Dict[str, PyramidNode]:
    """Build parent (L2/L1) summaries from children."""
    nodes: Dict[str, PyramidNode] = {}

    def _process_node(heading: HeadingNode):
        if not heading.children:
            return

        # Collect children summaries
        children_text = []
        children_ids = []
        for child in heading.children:
            child_node = leaf_nodes.get(child.id) or nodes.get(child.id)
            if child_node:
                children_text.append(f"[{child.id}] {child_node.summary}")
                children_ids.append(child.id)

        if not children_text:
            return

        combined = "\n\n".join(children_text)
        h = content_hash(combined)

        # Check cache
        if heading.id in existing_nodes and existing_nodes[heading.id].content_hash == h:
            nodes[heading.id] = existing_nodes[heading.id]
            return

        summary = _summarize_text(backend, combined, heading.title, max_tokens)
        nodes[heading.id] = PyramidNode(
            node_id=heading.id,
            level=level,
            heading=heading.title,
            anchor=heading.anchor,
            content_hash=h,
            summary=summary,
            children=children_ids,
            token_count=estimate_tokens(combined),
        )

        # Recurse for deeper nesting
        for child in heading.children:
            if child.children:
                _process_node(child)

    for top in tree:
        _process_node(top)

    return nodes


class MdPyramidKernel(Kernel):
    """Bottom-up hierarchical summaries for single Markdown document."""

    name = "md_pyramid"
    version = "1.0.0"
    category = "reviewer"
    stage = 2
    description = "Single-document hierarchical summaries"

    requires: List[str] = ["md_chunk"]
    provides: List[str] = ["pyramid", "glossary"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load document
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Load structure
        struct_path = input.workspace / "stage1" / "md_structure.json"
        if not struct_path.exists():
            raise RuntimeError("Missing md_structure output")
        struct_data = json.loads(struct_path.read_text())["data"]
        tree = [HeadingNode.from_dict(d) for d in struct_data["heading_tree"]]

        # Load chunks
        chunk_path = input.dependencies.get("md_chunk")
        chunks = []
        if chunk_path and chunk_path.exists():
            chunk_data = json.loads(chunk_path.read_text())["data"]
            chunks = [ReviewChunk.from_dict(d) for d in chunk_data["chunks"]]

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        llm_cfg = reviewer_cfg.get("llm", {})
        max_tokens = reviewer_cfg.get("summary_max_tokens", 200)

        # Initialize LLM backend for summarization
        backend = get_backend(
            backend=llm_cfg.get("backend", "ollama"),
            model=llm_cfg.get("pyramid_model", "granite3.1-moe:3b"),
            endpoint=llm_cfg.get("endpoint", "http://127.0.0.1:11434"),
            strict_sovereign=llm_cfg.get("strict_sovereign", True),
        )

        # Load existing pyramid for incremental update
        existing_nodes: Dict[str, PyramidNode] = {}
        existing_path = input.workspace / "stage2" / "pyramid.json"
        if existing_path.exists():
            try:
                old_data = json.loads(existing_path.read_text())
                for nd in old_data.get("nodes", []):
                    node = PyramidNode.from_dict(nd)
                    existing_nodes[node.node_id] = node
            except Exception:
                pass

        # Collect leaf sections
        leaves: List[HeadingNode] = []
        def _collect_leaves(nodes):
            for n in nodes:
                if not n.children:
                    leaves.append(n)
                else:
                    _collect_leaves(n.children)
        _collect_leaves(tree)

        # Phase 1: Level 3 (leaf nodes)
        llm_calls = 0
        leaf_nodes = _build_leaf_summaries(
            lines, leaves, chunks, backend, max_tokens, existing_nodes
        )
        llm_calls += sum(1 for nid in leaf_nodes if nid not in existing_nodes)

        # Phase 2-3: Level 2 and Level 1 (parent summaries)
        all_nodes = dict(leaf_nodes)
        parent_nodes = _build_parent_summaries(
            tree, all_nodes, backend, max_tokens, existing_nodes, level=1
        )
        all_nodes.update(parent_nodes)
        llm_calls += sum(1 for nid in parent_nodes if nid not in existing_nodes)

        # Phase 4: Level 0 (document abstract)
        top_summaries = []
        for top in tree:
            node = all_nodes.get(top.id)
            if node:
                top_summaries.append(node.summary)

        if top_summaries:
            abstract_text = "\n\n".join(top_summaries)
            abstract_hash = content_hash(abstract_text)
            if "root" not in existing_nodes or existing_nodes["root"].content_hash != abstract_hash:
                abstract = _summarize_text(
                    backend, abstract_text, "Document Overview", max_tokens
                )
                llm_calls += 1
            else:
                abstract = existing_nodes["root"].summary

            root_node = PyramidNode(
                node_id="root",
                level=0,
                heading="Document Abstract",
                content_hash=abstract_hash,
                summary=abstract,
                children=[top.id for top in tree],
                token_count=estimate_tokens(abstract_text),
            )
            all_nodes["root"] = root_node

        # Save pyramid
        stage_dir = input.workspace / "stage2"
        stage_dir.mkdir(parents=True, exist_ok=True)

        pyramid_data = {
            "nodes": [n.to_dict() for n in all_nodes.values()],
            "total_nodes": len(all_nodes),
            "llm_calls": llm_calls,
            "cached_nodes": len(all_nodes) - llm_calls,
        }
        (stage_dir / "pyramid.json").write_text(
            json.dumps(pyramid_data, indent=2), encoding="utf-8"
        )

        # Human-readable markdown
        md_lines = ["# Document Pyramid\n"]
        for node in sorted(all_nodes.values(), key=lambda n: (n.level, n.node_id)):
            indent = "  " * node.level
            md_lines.append(f"{indent}- **{node.node_id}** (L{node.level}): {node.summary[:120]}")
        (stage_dir / "pyramid.md").write_text("\n".join(md_lines), encoding="utf-8")

        logger.info(
            f"[md_pyramid] {len(all_nodes)} nodes, {llm_calls} LLM calls, "
            f"{len(all_nodes) - llm_calls} cached"
        )

        return {
            "nodes": [n.to_dict() for n in all_nodes.values()],
            "total_nodes": len(all_nodes),
            "llm_calls": llm_calls,
            "cached_nodes": len(all_nodes) - llm_calls,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        return (
            f"Pyramid: {data['total_nodes']} nodes, "
            f"{data['llm_calls']} LLM calls, "
            f"{data['cached_nodes']} from cache."
        )
