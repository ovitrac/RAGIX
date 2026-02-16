"""
summary_collect — Stage 1: Enumerate Files + Build Chunk Plan

Deterministic corpus ingestion. No LLM needed.
Wraps doc_tools.doc_list() + doc_tools.doc_chunk_plan().

V2.4: Delta detection — compares file hashes against stored registry
      to identify new, modified, and unchanged files.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_core.memory.doc_tools import doc_list, doc_chunk_plan, doc_stats


class SummaryCollectKernel(Kernel):
    name = "summary_collect"
    version = "1.1.0"
    category = "summary"
    stage = 1
    description = "Enumerate corpus files and build chunk plan (with delta detection)"
    requires = []
    provides = ["file_registry", "chunk_registry"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Enumerate corpus files and build chunk plan with optional delta detection."""
        cfg = input.config
        root = cfg.get("input_folder", str(input.workspace / "corpus"))
        strategy = cfg.get("strategy", "pages")
        max_chunk_tokens = cfg.get("max_chunk_tokens", 800)
        globs = cfg.get("globs", ["*.pdf", "*.md", "*.txt"])
        delta_mode = cfg.get("delta", False)

        # Step 1: Enumerate files
        files = doc_list(root, globs=globs, sort="name")
        file_registry = []
        for f in files:
            stats = doc_stats(f["path"])
            file_registry.append({
                "name": f["name"],
                "path": f["path"],
                "size_bytes": stats.size_bytes,
                "size_mb": stats.size_mb,
                "mime": stats.mime_hint,
                "pages": stats.pages,
                "sha256": stats.sha256,
            })

        # Step 2: V2.4 delta detection
        delta_info = None
        files_to_chunk = file_registry  # default: all files

        if delta_mode:
            from ragix_core.memory.store import MemoryStore
            db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
            fts_tokenizer = cfg.get("fts_tokenizer")
            store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
            delta = store.find_changed_files(file_registry)
            delta_info = {
                "new_count": len(delta["new"]),
                "modified_count": len(delta["modified"]),
                "unchanged_count": len(delta["unchanged"]),
                "deleted_count": len(delta["deleted"]),
                "new_files": [f["name"] for f in delta["new"]],
                "modified_files": [f["name"] for f in delta["modified"]],
                "deleted_files": [f.get("path", "") for f in delta["deleted"]],
            }
            # Only chunk new + modified files
            changed_paths = {f["path"] for f in delta["new"]} | {f["path"] for f in delta["modified"]}
            files_to_chunk = [f for f in file_registry if f["path"] in changed_paths]

        # Step 3: Build chunk plan per file (only changed files in delta mode)
        chunk_registry = []
        for f in files_to_chunk:
            chunks = doc_chunk_plan(
                f["path"], strategy=strategy,
                max_chunk_tokens=max_chunk_tokens,
            )
            for c in chunks:
                chunk_registry.append({
                    "doc_name": f["name"],
                    "chunk_id": c.chunk_id,
                    "heading": c.heading,
                    "start_page": c.start_page,
                    "end_page": c.end_page,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "token_estimate": c.token_estimate,
                    "content_hash": c.content_hash,
                })

        result = {
            "file_count": len(file_registry),
            "chunk_count": len(chunk_registry),
            "total_tokens": sum(c["token_estimate"] for c in chunk_registry),
            "files": file_registry,
            "chunks": chunk_registry,
            "delta_mode": delta_mode,
        }
        if delta_info is not None:
            result["delta"] = delta_info

        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of collection results including delta info."""
        base = (
            f"Collected {data['file_count']} files, "
            f"{data['chunk_count']} chunks, "
            f"~{data['total_tokens']} tokens"
        )
        if data.get("delta"):
            d = data["delta"]
            base += (
                f" (delta: {d['new_count']} new, "
                f"{d['modified_count']} modified, "
                f"{d['unchanged_count']} unchanged)"
            )
        return base
