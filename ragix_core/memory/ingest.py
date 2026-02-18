"""
Memory Ingest — File-to-Memory Chunking Pipeline

Splits files (or stdin) into paragraph-bounded chunks and writes them as
MemoryItems into the store.  Idempotent: re-ingesting an unchanged file
is a no-op (dedup via corpus_hashes).

Public API:
    chunk_paragraphs(text, max_tokens) -> [(chunk, start_line, end_line), ...]
    ingest_file(store, path, ...) -> IngestResult
    ingest_stdin(store, ...) -> IngestResult
    resolve_sources(args) -> [path, ...]  (expand dirs, globs, extensions)
    corpus_stats(file_paths) -> dict   (lines, tokens, per-file breakdown)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-18
"""

from __future__ import annotations

import glob as _glob_mod
import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Summary of an ingest operation."""

    files_processed: int = 0
    files_skipped: int = 0       # already in corpus_hashes with same sha256
    chunks_created: int = 0
    item_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Corpus measurement
# ---------------------------------------------------------------------------


def corpus_stats(file_paths: List[str]) -> Dict[str, Any]:
    """
    Measure a set of files: count lines, characters, and estimated tokens.

    Args:
        file_paths: Absolute or relative paths to measure.

    Returns:
        Dict with keys ``files``, ``total_lines``, ``total_tokens``,
        and ``per_file`` (list of dicts with ``name``, ``lines``, ``tokens``).
    """
    total_lines = 0
    total_chars = 0
    per_file: List[Dict[str, Any]] = []
    for p in file_paths:
        from ragix_core.memory.extract import read_file_text
        text = read_file_text(p)
        lines = text.count("\n") + 1
        chars = len(text)
        tokens = chars // 4
        total_lines += lines
        total_chars += chars
        per_file.append({"name": os.path.basename(p), "lines": lines, "tokens": tokens})
    return {
        "files": len(file_paths),
        "total_lines": total_lines,
        "total_tokens": total_chars // 4,
        "per_file": per_file,
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Extension → tag mapping for --format auto
_EXT_TAG_MAP = {
    ".md": "markdown",
    ".txt": "text",
    ".py": "python",
    ".java": "java",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".csv": "csv",
    ".rst": "rst",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".toml": "toml",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".sh": "shell",
    ".sql": "sql",
    ".css": "css",
    ".docx": "docx",
    ".odt": "odt",
    ".pptx": "pptx",
    ".odp": "odp",
    ".xlsx": "xlsx",
    ".ods": "ods",
    ".pdf": "pdf",
}


def chunk_paragraphs(
    text: str,
    max_tokens: int = 1800,
) -> List[Tuple[str, int, int]]:
    """
    Split *text* at paragraph boundaries (blank lines).

    Each chunk stays under *max_tokens* (estimated as ``len(text) // 4``).
    A single paragraph that exceeds the budget is emitted as-is (never split
    mid-paragraph).

    Returns:
        List of ``(chunk_text, start_line, end_line)`` tuples.
        Line numbers are 0-based and refer to the original text.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", text)

    chunks: List[Tuple[str, int, int]] = []
    current_paras: List[str] = []
    current_tokens = 0
    # Track line offsets through the original text
    line_offset = 0
    start_line = 0

    for i, para in enumerate(paragraphs):
        para_tokens = len(para) // 4  # rough char/4 estimate
        para_lines = para.count("\n") + 1

        # Flush current bucket if adding this paragraph would exceed budget
        if current_tokens + para_tokens > max_tokens and current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append((chunk_text, start_line, line_offset - 1))
            current_paras = []
            current_tokens = 0
            start_line = line_offset

        current_paras.append(para)
        current_tokens += para_tokens
        line_offset += para_lines
        # Account for the blank-line separator between paragraphs
        if i < len(paragraphs) - 1:
            line_offset += 1  # the \n\n separator counts as ~1 line gap

    # Flush remaining
    if current_paras:
        end_line = max(line_offset - 1, start_line)
        chunks.append(("\n\n".join(current_paras), start_line, end_line))

    return chunks


# ---------------------------------------------------------------------------
# File SHA-256
# ---------------------------------------------------------------------------


def _file_sha256(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _text_sha256(text: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Auto-format helpers
# ---------------------------------------------------------------------------


def _infer_title(text: str, fallback: str) -> str:
    """Extract title from the first markdown heading, or use fallback."""
    for line in text.split("\n")[:20]:
        m = re.match(r"^#+\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def _infer_tags_from_path(path: str) -> List[str]:
    """Extract meaningful tags from path segments."""
    p = Path(path)
    tags: List[str] = []
    # Extension tag
    ext_tag = _EXT_TAG_MAP.get(p.suffix.lower())
    if ext_tag:
        tags.append(ext_tag)
    # Parent directory names (last 2 non-trivial segments)
    parts = [part for part in p.parts[:-1] if part not in (".", "..", "/")]
    for part in parts[-2:]:
        tag = part.lower().replace(" ", "-")
        if tag and tag not in tags and len(tag) <= 40:
            tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# Source resolution (dirs, globs, extensions)
# ---------------------------------------------------------------------------

# File extensions accepted when recursing directories
# (import canonical set from extract module to avoid drift)
from ragix_core.memory.extract import ALL_INGESTABLE_EXTS as _INGESTABLE_EXTS


def resolve_sources(raw: List[str]) -> List[str]:
    """Expand a list of source arguments into concrete file paths.

    Each element of *raw* can be:
    - A regular file path         → kept as-is
    - A directory path            → recursed for files with known extensions
    - A glob pattern (has * or ?) → expanded via :func:`glob.glob`

    Duplicates are removed (preserving order).  Non-existent literal paths
    raise :class:`FileNotFoundError` so the caller gets a clear message.
    """
    seen: set[str] = set()
    result: List[str] = []

    for arg in raw:
        # Glob pattern?
        if "*" in arg or "?" in arg:
            expanded = sorted(_glob_mod.glob(arg, recursive=True))
            for p in expanded:
                if os.path.isfile(p):
                    ap = os.path.abspath(p)
                    if ap not in seen:
                        seen.add(ap)
                        result.append(p)
            continue

        # Directory?
        if os.path.isdir(arg):
            for root, _dirs, files in os.walk(arg):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in _INGESTABLE_EXTS:
                        fp = os.path.join(root, fname)
                        ap = os.path.abspath(fp)
                        if ap not in seen:
                            seen.add(ap)
                            result.append(fp)
            continue

        # Regular file (or missing → let caller get FileNotFoundError)
        if not os.path.isfile(arg):
            raise FileNotFoundError(
                f"Source not found: {arg!r}  "
                f"(resolved to {os.path.abspath(arg)!r}). "
                f"Pass a file, directory, or glob pattern."
            )
        ap = os.path.abspath(arg)
        if ap not in seen:
            seen.add(ap)
            result.append(arg)

    return result


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest_file(
    store: MemoryStore,
    path: str,
    *,
    workspace: str = "default",
    scope: str = "audit",
    corpus_id: Optional[str] = None,
    max_tokens: int = 1800,
    tags: Optional[List[str]] = None,
    format_mode: str = "text",
    injectable: bool = False,
) -> IngestResult:
    """
    Ingest a single file into memory.  Idempotent via corpus_hashes.

    Args:
        store: Target MemoryStore.
        path: File path to ingest.
        workspace: Workspace name (used in provenance, not for routing).
        scope: Memory scope for created items.
        corpus_id: Optional corpus identifier.
        max_tokens: Maximum tokens per chunk.
        tags: Extra tags to attach to every chunk.
        format_mode: "text" (plain) or "auto" (infer tags/title from path).
        injectable: Whether chunks are injectable (default False for raw ingest).

    Returns:
        IngestResult with counts and item IDs.
    """
    abs_path = os.path.abspath(path)

    # Read file (dispatches to binary extractors for Office/PDF formats)
    from ragix_core.memory.extract import read_file_text
    text = read_file_text(abs_path)

    # Compute SHA-256 for dedup
    sha256 = _file_sha256(abs_path)

    # Dedup check
    existing = store.read_corpus_hash(abs_path)
    if existing and existing["sha256"] == sha256:
        logger.debug("Skipping %s (unchanged, sha256=%s)", path, sha256[:16])
        return IngestResult(files_processed=0, files_skipped=1)

    # Auto-format: infer tags and title
    extra_tags = list(tags or [])
    title_base = Path(path).stem
    if format_mode == "auto":
        extra_tags.extend(_infer_tags_from_path(path))
        title_base = _infer_title(text, Path(path).stem)

    # Chunk
    chunks = chunk_paragraphs(text, max_tokens=max_tokens)
    if not chunks:
        logger.warning("No content to ingest from %s", path)
        return IngestResult(files_processed=1)

    # Write items
    item_ids: List[str] = []
    for i, (chunk_text, start_line, end_line) in enumerate(chunks):
        title = f"{title_base} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title_base

        provenance = MemoryProvenance(
            source_kind="doc",
            source_id=abs_path,
            chunk_ids=[f"{abs_path}:{i}"],
            content_hashes=[f"sha256:{sha256}"],
        )

        # Prefix content with provenance metadata
        header = f"[path:{path} chunk:{i} lines:{start_line}-{end_line}]"
        content = f"{header}\n{chunk_text}"

        item = MemoryItem(
            tier="stm",
            type="note",
            title=title,
            content=content,
            tags=list(extra_tags),  # copy to avoid shared ref
            provenance=provenance,
            scope=scope,
            corpus_id=corpus_id or "",
            injectable=injectable,
        )

        store.write_item(item, reason="ingest")
        item_ids.append(item.id)

    # Record corpus hash
    store.write_corpus_hash(abs_path, sha256, len(chunks), item_ids)

    logger.info(
        "Ingested %s: %d chunks, ids=%s",
        path, len(chunks), item_ids[0] if item_ids else "none",
    )

    return IngestResult(
        files_processed=1,
        chunks_created=len(chunks),
        item_ids=item_ids,
    )


def ingest_stdin(
    store: MemoryStore,
    *,
    workspace: str = "default",
    scope: str = "audit",
    corpus_id: Optional[str] = None,
    max_tokens: int = 1800,
    tags: Optional[List[str]] = None,
    injectable: bool = False,
) -> IngestResult:
    """
    Ingest stdin content into memory.

    Uses text SHA-256 for dedup (source_id = "<stdin>").
    """
    text = sys.stdin.read()
    if not text.strip():
        return IngestResult()

    sha256 = _text_sha256(text)

    # Dedup check
    existing = store.read_corpus_hash("<stdin>")
    if existing and existing["sha256"] == sha256:
        logger.debug("Skipping stdin (unchanged)")
        return IngestResult(files_skipped=1)

    extra_tags = list(tags or [])
    chunks = chunk_paragraphs(text, max_tokens=max_tokens)
    if not chunks:
        return IngestResult()

    item_ids: List[str] = []
    for i, (chunk_text, start_line, end_line) in enumerate(chunks):
        title = f"stdin [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else "stdin"

        provenance = MemoryProvenance(
            source_kind="doc",
            source_id="<stdin>",
            chunk_ids=[f"<stdin>:{i}"],
            content_hashes=[f"sha256:{sha256}"],
        )

        header = f"[path:<stdin> chunk:{i} lines:{start_line}-{end_line}]"
        content = f"{header}\n{chunk_text}"

        item = MemoryItem(
            tier="stm",
            type="note",
            title=title,
            content=content,
            tags=list(extra_tags),
            provenance=provenance,
            scope=scope,
            corpus_id=corpus_id or "",
            injectable=injectable,
        )

        store.write_item(item, reason="ingest")
        item_ids.append(item.id)

    store.write_corpus_hash("<stdin>", sha256, len(chunks), item_ids)

    return IngestResult(
        files_processed=1,
        chunks_created=len(chunks),
        item_ids=item_ids,
    )
