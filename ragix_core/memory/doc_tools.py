"""
Document Tools — Deterministic I/O for Large Document Processing

Provides auditable, chunk-based access to large documents (PDF, text)
without loading entire files into LLM context.

Tools:
    doc_list       - Enumerate files with metadata
    doc_stats      - File stats (bytes, pages, mime, last-modified)
    doc_read       - Read text from a range (page range for PDF, line range for text)
    doc_find       - Search for patterns within a document
    doc_chunk_plan - Plan chunking strategy (headings or windows)
    doc_extract_text - Full text extraction from PDF via pdftotext

Chunking strategies:
    "headings"  - Split at heading markers (§, numbered sections, Markdown-ish)
    "windows"   - Fixed-size token/line windows with overlap

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocStats:
    """File-level metadata."""
    path: str
    name: str
    size_bytes: int
    size_mb: float
    mime_hint: str  # pdf, md, txt, etc.
    last_modified: str
    pages: int = 0  # 0 for non-PDF
    line_count: int = 0
    sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize file stats to a plain dictionary."""
        return {
            "path": self.path, "name": self.name,
            "size_bytes": self.size_bytes, "size_mb": round(self.size_mb, 2),
            "mime_hint": self.mime_hint, "last_modified": self.last_modified,
            "pages": self.pages, "line_count": self.line_count,
            "sha256": self.sha256,
        }


@dataclass
class DocChunk:
    """A chunk of document text with provenance."""
    chunk_id: str
    doc_path: str
    doc_name: str
    start_page: int = 0   # for PDF (1-based)
    end_page: int = 0
    start_line: int = 0   # for text (1-based)
    end_line: int = 0
    heading: str = ""      # section heading if available
    text: str = ""
    token_estimate: int = 0
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk metadata to a plain dictionary (excludes text)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_path": self.doc_path, "doc_name": self.doc_name,
            "start_page": self.start_page, "end_page": self.end_page,
            "start_line": self.start_line, "end_line": self.end_line,
            "heading": self.heading,
            "token_estimate": self.token_estimate,
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Heading detection patterns (RIE-style: numbered sections, §, Article, etc.)
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    # Numbered sections: "1.", "1.2", "1.2.3", "1.2.3.4"
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s+[A-ZÉÈÊËÀÂÄÙÛÜÎÏÔÖÇ]"),
    # French article markers
    re.compile(r"^\s*(Article|Chapitre|Section|Annexe|Titre)\s+", re.IGNORECASE),
    # Section symbol
    re.compile(r"^\s*§\s*\d"),
    # ALL-CAPS headings (at least 4 chars, whole line)
    re.compile(r"^\s*[A-ZÉÈÊËÀÂÄÙÛÜÎÏÔÖÇ][A-ZÉÈÊËÀÂÄÙÛÜÎÏÔÖÇ\s\-:]{3,}$"),
    # Markdown headings
    re.compile(r"^\s*#{1,4}\s+"),
]


def _is_heading(line: str) -> bool:
    """Check if a line looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    return any(p.match(stripped) for p in _HEADING_PATTERNS)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4 (conservative for French)."""
    return max(1, len(text) // 4)


def _content_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(
    path: str,
    first_page: int = 1,
    last_page: int = 0,
    layout: bool = True,
) -> str:
    """
    Extract text from PDF using pdftotext (poppler).

    Args:
        path: path to PDF file
        first_page: first page (1-based)
        last_page: last page (0 = all)
        layout: preserve layout (-layout flag)
    """
    cmd = ["pdftotext"]
    if first_page > 1:
        cmd.extend(["-f", str(first_page)])
    if last_page > 0:
        cmd.extend(["-l", str(last_page)])
    if layout:
        cmd.append("-layout")
    cmd.extend([str(path), "-"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"pdftotext failed for {path}: {result.stderr}")
            return ""
        return result.stdout
    except FileNotFoundError:
        logger.error("pdftotext not found; install poppler-utils")
        return ""
    except subprocess.TimeoutExpired:
        logger.warning(f"pdftotext timed out for {path}")
        return ""


def count_pdf_pages(path: str) -> int:
    """Count pages in a PDF using pdfinfo."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# doc_list
# ---------------------------------------------------------------------------

def doc_list(
    root: str,
    globs: Optional[List[str]] = None,
    max_files: int = 200,
    sort: str = "name",
) -> List[Dict[str, Any]]:
    """
    Enumerate document files under root.

    Args:
        root: directory path
        globs: file patterns (default: ["*.pdf", "*.md", "*.txt"])
        max_files: maximum files to return
        sort: "name" | "size" | "date"
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    patterns = globs or ["*.pdf", "*.md", "*.txt"]
    files = []
    for pattern in patterns:
        files.extend(root_path.glob(pattern))

    # Deduplicate and sort
    files = sorted(set(files))
    if sort == "size":
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
    elif sort == "date":
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    results = []
    for f in files[:max_files]:
        stat = f.stat()
        mime = "pdf" if f.suffix.lower() == ".pdf" else f.suffix.lstrip(".")
        results.append({
            "path": str(f),
            "name": f.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "mime_hint": mime,
        })
    return results


# ---------------------------------------------------------------------------
# doc_stats
# ---------------------------------------------------------------------------

def doc_stats(path: str) -> DocStats:
    """Compute detailed stats for a single document."""
    p = Path(path)
    stat = p.stat()
    mime = "pdf" if p.suffix.lower() == ".pdf" else p.suffix.lstrip(".")

    pages = 0
    line_count = 0
    sha = ""

    if mime == "pdf":
        pages = count_pdf_pages(path)
        # Hash first 64KB for fingerprint
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read(65536)).hexdigest()[:16]
    else:
        text = p.read_text(encoding="utf-8", errors="replace")
        line_count = text.count("\n") + 1
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    from datetime import datetime
    mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return DocStats(
        path=str(p), name=p.name,
        size_bytes=stat.st_size,
        size_mb=stat.st_size / (1024 * 1024),
        mime_hint=mime, last_modified=mtime,
        pages=pages, line_count=line_count,
        sha256=f"sha256:{sha}",
    )


# ---------------------------------------------------------------------------
# doc_read
# ---------------------------------------------------------------------------

def doc_read(
    path: str,
    start_page: int = 1,
    end_page: int = 0,
    start_line: int = 0,
    num_lines: int = 0,
) -> str:
    """
    Read text from a document range.

    For PDF: extracts pages [start_page, end_page].
    For text: reads lines [start_line, start_line+num_lines).
    """
    p = Path(path)
    mime = "pdf" if p.suffix.lower() == ".pdf" else "text"

    if mime == "pdf":
        return extract_pdf_text(path, first_page=start_page, last_page=end_page)
    else:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if start_line > 0 and num_lines > 0:
            return "\n".join(lines[start_line - 1: start_line - 1 + num_lines])
        return text


# ---------------------------------------------------------------------------
# doc_find
# ---------------------------------------------------------------------------

def doc_find(
    path: str,
    pattern: str,
    max_hits: int = 50,
    context_lines: int = 1,
) -> List[Dict[str, Any]]:
    """
    Search for regex pattern within a document.

    Returns list of {line_num, text, context_before, context_after}.
    For PDF: extracts full text first, then searches.
    """
    p = Path(path)
    mime = "pdf" if p.suffix.lower() == ".pdf" else "text"

    if mime == "pdf":
        text = extract_pdf_text(path)
    else:
        text = p.read_text(encoding="utf-8", errors="replace")

    lines = text.splitlines()
    regex = re.compile(pattern, re.IGNORECASE)
    hits = []

    for i, line in enumerate(lines):
        if regex.search(line):
            before = lines[max(0, i - context_lines): i]
            after = lines[i + 1: i + 1 + context_lines]
            hits.append({
                "line_num": i + 1,
                "text": line.strip(),
                "context_before": [l.strip() for l in before],
                "context_after": [l.strip() for l in after],
            })
            if len(hits) >= max_hits:
                break

    return hits


# ---------------------------------------------------------------------------
# doc_chunk_plan — chunking strategies
# ---------------------------------------------------------------------------

def doc_chunk_plan(
    path: str,
    strategy: str = "headings",
    max_chunk_tokens: int = 800,
    overlap_lines: int = 5,
    window_lines: int = 200,
) -> List[DocChunk]:
    """
    Plan how to chunk a document for memory ingestion.

    Strategies:
        "headings": split at detected heading boundaries
        "windows": fixed-size line windows with overlap
        "pages": one chunk per page range (PDF only)

    Returns list of DocChunk with metadata (text populated).
    """
    p = Path(path)
    mime = "pdf" if p.suffix.lower() == ".pdf" else "text"

    # Extract full text
    if mime == "pdf":
        full_text = extract_pdf_text(path)
    else:
        full_text = p.read_text(encoding="utf-8", errors="replace")

    lines = full_text.splitlines()
    doc_name = p.name

    if strategy == "headings":
        return _chunk_by_headings(path, doc_name, lines, max_chunk_tokens)
    elif strategy == "windows":
        return _chunk_by_windows(path, doc_name, lines, window_lines, overlap_lines)
    elif strategy == "pages" and mime == "pdf":
        return _chunk_by_pages(path, doc_name, pages_per_chunk=5)
    else:
        # Fallback to windows
        return _chunk_by_windows(path, doc_name, lines, window_lines, overlap_lines)


def _chunk_by_headings(
    path: str, doc_name: str, lines: List[str], max_tokens: int,
) -> List[DocChunk]:
    """Split at heading boundaries, merge small sections."""
    # Find heading positions
    heading_positions: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if _is_heading(line):
            heading_positions.append((i, line.strip()))

    if not heading_positions:
        # No headings found, fall back to windows
        return _chunk_by_windows(path, doc_name, lines, 200, 5)

    chunks = []
    chunk_idx = 0

    for pos_idx, (start, heading) in enumerate(heading_positions):
        # End is the line before next heading (or EOF)
        if pos_idx + 1 < len(heading_positions):
            end = heading_positions[pos_idx + 1][0]
        else:
            end = len(lines)

        section_text = "\n".join(lines[start:end])
        tokens = _estimate_tokens(section_text)

        # If section is too large, split into sub-windows
        if tokens > max_tokens * 2:
            sub_lines = lines[start:end]
            sub_chunks = _chunk_by_windows(
                path, doc_name, sub_lines,
                window_lines=max(50, max_tokens * 4 // max(1, len("x" * 4))),
                overlap=3, base_line=start, heading_prefix=heading,
            )
            for sc in sub_chunks:
                sc.chunk_id = f"{doc_name}:H{chunk_idx:03d}"
                chunk_idx += 1
            chunks.extend(sub_chunks)
        else:
            chunks.append(DocChunk(
                chunk_id=f"{doc_name}:H{chunk_idx:03d}",
                doc_path=path, doc_name=doc_name,
                start_line=start + 1, end_line=end,
                heading=heading,
                text=section_text,
                token_estimate=tokens,
                content_hash=_content_hash(section_text),
            ))
            chunk_idx += 1

    return chunks


def _chunk_by_windows(
    path: str, doc_name: str, lines: List[str],
    window_lines: int = 200, overlap: int = 5,
    base_line: int = 0, heading_prefix: str = "",
) -> List[DocChunk]:
    """Fixed-size line windows with overlap."""
    chunks = []
    i = 0
    chunk_idx = 0

    while i < len(lines):
        end = min(i + window_lines, len(lines))
        text = "\n".join(lines[i:end])
        tokens = _estimate_tokens(text)

        heading = heading_prefix
        # Try to find a heading in this window
        for j in range(i, min(i + 10, end)):
            if _is_heading(lines[j]):
                heading = lines[j].strip()
                break

        chunks.append(DocChunk(
            chunk_id=f"{doc_name}:W{chunk_idx:03d}",
            doc_path=path, doc_name=doc_name,
            start_line=base_line + i + 1,
            end_line=base_line + end,
            heading=heading,
            text=text,
            token_estimate=tokens,
            content_hash=_content_hash(text),
        ))
        chunk_idx += 1
        i = end - overlap if end < len(lines) else end

    return chunks


def _chunk_by_pages(
    path: str, doc_name: str, pages_per_chunk: int = 5,
) -> List[DocChunk]:
    """One chunk per page range (PDF only)."""
    total_pages = count_pdf_pages(path)
    if total_pages == 0:
        total_pages = 100  # fallback

    chunks = []
    chunk_idx = 0

    for start in range(1, total_pages + 1, pages_per_chunk):
        end = min(start + pages_per_chunk - 1, total_pages)
        text = extract_pdf_text(path, first_page=start, last_page=end)
        tokens = _estimate_tokens(text)

        # Find first heading
        heading = ""
        for line in text.splitlines()[:20]:
            if _is_heading(line):
                heading = line.strip()
                break

        chunks.append(DocChunk(
            chunk_id=f"{doc_name}:P{chunk_idx:03d}",
            doc_path=path, doc_name=doc_name,
            start_page=start, end_page=end,
            heading=heading,
            text=text,
            token_estimate=tokens,
            content_hash=_content_hash(text),
        ))
        chunk_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Tool dispatcher (doc.* namespace)
# ---------------------------------------------------------------------------

def dispatch_doc_tool(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a doc.* tool action.

    Actions: doc.list, doc.stats, doc.read, doc.find, doc.chunk_plan
    """
    if action.startswith("doc."):
        action = action[4:]

    try:
        if action == "list":
            results = doc_list(
                root=params.get("root", "."),
                globs=params.get("globs"),
                max_files=params.get("max_files", 200),
                sort=params.get("sort", "name"),
            )
            return {"status": "ok", "files": results, "count": len(results)}

        elif action == "stats":
            stats = doc_stats(params["path"])
            return {"status": "ok", **stats.to_dict()}

        elif action == "read":
            text = doc_read(
                path=params["path"],
                start_page=params.get("start_page", 1),
                end_page=params.get("end_page", 0),
                start_line=params.get("start_line", 0),
                num_lines=params.get("num_lines", 0),
            )
            return {
                "status": "ok",
                "text": text,
                "token_estimate": _estimate_tokens(text),
            }

        elif action == "find":
            hits = doc_find(
                path=params["path"],
                pattern=params.get("pattern", ""),
                max_hits=params.get("max_hits", 50),
            )
            return {"status": "ok", "hits": hits, "count": len(hits)}

        elif action == "chunk_plan":
            chunks = doc_chunk_plan(
                path=params["path"],
                strategy=params.get("strategy", "headings"),
                max_chunk_tokens=params.get("max_chunk_tokens", 800),
            )
            return {
                "status": "ok",
                "chunks": [c.to_dict() for c in chunks],
                "count": len(chunks),
                "total_tokens": sum(c.token_estimate for c in chunks),
            }

        else:
            return {"status": "error", "message": f"Unknown doc action: {action}"}

    except Exception as e:
        logger.exception(f"doc.{action} failed")
        return {"status": "error", "message": str(e)}
