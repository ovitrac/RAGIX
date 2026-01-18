"""
Kernel: Document Metadata
Stage: 1 (Collection)

Extracts document metadata and statistics from the RAG index.
This is the foundation kernel for document summarization — all other
doc kernels depend on this data.

Provides:
- File inventory with kinds (DOCX, PDF, PPTX, XLSX, etc.)
- Chunk counts per file
- Statistics by document type
- File size distributions

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocMetadataKernel(Kernel):
    """
    Extract document metadata from RAG index.

    This kernel reads the RAG metadata store to collect:
    - All indexed files with their types
    - Chunk counts per file
    - Statistics aggregated by file kind

    Configuration options:
        project.path: Path to the indexed project (required)
        include_structure: Include file structure details (default: true)
        filter_docs_only: Only include document files, exclude code (default: true)

    Dependencies:
        None (reads directly from RAG index)

    Output:
        files: List of file metadata (file_id, path, kind, chunk_count, size_bytes)
        statistics: Aggregated statistics
        by_kind: Files grouped by kind
        by_extension: Files grouped by extension
    """

    name = "doc_metadata"
    version = "1.0.0"
    category = "docs"
    stage = 1
    description = "Extract document metadata from RAG index"

    requires = []  # First stage — reads from RAG directly
    provides = ["doc_metadata", "doc_statistics", "doc_inventory"]

    # Document file kinds (from FileKind enum)
    DOC_KINDS = {
        "doc_markdown", "doc_text", "doc_rst", "doc_pdf",
        "doc_docx", "doc_pptx", "doc_xlsx", "doc_odt", "doc_other",
    }

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Extract document metadata from RAG index."""
        from ragix_core.rag_project import RAGProject, MetadataStore

        # Get project path from config
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        project_path = Path(project_path_str)
        if not project_path.exists():
            raise RuntimeError(f"Project path does not exist: {project_path}")

        # Configuration
        include_structure = input.config.get("include_structure", True)
        filter_docs_only = input.config.get("filter_docs_only", True)

        logger.info(f"[doc_metadata] Loading RAG metadata from {project_path}")

        # Initialize RAG project and metadata store
        rag = RAGProject(project_path)

        if not rag.is_initialized():
            raise RuntimeError(f"RAG index not initialized for {project_path}")

        metadata = rag.metadata

        # Collect file information
        files: List[Dict[str, Any]] = []
        kind_counts: Counter = Counter()
        extension_counts: Counter = Counter()
        total_chunks = 0
        total_size = 0

        for file_meta in metadata.iter_files():
            # Filter to docs only if requested
            if filter_docs_only and file_meta.kind not in self.DOC_KINDS:
                continue

            file_info = {
                "file_id": file_meta.file_id,
                "path": file_meta.path,
                "kind": file_meta.kind,
                "chunk_count": file_meta.chunk_count,
                "size_bytes": file_meta.size_bytes,
                "language": file_meta.language,
                "indexed_at": file_meta.indexed_at,
            }

            # Extract extension
            ext = Path(file_meta.path).suffix.lower()
            file_info["extension"] = ext

            files.append(file_info)

            # Count statistics
            kind_counts[file_meta.kind] += 1
            extension_counts[ext] += 1
            total_chunks += file_meta.chunk_count
            total_size += file_meta.size_bytes

        logger.info(
            f"[doc_metadata] Found {len(files)} documents with {total_chunks} chunks"
        )

        # Build by_kind grouping
        by_kind: Dict[str, List[Dict[str, Any]]] = {}
        for f in files:
            kind = f["kind"]
            if kind not in by_kind:
                by_kind[kind] = []
            by_kind[kind].append({
                "file_id": f["file_id"],
                "path": f["path"],
                "chunk_count": f["chunk_count"],
            })

        # Build by_extension grouping
        by_extension: Dict[str, List[str]] = {}
        for f in files:
            ext = f["extension"]
            if ext not in by_extension:
                by_extension[ext] = []
            by_extension[ext].append(f["path"])

        # Compute statistics
        statistics = {
            "total_files": len(files),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "avg_chunks_per_file": round(total_chunks / len(files), 1) if files else 0,
            "avg_size_bytes": round(total_size / len(files)) if files else 0,
            "kind_counts": dict(kind_counts),
            "extension_counts": dict(extension_counts),
        }

        # Add structure info if requested
        structure = {}
        if include_structure:
            # Find common directory prefixes
            paths = [Path(f["path"]) for f in files]
            if paths:
                # Count files by top-level directory
                dir_counts: Counter = Counter()
                for p in paths:
                    parts = p.parts
                    if len(parts) > 1:
                        dir_counts[parts[0]] += 1
                    else:
                        dir_counts["(root)"] += 1

                structure = {
                    "top_directories": dict(dir_counts.most_common(20)),
                    "max_depth": max(len(p.parts) for p in paths) if paths else 0,
                }

        return {
            "files": files,
            "statistics": statistics,
            "by_kind": by_kind,
            "by_extension": by_extension,
            "structure": structure,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total_files = stats.get("total_files", 0)
        total_chunks = stats.get("total_chunks", 0)
        size_mb = stats.get("total_size_mb", 0)

        kind_counts = stats.get("kind_counts", {})
        top_kinds = sorted(kind_counts.items(), key=lambda x: -x[1])[:3]
        kinds_str = ", ".join(f"{k.replace('doc_', '').upper()}:{v}" for k, v in top_kinds)

        return (
            f"Document metadata: {total_files} files, {total_chunks} chunks, "
            f"{size_mb} MB. Top types: {kinds_str}."
        )
