"""
Kernel: Document Structure
Stage: 1 (Collection)

Detects document structure by analyzing chunk metadata for section
titles, headings, and organizational patterns.

For DOCX/PDF/PPTX documents, the chunker may have extracted:
- Section titles (from styles like Heading 1, Heading 2)
- Slide numbers (for PPTX)
- Sheet names (for XLSX)

This kernel aggregates that information to build per-document outlines.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import logging
import json
import re

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocStructureKernel(Kernel):
    """
    Detect document structure from chunk metadata.

    This kernel analyzes chunk metadata to identify:
    - Section titles and headings
    - Document outlines (TOC approximation)
    - Structural patterns across the corpus

    Configuration options:
        project.path: Path to the indexed project (required)
        detect_headings: Attempt heading detection from chunk text (default: true)
        extract_toc: Build table of contents per document (default: true)
        heading_patterns: Regex patterns for heading detection

    Dependencies:
        doc_metadata: File inventory

    Output:
        documents: Per-document structure information
        outlines: Document outlines (TOC)
        section_types: Common section types found
        statistics: Aggregated statistics
    """

    name = "doc_structure"
    version = "1.0.0"
    category = "docs"
    stage = 1
    description = "Detect document structure from chunk metadata"

    requires = ["doc_metadata"]
    provides = ["doc_structure", "doc_outlines", "section_types"]

    # Default heading patterns (for text-based detection)
    DEFAULT_HEADING_PATTERNS = [
        r"^#+\s+(.+)$",                          # Markdown headings
        r"^(\d+\.)+\s+(.+)$",                    # Numbered sections (1.2.3 Title)
        r"^(Chapter|Section|Part)\s+\d+[:\.]?\s*(.+)?$",  # Explicit section markers
        r"^([A-Z][A-Z\s]+[A-Z])$",               # ALL CAPS titles
        r"^(Introduction|Conclusion|Abstract|Summary|References|Appendix)",  # Common sections
    ]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Detect document structure from chunk metadata."""
        from ragix_core.rag_project import RAGProject, MetadataStore

        # Get project path from config
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        project_path = Path(project_path_str)

        # Configuration
        detect_headings = input.config.get("detect_headings", True)
        extract_toc = input.config.get("extract_toc", True)
        heading_patterns = input.config.get("heading_patterns", self.DEFAULT_HEADING_PATTERNS)

        logger.info(f"[doc_structure] Analyzing document structure from {project_path}")

        # Load doc_metadata to get file list
        metadata_path = input.dependencies.get("doc_metadata")
        if not metadata_path or not metadata_path.exists():
            raise RuntimeError("Missing required dependency: doc_metadata")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})

        doc_files = {f["file_id"]: f for f in metadata_data.get("files", [])}
        logger.info(f"[doc_structure] Processing {len(doc_files)} documents")

        # Initialize RAG project
        rag = RAGProject(project_path)
        metadata = rag.metadata

        # Compile heading patterns
        compiled_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                           for p in heading_patterns]

        # Analyze structure per document
        documents: Dict[str, Dict[str, Any]] = {}
        outlines: Dict[str, List[Dict[str, Any]]] = {}
        section_types: Dict[str, int] = defaultdict(int)

        total_sections = 0
        docs_with_structure = 0

        for file_id, file_info in doc_files.items():
            file_path = file_info["path"]
            file_kind = file_info["kind"]

            doc_structure = {
                "file_id": file_id,
                "path": file_path,
                "kind": file_kind,
                "sections": [],
                "has_toc": False,
                "section_count": 0,
            }

            # Get chunks for this file
            chunks = list(metadata.iter_chunks_for_file(file_id))

            if not chunks:
                documents[file_id] = doc_structure
                continue

            # Sort chunks by position
            chunks.sort(key=lambda c: (c.line_start, c.offset_start))

            # Detect sections from chunk metadata
            sections: List[Dict[str, Any]] = []

            for chunk in chunks:
                section_info = None

                # Check extra metadata for section titles
                extra = chunk.extra
                if extra:
                    if "section_title" in extra:
                        section_info = {
                            "title": extra["section_title"],
                            "level": extra.get("heading_level", 1),
                            "source": "metadata",
                            "chunk_id": chunk.chunk_id,
                            "line_start": chunk.line_start,
                        }
                    elif "slide_number" in extra:
                        slide_title = extra.get("slide_title", f"Slide {extra['slide_number']}")
                        section_info = {
                            "title": slide_title,
                            "level": 1,
                            "source": "slide",
                            "slide_number": extra["slide_number"],
                            "chunk_id": chunk.chunk_id,
                        }
                    elif "sheet_name" in extra:
                        section_info = {
                            "title": extra["sheet_name"],
                            "level": 1,
                            "source": "sheet",
                            "chunk_id": chunk.chunk_id,
                        }

                # Try to detect headings from text preview
                if not section_info and detect_headings and chunk.text_preview:
                    preview = chunk.text_preview.strip()
                    first_line = preview.split("\n")[0] if preview else ""

                    for pattern in compiled_patterns:
                        match = pattern.match(first_line)
                        if match:
                            # Extract title from match groups
                            groups = match.groups()
                            title = groups[-1] if groups else first_line
                            if not title:
                                title = first_line

                            # Determine level from pattern
                            level = 1
                            if first_line.startswith("#"):
                                level = len(first_line) - len(first_line.lstrip("#"))
                            elif re.match(r"^\d+\.", first_line):
                                level = first_line.count(".")

                            section_info = {
                                "title": title.strip(),
                                "level": min(level, 6),
                                "source": "detected",
                                "chunk_id": chunk.chunk_id,
                                "line_start": chunk.line_start,
                            }
                            break

                if section_info:
                    sections.append(section_info)
                    section_types[section_info.get("source", "unknown")] += 1

            # Deduplicate sections (same title at same position)
            seen_titles: Set[str] = set()
            unique_sections = []
            for sec in sections:
                key = (sec["title"].lower(), sec.get("line_start", 0))
                if key not in seen_titles:
                    seen_titles.add(key)
                    unique_sections.append(sec)

            doc_structure["sections"] = unique_sections
            doc_structure["section_count"] = len(unique_sections)
            doc_structure["has_toc"] = len(unique_sections) > 0

            if unique_sections:
                docs_with_structure += 1
                total_sections += len(unique_sections)

            documents[file_id] = doc_structure

            # Build outline (TOC) if requested
            if extract_toc and unique_sections:
                outlines[file_id] = [
                    {
                        "title": s["title"],
                        "level": s["level"],
                        "line": s.get("line_start"),
                    }
                    for s in unique_sections
                ]

        # Statistics
        statistics = {
            "total_documents": len(documents),
            "docs_with_structure": docs_with_structure,
            "docs_without_structure": len(documents) - docs_with_structure,
            "total_sections": total_sections,
            "avg_sections_per_doc": (
                round(total_sections / docs_with_structure, 1)
                if docs_with_structure else 0
            ),
            "section_sources": dict(section_types),
        }

        logger.info(
            f"[doc_structure] Found {total_sections} sections "
            f"in {docs_with_structure}/{len(documents)} documents"
        )

        return {
            "documents": documents,
            "outlines": outlines,
            "section_types": dict(section_types),
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total_docs = stats.get("total_documents", 0)
        docs_with = stats.get("docs_with_structure", 0)
        total_secs = stats.get("total_sections", 0)
        avg_secs = stats.get("avg_sections_per_doc", 0)

        sources = stats.get("section_sources", {})
        source_str = ", ".join(f"{k}:{v}" for k, v in sources.items())

        return (
            f"Structure: {docs_with}/{total_docs} docs have sections, "
            f"{total_secs} sections total (avg {avg_secs}/doc). "
            f"Sources: {source_str}."
        )
