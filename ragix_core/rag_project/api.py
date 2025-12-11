"""
RAGIX Project RAG - High-Level API

Provides the main RAGProject class for project-level RAG operations.
This is the primary interface used by ragix-web and the chat system.

Key features:
    - Project initialization and status
    - Indexing control (start, stop, progress)
    - Query interface (concept search, retrieval)
    - Chat context injection

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging

from .config import (
    RAGConfig,
    ProfileType,
    IndexingProfile,
    load_config,
    save_config,
    ensure_rag_initialized,
    has_rag_index,
    get_rag_dir,
)
from .metadata import MetadataStore, IndexingState, FileMetadata, ChunkMetadata
from .vector_store import VectorStore, SearchResult, CollectionType
from .graph import KnowledgeGraph, GraphNode, NodeType
from .worker import (
    IndexingWorker,
    WorkerProgress,
    start_indexing,
    stop_indexing,
    get_indexing_progress,
    is_indexing,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Project RAG Context (for chat injection)
# =============================================================================

@dataclass
class ProjectRAGContext:
    """
    Context retrieved from Project RAG for chat injection.

    Contains relevant chunks with citations for LLM prompting.
    """
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_chars: int = 0

    def is_empty(self) -> bool:
        return len(self.results) == 0

    def format_for_prompt(self, max_chars: int = 8000) -> str:
        """
        Format context for LLM prompt injection.

        Args:
            max_chars: Maximum characters to include

        Returns:
            Formatted context string with citations
        """
        if self.is_empty():
            return ""

        parts = [
            "## 📚 PROJECT CONTEXT (from RAG Index)\n",
            "**IMPORTANT:** The following content has been retrieved from the project index.",
            "Use this content to answer the user's question. Cite sources when referencing specific files.\n",
        ]

        char_count = sum(len(p) for p in parts)

        for i, result in enumerate(self.results, 1):
            citation = result.get_citation()
            content = result.content

            # Truncate if needed
            remaining = max_chars - char_count - 150  # Reserve space for formatting
            if len(content) > remaining:
                content = content[:remaining] + "..."

            # Add code complexity stats for code chunks (useful for AI reasoning)
            stats_note = ""
            if result.is_code:
                cc = result.cc_estimate
                if cc > 10:
                    stats_note = f" ⚠️ **HIGH COMPLEXITY** (CC={cc})"
                elif cc > 5:
                    stats_note = f" ⚡ Moderate complexity (CC={cc})"
                # Note: Low complexity (CC<=5) is not mentioned to reduce noise

            chunk_text = f"\n### [{i}] {citation}{stats_note}\n```\n{content}\n```\n"
            parts.append(chunk_text)
            char_count += len(chunk_text)

            if char_count >= max_chars:
                break

        parts.append(f"\n---\n*Retrieved {len(self.results)} relevant sections from project index.*\n")

        return "\n".join(parts)

    def get_citations(self) -> List[str]:
        """Get list of citation strings."""
        return [r.get_citation() for r in self.results]


# =============================================================================
# RAG Project
# =============================================================================

class RAGProject:
    """
    High-level API for project RAG operations.

    Main entry point for:
        - Checking project RAG status
        - Starting/stopping indexing
        - Querying the index
        - Retrieving context for chat
    """

    def __init__(self, project_root: Path):
        """
        Initialize RAG project.

        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root).resolve()
        self.rag_dir = get_rag_dir(self.project_root)

        # Lazy-loaded components
        self._config: Optional[RAGConfig] = None
        self._metadata: Optional[MetadataStore] = None
        self._vector_store: Optional[VectorStore] = None
        self._graph: Optional[KnowledgeGraph] = None

    # -------------------------------------------------------------------------
    # Status & Configuration
    # -------------------------------------------------------------------------

    @property
    def config(self) -> Optional[RAGConfig]:
        """Get RAG configuration (lazy load)."""
        if self._config is None:
            self._config = load_config(self.project_root)
        return self._config

    @property
    def metadata(self) -> MetadataStore:
        """Get metadata store (lazy init)."""
        if self._metadata is None:
            self._metadata = MetadataStore(self.project_root)
        return self._metadata

    @property
    def vector_store(self) -> VectorStore:
        """Get vector store (lazy init)."""
        if self._vector_store is None:
            self._vector_store = VectorStore(self.project_root)
        return self._vector_store

    @property
    def graph(self) -> KnowledgeGraph:
        """Get knowledge graph (lazy init)."""
        if self._graph is None:
            self._graph = KnowledgeGraph(self.project_root)
            self._graph.load()
        return self._graph

    def exists(self) -> bool:
        """Check if .RAG/ directory exists."""
        return self.rag_dir.exists()

    def is_initialized(self) -> bool:
        """Check if RAG index has been initialized."""
        return has_rag_index(self.project_root)

    def is_indexing(self) -> bool:
        """Check if indexing is currently running."""
        return is_indexing(self.project_root)

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive RAG status.

        Returns:
            Dict with status information
        """
        state = self.metadata.load_state()
        config = self.config

        status = {
            "project_root": str(self.project_root),
            "rag_dir": str(self.rag_dir),
            "exists": self.exists(),
            "initialized": self.is_initialized(),
            "is_indexing": self.is_indexing(),
            "state": {
                "status": state.status,
                "files_total": state.files_total,
                "files_indexed": state.files_indexed,
                "chunks_indexed": state.chunks_indexed,
                "progress_percent": state.progress_percent,
                "last_update": state.last_update,
                "error": state.error,
            },
        }

        if config:
            status["config"] = {
                "active_profile": config.active_profile,
                "project_name": config.project_name,
            }

        # Add vector store stats if initialized
        if self.is_initialized():
            try:
                vs_stats = self.vector_store.get_stats()
                status["vector_store"] = vs_stats
            except Exception as e:
                status["vector_store"] = {"error": str(e)}

        return status

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            "metadata": self.metadata.get_stats(),
        }

        if self.is_initialized():
            stats["vector_store"] = self.vector_store.get_stats()
            stats["graph"] = self.graph.get_stats()

        return stats

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(
        self,
        profile: ProfileType = ProfileType.MIXED_DOCS_CODE,
        project_name: Optional[str] = None,
    ) -> RAGConfig:
        """
        Initialize .RAG/ with configuration.

        Args:
            profile: Initial indexing profile
            project_name: Optional project name

        Returns:
            Created RAGConfig
        """
        config = ensure_rag_initialized(
            self.project_root,
            profile=profile,
            project_name=project_name or self.project_root.name,
        )
        self._config = config
        return config

    def set_profile(self, profile: ProfileType) -> None:
        """Change the active indexing profile."""
        if self.config:
            self.config.set_active_profile(profile.value)
            save_config(self.project_root, self.config)

    # -------------------------------------------------------------------------
    # Indexing Control
    # -------------------------------------------------------------------------

    def start_indexing(
        self,
        full_reindex: bool = False,
        on_progress: Optional[Callable[[WorkerProgress], None]] = None,
        on_complete: Optional[Callable[[bool, Optional[str]], None]] = None,
    ) -> IndexingWorker:
        """
        Start background indexing.

        Args:
            full_reindex: Clear and rebuild from scratch
            on_progress: Progress callback
            on_complete: Completion callback

        Returns:
            IndexingWorker instance
        """
        # Ensure initialized
        if not self.exists():
            self.initialize()

        return start_indexing(
            self.project_root,
            full_reindex=full_reindex,
            config=self.config,
            on_progress=on_progress,
            on_complete=on_complete,
        )

    def stop_indexing(self, wait: bool = True) -> bool:
        """
        Stop background indexing.

        Args:
            wait: Wait for worker to stop

        Returns:
            True if stopped
        """
        return stop_indexing(self.project_root, wait=wait)

    def get_indexing_progress(self) -> Optional[WorkerProgress]:
        """Get current indexing progress."""
        return get_indexing_progress(self.project_root)

    # -------------------------------------------------------------------------
    # Query Interface
    # -------------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        collection: str = CollectionType.MIXED,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Query the RAG index.

        Args:
            query_text: Search query
            top_k: Number of results
            collection: Collection to search (docs, code, mixed)
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        if not self.is_initialized():
            logger.warning("RAG not initialized, cannot query")
            return []

        return self.vector_store.query(
            query_text=query_text,
            top_k=top_k,
            collection_type=collection,
            filters=filters,
        )

    def query_code(
        self,
        query_text: str,
        top_k: int = 10,
        language: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Query code chunks specifically.

        Args:
            query_text: Search query
            top_k: Number of results
            language: Filter by programming language

        Returns:
            List of SearchResult objects
        """
        filters = None
        if language:
            filters = {"kind": f"code_{language}"}

        return self.query(
            query_text=query_text,
            top_k=top_k,
            collection=CollectionType.CODE,
            filters=filters,
        )

    def query_docs(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Query documentation chunks specifically."""
        return self.query(
            query_text=query_text,
            top_k=top_k,
            collection=CollectionType.DOCS,
        )

    def query_complex_code(
        self,
        query_text: str,
        top_k: int = 10,
        min_complexity: int = 5,
    ) -> List[SearchResult]:
        """
        Query code chunks with high complexity.

        Useful for finding complex code that may need refactoring.

        Args:
            query_text: Search query
            top_k: Number of results
            min_complexity: Minimum CC threshold (default 5)

        Returns:
            List of SearchResult for complex code chunks
        """
        # Query code with is_complex filter
        results = self.query(
            query_text=query_text,
            top_k=top_k * 2,  # Get more to filter
            collection=CollectionType.CODE,
            filters={"is_complex": True},
        )
        # Filter and sort by complexity
        filtered = [r for r in results if r.cc_estimate >= min_complexity]
        filtered.sort(key=lambda r: -r.cc_estimate)
        return filtered[:top_k]

    def search_concept(
        self,
        concept: str,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for a concept across the project.

        Returns both vector search results and graph relationships.

        Args:
            concept: Concept to search for
            top_k: Number of results

        Returns:
            Dict with 'results', 'files', 'related_concepts'
        """
        # Vector search
        results = self.query(concept, top_k=top_k)

        # Graph lookup
        concept_node = self.graph.get_concept_by_label(concept)

        files = []
        related_concepts = []

        if concept_node:
            # Get files mentioning this concept
            file_nodes = self.graph.get_files_for_concept(concept_node.id)
            files = [
                {"file_id": f.id, "path": f.data.get("path", ""), "kind": f.data.get("kind", "")}
                for f in file_nodes
            ]

            # Could add related concept lookup here

        return {
            "concept": concept,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "file_path": r.file_path,
                    "line_start": r.line_start,
                    "line_end": r.line_end,
                    "content": r.content[:500],
                }
                for r in results
            ],
            "files": files,
            "related_concepts": related_concepts,
            "graph_node_exists": concept_node is not None,
        }

    # -------------------------------------------------------------------------
    # Change Detection
    # -------------------------------------------------------------------------

    def detect_changes(self) -> Dict[str, Any]:
        """
        Detect file changes since last indexing.

        Lightweight scan comparing current file tree against indexed metadata:
        - New files: Present on disk but not in index
        - Modified files: mtime or hash changed
        - Deleted files: In index but not on disk

        Returns:
            Dict with 'new', 'modified', 'deleted' lists and 'has_changes' flag
        """
        from .ingest import FileIngester

        changes = {
            "new": [],
            "modified": [],
            "deleted": [],
            "has_changes": False,
            "summary": "",
        }

        if not self.is_initialized():
            return changes

        # Force fresh metadata reload (clear any stale cache)
        self.metadata._files_by_id = None
        self.metadata._files_by_path = None

        # Load indexed files
        indexed_files = self.metadata.load_files()
        indexed_paths = set(self.metadata._files_by_path.keys()) if self.metadata._files_by_path else set()

        # Scan current file tree (lightweight - no content reading)
        ingester = FileIngester(self.project_root, self.config)
        current_paths = set()

        for file_path in ingester.scan_files():
            rel_path = str(file_path.relative_to(self.project_root))
            current_paths.add(rel_path)

            if rel_path not in indexed_paths:
                # New file
                changes["new"].append(rel_path)
            else:
                # Check if modified (by mtime or size)
                indexed_meta = self.metadata._files_by_path.get(rel_path)
                if indexed_meta:
                    try:
                        stat = file_path.stat()
                        current_mtime = stat.st_mtime
                        current_size = stat.st_size

                        # Parse indexed mtime
                        from datetime import datetime
                        indexed_mtime = None
                        if indexed_meta.mtime:
                            try:
                                indexed_mtime = datetime.fromisoformat(indexed_meta.mtime).timestamp()
                            except ValueError:
                                pass

                        # Compare mtime (with 1 second tolerance) or size
                        if indexed_mtime is None:
                            pass  # Can't compare, assume unchanged
                        elif abs(current_mtime - indexed_mtime) > 1 or current_size != indexed_meta.size_bytes:
                            changes["modified"].append(rel_path)
                    except OSError:
                        pass  # File may have been deleted between scan and stat

        # Find deleted files
        for path in indexed_paths:
            if path not in current_paths:
                changes["deleted"].append(path)

        # Set flags
        changes["has_changes"] = bool(changes["new"] or changes["modified"] or changes["deleted"])

        # Build summary
        parts = []
        if changes["new"]:
            parts.append(f"{len(changes['new'])} new file(s)")
        if changes["modified"]:
            parts.append(f"{len(changes['modified'])} modified file(s)")
        if changes["deleted"]:
            parts.append(f"{len(changes['deleted'])} deleted file(s)")
        changes["summary"] = ", ".join(parts) if parts else "No changes detected"

        return changes

    def get_change_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a quick summary of changes (for notification display).

        Returns:
            None if no changes, otherwise dict with summary info
        """
        changes = self.detect_changes()
        if not changes["has_changes"]:
            return None

        return {
            "summary": changes["summary"],
            "new_count": len(changes["new"]),
            "modified_count": len(changes["modified"]),
            "deleted_count": len(changes["deleted"]),
            "total_changes": len(changes["new"]) + len(changes["modified"]) + len(changes["deleted"]),
        }

    # -------------------------------------------------------------------------
    # Chat Context Injection
    # -------------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        collection: str = CollectionType.MIXED,
        max_chars_per_chunk: int = 1500,
    ) -> ProjectRAGContext:
        """
        Retrieve context for chat prompt injection.

        This is the main method used by the chat system to get
        relevant project context for LLM prompts.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            collection: Collection to search
            max_chars_per_chunk: Max chars per chunk

        Returns:
            ProjectRAGContext for prompt injection
        """
        if not self.is_initialized():
            return ProjectRAGContext(query=query)

        results = self.query(
            query_text=query,
            top_k=top_k,
            collection=collection,
        )

        # Truncate content if needed
        for r in results:
            if len(r.content) > max_chars_per_chunk:
                r.content = r.content[:max_chars_per_chunk] + "..."

        total_chars = sum(len(r.content) for r in results)

        return ProjectRAGContext(
            query=query,
            results=results,
            total_chars=total_chars,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def clear(self) -> bool:
        """
        Clear all RAG data (full reset).

        Returns:
            True if cleared successfully
        """
        if self.is_indexing():
            self.stop_indexing()

        try:
            self.metadata.clear_all()
            self.vector_store.clear_all()
            self.graph.clear()
            self.graph.delete_graph_file()
            return True
        except Exception as e:
            logger.error(f"Failed to clear RAG: {e}")
            return False

    def close(self) -> None:
        """Close all connections."""
        if self._vector_store:
            self._vector_store.close()


# =============================================================================
# Module-Level Functions
# =============================================================================

def get_project_rag(project_root: Path) -> RAGProject:
    """
    Get RAGProject instance for a project.

    Args:
        project_root: Project root directory

    Returns:
        RAGProject instance
    """
    return RAGProject(project_root)


def retrieve_project_context(
    project_root: Path,
    query: str,
    top_k: int = 5,
    collection: str = CollectionType.MIXED,
) -> ProjectRAGContext:
    """
    Convenience function to retrieve context from project RAG.

    Args:
        project_root: Project root directory
        query: User query
        top_k: Number of results
        collection: Collection to search

    Returns:
        ProjectRAGContext for prompt injection
    """
    project = RAGProject(project_root)
    return project.retrieve_context(
        query=query,
        top_k=top_k,
        collection=collection,
    )


def check_project_rag_status(project_root: Path) -> Dict[str, Any]:
    """
    Quick check of project RAG status.

    Args:
        project_root: Project root directory

    Returns:
        Status dict with 'exists', 'initialized', 'is_indexing'
    """
    project = RAGProject(project_root)
    return {
        "exists": project.exists(),
        "initialized": project.is_initialized(),
        "is_indexing": project.is_indexing(),
    }
