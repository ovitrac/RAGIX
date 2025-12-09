"""
RAGIX Project RAG - Massive project-level RAG with ChromaDB and Knowledge Graph

This module provides project-wide RAG indexing for code audits, literature reviews,
and heterogeneous document analysis. It is designed to handle thousands of files
with GPU-accelerated vector search via ChromaDB.

Architecture:
    - Storage: .RAG/ folder inside each project
    - Vector Store: ChromaDB (CUDA-accelerated when available)
    - Knowledge Graph: file → chunk → concept relationships
    - Indexing: Background async worker with progress tracking

Two-Level RAG System:
    - Level 1 (this module): Project RAG - massive, persistent, ChromaDB-based
    - Level 2 (rag_chat): Chat RAG - light, session-scoped, BM25-based

The Project RAG is wired to the chat to provide natural project interrogation.

Usage:
    from ragix_core.rag_project import RAGProject, ProfileType

    # Initialize project RAG
    project = RAGProject("/path/to/project")
    project.initialize(profile=ProfileType.MIXED_DOCS_CODE)

    # Start background indexing
    project.start_indexing()

    # Query the index
    results = project.query("authentication middleware")

    # Get context for chat injection
    context = project.retrieve_context("How does login work?")
    prompt_text = context.format_for_prompt()

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

# Config
from .config import (
    RAGConfig,
    IndexingProfile,
    ProfileType,
    VectorStoreConfig as ConfigVectorStoreConfig,
    IndexingFilters,
    load_config,
    save_config,
    get_default_config,
    ensure_rag_initialized,
    has_rag_index,
    get_rag_dir,
    PROFILE_DOCS_ONLY,
    PROFILE_MIXED,
    PROFILE_CODE_ONLY,
)

# Metadata
from .metadata import (
    FileKind,
    FileMetadata,
    ChunkMetadata,
    IndexingState,
    IndexingStatus,
    MetadataStore,
)

# Chunking
from .chunking import (
    Chunk,
    Chunker,
    chunk_text,
    chunk_code,
)

# Vector Store
from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    SearchResult,
    CollectionType,
)

# Graph
from .graph import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)

# Ingest
from .ingest import (
    FileIngester,
    IngestResult,
    normalize_file,
    detect_file_kind,
    detect_language,
    is_code_file,
    is_doc_file,
    extract_tags,
)

# Worker
from .worker import (
    IndexingWorker,
    WorkerStatus,
    WorkerProgress,
    start_indexing,
    stop_indexing,
    get_indexing_progress,
    is_indexing,
    get_worker,
)

# API (main entry point)
from .api import (
    RAGProject,
    ProjectRAGContext,
    retrieve_project_context,
    get_project_rag,
    check_project_rag_status,
)

__all__ = [
    # Config
    "RAGConfig",
    "IndexingProfile",
    "ProfileType",
    "IndexingFilters",
    "load_config",
    "save_config",
    "get_default_config",
    "ensure_rag_initialized",
    "has_rag_index",
    "get_rag_dir",
    "PROFILE_DOCS_ONLY",
    "PROFILE_MIXED",
    "PROFILE_CODE_ONLY",
    # Metadata
    "FileKind",
    "FileMetadata",
    "ChunkMetadata",
    "IndexingState",
    "IndexingStatus",
    "MetadataStore",
    # Chunking
    "Chunk",
    "Chunker",
    "chunk_text",
    "chunk_code",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "SearchResult",
    "CollectionType",
    # Graph
    "KnowledgeGraph",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "EdgeType",
    # Ingest
    "FileIngester",
    "IngestResult",
    "normalize_file",
    "detect_file_kind",
    "detect_language",
    "is_code_file",
    "is_doc_file",
    "extract_tags",
    # Worker
    "IndexingWorker",
    "WorkerStatus",
    "WorkerProgress",
    "start_indexing",
    "stop_indexing",
    "get_indexing_progress",
    "is_indexing",
    "get_worker",
    # API
    "RAGProject",
    "ProjectRAGContext",
    "retrieve_project_context",
    "get_project_rag",
    "check_project_rag_status",
]
