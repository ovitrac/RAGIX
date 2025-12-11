"""
RAGIX Project RAG - Background Indexing Worker

Runs indexing in a separate thread to avoid blocking the UI.

Features:
    - Non-blocking async indexing
    - Progress tracking via state.json
    - Cancellation support
    - Incremental updates (only changed files)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import logging

from .config import RAGConfig, load_config, save_config, ensure_rag_initialized, ProfileType
from .metadata import (
    MetadataStore,
    IndexingState,
    IndexingStatus,
    FileMetadata,
    ChunkMetadata,
)
from .chunking import Chunker, Chunk
from .ingest import FileIngester, IngestResult, extract_tags, is_code_file, FileKind
from .vector_store import VectorStore, CollectionType
from .graph import KnowledgeGraph, GraphNode, GraphEdge, EdgeType

# Code metrics for RAG integration
from ..code_metrics import estimate_chunk_complexity, compute_file_stats_for_rag

logger = logging.getLogger(__name__)

# =============================================================================
# Worker Status
# =============================================================================

class WorkerStatus(str, Enum):
    """Worker thread status."""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerProgress:
    """Current worker progress."""
    status: str
    files_total: int = 0
    files_processed: int = 0
    chunks_indexed: int = 0
    current_file: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    elapsed_seconds: float = 0.0

    @property
    def progress_percent(self) -> float:
        if self.files_total == 0:
            return 0.0
        return (self.files_processed / self.files_total) * 100


# =============================================================================
# Indexing Worker
# =============================================================================

class IndexingWorker:
    """
    Background worker for project indexing.

    Runs in a separate thread, updates state.json for progress tracking.
    """

    def __init__(
        self,
        project_root: Path,
        config: Optional[RAGConfig] = None,
        on_progress: Optional[Callable[[WorkerProgress], None]] = None,
        on_complete: Optional[Callable[[bool, Optional[str]], None]] = None,
    ):
        """
        Initialize indexing worker.

        Args:
            project_root: Project root directory
            config: RAG configuration (loads from .RAG/ if None)
            on_progress: Callback for progress updates
            on_complete: Callback when indexing completes (success, error_msg)
        """
        self.project_root = project_root
        self.config = config or load_config(project_root) or ensure_rag_initialized(project_root)
        self.on_progress = on_progress
        self.on_complete = on_complete

        # Components
        self.metadata = MetadataStore(project_root)
        self.vector_store = VectorStore(project_root)
        self.graph = KnowledgeGraph(project_root)
        self.ingester = FileIngester(project_root, self.config)
        self.chunker = Chunker(self.config.get_active_profile())

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status = WorkerStatus.IDLE
        self._error: Optional[str] = None

        # Progress tracking
        self._progress = WorkerProgress(status=WorkerStatus.IDLE.value)

    @property
    def status(self) -> WorkerStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == WorkerStatus.RUNNING

    @property
    def progress(self) -> WorkerProgress:
        return self._progress

    # -------------------------------------------------------------------------
    # Control Methods
    # -------------------------------------------------------------------------

    def start(self, full_reindex: bool = False) -> bool:
        """
        Start indexing in background thread.

        Args:
            full_reindex: If True, clear existing index first

        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("Worker already running")
            return False

        self._stop_event.clear()
        self._status = WorkerStatus.RUNNING
        self._error = None

        self._thread = threading.Thread(
            target=self._run_indexing,
            args=(full_reindex,),
            daemon=True,
            name="RAGIndexingWorker",
        )
        self._thread.start()

        logger.info(f"Indexing worker started for {self.project_root}")
        return True

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Stop the indexing worker.

        Args:
            wait: If True, wait for thread to finish
            timeout: Maximum time to wait
        """
        if not self.is_running:
            return

        self._status = WorkerStatus.STOPPING
        self._stop_event.set()

        if wait and self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Worker thread did not stop within timeout")

        self._status = WorkerStatus.STOPPED

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for indexing to complete.

        Args:
            timeout: Maximum time to wait (None = forever)

        Returns:
            True if completed, False if timeout
        """
        if self._thread:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    # -------------------------------------------------------------------------
    # Main Indexing Loop
    # -------------------------------------------------------------------------

    def _run_indexing(self, full_reindex: bool) -> None:
        """Main indexing loop (runs in thread)."""
        start_time = time.time()
        self._progress.started_at = datetime.now().isoformat()

        try:
            # Initialize state
            state = IndexingState()
            profile = self.config.get_active_profile()

            # Full reindex: clear everything
            if full_reindex:
                logger.info("Full reindex: clearing existing data")
                self.metadata.clear_all()
                self.vector_store.clear_all()
                self.graph.clear()

            # Load graph
            self.graph.load()

            # Count files
            logger.info("Scanning files...")
            self._progress.status = IndexingStatus.SCANNING.value
            files_total = self.ingester.count_files()
            self._progress.files_total = files_total

            state.start(profile.name, files_total)
            self.metadata.save_state(state)

            # Process files
            logger.info(f"Indexing {files_total} files with profile '{profile.name}'")
            files_processed = 0
            chunks_indexed = 0

            for ingest_result in self.ingester.ingest_all(
                profile_name=profile.name,
                file_id_generator=self.metadata.next_file_id,
            ):
                # Check for stop signal
                if self._stop_event.is_set():
                    logger.info("Indexing cancelled by user")
                    state.cancel()
                    self.metadata.save_state(state)
                    self._status = WorkerStatus.STOPPED
                    return

                # Process result
                if ingest_result.success and ingest_result.file_metadata:
                    chunks = self._process_file(ingest_result, profile.name)
                    chunks_indexed += len(chunks)
                    ingest_result.file_metadata.chunk_count = len(chunks)

                    # Save file metadata
                    self.metadata.append_file(ingest_result.file_metadata)
                else:
                    state.files_failed += 1
                    if ingest_result.error:
                        state.failed_files.append(f"{ingest_result.file_path}: {ingest_result.error}")

                files_processed += 1

                # Update progress
                self._progress.files_processed = files_processed
                self._progress.chunks_indexed = chunks_indexed
                self._progress.current_file = ingest_result.file_path
                self._progress.elapsed_seconds = time.time() - start_time

                state.update_progress(files_processed, chunks_indexed, ingest_result.file_path)

                # Save state periodically (every 10 files)
                if files_processed % 10 == 0:
                    self.metadata.save_state(state)
                    self._notify_progress()

            # Finalize
            self.vector_store.persist()
            self.graph.save()

            # Calculate index size
            index_size = self._calculate_index_size()
            state.complete(index_size)
            self.metadata.save_state(state)

            self._progress.status = IndexingStatus.COMPLETED.value
            self._status = WorkerStatus.IDLE

            logger.info(
                f"Indexing complete: {files_processed} files, "
                f"{chunks_indexed} chunks in {time.time() - start_time:.1f}s"
            )

            if self.on_complete:
                self.on_complete(True, None)

        except Exception as e:
            logger.exception(f"Indexing failed: {e}")
            self._error = str(e)
            self._progress.error = str(e)
            self._progress.status = IndexingStatus.ERROR.value
            self._status = WorkerStatus.ERROR

            state = self.metadata.load_state()
            state.fail(str(e))
            self.metadata.save_state(state)

            if self.on_complete:
                self.on_complete(False, str(e))

    def _process_file(
        self,
        ingest_result: IngestResult,
        profile_name: str,
    ) -> List[ChunkMetadata]:
        """
        Process a single file: chunk, embed, add to graph.

        Args:
            ingest_result: Ingestion result with content
            profile_name: Active profile name

        Returns:
            List of chunk metadata created
        """
        file_meta = ingest_result.file_metadata
        content = ingest_result.content

        if not file_meta or not content:
            return []

        # Determine if code
        is_code = file_meta.kind.startswith("code_")

        # Compute file-level stats for RAG metadata (code files only)
        if is_code:
            try:
                file_stats = compute_file_stats_for_rag(
                    Path(file_meta.path),
                    content,
                    file_meta.language or ""
                )
                file_meta.extra.update(file_stats)
            except Exception as e:
                logger.debug(f"Could not compute stats for {file_meta.path}: {e}")

        # Chunk the content
        chunks = self.chunker.chunk(content, is_code=is_code)

        if not chunks:
            return []

        # Create file node in graph
        file_node = GraphNode.file_node(
            file_id=file_meta.file_id,
            path=file_meta.path,
            kind=file_meta.kind,
        )
        self.graph.add_node(file_node)

        # Process chunks
        chunk_metadatas = []
        chunk_dicts = []

        for chunk in chunks:
            chunk_id = self.metadata.next_chunk_id()

            # Extract tags
            tags = extract_tags(chunk.content, file_meta.path, FileKind(file_meta.kind))

            # Compute chunk-level complexity (for code chunks)
            chunk_extra = {}
            chunk_cc = 1  # Default complexity
            chunk_is_complex = False
            if is_code:
                try:
                    chunk_stats = estimate_chunk_complexity(
                        chunk.content,
                        file_meta.language or ""
                    )
                    chunk_extra = {
                        "cc_estimate": chunk_stats["cc_estimate"],
                        "loc": chunk_stats["loc"],
                        "is_complex": chunk_stats["is_complex"],
                        "complexity_level": chunk_stats["complexity_level"],
                    }
                    chunk_cc = chunk_stats["cc_estimate"]
                    chunk_is_complex = chunk_stats["is_complex"]
                except Exception:
                    pass

            # Create chunk metadata
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                file_id=file_meta.file_id,
                profile=profile_name,
                chunk_index=chunk.chunk_index,
                offset_start=chunk.offset_start,
                offset_end=chunk.offset_end,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                kind=file_meta.kind,
                tags=tags,
                text_preview=chunk.text_preview,
                extra=chunk_extra,
            )
            chunk_metadatas.append(chunk_meta)

            # Prepare for vector store (include CC for filtering)
            chunk_dict = {
                "chunk_id": chunk_id,
                "content": chunk.content,
                "file_path": file_meta.path,
                "file_id": file_meta.file_id,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "kind": file_meta.kind,
                "chunk_index": chunk.chunk_index,
                "tags": tags,
            }
            # Add CC fields for ChromaDB filtering (flat fields required)
            if is_code:
                chunk_dict["cc_estimate"] = chunk_cc
                chunk_dict["is_complex"] = chunk_is_complex
            chunk_dicts.append(chunk_dict)

            # Add chunk to graph
            chunk_node = GraphNode.chunk_node(
                chunk_id=chunk_id,
                file_id=file_meta.file_id,
                file_path=file_meta.path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                kind=file_meta.kind,
            )
            self.graph.add_node(chunk_node)
            self.graph.add_edge(GraphEdge.contains(file_meta.file_id, chunk_id))

            # Add concept edges for tags
            for tag in tags:
                concept = self.graph.get_or_create_concept(tag)
                self.graph.add_edge(GraphEdge.mentions(chunk_id, concept.id))

        # Save chunk metadata
        self.metadata.append_chunks(chunk_metadatas)

        # Add to vector store (determines collection by kind)
        if is_code:
            self.vector_store.add_chunks(chunk_dicts, CollectionType.CODE)
        else:
            self.vector_store.add_chunks(chunk_dicts, CollectionType.DOCS)

        # Also add to mixed collection
        self.vector_store.add_chunks(chunk_dicts, CollectionType.MIXED)

        return chunk_metadatas

    def _calculate_index_size(self) -> int:
        """Calculate total index size in bytes."""
        total = 0
        rag_dir = self.project_root / ".RAG"

        if rag_dir.exists():
            for path in rag_dir.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size

        return total

    def _notify_progress(self) -> None:
        """Notify progress callback."""
        if self.on_progress:
            self.on_progress(self._progress)


# =============================================================================
# Module-Level Functions
# =============================================================================

# Global worker registry (per project)
_workers: Dict[str, IndexingWorker] = {}


def get_worker(project_root: Path) -> Optional[IndexingWorker]:
    """Get existing worker for a project."""
    key = str(project_root.resolve())
    return _workers.get(key)


def start_indexing(
    project_root: Path,
    full_reindex: bool = False,
    config: Optional[RAGConfig] = None,
    on_progress: Optional[Callable[[WorkerProgress], None]] = None,
    on_complete: Optional[Callable[[bool, Optional[str]], None]] = None,
) -> IndexingWorker:
    """
    Start indexing for a project.

    Args:
        project_root: Project root directory
        full_reindex: Clear and rebuild index from scratch
        config: RAG configuration
        on_progress: Progress callback
        on_complete: Completion callback

    Returns:
        IndexingWorker instance
    """
    key = str(project_root.resolve())

    # Check for existing worker
    existing = _workers.get(key)
    if existing and existing.is_running:
        logger.warning(f"Indexing already running for {project_root}")
        return existing

    # Create new worker
    worker = IndexingWorker(
        project_root=project_root,
        config=config,
        on_progress=on_progress,
        on_complete=on_complete,
    )

    _workers[key] = worker
    worker.start(full_reindex=full_reindex)

    return worker


def stop_indexing(project_root: Path, wait: bool = True) -> bool:
    """
    Stop indexing for a project.

    Args:
        project_root: Project root directory
        wait: Wait for worker to stop

    Returns:
        True if worker was stopped
    """
    key = str(project_root.resolve())
    worker = _workers.get(key)

    if worker and worker.is_running:
        worker.stop(wait=wait)
        return True

    return False


def get_indexing_progress(project_root: Path) -> Optional[WorkerProgress]:
    """
    Get current indexing progress.

    Args:
        project_root: Project root directory

    Returns:
        WorkerProgress or None if not indexing
    """
    worker = get_worker(project_root)
    if worker:
        return worker.progress

    # Try to load from state.json
    metadata = MetadataStore(project_root)
    state = metadata.load_state()

    if state:
        return WorkerProgress(
            status=state.status,
            files_total=state.files_total,
            files_processed=state.files_indexed,
            chunks_indexed=state.chunks_indexed,
            current_file=state.current_file,
            error=state.error,
            started_at=state.started_at,
        )

    return None


def is_indexing(project_root: Path) -> bool:
    """Check if project is currently being indexed."""
    worker = get_worker(project_root)
    return worker is not None and worker.is_running
