"""
SWE Workflows - Chunked execution with checkpoints and resumption for large codebases

Provides robust workflow execution for Software Engineering tasks:
- Chunked processing of large file sets
- Checkpoint-based resumption
- Circuit breaker integration
- Progress tracking and estimation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from .monitoring import CircuitBreaker, get_metrics
from .resilience import RetryConfig, BackoffStrategy, retry

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """States of a chunked workflow."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkState(str, Enum):
    """States of individual chunks."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    chunk_id: str
    state: ChunkState
    items_processed: int = 0
    items_failed: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowCheckpoint:
    """
    Checkpoint for workflow resumption.

    Saved to disk to enable resumption after interruption.
    """
    workflow_id: str
    workflow_type: str
    state: WorkflowState
    created_at: str
    updated_at: str
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    current_chunk_id: Optional[str]
    chunk_results: Dict[str, Dict[str, Any]]  # chunk_id -> ChunkResult as dict
    context: Dict[str, Any]  # Workflow-specific context
    config: Dict[str, Any]  # Workflow configuration

    def to_json(self) -> str:
        """Serialize checkpoint to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowCheckpoint":
        """Deserialize checkpoint from JSON."""
        data = json.loads(json_str)
        data["state"] = WorkflowState(data["state"])
        return cls(**data)

    def save(self, checkpoint_dir: Path) -> Path:
        """
        Save checkpoint to disk.

        Args:
            checkpoint_dir: Directory for checkpoints

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{self.workflow_id}.checkpoint.json"

        with open(checkpoint_path, 'w') as f:
            f.write(self.to_json())

        logger.debug(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    @classmethod
    def load(cls, checkpoint_path: Path) -> Optional["WorkflowCheckpoint"]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            WorkflowCheckpoint or None
        """
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                return cls.from_json(f.read())
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


@dataclass
class ChunkConfig:
    """Configuration for chunk processing."""
    chunk_size: int = 50  # Items per chunk
    max_parallel: int = 1  # Parallel chunk processing (1 = sequential)
    timeout_per_item: float = 30.0  # Seconds per item
    retry_failed_chunks: bool = True
    max_chunk_retries: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    save_checkpoint_every: int = 1  # Save after N chunks


class ChunkedWorkflow:
    """
    Base class for chunked workflow execution.

    Provides:
    - Automatic chunking of work items
    - Checkpoint-based resumption
    - Circuit breaker protection
    - Progress tracking
    """

    def __init__(
        self,
        workflow_id: str,
        workflow_type: str,
        config: Optional[ChunkConfig] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize chunked workflow.

        Args:
            workflow_id: Unique identifier for this workflow run
            workflow_type: Type of workflow (e.g., "code_review", "migration")
            config: Chunk configuration
            checkpoint_dir: Directory for checkpoints
        """
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.config = config or ChunkConfig()
        self.checkpoint_dir = checkpoint_dir or Path(".ragix/checkpoints")

        self.state = WorkflowState.PENDING
        self.chunks: List[List[Any]] = []
        self.chunk_results: Dict[str, ChunkResult] = {}
        self.context: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Circuit breaker for failure protection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout,
        )

        # Metrics
        self.metrics = get_metrics()

    def _generate_chunk_id(self, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{self.workflow_id}_chunk_{chunk_index:04d}"

    def create_chunks(self, items: List[Any]) -> List[List[Any]]:
        """
        Split items into chunks.

        Args:
            items: List of items to process

        Returns:
            List of chunks
        """
        chunks = []
        for i in range(0, len(items), self.config.chunk_size):
            chunks.append(items[i:i + self.config.chunk_size])

        self.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks from {len(items)} items")
        return chunks

    def process_item(self, item: Any, context: Dict[str, Any]) -> Any:
        """
        Process a single item. Override in subclasses.

        Args:
            item: Item to process
            context: Workflow context

        Returns:
            Processing result
        """
        raise NotImplementedError("Subclasses must implement process_item")

    def process_chunk(
        self,
        chunk_id: str,
        items: List[Any],
        context: Dict[str, Any]
    ) -> ChunkResult:
        """
        Process a single chunk.

        Args:
            chunk_id: Unique chunk identifier
            items: Items in this chunk
            context: Workflow context

        Returns:
            ChunkResult
        """
        start_time = time.time()
        processed = 0
        failed = 0
        outputs = []
        errors = []

        for item in items:
            # Check circuit breaker
            if not self.circuit_breaker.is_allowed():
                logger.warning(f"Circuit breaker open, skipping remaining items in chunk {chunk_id}")
                break

            try:
                result = self.process_item(item, context)
                outputs.append(result)
                processed += 1
                self.circuit_breaker.record_success()

            except Exception as e:
                failed += 1
                errors.append(str(e))
                self.circuit_breaker.record_failure()
                logger.warning(f"Failed to process item in chunk {chunk_id}: {e}")

        duration = time.time() - start_time

        state = ChunkState.COMPLETED
        if failed > 0:
            state = ChunkState.FAILED if failed == len(items) else ChunkState.COMPLETED

        return ChunkResult(
            chunk_id=chunk_id,
            state=state,
            items_processed=processed,
            items_failed=failed,
            duration_seconds=duration,
            error="; ".join(errors) if errors else None,
            output=outputs,
        )

    def create_checkpoint(self) -> WorkflowCheckpoint:
        """Create current checkpoint."""
        now = datetime.now().isoformat()

        return WorkflowCheckpoint(
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
            state=self.state,
            created_at=self.start_time and datetime.fromtimestamp(self.start_time).isoformat() or now,
            updated_at=now,
            total_chunks=len(self.chunks),
            completed_chunks=sum(1 for r in self.chunk_results.values() if r.state == ChunkState.COMPLETED),
            failed_chunks=sum(1 for r in self.chunk_results.values() if r.state == ChunkState.FAILED),
            current_chunk_id=None,
            chunk_results={
                k: asdict(v) for k, v in self.chunk_results.items()
            },
            context=self.context,
            config=asdict(self.config),
        )

    def save_checkpoint(self) -> Path:
        """Save current checkpoint to disk."""
        checkpoint = self.create_checkpoint()
        return checkpoint.save(self.checkpoint_dir)

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint from disk.

        Returns:
            True if checkpoint loaded
        """
        checkpoint_path = self.checkpoint_dir / f"{self.workflow_id}.checkpoint.json"
        checkpoint = WorkflowCheckpoint.load(checkpoint_path)

        if not checkpoint:
            return False

        # Restore state
        self.state = checkpoint.state
        self.context = checkpoint.context

        # Restore chunk results
        for chunk_id, result_dict in checkpoint.chunk_results.items():
            result_dict["state"] = ChunkState(result_dict["state"])
            self.chunk_results[chunk_id] = ChunkResult(**result_dict)

        logger.info(
            f"Loaded checkpoint: {checkpoint.completed_chunks}/{checkpoint.total_chunks} chunks complete"
        )
        return True

    def get_pending_chunks(self) -> List[Tuple[int, str, List[Any]]]:
        """
        Get chunks that still need processing.

        Returns:
            List of (index, chunk_id, items) tuples
        """
        pending = []

        for i, chunk in enumerate(self.chunks):
            chunk_id = self._generate_chunk_id(i)

            # Check if already completed
            if chunk_id in self.chunk_results:
                result = self.chunk_results[chunk_id]
                if result.state == ChunkState.COMPLETED:
                    continue
                if result.state == ChunkState.FAILED and not self.config.retry_failed_chunks:
                    continue

            pending.append((i, chunk_id, chunk))

        return pending

    def run(
        self,
        items: List[Any],
        context: Optional[Dict[str, Any]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the chunked workflow.

        Args:
            items: Items to process
            context: Initial context
            resume: Try to resume from checkpoint

        Returns:
            Workflow results
        """
        self.start_time = time.time()
        self.context = context or {}
        self.state = WorkflowState.RUNNING

        # Try to resume
        if resume:
            self.load_checkpoint()

        # Create chunks if not already done
        if not self.chunks:
            self.create_chunks(items)

        # Process pending chunks
        pending = self.get_pending_chunks()
        total_pending = len(pending)

        logger.info(f"Processing {total_pending} pending chunks")

        chunks_since_checkpoint = 0

        for i, (chunk_index, chunk_id, chunk_items) in enumerate(pending):
            # Check if workflow was cancelled
            if self.state == WorkflowState.CANCELLED:
                break

            # Check circuit breaker state
            if self.circuit_breaker.state == "open":
                logger.warning("Circuit breaker is open, pausing workflow")
                self.state = WorkflowState.PAUSED
                self.save_checkpoint()
                break

            logger.info(f"Processing chunk {i+1}/{total_pending} ({chunk_id})")

            # Process chunk with retry
            result = self._process_chunk_with_retry(chunk_id, chunk_items)
            self.chunk_results[chunk_id] = result

            # Update metrics
            self.metrics.increment("swe_chunks_processed")
            self.metrics.observe("swe_chunk_duration", result.duration_seconds)

            # Save checkpoint periodically
            chunks_since_checkpoint += 1
            if chunks_since_checkpoint >= self.config.save_checkpoint_every:
                self.save_checkpoint()
                chunks_since_checkpoint = 0

        # Final state
        self.end_time = time.time()

        if self.state == WorkflowState.RUNNING:
            # Check if all completed
            completed = sum(1 for r in self.chunk_results.values() if r.state == ChunkState.COMPLETED)
            failed = sum(1 for r in self.chunk_results.values() if r.state == ChunkState.FAILED)

            if completed == len(self.chunks):
                self.state = WorkflowState.COMPLETED
            elif failed > 0:
                self.state = WorkflowState.FAILED
            else:
                self.state = WorkflowState.COMPLETED

        # Save final checkpoint
        self.save_checkpoint()

        return self.get_summary()

    def _process_chunk_with_retry(
        self,
        chunk_id: str,
        items: List[Any]
    ) -> ChunkResult:
        """Process chunk with retry logic."""
        retries = 0
        last_result = None

        while retries <= self.config.max_chunk_retries:
            result = self.process_chunk(chunk_id, items, self.context)

            if result.state == ChunkState.COMPLETED:
                return result

            last_result = result
            retries += 1

            if retries <= self.config.max_chunk_retries:
                logger.info(f"Retrying chunk {chunk_id} (attempt {retries + 1})")
                time.sleep(2 ** retries)  # Exponential backoff

        return last_result or ChunkResult(
            chunk_id=chunk_id,
            state=ChunkState.FAILED,
            error="Max retries exceeded",
        )

    def cancel(self):
        """Cancel the workflow."""
        self.state = WorkflowState.CANCELLED
        logger.info(f"Workflow {self.workflow_id} cancelled")

    def pause(self):
        """Pause the workflow."""
        self.state = WorkflowState.PAUSED
        self.save_checkpoint()
        logger.info(f"Workflow {self.workflow_id} paused")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        completed = sum(1 for r in self.chunk_results.values() if r.state == ChunkState.COMPLETED)
        failed = sum(1 for r in self.chunk_results.values() if r.state == ChunkState.FAILED)
        total = len(self.chunks)

        items_processed = sum(r.items_processed for r in self.chunk_results.values())
        items_failed = sum(r.items_failed for r in self.chunk_results.values())

        elapsed = 0.0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time

        # Estimate remaining time
        eta_seconds = None
        if completed > 0 and total > completed:
            avg_chunk_time = elapsed / completed
            eta_seconds = avg_chunk_time * (total - completed)

        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "total_chunks": total,
            "completed_chunks": completed,
            "failed_chunks": failed,
            "pending_chunks": total - completed - failed,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "items_processed": items_processed,
            "items_failed": items_failed,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "circuit_breaker_state": self.circuit_breaker.state,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        progress = self.get_progress()

        # Collect all outputs
        outputs = []
        for result in self.chunk_results.values():
            if result.output:
                outputs.extend(result.output)

        # Collect all errors
        errors = []
        for result in self.chunk_results.values():
            if result.error:
                errors.append({
                    "chunk_id": result.chunk_id,
                    "error": result.error,
                })

        return {
            **progress,
            "outputs": outputs,
            "errors": errors,
        }


class FileProcessingWorkflow(ChunkedWorkflow):
    """
    Workflow for processing files in large codebases.

    Handles:
    - File discovery and filtering
    - Chunked file processing
    - Results aggregation
    """

    def __init__(
        self,
        workflow_id: str,
        root_path: Path,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        config: Optional[ChunkConfig] = None,
    ):
        """
        Initialize file processing workflow.

        Args:
            workflow_id: Unique workflow ID
            root_path: Root directory for file discovery
            file_patterns: Glob patterns to include (e.g., ["*.py", "*.js"])
            exclude_patterns: Glob patterns to exclude
            config: Chunk configuration
        """
        super().__init__(workflow_id, "file_processing", config)

        self.root_path = Path(root_path)
        self.file_patterns = file_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or [
            "__pycache__/*",
            ".git/*",
            "node_modules/*",
            "*.pyc",
            ".venv/*",
        ]

    def discover_files(self) -> List[Path]:
        """
        Discover files matching patterns.

        Returns:
            List of file paths
        """
        files = set()

        for pattern in self.file_patterns:
            for file_path in self.root_path.rglob(pattern):
                if file_path.is_file():
                    files.add(file_path)

        # Apply exclusions
        excluded = set()
        for pattern in self.exclude_patterns:
            for file_path in self.root_path.rglob(pattern):
                excluded.add(file_path)

        files -= excluded

        # Sort for deterministic order
        return sorted(files)

    def process_file(self, file_path: Path, context: Dict[str, Any]) -> Any:
        """
        Process a single file. Override in subclasses.

        Args:
            file_path: Path to file
            context: Workflow context

        Returns:
            Processing result
        """
        raise NotImplementedError("Subclasses must implement process_file")

    def process_item(self, item: Any, context: Dict[str, Any]) -> Any:
        """Process item (delegates to process_file)."""
        return self.process_file(Path(item), context)

    def run_on_files(
        self,
        context: Optional[Dict[str, Any]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run workflow on discovered files.

        Args:
            context: Initial context
            resume: Try to resume from checkpoint

        Returns:
            Workflow results
        """
        # Discover files
        files = self.discover_files()
        logger.info(f"Discovered {len(files)} files to process")

        # Convert to strings for serialization
        file_strings = [str(f) for f in files]

        return self.run(file_strings, context, resume)


class CodeReviewWorkflow(FileProcessingWorkflow):
    """
    Workflow for chunked code review of large codebases.

    Processes files in chunks, collecting review findings.
    """

    def __init__(
        self,
        workflow_id: str,
        root_path: Path,
        review_func: Callable[[Path, Dict], Dict],
        file_patterns: Optional[List[str]] = None,
        config: Optional[ChunkConfig] = None,
    ):
        """
        Initialize code review workflow.

        Args:
            workflow_id: Unique workflow ID
            root_path: Root directory
            review_func: Function to review a single file
            file_patterns: File patterns to review
            config: Chunk configuration
        """
        super().__init__(
            workflow_id,
            root_path,
            file_patterns or ["*.py", "*.js", "*.ts", "*.java"],
            config=config,
        )
        self.workflow_type = "code_review"
        self.review_func = review_func

    def process_file(self, file_path: Path, context: Dict[str, Any]) -> Dict:
        """Review a single file."""
        return self.review_func(file_path, context)


class MigrationWorkflow(FileProcessingWorkflow):
    """
    Workflow for code migration across large codebases.

    Applies transformations to files in chunks with rollback support.
    """

    def __init__(
        self,
        workflow_id: str,
        root_path: Path,
        transform_func: Callable[[Path, Dict], Optional[str]],
        file_patterns: Optional[List[str]] = None,
        config: Optional[ChunkConfig] = None,
        dry_run: bool = True,
    ):
        """
        Initialize migration workflow.

        Args:
            workflow_id: Unique workflow ID
            root_path: Root directory
            transform_func: Function to transform file content
            file_patterns: File patterns to migrate
            config: Chunk configuration
            dry_run: If True, don't write changes
        """
        super().__init__(workflow_id, root_path, file_patterns, config=config)
        self.workflow_type = "migration"
        self.transform_func = transform_func
        self.dry_run = dry_run
        self.backups: Dict[str, str] = {}  # path -> original content

    def process_file(self, file_path: Path, context: Dict[str, Any]) -> Dict:
        """Transform a single file."""
        try:
            # Read original
            with open(file_path, 'r') as f:
                original = f.read()

            # Transform
            transformed = self.transform_func(file_path, context)

            if transformed is None or transformed == original:
                return {
                    "path": str(file_path),
                    "changed": False,
                }

            # Backup
            self.backups[str(file_path)] = original

            # Write if not dry run
            if not self.dry_run:
                with open(file_path, 'w') as f:
                    f.write(transformed)

            return {
                "path": str(file_path),
                "changed": True,
                "dry_run": self.dry_run,
            }

        except Exception as e:
            return {
                "path": str(file_path),
                "error": str(e),
            }

    def rollback(self) -> int:
        """
        Rollback all changes.

        Returns:
            Number of files rolled back
        """
        count = 0
        for path, content in self.backups.items():
            try:
                with open(path, 'w') as f:
                    f.write(content)
                count += 1
            except Exception as e:
                logger.error(f"Failed to rollback {path}: {e}")

        logger.info(f"Rolled back {count} files")
        return count


def list_checkpoints(checkpoint_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    List available workflow checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint summaries
    """
    checkpoint_dir = checkpoint_dir or Path(".ragix/checkpoints")

    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    for cp_file in checkpoint_dir.glob("*.checkpoint.json"):
        try:
            checkpoint = WorkflowCheckpoint.load(cp_file)
            if checkpoint:
                checkpoints.append({
                    "workflow_id": checkpoint.workflow_id,
                    "workflow_type": checkpoint.workflow_type,
                    "state": checkpoint.state.value,
                    "progress": f"{checkpoint.completed_chunks}/{checkpoint.total_chunks}",
                    "updated_at": checkpoint.updated_at,
                    "file": str(cp_file),
                })
        except Exception as e:
            logger.debug(f"Failed to read checkpoint {cp_file}: {e}")

    return sorted(checkpoints, key=lambda x: x["updated_at"], reverse=True)


def resume_workflow(
    workflow_id: str,
    workflow_class: type,
    checkpoint_dir: Optional[Path] = None,
    **kwargs
) -> Optional[ChunkedWorkflow]:
    """
    Resume a workflow from checkpoint.

    Args:
        workflow_id: ID of workflow to resume
        workflow_class: Class to instantiate
        checkpoint_dir: Directory containing checkpoints
        **kwargs: Additional arguments for workflow constructor

    Returns:
        Resumed workflow or None
    """
    checkpoint_dir = checkpoint_dir or Path(".ragix/checkpoints")
    checkpoint_path = checkpoint_dir / f"{workflow_id}.checkpoint.json"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {workflow_id}")
        return None

    # Create workflow instance
    workflow = workflow_class(workflow_id=workflow_id, **kwargs)

    # Load checkpoint
    if workflow.load_checkpoint():
        logger.info(f"Resumed workflow: {workflow_id}")
        return workflow

    return None
