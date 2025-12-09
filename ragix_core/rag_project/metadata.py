"""
RAGIX Project RAG - Metadata Storage

Manages .RAG/metadata/ directory with:
    - files.jsonl: Per-file metadata (path, kind, hash, mtime)
    - chunks.jsonl: Per-chunk metadata (file_id, line_start, line_end, tags)
    - state.json: Indexing state and progress

All metadata uses JSONL (JSON Lines) for efficient append-only operations
and streaming reads.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Set
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

METADATA_DIR = "metadata"
FILES_JSONL = "files.jsonl"
CHUNKS_JSONL = "chunks.jsonl"
STATE_JSON = "state.json"


class FileKind(str, Enum):
    """File type classification."""
    # Code
    CODE_JAVA = "code_java"
    CODE_PYTHON = "code_python"
    CODE_JAVASCRIPT = "code_javascript"
    CODE_TYPESCRIPT = "code_typescript"
    CODE_C = "code_c"
    CODE_CPP = "code_cpp"
    CODE_GO = "code_go"
    CODE_RUST = "code_rust"
    CODE_RUBY = "code_ruby"
    CODE_PHP = "code_php"
    CODE_KOTLIN = "code_kotlin"
    CODE_SCALA = "code_scala"
    CODE_SWIFT = "code_swift"
    CODE_SHELL = "code_shell"
    CODE_SQL = "code_sql"
    CODE_OTHER = "code_other"

    # Config
    CONFIG_XML = "config_xml"
    CONFIG_JSON = "config_json"
    CONFIG_YAML = "config_yaml"
    CONFIG_PROPERTIES = "config_properties"
    CONFIG_OTHER = "config_other"

    # Build
    BUILD_POM = "build_pom"
    BUILD_GRADLE = "build_gradle"
    BUILD_MAKEFILE = "build_makefile"
    BUILD_CMAKE = "build_cmake"
    BUILD_OTHER = "build_other"

    # Documentation
    DOC_MARKDOWN = "doc_markdown"
    DOC_TEXT = "doc_text"
    DOC_RST = "doc_rst"
    DOC_PDF = "doc_pdf"
    DOC_DOCX = "doc_docx"
    DOC_PPTX = "doc_pptx"
    DOC_XLSX = "doc_xlsx"
    DOC_ODT = "doc_odt"
    DOC_OTHER = "doc_other"

    # Data
    DATA_CSV = "data_csv"
    DATA_XSD = "data_xsd"
    DATA_OTHER = "data_other"

    # Unknown
    UNKNOWN = "unknown"


class IndexingStatus(str, Enum):
    """Indexing state status."""
    IDLE = "idle"
    SCANNING = "scanning"
    INDEXING = "indexing"
    UPDATING = "updating"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


# =============================================================================
# File Metadata
# =============================================================================

@dataclass
class FileMetadata:
    """
    Metadata for a single indexed file.

    Stored in files.jsonl, one JSON object per line.
    """
    file_id: str                    # Unique ID (e.g., "F000123")
    path: str                       # Relative path from project root
    kind: str                       # FileKind value
    language: Optional[str] = None  # Programming language or doc language
    hash: Optional[str] = None      # SHA-256 hash for change detection
    mtime: Optional[str] = None     # ISO format modification time
    size_bytes: int = 0
    profile: Optional[str] = None   # Profile used for indexing
    chunk_count: int = 0            # Number of chunks from this file
    indexed_at: Optional[str] = None  # When this file was indexed

    # Optional extra metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileMetadata":
        # Handle extra field
        extra = data.pop("extra", {})
        return cls(**data, extra=extra)

    @classmethod
    def from_json(cls, line: str) -> "FileMetadata":
        return cls.from_dict(json.loads(line))

    @staticmethod
    def compute_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash {file_path}: {e}")
            return ""

    @staticmethod
    def generate_id(index: int) -> str:
        """Generate a file ID."""
        return f"F{index:06d}"


# =============================================================================
# Chunk Metadata
# =============================================================================

@dataclass
class ChunkMetadata:
    """
    Metadata for a single chunk.

    Stored in chunks.jsonl, one JSON object per line.
    Critical for future AST integration: line_start/line_end enable
    code → doc pairing with line-level citations.
    """
    chunk_id: str                   # Unique ID (e.g., "C000987")
    file_id: str                    # Reference to parent file
    profile: str                    # Profile used for chunking
    chunk_index: int = 0            # Index within file

    # Position in original file (critical for citations)
    offset_start: int = 0           # Character offset start
    offset_end: int = 0             # Character offset end
    line_start: int = 0             # Line number start (1-indexed)
    line_end: int = 0               # Line number end (1-indexed)

    # Classification
    kind: str = "unknown"           # Same taxonomy as FileKind
    tags: List[str] = field(default_factory=list)

    # Vector store reference
    embedding_id: Optional[str] = None

    # Preview for UI display
    text_preview: str = ""          # First ~200 chars

    # Extra metadata (section_title, slide_number, sheet_name, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        tags = data.pop("tags", [])
        extra = data.pop("extra", {})
        return cls(**data, tags=tags, extra=extra)

    @classmethod
    def from_json(cls, line: str) -> "ChunkMetadata":
        return cls.from_dict(json.loads(line))

    @staticmethod
    def generate_id(index: int) -> str:
        """Generate a chunk ID."""
        return f"C{index:06d}"

    def get_citation(self, file_path: str) -> str:
        """Get a citation string for this chunk."""
        if self.line_start and self.line_end:
            if self.line_start == self.line_end:
                return f"{file_path}:{self.line_start}"
            return f"{file_path}:{self.line_start}-{self.line_end}"
        return file_path


# =============================================================================
# Indexing State
# =============================================================================

@dataclass
class IndexingState:
    """
    Current indexing state and progress.

    Stored in state.json, updated during indexing.
    Used by UI to show progress bar and stats.
    """
    status: str = IndexingStatus.IDLE.value
    active_profile: Optional[str] = None

    # Progress counters
    files_total: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0

    chunks_total: int = 0
    chunks_indexed: int = 0

    # Index stats
    index_size_bytes: int = 0

    # Timing
    started_at: Optional[str] = None
    last_update: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0

    # Error tracking
    error: Optional[str] = None
    failed_files: List[str] = field(default_factory=list)

    # Current file being processed (for progress display)
    current_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexingState":
        failed_files = data.pop("failed_files", [])
        return cls(**data, failed_files=failed_files)

    @classmethod
    def from_json(cls, content: str) -> "IndexingState":
        return cls.from_dict(json.loads(content))

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.files_total == 0:
            return 0.0
        return (self.files_indexed / self.files_total) * 100

    @property
    def is_running(self) -> bool:
        """Check if indexing is currently running."""
        return self.status in (
            IndexingStatus.SCANNING.value,
            IndexingStatus.INDEXING.value,
            IndexingStatus.UPDATING.value,
        )

    def start(self, profile: str, files_total: int) -> None:
        """Mark indexing as started."""
        self.status = IndexingStatus.INDEXING.value
        self.active_profile = profile
        self.files_total = files_total
        self.files_indexed = 0
        self.files_skipped = 0
        self.files_failed = 0
        self.chunks_total = 0
        self.chunks_indexed = 0
        self.started_at = datetime.now().isoformat()
        self.last_update = self.started_at
        self.completed_at = None
        self.error = None
        self.failed_files = []

    def update_progress(
        self,
        files_indexed: int,
        chunks_indexed: int,
        current_file: Optional[str] = None,
    ) -> None:
        """Update progress counters."""
        self.files_indexed = files_indexed
        self.chunks_indexed = chunks_indexed
        self.chunks_total = chunks_indexed  # Update total as we go
        self.current_file = current_file
        self.last_update = datetime.now().isoformat()

    def complete(self, index_size_bytes: int = 0) -> None:
        """Mark indexing as completed."""
        self.status = IndexingStatus.COMPLETED.value
        self.completed_at = datetime.now().isoformat()
        self.index_size_bytes = index_size_bytes
        self.current_file = None

        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_seconds = (end - start).total_seconds()

    def fail(self, error: str) -> None:
        """Mark indexing as failed."""
        self.status = IndexingStatus.ERROR.value
        self.error = error
        self.completed_at = datetime.now().isoformat()
        self.current_file = None

    def cancel(self) -> None:
        """Mark indexing as cancelled."""
        self.status = IndexingStatus.CANCELLED.value
        self.completed_at = datetime.now().isoformat()
        self.current_file = None


# =============================================================================
# Metadata Store
# =============================================================================

class MetadataStore:
    """
    Manages .RAG/metadata/ storage.

    Provides read/write access to files.jsonl, chunks.jsonl, and state.json.
    Uses append-only JSONL for files and chunks (efficient for large indices).
    """

    def __init__(self, project_root: Path):
        """
        Initialize metadata store.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.rag_dir = project_root / ".RAG"
        self.metadata_dir = self.rag_dir / METADATA_DIR

        # File paths
        self.files_path = self.metadata_dir / FILES_JSONL
        self.chunks_path = self.metadata_dir / CHUNKS_JSONL
        self.state_path = self.metadata_dir / STATE_JSON

        # In-memory indices (loaded on demand)
        self._files_by_id: Optional[Dict[str, FileMetadata]] = None
        self._files_by_path: Optional[Dict[str, FileMetadata]] = None
        self._chunks_by_id: Optional[Dict[str, ChunkMetadata]] = None
        self._file_id_counter: int = 0
        self._chunk_id_counter: int = 0

    def ensure_dirs(self) -> None:
        """Create metadata directory if needed."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def load_state(self) -> IndexingState:
        """Load current indexing state."""
        if not self.state_path.exists():
            return IndexingState()

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return IndexingState.from_json(f.read())
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return IndexingState()

    def save_state(self, state: IndexingState) -> None:
        """Save indexing state."""
        self.ensure_dirs()
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                f.write(state.to_json())
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    # -------------------------------------------------------------------------
    # File Metadata
    # -------------------------------------------------------------------------

    def iter_files(self) -> Iterator[FileMetadata]:
        """Iterate over all file metadata (streaming read)."""
        if not self.files_path.exists():
            return

        with open(self.files_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield FileMetadata.from_json(line)
                    except Exception as e:
                        logger.warning(f"Invalid file metadata line: {e}")

    def load_files(self) -> Dict[str, FileMetadata]:
        """Load all file metadata into memory (by file_id)."""
        if self._files_by_id is None:
            self._files_by_id = {}
            self._files_by_path = {}
            for fm in self.iter_files():
                self._files_by_id[fm.file_id] = fm
                self._files_by_path[fm.path] = fm
                # Track max ID for counter
                try:
                    num = int(fm.file_id[1:])
                    self._file_id_counter = max(self._file_id_counter, num + 1)
                except ValueError:
                    pass

        return self._files_by_id

    def get_file_by_id(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata by ID."""
        files = self.load_files()
        return files.get(file_id)

    def get_file_by_path(self, path: str) -> Optional[FileMetadata]:
        """Get file metadata by relative path."""
        self.load_files()
        return self._files_by_path.get(path) if self._files_by_path else None

    def append_file(self, file_meta: FileMetadata) -> None:
        """Append a file metadata entry."""
        self.ensure_dirs()
        with open(self.files_path, "a", encoding="utf-8") as f:
            f.write(file_meta.to_json() + "\n")

        # Update in-memory cache if loaded
        if self._files_by_id is not None:
            self._files_by_id[file_meta.file_id] = file_meta
        if self._files_by_path is not None:
            self._files_by_path[file_meta.path] = file_meta

    def next_file_id(self) -> str:
        """Generate next file ID."""
        self.load_files()  # Ensure counter is initialized
        file_id = FileMetadata.generate_id(self._file_id_counter)
        self._file_id_counter += 1
        return file_id

    def clear_files(self) -> None:
        """Clear all file metadata."""
        if self.files_path.exists():
            self.files_path.unlink()
        self._files_by_id = None
        self._files_by_path = None
        self._file_id_counter = 0

    # -------------------------------------------------------------------------
    # Chunk Metadata
    # -------------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[ChunkMetadata]:
        """Iterate over all chunk metadata (streaming read)."""
        if not self.chunks_path.exists():
            return

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield ChunkMetadata.from_json(line)
                    except Exception as e:
                        logger.warning(f"Invalid chunk metadata line: {e}")

    def iter_chunks_for_file(self, file_id: str) -> Iterator[ChunkMetadata]:
        """Iterate over chunks belonging to a specific file."""
        for chunk in self.iter_chunks():
            if chunk.file_id == file_id:
                yield chunk

    def load_chunks(self) -> Dict[str, ChunkMetadata]:
        """Load all chunk metadata into memory."""
        if self._chunks_by_id is None:
            self._chunks_by_id = {}
            for cm in self.iter_chunks():
                self._chunks_by_id[cm.chunk_id] = cm
                try:
                    num = int(cm.chunk_id[1:])
                    self._chunk_id_counter = max(self._chunk_id_counter, num + 1)
                except ValueError:
                    pass

        return self._chunks_by_id

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get chunk metadata by ID."""
        chunks = self.load_chunks()
        return chunks.get(chunk_id)

    def append_chunk(self, chunk_meta: ChunkMetadata) -> None:
        """Append a chunk metadata entry."""
        self.ensure_dirs()
        with open(self.chunks_path, "a", encoding="utf-8") as f:
            f.write(chunk_meta.to_json() + "\n")

        if self._chunks_by_id is not None:
            self._chunks_by_id[chunk_meta.chunk_id] = chunk_meta

    def append_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Append multiple chunk metadata entries (batch)."""
        self.ensure_dirs()
        with open(self.chunks_path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.to_json() + "\n")

                if self._chunks_by_id is not None:
                    self._chunks_by_id[chunk.chunk_id] = chunk

    def next_chunk_id(self) -> str:
        """Generate next chunk ID."""
        self.load_chunks()
        chunk_id = ChunkMetadata.generate_id(self._chunk_id_counter)
        self._chunk_id_counter += 1
        return chunk_id

    def clear_chunks(self) -> None:
        """Clear all chunk metadata."""
        if self.chunks_path.exists():
            self.chunks_path.unlink()
        self._chunks_by_id = None
        self._chunk_id_counter = 0

    # -------------------------------------------------------------------------
    # Change Detection
    # -------------------------------------------------------------------------

    def detect_changes(self) -> Dict[str, List[str]]:
        """
        Detect file changes since last indexing.

        Returns:
            Dict with keys: 'new', 'modified', 'deleted'
        """
        changes = {
            "new": [],
            "modified": [],
            "deleted": [],
        }

        # Load existing file metadata
        indexed_files = self.load_files()
        indexed_paths = set(self._files_by_path.keys()) if self._files_by_path else set()

        # This will be called by ingest.py which scans the actual files
        # Here we just provide the infrastructure

        return changes

    def get_indexed_paths(self) -> Set[str]:
        """Get set of all indexed file paths."""
        self.load_files()
        return set(self._files_by_path.keys()) if self._files_by_path else set()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get metadata statistics."""
        files = self.load_files()
        chunks = self.load_chunks()

        # Count by kind
        files_by_kind: Dict[str, int] = {}
        for fm in files.values():
            kind = fm.kind
            files_by_kind[kind] = files_by_kind.get(kind, 0) + 1

        return {
            "total_files": len(files),
            "total_chunks": len(chunks),
            "files_by_kind": files_by_kind,
            "metadata_size_bytes": self._get_metadata_size(),
        }

    def _get_metadata_size(self) -> int:
        """Get total size of metadata files."""
        total = 0
        for path in [self.files_path, self.chunks_path, self.state_path]:
            if path.exists():
                total += path.stat().st_size
        return total

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all metadata (full reset)."""
        self.clear_files()
        self.clear_chunks()
        if self.state_path.exists():
            self.state_path.unlink()
