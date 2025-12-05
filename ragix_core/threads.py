"""
RAGIX Thread Management - Multi-conversation threads with persistence

Provides a Thread abstraction for managing multiple conversation threads within
a session. Each thread maintains its own message history and can be persisted
to disk for later restoration.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-05
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Default storage location for threads
DEFAULT_THREADS_DIR = ".ragix/threads"


@dataclass
class Message:
    """A single message in a thread."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**data)


@dataclass
class Thread:
    """
    A conversation thread with its own message history.

    Attributes:
        id: Unique thread identifier
        name: Human-readable thread name
        created_at: ISO timestamp of creation
        updated_at: ISO timestamp of last update
        messages: List of messages in this thread
        model: Model used for this thread (can override session default)
        metadata: Additional thread metadata (tags, notes, etc.)
    """
    id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    messages: List[Message] = field(default_factory=list)
    model: Optional[str] = None  # Override session model if set
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a message to this thread."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()
        return msg

    def get_messages_for_llm(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context (role, content only)."""
        msgs = self.messages[-limit:] if limit else self.messages
        return [{"role": m.role, "content": m.content} for m in msgs]

    def clear_messages(self, keep_last: int = 0) -> int:
        """Clear messages, optionally keeping the last N."""
        count = len(self.messages)
        if keep_last > 0:
            self.messages = self.messages[-keep_last:]
        else:
            self.messages = []
        self.updated_at = datetime.now().isoformat()
        return count - len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thread to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "model": self.model,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thread":
        """Deserialize thread from dictionary."""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            messages=messages,
            model=data.get("model"),
            metadata=data.get("metadata", {}),
        )

    def export_markdown(self) -> str:
        """Export thread as markdown."""
        lines = [
            f"# {self.name}",
            f"",
            f"**Thread ID:** {self.id}",
            f"**Created:** {self.created_at}",
            f"**Messages:** {len(self.messages)}",
            f"",
            "---",
            "",
        ]

        for msg in self.messages:
            role_label = msg.role.capitalize()
            lines.append(f"## {role_label}")
            lines.append(f"*{msg.timestamp}*")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def export_json(self, indent: int = 2) -> str:
        """Export thread as JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ThreadManager:
    """
    Manages multiple threads within a session with disk persistence.

    Threads are stored in .ragix/threads/{session_id}/ as JSON files.
    Each thread is a separate file named {thread_id}.json.
    """

    def __init__(self, session_id: str, storage_root: Optional[str] = None):
        """
        Initialize thread manager.

        Args:
            session_id: Session identifier for namespacing threads
            storage_root: Root directory for thread storage (default: cwd)
        """
        self.session_id = session_id
        self.storage_root = Path(storage_root) if storage_root else Path.cwd()
        self.threads_dir = self.storage_root / DEFAULT_THREADS_DIR / session_id
        self._threads: Dict[str, Thread] = {}
        self._active_thread_id: Optional[str] = None

        # Ensure storage directory exists
        self.threads_dir.mkdir(parents=True, exist_ok=True)

        # Load existing threads from disk
        self._load_threads()

    def _get_thread_path(self, thread_id: str) -> Path:
        """Get the file path for a thread."""
        return self.threads_dir / f"{thread_id}.json"

    def _load_threads(self) -> None:
        """Load all threads from disk."""
        if not self.threads_dir.exists():
            return

        for path in self.threads_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                thread = Thread.from_dict(data)
                self._threads[thread.id] = thread
                logger.debug(f"Loaded thread: {thread.id} ({thread.name})")
            except Exception as e:
                logger.warning(f"Failed to load thread from {path}: {e}")

        # Set active thread to most recently updated
        if self._threads:
            most_recent = max(self._threads.values(), key=lambda t: t.updated_at)
            self._active_thread_id = most_recent.id
            logger.debug(f"Active thread: {self._active_thread_id}")

    def _save_thread(self, thread: Thread) -> None:
        """Save a thread to disk."""
        path = self._get_thread_path(thread.id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(thread.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved thread: {thread.id}")
        except Exception as e:
            logger.error(f"Failed to save thread {thread.id}: {e}")
            raise

    def create_thread(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Thread:
        """
        Create a new thread.

        Args:
            name: Thread name (auto-generated if not provided)
            model: Model override for this thread
            metadata: Additional metadata

        Returns:
            The newly created Thread
        """
        thread_id = str(uuid.uuid4())[:8]  # Short readable ID
        if name is None:
            name = f"Thread {len(self._threads) + 1}"

        thread = Thread(
            id=thread_id,
            name=name,
            model=model,
            metadata=metadata or {},
        )

        self._threads[thread_id] = thread
        self._active_thread_id = thread_id
        self._save_thread(thread)

        logger.info(f"Created thread: {thread_id} ({name})")
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID."""
        return self._threads.get(thread_id)

    def get_active_thread(self) -> Optional[Thread]:
        """Get the currently active thread."""
        if self._active_thread_id:
            return self._threads.get(self._active_thread_id)
        return None

    def set_active_thread(self, thread_id: str) -> Optional[Thread]:
        """Set the active thread."""
        if thread_id in self._threads:
            self._active_thread_id = thread_id
            return self._threads[thread_id]
        return None

    def list_threads(self, sort_by: str = "updated_at", descending: bool = True) -> List[Thread]:
        """
        List all threads.

        Args:
            sort_by: Field to sort by (created_at, updated_at, name)
            descending: Sort in descending order
        """
        threads = list(self._threads.values())

        if sort_by == "name":
            threads.sort(key=lambda t: t.name.lower(), reverse=descending)
        elif sort_by == "created_at":
            threads.sort(key=lambda t: t.created_at, reverse=descending)
        else:  # default: updated_at
            threads.sort(key=lambda t: t.updated_at, reverse=descending)

        return threads

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread."""
        if thread_id not in self._threads:
            return False

        # Remove from memory
        del self._threads[thread_id]

        # Remove file
        path = self._get_thread_path(thread_id)
        if path.exists():
            path.unlink()

        # Update active thread if needed
        if self._active_thread_id == thread_id:
            self._active_thread_id = None
            if self._threads:
                most_recent = max(self._threads.values(), key=lambda t: t.updated_at)
                self._active_thread_id = most_recent.id

        logger.info(f"Deleted thread: {thread_id}")
        return True

    def rename_thread(self, thread_id: str, new_name: str) -> Optional[Thread]:
        """Rename a thread."""
        thread = self._threads.get(thread_id)
        if thread:
            thread.name = new_name
            thread.updated_at = datetime.now().isoformat()
            self._save_thread(thread)
        return thread

    def add_message(
        self,
        role: str,
        content: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Message]:
        """
        Add a message to a thread.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            thread_id: Thread ID (uses active thread if not specified)
            metadata: Additional message metadata

        Returns:
            The created Message, or None if no thread found
        """
        tid = thread_id or self._active_thread_id
        if not tid:
            # Auto-create default thread if none exists
            thread = self.create_thread(name="Main")
            tid = thread.id

        thread = self._threads.get(tid)
        if thread:
            msg = thread.add_message(role, content, metadata)
            self._save_thread(thread)
            return msg
        return None

    def get_messages(
        self,
        thread_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages from a thread."""
        tid = thread_id or self._active_thread_id
        if not tid:
            return []

        thread = self._threads.get(tid)
        if thread:
            return thread.messages[-limit:] if limit else thread.messages
        return []

    def clear_thread(self, thread_id: Optional[str] = None, keep_last: int = 0) -> int:
        """Clear messages from a thread."""
        tid = thread_id or self._active_thread_id
        if not tid:
            return 0

        thread = self._threads.get(tid)
        if thread:
            count = thread.clear_messages(keep_last)
            self._save_thread(thread)
            return count
        return 0

    def export_thread(
        self,
        thread_id: Optional[str] = None,
        format: str = "markdown",
    ) -> Optional[str]:
        """
        Export a thread.

        Args:
            thread_id: Thread ID (uses active thread if not specified)
            format: Export format ("markdown" or "json")

        Returns:
            Exported content as string, or None if thread not found
        """
        tid = thread_id or self._active_thread_id
        if not tid:
            return None

        thread = self._threads.get(tid)
        if not thread:
            return None

        if format == "json":
            return thread.export_json()
        else:
            return thread.export_markdown()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all threads."""
        threads = self.list_threads()
        total_messages = sum(len(t.messages) for t in threads)

        return {
            "session_id": self.session_id,
            "thread_count": len(threads),
            "total_messages": total_messages,
            "active_thread_id": self._active_thread_id,
            "storage_path": str(self.threads_dir),
        }


# Global thread managers per session
_thread_managers: Dict[str, ThreadManager] = {}


def get_thread_manager(session_id: str, storage_root: Optional[str] = None) -> ThreadManager:
    """
    Get or create a ThreadManager for a session.

    Args:
        session_id: Session identifier
        storage_root: Root directory for storage (default: cwd)

    Returns:
        ThreadManager instance for the session
    """
    key = f"{storage_root or '.'}:{session_id}"
    if key not in _thread_managers:
        _thread_managers[key] = ThreadManager(session_id, storage_root)
    return _thread_managers[key]


def clear_thread_managers() -> None:
    """Clear all cached thread managers (for testing)."""
    _thread_managers.clear()
