"""
RAGIX Log Integrity and Hashing
================================

Provides SHA256 signing for log files to ensure tamper-evidence
and compliance with audit requirements.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LogEntry:
    """A single log entry with hash."""
    timestamp: str
    sequence: int
    content: str
    hash: str
    prev_hash: str


@dataclass
class LogIntegrityReport:
    """Report from log integrity verification."""
    valid: bool
    total_entries: int
    verified_entries: int
    first_invalid_entry: Optional[int]
    errors: List[str]
    log_file: str
    verification_time: str


# =============================================================================
# Hash Functions
# =============================================================================

def compute_hash(content: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of content.

    Args:
        content: String content to hash
        algorithm: Hash algorithm (sha256, sha512, md5)

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "sha256":
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(content.encode('utf-8')).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of entire file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


# =============================================================================
# Chained Log Hasher
# =============================================================================

class ChainedLogHasher:
    """
    Maintains a blockchain-style hash chain for log entries.

    Each entry's hash includes the previous entry's hash, creating
    a tamper-evident chain.
    """

    GENESIS_HASH = "0" * 64  # Genesis block hash

    def __init__(
        self,
        log_dir: Path,
        log_file: str = "commands.log",
        hash_file: str = "commands.log.sha256",
        algorithm: str = "sha256"
    ):
        """
        Initialize chained log hasher.

        Args:
            log_dir: Directory for log files
            log_file: Main log file name
            hash_file: Hash chain file name
            algorithm: Hash algorithm
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file
        self.hash_file = self.log_dir / hash_file
        self.algorithm = algorithm
        self.sequence = 0
        self.prev_hash = self.GENESIS_HASH

        # Load existing chain state
        self._load_chain_state()

    def _load_chain_state(self) -> None:
        """Load the last hash from the chain file."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            entry = json.loads(last_line)
                            self.sequence = entry.get("sequence", 0)
                            self.prev_hash = entry.get("hash", self.GENESIS_HASH)
            except Exception as e:
                logger.warning(f"Failed to load hash chain state: {e}")

    def hash_entry(self, content: str) -> LogEntry:
        """
        Create a hashed log entry.

        Args:
            content: Log entry content

        Returns:
            LogEntry with hash
        """
        self.sequence += 1
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Hash includes: sequence + timestamp + content + previous hash
        hash_input = f"{self.sequence}|{timestamp}|{content}|{self.prev_hash}"
        current_hash = compute_hash(hash_input, self.algorithm)

        entry = LogEntry(
            timestamp=timestamp,
            sequence=self.sequence,
            content=content,
            hash=current_hash,
            prev_hash=self.prev_hash
        )

        self.prev_hash = current_hash
        return entry

    def write_entry(self, content: str) -> LogEntry:
        """
        Write a hashed entry to both log and hash files.

        Args:
            content: Log entry content

        Returns:
            LogEntry that was written
        """
        # Ensure directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create hashed entry
        entry = self.hash_entry(content)

        # Write to main log file
        with open(self.log_file, 'a') as f:
            f.write(f"[{entry.timestamp}] [{entry.sequence}] {content}\n")

        # Write to hash chain file
        with open(self.hash_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        return entry

    def verify_chain(self) -> LogIntegrityReport:
        """
        Verify the integrity of the hash chain.

        Returns:
            LogIntegrityReport with verification results
        """
        errors = []
        verified = 0
        first_invalid = None
        total = 0

        if not self.hash_file.exists():
            return LogIntegrityReport(
                valid=True,
                total_entries=0,
                verified_entries=0,
                first_invalid_entry=None,
                errors=["Hash file does not exist"],
                log_file=str(self.log_file),
                verification_time=datetime.utcnow().isoformat() + "Z"
            )

        try:
            prev_hash = self.GENESIS_HASH

            with open(self.hash_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    total += 1

                    try:
                        entry = json.loads(line)

                        # Verify previous hash linkage
                        if entry.get("prev_hash") != prev_hash:
                            if first_invalid is None:
                                first_invalid = line_num
                            errors.append(f"Line {line_num}: prev_hash mismatch")
                            continue

                        # Recompute hash
                        hash_input = (
                            f"{entry['sequence']}|{entry['timestamp']}|"
                            f"{entry['content']}|{entry['prev_hash']}"
                        )
                        computed_hash = compute_hash(hash_input, self.algorithm)

                        if computed_hash != entry.get("hash"):
                            if first_invalid is None:
                                first_invalid = line_num
                            errors.append(f"Line {line_num}: hash mismatch")
                            continue

                        verified += 1
                        prev_hash = entry["hash"]

                    except json.JSONDecodeError as e:
                        if first_invalid is None:
                            first_invalid = line_num
                        errors.append(f"Line {line_num}: JSON parse error: {e}")

        except Exception as e:
            errors.append(f"File read error: {e}")

        return LogIntegrityReport(
            valid=len(errors) == 0,
            total_entries=total,
            verified_entries=verified,
            first_invalid_entry=first_invalid,
            errors=errors,
            log_file=str(self.log_file),
            verification_time=datetime.utcnow().isoformat() + "Z"
        )

    def get_latest_hash(self) -> str:
        """Get the latest hash in the chain."""
        return self.prev_hash

    def get_chain_summary(self) -> Dict[str, Any]:
        """Get summary of the hash chain."""
        return {
            "log_file": str(self.log_file),
            "hash_file": str(self.hash_file),
            "algorithm": self.algorithm,
            "total_entries": self.sequence,
            "latest_hash": self.prev_hash[:16] + "...",
            "genesis_hash": self.GENESIS_HASH[:16] + "...",
        }


# =============================================================================
# Simple File Hasher (for existing logs)
# =============================================================================

class SimpleFileHasher:
    """
    Simple file-level hashing for existing log files.

    Creates a .sha256 sidecar file with the hash of the main log file.
    """

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm

    def hash_file(self, file_path: Path) -> Tuple[str, Path]:
        """
        Hash a file and write sidecar hash file.

        Args:
            file_path: Path to file to hash

        Returns:
            Tuple of (hash, hash_file_path)
        """
        file_path = Path(file_path)
        hash_path = file_path.with_suffix(file_path.suffix + f".{self.algorithm}")

        file_hash = compute_file_hash(file_path, self.algorithm)

        with open(hash_path, 'w') as f:
            f.write(f"{file_hash}  {file_path.name}\n")

        return file_hash, hash_path

    def verify_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Verify a file against its sidecar hash.

        Args:
            file_path: Path to file to verify

        Returns:
            Tuple of (is_valid, message)
        """
        file_path = Path(file_path)
        hash_path = file_path.with_suffix(file_path.suffix + f".{self.algorithm}")

        if not hash_path.exists():
            return False, f"Hash file not found: {hash_path}"

        try:
            with open(hash_path, 'r') as f:
                line = f.readline().strip()
                expected_hash = line.split()[0]

            actual_hash = compute_file_hash(file_path, self.algorithm)

            if actual_hash == expected_hash:
                return True, "Hash verified successfully"
            else:
                return False, f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."

        except Exception as e:
            return False, f"Verification error: {e}"


# =============================================================================
# Audit Log Manager
# =============================================================================

class AuditLogManager:
    """
    Unified audit log manager with integrity features.

    Manages command logging with optional chained hashing for
    compliance and forensic requirements.
    """

    def __init__(
        self,
        log_dir: str = ".agent_logs",
        enable_hashing: bool = True,
        algorithm: str = "sha256"
    ):
        """
        Initialize audit log manager.

        Args:
            log_dir: Directory for log files
            enable_hashing: Enable chained hashing
            algorithm: Hash algorithm
        """
        self.log_dir = Path(log_dir)
        self.enable_hashing = enable_hashing
        self.algorithm = algorithm

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if enable_hashing:
            self.hasher = ChainedLogHasher(
                log_dir=self.log_dir,
                algorithm=algorithm
            )
        else:
            self.hasher = None

        self.log_file = self.log_dir / "commands.log"

    def log_command(
        self,
        command: str,
        return_code: int,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[LogEntry]:
        """
        Log a command execution.

        Args:
            command: The command that was executed
            return_code: Command return code
            execution_time: Execution time in seconds
            metadata: Additional metadata

        Returns:
            LogEntry if hashing enabled, None otherwise
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Build log content
        content = f"CMD: {command} | RC: {return_code} | TIME: {execution_time:.3f}s"
        if metadata:
            content += f" | META: {json.dumps(metadata)}"

        if self.enable_hashing and self.hasher:
            return self.hasher.write_entry(content)
        else:
            # Write without hashing
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {content}\n")
            return None

    def log_edit(
        self,
        file_path: str,
        action: str,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None
    ) -> Optional[LogEntry]:
        """
        Log a file edit operation.

        Args:
            file_path: Path to edited file
            action: Action type (create, edit, delete)
            old_content: Original content (truncated)
            new_content: New content (truncated)

        Returns:
            LogEntry if hashing enabled
        """
        content = f"EDIT: {action} | FILE: {file_path}"
        if old_content:
            content += f" | OLD: {old_content[:100]}..."
        if new_content:
            content += f" | NEW: {new_content[:100]}..."

        if self.enable_hashing and self.hasher:
            return self.hasher.write_entry(content)
        else:
            timestamp = datetime.utcnow().isoformat() + "Z"
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {content}\n")
            return None

    def log_event(self, event_type: str, details: str) -> Optional[LogEntry]:
        """
        Log a general event.

        Args:
            event_type: Type of event (startup, shutdown, error, etc.)
            details: Event details

        Returns:
            LogEntry if hashing enabled
        """
        content = f"EVENT: {event_type} | {details}"

        if self.enable_hashing and self.hasher:
            return self.hasher.write_entry(content)
        else:
            timestamp = datetime.utcnow().isoformat() + "Z"
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {content}\n")
            return None

    def verify_integrity(self) -> LogIntegrityReport:
        """
        Verify log integrity.

        Returns:
            LogIntegrityReport with verification results
        """
        if self.enable_hashing and self.hasher:
            return self.hasher.verify_chain()
        else:
            return LogIntegrityReport(
                valid=True,
                total_entries=0,
                verified_entries=0,
                first_invalid_entry=None,
                errors=["Hashing not enabled"],
                log_file=str(self.log_file),
                verification_time=datetime.utcnow().isoformat() + "Z"
            )

    def get_recent_entries(self, n: int = 50) -> List[str]:
        """
        Get the most recent log entries.

        Args:
            n: Number of entries to return

        Returns:
            List of log entry strings
        """
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            return [line.strip() for line in lines[-n:]]
        except Exception as e:
            logger.error(f"Failed to read log entries: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        stats = {
            "log_dir": str(self.log_dir),
            "log_file": str(self.log_file),
            "hashing_enabled": self.enable_hashing,
            "algorithm": self.algorithm,
        }

        if self.log_file.exists():
            stats["log_size_bytes"] = self.log_file.stat().st_size
            stats["log_size_mb"] = stats["log_size_bytes"] / (1024 * 1024)

            with open(self.log_file, 'r') as f:
                stats["total_entries"] = sum(1 for _ in f)

        if self.enable_hashing and self.hasher:
            stats["chain_summary"] = self.hasher.get_chain_summary()

        return stats
