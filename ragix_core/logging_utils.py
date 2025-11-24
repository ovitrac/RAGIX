"""
Logging Utilities for RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(str, Enum):
    """Log levels for RAGIX agent."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Patterns for secret masking (environment variables, tokens, keys)
SECRET_PATTERNS = [
    (re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH)[=:]\s*['\"]?([^'\"\ \n]+)", re.I), r"\1=***"),
    (re.compile(r"(Bearer|token)\s+([a-zA-Z0-9_\-\.]+)", re.I), r"\1 ***"),
    (re.compile(r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----", re.DOTALL), "*** SSH/PGP KEY ***"),
]


def mask_secrets(text: str) -> str:
    """
    Mask secrets in text before logging.

    Args:
        text: Raw text that may contain secrets

    Returns:
        Text with secrets replaced by ***
    """
    masked = text
    for pattern, replacement in SECRET_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


class AgentLogger:
    """
    File-based logger for RAGIX agent operations with structured JSONL support.

    Logs are written to:
    - {log_dir}/commands.log - Human-readable text log
    - {log_dir}/events.jsonl - Structured JSONL log
    """

    def __init__(self, log_dir: Path, min_level: LogLevel = LogLevel.INFO, mask_secrets_enabled: bool = True):
        """
        Initialize agent logger.

        Args:
            log_dir: Directory for log files
            min_level: Minimum log level to write
            mask_secrets_enabled: Whether to mask secrets in logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.min_level = min_level
        self.mask_secrets_enabled = mask_secrets_enabled

        self.text_log = self.log_dir / "commands.log"
        self.json_log = self.log_dir / "events.jsonl"

    def _should_log(self, level: LogLevel) -> bool:
        """Check if a log level should be written."""
        levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        return levels.index(level) >= levels.index(self.min_level)

    def _mask_if_enabled(self, text: str) -> str:
        """Mask secrets if enabled."""
        if self.mask_secrets_enabled:
            return mask_secrets(text)
        return text

    def log_text(self, line: str, level: LogLevel = LogLevel.INFO):
        """
        Append a timestamped line to the text log.

        Args:
            line: Log message (timestamp will be prepended)
            level: Log level
        """
        if not self._should_log(level):
            return

        timestamp = datetime.utcnow().isoformat()
        masked_line = self._mask_if_enabled(line)
        with self.text_log.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level.value}] {masked_line}\n")

    def log_jsonl(self, event_type: str, data: Dict[str, Any], level: LogLevel = LogLevel.INFO):
        """
        Append structured JSONL event to the events log.

        Args:
            event_type: Type of event (e.g., "command", "edit", "error")
            data: Event data dictionary
            level: Log level
        """
        if not self._should_log(level):
            return

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "type": event_type,
            "data": data
        }

        # Mask secrets in JSON data
        if self.mask_secrets_enabled:
            event_str = json.dumps(event)
            event_str = mask_secrets(event_str)
        else:
            event_str = json.dumps(event)

        with self.json_log.open("a", encoding="utf-8") as f:
            f.write(event_str + "\n")

    def log_event(self, event: str, details: str = "", level: LogLevel = LogLevel.INFO):
        """
        Log an event to both text and JSONL logs.

        Args:
            event: Event type/name
            details: Optional event details
            level: Log level
        """
        line = f"EVENT={event}"
        if details:
            line += f" DETAILS={details}"
        self.log_text(line, level)

        self.log_jsonl(event, {"details": details}, level)

    def log_command(self, command: str, cwd: str, returncode: int,
                   stdout: Optional[str] = None, stderr: Optional[str] = None,
                   duration_ms: Optional[float] = None):
        """
        Log a shell command execution to both text and JSONL logs.

        Args:
            command: The command that was executed
            cwd: Working directory
            returncode: Command exit code
            stdout: Command stdout (optional, truncated if long)
            stderr: Command stderr (optional, truncated if long)
            duration_ms: Execution duration in milliseconds
        """
        # Text log (brief)
        summary = f'CMD="{command}" CWD=\'{cwd}\' RC={returncode}'
        if duration_ms is not None:
            summary += f" DURATION={duration_ms:.1f}ms"

        level = LogLevel.ERROR if returncode != 0 else LogLevel.INFO
        self.log_text(summary, level)

        # JSONL log (full structured data)
        data = {
            "command": command,
            "cwd": cwd,
            "returncode": returncode,
        }
        if stdout:
            data["stdout"] = stdout[:1000]  # Truncate long outputs
        if stderr:
            data["stderr"] = stderr[:1000]
        if duration_ms is not None:
            data["duration_ms"] = duration_ms

        self.log_jsonl("command", data, level)

    def debug(self, message: str):
        """Log debug message."""
        self.log_text(f"DEBUG: {message}", LogLevel.DEBUG)

    def info(self, message: str):
        """Log info message."""
        self.log_text(f"INFO: {message}", LogLevel.INFO)

    def warning(self, message: str):
        """Log warning message."""
        self.log_text(f"WARNING: {message}", LogLevel.WARNING)

    def error(self, message: str):
        """Log error message."""
        self.log_text(f"ERROR: {message}", LogLevel.ERROR)
