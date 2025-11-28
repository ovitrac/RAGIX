"""
RAGIX Logs Router - Log management and integrity API endpoints

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/logs", tags=["logs"])

# Try to import ragix_core components
try:
    from ragix_core import AuditLogManager
    INTEGRITY_AVAILABLE = True
except ImportError:
    INTEGRITY_AVAILABLE = False
    AuditLogManager = None

# Reference to sessions store
_active_sessions: Dict[str, Dict[str, Any]] = {}


def set_sessions_store(sessions: Dict[str, Dict[str, Any]]):
    """Set the sessions store reference from server.py"""
    global _active_sessions
    _active_sessions = sessions


@router.get("/server")
async def get_server_logs(limit: int = 100):
    """Get server-side logs (from all active sessions)."""
    all_logs = []

    # Check all active sessions
    for session_id, session in _active_sessions.items():
        sandbox_root = session.get("sandbox_root", "")
        if sandbox_root:
            log_file = Path(sandbox_root) / ".agent_logs" / "commands.log"
            if log_file.exists():
                try:
                    lines = log_file.read_text().strip().split('\n')
                    for line in lines[-50:]:
                        all_logs.append({
                            "session": session_id,
                            "log": line
                        })
                except Exception:
                    pass

    # Also check default location
    default_log = Path(".agent_logs/commands.log")
    if default_log.exists():
        try:
            lines = default_log.read_text().strip().split('\n')
            for line in lines[-50:]:
                all_logs.append({
                    "session": "default",
                    "log": line
                })
        except Exception:
            pass

    return {"logs": all_logs[-limit:], "count": len(all_logs)}


@router.get("/integrity")
async def verify_log_integrity(session_id: Optional[str] = None):
    """
    Verify integrity of command logs using hash chains.

    Returns integrity status and any detected tampering.
    """
    if not INTEGRITY_AVAILABLE:
        return {
            "available": False,
            "message": "Log integrity module not available"
        }

    # Determine log directory
    if session_id and session_id in _active_sessions:
        sandbox_root = _active_sessions[session_id].get("sandbox_root", "")
        log_dir = Path(sandbox_root) / ".agent_logs" if sandbox_root else Path(".agent_logs")
    else:
        log_dir = Path(".agent_logs")

    if not log_dir.exists():
        return {
            "available": True,
            "verified": False,
            "message": "No logs found",
            "log_dir": str(log_dir)
        }

    try:
        manager = AuditLogManager(log_dir)
        report = manager.verify_integrity()

        return {
            "available": True,
            "verified": report.is_valid,
            "total_entries": report.total_entries,
            "valid_entries": report.valid_entries,
            "invalid_entries": report.invalid_entries,
            "tampering_detected": report.tampering_detected,
            "issues": report.issues[:10],
            "log_dir": str(log_dir)
        }
    except Exception as e:
        return {
            "available": True,
            "verified": False,
            "error": str(e),
            "log_dir": str(log_dir)
        }


@router.get("/chain")
async def get_log_chain(session_id: Optional[str] = None, limit: int = 50):
    """
    Get log entries with their hash chain for visualization.

    Shows the cryptographic chain linking each log entry.
    """
    # Determine log directory
    if session_id and session_id in _active_sessions:
        sandbox_root = _active_sessions[session_id].get("sandbox_root", "")
        log_dir = Path(sandbox_root) / ".agent_logs" if sandbox_root else Path(".agent_logs")
    else:
        log_dir = Path(".agent_logs")

    hash_file = log_dir / "hashes.jsonl"

    if not hash_file.exists():
        return {"chain": [], "count": 0}

    chain = []
    try:
        lines = hash_file.read_text().strip().split('\n')
        for line in lines[-limit:]:
            if line:
                entry = json.loads(line)
                chain.append({
                    "index": entry.get("index", 0),
                    "timestamp": entry.get("timestamp", ""),
                    "hash": entry.get("hash", "")[:16] + "...",
                    "prev_hash": (entry.get("prev_hash", "")[:16] + "...") if entry.get("prev_hash") else None,
                    "command_preview": entry.get("command", "")[:50] + "..." if entry.get("command") else None
                })
    except Exception as e:
        return {"chain": [], "error": str(e)}

    return {"chain": chain, "count": len(chain)}


@router.get("/stats")
async def get_log_stats(session_id: Optional[str] = None):
    """
    Get statistics about logs for a session.

    Returns counts, sizes, and summary information.
    """
    # Determine log directory
    if session_id and session_id in _active_sessions:
        sandbox_root = _active_sessions[session_id].get("sandbox_root", "")
        log_dir = Path(sandbox_root) / ".agent_logs" if sandbox_root else Path(".agent_logs")
    else:
        log_dir = Path(".agent_logs")

    stats = {
        "log_dir": str(log_dir),
        "exists": log_dir.exists(),
        "files": {},
        "total_size_bytes": 0,
        "total_entries": 0
    }

    if not log_dir.exists():
        return stats

    # Check each log file
    log_files = ["commands.log", "events.jsonl", "hashes.jsonl", "episodic_log.jsonl"]
    for filename in log_files:
        filepath = log_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            lines = len(filepath.read_text().strip().split('\n'))
            stats["files"][filename] = {
                "size_bytes": size,
                "size_kb": round(size / 1024, 2),
                "entries": lines
            }
            stats["total_size_bytes"] += size
            stats["total_entries"] += lines

    stats["total_size_kb"] = round(stats["total_size_bytes"] / 1024, 2)

    return stats
