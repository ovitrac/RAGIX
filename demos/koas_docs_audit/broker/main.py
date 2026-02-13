"""
Minimal KOAS Broker — Demo Implementation

A local authenticated job gateway for KOAS workflows.
"Front desk + guard + bookkeeper" — not a reimplementation of KOAS.

Responsibilities:
- AuthN/AuthZ (keys + scopes)
- Job queue (in-memory + SQLite)
- Output sanitization
- Activity logging

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-30
"""

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DEMO_DIR = Path(__file__).parent.parent
WORKSPACE = DEMO_DIR / "workspace"
ACL_FILE = WORKSPACE / ".KOAS" / "auth" / "acl.yaml"
ACTIVITY_LOG = WORKSPACE / ".KOAS" / "activity" / "events.jsonl"
DB_PATH = WORKSPACE / ".KOAS" / "broker_jobs.db"

# ============================================================================
# Activity Event Schema (koas.event/1.0)
# ============================================================================

SCHEMA_VERSION = "koas.event/1.0"

@dataclass
class Actor:
    type: str  # system | operator | external_orchestrator
    id: str
    auth: str = "api_key"
    session: Optional[str] = None

@dataclass
class ActivityEvent:
    scope: str
    actor: Actor
    phase: str = ""
    refs: Dict[str, str] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    v: str = field(default=SCHEMA_VERSION)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["actor"] = asdict(self.actor)
        return json.dumps(d, separators=(',', ':'))


class ActivityWriter:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: ActivityEvent) -> None:
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(event.to_json() + "\n")


# Global activity writer
activity = ActivityWriter(ACTIVITY_LOG)

# ============================================================================
# ACL / Auth
# ============================================================================

@dataclass
class Client:
    id: str
    type: str
    scopes: List[str]
    key_hash: Optional[str]
    rate_limit: Optional[str] = None
    restrictions: List[str] = field(default_factory=list)


class ACLManager:
    def __init__(self, acl_path: Path):
        self.clients: Dict[str, Client] = {}
        if acl_path.exists() and YAML_AVAILABLE:
            with open(acl_path) as f:
                config = yaml.safe_load(f)
            for cid, cfg in config.get("clients", {}).items():
                self.clients[cid] = Client(
                    id=cid,
                    type=cfg.get("type", "unknown"),
                    scopes=cfg.get("scopes", []),
                    key_hash=cfg.get("key_hash"),
                    rate_limit=cfg.get("rate_limit"),
                    restrictions=cfg.get("restrictions", []),
                )
        logger.info(f"ACL loaded: {len(self.clients)} clients")

    def authenticate(self, api_key: str) -> Optional[Client]:
        """Authenticate by API key, return client if valid."""
        key_hash = f"sha256:{hashlib.sha256(api_key.encode()).hexdigest()}"
        for client in self.clients.values():
            if client.key_hash == key_hash:
                return client
        return None

    def authorize(self, client: Client, scope: str) -> bool:
        """Check if client has required scope."""
        if "*" in client.scopes:
            return True
        return scope in client.scopes


# Global ACL manager
acl = ACLManager(ACL_FILE)

# ============================================================================
# Job Management
# ============================================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    workspace: str
    mode: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    client_id: str = ""


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                workspace TEXT,
                mode TEXT,
                created_at TEXT,
                started_at TEXT,
                ended_at TEXT,
                progress TEXT,
                metrics TEXT,
                error TEXT,
                client_id TEXT
            )
        """)
        conn.commit()
        conn.close()

    def create(self, job: Job) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO jobs (id, status, workspace, mode, created_at, client_id, progress, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (job.id, job.status.value, job.workspace, job.mode, job.created_at,
              job.client_id, json.dumps(job.progress), json.dumps(job.metrics)))
        conn.commit()
        conn.close()

    def get(self, job_id: str) -> Optional[Job]:
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return Job(
            id=row[0],
            status=JobStatus(row[1]),
            workspace=row[2],
            mode=row[3],
            created_at=row[4],
            started_at=row[5],
            ended_at=row[6],
            progress=json.loads(row[7] or "{}"),
            metrics=json.loads(row[8] or "{}"),
            error=row[9],
            client_id=row[10],
        )

    def update(self, job: Job) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            UPDATE jobs SET status=?, started_at=?, ended_at=?, progress=?, metrics=?, error=?
            WHERE id=?
        """, (job.status.value, job.started_at, job.ended_at,
              json.dumps(job.progress), json.dumps(job.metrics), job.error, job.id))
        conn.commit()
        conn.close()


# Global job store
jobs = JobStore(DB_PATH)

# ============================================================================
# KOAS Runner
# ============================================================================

def run_koas_job(job_id: str):
    """Execute KOAS pipeline in background."""
    job = jobs.get(job_id)
    if not job:
        return

    # Mark started
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc).isoformat()
    jobs.update(job)

    activity.emit(ActivityEvent(
        scope="broker.job",
        actor=Actor(type="system", id="broker"),
        phase="started",
        refs={"job_id": job_id},
    ))

    # Run KOAS CLI
    try:
        result = subprocess.run([
            "python", "-m", "ragix_kernels.run_doc_koas", "run",
            "--workspace", str(WORKSPACE),
            "--all",
            "--output-level", "external",
            "--quiet"
        ], capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            job.status = JobStatus.COMPLETED
            job.metrics = {
                "exit_code": 0,
                "stdout_lines": len(result.stdout.splitlines()),
            }
            # Try to extract metrics from output
            if (WORKSPACE / ".KOAS" / "final_report.md").exists():
                job.metrics["report_size"] = (WORKSPACE / ".KOAS" / "final_report.md").stat().st_size
        else:
            job.status = JobStatus.FAILED
            job.error = f"Exit code {result.returncode}"
            job.metrics = {"exit_code": result.returncode}

    except subprocess.TimeoutExpired:
        job.status = JobStatus.FAILED
        job.error = "Timeout (600s)"
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(type(e).__name__)

    job.ended_at = datetime.now(timezone.utc).isoformat()
    jobs.update(job)

    activity.emit(ActivityEvent(
        scope="broker.job",
        actor=Actor(type="system", id="broker"),
        phase="completed" if job.status == JobStatus.COMPLETED else "failed",
        refs={"job_id": job_id},
        metrics=job.metrics,
        decision={"status": job.status.value},
    ))


# ============================================================================
# FastAPI Application
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(title="KOAS Broker", version="0.1.0")

    # --- Auth Dependency ---

    async def get_current_client(authorization: str = Header(None)) -> Client:
        if not authorization or not authorization.startswith("Bearer "):
            activity.emit(ActivityEvent(
                scope="system.auth",
                actor=Actor(type="unknown", id="anonymous", auth="none"),
                phase="denied",
                decision={"reason": "missing_token"},
            ))
            raise HTTPException(status_code=401, detail="Missing authorization header")

        api_key = authorization.replace("Bearer ", "")
        client = acl.authenticate(api_key)

        if not client:
            activity.emit(ActivityEvent(
                scope="system.auth",
                actor=Actor(type="unknown", id="anonymous", auth="api_key"),
                phase="denied",
                decision={"reason": "invalid_key"},
            ))
            raise HTTPException(status_code=401, detail="Invalid API key")

        activity.emit(ActivityEvent(
            scope="system.auth",
            actor=Actor(type=client.type, id=client.id, auth="api_key"),
            phase="granted",
        ))

        return client

    # --- Endpoints ---

    @app.get("/koas/v1/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    class JobRequest(BaseModel):
        mode: str = "pure_docs"
        profile: str = "default"
        workspace: str = "./workspace"
        actions: List[str] = ["koas_audit"]

    @app.post("/koas/v1/jobs")
    async def create_job(
        request: JobRequest,
        background_tasks: BackgroundTasks,
        client: Client = Depends(get_current_client)
    ):
        # Check scope
        if not acl.authorize(client, "docs.trigger"):
            raise HTTPException(status_code=403, detail="Scope 'docs.trigger' required")

        # Create job
        job = Job(
            id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            status=JobStatus.QUEUED,
            workspace=str(WORKSPACE),
            mode=request.mode,
            created_at=datetime.now(timezone.utc).isoformat(),
            client_id=client.id,
        )
        jobs.create(job)

        activity.emit(ActivityEvent(
            scope="broker.job",
            actor=Actor(type=client.type, id=client.id, auth="api_key"),
            phase="created",
            refs={"job_id": job.id},
            decision={"mode": request.mode},
        ))

        # Run in background
        background_tasks.add_task(run_koas_job, job.id)

        return {"job_id": job.id, "status": job.status.value}

    @app.get("/koas/v1/jobs/{job_id}")
    async def get_job_status(job_id: str, client: Client = Depends(get_current_client)):
        # Check scope
        if not acl.authorize(client, "docs.status"):
            raise HTTPException(status_code=403, detail="Scope 'docs.status' required")

        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Return metrics only (no internal details)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "progress": job.progress,
            "metrics": job.metrics,
        }

    @app.get("/koas/v1/jobs/{job_id}/artifact")
    async def get_artifact(
        job_id: str,
        view: str = "external",
        client: Client = Depends(get_current_client)
    ):
        # Check scope based on view
        required_scope = "docs.export_internal" if view == "internal" else "docs.export_external"
        if not acl.authorize(client, required_scope):
            raise HTTPException(status_code=403, detail=f"Scope '{required_scope}' required")

        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status.value})")

        # Return report file
        report_path = WORKSPACE / ".KOAS" / "final_report.md"
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        activity.emit(ActivityEvent(
            scope="broker.artifact",
            actor=Actor(type=client.type, id=client.id, auth="api_key"),
            phase="downloaded",
            refs={"job_id": job_id},
            decision={"view": view},
        ))

        return FileResponse(
            path=str(report_path),
            media_type="text/markdown",
            filename=f"report_{job_id}.md"
        )


# ============================================================================
# Main
# ============================================================================

def main():
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed")
        print("Run: pip install fastapi uvicorn pyyaml")
        return

    import uvicorn
    logger.info(f"Starting KOAS Broker on http://localhost:8080")
    logger.info(f"Workspace: {WORKSPACE}")
    logger.info(f"ACL: {ACL_FILE}")
    logger.info(f"Activity: {ACTIVITY_LOG}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
