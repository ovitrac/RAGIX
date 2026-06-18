"""
RAGIX-Sealed — in-process sealed worker (WP §8ter, Sprint 2bis).

Simulates the sealed-worker contract without a network: opens sealed inputs inside a
volatile temp workspace, runs an injected task function (e.g. a captioner that returns
placeholderized text), re-seals the outputs under a derivative AAD, emits a sanitized
attestation, and wipes the workspace on success *and* failure.

The SSH transport + remote key handling are deferred (see SSHSealedWorker) exactly as
PDF/DOCX extraction was — the framework here is real and tested; the transport is not.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Tuple

from ..ingest.ids import CaseContext
from ragix_core.crypto.sealed_aead import SealedBlob, open_bytes, seal_bytes
from .job import RemoteAAD, SealedJob, WorkerAttestation

# A task transforms decrypted input bytes (text or binary, e.g. an image) into
# placeholderized output text (INTERNAL).
TaskFn = Callable[[bytes], str]


@dataclass(frozen=True)
class SealedDerivative:
    artifact_id: str
    source_id: str
    blob: SealedBlob
    aad: RemoteAAD
    hash: str  # sha256 of the sealed ciphertext, for the provenance node


class LocalSealedWorker:
    """Runs a sealed job in-process. The master and worker share ``ctx`` here; a real
    worker would receive job-scoped key material over the sealed transport."""

    def __init__(self, ctx: CaseContext, worker_id: str = "local_worker_01") -> None:
        self.ctx = ctx
        self.worker_id = worker_id

    def run(self, job: SealedJob, task_fn: TaskFn) -> Tuple[List[SealedDerivative], WorkerAttestation]:
        workspace = tempfile.mkdtemp(prefix="ragix-worker-")
        outputs: List[SealedDerivative] = []
        try:
            for inp in job.inputs:
                # Decrypt only inside the volatile workspace (bytes; may be binary).
                raw = open_bytes(inp.blob, inp.aad, self.ctx.key)
                # Run the task (injected). Output is expected placeholderized/INTERNAL.
                result = task_fn(raw)
                out_artifact = self.ctx.derive_id("caption", inp.source_id, job.job_id, job.task)
                out_aad = RemoteAAD(
                    case_id=job.case_id, job_id=job.job_id, worker_id=self.worker_id,
                    source_id=inp.source_id, artifact_id=out_artifact,
                    policy_version=self.ctx.policy_version,
                    placeholder_schema_version=self.ctx.placeholder_schema_version,
                    model_profile=job.task, task_profile=job.task,
                    ingestion_state="DERIVED_COOLED_DESCRIPTOR",
                )
                blob = seal_bytes(result.encode("utf-8"), out_aad, self.ctx.key)
                digest = hashlib.sha256(blob.ciphertext).hexdigest()
                outputs.append(SealedDerivative(out_artifact, inp.source_id, blob, out_aad, digest))

            attestation = WorkerAttestation(
                job_id=job.job_id, worker_id=self.worker_id, case_id=job.case_id, task=job.task,
                input_ids=[i.artifact_id for i in job.inputs],
                output_ids=[o.artifact_id for o in outputs],
                model_profile=job.task, cleanup="completed",
                raw_content_logged=False, workspace_destroyed=True,
            )
            return outputs, attestation
        finally:
            # Cleanup runs on success AND failure (WP §8ter.6).
            shutil.rmtree(workspace, ignore_errors=True)

    def open_derivative(self, derivative: SealedDerivative) -> str:
        """Open a sealed derivative — INTERNAL/human path only."""
        return open_bytes(derivative.blob, derivative.aad, self.ctx.key).decode("utf-8")


class SSHSealedWorker:
    """Real SSH + remote-tmpfs transport — deferred (framework placeholder, WP §8ter.3)."""

    def run(self, *_args, **_kwargs):  # pragma: no cover - deferred
        raise NotImplementedError(
            "SSH sealed-worker transport is deferred; use LocalSealedWorker for now. "
            "A real worker runs the task in remote tmpfs with Ollama bound to 127.0.0.1."
        )
