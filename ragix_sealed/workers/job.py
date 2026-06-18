"""
RAGIX-Sealed — sealed worker job protocol (WP §8ter, Sprint 2bis).

A sealed job carries inputs sealed under a *remote AAD* that binds case, job, worker,
source, artifact, policy/schema versions, model/task profile, and state — defeating
cross-worker / cross-case / cross-policy / cross-task substitution (§8ter.5).

The worker returns sealed derivatives plus a *sanitized* attestation (no raw content).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..ingest.ids import CaseContext
from ragix_core.crypto.sealed_aead import SealedBlob, seal_bytes


@dataclass(frozen=True)
class RemoteAAD:
    """Associated data bound into every sealed remote artifact (WP §8ter.5).

    Duck-types as a sealing AAD: it provides ``to_bytes()``, so it can be passed to
    ``seal_bytes`` / ``open_bytes`` directly. All fields are non-secret identifiers.
    """

    case_id: str
    job_id: str
    worker_id: str
    source_id: str
    artifact_id: str
    policy_version: str
    placeholder_schema_version: str
    model_profile: str
    task_profile: str
    ingestion_state: str

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "case_id": self.case_id, "job_id": self.job_id, "worker_id": self.worker_id,
                "source_id": self.source_id, "artifact_id": self.artifact_id,
                "policy_version": self.policy_version,
                "placeholder_schema_version": self.placeholder_schema_version,
                "model_profile": self.model_profile, "task_profile": self.task_profile,
                "ingestion_state": self.ingestion_state,
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")


@dataclass(frozen=True)
class SealedInput:
    artifact_id: str
    source_id: str
    blob: SealedBlob
    aad: RemoteAAD


@dataclass(frozen=True)
class SealedJob:
    case_id: str
    job_id: str
    worker_id: str
    task: str
    inputs: List[SealedInput]


@dataclass(frozen=True)
class WorkerAttestation:
    """Sanitized worker attestation (WP §8ter.7). No raw text/filename/prompt/caption."""

    job_id: str
    worker_id: str
    case_id: str
    task: str
    input_ids: List[str]
    output_ids: List[str]
    model_profile: str
    cleanup: str = "completed"
    raw_content_logged: bool = False
    workspace_destroyed: bool = True

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id, "worker_id": self.worker_id, "case_id": self.case_id,
            "task": self.task, "inputs": list(self.input_ids), "outputs": list(self.output_ids),
            "model_profile": self.model_profile, "cleanup": self.cleanup,
            "raw_content_logged": self.raw_content_logged,
            "workspace_destroyed": self.workspace_destroyed,
        }


def build_job_package(
    ctx: CaseContext,
    job_id: str,
    worker_id: str,
    task: str,
    raw_inputs: List[Dict[str, Any]],
) -> SealedJob:
    """Seal raw inputs into a job package.

    ``raw_inputs``: list of ``{"artifact_id", "source_id", "data": bytes}``. Each is sealed
    under a RemoteAAD with ``ingestion_state="DERIVED_RAW_INTERNAL"``.
    """
    inputs: List[SealedInput] = []
    for item in raw_inputs:
        aad = RemoteAAD(
            case_id=ctx.case_id, job_id=job_id, worker_id=worker_id,
            source_id=item["source_id"], artifact_id=item["artifact_id"],
            policy_version=ctx.policy_version,
            placeholder_schema_version=ctx.placeholder_schema_version,
            model_profile=task, task_profile=task, ingestion_state="DERIVED_RAW_INTERNAL",
        )
        blob = seal_bytes(item["data"], aad, ctx.key)
        inputs.append(SealedInput(item["artifact_id"], item["source_id"], blob, aad))
    return SealedJob(case_id=ctx.case_id, job_id=job_id, worker_id=worker_id, task=task, inputs=inputs)
