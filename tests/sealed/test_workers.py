"""
Tests for the Sprint 2bis sealed worker protocol (WP §8ter).

Job sealing under remote AAD, in-process worker run, derivative re-sealing, sanitized
attestation (no raw), workspace cleanup, and remote-AAD tamper rejection. Synthetic only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import dataclasses
import json

import pytest

from ragix_core.crypto.sealed_aead import SealedAEADError, open_bytes
from ragix_sealed.ingest import new_case_context
from ragix_sealed.workers import (
    LocalSealedWorker,
    SSHSealedWorker,
    build_job_package,
)

# A synthetic captioner: pretends to describe an image, returning placeholderized text.
def _captioner(_raw: bytes) -> str:
    return "The image shows [PERSON_014] near [LOCATION_008]."


def _job(ctx):
    return build_job_package(
        ctx, job_id="job_001", worker_id="local_worker_01", task="image_caption",
        raw_inputs=[{"artifact_id": "img_001", "source_id": "src_abc",
                     "data": b"\x89PNG synthetic image bytes"}],
    )


def test_worker_runs_and_returns_sealed_derivative():
    ctx = new_case_context("case_w_001")
    worker = LocalSealedWorker(ctx)
    outputs, attestation = worker.run(_job(ctx), _captioner)
    assert len(outputs) == 1
    # Derivative opens (human/internal path) to the placeholderized caption.
    assert worker.open_derivative(outputs[0]) == "The image shows [PERSON_014] near [LOCATION_008]."


def test_attestation_is_sanitized():
    ctx = new_case_context("case_w_001")
    worker = LocalSealedWorker(ctx)
    _, attestation = worker.run(_job(ctx), _captioner)
    pub = attestation.to_public_dict()
    assert pub["raw_content_logged"] is False
    assert pub["workspace_destroyed"] is True
    # No raw bytes/text leak into the attestation.
    blob = json.dumps(pub)
    assert "PNG" not in blob and "image shows" not in blob


def test_remote_aad_binding_rejects_tamper():
    ctx = new_case_context("case_w_001")
    job = _job(ctx)
    sealed_in = job.inputs[0]
    # Opening with a mismatched AAD (wrong job_id) must fail authentication.
    bad_aad = dataclasses.replace(sealed_in.aad, job_id="job_999")
    with pytest.raises(SealedAEADError):
        open_bytes(sealed_in.blob, bad_aad, ctx.key)
    # Correct AAD opens.
    assert open_bytes(sealed_in.blob, sealed_in.aad, ctx.key) == b"\x89PNG synthetic image bytes"


def test_output_artifact_id_is_opaque():
    ctx = new_case_context("case_w_001")
    worker = LocalSealedWorker(ctx)
    outputs, _ = worker.run(_job(ctx), _captioner)
    assert outputs[0].artifact_id.startswith("caption_")
    assert "src_abc" not in outputs[0].artifact_id


def test_ssh_worker_deferred():
    with pytest.raises(NotImplementedError, match="deferred"):
        SSHSealedWorker().run()
