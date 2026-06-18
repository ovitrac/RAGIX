"""
RAGIX-Sealed sealed LAN workers (WP §8ter, Sprint 2bis).

Master-authoritative job protocol: inputs are sealed under a remote AAD, the worker runs
the task in a volatile workspace and returns sealed derivatives + a sanitized attestation,
then destroys the workspace. SSH transport is deferred; LocalSealedWorker runs in-process.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .job import (
    RemoteAAD,
    SealedInput,
    SealedJob,
    WorkerAttestation,
    build_job_package,
)
from .local import LocalSealedWorker, SealedDerivative, SSHSealedWorker

__all__ = [
    "RemoteAAD",
    "SealedInput",
    "SealedJob",
    "WorkerAttestation",
    "build_job_package",
    "LocalSealedWorker",
    "SealedDerivative",
    "SSHSealedWorker",
]
