"""
Conftest for KOAS Reviewer regression tests.

Registers pytest CLI options for workspace paths.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-07
"""

from pathlib import Path


_DEFAULT_WS = Path(
    "/home/olivi/Documents/Adservio/audit/SIAS/v7_revised/"
    "koas_review_test/workspace_oss"
)


def pytest_addoption(parser):
    parser.addoption(
        "--status-jsonl",
        default=str(_DEFAULT_WS / "stage2" / "ops" / "status.jsonl"),
        help="Path to status.jsonl from a completed integration test run",
    )
    parser.addoption(
        "--fingerprint-json",
        default=str(_DEFAULT_WS / "stage2" / "md_fingerprint_chunk.json"),
        help="Path to md_fingerprint_chunk.json",
    )
    parser.addoption(
        "--masks-dir",
        default=str(_DEFAULT_WS / "stage2" / "masks"),
        help="Path to masks/ directory with recipe artifacts",
    )
