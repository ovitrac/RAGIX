"""Run optional-renderer falsifiers without loading it into the default suite.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_pdf_clean_synthetic_falsifiers_in_isolated_process():
    for dependency in ("pikepdf", "pymupdf", "numpy"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Optional pdf-clean dependency absent: {dependency}")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).with_name("pdf_clean_cases.py")), "-q"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
