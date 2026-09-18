"""Opt-in pytest capture of immutable quantity outputs for a before/after audit.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import json
from dataclasses import asdict
from pathlib import Path
import pytest

CAPTURE = {}


def pytest_addoption(parser):
    parser.addoption("--quantity-audit-output", type=Path)


def pytest_configure(config):
    if config.getoption("--quantity-audit-output") is None:
        raise pytest.UsageError("quantity capture needs an explicit output path")
    import ragix_kernels.harvest.quantitative as quantities
    import ragix_kernels.saqqara.profile_readers as readers

    original = quantities.harvest

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        key = json.dumps((args, kwargs), sort_keys=True, ensure_ascii=False, default=asdict)
        value = [asdict(candidate) for candidate in result]
        if key in CAPTURE:
            assert CAPTURE[key] == value
        CAPTURE[key] = value
        return result

    quantities.harvest = capture
    readers.harvest = capture


def pytest_sessionfinish(session, exitstatus):
    target = session.config.getoption("--quantity-audit-output")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(CAPTURE, sort_keys=True, ensure_ascii=False) + "\n")
