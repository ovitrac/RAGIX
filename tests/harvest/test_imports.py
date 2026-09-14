"""Every module of the harvest family imports from the package, and from nowhere else.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The modules were moved with their imports rewritten from the lab's `sys.path` arrangements to
package-relative ones. Most of them are drivers with no test of their own; importing each one is the
least that proves the rewrite resolved, and scanning each source proves no path hack came along.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import ragix_kernels.harvest as harvest

MODULES = sorted(name for _, name, _ in pkgutil.walk_packages([str(Path(harvest.__file__).parent)],
                                                             prefix="ragix_kernels.harvest."))


def test_the_family_has_its_modules():
    assert {"ragix_kernels.harvest.form", "ragix_kernels.harvest.runner", "ragix_kernels.harvest.core",
            "ragix_kernels.harvest.pass1", "ragix_kernels.harvest.derived"} <= set(MODULES)


@pytest.mark.parametrize("name", MODULES)
def test_the_module_imports_and_carries_no_path_hack(name):
    module = importlib.import_module(name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*sys\.path\.insert", source, re.M), f"{name} edits sys.path"
    assert not re.search(r"^\s*(from|import) tender\b", source, re.M), f"{name} imports the lab's package"
