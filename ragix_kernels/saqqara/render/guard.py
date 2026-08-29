"""
saqqara.render.guard — proof that the AGPL renderer is not on any default path.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-29

Specified by K6.16 in SPEC.md.

Two checks, because either alone is insufficient. A **source scan** catches an import written into
the kernel; a **runtime check** catches one arriving through a dependency, a plugin, or a caller.
A licence obligation that depends on which code path happened to execute is not an obligation
anyone can audit, so the claim is the strong one: not "we do not call it", but "it is not loaded".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

__all__ = ["AGPL_MODULES", "EXEMPT", "loaded_agpl_modules", "scan_sources"]

#: The module names that carry the obligation. `fitz` is the same library under
#: its older name, and a scan that knew only one of them would pass while the
#: other was imported on the line below.
AGPL_MODULES = ("pymupdf", "fitz", "pymupdf4llm")

#: The single file allowed to name them, relative to the kernel root.
EXEMPT = ("render/mupdf.py",)

_IMPORT = re.compile(
    r"^\s*(?:import\s+(?P<a>[A-Za-z_][\w.]*)|from\s+(?P<b>[A-Za-z_][\w.]*)\s+import)",
    re.MULTILINE,
)


def scan_sources(root: Path | None = None) -> list[tuple[str, int, str]]:
    """Every import of an AGPL module under the kernel, outside the exemption.

    Returns (path, line number, module). An empty list is the claim K6.16 makes
    about this package's source.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    violations: list[tuple[str, int, str]] = []
    for source in sorted(root.rglob("*.py")):
        relative = source.relative_to(root).as_posix()
        if relative in EXEMPT:
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for match in _IMPORT.finditer(text):
            module = (match.group("a") or match.group("b") or "").split(".")[0]
            if module in AGPL_MODULES:
                line = text.count("\n", 0, match.start()) + 1
                violations.append((relative, line, module))
    return violations


def loaded_agpl_modules() -> list[str]:
    """Which AGPL modules are in `sys.modules` right now.

    The check that matters: not whether this package calls the library, but
    whether the process has loaded it at all.
    """
    return [name for name in AGPL_MODULES if name in sys.modules]
