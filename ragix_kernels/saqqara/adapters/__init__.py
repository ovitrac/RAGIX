"""
saqqara.adapters — one reader per format, raw facts only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K2 in SPEC.md. Readers for spreadsheets, word-processing documents, presentations
and markdown land in P2; the laid-out-document reader in P2'.

Each reader claims a set of extensions and answers a single contract: read a path, yield Mastaba
raw facts. A fact is what the file says — a cell's data type, a run's boldness, a number format, a
lock flag, a merge state — never what those facts might mean. Interpretation is the analyzers'
work, and keeping the two apart is what lets a new fact be added without disturbing any rule that
does not use it.

Failure is explicit: `read_path` raises on a file no reader claims or that its reader cannot
parse, and `read_paths` turns those into counted refusals rather than a quietly shorter result.
"""

from .contract import (  # noqa: F401
    FIGURE_FACTS,
    FIGURE_SOURCES,
    PART_SKIPS,
    GRID_CELL_FACTS,
    GRID_TABLE_FACTS,
    Adapter,
    Mastaba,
    OpenVocabulary,
    ReadReport,
    Refusal,
    UnreadableFile,
    UnsupportedFormat,
    adapter_for,
    read_corpus,
    read_path,
    read_paths,
    register_adapter,
    registered_adapters,
)
from . import docx as _docx      # noqa: F401  registers the word-processing reader
from . import md as _md          # noqa: F401  registers the markdown reader
from . import pdf as _pdf        # noqa: F401  registers the laid-out-document reader
from . import pptx as _pptx      # noqa: F401  registers the presentation reader
from . import xlsx as _xlsx      # noqa: F401  registers the spreadsheet reader

__all__ = [
    "Adapter",
    "FIGURE_FACTS",
    "FIGURE_SOURCES",
    "PART_SKIPS",
    "GRID_CELL_FACTS",
    "GRID_TABLE_FACTS",
    "Mastaba",
    "OpenVocabulary",
    "ReadReport",
    "Refusal",
    "UnreadableFile",
    "UnsupportedFormat",
    "adapter_for",
    "read_path",
    "read_paths",
    "register_adapter",
    "registered_adapters",
]
