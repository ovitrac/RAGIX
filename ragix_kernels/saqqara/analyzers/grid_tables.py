"""
saqqara.analyzers.grid_tables — is this a table, or a page layout drawn with one?

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.29, K3.30 and K3.32 of SPEC.md.

This runs over every format whose tables are not implied by a grid of cells — word-processing
documents and presentations. A spreadsheet does not need it: there, a table is found by
segmentation, not declared by the file.

Word-processing documents use tables for two unrelated purposes. Some hold data. Others exist only
to put two paragraphs side by side, and their author never thought of them as tables at all. They
are the same construct in the file, and telling them apart is a prerequisite for everything
downstream: running header recognition over a layout table produces a column header for a
paragraph, and every answer read out of it afterwards is attached to a heading that does not exist.

The baseline this replaces is "every table is data", which is what a reader does when it stops at
the file format. It is asserted against rather than assumed to be worse (K3.30).

The rules are ordered, and they are **fail-closed**: positive evidence promotes a table to
data-form, and in its absence the table abstains rather than being demoted to layout. That
asymmetry is deliberate. Calling a data table "layout" silently drops its content from every later
stage; calling it uncertain puts it in front of a human. The two errors do not cost the same.
"""

from __future__ import annotations

from typing import Any

from ..model import Node
from .contract import Analyzer, AnalyzerResult
from .geometry import rng

__all__ = ["GRID_TABLE_FORMATS", "DOCX_TYPES", "GridTablesAnalyzer", "type_table"]

#: Formats whose tables are declared by the file rather than found by segmentation.
GRID_TABLE_FORMATS = ("docx", "pptx")

DOCX_TYPES = ("data-form", "layout", "table_uncertain")

#: Styles that say nothing: the default a document gets when nobody chose one.
_NEUTRAL_STYLES = {None, "", "Normal Table"}


def type_table(table: Node) -> tuple[str, dict[str, Any]]:
    """Ordered hard rules. The first that fires decides, and names itself."""
    cells = [c for c in table.children if c.kind == "cell"]
    rows = int(table.facts.get("n_rows") or 0)
    cols = int(table.facts.get("n_grid_cols") or 0)
    style = table.facts.get("style")
    styled = style not in _NEUTRAL_STYLES

    signals: dict[str, Any] = {
        "n_rows": rows,
        "n_grid_cols": cols,
        "style": style,
        "styled": styled,
        "markers": sum(1 for c in cells if c.facts.get("marker")),
        "bold_cells": sum(1 for c in cells if c.facts.get("bold")),
        "shaded_cells": sum(1 for c in cells if c.facts.get("shaded")),
    }

    if signals["markers"]:
        signals["rule"] = "D1-fillable-markers"
        return "data-form", signals

    if styled and rows >= 2 and cols >= 2:
        signals["rule"] = "D2-ruled-grid-of-two-dimensions"
        return "data-form", signals

    if cols >= 2 and rows >= 2 and (signals["bold_cells"] or signals["shaded_cells"]):
        signals["rule"] = "D3-header-evidence"
        return "data-form", signals

    if not styled and rows == 1:
        signals["rule"] = "D4-unstyled-single-row"
        return "layout", signals

    # No positive evidence either way. Abstain rather than demote: calling a data
    # table "layout" drops its content silently, calling it uncertain surfaces it.
    signals["rule"] = "D5-no-positive-evidence"
    return "table_uncertain", signals


class GridTablesAnalyzer(Analyzer):
    """Type every declared table, and give the data ones a block to analyse."""

    name = "grid_tables"
    version = "0.2.0"          # 0.2.0: presentations join word-processing documents

    def run(self, tree) -> AnalyzerResult:
        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "tables": 0,
            "by_type": {name: 0 for name in DOCX_TYPES},
            "abstained": [],
        }

        for node in tree.walk():
            if node.kind != "table" or node.provenance.source_format not in GRID_TABLE_FORMATS:
                continue
            table_type, signals = type_table(node)
            trace["tables"] += 1
            trace["by_type"][table_type] += 1

            node.facts["table_type"] = table_type
            node.facts["signals"] = signals
            node.facts["block_range"] = rng(
                1, 1, int(node.facts.get("n_rows") or 1), int(node.facts.get("n_grid_cols") or 1)
            )
            # Only a data-form table becomes a block the header rules will read.
            # Abstention by typing: a layout or uncertain table is not analysed,
            # and the reason it was not is on the node (K3.32).
            node.facts["block_type"] = "table" if table_type == "data-form" else "excluded"

            if table_type != "data-form":
                trace["abstained"].append(
                    {
                        "flow": getattr(node.provenance.leaf, "flow", None),
                        "table_index": getattr(node.provenance.leaf, "table_index", None),
                        "type": table_type,
                        "rule": signals["rule"],
                    }
                )

        return AnalyzerResult(tree=tree, trace=trace)
