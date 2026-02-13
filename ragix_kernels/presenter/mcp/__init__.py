"""
KOAS Presenter MCP tools for Claude Code / Claude Desktop integration.

3 tools:
    presenter_render    Run the full S1->S2->S3 pipeline
    presenter_export    Export existing workspace to PDF/HTML
    presenter_status    Query workspace state and metadata

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from ragix_kernels.presenter.mcp.tools import (
    register_presenter_tools,
    presenter_render,
    presenter_export,
    presenter_status,
)
