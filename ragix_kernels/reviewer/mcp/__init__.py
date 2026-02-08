"""
KOAS Reviewer MCP tools for Claude Code / Claude Desktop integration.

4 tools:
    review_md_run          Run the full review pipeline
    review_md_status       Query review status and statistics
    review_md_revert       Revert one or more changes
    review_md_show_change  Show details of a specific change

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from ragix_kernels.reviewer.mcp.tools import (
    register_reviewer_tools,
    review_md_revert,
    review_md_run,
    review_md_show_change,
    review_md_status,
)
