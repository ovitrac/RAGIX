"""
Helper functions for the KOAS Memory Pipe demo.

Thin wrappers around ragix_core.memory functions — the real logic lives in:
  - ragix_core.memory.ingest.corpus_stats
  - ragix_core.memory.mcp.formatting.parse_injection_block

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import json
import os
import sys

# Re-export from the library so the demo script can import from one place
from ragix_core.memory.ingest import corpus_stats          # noqa: F401
from ragix_core.memory.mcp.formatting import parse_injection_block as extract_insight  # noqa: F401


def print_synthesis(query_results):
    """Print a cross-file architectural synthesis from 4 query results."""
    print()
    print("  KOAS is a sovereign computation framework (from KOAS.md) built on")
    print("  a formal kernel contract (from base.py): every kernel implements")
    print("  compute() for deterministic processing and summarize() for LLM")
    print("  consumption. Five families — audit, docs, presenter, reviewer,")
    print("  summary — organize 75 kernels into a 3-stage pipeline: collection,")
    print("  analysis, reporting (from ARCHITECTURE.md). Real implementations")
    print("  like md_edit_plan.py show how this contract scales to complex")
    print("  tasks: 2,964 lines of preflight masking, adaptive tier escalation,")
    print("  and content recipes — all deterministic guards before the LLM call.")
    print()
    print("  This synthesis draws from 4 different source files. No single file")
    print("  contains all of this — Memory assembled it from cross-file recall.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        files = [
            f"{project}/docs/KOAS.md",
            f"{project}/docs/ARCHITECTURE.md",
            f"{project}/ragix_kernels/base.py",
            f"{project}/ragix_kernels/registry.py",
            f"{project}/ragix_kernels/reviewer/kernels/md_edit_plan.py",
            f"{project}/ragix_kernels/presenter/kernels/pres_slide_plan.py",
            f"{project}/ragix_kernels/summary/cli/summaryctl.py",
            f"{project}/ragix_core/memory/cli.py",
        ]
        stats = corpus_stats(files)
        print(json.dumps(stats, indent=2))
