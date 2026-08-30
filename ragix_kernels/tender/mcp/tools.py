"""
tender MCP tools.

    koas_tender_probe   - report what a document store holds
    koas_tender_status  - what the configuration resolves to, without opening it

Both delegate to the CLI so the two surfaces cannot answer differently.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def register_tender_tools(mcp_server) -> None:
    """Register the tender tools with a FastMCP server instance."""

    @mcp_server.tool()
    def koas_tender_probe(config: str = "") -> Dict[str, Any]:
        """
        Report what a document store holds, without modifying it.

        Opens the store named by the configuration, asks it what it contains, and
        returns those counts. It never builds a store and never writes to one.

        Parameters
        ----------
        config : str
            Optional path to a tender.yaml overlaying the packaged defaults.

        Returns
        -------
        dict
            {"probe": {"store_path", "documents", "chunks", "dense_enabled",
            "dense"}} — identical to `tenderctl probe --json`. `dense_enabled` is
            a field rather than something inferred from a count: a store with no
            vectors because nothing was embedded and one with none because
            embedding failed are different situations.
        """
        try:
            import argparse
            import contextlib
            import io
            import json as _json

            from ragix_kernels.tender.cli.tenderctl import cmd_probe

            args = argparse.Namespace(config=config or None, json=True, verbose=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = cmd_probe(args)
            if code != 0:
                return {"error": f"probe failed with code {code}"}
            return _json.loads(buffer.getvalue() or "{}")
        except Exception as e:
            return {"error": str(e)}

    @mcp_server.tool()
    def koas_tender_status(config: str = "") -> Dict[str, Any]:
        """
        What the configuration resolves to, without opening the store.

        Answers "which store did you mean", which is the question a refusal from
        `koas_tender_probe` raises.

        Parameters
        ----------
        config : str
            Optional path to a tender.yaml.

        Returns
        -------
        dict
            {"store_path", "exists", "config"}
        """
        try:
            from pathlib import Path as _Path

            from ragix_kernels.tender.config import load_config

            resolved = load_config(config or None)
            store = _Path(resolved.get("store.path"))
            return {"store_path": str(store), "exists": store.is_file(),
                    "config": resolved.to_dict()}
        except Exception as e:
            return {"error": str(e)}
