"""
saqqara MCP tools.

    koas_saqqara_run     - read documents into typed trees and recognise structure
    koas_saqqara_status  - read back what a previous run found
    koas_saqqara_index   - chunk what was read into a store, embedding what is missing
    koas_saqqara_search  - query that store, with both lane ranks and every citation

The tool names, parameters and return shapes are the frozen public envelope: this
module is where they are DEFINED, not a second definition of them. They moved here
from the monolithic server so the family carries its own surface, exactly as the
other families do, and the server registers it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def register_saqqara_tools(mcp_server) -> None:
    """Register the saqqara tools with a FastMCP server instance."""

    @mcp_server.tool()
    def koas_saqqara_run(
        source: str,
        workspace: str = "",
        formats: str = "",
        promote_outline: bool = False,
    ) -> Dict[str, Any]:
        """
        Read documents into typed trees with provenance and recognise their structure.

        Reads pdf, word-processing, spreadsheet, presentation and markdown files into one
        typed tree each, with an exact citation on every node, then runs the structure
        analyzers over them. Pure computation: no model, no retrieval, no store.

        Parameters
        ----------
        source : str
            File or directory to read.
        workspace : str
            Where to write the result (default: a temporary workspace).
        formats : str
            Comma-separated extensions to narrow the scan, e.g. ".docx,.xlsx".
            Left empty, every file found is offered to the readers and whatever
            they cannot claim is counted as a refusal rather than skipped quietly.
        promote_outline : bool
            Also run the opt-in typed-outline pass, which ADDS inferred headings
            beside the paragraphs they came from. Off by default: a promotion is an
            inference, and it is not enabled on a caller's behalf.

        Returns
        -------
        dict
            {"success", "summary", "documents", "merkle_root", "source_root", "report"}
            `merkle_root` is stable across runs of the same input; `source_root`
            follows the bytes read and not where they sit.
        """
        try:
            from pathlib import Path as _Path

            from ragix_kernels.base import KernelInput
            from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel

            target = _Path(workspace) if workspace else _Path(source).resolve().parent
            config: Dict[str, Any] = {"source": {"path": source}}
            if formats.strip():
                config["formats"] = [f.strip() for f in formats.split(",") if f.strip()]
            if promote_outline:
                config["promote_outline"] = True

            output = SaqqaraKernel().run(KernelInput(workspace=target, config=config))
            return {
                "success": output.success,
                "summary": output.summary,
                "output_file": str(output.output_file),
                "documents": [
                    {
                        "path": d["path"],
                        "format": d["format"],
                        "sha256": d["sha256"],
                        "nodes": SaqqaraKernel._count(d["tree"]["root"]),
                    }
                    for d in (output.data.get("documents") or [])
                ],
                "merkle_root": output.data.get("merkle_root"),
                "source_root": output.data.get("source_root"),
                "report": output.data.get("report"),
                "errors": output.errors,
            }
        except Exception as e:
            return {"error": str(e)}


    @mcp_server.tool()
    def koas_saqqara_status(workspace: str) -> Dict[str, Any]:
        """
        What a previous saqqara run found, read back from its stored result.

        Reports what was read, what was refused and why, what each analyzer abstained
        on, and the roots — so a caller can check a citation without re-reading the
        corpus.

        Parameters
        ----------
        workspace : str
            The workspace a previous run wrote to.

        Returns
        -------
        dict
            {"documents", "merkle_root", "source_root", "report", "abstentions"}
        """
        try:
            import json as _json
            from pathlib import Path as _Path

            stored = _Path(workspace) / "stage1" / "saqqara.json"
            if not stored.is_file():
                return {"error": f"no saqqara result in {workspace}"}

            payload = _json.loads(stored.read_text())
            data = payload.get("data", {})

            abstentions = []
            for document in data.get("documents", []):
                for name, trace in (document.get("traces") or {}).items():
                    if not isinstance(trace, dict):
                        continue
                    for entry in trace.get("abstentions", []) or []:
                        abstentions.append({"document": document["path"], "analyzer": name, **entry})

            return {
                "meta": payload.get("_meta"),
                "documents": [
                    {"path": d["path"], "format": d["format"], "sha256": d["sha256"]}
                    for d in data.get("documents", [])
                ],
                "merkle_root": data.get("merkle_root"),
                "source_root": data.get("source_root"),
                "report": data.get("report"),
                "abstentions": abstentions,
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp_server.tool()
    def koas_saqqara_index(workspace: str, config: str = "") -> Dict[str, Any]:
        """
        Chunk what a previous read produced into a queryable store.

        Reads the trees stage 1 wrote, cuts them along their own structure, stores
        them in one SQLite file and embeds only the chunks that model has not seen.
        With `embedder.provider: none` the store is lexical-only and says so — no
        zero vectors are written.

        Parameters
        ----------
        workspace : str
            The workspace a previous saqqara run wrote to.
        config : str
            Optional path to a saqqara.yaml overlaying the packaged defaults.

        Returns
        -------
        dict
            {"success", "summary", "documents", "status", "embedded", "skipped"}
            `status.dense` states the model and vector count, or that dense is
            disabled and why.
        """
        try:
            from pathlib import Path as _Path

            from ragix_kernels.base import KernelInput
            from ragix_kernels.saqqara.kernels.saqqara_index import SaqqaraIndexKernel

            target = _Path(workspace)
            target.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {}
            if config.strip():
                payload["config"] = config
            read = target / "stage1" / "saqqara.json"
            if not read.is_file():
                return {"error": f"nothing has been read into {workspace}; "
                                 "run koas_saqqara_run first"}
            output = SaqqaraIndexKernel().run(KernelInput(
                workspace=target, config=payload,
                dependencies={"document_tree": read}))
            return {
                "success": output.success,
                "summary": output.summary,
                "output_file": str(output.output_file),
                "documents": output.data.get("documents") or [],
                "status": output.data.get("status") or {},
                "embedded": output.data.get("embedded", 0),
                "skipped": output.data.get("skipped", 0),
                "errors": output.errors,
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp_server.tool()
    def koas_saqqara_search(workspace: str, query: str, k: int = 10,
                            config: str = "") -> Dict[str, Any]:
        """
        Query the store, keeping both lane ranks and every hit's citation.

        A hit carries `dense_rank`, `lexical_rank` and `final_rank` separately: the
        fused rank never replaces the ranks it was computed from, and a hit found
        by one lane carries null for the other rather than a worst-case number.
        Every hit resolves to the provenance chain of the nodes it was cut from.

        Parameters
        ----------
        workspace : str
            The workspace holding the store.
        query : str
            What to look for.
        k : int
            How many hits to return.
        config : str
            Optional path to a saqqara.yaml.

        Returns
        -------
        dict
            {"hits": [...], "dense": str} — `hits` is exactly what `saqqaractl
            search --json` prints, so the two surfaces cannot drift apart.
        """
        try:
            import argparse
            import contextlib
            import io
            import json as _json

            from ragix_kernels.saqqara.cli.saqqaractl import cmd_search

            args = argparse.Namespace(workspace=workspace, query=query, config=config or None,
                                      top_k=k, json=True, verbose=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = cmd_search(args)
            if code != 0:
                return {"error": f"search failed with code {code}"}
            return {"hits": _json.loads(buffer.getvalue() or "[]")}
        except Exception as e:
            return {"error": str(e)}
