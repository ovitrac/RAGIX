"""
Kernel: Saqqara document substrate
Stage: 1 (Collection)

Reads documents into typed trees with provenance on every node, runs the structure analyzers over
them, and emits the trees, the traces the analyzers produced, and a Merkle root over both. No LLM,
no retrieval, no store — pure computation over files.

This is the ONLY module of the family that exposes a Kernel subclass: the registry discovers
kernels by walking the package, so a second one here would register as an independent kernel.

Two roots are reported, and the difference between them is the point.

`merkle_root` covers the trees. Two runs over the same input produce the same root, which is what
makes a result citable: a claim about a document can name the exact structure it was read from.

`source_root` covers the bytes of the input files and deliberately excludes their paths. Moving a
document does not change what it says; editing one does. A root that followed the path would
report a change where there was none and miss one where there was.

Neither is the hash of this kernel's output file. The shared envelope stamps a wall-clock timestamp
into every output it writes, so two identical runs never produce identical files — correct for the
envelope's own purpose, and the reason reproducibility here is measured on the tree.

Provides:
- documents: one entry per file read — its tree, its traces, its content hash
- merkle_root: over the trees
- source_root: over the input bytes, path-excluded
- report: what was read, what was refused, what was a duplicate

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.merkle import compute_inputs_merkle_root

from ..adapters import adapter_for, read_corpus
from ..analyzers import PIPELINE, OutlineAnalyzer
from ..analyzers.contract import (abstention_records, count_reported,
                                  reports_abstention, tree_abstention_records)
from ..builder import build_tree
from ..model import CANONICAL_JSON

#: Extensions this kernel will pick up when scanning a directory.
READABLE = (".pdf", ".docx", ".xlsx", ".xlsm", ".pptx", ".md", ".markdown")


class SaqqaraKernel(Kernel):
    """
    Read documents into typed trees with provenance, and recognise their structure.

    Configuration options:
        source.path: file or directory to read (required)
        formats: extensions to accept (default: every registered reader)
        promote_outline: also run the opt-in typed-outline pass (default: false)

    Dependencies:
        None — reads source files directly.

    Output:
        documents: per file, the tree, the analyzer traces and the content hash
        merkle_root: over the trees, stable across runs
        source_root: over the input bytes, independent of where they sit
        report: counts of what was read, refused and de-duplicated
    """

    name = "saqqara"
    version = "0.1.0"
    category = "saqqara"
    stage = 1
    description = "Read documents into typed trees with provenance and recognise their structure"

    requires: List[str] = []
    provides = ["document_tree", "traces", "merkle_root"]

    # ------------------------------------------------------------------ input

    def validate_input(self, input: KernelInput) -> List[str]:
        errors = super().validate_input(input)
        if not (input.config or {}).get("source", {}).get("path"):
            errors.append("source.path is required")
        return errors

    @staticmethod
    def _paths(config: Dict[str, Any]) -> List[Path]:
        """Every file under the source, unless the caller narrowed it deliberately.

        Scanning a directory and quietly keeping only the extensions this kernel
        likes would report "read 4" over a folder of forty, and the thirty-six it
        ignored would appear nowhere. Everything found is offered to the readers,
        which refuse what they cannot claim and count the refusal. A caller who
        wants the narrowing says so with `formats`, and that is then a stated
        choice rather than a silent one.
        """
        root = Path(config["source"]["path"]).expanduser()
        if root.is_file():
            return [root]
        found = sorted(p for p in root.rglob("*") if p.is_file())
        wanted = config.get("formats")
        if wanted:
            return [p for p in found if p.suffix.lower() in tuple(wanted)]
        return found

    # ---------------------------------------------------------------- compute

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        config = input.config or {}
        paths = self._paths(config)
        by_path, report = read_corpus(paths)

        documents = []
        for path in report.read:
            observations = by_path.get(path, [])
            adapter = adapter_for(Path(path))
            built = build_tree(
                observations, path, adapter.format, adapter.format, adapter.version
            )
            tree = built.tree

            traces = {"builder": built.trace}
            for analyzer_class in PIPELINE:
                analyzer = analyzer_class()
                result = analyzer.run(tree)
                tree = result.tree
                traces[analyzer.name] = result.trace

            if config.get("promote_outline"):
                # Opt-in, and after sections: a promotion is an inference, and
                # running it first would feed this package its own conclusions.
                promoted = OutlineAnalyzer().run(tree)
                tree = promoted.tree
                traces["outline"] = promoted.trace

            documents.append(
                {
                    "path": path,
                    "format": adapter.format,
                    "sha256": self._digest(Path(path)),
                    "tree": tree.to_dict(),
                    "traces": traces,
                }
            )

        return {
            "documents": documents,
            "merkle_root": self._tree_root(documents),
            "source_root": self._source_root(documents),
            "report": {
                "counts": report.counts,
                "refusals": [
                    {"path": r.path, "reason": r.reason, "detail": r.detail}
                    for r in report.refusals
                ],
                "duplicates": report.duplicates,
                "abstentions": self._abstentions(documents),
            },
        }

    # ------------------------------------------------------ abstention register

    @staticmethod
    def _abstentions(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Every abstention the run produced, listed rather than counted (K4.3).

        A refusal has always been reported with its path and its reason. An
        abstention was reported as a number in one line of prose, so a document
        that abstained could not be found from the output — only by re-running.
        The two are the same kind of fact and are now listed the same way.

        Each record carries the document it is about beside what the analyzer
        kept: who abstained, where (or `None` where the producer kept only a
        count), why, and the signals it saw.
        """
        register: List[Dict[str, Any]] = []
        for document in documents:
            for name, trace in (document.get("traces") or {}).items():
                if not isinstance(trace, dict) or not reports_abstention(trace):
                    continue
                for record in abstention_records(name, trace):
                    register.append({"path": document["path"], **record})
            # A reader that declines a page abstains as an analyzer does; its
            # record lives in the tree rather than in a trace (K6.19).
            for record in tree_abstention_records(document.get("tree") or {}):
                register.append({"path": document["path"], **record})
        return register

    # ------------------------------------------------------------------ roots

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _tree_root(documents: List[Dict[str, Any]]) -> str:
        """A root over the trees, stable across runs of the same input."""
        return compute_inputs_merkle_root(
            [
                {
                    "file_path": document["path"],
                    "chunk_index": 0,
                    "content": json.dumps(document["tree"], **CANONICAL_JSON),
                }
                for document in documents
            ]
        )

    @staticmethod
    def _source_root(documents: List[Dict[str, Any]]) -> str:
        """A root over the bytes read, with the paths deliberately left out.

        Sorting by content hash rather than by name is what makes it independent
        of where the files sit: the same documents in another directory, or under
        other names, produce the same root.
        """
        return compute_inputs_merkle_root(
            [
                {"file_path": "", "chunk_index": index, "content_hash": digest}
                for index, digest in enumerate(sorted(d["sha256"] for d in documents))
            ]
        )

    # ---------------------------------------------------------------- summary

    def summarize(self, data: Dict[str, Any]) -> str:
        """One line on what was read, what was refused, and what did not decide."""
        documents = data.get("documents") or []
        report = (data.get("report") or {}).get("counts") or {}

        nodes = sum(self._count(d["tree"]["root"]) for d in documents)

        # Derived from the register, never counted a second way: a number and a
        # list that are computed apart are a number and a list that disagree.
        # A payload from before K4.3 carries no register, so one is built from
        # its traces by the same function rather than by a second rule.
        register = (data.get("report") or {}).get("abstentions")
        if register is None:
            register = self._abstentions(documents)
        abstentions = sum(record.get("count", 1) for record in register)

        drops = 0
        for document in documents:
            for trace in (document.get("traces") or {}).values():
                if isinstance(trace, dict):
                    # Not `int(...)`: an analyzer reports what it let go in the
                    # shape that suits what it saw, and four are in use. The cast
                    # was right for two of them and raised for a third (K4.3).
                    drops += count_reported(trace.get("dropped"))

        return (
            f"{len(documents)} document(s), {nodes} nodes. "
            f"Refused {report.get('refused', 0)}, duplicate {report.get('duplicate', 0)}. "
            f"{abstentions} abstention(s), {drops} drop(s) counted. "
            f"root {(data.get('merkle_root') or '')[:12]}."
        )

    @classmethod
    def _count(cls, node: Dict[str, Any]) -> int:
        return 1 + sum(cls._count(child) for child in node.get("children", []))
