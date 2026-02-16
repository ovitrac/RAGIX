"""
summary_verify — Stage 3: Citation Verification

Deterministic post-processor that verifies [MID: xxx] citations and
builds the citation map.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput


class SummaryVerifyKernel(Kernel):
    name = "summary_verify"
    version = "1.0.0"
    category = "summary"
    stage = 3
    description = "Verify MID citations in generated summary"
    requires = ["summary_generate"]
    provides = ["citation_report", "citation_map"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Verify [MID: xxx] citations in generated summary against memory store."""
        from ragix_core.memory.citation_verify import (
            verify_citations,
            build_citation_map,
        )
        from ragix_core.memory.store import MemoryStore

        cfg = input.config
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
        scope = cfg.get("scope", "project")

        # Load generated sections
        gen_file = input.dependencies.get("summary_generate")
        if gen_file and gen_file.exists():
            gen_data = json.loads(gen_file.read_text())["data"]
        else:
            raise RuntimeError("Missing summary_generate dependency")

        # Combine all sections into single text for verification
        sections = gen_data.get("sections", [])
        full_text = "\n\n".join(s["content"] for s in sections)

        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)

        # Verify citations
        report = verify_citations(full_text, store, scope)

        # Build citation map
        citation_map = build_citation_map(full_text, store)

        return {
            "citation_report": report.to_dict(),
            "citation_map": citation_map,
            "sections_verified": len(sections),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of citation verification results."""
        cr = data.get("citation_report", {})
        return (
            f"Verified {cr.get('total_bullets', 0)} bullets: "
            f"{cr.get('bullets_with_citation', 0)} cited, "
            f"{cr.get('valid_mids', 0)} valid MIDs, "
            f"{cr.get('invalid_mids', 0)} invalid"
        )
