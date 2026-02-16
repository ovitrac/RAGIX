"""
summary_budgeted_recall — Stage 2: Per-Domain Budgeted Retrieval

Retrieves memory items with per-domain quotas to ensure balanced coverage.
This is the key kernel that fixes RHEL-domination in summaries.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput


class SummaryBudgetedRecallKernel(Kernel):
    name = "summary_budgeted_recall"
    version = "1.0.0"
    category = "summary"
    stage = 2
    description = "Per-domain budgeted retrieval from memory"
    requires = ["summary_ingest"]
    provides = ["budgeted_items", "domain_coverage"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Retrieve memory items with per-domain quotas to ensure balanced coverage."""
        from ragix_core.memory.budgeted_recall import (
            recall_budgeted,
            format_budgeted_inject,
        )
        from ragix_core.memory.store import MemoryStore

        cfg = input.config
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
        scope = cfg.get("scope", "project")
        max_tokens = cfg.get("max_inject_tokens", 12000)
        min_per_domain = cfg.get("min_items_per_domain", 3)
        max_per_domain = cfg.get("max_items_per_domain", 25)

        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
        budgeted = recall_budgeted(
            store=store,
            scope=scope,
            max_tokens=max_tokens,
            min_items_per_domain=min_per_domain,
            max_items_per_domain=max_per_domain,
        )

        meta = budgeted.pop("_meta", {})

        # Build injection text
        inject_text = format_budgeted_inject(budgeted)

        # Serialize item IDs per domain
        domain_items = {}
        for domain, items in budgeted.items():
            domain_items[domain] = [
                {"id": it.id, "title": it.title, "type": it.type}
                for it in items
            ]

        return {
            "meta": meta,
            "domain_items": domain_items,
            "inject_text": inject_text,
            "inject_tokens": meta.get("total_tokens", 0),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of budgeted recall results."""
        meta = data.get("meta", {})
        return (
            f"Budgeted recall: {meta.get('total_items', 0)} items across "
            f"{meta.get('domains', 0)} domains, "
            f"~{meta.get('total_tokens', 0)} tokens"
        )
