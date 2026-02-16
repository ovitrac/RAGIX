"""
summary_extract_entities — Stage 2: Backfill Entities on Memory Items

Deterministic entity extraction on existing memory items (no LLM).
Populates item.entities so that the graph build can create 'mentions' edges.

Runs before summary_build_graph in the pipeline.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class SummaryExtractEntitiesKernel(Kernel):
    name = "summary_extract_entities"
    version = "1.0.0"
    category = "summary"
    stage = 2
    description = "Backfill entities on existing memory items (deterministic)"
    requires = ["summary_ingest"]
    provides = ["entity_stats"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Backfill entities on memory items using deterministic extraction."""
        from ragix_core.memory.entity_extract import extract_entities
        from ragix_core.memory.store import MemoryStore

        cfg = input.config
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))

        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
        items = store.list_items(exclude_archived=True, limit=10000)

        items_processed = 0
        items_updated = 0
        items_unchanged = 0
        all_entities: Counter = Counter()

        for item in items:
            items_processed += 1

            # Skip items that already have entities
            if item.entities:
                items_unchanged += 1
                for ent in item.entities:
                    all_entities[ent] += 1
                continue

            # Extract from title + content
            text = f"{item.title} {item.content}"
            entities = extract_entities(text)

            if entities:
                store.update_item(item.id, {"entities": entities})
                items_updated += 1
                for ent in entities:
                    all_entities[ent] += 1
            else:
                items_unchanged += 1

        # Compute top entities
        top_entities = all_entities.most_common(30)

        # Classify entity types
        type_counts: Dict[str, int] = {
            "product": 0, "cve": 0, "version": 0,
            "compliance": 0, "path": 0, "port": 0,
        }
        for ent, count in all_entities.items():
            ent_upper = ent.upper()
            if ent_upper.startswith("CVE-"):
                type_counts["cve"] += count
            elif ent_upper in ("MUST", "SHALL", "SHOULD", "PROHIBITED", "REQUIRED"):
                type_counts["compliance"] += count
            elif "/" in ent and ("tcp" in ent or "udp" in ent):
                type_counts["port"] += count
            elif ent.startswith("/"):
                type_counts["path"] += count
            elif any(c.isdigit() for c in ent) and "." in ent:
                type_counts["version"] += count
            else:
                type_counts["product"] += count

        artifact = {
            "items_processed": items_processed,
            "items_updated": items_updated,
            "items_unchanged": items_unchanged,
            "coverage_pct": round(
                100 * (items_updated + sum(1 for i in items if i.entities))
                / max(items_processed, 1), 1,
            ),
            "unique_entities": len(all_entities),
            "total_entity_refs": sum(all_entities.values()),
            "entity_type_counts": type_counts,
            "top_entities": [
                {"entity": ent, "count": cnt} for ent, cnt in top_entities
            ],
        }

        # Write artifact
        artifact_path = input.workspace / "stage2" / "entity_stats.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        logger.info(
            f"Entity backfill: {items_updated}/{items_processed} updated, "
            f"{len(all_entities)} unique entities → {artifact_path}"
        )

        return artifact

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of entity extraction results."""
        return (
            f"Entities: {data.get('items_updated', 0)}/{data.get('items_processed', 0)} "
            f"items updated ({data.get('coverage_pct', 0)}% coverage), "
            f"{data.get('unique_entities', 0)} unique entities, "
            f"{data.get('total_entity_refs', 0)} total references"
        )
