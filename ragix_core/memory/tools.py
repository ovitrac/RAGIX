"""
Memory Tool API — JSON Action Dispatcher

Stable JSON I/O for the memory.* tool namespace:
    memory.propose   - Submit memory candidates (goes through policy)
    memory.write     - Direct write (policy-checked)
    memory.search    - Hybrid search
    memory.read      - Read items by ID
    memory.update    - Patch item fields
    memory.link      - Create link between items
    memory.consolidate - Trigger consolidation pipeline
    memory.palace.list - Browse palace locations
    memory.palace.get  - Get specific palace location

Every action writes an audit event to memory_events.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ragix_core.memory.config import MemoryConfig
from ragix_core.memory.embedder import MemoryEmbedder, create_embedder
from ragix_core.memory.policy import MemoryPolicy, PolicyVerdict
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import (
    MemoryItem,
    MemoryLink,
    MemoryProposal,
    _generate_id,
    content_hash,
)

logger = logging.getLogger(__name__)


class MemoryToolDispatcher:
    """
    Central dispatcher for all memory.* tool actions.

    Integrates store, policy engine, and embedder.
    All actions are audited via store event logging.
    """

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy,
        embedder: MemoryEmbedder,
        config: Optional[MemoryConfig] = None,
    ):
        """Initialize dispatcher with store, policy engine, embedder, and config."""
        self._store = store
        self._policy = policy
        self._embedder = embedder
        self._config = config or MemoryConfig()
        # Lazy imports for recall/qsearch/palace/consolidate
        self._recall = None
        self._qsearch = None
        self._palace = None
        self._consolidator = None

    @property
    def store(self) -> MemoryStore:
        """Return the underlying memory store."""
        return self._store

    @property
    def policy(self) -> MemoryPolicy:
        """Return the policy engine."""
        return self._policy

    @property
    def embedder(self) -> MemoryEmbedder:
        """Return the embedding backend."""
        return self._embedder

    def dispatch(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a memory.* action to the appropriate handler.

        Args:
            action: dotted action name (e.g. "memory.propose", "memory.search")
            params: action parameters dict

        Returns:
            Result dict with at minimum {"status": "ok"|"error", ...}
        """
        # Strip "memory." prefix if present
        if action.startswith("memory."):
            action = action[len("memory."):]

        handlers = {
            "propose": self._handle_propose,
            "write": self._handle_write,
            "search": self._handle_search,
            "read": self._handle_read,
            "update": self._handle_update,
            "link": self._handle_link,
            "consolidate": self._handle_consolidate,
            "palace.list": self._handle_palace_list,
            "palace.get": self._handle_palace_get,
        }

        handler = handlers.get(action)
        if handler is None:
            return {"status": "error", "message": f"Unknown action: memory.{action}"}

        try:
            return handler(params)
        except Exception as e:
            logger.exception(f"Error in memory.{action}")
            return {"status": "error", "message": str(e)}

    # -- Handlers ----------------------------------------------------------

    def _handle_propose(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory proposals from LLM.
        Each proposal goes through policy evaluation.
        """
        items_data = params.get("items", [])
        if not items_data:
            return {"status": "error", "message": "No items in proposal"}

        results = []
        for item_d in items_data:
            proposal = MemoryProposal.from_dict(item_d)
            verdict = self._policy.evaluate_proposal(proposal)

            if verdict.rejected:
                results.append({
                    "title": proposal.title,
                    "action": "rejected",
                    "reasons": verdict.reasons,
                })
                continue

            # Convert to MemoryItem
            if verdict.action == "quarantine":
                item = proposal.to_memory_item(
                    tier=verdict.forced_tier or "stm",
                    confidence=0.3,
                )
                item.validation = verdict.forced_validation or "unverified"
                item.expires_at = verdict.forced_expires_at
                # V3.3: instructional-content quarantine → non-injectable
                if verdict.forced_non_injectable:
                    item.injectable = False
            else:
                item = proposal.to_memory_item(tier="stm")

            # Policy check on the resulting item
            item_verdict = self._policy.evaluate_item(item)
            if item_verdict.rejected:
                results.append({
                    "title": proposal.title,
                    "action": "rejected",
                    "reasons": item_verdict.reasons,
                })
                continue

            # Store item
            stored = self._store.write_item(item, reason="propose")

            # Generate and store embedding
            try:
                vec = self._embedder.embed_text(f"{item.title} {item.content}")
                self._store.write_embedding(
                    item.id, vec, self._embedder.model_name, self._embedder.dimension,
                )
            except Exception as e:
                logger.warning(f"Embedding failed for {item.id}: {e}")

            results.append({
                "id": stored.id,
                "title": stored.title,
                "action": verdict.action,
                "tier": stored.tier,
            })

        accepted = sum(1 for r in results if r.get("action") != "rejected")
        return {
            "status": "ok",
            "accepted": accepted,
            "rejected": len(results) - accepted,
            "items": results,
        }

    def _handle_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Direct write of a memory item (policy-checked)."""
        item = MemoryItem.from_dict(params)
        verdict = self._policy.evaluate_item(item)
        if verdict.rejected:
            return {
                "status": "rejected",
                "reasons": verdict.reasons,
            }

        stored = self._store.write_item(item, reason="direct_write")

        # Embedding
        try:
            vec = self._embedder.embed_text(f"{item.title} {item.content}")
            self._store.write_embedding(
                item.id, vec, self._embedder.model_name, self._embedder.dimension,
            )
        except Exception as e:
            logger.warning(f"Embedding failed for {item.id}: {e}")

        return {"status": "ok", "id": stored.id}

    def _handle_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid search across memory items."""
        query = params.get("query", "")
        tags = params.get("tags", [])
        tier = params.get("tier")
        type_filter = params.get("type")
        scope = params.get("scope")
        k = params.get("k", self._config.recall.catalog_k)

        # Lazy import recall engine
        if self._recall is None:
            from ragix_core.memory.recall import RecallEngine
            self._recall = RecallEngine(
                store=self._store,
                embedder=self._embedder,
                config=self._config.recall,
            )

        results = self._recall.search(
            query=query, tags=tags, tier=tier,
            type_filter=type_filter, scope=scope, k=k,
        )
        return {
            "status": "ok",
            "count": len(results),
            "items": [item.format_catalog_entry() for item in results],
        }

    def _handle_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read items by ID."""
        ids = params.get("ids", [])
        if not ids:
            return {"status": "error", "message": "No ids provided"}

        items = self._store.read_items(ids)
        return {
            "status": "ok",
            "items": [item.to_dict() for item in items],
        }

    def _handle_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update (patch) an existing item."""
        item_id = params.get("id")
        if not item_id:
            return {"status": "error", "message": "Missing id"}

        patch = {k: v for k, v in params.items() if k != "id"}
        updated = self._store.update_item(item_id, patch)
        if updated is None:
            return {"status": "error", "message": f"Item not found: {item_id}"}

        # Re-embed if content changed
        if "content" in patch or "title" in patch:
            try:
                vec = self._embedder.embed_text(
                    f"{updated.title} {updated.content}"
                )
                self._store.write_embedding(
                    item_id, vec, self._embedder.model_name, self._embedder.dimension,
                )
            except Exception as e:
                logger.warning(f"Re-embedding failed for {item_id}: {e}")

        return {"status": "ok", "id": item_id}

    def _handle_link(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a typed link between two items."""
        link = MemoryLink(
            src_id=params.get("src_id", ""),
            dst_id=params.get("dst_id", ""),
            rel=params.get("rel", ""),
        )
        if not link.src_id or not link.dst_id or not link.rel:
            return {"status": "error", "message": "Missing src_id, dst_id, or rel"}

        self._store.write_link(link)
        return {"status": "ok", "src_id": link.src_id, "dst_id": link.dst_id}

    def _handle_consolidate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger consolidation pipeline."""
        if self._consolidator is None:
            from ragix_core.memory.consolidate import ConsolidationPipeline
            self._consolidator = ConsolidationPipeline(
                store=self._store,
                embedder=self._embedder,
                config=self._config.consolidate,
            )

        scope = params.get("scope", "project")
        tiers = params.get("tiers", ["stm"])
        promote = params.get("promote", True)

        result = self._consolidator.run(
            scope=scope, tiers=tiers, promote=promote,
        )
        return {"status": "ok", **result}

    def _handle_palace_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List palace locations."""
        path = params.get("path", "")
        parts = path.strip("/").split("/") if path else []
        domain = parts[0] if len(parts) > 0 else None
        room = parts[1] if len(parts) > 1 else None

        locations = self._store.list_palace_locations(domain=domain, room=room)
        return {"status": "ok", "locations": locations}

    def _handle_palace_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific palace location + item."""
        item_id = params.get("item_id")
        if not item_id:
            # Try to parse location path
            location = params.get("location", "")
            # Find by location path - would need palace module
            return {"status": "error", "message": "Provide item_id or location"}

        loc = self._store.read_palace_location(item_id)
        item = self._store.read_item(item_id)
        if loc is None or item is None:
            return {"status": "error", "message": f"Not found: {item_id}"}

        return {
            "status": "ok",
            "location": loc,
            "item": item.to_dict(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_dispatcher(config: Optional[MemoryConfig] = None) -> MemoryToolDispatcher:
    """Create a fully wired MemoryToolDispatcher from config."""
    cfg = config or MemoryConfig()

    store = MemoryStore(
        db_path=cfg.store.db_path,
        wal_mode=cfg.store.wal_mode,
    )
    policy = MemoryPolicy(config=cfg.policy)
    embedder = create_embedder(
        backend=cfg.embedder.backend,
        model=cfg.embedder.model,
        dimension=cfg.embedder.dimension,
        ollama_url=cfg.embedder.ollama_url,
        mock_seed=cfg.embedder.mock_seed,
    )

    return MemoryToolDispatcher(
        store=store, policy=policy, embedder=embedder, config=cfg,
    )
