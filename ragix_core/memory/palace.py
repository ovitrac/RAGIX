"""
Memory Palace — Location Index and Browse API

Provides deterministic browsing of memory items via a hierarchical
location scheme: domain/room/shelf/card.

  - domain: project or corpus identifier
  - room:   topic cluster (derived from tags/entities)
  - shelf:  memory type or document section
  - card:   individual item ID

Complements embedding-based retrieval with structured navigation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem

logger = logging.getLogger(__name__)


class MemoryPalace:
    """
    Hierarchical location index for memory items.

    Provides browse operations (list, get, assign) and
    auto-assignment based on item metadata.
    """

    def __init__(self, store: MemoryStore, default_domain: str = "default"):
        """Initialize memory palace with store and default domain."""
        self._store = store
        self._default_domain = default_domain

    def assign(
        self,
        item: MemoryItem,
        domain: Optional[str] = None,
        room: Optional[str] = None,
        shelf: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Assign a palace location to an item.

        If room/shelf not provided, derives from item metadata:
        - room: primary tag or first entity
        - shelf: item type
        """
        domain = domain or self._default_domain
        room = room or self._derive_room(item)
        shelf = shelf or item.type
        card = item.id

        self._store.write_palace_location(item.id, domain, room, shelf, card)

        location = {
            "domain": domain, "room": room, "shelf": shelf, "card": card,
        }
        logger.debug(f"Palace: assigned {item.id} -> {domain}/{room}/{shelf}/{card}")
        return location

    def auto_assign(self, item: MemoryItem) -> Dict[str, str]:
        """Auto-assign location based on item metadata."""
        return self.assign(item)

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get palace location and item data for a given item."""
        loc = self._store.read_palace_location(item_id)
        if loc is None:
            return None
        item = self._store.read_item(item_id)
        if item is None:
            return None
        return {
            "location": loc,
            "item": item.to_dict(),
            "path": f"{loc['domain']}/{loc['room']}/{loc['shelf']}/{loc['card']}",
        }

    def list_path(self, path: str = "") -> Dict[str, Any]:
        """
        Browse palace at a given path level.

        Examples:
          list_path("")          -> list all domains
          list_path("default")   -> list rooms in domain
          list_path("default/architecture") -> list shelves in room
          list_path("default/architecture/decision") -> list cards on shelf
        """
        parts = [p for p in path.strip("/").split("/") if p] if path else []
        depth = len(parts)

        if depth == 0:
            return self._list_domains()
        elif depth == 1:
            return self._list_rooms(domain=parts[0])
        elif depth == 2:
            return self._list_shelves(domain=parts[0], room=parts[1])
        elif depth >= 3:
            return self._list_cards(
                domain=parts[0], room=parts[1], shelf=parts[2],
            )

    def _list_domains(self) -> Dict[str, Any]:
        """List all distinct domains."""
        all_locs = self._store.list_palace_locations()
        domains = sorted(set(loc["domain"] for loc in all_locs))
        return {
            "level": "domain",
            "path": "/",
            "children": [
                {"name": d, "type": "domain", "path": d}
                for d in domains
            ],
            "count": len(domains),
        }

    def _list_rooms(self, domain: str) -> Dict[str, Any]:
        """List all rooms in a domain."""
        locs = self._store.list_palace_locations(domain=domain)
        rooms = sorted(set(loc["room"] for loc in locs))
        return {
            "level": "room",
            "path": domain,
            "children": [
                {
                    "name": r, "type": "room",
                    "path": f"{domain}/{r}",
                    "item_count": sum(1 for l in locs if l["room"] == r),
                }
                for r in rooms
            ],
            "count": len(rooms),
        }

    def _list_shelves(self, domain: str, room: str) -> Dict[str, Any]:
        """List all shelves in a room."""
        locs = self._store.list_palace_locations(domain=domain, room=room)
        shelves = sorted(set(loc["shelf"] for loc in locs))
        return {
            "level": "shelf",
            "path": f"{domain}/{room}",
            "children": [
                {
                    "name": s, "type": "shelf",
                    "path": f"{domain}/{room}/{s}",
                    "item_count": sum(
                        1 for l in locs if l["shelf"] == s
                    ),
                }
                for s in shelves
            ],
            "count": len(shelves),
        }

    def _list_cards(self, domain: str, room: str, shelf: str) -> Dict[str, Any]:
        """List all cards (items) on a shelf."""
        locs = self._store.list_palace_locations(domain=domain, room=room)
        cards = [
            loc for loc in locs if loc["shelf"] == shelf
        ]
        # Enrich with item titles
        enriched = []
        for card in cards:
            item = self._store.read_item(card["item_id"])
            enriched.append({
                "item_id": card["item_id"],
                "card": card["card"],
                "title": item.title if item else "(unknown)",
                "type": item.type if item else "",
                "tier": item.tier if item else "",
                "path": f"{domain}/{room}/{shelf}/{card['card']}",
            })
        return {
            "level": "card",
            "path": f"{domain}/{room}/{shelf}",
            "children": enriched,
            "count": len(enriched),
        }

    # -- Derivation helpers ------------------------------------------------

    def _derive_room(self, item: MemoryItem) -> str:
        """Derive room name from item metadata."""
        # Use primary tag, or first entity, or "misc"
        if item.tags:
            return item.tags[0].lower().replace(" ", "-")
        if item.entities:
            return item.entities[0].lower().replace(" ", "-")
        return "misc"
