"""
Chat Pipeline Integration — Memory Middleware

Hooks into the RAGIX chat loop at 4 points:
  1. pre_call    — inject relevant memory into context
  2. post_call   — parse proposals, govern, store accepted items
  3. pre_return  — Q*-search recall pass before final response
  4. intercept   — detect and handle explicit recall requests

These hooks divert the Ollama process behavior without modifying Ollama itself.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ragix_core.memory.config import MemoryConfig
from ragix_core.memory.proposer import MemoryProposer
from ragix_core.memory.tools import MemoryToolDispatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recall intent detection patterns
# ---------------------------------------------------------------------------

_RECALL_PATTERNS = [
    re.compile(r"\brecall\b", re.IGNORECASE),
    re.compile(r"what\s+do\s+we\s+know\s+about\b", re.IGNORECASE),
    re.compile(r"from\s+memory\b", re.IGNORECASE),
    re.compile(r"as\s+we\s+decided\s+earlier\b", re.IGNORECASE),
    re.compile(r"remember\s+(?:when|that|what)\b", re.IGNORECASE),
    re.compile(r"previous(?:ly)?\s+(?:discussed|decided|noted)\b", re.IGNORECASE),
]


class MemoryMiddleware:
    """
    Memory middleware for the chat pipeline.

    Integrates with the LLM orchestration loop to provide persistent
    memory across turns without modifying Ollama.

    Consolidation triggers (both deterministic):
      1. Context fraction: consolidate when injected+STM tokens >= fraction of ctx limit
      2. STM count: consolidate when STM item count exceeds threshold
    """

    def __init__(
        self,
        dispatcher: MemoryToolDispatcher,
        config: Optional[MemoryConfig] = None,
    ):
        """Initialize middleware with tool dispatcher and memory config."""
        self._dispatcher = dispatcher
        self._config = config or MemoryConfig()
        self._proposer = MemoryProposer(config=self._config.proposer)
        # Track context for the current turn
        self._last_query: str = ""
        self._last_injected_ids: List[str] = []
        # Running STM token estimate (for context-fraction trigger)
        self._stm_tokens: int = 0
        self._consolidation_count: int = 0

    @property
    def proposer(self) -> MemoryProposer:
        """Return the LLM output parser for memory proposals."""
        return self._proposer

    @property
    def system_instruction(self) -> str:
        """System instruction to append to LLM context for memory awareness."""
        return self._proposer.system_instruction

    # -- Hook 1: Pre-call (inject memory into context) ---------------------

    def pre_call(
        self,
        user_query: str,
        system_context: str = "",
        turn_id: str = "",
    ) -> str:
        """
        Inject relevant memory items into the system context.

        Called before the LLM call. Returns augmented system context.
        """
        self._last_query = user_query
        self._last_injected_ids = []

        mode = self._config.recall.mode
        if mode not in ("inject", "hybrid"):
            return system_context

        # Search for relevant memory
        result = self._dispatcher.dispatch("search", {
            "query": user_query,
            "k": min(5, self._config.recall.catalog_k),
        })

        if result.get("status") != "ok" or not result.get("items"):
            return system_context

        # Build injection text within token budget
        inject_parts = []
        budget = self._config.recall.inject_budget_tokens
        used = 0

        for item_entry in result["items"]:
            item_id = item_entry["id"]
            # Read full item
            full = self._dispatcher.store.read_item(item_id)
            if full is None:
                continue
            inject_text = full.format_inject()
            # Rough token estimate: chars / 4
            tokens_est = len(inject_text) // 4
            if used + tokens_est > budget:
                break
            inject_parts.append(inject_text)
            self._last_injected_ids.append(item_id)
            used += tokens_est

        if not inject_parts:
            return system_context

        memory_block = "\n\n".join(inject_parts)
        augmented = (
            f"{system_context}\n\n"
            f"--- Relevant Memory ---\n{memory_block}\n--- End Memory ---"
        )
        logger.info(
            f"Injected {len(inject_parts)} memory item(s) "
            f"(~{used} tokens) into context"
        )
        return augmented

    # -- Hook 2: Post-call (parse proposals, govern, store) ----------------

    def post_call(
        self,
        response_text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        turn_id: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parse memory proposals from LLM response, apply policy, store.

        Returns (cleaned_response_text, proposal_summary).
        """
        cleaned, proposals = self._proposer.extract_proposals(
            response_text=response_text,
            tool_calls=tool_calls,
        )

        if not proposals:
            return cleaned, {"proposals_found": 0, "consolidation_triggered": False}

        # Submit proposals through the tool dispatcher
        items_data = [p.to_dict() for p in proposals]
        result = self._dispatcher.dispatch("propose", {"items": items_data})

        # Update running STM token estimate
        for item_info in result.get("items", []):
            if item_info.get("action") != "rejected":
                self._stm_tokens += 100  # rough per-item estimate

        summary = {
            "proposals_found": len(proposals),
            "accepted": result.get("accepted", 0),
            "rejected": result.get("rejected", 0),
            "items": result.get("items", []),
            "consolidation_triggered": False,
        }

        # Check context-fraction consolidation trigger
        if self._should_consolidate():
            logger.info(
                f"Context-fraction consolidation triggered "
                f"(~{self._stm_tokens} STM tokens, "
                f"consolidation #{self._consolidation_count + 1})"
            )
            self._trigger_consolidation()
            summary["consolidation_triggered"] = True

        logger.info(
            f"Post-call: {summary['accepted']} accepted, "
            f"{summary['rejected']} rejected out of {len(proposals)} proposals"
        )

        return cleaned, summary

    # -- Hook 3: Pre-return (Q*-search recall pass) ------------------------

    def pre_return(
        self,
        user_query: str,
        assistant_response: str,
        turn_id: str = "",
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Run Q*-search recall before returning final response.

        In 'catalog' or 'hybrid' mode, returns a memory catalog.
        In 'inject' mode, this is a no-op (injection happens in pre_call).
        """
        mode = self._config.recall.mode

        if mode == "inject":
            return assistant_response, None

        if not self._config.qsearch.enabled:
            return assistant_response, None

        # Run Q*-search
        try:
            from ragix_core.memory.qsearch import QSearchEngine
            engine = QSearchEngine(
                store=self._dispatcher.store,
                embedder=self._dispatcher.embedder,
                config=self._config.qsearch,
            )
            search_result = engine.search(user_query)
        except ImportError:
            logger.warning("qsearch module not available; skipping pre-return recall")
            return assistant_response, None
        except Exception as e:
            logger.warning(f"Q*-search failed: {e}")
            return assistant_response, None

        if not search_result.get("items"):
            return assistant_response, None

        catalog = {
            "memory_catalog": search_result["items"],
            "score": search_result.get("best_score", 0),
        }

        if mode == "hybrid":
            # Also inject top items into response context
            inject_items = search_result["items"][:3]
            if inject_items:
                ids = [i["id"] for i in inject_items if "id" in i]
                items = self._dispatcher.store.read_items(ids)
                if items:
                    memory_block = "\n".join(i.format_inject() for i in items)
                    assistant_response = (
                        f"{assistant_response}\n\n"
                        f"[Memory context available: {len(items)} item(s)]"
                    )

        return assistant_response, catalog

    # -- Hook 4: Intercept recall requests ---------------------------------

    def intercept_recall(
        self,
        user_query: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect explicit memory recall requests.

        Returns recall results if detected, None otherwise.
        Tool calls take precedence over text pattern detection.
        """
        # Check tool calls first
        if tool_calls:
            for call in tool_calls:
                action = call.get("action", "") or call.get("name", "")
                if action.startswith("memory.search") or action.startswith("memory.browse"):
                    params = call.get("arguments", call)
                    if isinstance(params, str):
                        import json
                        try:
                            params = json.loads(params)
                        except Exception:
                            params = {"query": user_query}
                    return self._dispatcher.dispatch("search", params)

        # Text-based intent detection (fallback)
        if self._detect_recall_intent(user_query):
            logger.info(f"Recall intent detected in query")
            return self._dispatcher.dispatch("search", {
                "query": user_query,
                "k": self._config.recall.catalog_k,
            })

        return None

    # -- Consolidation triggers --------------------------------------------

    def _should_consolidate(self) -> bool:
        """
        Check if consolidation should be triggered.

        Two deterministic conditions (either triggers):
          1. STM token estimate >= ctx_fraction_trigger * ctx_limit_tokens
          2. STM count >= stm_threshold
        """
        cfg = self._config.consolidate
        ctx_limit = cfg.ctx_limit_tokens
        fraction = cfg.ctx_fraction_trigger

        # Context fraction trigger
        if ctx_limit > 0 and fraction > 0:
            if self._stm_tokens / ctx_limit >= fraction:
                return True

        # STM count trigger
        stm_count = self._dispatcher.store.count_items(tier="stm")
        if stm_count >= cfg.stm_threshold:
            return True

        return False

    def _trigger_consolidation(self, scope: str = "project") -> None:
        """Run consolidation and reset STM token counter."""
        result = self._dispatcher.dispatch("consolidate", {
            "scope": scope,
            "tiers": ["stm"],
            "promote": True,
        })
        self._consolidation_count += 1
        merged = result.get("items_merged", 0)
        # Reduce token estimate (merged items are smaller)
        self._stm_tokens = max(0, self._stm_tokens - (merged * 80))
        logger.info(
            f"Consolidation #{self._consolidation_count}: "
            f"merged={merged}, promoted={result.get('items_promoted', 0)}, "
            f"remaining ~{self._stm_tokens} STM tokens"
        )

    @property
    def consolidation_count(self) -> int:
        """Number of consolidation cycles triggered so far."""
        return self._consolidation_count

    @property
    def stm_tokens_estimate(self) -> int:
        """Running estimate of STM tokens accumulated."""
        return self._stm_tokens

    def _detect_recall_intent(self, text: str) -> bool:
        """Check if text contains recall-intent patterns."""
        return any(p.search(text) for p in _RECALL_PATTERNS)
