"""
RAGIX Memory MCP Tools — Thin wrappers around MemoryToolDispatcher.

Each MCP tool parses arguments, delegates to dispatcher.dispatch(),
and formats the response.  Zero business logic in this layer.

Tool hierarchy (from PLAN §3.2):
    PRIMARY:    memory_recall   — token-budgeted injection (canonical contract)
    SECONDARY:  memory_search   — interactive discovery
    WRITE:      memory_propose  — default (governed), memory_write — privileged
    CRUD:       memory_read, memory_update, memory_link
    LIFECYCLE:  memory_consolidate, memory_stats, memory_palace_list, memory_palace_get
    MANAGEMENT: memory_workspace_list, memory_workspace_register,
                memory_workspace_remove, memory_metrics

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from ragix_core.memory.mcp.formatting import (
    FORMAT_VERSION,
    format_injection_block,
    format_search_results,
)
from ragix_core.memory.mcp.session import SessionManager
from ragix_core.memory.tools import MemoryToolDispatcher

logger = logging.getLogger(__name__)

# Default session ID for non-session-aware tools.
_DEFAULT_SESSION = "mcp"


def register_memory_tools(
    mcp,
    dispatcher: MemoryToolDispatcher,
    session_mgr: Optional[SessionManager] = None,
    workspace_router=None,
    metrics=None,
    rate_limiter=None,
) -> None:
    """
    Register all memory MCP tools on a FastMCP server instance.

    Phase 1: 11 core tools (recall, search, propose, write, read, update,
             link, consolidate, stats, palace_list, palace_get).
    Phase 2: +2 session bridge tools (session_inject, session_store).
    Phase 5: +4 management tools (workspace_list, workspace_register,
             workspace_remove, metrics).

    Args:
        mcp: FastMCP server instance.
        dispatcher: Fully wired MemoryToolDispatcher.
        session_mgr: Optional session manager for session bridge tools.
        workspace_router: Optional WorkspaceRouter for named workspace support.
        metrics: Optional MetricsCollector for per-tool call tracking.
        rate_limiter: Optional RateLimiter for per-session rate limiting.
    """

    # -- Guard helpers (closures capturing rate_limiter / metrics) ----------

    def _rate_guard(tool_name: str, session_id: str = _DEFAULT_SESSION) -> Optional[Dict[str, Any]]:
        """Check rate limit.  Returns error dict if blocked, None if OK."""
        if rate_limiter is None:
            return None
        rl = rate_limiter.check_rate(session_id)
        if not rl.allowed:
            # Record the blocked call as an error in metrics
            if metrics is not None:
                metrics.record_call(tool_name, latency_ms=0.0, error=True)
            return {
                "status": "error",
                "message": f"Rate limit exceeded for {tool_name}",
                "reason": rl.reason,
                "retry_after_ms": rl.retry_after_ms,
            }
        rate_limiter.consume(session_id)
        return None

    @contextmanager
    def _timed(tool_name: str) -> Iterator[None]:
        """Context manager: record call latency in metrics collector."""
        if metrics is not None:
            with metrics.timed_call(tool_name):
                yield
        else:
            yield

    def _resolve_workspace(
        workspace: Optional[str],
        fallback_scope: Optional[str],
    ) -> tuple:
        """Resolve workspace name to (scope, corpus_id).

        Returns (scope, corpus_id, error_dict_or_None).
        If workspace is None or workspace_router is absent, falls back to
        (fallback_scope, None, None).
        """
        if not workspace or workspace_router is None:
            return (fallback_scope, None, None)
        try:
            scope, corpus_id = workspace_router.resolve(workspace)
            return (scope, corpus_id, None)
        except KeyError as e:
            return (None, None, {
                "status": "error",
                "message": f"Unknown workspace: {workspace}",
            })

    # -- PRIMARY: Token-budgeted injection ---------------------------------

    @mcp.tool()
    def memory_recall(
        query: str,
        budget_tokens: int = 1500,
        mode: str = "hybrid",
        tier: Optional[str] = None,
        scope: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Token-budgeted memory retrieval for context injection.

        PRIMARY tool — returns items formatted for direct insertion into
        LLM context, respecting a token budget. Use this as the default
        way to retrieve prior knowledge.

        Args:
            query: Natural language search query.
            budget_tokens: Maximum tokens for injection block (default 1500).
            mode: Retrieval mode — inject|catalog|hybrid (default hybrid).
            tier: Filter by tier (stm|mtm|ltm). None = all.
            scope: Filter by scope. None = all.
            workspace: Named workspace (resolves to scope + corpus_id).

        Returns:
            inject_text: Formatted injection block (format_version=1).
            items: Structured item list for catalog mode.
            tokens_used: Actual tokens consumed.
            matched: Total items matching query.
        """
        blocked = _rate_guard("memory_recall")
        if blocked:
            return blocked

        eff_scope, eff_corpus, ws_err = _resolve_workspace(workspace, scope)
        if ws_err:
            return ws_err

        with _timed("memory_recall"):
            # Delegate search to dispatcher
            search_params: Dict[str, Any] = {
                "query": query,
                "tier": tier,
                "scope": eff_scope,
                "k": 50,  # fetch more, then budget-truncate
            }
            if eff_corpus:
                search_params["corpus_id"] = eff_corpus

            search_result = dispatcher.dispatch("search", search_params)

            if search_result.get("status") != "ok":
                return search_result

            raw_items = search_result.get("items", [])

            # Enrich with full content for injection formatting
            item_ids = [it.get("id", "") for it in raw_items if it.get("id")]
            if item_ids:
                read_result = dispatcher.dispatch("read", {"ids": item_ids})
                full_items = {
                    it["id"]: it
                    for it in read_result.get("items", [])
                }
            else:
                full_items = {}

            # Merge catalog entries with full content, filtering non-injectable
            enriched = []
            for cat_item in raw_items:
                iid = cat_item.get("id", "")
                full = full_items.get(iid, {})
                merged = {**cat_item, **full}
                # V3.3: exclude non-injectable items from recall
                if not merged.get("injectable", True):
                    continue
                enriched.append(merged)

            # Format injection block
            inject_text = ""
            if mode in ("inject", "hybrid"):
                inject_text = format_injection_block(
                    enriched,
                    budget_tokens=budget_tokens,
                    total_matched=len(enriched),
                    injection_type="memory_recall",
                )

            catalog = []
            if mode in ("catalog", "hybrid"):
                catalog = format_search_results(enriched, query=query)

            tokens_used = len(inject_text) // 4 if inject_text else 0

            return {
                "status": "ok",
                "inject_text": inject_text,
                "items": catalog,
                "tokens_used": tokens_used,
                "matched": len(enriched),
                "format_version": FORMAT_VERSION,
            }

    # -- SECONDARY: Interactive search -------------------------------------

    @mcp.tool()
    def memory_search(
        query: str,
        tags: Optional[str] = None,
        tier: Optional[str] = None,
        type_filter: Optional[str] = None,
        domain: Optional[str] = None,
        scope: Optional[str] = None,
        workspace: Optional[str] = None,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Search memory items by text query, tags, and filters.

        SECONDARY tool — for interactive discovery and exploration.
        Returns structured results (not formatted for injection).

        Args:
            query: Search text (FTS5 BM25 ranked).
            tags: Comma-separated tags to filter by.
            tier: Filter by tier (stm|mtm|ltm).
            type_filter: Filter by type (fact|decision|definition|...).
            domain: Filter by document domain (post-filter).
            scope: Filter by scope.
            workspace: Named workspace (resolves to scope + corpus_id).
            k: Max results (default 10).

        Returns:
            count: Number of results.
            items: List of matching items with preview.
        """
        blocked = _rate_guard("memory_search")
        if blocked:
            return blocked

        eff_scope, eff_corpus, ws_err = _resolve_workspace(workspace, scope)
        if ws_err:
            return ws_err

        with _timed("memory_search"):
            params: Dict[str, Any] = {"query": query, "k": k}
            if tags:
                params["tags"] = [t.strip() for t in tags.split(",")]
            if tier:
                params["tier"] = tier
            if type_filter:
                params["type"] = type_filter
            if eff_scope:
                params["scope"] = eff_scope
            if eff_corpus:
                params["corpus_id"] = eff_corpus

            result = dispatcher.dispatch("search", params)

            # Post-filter by domain if requested (domain is not a store field)
            if domain and result.get("status") == "ok":
                filtered = [
                    it for it in result.get("items", [])
                    if _item_matches_domain(it, domain)
                ]
                result["items"] = filtered
                result["count"] = len(filtered)

            # V3.3: Flag non-injectable items as [QUARANTINED] in search results
            if result.get("status") == "ok":
                for it in result.get("items", []):
                    if not it.get("injectable", True):
                        it["quarantined"] = True

            return result

    # -- WRITE PATH --------------------------------------------------------

    @mcp.tool()
    def memory_propose(
        items: str,
        scope: str = "project",
        workspace: Optional[str] = None,
        source_doc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit memory candidates for policy evaluation and storage.

        DEFAULT write path — items are validated against governance policy
        (secret detection, injection patterns, size limits). Approved items
        are stored in STM.

        Args:
            items: JSON array of proposals. Each: {title, content, tags[], type}.
                   Optional: entities[], confidence, provenance_hint{}.
            scope: Memory scope (default "project").
            workspace: Named workspace (resolves to scope + corpus_id).
            source_doc: Source document for provenance tracking.

        Returns:
            accepted: Count of stored items.
            rejected: Count of blocked items.
            items: Per-item results with status.
        """
        blocked = _rate_guard("memory_propose")
        if blocked:
            return blocked

        eff_scope, eff_corpus, ws_err = _resolve_workspace(workspace, scope)
        if ws_err:
            return ws_err
        if eff_scope:
            scope = eff_scope

        with _timed("memory_propose"):
            try:
                item_list = json.loads(items)
            except (json.JSONDecodeError, TypeError) as e:
                return {"status": "error", "message": f"Invalid JSON in items: {e}"}

            if not isinstance(item_list, list):
                item_list = [item_list]

            # Check proposal rate limit (per-turn cap)
            if rate_limiter is not None:
                turn = 0
                if session_mgr is not None:
                    turn = session_mgr.get_turn_count(_DEFAULT_SESSION)
                pl = rate_limiter.check_proposal_limit(
                    _DEFAULT_SESSION, turn=turn, count=len(item_list),
                )
                if not pl.allowed:
                    return {
                        "status": "error",
                        "message": "Proposal limit exceeded for this turn",
                        "reason": pl.reason,
                        "remaining": pl.remaining,
                    }

            # Inject provenance from source_doc if provided
            if source_doc:
                for it in item_list:
                    if "provenance_hint" not in it:
                        it["provenance_hint"] = {}
                    it["provenance_hint"]["source_id"] = source_doc
                    it["provenance_hint"]["source_kind"] = "doc"
                    it.setdefault("scope", scope)

            result = dispatcher.dispatch("propose", {"items": item_list})

            # V3.3: Auto-consolidation trigger
            if result.get("accepted", 0) > 0:
                consol = _maybe_auto_consolidate(dispatcher, scope=scope)
                if consol is not None:
                    result["consolidation_triggered"] = True
                    result["consolidation_merged"] = consol.get("items_merged", 0)

            return result

    @mcp.tool()
    def memory_write(
        title: str,
        content: str,
        tags: Optional[str] = None,
        tier: str = "stm",
        type: str = "note",
        scope: str = "project",
        workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Direct write to memory store (privileged, policy-checked).

        DEV/ADMIN only — bypasses proposal workflow but still runs policy
        checks (secret + injection detection). Use memory_propose for
        normal write operations.

        Args:
            title: Item title.
            content: Item content (max 2000 chars).
            tags: Comma-separated tags.
            tier: Memory tier — stm|mtm|ltm (default stm).
            type: Item type — fact|decision|definition|constraint|pattern|todo|pointer|note.
            scope: Memory scope (default "project").
            workspace: Named workspace (resolves to scope + corpus_id).

        Returns:
            id: Stored item ID (or rejection reason).
        """
        blocked = _rate_guard("memory_write")
        if blocked:
            return blocked

        eff_scope, _, ws_err = _resolve_workspace(workspace, scope)
        if ws_err:
            return ws_err
        if eff_scope:
            scope = eff_scope

        with _timed("memory_write"):
            params: Dict[str, Any] = {
                "title": title,
                "content": content,
                "tier": tier,
                "type": type,
                "scope": scope,
            }
            if tags:
                params["tags"] = [t.strip() for t in tags.split(",")]

            return dispatcher.dispatch("write", params)

    # -- CRUD --------------------------------------------------------------

    @mcp.tool()
    def memory_read(
        ids: str,
    ) -> Dict[str, Any]:
        """Read memory items by their IDs.

        Args:
            ids: Comma-separated item IDs (e.g. "MEM-abc123,MEM-def456").

        Returns:
            items: Full item data for each found ID.
        """
        blocked = _rate_guard("memory_read")
        if blocked:
            return blocked

        with _timed("memory_read"):
            id_list = [i.strip() for i in ids.split(",") if i.strip()]
            return dispatcher.dispatch("read", {"ids": id_list})

    @mcp.tool()
    def memory_update(
        id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[str] = None,
        tier: Optional[str] = None,
        confidence: Optional[float] = None,
        validation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update fields on an existing memory item.

        Args:
            id: Item ID to update.
            title: New title (optional).
            content: New content (optional).
            tags: New comma-separated tags (optional, replaces existing).
            tier: New tier — stm|mtm|ltm (optional).
            confidence: New confidence score 0.0-1.0 (optional).
            validation: New validation state (optional).

        Returns:
            id: Updated item ID.
        """
        blocked = _rate_guard("memory_update")
        if blocked:
            return blocked

        with _timed("memory_update"):
            patch: Dict[str, Any] = {"id": id}
            if title is not None:
                patch["title"] = title
            if content is not None:
                patch["content"] = content
            if tags is not None:
                patch["tags"] = [t.strip() for t in tags.split(",")]
            if tier is not None:
                patch["tier"] = tier
            if confidence is not None:
                patch["confidence"] = confidence
            if validation is not None:
                patch["validation"] = validation
            return dispatcher.dispatch("update", patch)

    @mcp.tool()
    def memory_link(
        src_id: str,
        dst_id: str,
        relation: str,
    ) -> Dict[str, Any]:
        """Create a typed relationship between two memory items.

        Args:
            src_id: Source item ID.
            dst_id: Destination item ID.
            relation: Relationship type (supports|contradicts|refines|supersedes).

        Returns:
            src_id, dst_id: Linked item IDs.
        """
        blocked = _rate_guard("memory_link")
        if blocked:
            return blocked

        with _timed("memory_link"):
            return dispatcher.dispatch("link", {
                "src_id": src_id,
                "dst_id": dst_id,
                "rel": relation,
            })

    # -- LIFECYCLE ---------------------------------------------------------

    @mcp.tool()
    def memory_consolidate(
        scope: str = "project",
        workspace: Optional[str] = None,
        tiers: Optional[str] = None,
        promote: bool = True,
    ) -> Dict[str, Any]:
        """Trigger memory consolidation: deduplication, merge, and tier promotion.

        Args:
            scope: Scope to consolidate (default "project").
            workspace: Named workspace (resolves to scope).
            tiers: Comma-separated tiers to process (default "stm").
            promote: Whether to promote items to higher tiers (default True).

        Returns:
            Consolidation results (clusters merged, items promoted, etc.).
        """
        blocked = _rate_guard("memory_consolidate")
        if blocked:
            return blocked

        eff_scope, _, ws_err = _resolve_workspace(workspace, scope)
        if ws_err:
            return ws_err
        if eff_scope:
            scope = eff_scope

        with _timed("memory_consolidate"):
            tier_list = (
                [t.strip() for t in tiers.split(",")]
                if tiers else ["stm"]
            )
            return dispatcher.dispatch("consolidate", {
                "scope": scope,
                "tiers": tier_list,
                "promote": promote,
            })

    @mcp.tool()
    def memory_stats(
        scope: Optional[str] = None,
        workspace: Optional[str] = None,
        corpus_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Memory store statistics: item counts, tier distribution, search status.

        Args:
            scope: Filter stats by scope (optional).
            workspace: Named workspace (resolves to scope + corpus_id).
            corpus_id: Filter stats by corpus (optional).

        Returns:
            total_items, by_tier, by_type, events_count, fts5_available, etc.
        """
        blocked = _rate_guard("memory_stats")
        if blocked:
            return blocked

        eff_scope, eff_corpus, ws_err = _resolve_workspace(workspace, scope)
        if ws_err:
            return ws_err
        # Use resolved corpus if not explicitly provided
        if eff_corpus and not corpus_id:
            corpus_id = eff_corpus

        with _timed("memory_stats"):
            stats = dispatcher.store.stats()
            stats["status"] = "ok"
            stats["format_version"] = FORMAT_VERSION
            return stats

    @mcp.tool()
    def memory_palace_list(
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Browse memory palace hierarchy (domain/room/shelf/card).

        Args:
            path: Palace path to browse (e.g. "oracle/security").
                  Empty or None returns top-level domains.

        Returns:
            locations: List of palace locations.
        """
        blocked = _rate_guard("memory_palace_list")
        if blocked:
            return blocked

        with _timed("memory_palace_list"):
            return dispatcher.dispatch("palace.list", {"path": path or ""})

    @mcp.tool()
    def memory_palace_get(
        item_id: str,
    ) -> Dict[str, Any]:
        """Get a memory item with its palace location.

        Args:
            item_id: Item ID to retrieve.

        Returns:
            item: Full item data.
            location: Palace location (domain/room/shelf/card).
        """
        blocked = _rate_guard("memory_palace_get")
        if blocked:
            return blocked

        with _timed("memory_palace_get"):
            return dispatcher.dispatch("palace.get", {"item_id": item_id})

    # -- SESSION BRIDGE (Phase 2) ------------------------------------------

    if session_mgr is not None:
        @mcp.tool()
        def memory_session_inject(
            query: str,
            session_id: str,
            system_context: str = "",
            budget_tokens: int = 1500,
        ) -> Dict[str, Any]:
            """Claude Session Bridge — pre-call injection.

            Call at the START of each turn to augment system context with
            relevant prior knowledge from memory.  Returns augmented context
            with injected memory block (format_version=1).

            Args:
                query: The user's query for this turn.
                session_id: Session identifier (project_id:conversation_id).
                system_context: Existing system context to augment.
                budget_tokens: Maximum tokens for injection block (default 1500).

            Returns:
                augmented_context: system_context + injected memory block.
                items_injected: Number of items included.
                tokens_used: Tokens consumed by injection.
                session_turn: Monotonic turn counter for this session.
            """
            blocked = _rate_guard("memory_session_inject", session_id)
            if blocked:
                return blocked

            with _timed("memory_session_inject"):
                # Get or create session, increment turn
                session = session_mgr.get_or_create(session_id)
                turn = session_mgr.increment_turn(session_id)

                # Search for relevant items
                search_result = dispatcher.dispatch("search", {
                    "query": query,
                    "scope": session.scope,
                    "k": 50,
                })

                if search_result.get("status") != "ok":
                    return {
                        "status": search_result.get("status", "error"),
                        "message": search_result.get("message", "Search failed"),
                        "augmented_context": system_context,
                        "items_injected": 0,
                        "tokens_used": 0,
                        "session_turn": turn,
                    }

                raw_items = search_result.get("items", [])

                # Enrich with full content
                item_ids = [it.get("id", "") for it in raw_items if it.get("id")]
                if item_ids:
                    read_result = dispatcher.dispatch("read", {"ids": item_ids})
                    full_items = {
                        it["id"]: it for it in read_result.get("items", [])
                    }
                else:
                    full_items = {}

                # Merge and filter non-injectable
                enriched = []
                for cat_item in raw_items:
                    iid = cat_item.get("id", "")
                    full = full_items.get(iid, {})
                    merged = {**cat_item, **full}
                    if not merged.get("injectable", True):
                        continue
                    enriched.append(merged)

                # Format injection block
                inject_text = format_injection_block(
                    enriched,
                    budget_tokens=budget_tokens,
                    total_matched=len(enriched),
                    injection_type="session_inject",
                )

                tokens_used = len(inject_text) // 4 if inject_text else 0

                # Augment system context
                if inject_text:
                    augmented = f"{system_context}\n\n{inject_text}" if system_context else inject_text
                else:
                    augmented = system_context

                return {
                    "status": "ok",
                    "augmented_context": augmented,
                    "items_injected": len(enriched),
                    "tokens_used": tokens_used,
                    "session_turn": turn,
                    "format_version": FORMAT_VERSION,
                }

        @mcp.tool()
        def memory_session_store(
            response_text: str,
            session_id: str,
            source_doc: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Claude Session Bridge — post-call storage.

            Call at the END of each turn to extract and persist any
            knowledge the LLM produced.  Parses structured proposals
            from response text and applies policy governance.

            Args:
                response_text: The LLM's response text for this turn.
                session_id: Session identifier (project_id:conversation_id).
                source_doc: Source document for provenance tracking.

            Returns:
                accepted: Items stored (tier=STM).
                rejected: Items blocked by policy.
                quarantined: Items stored with injectable=False.
                session_turn: Current turn counter.
            """
            blocked = _rate_guard("memory_session_store", session_id)
            if blocked:
                return blocked

            with _timed("memory_session_store"):
                session = session_mgr.get_or_create(session_id)
                turn = session_mgr.get_turn_count(session_id)

                # Parse proposals from response text using Proposer
                try:
                    from ragix_core.memory.proposer import MemoryProposer
                    proposer = MemoryProposer(config=dispatcher._config.proposer)
                    proposals = proposer.parse(response_text)
                except (ImportError, Exception) as e:
                    logger.debug(f"Proposer parse: {e}")
                    proposals = []

                if not proposals:
                    return {
                        "status": "ok",
                        "accepted": 0,
                        "rejected": 0,
                        "quarantined": 0,
                        "session_turn": turn,
                    }

                # Convert proposals to dispatcher format and submit
                items_data = []
                for p in proposals:
                    item_d = p if isinstance(p, dict) else p.to_dict()
                    if source_doc:
                        if "provenance_hint" not in item_d:
                            item_d["provenance_hint"] = {}
                        item_d["provenance_hint"]["source_id"] = source_doc
                        item_d["provenance_hint"]["source_kind"] = "doc"
                    item_d.setdefault("scope", session.scope)
                    items_data.append(item_d)

                result = dispatcher.dispatch("propose", {"items": items_data})

                # Count quarantined (accepted but non-injectable)
                quarantined = 0
                for it_result in result.get("items", []):
                    if it_result.get("action") == "quarantine":
                        quarantined += 1

                # V3.3: Auto-consolidation trigger
                consolidation_triggered = False
                if result.get("accepted", 0) > 0:
                    consol = _maybe_auto_consolidate(dispatcher, scope=session.scope)
                    if consol is not None:
                        consolidation_triggered = True

                return {
                    "status": "ok",
                    "accepted": result.get("accepted", 0),
                    "rejected": result.get("rejected", 0),
                    "quarantined": quarantined,
                    "consolidation_triggered": consolidation_triggered,
                    "session_turn": turn,
                }

    # -- MANAGEMENT (Phase 5) ----------------------------------------------

    if workspace_router is not None:
        @mcp.tool()
        def memory_workspace_list() -> Dict[str, Any]:
            """List all registered memory workspaces.

            Returns:
                workspaces: List of {name, scope, corpus_id, description, created_at}.
            """
            blocked = _rate_guard("memory_workspace_list")
            if blocked:
                return blocked

            with _timed("memory_workspace_list"):
                return {
                    "status": "ok",
                    "workspaces": workspace_router.list_workspaces(),
                }

        @mcp.tool()
        def memory_workspace_register(
            name: str,
            scope: Optional[str] = None,
            corpus_id: Optional[str] = None,
            description: str = "",
        ) -> Dict[str, Any]:
            """Register or update a named memory workspace.

            A workspace maps a human-friendly name to a (scope, corpus_id) pair
            used by memory recall and search.

            Args:
                name: Unique workspace name.
                scope: Memory scope (defaults to name if omitted).
                corpus_id: Corpus identifier for cross-corpus operations.
                description: Free-text description.

            Returns:
                workspace: The registered workspace record.
            """
            blocked = _rate_guard("memory_workspace_register")
            if blocked:
                return blocked

            with _timed("memory_workspace_register"):
                try:
                    info = workspace_router.register(
                        name=name,
                        scope=scope,
                        corpus_id=corpus_id,
                        description=description,
                    )
                    return {
                        "status": "ok",
                        "workspace": {
                            "name": info.name,
                            "scope": info.scope,
                            "corpus_id": info.corpus_id,
                            "description": info.description,
                            "created_at": info.created_at,
                        },
                    }
                except ValueError as e:
                    return {"status": "error", "message": str(e)}

        @mcp.tool()
        def memory_workspace_remove(
            name: str,
        ) -> Dict[str, Any]:
            """Remove a named memory workspace.

            The "default" workspace cannot be removed.

            Args:
                name: Workspace name to remove.

            Returns:
                removed: True if workspace was removed.
            """
            blocked = _rate_guard("memory_workspace_remove")
            if blocked:
                return blocked

            with _timed("memory_workspace_remove"):
                removed = workspace_router.remove(name)
                return {
                    "status": "ok",
                    "removed": removed,
                    "name": name,
                }

    if metrics is not None:
        @mcp.tool()
        def memory_metrics(
            tool_name: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Memory MCP server metrics: call counts, latency, errors.

            Args:
                tool_name: Filter to a single tool (optional).
                           None returns aggregate summary + per-tool breakdown.

            Returns:
                summary or per-tool metrics dict.
            """
            # No rate guard for metrics (read-only, always allowed)
            with _timed("memory_metrics"):
                if tool_name:
                    return {
                        "status": "ok",
                        "metrics": metrics.get_metrics(tool_name),
                    }
                summary = metrics.get_summary()
                # Append rate limiter status if available
                if rate_limiter is not None:
                    summary["rate_limiter"] = rate_limiter.get_status(_DEFAULT_SESSION)
                summary["status"] = "ok"
                return summary

    # -- Log registered tool count -----------------------------------------

    tool_count = 11  # core tools
    if session_mgr is not None:
        tool_count += 2
    if workspace_router is not None:
        tool_count += 3
    if metrics is not None:
        tool_count += 1
    logger.info("Registered %d memory MCP tools", tool_count)


# -- Helpers ---------------------------------------------------------------

def _maybe_auto_consolidate(
    dispatcher: MemoryToolDispatcher,
    scope: str = "project",
) -> Optional[Dict[str, Any]]:
    """
    Check if STM count exceeds threshold and trigger consolidation.

    Returns consolidation result dict if triggered, None otherwise.
    Deterministic trigger: STM count >= config.consolidate.stm_threshold.
    """
    cfg = dispatcher._config.consolidate
    if not cfg.enabled:
        return None

    stm_count = dispatcher.store.count_items(tier="stm", scope=scope)
    if stm_count < cfg.stm_threshold:
        return None

    logger.info(
        f"Auto-consolidation triggered: STM count {stm_count} >= threshold {cfg.stm_threshold}"
    )
    return dispatcher.dispatch("consolidate", {
        "scope": scope,
        "tiers": ["stm"],
        "promote": True,
    })


def _item_matches_domain(item: Dict[str, Any], domain: str) -> bool:
    """Check if an item matches a domain filter (tag-based heuristic)."""
    domain_lower = domain.lower()
    tags = item.get("tags", [])
    title = item.get("title", "").lower()
    if any(domain_lower in t.lower() for t in tags):
        return True
    if domain_lower in title:
        return True
    return False
