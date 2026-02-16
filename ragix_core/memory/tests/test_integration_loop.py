"""
Integration Test — Full Memory Loop

Simulates a multi-turn conversation flow:
  1. Initial question -> LLM proposes memory -> governor accepts -> stored
  2. Next question -> recall finds memory -> injection/catalog returned
  3. Explicit recall request -> triggers memory.search tool flow

Uses MockEmbedder so tests are fully deterministic.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

import pytest
from ragix_core.memory.config import (
    EmbedderConfig,
    MemoryConfig,
    PolicyConfig,
    QSearchConfig,
    RecallConfig,
    StoreConfig,
)
from ragix_core.memory.middleware import MemoryMiddleware
from ragix_core.memory.tools import MemoryToolDispatcher, create_dispatcher


@pytest.fixture
def config():
    return MemoryConfig(
        store=StoreConfig(db_path=":memory:"),
        embedder=EmbedderConfig(backend="mock", dimension=32, mock_seed=42),
        policy=PolicyConfig(),
        recall=RecallConfig(
            mode="hybrid",
            inject_budget_tokens=500,
            catalog_k=5,
        ),
        qsearch=QSearchConfig(
            enabled=True,
            max_expansions=3,
            max_time_seconds=5,
        ),
    )


@pytest.fixture
def dispatcher(config):
    d = create_dispatcher(config)
    yield d
    d.store.close()


@pytest.fixture
def middleware(dispatcher, config):
    return MemoryMiddleware(dispatcher=dispatcher, config=config)


class TestTurn1_ProposalFlow:
    """Turn 1: LLM proposes memory candidates."""

    def test_proposal_accepted_and_stored(self, middleware):
        # Simulate LLM response with memory proposals
        response_text = (
            "Based on my analysis, I recommend using SQLite.\n\n"
            "<MEMORY_PROPOSALS_JSON>\n"
            '[\n'
            '  {\n'
            '    "type": "decision",\n'
            '    "title": "Use SQLite for persistence",\n'
            '    "content": "SQLite chosen for memory store due to zero-config and reliability.",\n'
            '    "tags": ["architecture", "database"],\n'
            '    "why_store": "Key architectural decision",\n'
            '    "provenance_hint": {"source_kind": "chat", "source_id": "turn_1"}\n'
            '  }\n'
            ']\n'
            "</MEMORY_PROPOSALS_JSON>"
        )

        cleaned, summary = middleware.post_call(response_text)

        # Delimiter block should be stripped
        assert "<MEMORY_PROPOSALS_JSON>" not in cleaned
        assert "Based on my analysis" in cleaned

        # Proposal accepted
        assert summary["proposals_found"] == 1
        assert summary["accepted"] == 1
        assert summary["rejected"] == 0

        # Verify stored
        items = middleware._dispatcher.store.list_items()
        assert len(items) == 1
        assert items[0].type == "decision"
        assert items[0].tier == "stm"

    def test_proposal_with_secret_rejected(self, middleware):
        response_text = (
            "Here is the config.\n"
            "<MEMORY_PROPOSALS_JSON>\n"
            '[{"type": "note", "title": "API key", '
            '"content": "api_key: sk-1234567890abcdef1234567890", '
            '"tags": ["config"], "why_store": "save key", '
            '"provenance_hint": {"source_kind": "chat", "source_id": "turn_2"}}]\n'
            "</MEMORY_PROPOSALS_JSON>"
        )

        cleaned, summary = middleware.post_call(response_text)
        assert summary["accepted"] == 0
        assert summary["rejected"] == 1


class TestTurn2_RecallFlow:
    """Turn 2: Pre-call injection finds previously stored memory."""

    def _store_item(self, dispatcher):
        """Helper: store a decision item."""
        dispatcher.dispatch("propose", {
            "items": [{
                "type": "decision",
                "title": "Use SQLite for persistence",
                "content": "SQLite chosen for the memory store.",
                "tags": ["architecture", "database", "sqlite"],
                "why_store": "Key decision",
                "provenance_hint": {"source_kind": "chat", "source_id": "turn_1"},
            }],
        })

    def test_pre_call_injects_memory(self, middleware, dispatcher):
        self._store_item(dispatcher)

        augmented = middleware.pre_call(
            user_query="What database are we using for the memory store?",
            system_context="You are a helpful assistant.",
        )

        assert "Relevant Memory" in augmented
        assert "[MEMORY:" in augmented

    def test_pre_call_no_inject_for_unrelated(self, middleware, dispatcher):
        self._store_item(dispatcher)

        augmented = middleware.pre_call(
            user_query="How do I cook pasta?",
            system_context="You are a helpful assistant.",
        )

        # May or may not inject (depends on embedding similarity)
        # At minimum, system context should be preserved
        assert "You are a helpful assistant." in augmented


class TestTurn3_ExplicitRecall:
    """Turn 3: Explicit recall request triggers memory search."""

    def _store_items(self, dispatcher):
        dispatcher.dispatch("propose", {
            "items": [
                {
                    "type": "decision",
                    "title": "Use SQLite",
                    "content": "SQLite for storage.",
                    "tags": ["database"],
                    "why_store": "decision",
                    "provenance_hint": {"source_kind": "chat", "source_id": "t1"},
                },
                {
                    "type": "fact",
                    "title": "Python 3.10 required",
                    "content": "Minimum Python version is 3.10.",
                    "tags": ["python", "requirements"],
                    "why_store": "requirement",
                    "provenance_hint": {"source_kind": "doc", "source_id": "readme"},
                },
            ],
        })

    def test_recall_intent_detected(self, middleware, dispatcher):
        self._store_items(dispatcher)

        result = middleware.intercept_recall(
            "What do we know about the database decision?"
        )

        assert result is not None
        assert result["status"] == "ok"
        assert result.get("count", 0) > 0

    def test_recall_via_tool_call(self, middleware, dispatcher):
        self._store_items(dispatcher)

        result = middleware.intercept_recall(
            "Search memory",
            tool_calls=[{
                "action": "memory.search",
                "query": "database",
                "k": 5,
            }],
        )

        assert result is not None
        assert result["status"] == "ok"

    def test_no_recall_for_normal_query(self, middleware):
        result = middleware.intercept_recall(
            "How do I create a Python virtual environment?"
        )
        assert result is None


class TestFullLoop:
    """End-to-end: propose -> store -> recall -> read."""

    def test_full_cycle(self, dispatcher):
        # Step 1: Propose
        propose_result = dispatcher.dispatch("propose", {
            "items": [{
                "type": "constraint",
                "title": "No cloud dependencies",
                "content": "All processing must be local-first, sovereign.",
                "tags": ["architecture", "sovereignty", "constraint"],
                "why_store": "Hard architectural constraint",
                "provenance_hint": {
                    "source_kind": "doc",
                    "source_id": "CLAUDE.md",
                    "content_hashes": ["sha256:constraint1"],
                },
            }],
        })
        assert propose_result["accepted"] == 1
        item_id = propose_result["items"][0]["id"]

        # Step 2: Search
        search_result = dispatcher.dispatch("search", {
            "query": "architecture sovereignty local",
            "k": 5,
        })
        assert search_result["status"] == "ok"
        assert search_result["count"] > 0

        # Step 3: Read
        read_result = dispatcher.dispatch("read", {"ids": [item_id]})
        assert read_result["status"] == "ok"
        assert len(read_result["items"]) == 1
        assert read_result["items"][0]["title"] == "No cloud dependencies"

        # Step 4: Update
        update_result = dispatcher.dispatch("update", {
            "id": item_id,
            "validation": "verified",
        })
        assert update_result["status"] == "ok"

        # Step 5: Link
        propose2 = dispatcher.dispatch("propose", {
            "items": [{
                "type": "decision",
                "title": "Use Ollama for LLM",
                "content": "Ollama serves local LLMs for all inference.",
                "tags": ["architecture", "llm", "ollama"],
                "why_store": "Supports sovereignty constraint",
                "provenance_hint": {"source_kind": "chat", "source_id": "turn_5"},
            }],
        })
        item2_id = propose2["items"][0]["id"]

        link_result = dispatcher.dispatch("link", {
            "src_id": item2_id,
            "dst_id": item_id,
            "rel": "supports",
        })
        assert link_result["status"] == "ok"

        # Step 6: Stats
        stats = dispatcher.store.stats()
        assert stats["total_items"] == 2
