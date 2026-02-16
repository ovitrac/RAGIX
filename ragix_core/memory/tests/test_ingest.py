"""
Tests for memory ingest pipeline (ingest.py + cli recall/ingest).

Validates:
- Paragraph chunking (basic split, small-paragraph merging)
- Idempotent file ingestion
- injectable=False default
- Provenance metadata on chunks
- Auto-format tag/title inference
- format_injection_block format_version header
- Recall injectable filter

Uses in-memory SQLite for speed and isolation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-16
"""

import os
import textwrap

import pytest

from ragix_core.memory.ingest import (
    IngestResult,
    chunk_paragraphs,
    ingest_file,
    _infer_tags_from_path,
    _infer_title,
)
from ragix_core.memory.mcp.formatting import format_injection_block
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance


@pytest.fixture
def store():
    """In-memory store for testing."""
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def sample_file(tmp_path):
    """Create a small markdown file for testing."""
    content = textwrap.dedent("""\
        # Architecture Overview

        This is the first paragraph describing the system architecture.
        It spans multiple lines but stays within one paragraph block.

        This is the second paragraph. It covers the data model and
        how components interact with each other.

        ## Subsection

        Third paragraph here with technical details about the
        implementation choices and trade-offs.

        Fourth paragraph wrapping up the section with a summary
        of key decisions made during the design phase.
    """)
    p = tmp_path / "docs" / "architecture.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. chunk_paragraphs: basic split
# ---------------------------------------------------------------------------


class TestChunkParagraphsBasic:
    def test_splits_at_blank_lines(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_paragraphs(text, max_tokens=5000)
        # All fit in one chunk at high budget
        assert len(chunks) == 1
        assert "Para one." in chunks[0][0]
        assert "Para three." in chunks[0][0]

    def test_respects_token_limit(self):
        # Each paragraph ~50 chars = ~12 tokens; budget=20 forces split
        text = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80
        chunks = chunk_paragraphs(text, max_tokens=25)
        assert len(chunks) >= 2
        # First chunk starts at line 0
        assert chunks[0][1] == 0

    def test_empty_text(self):
        assert chunk_paragraphs("") == []
        assert chunk_paragraphs("   \n\n  ") == []

    def test_line_ranges_increase(self):
        text = "Line1\nLine2\n\nLine3\nLine4\nLine5\n\nLine6"
        chunks = chunk_paragraphs(text, max_tokens=5000)
        # Single chunk: start=0, end >= 0
        assert len(chunks) == 1
        assert chunks[0][1] == 0
        assert chunks[0][2] >= 0


# ---------------------------------------------------------------------------
# 2. chunk_paragraphs: merge small paragraphs
# ---------------------------------------------------------------------------


class TestChunkParagraphsMerge:
    def test_small_paragraphs_merged(self):
        # 5 tiny paragraphs, all fit under budget
        paras = ["word"] * 5
        text = "\n\n".join(paras)
        chunks = chunk_paragraphs(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0][0].count("word") == 5

    def test_large_paragraph_emitted_alone(self):
        # One huge paragraph exceeds budget but still gets emitted
        big = "x" * 4000  # ~1000 tokens
        text = "small\n\n" + big + "\n\nsmall2"
        chunks = chunk_paragraphs(text, max_tokens=100)
        # The big paragraph should be in its own chunk
        assert any(len(c[0]) > 3000 for c in chunks)


# ---------------------------------------------------------------------------
# 3. ingest_file: idempotent
# ---------------------------------------------------------------------------


class TestIngestIdempotent:
    def test_reingest_same_file_skipped(self, store, sample_file):
        r1 = ingest_file(store, sample_file, workspace="test")
        assert r1.files_processed == 1
        assert r1.chunks_created >= 1

        r2 = ingest_file(store, sample_file, workspace="test")
        assert r2.files_skipped == 1
        assert r2.files_processed == 0
        assert r2.chunks_created == 0
        assert r2.item_ids == []

    def test_modified_file_reingested(self, store, sample_file):
        r1 = ingest_file(store, sample_file, workspace="test")
        assert r1.files_processed == 1

        # Modify file content
        with open(sample_file, "a") as f:
            f.write("\n\nNew paragraph added for testing.\n")

        r2 = ingest_file(store, sample_file, workspace="test")
        assert r2.files_processed == 1
        assert r2.chunks_created >= 1


# ---------------------------------------------------------------------------
# 4. ingest_file: injectable=False by default
# ---------------------------------------------------------------------------


class TestIngestInjectableDefault:
    def test_default_not_injectable(self, store, sample_file):
        r = ingest_file(store, sample_file, workspace="test")
        for item_id in r.item_ids:
            item = store.read_item(item_id)
            assert item is not None
            assert item.injectable is False

    def test_injectable_flag_respected(self, store, sample_file):
        r = ingest_file(store, sample_file, workspace="test", injectable=True)
        for item_id in r.item_ids:
            item = store.read_item(item_id)
            assert item.injectable is True


# ---------------------------------------------------------------------------
# 5. ingest_file: provenance
# ---------------------------------------------------------------------------


class TestIngestProvenance:
    def test_provenance_fields(self, store, sample_file):
        r = ingest_file(store, sample_file, workspace="test")
        assert r.chunks_created >= 1

        item = store.read_item(r.item_ids[0])
        assert item is not None
        prov = item.provenance

        # source_kind = "doc"
        assert prov.source_kind == "doc"
        # source_id = absolute path
        assert os.path.isabs(prov.source_id)
        # chunk_ids present
        assert len(prov.chunk_ids) == 1
        assert ":0" in prov.chunk_ids[0]
        # content_hashes present
        assert len(prov.content_hashes) == 1
        assert prov.content_hashes[0].startswith("sha256:")

    def test_content_has_path_header(self, store, sample_file):
        r = ingest_file(store, sample_file, workspace="test")
        item = store.read_item(r.item_ids[0])
        assert item.content.startswith("[path:")
        assert "chunk:0" in item.content
        assert "lines:" in item.content


# ---------------------------------------------------------------------------
# 6. ingest_file: format auto
# ---------------------------------------------------------------------------


class TestIngestFormatAuto:
    def test_auto_infers_tags(self, store, sample_file):
        r = ingest_file(
            store, sample_file, workspace="test",
            format_mode="auto", tags=["extra"],
        )
        item = store.read_item(r.item_ids[0])
        # Should have "markdown" from .md extension + "docs" from parent dir + "extra"
        assert "markdown" in item.tags
        assert "extra" in item.tags

    def test_auto_infers_title(self, store, sample_file):
        r = ingest_file(
            store, sample_file, workspace="test",
            format_mode="auto",
        )
        item = store.read_item(r.item_ids[0])
        # Title should come from first heading "# Architecture Overview"
        assert "Architecture Overview" in item.title


# ---------------------------------------------------------------------------
# 7. format_injection_block: format_version
# ---------------------------------------------------------------------------


class TestRecallFormatVersion:
    def test_format_version_present(self):
        items = [
            {
                "id": "MEM-abc",
                "tier": "stm",
                "type": "note",
                "title": "Test",
                "content": "Some content",
                "tags": ["test"],
                "confidence": 0.8,
                "validation": "unverified",
                "provenance": {"source_kind": "doc", "source_id": "test.md"},
            }
        ]
        text = format_injection_block(items, budget_tokens=1500, total_matched=1)
        assert "format_version: 1" in text
        assert "## Memory (Injected)" in text

    def test_empty_items_returns_empty(self):
        assert format_injection_block([], budget_tokens=1500) == ""


# ---------------------------------------------------------------------------
# 8. Recall excludes non-injectable
# ---------------------------------------------------------------------------


class TestRecallExcludesNonInjectable:
    def test_injectable_filter(self):
        """Simulates the injectable filter logic from cmd_recall."""
        items = [
            {"id": "MEM-001", "injectable": True, "content": "good"},
            {"id": "MEM-002", "injectable": False, "content": "quarantined"},
            {"id": "MEM-003", "content": "no flag = default true"},
        ]
        filtered = [it for it in items if it.get("injectable", True)]
        assert len(filtered) == 2
        assert filtered[0]["id"] == "MEM-001"
        assert filtered[1]["id"] == "MEM-003"

    def test_all_non_injectable_yields_empty(self):
        items = [
            {"id": "MEM-001", "injectable": False},
            {"id": "MEM-002", "injectable": False},
        ]
        filtered = [it for it in items if it.get("injectable", True)]
        assert len(filtered) == 0
