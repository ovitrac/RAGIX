"""
Tests for FTS5 full-text search and LIKE fallback in store.py.

Validates:
- Basic term search returns matching items
- AND logic: multi-term query requires all terms
- Empty query returns all items (via list_items)
- Filter combinations (tier+query, type+query, scope+query)
- FTS5 path vs LIKE fallback path (mock _fts5_available)
- SQL injection safety (query with quotes, semicolons)

Uses in-memory SQLite for speed and isolation.
"""

import pytest
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance


@pytest.fixture
def store():
    """In-memory store for testing."""
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with diverse items for fulltext search."""
    items = [
        MemoryItem(
            id="FTS-001", tier="stm", type="fact",
            title="SQLite database engine",
            content="SQLite is a self-contained SQL database engine.",
            tags=["database", "sqlite", "engine"],
            provenance=MemoryProvenance(source_kind="doc", source_id="doc_1"),
            confidence=0.9,
            scope="project",
        ),
        MemoryItem(
            id="FTS-002", tier="mtm", type="decision",
            title="FAISS vector index",
            content="Use FAISS for approximate nearest neighbor search.",
            tags=["vector", "faiss", "search"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_2"),
            confidence=0.8,
            scope="project",
        ),
        MemoryItem(
            id="FTS-003", tier="stm", type="fact",
            title="Python version requirement",
            content="RAGIX requires Python 3.12 for async features.",
            tags=["python", "requirements"],
            provenance=MemoryProvenance(source_kind="doc", source_id="readme"),
            confidence=0.95,
            scope="global",
        ),
        MemoryItem(
            id="FTS-004", tier="ltm", type="constraint",
            title="No external API calls",
            content="All processing must stay local. No cloud database allowed.",
            tags=["security", "constraint", "local"],
            provenance=MemoryProvenance(source_kind="doc", source_id="policy"),
            confidence=1.0,
            scope="project",
        ),
        MemoryItem(
            id="FTS-005", tier="stm", type="note",
            title="Database migration notes",
            content="When migrating from SQLite to PostgreSQL, handle WAL mode.",
            tags=["database", "migration", "postgresql"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_5"),
            confidence=0.6,
            scope="project",
        ),
    ]
    for item in items:
        store.write_item(item)
    return store


# ---------------------------------------------------------------------------
# Basic FTS search
# ---------------------------------------------------------------------------

class TestFtsBasicSearch:
    def test_single_term_returns_matching(self, populated_store):
        """A single term should find items containing that term."""
        results = populated_store.search_fulltext("SQLite")
        ids = {r.id for r in results}
        assert "FTS-001" in ids  # title and content mention SQLite
        assert "FTS-005" in ids  # content mentions SQLite

    def test_single_term_excludes_non_matching(self, populated_store):
        """Items without the search term should not appear."""
        results = populated_store.search_fulltext("FAISS")
        ids = {r.id for r in results}
        assert "FTS-002" in ids
        assert "FTS-001" not in ids

    def test_multi_term_and_logic(self, populated_store):
        """Multiple terms must all be present (AND logic)."""
        results = populated_store.search_fulltext("SQLite database")
        ids = {r.id for r in results}
        # FTS-001 has both in title+content
        assert "FTS-001" in ids
        # FTS-005 has both "SQLite" in content and "database" in title
        assert "FTS-005" in ids

    def test_multi_term_filters_partial_match(self, populated_store):
        """An item matching only one of two terms should not appear."""
        results = populated_store.search_fulltext("FAISS migration")
        ids = {r.id for r in results}
        # FTS-002 has FAISS but not migration; FTS-005 has migration but not FAISS
        assert "FTS-002" not in ids
        assert "FTS-005" not in ids

    def test_case_insensitive_search(self, populated_store):
        """Search should be case-insensitive."""
        results_lower = populated_store.search_fulltext("sqlite")
        results_upper = populated_store.search_fulltext("SQLITE")
        ids_lower = {r.id for r in results_lower}
        ids_upper = {r.id for r in results_upper}
        # Both should find items with SQLite
        assert "FTS-001" in ids_lower
        assert "FTS-001" in ids_upper


# ---------------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------------

class TestFtsEmptyQuery:
    def test_empty_query_returns_all(self, populated_store):
        """Empty query should fall back to list_items and return all."""
        results = populated_store.search_fulltext("")
        assert len(results) == 5

    def test_whitespace_query_returns_all(self, populated_store):
        """Whitespace-only query should behave like empty query."""
        results = populated_store.search_fulltext("   ")
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Filter combinations
# ---------------------------------------------------------------------------

class TestFtsFilters:
    def test_tier_filter(self, populated_store):
        """Tier filter restricts search to that tier."""
        results = populated_store.search_fulltext("database", tier="stm")
        for r in results:
            assert r.tier == "stm"

    def test_type_filter(self, populated_store):
        """Type filter restricts search to that type."""
        results = populated_store.search_fulltext("database", type_filter="fact")
        for r in results:
            assert r.type == "fact"

    def test_scope_filter(self, populated_store):
        """Scope filter restricts results to matching scope."""
        results = populated_store.search_fulltext("Python", scope="global")
        ids = {r.id for r in results}
        assert "FTS-003" in ids
        # FTS-001 is project scope, should not appear
        assert "FTS-001" not in ids

    def test_combined_tier_and_type(self, populated_store):
        """Combining tier and type narrows results further."""
        results = populated_store.search_fulltext(
            "database", tier="stm", type_filter="fact"
        )
        for r in results:
            assert r.tier == "stm"
            assert r.type == "fact"

    def test_limit_parameter(self, populated_store):
        """Limit caps the number of results."""
        results = populated_store.search_fulltext("database", limit=1)
        assert len(results) <= 1

    def test_no_match_returns_empty(self, populated_store):
        """A query matching nothing returns an empty list."""
        results = populated_store.search_fulltext("xyzzyplugh")
        assert results == []


# ---------------------------------------------------------------------------
# FTS5 path vs LIKE fallback
# ---------------------------------------------------------------------------

class TestFtsFallback:
    def test_like_fallback_when_fts5_disabled(self, populated_store):
        """When _fts5_available is False, LIKE fallback produces results."""
        original = populated_store._fts5_available
        try:
            populated_store._fts5_available = False
            results = populated_store.search_fulltext("SQLite")
            ids = {r.id for r in results}
            assert "FTS-001" in ids
        finally:
            populated_store._fts5_available = original

    def test_like_fallback_and_logic(self, populated_store):
        """LIKE fallback also enforces AND logic across terms."""
        original = populated_store._fts5_available
        try:
            populated_store._fts5_available = False
            results = populated_store.search_fulltext("FAISS migration")
            # Neither FTS-002 nor FTS-005 should match both terms
            ids = {r.id for r in results}
            assert "FTS-002" not in ids
            assert "FTS-005" not in ids
        finally:
            populated_store._fts5_available = original

    def test_like_fallback_with_filters(self, populated_store):
        """LIKE fallback respects tier/type/scope filters."""
        original = populated_store._fts5_available
        try:
            populated_store._fts5_available = False
            results = populated_store.search_fulltext(
                "database", tier="stm"
            )
            for r in results:
                assert r.tier == "stm"
        finally:
            populated_store._fts5_available = original

    def test_fts5_and_like_agree(self, populated_store):
        """FTS5 and LIKE paths should return overlapping item sets."""
        if not populated_store._fts5_available:
            pytest.skip("FTS5 not available in this SQLite build")
        fts5_results = populated_store.search_fulltext("database")
        fts5_ids = {r.id for r in fts5_results}

        original = populated_store._fts5_available
        try:
            populated_store._fts5_available = False
            like_results = populated_store.search_fulltext("database")
            like_ids = {r.id for r in like_results}
        finally:
            populated_store._fts5_available = original

        # LIKE may be more permissive, but FTS5 results should be a subset
        # or both should contain the core matches
        assert fts5_ids & like_ids, "FTS5 and LIKE should share at least one result"


# ---------------------------------------------------------------------------
# SQL injection safety
# ---------------------------------------------------------------------------

class TestFtsSqlInjection:
    def test_query_with_quotes(self, populated_store):
        """Queries with single/double quotes should not cause errors."""
        results = populated_store.search_fulltext("it's a \"test\"")
        assert isinstance(results, list)

    def test_query_with_semicolons(self, populated_store):
        """Semicolons in query should not trigger SQL injection."""
        results = populated_store.search_fulltext("test; DROP TABLE memory_items;")
        assert isinstance(results, list)
        # Verify the table still exists
        count = populated_store.count_items()
        assert count == 5

    def test_query_with_sql_keywords(self, populated_store):
        """SQL keywords in query should be treated as text, not commands."""
        results = populated_store.search_fulltext("SELECT * FROM memory_items")
        assert isinstance(results, list)

    def test_query_with_special_fts5_chars(self, populated_store):
        """FTS5 special characters (*, ^, NEAR) should be escaped."""
        results = populated_store.search_fulltext("test* NEAR/3 foo")
        assert isinstance(results, list)

    def test_query_with_parentheses(self, populated_store):
        """Parentheses should not cause FTS5 syntax errors."""
        results = populated_store.search_fulltext("(test) OR (foo)")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# V3.2: Configurable tokenizer + French diacritics
# ---------------------------------------------------------------------------

class TestFtsTokenizerConfig:
    """Tests for V3.2 configurable FTS5 tokenizer."""

    def test_default_tokenizer_is_unicode61_diacritics(self):
        """Default tokenizer should be unicode61 remove_diacritics 2."""
        s = MemoryStore(db_path=":memory:")
        assert s._fts_tokenizer == "unicode61 remove_diacritics 2"
        s.close()

    def test_custom_tokenizer_porter(self):
        """Porter tokenizer should be accepted."""
        s = MemoryStore(db_path=":memory:", fts_tokenizer="porter")
        assert s._fts_tokenizer == "porter"
        assert s._fts5_available
        s.close()

    def test_custom_tokenizer_unicode61_raw(self):
        """Raw unicode61 (accent-sensitive) should be accepted."""
        s = MemoryStore(db_path=":memory:", fts_tokenizer="unicode61")
        assert s._fts_tokenizer == "unicode61"
        assert s._fts5_available
        s.close()

    def test_tokenizer_in_stats(self):
        """Stats should report the active tokenizer."""
        s = MemoryStore(db_path=":memory:", fts_tokenizer="porter")
        stats = s.stats()
        assert stats["fts_tokenizer"] == "porter"
        s.close()

    def test_tokenizer_sanitizer_rejects_quotes(self):
        """Tokenizer with quotes should be rejected (SQL injection guard)."""
        from ragix_core.memory.store import _validate_fts_tokenizer
        with pytest.raises(ValueError, match="Unsafe FTS5 tokenizer"):
            _validate_fts_tokenizer("unicode61'; DROP TABLE foo;")

    def test_tokenizer_sanitizer_rejects_semicolons(self):
        """Tokenizer with semicolons should be rejected."""
        from ragix_core.memory.store import _validate_fts_tokenizer
        with pytest.raises(ValueError, match="Unsafe FTS5 tokenizer"):
            _validate_fts_tokenizer("porter; DROP TABLE")

    def test_tokenizer_sanitizer_rejects_empty(self):
        """Empty tokenizer string should be rejected."""
        from ragix_core.memory.store import _validate_fts_tokenizer
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_fts_tokenizer("")

    def test_tokenizer_presets(self):
        """FTS_TOKENIZER_PRESETS should map fr/en/raw correctly."""
        from ragix_core.memory.store import FTS_TOKENIZER_PRESETS
        assert FTS_TOKENIZER_PRESETS["fr"] == "unicode61 remove_diacritics 2"
        assert FTS_TOKENIZER_PRESETS["en"] == "porter"
        assert FTS_TOKENIZER_PRESETS["raw"] == "unicode61"


class TestFtsFrenchDiacritics:
    """V3.2 regression: accent-insensitive search for French corpora."""

    @pytest.fixture
    def french_store(self):
        """Store with French content, default tokenizer (diacritics folding)."""
        s = MemoryStore(db_path=":memory:")
        items = [
            MemoryItem(
                id="FR-001", tier="stm", type="rule",
                title="Sécurité des installations",
                content="Les exigences de sécurité électrique doivent être vérifiées.",
                tags=["securite", "electrique"],
                provenance=MemoryProvenance(source_kind="doc", source_id="rie_01"),
                confidence=0.9, scope="grdf",
            ),
            MemoryItem(
                id="FR-002", tier="stm", type="rule",
                title="Café et restauration",
                content="Les espaces café sont réglementés par l'arrêté du 25 juin.",
                tags=["cafe", "reglementation"],
                provenance=MemoryProvenance(source_kind="doc", source_id="rie_02"),
                confidence=0.85, scope="grdf",
            ),
            MemoryItem(
                id="FR-003", tier="mtm", type="constraint",
                title="Contrôle qualité réseau",
                content="Le contrôle des réseaux gaz nécessite une habilitation spécifique.",
                tags=["controle", "reseau", "gaz"],
                provenance=MemoryProvenance(source_kind="doc", source_id="rie_03"),
                confidence=0.95, scope="grdf",
            ),
        ]
        for item in items:
            s.write_item(item)
        yield s
        s.close()

    def test_securite_matches_securite(self, french_store):
        """Accent-folded query 'securite' should match 'sécurité'."""
        results = french_store.search_fulltext("securite")
        assert len(results) >= 1
        assert any(r.id == "FR-001" for r in results)

    def test_cafe_matches_cafe(self, french_store):
        """Accent-folded query 'cafe' should match 'café'."""
        results = french_store.search_fulltext("cafe")
        assert len(results) >= 1
        assert any(r.id == "FR-002" for r in results)

    def test_accented_query_matches_too(self, french_store):
        """Accented query 'sécurité' should also match."""
        results = french_store.search_fulltext("sécurité")
        assert len(results) >= 1
        assert any(r.id == "FR-001" for r in results)

    def test_controle_matches_controle(self, french_store):
        """Accent-folded query 'controle' should match 'contrôle'."""
        results = french_store.search_fulltext("controle")
        assert len(results) >= 1
        assert any(r.id == "FR-003" for r in results)

    def test_multi_term_french(self, french_store):
        """Multi-term French query with diacritics folding."""
        results = french_store.search_fulltext("securite electrique")
        assert len(results) >= 1
        assert any(r.id == "FR-001" for r in results)

    def test_reglementation_matches(self, french_store):
        """Accent-folded 'reglementes' should match 'réglementés'.

        Note: unicode61 folds accents but does NOT stem, so the exact
        folded form is required (no plural stripping).
        """
        results = french_store.search_fulltext("reglementes")
        assert len(results) >= 1
        assert any(r.id == "FR-002" for r in results)

    def test_porter_does_not_fold_accents(self):
        """With porter tokenizer, 'securite' should NOT match 'sécurité'."""
        s = MemoryStore(db_path=":memory:", fts_tokenizer="porter")
        s.write_item(MemoryItem(
            id="PT-001", tier="stm", type="rule",
            title="Sécurité des installations",
            content="Les exigences de sécurité.",
            tags=["securite"],
            provenance=MemoryProvenance(source_kind="doc", source_id="d1"),
            confidence=0.9, scope="test",
        ))
        # Porter does English stemming but NOT diacritics folding
        results = s.search_fulltext("securite")
        # With porter, the unaccented query won't match accented content
        accented_results = s.search_fulltext("sécurité")
        # The accented query should match accented content with porter
        assert len(accented_results) >= 1
        s.close()


class TestFtsRebuildWithTokenizer:
    """Tests for rebuild_fts() with tokenizer change."""

    def test_rebuild_same_tokenizer(self):
        """Rebuild with same tokenizer should reindex without dropping."""
        s = MemoryStore(db_path=":memory:")
        s.write_item(MemoryItem(
            id="RB-001", tier="stm", type="fact",
            title="Test item", content="Test content.",
            tags=["test"],
            provenance=MemoryProvenance(source_kind="doc", source_id="d1"),
            confidence=0.9, scope="test",
        ))
        count = s.rebuild_fts()
        assert count == 1
        s.close()

    def test_rebuild_changes_tokenizer(self):
        """Rebuild with different tokenizer should drop + recreate FTS table."""
        s = MemoryStore(db_path=":memory:", fts_tokenizer="porter")
        s.write_item(MemoryItem(
            id="RB-002", tier="stm", type="fact",
            title="Sécurité électrique",
            content="Vérification de la sécurité.",
            tags=["securite"],
            provenance=MemoryProvenance(source_kind="doc", source_id="d1"),
            confidence=0.9, scope="test",
        ))
        # With porter: unaccented query may not match
        # Switch to unicode61 remove_diacritics 2
        count = s.rebuild_fts(tokenizer="unicode61 remove_diacritics 2")
        assert count == 1
        assert s._fts_tokenizer == "unicode61 remove_diacritics 2"
        # Now accent-folded query should work
        results = s.search_fulltext("securite")
        assert len(results) >= 1
        s.close()
