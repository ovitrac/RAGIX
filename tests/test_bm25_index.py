"""
Tests for BM25 Index

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import pytest
from pathlib import Path

from ragix_core.bm25_index import (
    BM25Document,
    BM25SearchResult,
    Tokenizer,
    BM25Index,
    build_bm25_index_from_chunks,
)


class TestTokenizer:
    """Tests for code-aware tokenizer."""

    def test_basic_tokenization(self):
        """Test basic text tokenization."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_camel_case_splitting(self):
        """Test camelCase splitting."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("processDataItem")
        assert "process" in tokens
        assert "data" in tokens
        assert "item" in tokens

    def test_snake_case_splitting(self):
        """Test snake_case splitting."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("process_data_item")
        assert "process" in tokens
        assert "data" in tokens
        assert "item" in tokens

    def test_lowercase_normalization(self):
        """Test tokens are lowercased."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("HELLO World TEST")
        assert all(t == t.lower() for t in tokens)

    def test_stop_word_removal(self):
        """Test stop words are removed."""
        tokenizer = Tokenizer(use_stopwords=True)
        tokens = tokenizer.tokenize("the quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_no_stop_word_removal(self):
        """Test stop words kept when disabled."""
        tokenizer = Tokenizer(use_stopwords=False)
        tokens = tokenizer.tokenize("the quick brown fox")
        assert "the" in tokens

    def test_minimum_length_filter(self):
        """Test short tokens are filtered."""
        tokenizer = Tokenizer(min_token_length=3)
        tokens = tokenizer.tokenize("a ab abc abcd")
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens
        assert "abcd" in tokens

    def test_code_tokens(self):
        """Test tokenization of code-like text."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("def calculate_total_sum(items: List[int]) -> int:")
        assert "calculate" in tokens
        assert "total" in tokens
        assert "sum" in tokens
        assert "items" in tokens
        assert "list" in tokens
        assert "int" in tokens


class TestBM25Document:
    """Tests for BM25Document."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = BM25Document(
            doc_id="doc_001",
            tokens=["hello", "world"],
            metadata={"file": "test.py"},
        )
        assert doc.doc_id == "doc_001"
        assert doc.tokens == ["hello", "world"]
        assert doc.metadata["file"] == "test.py"


class TestBM25Index:
    """Tests for BM25 index."""

    def test_index_creation(self):
        """Test creating an empty index."""
        index = BM25Index()
        assert index.doc_count() == 0

    def test_add_document(self):
        """Test adding documents to index."""
        index = BM25Index()
        doc = BM25Document(
            doc_id="doc_001",
            tokens=["hello", "world", "python"],
            metadata={},
        )
        index.add_document(doc)

        assert index.doc_count() == 1

    def test_search_basic(self):
        """Test basic search."""
        index = BM25Index()

        # Add documents
        docs = [
            BM25Document("doc1", ["python", "programming", "language"], {}),
            BM25Document("doc2", ["java", "programming", "language"], {}),
            BM25Document("doc3", ["python", "data", "science"], {}),
        ]
        for doc in docs:
            index.add_document(doc)

        # Search for "python"
        results = index.search("python", k=3)

        assert len(results) == 2
        assert results[0].doc_id in ["doc1", "doc3"]
        assert results[1].doc_id in ["doc1", "doc3"]

    def test_search_ranking(self):
        """Test that more relevant docs rank higher."""
        index = BM25Index()

        # Add documents with varying relevance
        docs = [
            BM25Document("doc1", ["python"] * 5, {}),  # High relevance
            BM25Document("doc2", ["python"] * 2, {}),  # Medium relevance
            BM25Document("doc3", ["python"], {}),  # Low relevance
        ]
        for doc in docs:
            index.add_document(doc)

        results = index.search("python", k=3)

        # More occurrences should rank higher (with TF saturation)
        assert results[0].doc_id == "doc1"

    def test_search_matched_terms(self):
        """Test matched terms in results."""
        index = BM25Index()
        index.add_document(BM25Document(
            "doc1",
            ["python", "programming", "data", "science"],
            {},
        ))

        results = index.search("python data", k=1)

        assert len(results) == 1
        assert "python" in results[0].matched_terms
        assert "data" in results[0].matched_terms

    def test_search_no_results(self):
        """Test search with no matches."""
        index = BM25Index()
        index.add_document(BM25Document("doc1", ["hello", "world"], {}))

        results = index.search("nonexistent query", k=5)

        assert len(results) == 0

    def test_idf_weighting(self):
        """Test IDF weighting for rare terms."""
        index = BM25Index()

        # Add documents - "rare" appears only once, "common" appears in all
        docs = [
            BM25Document("doc1", ["common", "term"], {}),
            BM25Document("doc2", ["common", "term"], {}),
            BM25Document("doc3", ["common", "rare"], {}),
        ]
        for doc in docs:
            index.add_document(doc)

        # Search for both terms in doc3
        results = index.search("common rare", k=3)

        # doc3 should rank highest due to rare term
        assert results[0].doc_id == "doc3"

    def test_save_and_load(self, temp_dir: Path):
        """Test saving and loading index."""
        index = BM25Index()
        docs = [
            BM25Document("doc1", ["hello", "world"], {"file": "test1.py"}),
            BM25Document("doc2", ["python", "code"], {"file": "test2.py"}),
        ]
        for doc in docs:
            index.add_document(doc)

        # Save
        save_path = temp_dir / "bm25_index"
        index.save(save_path)

        # Load
        loaded_index = BM25Index.load(save_path)

        assert loaded_index.doc_count() == 2
        results = loaded_index.search("python", k=1)
        assert len(results) == 1
        assert results[0].doc_id == "doc2"

    def test_custom_parameters(self):
        """Test index with custom BM25 parameters."""
        index = BM25Index(k1=1.5, b=0.8)
        index.add_document(BM25Document("doc1", ["test", "query"], {}))

        results = index.search("test", k=1)
        assert len(results) == 1


class TestBuildBM25IndexFromChunks:
    """Tests for building BM25 index from code chunks."""

    def test_build_from_chunks(self, sample_chunks: list):
        """Test building index from code chunks."""
        # Convert fixture to proper format
        chunks = []
        for chunk in sample_chunks:
            from ragix_core.chunking import CodeChunk, ChunkType
            chunks.append(CodeChunk(
                file_path=chunk["file_path"],
                chunk_type=ChunkType.FUNCTION if chunk["chunk_type"] == "function" else ChunkType.CLASS,
                name=chunk["name"],
                content=chunk["content"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
            ))

        index = build_bm25_index_from_chunks(chunks)

        assert index.doc_count() == len(chunks)

    def test_search_chunks(self, sample_chunks: list):
        """Test searching code chunks."""
        from ragix_core.chunking import CodeChunk, ChunkType

        chunks = []
        for chunk in sample_chunks:
            chunks.append(CodeChunk(
                file_path=chunk["file_path"],
                chunk_type=ChunkType.FUNCTION if chunk["chunk_type"] == "function" else ChunkType.CLASS,
                name=chunk["name"],
                content=chunk["content"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
            ))

        index = build_bm25_index_from_chunks(chunks)

        # Search for "calculate"
        results = index.search("calculate sum", k=5)

        assert len(results) > 0
        # Should find the calculate_sum function
        found = any("calculate_sum" in str(r.metadata.get("name", "")) for r in results)
        assert found
