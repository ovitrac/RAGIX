"""
BM25 Index - Sparse keyword-based search for hybrid retrieval

Implements BM25 (Best Matching 25) algorithm for keyword search,
complementing vector similarity search.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """A document in the BM25 index."""

    doc_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0

    def __post_init__(self):
        self.token_count = len(self.tokens)


@dataclass
class BM25SearchResult:
    """Result from BM25 search."""

    doc_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    score: float
    matched_terms: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<BM25Result {self.file_path}:{self.name} score={self.score:.3f}>"


class Tokenizer:
    """
    Simple tokenizer for code and text.

    Handles camelCase, snake_case, and common programming constructs.
    """

    # Common stop words for code
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "if", "else", "elif",
        "then", "than", "so", "such", "no", "not", "only", "own", "same",
        "def", "class", "return", "import", "from", "self", "none", "true",
        "false", "pass", "break", "continue", "try", "except", "finally",
        "raise", "assert", "yield", "lambda", "with", "as", "global", "local",
    }

    def __init__(self, min_length: int = 2, use_stemming: bool = False):
        """
        Initialize tokenizer.

        Args:
            min_length: Minimum token length
            use_stemming: Whether to apply simple stemming
        """
        self.min_length = min_length
        self.use_stemming = use_stemming

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into normalized tokens.

        Handles:
        - camelCase splitting
        - snake_case splitting
        - Number removal
        - Punctuation removal
        - Lowercasing
        - Stop word removal

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Split camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Split snake_case
        text = text.replace('_', ' ')

        # Remove numbers (but keep words with numbers like 'v2')
        text = re.sub(r'\b\d+\b', ' ', text)

        # Remove punctuation except dots in file paths
        text = re.sub(r'[^\w\s\.]', ' ', text)

        # Split on whitespace and dots
        words = re.split(r'[\s\.]+', text.lower())

        # Filter and normalize
        tokens = []
        for word in words:
            if len(word) < self.min_length:
                continue
            if word in self.STOP_WORDS:
                continue
            if word.isdigit():
                continue

            # Simple stemming (remove common suffixes)
            if self.use_stemming:
                word = self._simple_stem(word)

            tokens.append(word)

        return tokens

    def _simple_stem(self, word: str) -> str:
        """Apply simple suffix stripping."""
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es', 'ies']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word


class BM25Index:
    """
    BM25 index for keyword-based code search.

    Implements the BM25 ranking algorithm with configurable parameters.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (typical: 1.2-2.0)
            b: Document length normalization (0=none, 1=full)
            tokenizer: Custom tokenizer (uses default if None)
        """
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or Tokenizer()

        # Index structures
        self.documents: Dict[str, BM25Document] = {}
        self.inverted_index: Dict[str, Set[str]] = {}  # term -> doc_ids
        self.term_frequencies: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: count}
        self.document_frequencies: Dict[str, int] = {}  # term -> num docs containing term

        # Statistics
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.total_tokens = 0

    def add_document(self, doc: BM25Document):
        """
        Add a document to the index.

        Args:
            doc: Document to add
        """
        if doc.doc_id in self.documents:
            # Remove old version first
            self._remove_document(doc.doc_id)

        self.documents[doc.doc_id] = doc
        self.total_docs += 1
        self.total_tokens += doc.token_count

        # Update inverted index
        term_counts = Counter(doc.tokens)
        for term, count in term_counts.items():
            # Inverted index
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(doc.doc_id)

            # Term frequencies
            if term not in self.term_frequencies:
                self.term_frequencies[term] = {}
            self.term_frequencies[term][doc.doc_id] = count

            # Document frequencies
            if term not in self.document_frequencies:
                self.document_frequencies[term] = 0
            self.document_frequencies[term] += 1

        # Update average document length
        self.avg_doc_length = self.total_tokens / self.total_docs

    def _remove_document(self, doc_id: str):
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return

        doc = self.documents[doc_id]
        term_counts = Counter(doc.tokens)

        for term in term_counts:
            if term in self.inverted_index:
                self.inverted_index[term].discard(doc_id)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

            if term in self.term_frequencies and doc_id in self.term_frequencies[term]:
                del self.term_frequencies[term][doc_id]
                if not self.term_frequencies[term]:
                    del self.term_frequencies[term]

            if term in self.document_frequencies:
                self.document_frequencies[term] -= 1
                if self.document_frequencies[term] <= 0:
                    del self.document_frequencies[term]

        self.total_tokens -= doc.token_count
        self.total_docs -= 1
        del self.documents[doc_id]

        if self.total_docs > 0:
            self.avg_doc_length = self.total_tokens / self.total_docs
        else:
            self.avg_doc_length = 0.0

    def search(self, query: str, k: int = 10) -> List[BM25SearchResult]:
        """
        Search the index using BM25 scoring.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of BM25SearchResult sorted by score (descending)
        """
        if self.total_docs == 0:
            return []

        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Calculate scores
        scores: Dict[str, Tuple[float, List[str]]] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            # IDF component
            df = self.document_frequencies.get(term, 0)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            # Score each document containing this term
            for doc_id in self.inverted_index[term]:
                doc = self.documents[doc_id]
                tf = self.term_frequencies[term].get(doc_id, 0)

                # BM25 term score
                doc_len = doc.token_count
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                term_score = idf * numerator / denominator

                if doc_id not in scores:
                    scores[doc_id] = (0.0, [])

                current_score, matched = scores[doc_id]
                scores[doc_id] = (current_score + term_score, matched + [term])

        # Sort by score
        sorted_results = sorted(
            [(doc_id, score, matched) for doc_id, (score, matched) in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Build results
        results = []
        for doc_id, score, matched_terms in sorted_results:
            doc = self.documents[doc_id]
            results.append(BM25SearchResult(
                doc_id=doc.doc_id,
                file_path=doc.file_path,
                start_line=doc.start_line,
                end_line=doc.end_line,
                chunk_type=doc.chunk_type,
                name=doc.name,
                score=score,
                matched_terms=list(set(matched_terms)),
            ))

        return results

    def save(self, path: Path):
        """
        Save index to disk.

        Args:
            path: Directory path to save index
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save documents
        docs_data = {}
        for doc_id, doc in self.documents.items():
            docs_data[doc_id] = {
                "file_path": doc.file_path,
                "start_line": doc.start_line,
                "end_line": doc.end_line,
                "chunk_type": doc.chunk_type,
                "name": doc.name,
                "tokens": doc.tokens,
            }

        with open(path / "bm25_documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f)

        # Save metadata
        metadata = {
            "k1": self.k1,
            "b": self.b,
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "total_tokens": self.total_tokens,
            "vocabulary_size": len(self.inverted_index),
        }

        with open(path / "bm25_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved BM25 index to {path} ({self.total_docs} documents)")

    def load(self, path: Path):
        """
        Load index from disk.

        Args:
            path: Directory path containing saved index
        """
        # Load metadata
        with open(path / "bm25_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.k1 = metadata["k1"]
        self.b = metadata["b"]

        # Load documents
        with open(path / "bm25_documents.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        # Clear and rebuild index
        self.documents.clear()
        self.inverted_index.clear()
        self.term_frequencies.clear()
        self.document_frequencies.clear()
        self.total_docs = 0
        self.total_tokens = 0

        for doc_id, data in docs_data.items():
            doc = BM25Document(
                doc_id=doc_id,
                file_path=data["file_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                chunk_type=data["chunk_type"],
                name=data["name"],
                tokens=data["tokens"],
            )
            self.add_document(doc)

        logger.info(f"Loaded BM25 index from {path} ({self.total_docs} documents)")

    @property
    def size(self) -> int:
        """Number of documents in index."""
        return self.total_docs

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in index."""
        return len(self.inverted_index)


def build_bm25_index_from_chunks(
    chunks: List,  # List of CodeChunk from chunking.py
    tokenizer: Optional[Tokenizer] = None,
) -> BM25Index:
    """
    Build a BM25 index from code chunks.

    Args:
        chunks: List of CodeChunk objects
        tokenizer: Optional custom tokenizer

    Returns:
        Populated BM25Index
    """
    index = BM25Index(tokenizer=tokenizer)

    for chunk in chunks:
        doc_id = f"{chunk.file_path}:{chunk.name}"
        tokens = index.tokenizer.tokenize(chunk.content)

        doc = BM25Document(
            doc_id=doc_id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
            name=chunk.name,
            tokens=tokens,
        )

        index.add_document(doc)

    logger.info(f"Built BM25 index: {index.size} documents, {index.vocabulary_size} terms")
    return index
