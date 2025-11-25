#!/usr/bin/env python3
"""
RAGIX Hybrid Search Example - BM25 + Vector Search

This example demonstrates how to use RAGIX v0.7 hybrid search:
- BM25 sparse keyword search
- Vector semantic search
- Multiple fusion strategies
- Code-aware tokenization

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core import (
    # BM25 Index
    BM25Index,
    BM25Document,
    Tokenizer,
    # Hybrid Search
    FusionStrategy,
    # Chunking
    CodeChunk,
    ChunkType,
)


def demonstrate_tokenizer():
    """Show code-aware tokenization."""
    print("=" * 60)
    print("RAGIX v0.7 - Code-Aware Tokenization")
    print("=" * 60)

    tokenizer = Tokenizer(min_length=2, use_stemming=False)

    examples = [
        "def processDataItem(item):",
        "calculate_total_sum",
        "getUserByIdAndName",
        "The quick brown fox jumps",
        "HTTPRequestHandler class",
    ]

    print("\nTokenization Examples:")
    print("-" * 40)

    for text in examples:
        tokens = tokenizer.tokenize(text)
        print(f"  Input:  {text}")
        print(f"  Tokens: {tokens}")
        print()


def demonstrate_bm25_search():
    """Show BM25 keyword search."""
    print("=" * 60)
    print("RAGIX v0.7 - BM25 Search")
    print("=" * 60)

    # Create index
    index = BM25Index(k1=1.5, b=0.75)

    # Add sample documents (simulating code chunks)
    documents = [
        BM25Document(
            doc_id="main.py:1",
            file_path="main.py",
            start_line=1,
            end_line=10,
            chunk_type="function",
            name="main",
            tokens=["def", "main", "entry", "point", "application", "start"],
        ),
        BM25Document(
            doc_id="main.py:10",
            file_path="main.py",
            start_line=10,
            end_line=20,
            chunk_type="function",
            name="calculate_sum",
            tokens=["def", "calculate", "sum", "add", "numbers", "return", "int"],
        ),
        BM25Document(
            doc_id="utils.py:1",
            file_path="utils.py",
            start_line=1,
            end_line=15,
            chunk_type="function",
            name="load_config",
            tokens=["def", "load", "config", "json", "file", "read", "parse"],
        ),
        BM25Document(
            doc_id="handler.py:1",
            file_path="handler.py",
            start_line=1,
            end_line=30,
            chunk_type="class",
            name="DataHandler",
            tokens=["class", "data", "handler", "process", "item", "validate"],
        ),
        BM25Document(
            doc_id="handler.py:20",
            file_path="handler.py",
            start_line=20,
            end_line=35,
            chunk_type="function",
            name="process_data",
            tokens=["def", "process", "data", "transform", "validate", "return"],
        ),
    ]

    for doc in documents:
        index.add_document(doc)

    print(f"\nIndexed {index.size} documents")
    print(f"Vocabulary size: {index.vocabulary_size}")

    # Search examples
    queries = [
        "calculate sum numbers",
        "process data handler",
        "load config file",
        "main entry point",
    ]

    print("\nSearch Results:")
    print("-" * 40)

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = index.search(query, k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path}:{result.name} (score: {result.score:.3f})")
            print(f"     Matched terms: {result.matched_terms}")


def demonstrate_fusion_strategies():
    """Show different fusion strategies."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Fusion Strategies")
    print("=" * 60)

    print("\nAvailable Fusion Strategies:")
    print("-" * 40)

    strategies = [
        (FusionStrategy.RRF, "Reciprocal Rank Fusion - Best for balanced results"),
        (FusionStrategy.WEIGHTED, "Weighted combination - Tunable BM25/vector balance"),
        (FusionStrategy.INTERLEAVE, "Round-robin merging - Diverse results"),
        (FusionStrategy.BM25_RERANK, "Vector search, BM25 rerank - Keyword precision"),
        (FusionStrategy.VECTOR_RERANK, "BM25 search, vector rerank - Semantic precision"),
    ]

    for strategy, description in strategies:
        print(f"  - {strategy.value}: {description}")

    print("\nUsage Example:")
    print("-" * 40)
    print("""
    from ragix_core import create_hybrid_engine, FusionStrategy

    engine = create_hybrid_engine(
        index_path=Path(".ragix/index"),
        embedding_model="all-MiniLM-L6-v2",
    )

    # RRF fusion (default)
    results = engine.search("database connection error", k=10)

    # Weighted fusion with more emphasis on semantic
    results = engine.search(
        "database connection error",
        k=10,
        strategy=FusionStrategy.WEIGHTED,
        bm25_weight=0.3,
        vector_weight=0.7,
    )

    # Results include source tracking
    for r in results:
        print(f"{r.file_path}:{r.name}")
        print(f"  Score: {r.combined_score:.3f}")
        print(f"  Source: {r.source}")  # 'bm25', 'vector', or 'both'
        print(f"  BM25 terms: {r.bm25_matched_terms}")
    """)


def main():
    """Run all demonstrations."""
    demonstrate_tokenizer()
    demonstrate_bm25_search()
    demonstrate_fusion_strategies()

    print("\n" + "=" * 60)
    print("Hybrid search demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
