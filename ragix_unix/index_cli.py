#!/usr/bin/env python3
"""
RAGIX Indexing CLI - Build semantic code index

Usage:
    ragix-index <path> [options]

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

from ragix_core import (
    chunk_file,
    CodeChunk,
    create_embedding_backend,
    EmbeddingConfig,
    embed_code_chunks,
    save_embeddings,
    build_index_from_embeddings,
    BM25Index,
    build_bm25_index_from_chunks,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def discover_files(
    root_path: Path,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """
    Discover files to index in a directory tree.

    Args:
        root_path: Root directory to search
        include_patterns: File patterns to include (e.g., ["*.py", "*.md"])
        exclude_patterns: Patterns to exclude (e.g., ["**/test_*", "**/__pycache__"])

    Returns:
        List of file paths to index
    """
    if include_patterns is None:
        include_patterns = ["*.py", "*.md", "*.txt", "*.rst"]

    if exclude_patterns is None:
        exclude_patterns = [
            "**/__pycache__/**",
            "**/.*",
            "**/node_modules/**",
            "**/venv/**",
            "**/env/**",
            "**/.git/**",
            "**/build/**",
            "**/dist/**",
        ]

    files = []

    for pattern in include_patterns:
        for path in root_path.rglob(pattern):
            if not path.is_file():
                continue

            # Check if path matches any exclude pattern
            excluded = False
            for exclude in exclude_patterns:
                if path.match(exclude):
                    excluded = True
                    break

            if not excluded:
                files.append(path)

    return sorted(set(files))


def index_project(
    project_path: Path,
    output_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    backend_type: str = "numpy",
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    batch_size: int = 32,
    build_bm25: bool = True,
) -> int:
    """
    Index a project directory.

    Args:
        project_path: Path to project root
        output_dir: Directory to save index
        model_name: Embedding model name
        backend_type: Vector index backend ('numpy' or 'faiss')
        include_patterns: File patterns to include
        exclude_patterns: Patterns to exclude
        batch_size: Embedding batch size
        build_bm25: Whether to build BM25 index for hybrid search

    Returns:
        Number of chunks indexed
    """
    logger.info(f"Indexing project: {project_path}")
    logger.info(f"Output directory: {output_dir}")

    # Discover files
    logger.info("Discovering files...")
    files = discover_files(project_path, include_patterns, exclude_patterns)
    logger.info(f"Found {len(files)} files to index")

    if not files:
        logger.warning("No files found to index")
        return 0

    # Chunk files
    logger.info("Chunking files...")
    all_chunks: List[CodeChunk] = []
    for file_path in files:
        try:
            chunks = chunk_file(file_path)
            all_chunks.extend(chunks)
            logger.debug(f"  {file_path.relative_to(project_path)}: {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"  Failed to chunk {file_path}: {e}")

    logger.info(f"Generated {len(all_chunks)} chunks from {len(files)} files")

    if not all_chunks:
        logger.warning("No chunks generated")
        return 0

    # Create embedding backend
    logger.info(f"Initializing embedding model: {model_name}")
    config = EmbeddingConfig(model_name=model_name, batch_size=batch_size)
    embedding_backend = create_embedding_backend("sentence-transformers", config)

    # Generate embeddings
    logger.info("Generating embeddings...")
    chunk_embeddings = embed_code_chunks(all_chunks, embedding_backend, batch_size)
    logger.info(f"Generated {len(chunk_embeddings)} embeddings")

    # Save embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.json"
    logger.info(f"Saving embeddings to {embeddings_path}")
    save_embeddings(chunk_embeddings, embeddings_path)

    # Build vector index
    logger.info(f"Building {backend_type} vector index...")
    index = build_index_from_embeddings(chunk_embeddings, backend_type)
    logger.info(f"Index size: {index.size} vectors")

    # Save index
    index_path = output_dir / "index"
    logger.info(f"Saving index to {index_path}")
    index.save(index_path)

    # Build BM25 index for hybrid search
    bm25_index = None
    if build_bm25:
        logger.info("Building BM25 index for hybrid search...")
        bm25_index = build_bm25_index_from_chunks(all_chunks)
        bm25_path = output_dir / "bm25"
        logger.info(f"Saving BM25 index to {bm25_path}")
        bm25_index.save(bm25_path)
        logger.info(f"BM25 index: {bm25_index.size} documents, {bm25_index.vocabulary_size} terms")

    # Save metadata
    metadata = {
        "project_path": str(project_path.absolute()),
        "num_files": len(files),
        "num_chunks": len(all_chunks),
        "model_name": model_name,
        "backend_type": backend_type,
        "dimension": embedding_backend.dimension,
        "has_bm25": build_bm25,
        "bm25_vocabulary_size": bm25_index.vocabulary_size if bm25_index else 0,
    }

    import json

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ Indexing complete!")
    logger.info(f"   Files: {len(files)}")
    logger.info(f"   Chunks: {len(all_chunks)}")
    logger.info(f"   Vector dimension: {embedding_backend.dimension}")
    if bm25_index:
        logger.info(f"   BM25 vocabulary: {bm25_index.vocabulary_size} terms")

    return len(all_chunks)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGIX Indexing - Build semantic code index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current directory
  ragix-index .

  # Index specific directory with custom output
  ragix-index ~/my-project --output-dir .ragix/index

  # Use specific model
  ragix-index . --model sentence-transformers/all-mpnet-base-v2

  # Use FAISS backend for large projects
  ragix-index . --backend faiss

  # Custom file patterns
  ragix-index . --include "*.py" "*.js" --exclude "**/test_*"
""",
    )

    parser.add_argument("path", type=Path, help="Project directory to index")

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for index (default: <path>/.ragix/index)",
    )

    parser.add_argument(
        "--model",
        "-m",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--backend",
        "-b",
        choices=["numpy", "faiss"],
        default="numpy",
        help="Vector index backend (default: numpy)",
    )

    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="File patterns to include (default: *.py *.md *.txt *.rst)",
    )

    parser.add_argument(
        "--exclude", nargs="+", default=None, help="File patterns to exclude"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size (default: 32)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Skip BM25 index (only build vector index)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Resolve paths
    project_path = args.path.resolve()
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        sys.exit(1)

    if not project_path.is_dir():
        logger.error(f"Project path is not a directory: {project_path}")
        sys.exit(1)

    output_dir = args.output_dir or (project_path / ".ragix" / "index")

    # Run indexing
    try:
        num_chunks = index_project(
            project_path=project_path,
            output_dir=output_dir,
            model_name=args.model,
            backend_type=args.backend,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
            batch_size=args.batch_size,
            build_bm25=not args.no_bm25,
        )

        if num_chunks == 0:
            logger.warning("No chunks indexed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nIndexing cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
