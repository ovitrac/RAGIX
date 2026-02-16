"""
summary_ingest — Stage 1: Rule Extraction + Memory Storage

LLM-assisted extraction via DocReader. Stores rules as policy-governed
memory items and creates pointer provenance.

V2.4: Delta mode — skips chunks from unchanged files, records corpus
      hashes after ingestion for future delta detection.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput


class SummaryIngestKernel(Kernel):
    name = "summary_ingest"
    version = "1.1.0"
    category = "summary"
    stage = 1
    description = "Extract rules from corpus via LLM and store in memory"
    requires = ["summary_collect"]
    provides = ["memory_items", "ingestion_metrics"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Extract rules from corpus chunks via LLM and store as policy-governed memory items."""
        from ragix_core.memory.config import MemoryConfig
        from ragix_core.memory.doc_reader import DocReader
        from ragix_core.memory.store import MemoryStore
        from ragix_core.memory.tools import MemoryToolDispatcher
        from ragix_core.memory.embedder import create_embedder
        from ragix_core.memory.policy import MemoryPolicy

        cfg = input.config
        scope = cfg.get("scope", "project")
        model = cfg.get("model", "ibm/granite4:32b-a9b-h")
        ollama_url = cfg.get("ollama_url", "http://localhost:11434")
        strategy = cfg.get("strategy", "pages")
        max_chunk_tokens = cfg.get("max_chunk_tokens", 800)
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
        embedder_backend = cfg.get("embedder_backend", "mock")
        embedder_model = cfg.get("embedder_model", "nomic-embed-text")
        input_folder = cfg.get("input_folder", str(input.workspace / "corpus"))
        delta_mode = cfg.get("delta", False)
        corpus_id = cfg.get("corpus_id", None)  # V3.0

        # Initialize memory subsystem
        mem_config = MemoryConfig()
        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
        embedder = create_embedder(
            backend=embedder_backend, model=embedder_model,
        )
        policy = MemoryPolicy(mem_config.policy)
        dispatcher = MemoryToolDispatcher(
            store=store, embedder=embedder, config=mem_config, policy=policy,
        )

        # V2.4: In delta mode, load collect output to find changed files
        skip_docs = set()
        if delta_mode:
            collect_path = input.workspace / "stage1" / "summary_collect.json"
            if collect_path.exists():
                with open(collect_path) as f:
                    collect_data = json.load(f).get("data", {})
                delta_info = collect_data.get("delta")
                if delta_info:
                    # Build set of unchanged file names to skip
                    unchanged_count = delta_info.get("unchanged_count", 0)
                    # All files minus new+modified = unchanged
                    all_names = {f["name"] for f in collect_data.get("files", [])}
                    changed_names = set(delta_info.get("new_files", []))
                    changed_names.update(delta_info.get("modified_files", []))
                    skip_docs = all_names - changed_names

        # Run ingestion (DocReader handles per-file processing)
        reader = DocReader(
            dispatcher=dispatcher,
            config=mem_config,
            llm_model=model,
            ollama_url=ollama_url,
        )
        metrics = reader.ingest_folder(
            root=input_folder,
            strategy=strategy,
            max_chunk_tokens=max_chunk_tokens,
            scope=scope,
            verbose=True,
            skip_docs=skip_docs if skip_docs else None,
        )

        # V2.4: Record corpus hashes after ingestion
        collect_path = input.workspace / "stage1" / "summary_collect.json"
        if collect_path.exists():
            with open(collect_path) as f:
                collect_data = json.load(f).get("data", {})
            for f_info in collect_data.get("files", []):
                store.write_corpus_hash(
                    file_path=f_info.get("path", f_info["name"]),
                    sha256=f_info.get("sha256", ""),
                    chunk_count=sum(
                        1 for c in collect_data.get("chunks", [])
                        if c.get("doc_name") == f_info["name"]
                    ),
                )

        # V3.0: Stamp corpus_id on all items from this ingestion
        if corpus_id:
            from ragix_core.memory.types import CorpusMetadata
            items = store.list_items(scope=scope, exclude_archived=True, limit=10000)
            stamped = 0
            for item in items:
                if item.corpus_id != corpus_id:
                    store.update_item(item.id, {"corpus_id": corpus_id})
                    stamped += 1
            # Register corpus metadata
            store.write_corpus_metadata(CorpusMetadata(
                corpus_id=corpus_id,
                corpus_label=cfg.get("corpus_label", corpus_id),
                parent_corpus_id=cfg.get("parent_corpus_id", None),
                doc_count=metrics.files_processed if hasattr(metrics, 'files_processed') else 0,
                item_count=len(items),
                scope=scope,
            ))

        # Store stats
        store_stats = store.stats()

        # V3.0: Collect new item IDs for delta consolidation downstream
        # Query STM items created during this ingestion run
        new_item_ids = []
        if delta_mode:
            stm_items = store.list_items(scope=scope, tier="stm", limit=10000)
            new_item_ids = [item.id for item in stm_items]

        result = {
            "ingestion": metrics.to_dict(),
            "store": store_stats,
            "db_path": db_path,
            "scope": scope,
            "model": model,
            "new_item_ids": new_item_ids,
            "corpus_id": corpus_id,
        }
        if delta_mode:
            result["delta_skipped_docs"] = len(skip_docs)

        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of ingestion results."""
        ing = data.get("ingestion", {})
        st = data.get("store", {})
        base = (
            f"Ingested {ing.get('files_processed', 0)} files, "
            f"{ing.get('chunks_processed', 0)} chunks. "
            f"Accepted {ing.get('items_accepted', 0)} rules. "
            f"Store: {st.get('total_items', 0)} items."
        )
        if data.get("delta_skipped_docs"):
            base += f" (delta: skipped {data['delta_skipped_docs']} unchanged docs)"
        return base
