"""
DocReader — Large Document Ingestion Agent

Reads documents chunk-by-chunk, extracts rule candidates via LLM,
stores as MemoryItems, and triggers consolidation when context
fraction threshold is reached.

Two-phase protocol:
  Phase 1 (Ingestion): DocReader reads chunks -> LLM extracts rules ->
                        policy-governed storage -> periodic consolidation
  Phase 2 (Summarization): Main model uses only recalled memory items

The DocReader never sends the full document to the LLM. Each chunk is
processed independently with a focused extraction prompt.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_core.memory.config import ConsolidateConfig, MemoryConfig
from ragix_core.memory.doc_tools import (
    DocChunk,
    DocStats,
    doc_chunk_plan,
    doc_list,
    doc_stats,
)
from ragix_core.memory.tools import MemoryToolDispatcher
from ragix_core.memory.types import MemoryProposal, _now_iso
from ragix_core.shared.text_utils import normalize_whitespace, estimate_tokens

logger = logging.getLogger(__name__)



def _estimate_tokens_text(text: str) -> int:
    """Rough token count: chars / 4."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Triage configuration
# ---------------------------------------------------------------------------

@dataclass
class DocTriage:
    """Triage entry for a document."""
    path: str
    name: str
    tier: str = "T1"  # T1=direct match, T2=infrastructure, T3=skip
    relevance: str = ""
    max_pages: int = 0  # 0 = all pages
    priority: int = 1   # lower = higher priority


# ---------------------------------------------------------------------------
# Ingestion metrics
# ---------------------------------------------------------------------------

@dataclass
class IngestionMetrics:
    """Tracks metrics during ingestion."""
    files_processed: int = 0
    chunks_processed: int = 0
    tokens_read: int = 0
    items_proposed: int = 0
    items_accepted: int = 0
    items_rejected: int = 0
    consolidations_triggered: int = 0
    total_stm_tokens: int = 0  # running estimate of STM memory tokens
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ingestion metrics to a plain dictionary."""
        return {
            "files_processed": self.files_processed,
            "chunks_processed": self.chunks_processed,
            "tokens_read": self.tokens_read,
            "items_proposed": self.items_proposed,
            "items_accepted": self.items_accepted,
            "items_rejected": self.items_rejected,
            "consolidations_triggered": self.consolidations_triggered,
            "total_stm_tokens": self.total_stm_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


# ---------------------------------------------------------------------------
# Rule extraction prompt
# ---------------------------------------------------------------------------

_RULE_EXTRACT_PROMPT = """\
You are a technical compliance analyst extracting rules from a regulatory document.

**Document**: {doc_name}
**Section**: {heading}
**Context**: This is a "Référentiel d'Ingénierie et d'Exploitation" (RIE) from CORP-ENERGY.

Read the following chunk and extract ALL rules, constraints, definitions, and decisions.
For each, output a JSON array of objects with these fields:
- "type": one of "constraint", "definition", "decision", "pattern"
- "title": concise rule title starting with the technology name and version (e.g., "PostgreSQL 13: connection pooling limit", "RHEL 9: SSH AllowGroups policy")
- "content": the rule text, concise and canonical (max 200 chars)
- "tags": relevant tags (technology, domain, security level, etc.)
- "why_store": brief justification for storing this rule
- "rule_id": the document rule ID if present (e.g., "RIE-RHEL-042"), otherwise null

IMPORTANT:
- Each rule title MUST start with the technology name and version as found in the document.
- If the document contains rule IDs (e.g., RIE-RHEL-NNN, REX-PG-NNN patterns), include them.

Output ONLY the JSON array. If no rules found, output [].

---
CHUNK TEXT:
{chunk_text}
---

JSON rules array:"""

_RULE_EXTRACT_PROMPT_SIMPLE = """\
Extract rules from this RIE document chunk. Output a JSON array of objects.
Each object: {{"type": "constraint"|"definition"|"decision"|"pattern", "title": "...", "content": "...", "tags": [...], "why_store": "...", "rule_id": "..." or null}}
IMPORTANT: Each title MUST start with the technology name and version (e.g., "PostgreSQL 13: ..."). Include rule IDs if present in the document (e.g., RIE-RHEL-042).
If no rules, output [].

Document: {doc_name} | Section: {heading}

{chunk_text}

JSON:"""


# ---------------------------------------------------------------------------
# DocReader Agent
# ---------------------------------------------------------------------------

class DocReader:
    """
    Large document ingestion agent.

    Reads documents chunk-by-chunk, extracts rules via LLM,
    stores as policy-governed memory items, and triggers
    consolidation when context fraction threshold is reached.
    """

    def __init__(
        self,
        dispatcher: MemoryToolDispatcher,
        config: Optional[MemoryConfig] = None,
        llm_model: str = "gpt-oss-safeguard:120b",
        ollama_url: str = "http://localhost:11434",
    ):
        """Initialize DocReader with memory dispatcher, config, and LLM settings."""
        self._dispatcher = dispatcher
        self._config = config or MemoryConfig()
        self._llm_model = llm_model
        self._ollama_url = ollama_url.rstrip("/")
        self._metrics = IngestionMetrics()
        # Context tracking
        self._ctx_limit = self._config.consolidate.ctx_limit_tokens
        self._ctx_fraction = self._config.consolidate.ctx_fraction_trigger
        self._scope = "project"  # updated per ingest call

    @property
    def metrics(self) -> IngestionMetrics:
        """Return current ingestion metrics."""
        return self._metrics

    def ingest_folder(
        self,
        root: str,
        triage: Optional[Dict[str, DocTriage]] = None,
        strategy: str = "pages",
        pages_per_chunk: int = 5,
        max_chunk_tokens: int = 800,
        scope: str = "project",
        verbose: bool = False,
    ) -> IngestionMetrics:
        """
        Ingest all documents in a folder.

        Args:
            root: directory containing documents
            triage: optional triage map {filename: DocTriage}
            strategy: chunking strategy ("pages", "headings", "windows")
            pages_per_chunk: pages per chunk for "pages" strategy
            max_chunk_tokens: max tokens per chunk
            scope: memory scope label
            verbose: print progress
        """
        start = time.monotonic()
        self._metrics = IngestionMetrics()
        self._scope = scope

        # List files
        files = doc_list(root, globs=["*.pdf", "*.md", "*.txt"], sort="name")
        if verbose:
            print(f"Found {len(files)} files in {root}")

        for file_info in files:
            name = file_info["name"]

            # Apply triage
            if triage:
                entry = triage.get(name)
                if entry is None or entry.tier == "T3":
                    if verbose:
                        print(f"  SKIP: {name} (T3 or not in triage)")
                    continue
            else:
                entry = None

            if verbose:
                tier = entry.tier if entry else "?"
                print(f"  [{tier}] Processing: {name} ({file_info['size_mb']} MB)")

            self._ingest_file(
                path=file_info["path"],
                doc_name=name,
                strategy=strategy,
                pages_per_chunk=pages_per_chunk,
                max_chunk_tokens=max_chunk_tokens,
                scope=scope,
                triage_entry=entry,
                verbose=verbose,
            )
            self._metrics.files_processed += 1

        self._metrics.elapsed_seconds = time.monotonic() - start
        if verbose:
            print(f"\nIngestion complete: {json.dumps(self._metrics.to_dict(), indent=2)}")
        return self._metrics

    def ingest_file(
        self,
        path: str,
        strategy: str = "pages",
        pages_per_chunk: int = 5,
        max_chunk_tokens: int = 800,
        scope: str = "project",
        verbose: bool = False,
    ) -> IngestionMetrics:
        """Ingest a single file."""
        start = time.monotonic()
        self._metrics = IngestionMetrics()
        self._scope = scope
        name = Path(path).name

        self._ingest_file(
            path=path, doc_name=name,
            strategy=strategy, pages_per_chunk=pages_per_chunk,
            max_chunk_tokens=max_chunk_tokens, scope=scope,
            verbose=verbose,
        )
        self._metrics.files_processed = 1
        self._metrics.elapsed_seconds = time.monotonic() - start
        return self._metrics

    def _ingest_file(
        self,
        path: str,
        doc_name: str,
        strategy: str,
        pages_per_chunk: int,
        max_chunk_tokens: int,
        scope: str,
        triage_entry: Optional[DocTriage] = None,
        verbose: bool = False,
    ) -> None:
        """Internal: process a single file chunk-by-chunk."""
        chunks = doc_chunk_plan(
            path, strategy=strategy,
            max_chunk_tokens=max_chunk_tokens,
        )

        if verbose:
            total_tokens = sum(c.token_estimate for c in chunks)
            print(f"    {len(chunks)} chunks, ~{total_tokens} tokens total")

        for i, chunk in enumerate(chunks):
            if not chunk.text.strip():
                continue

            self._metrics.chunks_processed += 1
            self._metrics.tokens_read += chunk.token_estimate

            # Extract rules via LLM
            proposals = self._extract_rules(chunk, doc_name)

            if proposals:
                # Submit through governance pipeline
                items_data = []
                for p in proposals:
                    p.scope = scope
                    p.provenance_hint = {
                        "source_kind": "doc",
                        "source_id": f"{doc_name}:{chunk.chunk_id}",
                        "chunk_ids": [chunk.chunk_id],
                        "content_hashes": [chunk.content_hash],
                    }
                    d = p.to_dict()
                    # Propagate rule_id if the LLM extracted one
                    if hasattr(p, "rule_id") and p.rule_id:
                        d["rule_id"] = p.rule_id
                    items_data.append(d)

                result = self._dispatcher.dispatch("propose", {"items": items_data})
                accepted = result.get("accepted", 0)
                rejected = result.get("rejected", 0)

                self._metrics.items_proposed += len(proposals)
                self._metrics.items_accepted += accepted
                self._metrics.items_rejected += rejected

                # Update STM token estimate
                for item_info in result.get("items", []):
                    if item_info.get("action") != "rejected":
                        self._metrics.total_stm_tokens += 100  # rough estimate per item

                if verbose:
                    print(
                        f"    chunk {i+1}/{len(chunks)}: "
                        f"{len(proposals)} rules, {accepted} accepted, {rejected} rejected"
                    )
            elif verbose:
                print(f"    chunk {i+1}/{len(chunks)}: 0 rules extracted")

            # Also store a pointer to the chunk
            self._store_pointer(chunk, doc_name, scope)

            # Check context-fraction trigger
            if self._should_consolidate():
                if verbose:
                    frac = self._metrics.total_stm_tokens / self._ctx_limit
                    print(f"    >> CONSOLIDATION triggered ({frac:.1%} of context)")
                self._trigger_consolidation(scope)

    def _extract_rules(
        self, chunk: DocChunk, doc_name: str,
    ) -> List[MemoryProposal]:
        """Call LLM to extract rules from a chunk."""
        # Normalize Unicode whitespace (U+202F etc.) before sending to LLM
        chunk_text = normalize_whitespace(chunk.text[:3000])
        prompt = _RULE_EXTRACT_PROMPT_SIMPLE.format(
            doc_name=doc_name,
            heading=chunk.heading or "(no heading)",
            chunk_text=chunk_text,
        )

        try:
            import requests
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 2000},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM extraction failed for {chunk.chunk_id}: {e}")
            return []

        # Parse JSON array from response
        return self._parse_rules_json(text)

    def _parse_rules_json(self, text: str) -> List[MemoryProposal]:
        """Parse LLM response as JSON array of rule proposals."""
        import re

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [MemoryProposal.from_dict(d) for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [MemoryProposal.from_dict(d) for d in data if isinstance(d, dict)]
            except json.JSONDecodeError:
                pass

        # Try individual JSON objects
        proposals = []
        for obj_match in re.finditer(r"\{[^{}]+\}", text):
            try:
                d = json.loads(obj_match.group())
                if "title" in d or "content" in d:
                    proposals.append(MemoryProposal.from_dict(d))
            except json.JSONDecodeError:
                continue

        return proposals

    def _store_pointer(self, chunk: DocChunk, doc_name: str, scope: str) -> None:
        """Store a pointer memory item for chunk provenance."""
        self._dispatcher.dispatch("propose", {
            "items": [{
                "type": "pointer",
                "title": f"[{doc_name}] {chunk.heading or f'chunk {chunk.chunk_id}'}",
                "content": (
                    f"Document: {doc_name}\n"
                    f"Chunk: {chunk.chunk_id}\n"
                    f"Pages: {chunk.start_page}-{chunk.end_page}\n"
                    f"Lines: {chunk.start_line}-{chunk.end_line}\n"
                    f"Tokens: ~{chunk.token_estimate}\n"
                    f"Hash: {chunk.content_hash}"
                ),
                "tags": ["pointer", "chunk", doc_name.lower().replace(" ", "-")],
                "why_store": "Document chunk provenance pointer",
                "scope": scope,
                "provenance_hint": {
                    "source_kind": "doc",
                    "source_id": f"{doc_name}:{chunk.chunk_id}",
                    "chunk_ids": [chunk.chunk_id],
                    "content_hashes": [chunk.content_hash],
                },
            }],
        })

    # -- Context-fraction trigger ------------------------------------------

    def _should_consolidate(self) -> bool:
        """
        Check if consolidation should be triggered.

        Two conditions (either triggers):
          1. STM token estimate >= ctx_fraction * ctx_limit
          2. STM count >= stm_threshold (from config)
        """
        # Context fraction trigger
        if self._ctx_limit > 0:
            fraction = self._metrics.total_stm_tokens / self._ctx_limit
            if fraction >= self._ctx_fraction:
                return True

        # STM count trigger (scope-aware)
        stm_count = self._dispatcher.store.count_items(tier="stm", scope=self._scope)
        if stm_count >= self._config.consolidate.stm_threshold:
            return True

        return False

    def _trigger_consolidation(self, scope: str) -> None:
        """Run consolidation and reset token counter."""
        result = self._dispatcher.dispatch("consolidate", {
            "scope": scope,
            "tiers": ["stm"],
            "promote": True,
        })
        self._metrics.consolidations_triggered += 1
        # Reset STM token estimate (consolidated items are smaller)
        merged = result.get("items_merged", 0)
        self._metrics.total_stm_tokens = max(
            0,
            self._metrics.total_stm_tokens - (merged * 80),  # estimate
        )
        logger.info(
            f"Consolidation #{self._metrics.consolidations_triggered}: "
            f"merged={merged}, promoted={result.get('items_promoted', 0)}"
        )
