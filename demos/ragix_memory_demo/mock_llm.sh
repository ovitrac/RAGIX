#!/usr/bin/env bash
# =========================================================================
# mock_llm.sh — Deterministic mock LLM for RAGIX loop demo (Act 9a)
#
# Simulates a JSON-protocol-compliant LLM that:
#   - Iteration 1: requests more context about "memory consolidation"
#   - Iteration 2: requests more context about "injection block format"
#   - Iteration 3+: produces a final answer (stop=true)
#
# The mock reads stdin (the prompt) and outputs a response based on
# how many times it has been called (tracked via a counter file).
#
# Usage:
#   echo "prompt" | bash demos/ragix_memory_demo/mock_llm.sh
#   # or via ragix-memory loop:
#   ragix-memory loop --llm "bash demos/ragix_memory_demo/mock_llm.sh" ...
#
# The counter file path is controlled by MOCK_LLM_STATE env var.
# Default: /tmp/ragix_mock_llm_state
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# =========================================================================

set -euo pipefail

STATE_FILE="${MOCK_LLM_STATE:-/tmp/ragix_mock_llm_state}"

# Read stdin (prompt) — required by the protocol
PROMPT=$(cat)

# Read and increment call counter
if [[ -f "$STATE_FILE" ]]; then
    CALL=$(cat "$STATE_FILE")
else
    CALL=0
fi
CALL=$((CALL + 1))
echo "$CALL" > "$STATE_FILE"

# Respond based on iteration
case $CALL in
    1)
        cat <<'RESPONSE'
{"need_more": true, "query": "memory consolidation tier promotion", "rationale": "The context describes ingestion and recall but lacks details on how short-term memory items are consolidated and promoted to longer-term tiers.", "stop": false}

Based on the initial context, the RAGIX memory subsystem provides a complete pipeline for document ingestion and retrieval. Documents are chunked at paragraph boundaries, stored with SHA-256 content hashes for deduplication, and indexed via FTS5/BM25 for full-text search.

Key findings so far:
- Documents are ingested via ragix-memory ingest with automatic chunking
- FTS5/BM25 provides full-text search with BM25 ranking
- Token-budgeted injection blocks are formatted for LLM context windows
- All items stored in SQLite with WAL mode

I need more information about how memory items are consolidated from STM to MTM and LTM tiers.
RESPONSE
        ;;
    2)
        cat <<'RESPONSE'
{"need_more": true, "query": "injection block format token budget pipe", "rationale": "Consolidation is now understood but the exact format of injection blocks and how pipe composes recall with token budgets needs clarification.", "stop": false}

The RAGIX memory data flow is becoming clearer. Adding consolidation details:

Architecture overview:
- Documents are ingested and chunked at paragraph boundaries (~1800 tokens)
- SHA-256 content hashes ensure idempotent re-ingestion (zero duplicates)
- Items start in STM (short-term memory) tier
- Consolidation clusters similar STM items via Jaccard similarity on tags
- Merged items are promoted to MTM (medium-term memory)
- High-usage MTM items can be promoted to LTM (long-term memory)

The three-tier model (STM -> MTM -> LTM) ensures knowledge matures through verification and usage. However, I still need to understand the injection block format and how the pipe command composes recall with token budgets.
RESPONSE
        ;;
    *)
        cat <<'RESPONSE'
{"need_more": false, "query": null, "rationale": null, "stop": true}

# RAGIX Memory Data Flow — Complete Analysis

## 1. Document Ingestion
- Source files are read via format-specific extractors (Markdown, PDF, DOCX, etc.)
- Text is chunked at paragraph boundaries (~1800 tokens per chunk)
- Each chunk receives a SHA-256 content hash for deduplication
- Items are stored in SQLite with FTS5 indexing (BM25 ranking)
- Provenance tracked: source path, chunk index, content hash

## 2. Recall & Retrieval
- FTS5/BM25 full-text search with tier and type filtering
- Token-budgeted recall: items ranked by relevance, trimmed to fit budget
- Budget enforcement: sum of item token estimates must not exceed limit

## 3. Injection Block Format (FORMAT_VERSION=1)
- Header: format_version, budget_tokens, matched count, used count
- Body: numbered items with tier, type, title, and content
- Footer: end marker with format_version for block detection
- The pipe command composes ingestion + recall into a single stdout stream

## 4. Memory Lifecycle (Three-Tier Model)
- **STM** (Short-Term): Raw ingested items, unverified observations
- **MTM** (Medium-Term): Consolidated clusters, verified knowledge
- **LTM** (Long-Term): Stable decisions, promoted by usage frequency
- Consolidation: deterministic Jaccard clustering, no LLM required

## 5. LLM Integration
- The pipe command outputs injection blocks to stdout
- Any LLM can consume via Unix pipe: ollama, claude, or custom
- The loop controller adds iterative refinement with convergence detection
- Pull captures LLM output back into memory (feedback loop)
RESPONSE
        ;;
esac
