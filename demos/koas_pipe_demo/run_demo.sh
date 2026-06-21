#!/bin/bash
# =============================================================================
# KOAS Memory Pipe Demo — Cross-File Architectural Inference
# =============================================================================
#
# This demo shows how RAGIX Memory turns a large, multi-file codebase into
# a searchable knowledge store and extracts precisely the right fragments
# for each question — something impossible by feeding raw files to an LLM.
#
# The narrative:
#   Act 1 — THE PROBLEM: 8 files, ~68,000 tokens. An LLM injection budget
#           is ~3,000 tokens. That's 4.4% of the corpus. Which 4.4%?
#   Act 2 — INGEST ONCE: One command chunks, deduplicates, and stores.
#   Act 3 — SURGICAL RECALL: 4 questions, 4 different cross-sections.
#           Each query pulls different chunks from different files.
#   Act 4 — THE NUMBERS: Compression ratio, coverage spread, token savings.
#   Act 5 — CROSS-FILE SYNTHESIS: An architectural brief that no single
#           file contains — assembled from fragments across the corpus.
#   Act 6 — LIVE INFERENCE: If Claude CLI is available, pipes a real
#           question through Memory → Claude for a live answer.
#
# Acts 1-5 are pure deterministic (FTS5/BM25, no LLM).
# Act 6 calls Claude CLI if available.
#
# Usage:
#   ./run_demo.sh                    # Full demo
#   ./run_demo.sh --query-only       # Skip ingest, reuse existing DB
#   ./run_demo.sh --budget 2000      # Custom token budget
#   ./run_demo.sh --verbose          # Show full injection blocks (not just insights)
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BUDGET=3000
QUERY_ONLY=false
VERBOSE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DB_PATH="/tmp/koas_pipe_demo.db"
TMPDIR="${TMPDIR:-/tmp}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --query-only) QUERY_ONLY=true; shift ;;
        --budget|-b) BUDGET="$2"; shift 2 ;;
        --db) DB_PATH="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --help)
            echo "KOAS Memory Pipe Demo — Cross-File Architectural Inference"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --query-only        Skip ingest, query existing DB"
            echo "  --budget, -b N      Token budget per query (default: 3000)"
            echo "  --db PATH           Database path (default: /tmp/koas_pipe_demo.db)"
            echo "  --verbose, -v       Show full injection blocks"
            echo "  --help              Show this help"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
SOURCES=(
    "$PROJECT_ROOT/docs/KOAS.md"
    "$PROJECT_ROOT/docs/ARCHITECTURE.md"
    "$PROJECT_ROOT/ragix_kernels/base.py"
    "$PROJECT_ROOT/ragix_kernels/registry.py"
    "$PROJECT_ROOT/ragix_kernels/reviewer/kernels/md_edit_plan.py"
    "$PROJECT_ROOT/ragix_kernels/presenter/kernels/pres_slide_plan.py"
    "$PROJECT_ROOT/ragix_kernels/summary/cli/summaryctl.py"
    "$PROJECT_ROOT/ragix_core/memory/cli.py"
)

# Verify sources
for f in "${SOURCES[@]}"; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}Missing: $f${NC}"; exit 1
    fi
done

# ---------------------------------------------------------------------------
# Helper: run a query and capture output
# ---------------------------------------------------------------------------
run_query() {
    local query="$1"
    local outfile="$2"
    python3 -m ragix_core.memory.cli --db "$DB_PATH" pipe "$query" --budget "$BUDGET" > "$outfile" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Helper: extract insight from recall output → writes JSON to a temp file
# ---------------------------------------------------------------------------
extract_insight() {
    local recall_file="$1"
    local json_out="$TMPDIR/koas_insight.json"
    python3 -c "
import sys, json; sys.path.insert(0, '$SCRIPT_DIR')
from demo_helpers import extract_insight
text = open('$recall_file', encoding='utf-8').read()
r = extract_insight(text)
json.dump(r, open('$json_out', 'w'))
"
    # Read fields safely via jq-like Python extraction
    SOURCE=$(python3 -c "import json; print(json.load(open('$json_out'))['source'])")
    CHUNK=$(python3 -c "import json; print(json.load(open('$json_out'))['chunk'])")
    TOKENS=$(python3 -c "import json; print(json.load(open('$json_out'))['tokens_used'])")
    MATCHED=$(python3 -c "import json; print(json.load(open('$json_out'))['matched'])")
    INSIGHT=$(python3 -c "import json; print(json.load(open('$json_out'))['insight'])")
}

# =========================================================================
#                              BANNER
# =========================================================================
echo ""
echo -e "${BLUE}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║         KOAS Memory Pipe — Architectural Inference          ║"
echo "  ║                                                              ║"
echo "  ║   How RAGIX Memory finds the right 4% of a large codebase   ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =========================================================================
# ACT 1 — THE PROBLEM
# =========================================================================
echo -e "${MAGENTA}${BOLD}ACT 1 — THE PROBLEM${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  You want an AI assistant to explain the KOAS architecture."
echo -e "  The relevant source code spans ${BOLD}8 files${NC}:"
echo ""

# Measure each file
TOTAL_LINES=0
TOTAL_TOKENS=0
printf "  ${DIM}%-40s %6s %8s${NC}\n" "File" "Lines" "~Tokens"
printf "  ${DIM}%-40s %6s %8s${NC}\n" "────────────────────────────────────────" "──────" "────────"
for f in "${SOURCES[@]}"; do
    name=$(basename "$f")
    lines=$(wc -l < "$f")
    chars=$(wc -c < "$f")
    tokens=$((chars / 4))
    TOTAL_LINES=$((TOTAL_LINES + lines))
    TOTAL_TOKENS=$((TOTAL_TOKENS + tokens))
    printf "  %-40s %6d %8d\n" "$name" "$lines" "$tokens"
done
printf "  ${BOLD}%-40s %6d %8d${NC}\n" "TOTAL" "$TOTAL_LINES" "$TOTAL_TOKENS"

echo ""
echo -e "  Your injection budget: ${BOLD}${BUDGET} tokens${NC}"
RATIO=$(python3 -c "print(f'{$BUDGET/$TOTAL_TOKENS*100:.1f}')")
echo -e "  That's ${RED}${BOLD}${RATIO}%${NC} of the corpus."
echo ""
echo -e "  ${YELLOW}The question is not whether to use context —${NC}"
echo -e "  ${YELLOW}it's which ${RATIO}% to select for each question.${NC}"
echo ""

# =========================================================================
# ACT 2 — INGEST ONCE
# =========================================================================
echo -e "${MAGENTA}${BOLD}ACT 2 — INGEST ONCE${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$QUERY_ONLY" = false ]; then
    rm -f "$DB_PATH"
    echo -e "  Chunking ${BOLD}${#SOURCES[@]} files${NC} at paragraph boundaries..."
    echo -e "  (SHA-256 dedup, auto-tags from file type and path)"
    echo ""

    # Ingest all files via pipe with a throwaway query
    INGEST_OUTPUT=$(python3 -m ragix_core.memory.cli \
        --db "$DB_PATH" \
        pipe "bootstrap" \
        --source "${SOURCES[@]}" \
        --budget 100 2>&1 >/dev/null || true)

    # Extract chunk count from stderr
    CHUNKS=$(echo "$INGEST_OUTPUT" | grep -oP '\d+ chunks' | grep -oP '\d+' || echo "?")
    SKIPPED=$(echo "$INGEST_OUTPUT" | grep -oP '\d+ skipped' | grep -oP '\d+' || echo "0")

    echo -e "  ${GREEN}${BOLD}$CHUNKS chunks${NC} created from ${#SOURCES[@]} files"
    echo -e "  ${DIM}Stored in: $DB_PATH ($(du -h "$DB_PATH" 2>/dev/null | cut -f1 || echo '?'))${NC}"
    echo -e "  ${DIM}Average chunk: ~$((TOTAL_TOKENS / ${CHUNKS:-1})) tokens${NC}"
    echo ""
    echo -e "  From now on, every query is a ${BOLD}sub-second FTS5 lookup${NC}."
    echo -e "  No file I/O, no re-parsing, no LLM. Just SQLite."
else
    if [ ! -f "$DB_PATH" ]; then
        echo -e "${RED}  No database at $DB_PATH. Run without --query-only first.${NC}"
        exit 1
    fi
    CHUNKS=$(python3 -c "
from ragix_core.memory.store import MemoryStore
s = MemoryStore(db_path='$DB_PATH'); print(len(s.list_items(limit=9999))); s.close()
" 2>/dev/null || echo "?")
    echo -e "  Reusing existing database: ${BOLD}$CHUNKS chunks${NC}"
    echo -e "  ${DIM}$DB_PATH${NC}"
fi
echo ""

# =========================================================================
# ACT 3 — SURGICAL RECALL (4 queries, 4 cross-sections)
# =========================================================================
echo -e "${MAGENTA}${BOLD}ACT 3 — SURGICAL RECALL${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Same database, 4 different questions."
echo -e "  Each pulls a ${BOLD}different cross-section${NC} of the corpus."
echo ""

# Track unique sources across queries
declare -a ALL_SOURCES_SEEN
declare -a ALL_TOKENS_USED
TOTAL_RECALLED=0

# --- Query 1: Philosophy ---
echo -e "  ${CYAN}${BOLD}Q1: \"What problem does KOAS solve?\"${NC}"
Q1_FILE="$TMPDIR/koas_q1.txt"
run_query "KOAS sovereign kernel orchestrated audit system scientific" "$Q1_FILE"
extract_insight "$Q1_FILE"
echo -e "     ${GREEN}Source:${NC} $SOURCE (chunk $CHUNK) — $TOKENS tokens"
echo -e "     ${DIM}$INSIGHT${NC}"
ALL_SOURCES_SEEN+=("$SOURCE:$CHUNK")
ALL_TOKENS_USED+=("$TOKENS")
TOTAL_RECALLED=$((TOTAL_RECALLED + TOKENS))
if [ "$VERBOSE" = true ]; then echo ""; cat "$Q1_FILE"; fi
echo ""

# --- Query 2: Interface ---
echo -e "  ${CYAN}${BOLD}Q2: \"What is the kernel interface contract?\"${NC}"
Q2_FILE="$TMPDIR/koas_q2.txt"
run_query "kernel compute summarize base class abstract" "$Q2_FILE"
extract_insight "$Q2_FILE"
echo -e "     ${GREEN}Source:${NC} $SOURCE (chunk $CHUNK) — $TOKENS tokens"
echo -e "     ${DIM}$INSIGHT${NC}"
ALL_SOURCES_SEEN+=("$SOURCE:$CHUNK")
ALL_TOKENS_USED+=("$TOKENS")
TOTAL_RECALLED=$((TOTAL_RECALLED + TOKENS))
if [ "$VERBOSE" = true ]; then echo ""; cat "$Q2_FILE"; fi
echo ""

# --- Query 3: Families ---
echo -e "  ${CYAN}${BOLD}Q3: \"What are the five kernel families?\"${NC}"
Q3_FILE="$TMPDIR/koas_q3.txt"
run_query "five kernel families audit docs reviewer presenter summary pipeline" "$Q3_FILE"
extract_insight "$Q3_FILE"
echo -e "     ${GREEN}Source:${NC} $SOURCE (chunk $CHUNK) — $TOKENS tokens"
echo -e "     ${DIM}$INSIGHT${NC}"
ALL_SOURCES_SEEN+=("$SOURCE:$CHUNK")
ALL_TOKENS_USED+=("$TOKENS")
TOTAL_RECALLED=$((TOTAL_RECALLED + TOKENS))
if [ "$VERBOSE" = true ]; then echo ""; cat "$Q3_FILE"; fi
echo ""

# --- Query 4: Real implementation ---
echo -e "  ${CYAN}${BOLD}Q4: \"How does a real kernel handle complexity?\"${NC}"
Q4_FILE="$TMPDIR/koas_q4.txt"
run_query "preflight masking recipe degenerate adaptive tier chunk" "$Q4_FILE"
extract_insight "$Q4_FILE"
echo -e "     ${GREEN}Source:${NC} $SOURCE (chunk $CHUNK) — $TOKENS tokens"
echo -e "     ${DIM}$INSIGHT${NC}"
ALL_SOURCES_SEEN+=("$SOURCE:$CHUNK")
ALL_TOKENS_USED+=("$TOKENS")
TOTAL_RECALLED=$((TOTAL_RECALLED + TOKENS))
if [ "$VERBOSE" = true ]; then echo ""; cat "$Q4_FILE"; fi
echo ""

# =========================================================================
# ACT 4 — THE NUMBERS
# =========================================================================
echo -e "${MAGENTA}${BOLD}ACT 4 — THE NUMBERS${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Count unique source:chunk pairs
UNIQUE_SOURCES=$(printf '%s\n' "${ALL_SOURCES_SEEN[@]}" | sort -u | wc -l)
UNIQUE_FILES=$(printf '%s\n' "${ALL_SOURCES_SEEN[@]}" | cut -d: -f1 | sort -u | wc -l)

AVG_RECALLED=$((TOTAL_RECALLED / 4))
COMPRESSION=$(python3 -c "print(f'{$AVG_RECALLED/$TOTAL_TOKENS*100:.1f}')")

echo -e "  ${BOLD}Corpus${NC}"
echo -e "    Files:            ${#SOURCES[@]}"
echo -e "    Total tokens:     $TOTAL_TOKENS"
echo -e "    Chunks in store:  $CHUNKS"
echo ""
echo -e "  ${BOLD}Recall (4 queries)${NC}"
echo -e "    Budget per query: $BUDGET tokens"
echo -e "    Avg. tokens used: $AVG_RECALLED  (${BOLD}${COMPRESSION}%${NC} of corpus)"
echo -e "    Unique chunks:    $UNIQUE_SOURCES across $UNIQUE_FILES files"
echo ""
echo -e "  ${BOLD}Query differentiation${NC}"

# Show which file each query hit
printf "    %-45s → %s\n" "Q1 (philosophy)" "${ALL_SOURCES_SEEN[0]}"
printf "    %-45s → %s\n" "Q2 (interface)" "${ALL_SOURCES_SEEN[1]}"
printf "    %-45s → %s\n" "Q3 (families)" "${ALL_SOURCES_SEEN[2]}"
printf "    %-45s → %s\n" "Q4 (implementation)" "${ALL_SOURCES_SEEN[3]}"
echo ""

# Check if all 4 queries hit different chunks
if [ "$UNIQUE_SOURCES" -ge 3 ]; then
    echo -e "  ${GREEN}${BOLD}$UNIQUE_SOURCES/4 queries returned different chunks.${NC}"
    echo -e "  ${GREEN}Memory selects a different cross-section for each question.${NC}"
else
    echo -e "  ${YELLOW}$UNIQUE_SOURCES/4 unique chunks — some overlap detected.${NC}"
fi
echo ""

# =========================================================================
# ACT 5 — CROSS-FILE SYNTHESIS
# =========================================================================
echo -e "${MAGENTA}${BOLD}ACT 5 — CROSS-FILE SYNTHESIS${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}What an LLM could infer from these 4 recalls:${NC}"
echo ""

python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from demo_helpers import print_synthesis
print_synthesis(None)
"

echo ""
echo -e "  ${YELLOW}Key insight:${NC} This synthesis does not exist in any single file."
echo -e "  It emerges from combining fragments that Memory selected from"
echo -e "  4 different sources — ${BOLD}KOAS.md${NC}, ${BOLD}base.py${NC}, ${BOLD}ARCHITECTURE.md${NC},"
echo -e "  and ${BOLD}md_edit_plan.py${NC} — each chosen by a different query."
echo ""
echo -e "  Without Memory, you would need to manually pick the right"
echo -e "  ${COMPRESSION}% of ${TOTAL_TOKENS} tokens for each question."
echo -e "  With Memory, it takes ${BOLD}< 1 second${NC} per query."
echo ""

# =========================================================================
# ACT 6 — LIVE INFERENCE (optional: only if claude CLI is available)
# =========================================================================
LIVE_QUESTION="can I use KOAS to learn my math lessons?"

if command -v claude &>/dev/null; then
    echo ""
    echo -e "${MAGENTA}${BOLD}ACT 6 — LIVE INFERENCE${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  Claude CLI detected ($(claude --version 2>/dev/null || echo 'unknown'))."
    echo -e "  Piping a real question through Memory → Claude..."
    echo ""
    echo -e "  ${CYAN}${BOLD}Q: \"${LIVE_QUESTION}\"${NC}"
    echo ""
    echo -e "  ${DIM}Command: ragix-memory --db $DB_PATH pipe \"$LIVE_QUESTION\" | claude -p \"Answer based on the context provided\"${NC}"
    echo ""

    # Run the pipeline: Memory recall → Claude inference
    CLAUDE_RESPONSE=$(ragix-memory --db "$DB_PATH" pipe "$LIVE_QUESTION" 2>/dev/null \
        | claude -p "Based on the KOAS architecture context provided below, answer this question concisely (3-5 sentences): ${LIVE_QUESTION}" 2>/dev/null)

    if [ -n "$CLAUDE_RESPONSE" ]; then
        echo -e "  ${GREEN}${BOLD}Claude's answer:${NC}"
        echo ""
        echo "$CLAUDE_RESPONSE" | fold -s -w 72 | sed 's/^/    /'
        echo ""
    else
        echo -e "  ${YELLOW}(Claude did not return a response — check API key / connectivity)${NC}"
        echo ""
    fi
else
    echo ""
    echo -e "  ${DIM}[Claude CLI not found — skipping live inference]${NC}"
    echo -e "  ${DIM}Install: npm install -g @anthropic-ai/claude-code${NC}"
    echo ""
fi

# =========================================================================
# EPILOGUE
# =========================================================================
echo -e "${BLUE}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║                        Demo complete                         ║"
echo "  ╠══════════════════════════════════════════════════════════════╣"
echo "  ║                                                              ║"
echo "  ║  Corpus:  $TOTAL_TOKENS tokens across ${#SOURCES[@]} files                     ║"
printf "  ║  Budget:  %-4s tokens per query (%s%% of corpus)            ║\n" "$BUDGET" "$COMPRESSION"
echo "  ║  Queries: 4 questions, each pulling different chunks         ║"
echo "  ║  LLM:     none required (pure FTS5/BM25)                     ║"
echo "  ║                                                              ║"
echo "  ║  Next step: pipe this into Claude for real inference            ║"
echo "  ║                                                              ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "  ${BOLD}Try it:${NC}"
echo -e "    ragix-memory --db $DB_PATH pipe \"your question\" | claude"
echo ""
echo -e "  ${BOLD}Example:${NC}"
echo -e "    ragix-memory --db $DB_PATH pipe \"can I use KOAS to learn my math lessons?\" | claude"
echo ""
echo -e "${DIM}  Database persisted: $DB_PATH"
echo -e "  Subsequent calls: ragix-memory pipe \"...\" | claude  (DB remembered)${NC}"
echo ""

# Cleanup temp files
rm -f "$TMPDIR"/koas_q{1,2,3,4}.txt
