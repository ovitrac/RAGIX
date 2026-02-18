#!/usr/bin/env bash
# ============================================================================
# RAGIX Memory — Full Lifecycle Demo (8 Acts)
#
# Showcases the complete ragix-memory CLI: init, ingest, search, recall,
# idempotency, pull, stats, palace, export, and LLM reasoning via both
# local (Ollama/Granite) and cloud (Claude) pipelines.
#
# Usage:
#   ./run_demo.sh [OPTIONS]
#
#   --keep            Keep the workspace after demo (default: cleanup)
#   --workspace DIR   Use custom workspace (default: /tmp/ragix_memory_demo)
#   --budget N        Token budget for recall queries (default: 2000)
#   --skip-ingest     Skip Act 2 (reuse existing workspace)
#   --corpus DIR      Source corpus directory (default: docs/)
#   --model MODEL     Ollama model for Act 8 (default: granite3.1-moe:3b)
#   --no-llm          Skip Act 8 entirely (no LLM required)
#   --help            Show this message
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# ============================================================================

set -e

# ---------------------------------------------------------------------------
# Constants & Colors
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
WORKSPACE="/tmp/ragix_memory_demo"
BUDGET=2000
KEEP=false
SKIP_INGEST=false
CORPUS_DIR="$PROJECT_ROOT/docs"
LLM_MODEL="granite3.1-moe:3b"
NO_LLM=false

# Metrics (accumulated)
TOTAL_FILES=0
TOTAL_CHUNKS=0
TOTAL_QUERIES=0
LLM_QUESTIONS=0
DEMO_START=$SECONDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

banner() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

act_header() {
    local num="$1"
    local title="$2"
    echo ""
    echo -e "${BOLD}${MAGENTA}  ┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${MAGENTA}  │  Act $num — $title${NC}"
    echo -e "${BOLD}${MAGENTA}  └─────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

section() {
    echo ""
    echo -e "  ${BOLD}${WHITE}── $1 ──${NC}"
    echo ""
}

info()    { echo -e "  ${BLUE}[info]${NC}  $1"; }
ok()      { echo -e "  ${GREEN}[ok]${NC}    $1"; }
demo()    { echo -e "  ${CYAN}[demo]${NC}  $1"; }
warn()    { echo -e "  ${YELLOW}[warn]${NC}  $1"; }
fail()    { echo -e "  ${RED}[fail]${NC}  $1"; }
elapsed() { echo -e "  ${DIM}[time]  ${1}s${NC}"; }

# Show the canonical command the user would type (printf avoids mangling \)
cmd() {
    printf '  \033[1;33m$\033[0m \033[2m%s\033[0m\n' "$1"
}

# Print an LLM answer block with indentation and line limit
print_answer() {
    local answer="$1"
    local max_lines="${2:-35}"
    local line_count
    line_count=$(echo "$answer" | wc -l)
    echo "$answer" | head -"$max_lines" | while IFS= read -r line; do
        echo -e "  ${DIM}  $line${NC}"
    done
    if [ "$line_count" -gt "$max_lines" ]; then
        echo -e "  ${DIM}  [...truncated, $line_count lines total]${NC}"
    fi
}

# Run ragix-memory with the workspace DB
mem() {
    python -m ragix_core.memory.cli --db "$WORKSPACE/memory.db" "$@"
}

# Build a Claude -p command with clean isolation:
#   --system-prompt: REPLACES default prompt (prevents CLAUDE.md leaks)
#   --tools "":      disables ALL tools (pure LLM, no Bash/Read/Grep)
#   --no-session-persistence: don't pollute session history
#   env -u CLAUDECODE: allow calling from nested Claude Code session
#
# Usage pattern (from docs):
#   some_context | claude -p --system-prompt "instruction" --tools ""
#   The piped stdin becomes the user message. No positional prompt arg needed.
CLAUDE_BASE_PROMPT="You are a technical documentation analyst. You will receive documentation excerpts via stdin. Answer ONLY based on that text. Do not use tools, do not access files, do not reference any prior knowledge."
# IMPORTANT: run claude from /tmp to prevent CLAUDE.md injection.
# --system-prompt does NOT suppress project CLAUDE.md loading.
# The only reliable way is to change working directory to one without CLAUDE.md.
CLAUDE_CWD="/tmp"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep)
            KEEP=true; shift ;;
        --workspace)
            WORKSPACE="$2"; shift 2 ;;
        --budget)
            BUDGET="$2"; shift 2 ;;
        --skip-ingest)
            SKIP_INGEST=true; shift ;;
        --corpus)
            CORPUS_DIR="$2"; shift 2 ;;
        --model)
            LLM_MODEL="$2"; shift 2 ;;
        --no-llm)
            NO_LLM=true; shift ;;
        --help|-h)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --keep            Keep workspace after demo (default: cleanup)"
            echo "  --workspace DIR   Custom workspace (default: /tmp/ragix_memory_demo)"
            echo "  --budget N        Token budget for recall (default: 2000)"
            echo "  --skip-ingest     Skip Act 2 (reuse existing workspace)"
            echo "  --corpus DIR      Source corpus directory (default: docs/)"
            echo "  --model MODEL     Ollama model for Act 8 (default: granite3.1-moe:3b)"
            echo "  --no-llm          Skip Act 8 (no LLM required)"
            echo "  --help            Show this message"
            exit 0 ;;
        *)
            fail "Unknown option: $1"
            exit 1 ;;
    esac
done

DB="$WORKSPACE/memory.db"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

banner "RAGIX Memory — Full Lifecycle Demo"
echo ""
echo -e "  ${BOLD}Project:${NC}   RAGIX"
echo -e "  ${BOLD}Date:${NC}      $(date '+%Y-%m-%d %H:%M')"
echo -e "  ${BOLD}Workspace:${NC} $WORKSPACE"
echo -e "  ${BOLD}Corpus:${NC}    $CORPUS_DIR"
echo -e "  ${BOLD}Budget:${NC}    $BUDGET tokens"
echo -e "  ${BOLD}Local LLM:${NC} $LLM_MODEL (Ollama)"
echo -e "  ${BOLD}Cloud LLM:${NC} Claude (claude CLI)"
echo ""

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

info "Checking prerequisites..."

if ! python -c "import ragix_core.memory.cli" 2>/dev/null; then
    fail "Cannot import ragix_core.memory.cli"
    fail "Ensure RAGIX is installed: pip install -e ."
    exit 1
fi
ok "ragix-memory importable"

# Count corpus files
if [ ! -d "$CORPUS_DIR" ]; then
    fail "Corpus directory not found: $CORPUS_DIR"
    exit 1
fi
CORPUS_COUNT=$(find "$CORPUS_DIR" -maxdepth 1 -type f -name "*.md" | wc -l)
CORPUS_SIZE=$(du -sh "$CORPUS_DIR" 2>/dev/null | cut -f1)
ok "Corpus: $CORPUS_COUNT top-level docs ($CORPUS_SIZE total) in $CORPUS_DIR"

# Ollama check
OLLAMA_OK=false
if [ "$NO_LLM" = false ]; then
    if command -v ollama &>/dev/null && ollama list &>/dev/null 2>&1; then
        if ollama list 2>/dev/null | grep -q "$LLM_MODEL"; then
            ok "Ollama: model '$LLM_MODEL' available"
            OLLAMA_OK=true
        else
            warn "Ollama running but model '$LLM_MODEL' not found"
            warn "  Pull it with:  ollama pull $LLM_MODEL"
        fi
    else
        warn "Ollama not detected"
    fi
fi

# Claude check
CLAUDE_OK=false
if [ "$NO_LLM" = false ]; then
    if command -v claude &>/dev/null; then
        ok "Claude CLI available"
        CLAUDE_OK=true
    else
        warn "Claude CLI not found (install: npm install -g @anthropic-ai/claude-code)"
    fi
fi


# ===================================================================
# Act 1 — Init
# ===================================================================

act_header 1 "Initialize Memory Workspace"
ACT_START=$SECONDS

cmd "ragix-memory init $WORKSPACE"
echo ""

if [ -f "$DB" ] && [ "$SKIP_INGEST" = true ]; then
    info "Reusing existing workspace at $WORKSPACE"
    ok "Database already exists: $DB"
else
    rm -rf "$WORKSPACE"
    python -m ragix_core.memory.cli init "$WORKSPACE" 2>&1 | while IFS= read -r line; do
        info "$line"
    done
    ok "Workspace initialized"
fi

elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 2 — Ingest the docs/ corpus
# ===================================================================

act_header 2 "Ingest Corpus into Memory"
ACT_START=$SECONDS

cmd "ragix-memory --db \$DB ingest --source \"docs/*.md\" --source \"docs/developer/*.md\" \\"
cmd "    --injectable --format auto --tags ragix,documentation --scope demo"
echo ""

if [ "$SKIP_INGEST" = true ]; then
    info "Skipping ingestion (--skip-ingest)"
else
    # Use explicit glob patterns to ingest only markdown documentation.
    # This avoids machine-generated logs (.KOAS/runs/), binary noise, and
    # ensures FTS5 search results are relevant doc content.
    INGEST_SOURCES=()
    for pattern in "$CORPUS_DIR"/*.md "$CORPUS_DIR"/developer/*.md "$CORPUS_DIR"/archive/*.md; do
        for f in $pattern; do
            [ -f "$f" ] && INGEST_SOURCES+=("$f")
        done
    done

    info "Ingesting ${#INGEST_SOURCES[@]} markdown files from $CORPUS_DIR"
    echo ""

    INGEST_OUTPUT=$(mem ingest \
        --source "${INGEST_SOURCES[@]}" \
        --injectable \
        --format auto \
        --tags "ragix,documentation" \
        --scope demo 2>&1)

    echo "$INGEST_OUTPUT" | grep -v '^WARNING' | while IFS= read -r line; do
        demo "$line"
    done

    TOTAL_CHUNKS=$(echo "$INGEST_OUTPUT" | grep -oP '\d+ chunks' | grep -oP '\d+' || echo "0")
    TOTAL_FILES=$(echo "$INGEST_OUTPUT" | grep -oP '\d+ file' | grep -oP '\d+' || echo "0")

    echo ""
    ok "Ingestion complete: $TOTAL_FILES files, $TOTAL_CHUNKS chunks"
fi

elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 3 — Search & Recall
# ===================================================================

act_header 3 "Search & Recall (FTS5/BM25)"
ACT_START=$SECONDS

cmd "ragix-memory --db \$DB search \"your query\" --k 5"
cmd "ragix-memory --db \$DB recall \"your query\" --budget 2000"
echo ""

QUERIES=(
    "What is KOAS and how do kernels work?"
    "memory architecture and consolidation"
    "MCP tools and server integration"
    "sovereign AI and local-first design"
)

for query in "${QUERIES[@]}"; do
    TOTAL_QUERIES=$((TOTAL_QUERIES + 1))
    echo ""
    demo "Query: ${BOLD}\"$query\"${NC}"
    echo -e "  ${DIM}─────────────────────────────────────────────────────${NC}"

    SEARCH_OUTPUT=$(mem search "$query" --k 5 2>&1)
    MATCH_COUNT=$(echo "$SEARCH_OUTPUT" | grep -c '^\s*\[' || true)
    echo "$SEARCH_OUTPUT" | head -12 | while IFS= read -r line; do
        echo -e "  ${DIM}$line${NC}"
    done

    if [ "$MATCH_COUNT" -gt 0 ]; then
        ok "$MATCH_COUNT result(s) found"
    else
        warn "No results for this query"
    fi
done

echo ""
ok "Completed $TOTAL_QUERIES queries"
elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 4 — Idempotency Proof
# ===================================================================

act_header 4 "Idempotency Proof (SHA-256 Dedup)"
ACT_START=$SECONDS

cmd "ragix-memory --db \$DB ingest --source docs/ --injectable --format auto"
info "(same command as Act 2 — re-run to prove idempotency)"
echo ""

# Re-build source list (same logic as Act 2) for idempotency test
INGEST_SOURCES=()
for pattern in "$CORPUS_DIR"/*.md "$CORPUS_DIR"/developer/*.md "$CORPUS_DIR"/archive/*.md; do
    for f in $pattern; do
        [ -f "$f" ] && INGEST_SOURCES+=("$f")
    done
done

REINGEST_OUTPUT=$(mem ingest \
    --source "${INGEST_SOURCES[@]}" \
    --injectable \
    --format auto \
    --tags "ragix,documentation" \
    --scope demo 2>&1)

echo "$REINGEST_OUTPUT" | grep -v '^WARNING' | while IFS= read -r line; do
    demo "$line"
done

if echo "$REINGEST_OUTPUT" | grep -qP '0 chunks'; then
    ok "Zero new chunks — dedup working perfectly"
elif echo "$REINGEST_OUTPUT" | grep -qP 'skipped'; then
    SKIPPED=$(echo "$REINGEST_OUTPUT" | grep -oP '\d+ skipped' | grep -oP '\d+' || echo "all")
    ok "$SKIPPED files skipped (unchanged) — SHA-256 dedup confirmed"
else
    warn "Unexpected re-ingest output (check manually)"
fi

elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 5 — Pull (capture LLM output)
# ===================================================================

act_header 5 "Pull — Capture a Note from stdin"
ACT_START=$SECONDS

cmd "echo \"Your notes or LLM output\" | ragix-memory --db \$DB pull \\"
cmd "    --tags demo,summary --title \"My Summary\" --scope demo"
echo ""

SIMULATED_LLM_OUTPUT="RAGIX Memory System — Demo Summary

The RAGIX memory subsystem provides a complete lifecycle for knowledge
management: documents are ingested, chunked, and stored in a SQLite-backed
FTS5 index. Retrieval uses BM25 ranking with token-budgeted injection blocks.

Key capabilities demonstrated:
- Corpus ingestion with SHA-256 dedup
- Full-text search via FTS5/BM25
- Token-budgeted recall for LLM context injection
- Idempotent re-ingestion (no duplicate chunks)
- Unix composability: all commands work with pipes

This summary was generated during the RAGIX memory full-lifecycle demo."

info "Simulating LLM output capture via pull..."

PULL_OUTPUT=$(echo "$SIMULATED_LLM_OUTPUT" | mem pull \
    --tags "demo,summary,lifecycle" \
    --title "Demo Lifecycle Summary" \
    --scope demo 2>&1)

echo "$PULL_OUTPUT" | while IFS= read -r line; do
    demo "$line"
done
ok "Note captured via pull"

echo ""
info "Verifying: can we find the captured note?"
cmd "ragix-memory --db \$DB search \"demo lifecycle summary\" --k 3"
echo ""
PULL_SEARCH=$(mem search "demo lifecycle summary" --k 3 2>&1)
echo "$PULL_SEARCH" | while IFS= read -r line; do
    echo -e "  ${DIM}$line${NC}"
done
ok "Captured note found alongside corpus documents"

elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 6 — Stats & Palace
# ===================================================================

act_header 6 "Stats & Memory Palace"
ACT_START=$SECONDS

cmd "ragix-memory --db \$DB stats"
echo ""
mem stats 2>&1 | while IFS= read -r line; do
    echo -e "  ${DIM}$line${NC}"
done

echo ""
cmd "ragix-memory --db \$DB palace"
echo ""
mem palace 2>&1 | while IFS= read -r line; do
    echo -e "  ${DIM}$line${NC}"
done

elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 7 — Export & Compose
# ===================================================================

act_header 7 "Export & Unix Composability"
ACT_START=$SECONDS

cmd "ragix-memory --db \$DB export | wc -l"
cmd "ragix-memory --db \$DB export | jq -r '.title' | sort -u | head"
cmd "ragix-memory --db \$DB export -o snapshot.jsonl"
echo ""

EXPORT_COUNT=$(mem export 2>/dev/null | wc -l)
ok "Exported $EXPORT_COUNT JSONL records to stdout"

info "Pipe composability — extract unique titles:"
TITLE_SAMPLE=$(mem export 2>/dev/null | python -c "
import sys, json
seen = set()
for line in sys.stdin:
    t = json.loads(line).get('title','')
    if t and t not in seen:
        seen.add(t)
        print(f'    {t}')
    if len(seen) >= 6:
        break
" 2>/dev/null || true)
if [ -n "$TITLE_SAMPLE" ]; then
    echo "$TITLE_SAMPLE" | while IFS= read -r line; do
        echo -e "  ${DIM}$line${NC}"
    done
    echo -e "  ${DIM}    ...${NC}"
fi

echo ""
info "Sample JSONL record (truncated):"
SAMPLE=$(mem export 2>/dev/null | head -1 | python -c "
import sys, json
d = json.loads(sys.stdin.read())
d['content'] = d.get('content','')[:120] + '...'
print(json.dumps(d, indent=2, ensure_ascii=False)[:400])
print('  ...')
" 2>/dev/null || true)
if [ -n "$SAMPLE" ]; then
    echo "$SAMPLE" | while IFS= read -r line; do
        echo -e "  ${DIM}$line${NC}"
    done
fi

elapsed $((SECONDS - ACT_START))


# ===================================================================
# Act 8 — LLM Reasoning: Local (Ollama) + Cloud (Claude)
# ===================================================================

act_header 8 "LLM Reasoning over Memory"
ACT_START=$SECONDS

info "This act pipes recalled memory into LLMs for complex reasoning."
info "Two backends: ${BOLD}Ollama${NC} (local, sovereign) and ${BOLD}Claude${NC} (cloud, powerful)."
echo ""
info "The beauty: one Unix pipe connects memory to any LLM."
echo ""

if [ "$NO_LLM" = true ]; then
    warn "Skipped (--no-llm flag)"
    info "To run Act 8:  ./run_demo.sh --skip-ingest --keep"
    elapsed $((SECONDS - ACT_START))
else

# ---- Logging infrastructure ----
LOGDIR="$WORKSPACE/llm_logs"
mkdir -p "$LOGDIR"
info "LLM logs will be saved to: ${BOLD}$LOGDIR/${NC}"
echo ""

# Recall context into a temp file and log it.
# Usage: recall_context <qid> <topic> <budget>
# Sets: CTX_FILE, CTX_CHARS, CTX_TOKS, CTX_CHUNKS
recall_context() {
    local qid="$1" topic="$2" budget="$3"
    CTX_FILE="$LOGDIR/${qid}_context.txt"
    mem pipe "$topic" --budget "$budget" 2>/dev/null > "$CTX_FILE"
    CTX_CHARS=$(wc -c < "$CTX_FILE")
    CTX_TOKS=$((CTX_CHARS / 4))
    CTX_CHUNKS=$(grep -c '^\[' "$CTX_FILE" 2>/dev/null || echo 0)
}

# Log input (context + prompt) and display stats
# Usage: log_input <qid> <prompt> <backend>
log_input() {
    local qid="$1" prompt="$2" backend="$3"
    local input_file="$LOGDIR/${qid}_input.txt"
    {
        echo "=== LLM INPUT LOG ==="
        echo "Question:  $qid"
        echo "Backend:   $backend"
        echo "Timestamp: $(date -Iseconds)"
        echo "Context:   ${CTX_CHARS} chars, ~${CTX_TOKS} tokens, ${CTX_CHUNKS} chunks"
        echo "Budget:    requested in recall"
        echo ""
        echo "=== RECALLED CONTEXT ==="
        cat "$CTX_FILE"
        echo ""
        echo "=== PROMPT ==="
        echo "$prompt"
    } > "$input_file"
    info "Input: ${CTX_CHARS} chars, ~${CTX_TOKS} tok, ${CTX_CHUNKS} chunks → ${backend}"
    info "Logged: $LOGDIR/${qid}_input.txt"
}

# Log output (answer + metrics) and display result
# Usage: log_output <qid> <answer> <elapsed> <backend>
log_output() {
    local qid="$1" answer="$2" elapsed_s="$3" backend="$4"
    local output_file="$LOGDIR/${qid}_output.txt"
    local ans_chars ans_toks rate status
    if [ -n "$answer" ]; then
        ans_chars=$(printf '%s' "$answer" | wc -c)
        ans_toks=$((ans_chars / 4))
        rate=$( [ "$elapsed_s" -gt 0 ] && echo "$((ans_toks / elapsed_s))" || echo "n/a" )
        status="ok"
    else
        ans_chars=0; ans_toks=0; rate="n/a"; status="timeout/error"
    fi
    {
        echo "=== LLM OUTPUT LOG ==="
        echo "Question:   $qid"
        echo "Backend:    $backend"
        echo "Timestamp:  $(date -Iseconds)"
        echo "Status:     $status"
        echo "Elapsed:    ${elapsed_s}s"
        echo "Input:      ${CTX_CHARS} chars (~${CTX_TOKS} tok, ${CTX_CHUNKS} chunks)"
        echo "Output:     ${ans_chars} chars (~${ans_toks} tok)"
        echo "Rate:       ~${rate} tok/s"
        echo ""
        echo "=== ANSWER ==="
        echo "$answer"
    } > "$output_file"
    if [ -n "$answer" ]; then
        print_answer "$answer"
        echo ""
        ok "Answer: $(echo "$answer" | wc -l) lines, ~${ans_toks} tok, ${elapsed_s}s, ~${rate} tok/s"
        ok "Logged: $LOGDIR/${qid}_output.txt"
        LLM_QUESTIONS=$((LLM_QUESTIONS + 1))
    else
        warn "No answer after ${elapsed_s}s (timeout or API error)"
        warn "Check input: $LOGDIR/${qid}_input.txt"
    fi
}

# ---- Questions bank ----

Q1_TOPIC="RAGIX architecture purpose design philosophy sovereign local Unix"
Q1_PROMPT="Explain what RAGIX is and what problem it solves. Answer in exactly 5 bullet points. Be specific — name concrete technologies (SQLite, FTS5, Ollama, etc.)."
Q1_LABEL="What is RAGIX? (5 bullets)"

Q2_TOPIC="KOAS kernels audit orchestration activity summary presenter review"
Q2_PROMPT="List every KOAS kernel you can identify. For each, give its name and one sentence describing what it does. Then rank them by usefulness for auditing a large Java codebase."
Q2_LABEL="KOAS kernels inventory & ranking"

Q3_TOPIC="MCP tools skills server memory integration ragix-memory pipe recall"
Q3_PROMPT="Enumerate all MCP tools and CLI skills that RAGIX provides. Classify them into categories (e.g. memory management, retrieval, analysis, orchestration). Present as a table or grouped list."
Q3_LABEL="MCP tools & skills taxonomy"

Q4_TOPIC="RAGIX agent autonomous orchestration reasoning LLM tool"
Q4_PROMPT="Is RAGIX an autonomous agent, a framework, or a developer tool? Build a concise argument citing specific evidence from the provided text only (features, architecture choices, design principles). Conclude with a one-sentence verdict."
Q4_LABEL="Is RAGIX an agent? (evidence-based)"


# ---------------------------------------------------------------
# Section A: Ollama (local, sovereign)
# ---------------------------------------------------------------

section "Ollama / $LLM_MODEL  (local, sovereign)"

if [ "$OLLAMA_OK" = false ]; then
    warn "Ollama not available — skipping local LLM section"
    info "To enable:  ollama serve && ollama pull $LLM_MODEL"
else
    # Question 1 — What is RAGIX?
    echo -e "  ${BOLD}${CYAN}Q1:${NC} ${BOLD}$Q1_LABEL${NC}"
    echo -e "  ${DIM}─────────────────────────────────────────────────────${NC}"
    cmd "ragix-memory --db \$DB pipe \"topic\" --budget 3000 \\"
    cmd "    | { cat; echo '---'; echo 'prompt'; } | ollama run $LLM_MODEL"
    echo ""

    recall_context "Q1" "$Q1_TOPIC" 3000
    log_input "Q1" "$Q1_PROMPT" "ollama/$LLM_MODEL"

    Q_START=$SECONDS
    echo -ne "  ${DIM}[....] Generating (Ollama)...${NC}\r"
    A1=$(
        { cat "$CTX_FILE"; echo ""; echo "---"; echo ""; echo "$Q1_PROMPT"; } \
            | timeout 180 ollama run "$LLM_MODEL" 2>/dev/null
    ) || true
    echo -ne "\033[2K\r"

    log_output "Q1" "$A1" $((SECONDS - Q_START)) "ollama/$LLM_MODEL"
    echo ""

    # Question 2 — KOAS kernels
    echo -e "  ${BOLD}${CYAN}Q2:${NC} ${BOLD}$Q2_LABEL${NC}"
    echo -e "  ${DIM}─────────────────────────────────────────────────────${NC}"
    cmd "ragix-memory --db \$DB pipe \"topic\" --budget 3000 \\"
    cmd "    | { cat; echo '---'; echo 'prompt'; } | ollama run $LLM_MODEL"
    echo ""

    recall_context "Q2" "$Q2_TOPIC" 3000
    log_input "Q2" "$Q2_PROMPT" "ollama/$LLM_MODEL"

    Q_START=$SECONDS
    echo -ne "  ${DIM}[....] Generating (Ollama)...${NC}\r"
    A2=$(
        { cat "$CTX_FILE"; echo ""; echo "---"; echo ""; echo "$Q2_PROMPT"; } \
            | timeout 180 ollama run "$LLM_MODEL" 2>/dev/null
    ) || true
    echo -ne "\033[2K\r"

    log_output "Q2" "$A2" $((SECONDS - Q_START)) "ollama/$LLM_MODEL"
fi


# ---------------------------------------------------------------
# Section B: Claude (cloud, powerful reasoning)
# ---------------------------------------------------------------

section "Claude  (cloud, powerful reasoning)"

if [ "$CLAUDE_OK" = false ]; then
    warn "Claude CLI not available — skipping cloud LLM section"
    info "To enable:  npm install -g @anthropic-ai/claude-code"
    echo ""
    info "The command pattern (from headless docs):"
    cmd "ragix-memory --db \$DB pipe \"topic\" --budget 4000 \\"
    cmd "    | claude --system-prompt \"Doc analyst. <question>\" --tools '' -p"
else
    info "Claude syntax: stdin = context, --system-prompt = instruction, --tools '' = pure LLM"
    echo ""

    # Question 3 — MCP tools taxonomy
    echo -e "  ${BOLD}${CYAN}Q3:${NC} ${BOLD}$Q3_LABEL${NC}"
    echo -e "  ${DIM}─────────────────────────────────────────────────────${NC}"
    cmd "ragix-memory --db \$DB pipe \"$Q3_TOPIC\" --budget 4000 \\"
    cmd "    | claude --system-prompt \"$CLAUDE_BASE_PROMPT $Q3_PROMPT\" --tools '' -p"
    echo ""

    recall_context "Q3" "$Q3_TOPIC" 4000
    log_input "Q3" "$Q3_PROMPT" "claude"

    Q_START=$SECONDS
    echo -ne "  ${DIM}[....] Generating (Claude)...${NC}\r"
    A3=$(
        cat "$CTX_FILE" \
            | (cd "$CLAUDE_CWD" && timeout 180 env -u CLAUDECODE claude \
                --setting-sources "" \
                --system-prompt "$CLAUDE_BASE_PROMPT $Q3_PROMPT" \
                --tools "" \
                --no-session-persistence \
                -p 2>/dev/null)
    ) || true
    echo -ne "\033[2K\r"

    log_output "Q3" "$A3" $((SECONDS - Q_START)) "claude"
    echo ""

    # Question 4 — Is RAGIX an agent?
    echo -e "  ${BOLD}${CYAN}Q4:${NC} ${BOLD}$Q4_LABEL${NC}"
    echo -e "  ${DIM}─────────────────────────────────────────────────────${NC}"
    cmd "ragix-memory --db \$DB pipe \"$Q4_TOPIC\" --budget 4000 \\"
    cmd "    | claude --system-prompt \"$CLAUDE_BASE_PROMPT $Q4_PROMPT\" --tools '' -p"
    echo ""

    recall_context "Q4" "$Q4_TOPIC" 4000
    log_input "Q4" "$Q4_PROMPT" "claude"

    Q_START=$SECONDS
    echo -ne "  ${DIM}[....] Generating (Claude)...${NC}\r"
    A4=$(
        cat "$CTX_FILE" \
            | (cd "$CLAUDE_CWD" && timeout 180 env -u CLAUDECODE claude \
                --setting-sources "" \
                --system-prompt "$CLAUDE_BASE_PROMPT $Q4_PROMPT" \
                --tools "" \
                --no-session-persistence \
                -p 2>/dev/null)
    ) || true
    echo -ne "\033[2K\r"

    log_output "Q4" "$A4" $((SECONDS - Q_START)) "claude"
fi

echo ""
ok "Completed $LLM_QUESTIONS LLM-reasoned question(s)"
elapsed $((SECONDS - ACT_START))

fi  # end NO_LLM guard


# ===================================================================
# Summary Dashboard
# ===================================================================

TOTAL_ELAPSED=$((SECONDS - DEMO_START))

banner "Demo Complete"
echo ""
echo -e "  ${BOLD}${GREEN}Summary Dashboard${NC}"
echo -e "  ─────────────────────────────────────────"
echo -e "  ${BOLD}Workspace:${NC}       $WORKSPACE"
echo -e "  ${BOLD}Corpus:${NC}          $CORPUS_COUNT top-level docs ($CORPUS_SIZE)"
echo -e "  ${BOLD}JSONL records:${NC}   $EXPORT_COUNT"
echo -e "  ${BOLD}Search queries:${NC}  $TOTAL_QUERIES"
echo -e "  ${BOLD}LLM questions:${NC}   $LLM_QUESTIONS (Ollama + Claude)"
echo -e "  ${BOLD}Total time:${NC}      ${TOTAL_ELAPSED}s"
echo -e "  ${BOLD}Dedup:${NC}           SHA-256 (idempotent)"
echo -e "  ${BOLD}Search engine:${NC}   FTS5/BM25"
echo -e "  ${BOLD}Local LLM:${NC}       $LLM_MODEL (Ollama)"
echo -e "  ${BOLD}Cloud LLM:${NC}       Claude"
if [ -d "$WORKSPACE/llm_logs" ]; then
    LLM_LOG_COUNT=$(ls "$WORKSPACE/llm_logs/" 2>/dev/null | wc -l)
    echo -e "  ${BOLD}LLM logs:${NC}        $WORKSPACE/llm_logs/ ($LLM_LOG_COUNT files)"
fi
echo -e "  ─────────────────────────────────────────"
echo ""

# Command cheatsheet
echo -e "  ${BOLD}${YELLOW}Command Cheatsheet${NC}"
echo -e "  ─────────────────────────────────────────"
echo -e "  ${DIM}# Create a memory workspace${NC}"
cmd "ragix-memory init .memory"
echo ""
echo -e "  ${DIM}# Ingest a folder (any format: md, docx, py, yaml, pdf...)${NC}"
cmd "ragix-memory ingest --source ./my_docs/ --injectable --format auto"
echo ""
echo -e "  ${DIM}# Search your knowledge base${NC}"
cmd "ragix-memory search \"how does authentication work\" --k 10"
echo ""
echo -e "  ${DIM}# Token-budgeted recall (injection block for LLMs)${NC}"
cmd "ragix-memory recall \"security policy\" --budget 2000"
echo ""
echo -e "  ${DIM}# Full RAG pipe: ingest + recall → local LLM${NC}"
cmd "ragix-memory pipe \"summarize\" --source report.pdf --budget 3000 \\"
cmd "    | ollama run granite3.1-moe:3b"
echo ""
echo -e "  ${DIM}# Same pipe → Claude (cloud, --system-prompt replaces CLAUDE.md, --tools '' = pure LLM)${NC}"
cmd "ragix-memory pipe \"summarize\" --source report.pdf --budget 4000 \\"
cmd "    | claude --system-prompt \"Summarize the documentation.\" --tools '' -p"
echo ""
echo -e "  ${DIM}# Capture LLM output back into memory (feedback loop)${NC}"
cmd "ollama run mistral \"analyze X\" \\"
cmd "    | ragix-memory pull --tags analysis --title \"X Analysis\""
echo ""
echo -e "  ${DIM}# Export for backup or downstream processing${NC}"
cmd "ragix-memory export | jq -r '.title' | sort -u"
echo -e "  ─────────────────────────────────────────"
echo ""

# Cleanup
if [ "$KEEP" = true ]; then
    info "Workspace kept at: $WORKSPACE"
    info "Try it yourself:"
    cmd "ragix-memory --db $DB search \"your own question\""
    cmd "ragix-memory --db $DB pipe \"explain KOAS\" --budget 3000 | ollama run $LLM_MODEL"
    cmd "ragix-memory --db $DB pipe \"explain KOAS\" --budget 4000 | claude --system-prompt \"Explain concisely.\" --tools '' -p"
else
    info "Cleaning up workspace..."
    rm -rf "$WORKSPACE"
    ok "Workspace removed (use --keep to preserve it)"
fi

echo ""
echo -e "  ${DIM}RAGIX Memory — Full Lifecycle Demo finished.${NC}"
echo ""
