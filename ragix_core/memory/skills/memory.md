---
name: memory
description: RAGIX persistent structured memory — search, store, and recall knowledge
trigger: /memory
---

Interact with RAGIX persistent memory.  Memory persists across sessions
and provides governed, provenance-anchored knowledge storage.

## Subcommands

### /memory search <query> [--tier stm|mtm|ltm] [--domain <domain>] [--tags t1,t2] [--k <count>]
Search the memory store for items matching a query.
Uses FTS5 (BM25 ranked) with optional hybrid scoring.
Example: /memory search "oracle CVE" --domain oracle --k 5

### /memory recall <query> [--budget <tokens>]
Token-budgeted retrieval for context injection.
Returns items formatted for direct insertion into LLM context.
Default budget: 1500 tokens.
Example: /memory recall "migration security rules" --budget 2000

### /memory add "<title>" --tags tag1,tag2 --type rule|observation|decision
Store a finding into memory via policy evaluation (propose path).
Items are validated against governance (no secrets, no injection).
Stored in STM; promoted to MTM/LTM through consolidation.
Example: /memory add "Oracle 19c requires CPU patch Jan 2025" --tags oracle,CVE --type rule

### /memory stats [--scope <scope>]
Show memory store statistics: item counts, tier distribution, domains.

### /memory consolidate [--scope <scope>] [--promote]
Trigger deduplication, merge, and tier promotion cycle.

## How It Works
All subcommands delegate to the RAGIX Memory MCP server tools.
- search -> memory_search
- recall -> memory_recall
- add -> memory_propose
- stats -> memory_stats
- consolidate -> memory_consolidate
