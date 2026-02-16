---
name: memory-search
description: Search RAGIX memory (alias for /memory search)
trigger: /memory-search
---

Alias for `/memory search`. Searches the RAGIX memory store for items
matching a query using FTS5 (BM25 ranked) with optional hybrid scoring.

Usage: /memory-search <query> [--tier stm|mtm|ltm] [--domain <domain>] [--tags t1,t2] [--k <count>]

Delegates to: memory_search MCP tool.
