---
name: memory-recall
description: Token-budgeted memory recall (alias for /memory recall)
trigger: /memory-recall
---

Alias for `/memory recall`. Token-budgeted retrieval for context injection.
Returns items formatted for direct insertion into LLM context.

Usage: /memory-recall <query> [--budget <tokens>]

Delegates to: memory_recall MCP tool.
