---
name: memory-add
description: Store a finding into RAGIX memory (alias for /memory add)
trigger: /memory-add
---

Alias for `/memory add`. Stores a finding into memory via policy evaluation.
Items are validated against governance (no secrets, no injection).

Usage: /memory-add "<title>" --tags tag1,tag2 --type rule|observation|decision

Delegates to: memory_propose MCP tool.
