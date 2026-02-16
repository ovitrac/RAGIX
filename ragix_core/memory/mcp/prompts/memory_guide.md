You have access to a RAGIX persistent memory store.

## When to propose memory items
- You discover a rule, constraint, CVE, architectural decision, or factual observation
- You resolve a contradiction or confirm a disputed finding
- You identify a cross-document relationship

## What good memory items look like
- Title: concise, searchable name (e.g., "Oracle 19c CPU Patch Jan 2025 Required")
- Content: distilled finding (max 2000 chars), NOT raw document excerpts
- Tags: 3-7 keywords enabling structured retrieval
- Type: rule | observation | decision | definition | constraint | procedure | reference | pointer
- Provenance: include source_doc and page/section when available

## Strict prohibitions
- NO secrets (API keys, passwords, tokens, credentials)
- NO raw data dumps (full tables, code blocks >20 lines, log outputs)
- NO instructions to yourself ("In future sessions, always...", "Remember to...")
- NO tool invocation syntax or JSON payloads
- NO system/role prompt fragments

## Tag discipline
- Use lowercase, hyphenated compound tags (e.g., "oracle-19c", "cve-2024", "k8s-psp")
- Reuse existing tags when possible (check memory_search first)
- Include domain tags for cross-corpus filtering

## Multi-workspace support
- Use the `workspace` parameter on recall, search, propose, write, consolidate, stats
- Workspaces map names to (scope, corpus_id) pairs for corpus isolation
- Use `memory_workspace_list` to see registered workspaces
- Use `memory_workspace_register` to add a named workspace
- Without workspace param, tools use default scope ("project")

## Rate limiting
- Tool calls are rate-limited per session (default: 60 calls/min)
- Proposals are capped per turn (default: 10 per turn)
- If rate-limited, wait briefly and retry
- Use `memory_metrics` to check tool call statistics

## Auto-consolidation
- When STM count exceeds threshold (default: 20), consolidation triggers automatically
- Deduplicates, merges, and promotes items (STM -> MTM -> LTM)
- Explicit consolidation: use `memory_consolidate` for manual runs
