# ragix_kernels.harvest — the harvest family

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Knowledge read out of a `saqqara` document store, node by node, under a contract: deterministic
French readers for every critical value, a form that refuses what a model may not write, jobs that
call a local Ollama and keep every answer, abstracts by recurrence, node vectors, a derived graph,
composition under a fail-closed checker, and the registers and gold that audit them.

These are the tools as they were built for tender work, moved as they are: each module keeps its
name and logic; only its imports, its author line and a few names changed (see `MIGRATION.md`).
They are libraries and command-line drivers. **No KOAS kernel wraps them yet** and no MCP tool
exposes them: the registry discovers nothing here.

## What it never does

Every module that reads the store opens it read-only: `mode=ro&immutable=1`, or `immutable=1` (the
runner's bake-off and node jobs), or `mode=ro` (the gold machinery). What is derived
goes to files in an output directory or to the sidecar derived store (`derived.py`), keyed by the
store's own identifiers and the hash of the source a row was computed from. No module imports a PDF
library: documents are read through the store only. The Ollama host is `http://127.0.0.1:11434`
unless a driver is given another, and several drivers refuse any host that is not loopback.

## The modules

| module | what it is | run as |
|---|---|---|
| `fr.dates`, `fr.grammars`, `fr.cut` | the French readers: dates and clock times; amounts, percentages, durations, references, quantities, periods; the values a text layer cut or joined | library |
| `form` | the harvest form (`harvest-form/0.8`): validation, the refusals, `substitute`, `marker`, the grain rules, the extractive floor | library |
| `runner` | the bake-off, the node roll-ups and pass 2 at window grain | `python -m ragix_kernels.harvest.runner {bakeoff,nodes,window} …` |
| `pass1`, `core` | abstracts by recurrence: the ladder 150/350/800 and rules R1–R4, windows to documents to families to the whole | `python -m ragix_kernels.harvest.core …` |
| `pieces` | the piece map's parser | library |
| `families` | the family rung between the documents and the whole, sized | `python -m ragix_kernels.harvest.families …` |
| `derived` | the sidecar derived store | library |
| `embed_abstracts` | one vector per abstracted node | `python -m ragix_kernels.harvest.embed_abstracts …` |
| `map_test`, `descent` | the lexical map test, and the descent question → document → window → leaf | `python -m ragix_kernels.harvest.descent …` |
| `build_edges` | derived edges materialised from what already exists | `python -m ragix_kernels.harvest.build_edges …` |
| `ask` | descent and register lanes, composition under a JSON schema, the fail-closed checker | `python -m ragix_kernels.harvest.ask …` |
| `rescore`, `rescore_repeats` | recorded runs re-scored offline under the rules as they stand | `python -m ragix_kernels.harvest.rescore …` |
| `check_brief`, `scan_cut` | the checker of a composed brief (nine checks), and the cut-value scan | `python -m ragix_kernels.harvest.check_brief …` |
| `registers.*` | `family`, `template`, `commitments`, `clauses`, `traps`, `render_fr` | `python -m ragix_kernels.harvest.registers.family …` |
| `gold.*` | `read`, `agreement`, `select_pass2`, `coord_check` | `python -m ragix_kernels.harvest.gold.read …` |
| `provenance` | the T0 manifest of a corpus by content hash | `python -m ragix_kernels.harvest.provenance` |
| `models/` | the Granite 4.2 Modelfiles and the server monitors (not Python) | see `models/README.md` |

The chained event journal the drivers' runs are recorded in lives beside the family, in
`ragix_kernels.shared.journal`.

    from ragix_kernels.harvest.fr.grammars import read_values
    from ragix_kernels.harvest.form import validate, substitute, grain_rules
    from ragix_kernels.harvest.derived import DerivedStore, KnowledgeRow

## Environment

| variable | read by | meaning |
|---|---|---|
| `HARVEST_CONSULTATION` | `core`, `families`, `check_brief`, `registers.family`, `.clauses`, `.template`, `.render_fr` | the consultation's reference as its file names and headers print it |
| `HARVEST_PASS1_DIR` | `rescore` | where the recorded pass-1 runs live |
| `HARVEST_LAB` | `gold.*` | the root the gold's `demoE2E/...` paths hang from |
| `HARVEST_DCE_ZIP`, `HARVEST_DCE_DIR`, `HARVEST_RECORD_ZIP`, `HARVEST_PROVENANCE_DIR` | `provenance` | the archive, its unpacked copy, the archive of record, the output directory |
| `KOAS_JOURNAL_ROOT` | `shared.journal` | the directory the journal is written under |

## What is gated and what is not

Gated by `tests/harvest/`: the readers version by version, the form's refusals and grains, the
runner's three jobs against a fake server on synthetic stores, the recurrence's rules, the derived
store's schema and append-only law, the embedder's determinism checks, the descent's arithmetic and
refusals, and an import of every module. The tests that need the lab's reference store skip without
it (`HARVEST_REFERENCE_STORE`). Not gated here: the registers, the gold machinery, `ask`, `rescore`,
`check_brief` and `scan_cut` beyond their import — their own tests were measurements on the lab's
corpus and stayed with it.
