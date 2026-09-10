# saqqara

**Document substrate kernels: typed trees with provenance on every node.**

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

A document is an object before it is a fragment. `saqqara` reads pdf, word-processing,
spreadsheet, presentation and markdown files into one typed tree, keeps an exact citation on every
node, and recognises the structures a reader sees — sections, tables, header bands, label tilings —
using ordered hard rules that abstain rather than guess.

It computes: no language model runs anywhere in it. Beside the reader, it keeps one SQLite store
of trees, chunks, objects, edges and vectors, queried on a lexical lane and, when an embedder is
configured, on a dense lane, every hit leading back to its nodes. The only model it can call is
that embedder, from indexing and search.

## Status

Gated. `SPEC.md` holds 153 falsifiable propositions and every one is carried by a test.

| layer | what it does | gate |
|---|---|---|
| `model.py` | nodes, kinds, locators, provenance, stable JSON | K1 |
| `adapters/` | one reader per format — pdf, docx, xlsx, pptx, md | K2 |
| `builder.py` | observations into a tree, with the accounting reconciled | K3.j |
| `analyzers/` | tables, header bands, islands, chains, sections, outline, headings shown by size or weight | K3.a–K3.k |
| `services.py` | title cascade, page policy, lookup | K3.g |
| `assets.py`, `render/`, object analyzers | figures, captions and drawn regions — what a document shows rather than says | K6 |
| `kernels/saqqara_run.py` | the reading kernel: the envelope, the roots, the abstention register | K4 |
| `store/`, `kernels/saqqara_index.py` | one SQLite file: trees, chunks, objects, edges and their vectors | K7 |
| `cli/`, `mcp/` | `saqqaractl` and the four MCP tools, answering identically | K7.16 |

## Using it

```python
from ragix_kernels.base import KernelInput
from ragix_kernels.saqqara.kernel import SaqqaraKernel

result = SaqqaraKernel().run(
    KernelInput(workspace="/tmp/work", config={"source": {"path": "corpus/"}})
)
print(result.summary)
```

Through MCP: `koas_saqqara_run(source, workspace, formats, promote_outline)`,
`koas_saqqara_status(workspace)`, `koas_saqqara_index(workspace, config)` and
`koas_saqqara_search(workspace, query, k, config)`. From the command line:
`python -m ragix_kernels.saqqara.cli.saqqaractl run|status|index|search`. The user reference is
`docs/KOAS_SAQQARA.md`, the developer reference `docs/KOAS_SAQQARA_DEV.md`.

Two roots come back, and the difference is the point. `merkle_root` covers the trees and is stable
across runs, so a claim about a document can name the structure it was read from. `source_root`
covers the bytes and deliberately excludes the paths: moving a document does not change what it
says, editing one does.

## The analyzer order

`tables` → `docx_tables` → `header_bands` → `islands` → `sections`.

`outline` is opt-in and runs **after** `sections`, never before. It promotes numbered lines to
headings, and run first it would hand the section channels its own conclusions to read back. It
adds nodes rather than rewriting them, so an observation is never silently replaced by an
inference, and a caller who wants promotions inside the ancestry runs `sections` again afterwards.

`chains` is not a pass. It computes the ancestry of one position on demand and stores nothing —
making it a pipeline stage would mean writing a derived value into the tree, which is the one thing
a view must not do.

## Design rules

These are not style preferences. Each is carried by a proposition in `SPEC.md` and by a test.

- **No node without provenance.** A citation is part of constructing a node, not something added
  afterwards. A projection that loses provenance is a defect.
- **Abstention is an object.** When the evidence does not decide, the output says so, carries a
  reason from a closed vocabulary, and travels onward as a fact. Abstention is never a default
  value and never a silent fallback.
- **Ordered hard rules, never a blended score.** Recognition applies declared rules in a declared
  order and exposes a decomposed trace. A single number standing in for several kinds of evidence
  hides exactly the disagreements worth seeing.
- **What is dropped is counted.** Every discard is counted and described. Nothing is silently
  repaired, truncated, or completed.
- **Facts before interpretation.** Adapters emit what the file says; analyzers decide what it
  means. A new fact can then be added without disturbing any rule that does not use it.
- **Every advanced method ships with its baseline.** A method that is never compared to the simple
  thing it replaces cannot be known to be an improvement. Ties are recorded, not omitted.

## Layout

```
saqqara/
├── SPEC.md        the specification: 153 propositions, gates K1-K4, K6 and K7
├── README.md      this file
├── __init__.py    the family's docstring
├── kernel.py      re-exports SaqqaraKernel for older callers
├── model.py       Node, KindRegistry, Locator, Provenance, Tree
├── builder.py     observations into a tree, one format plan per reader
├── services.py    title cascade, page policy, lookup
├── views.py       lazy projections: the structure signature
├── assets.py      picture bytes beside the tree, addressed by their hash
├── adapters/      one reader per format; each emits Mastaba raw facts
├── analyzers/     the pipeline (tables, grid_tables, header_bands, islands, sections), the opt-in
│                  outline, chains, and the library-only format_headings, caption_binding,
│                  vector_regions
├── render/        the renderer port, the default renderer, the opt-in one, the licence guard
├── kernels/       saqqara_run (stage 1) and saqqara_index (stage 2)  <- the only Kernel subclasses
├── store/         records, chunker, feed, embeddings, refinement, retrieval, configuration, sqlite
├── cli/           saqqaractl: run, status, index, search
└── mcp/           the four MCP tools
```

Exactly two modules define a `Kernel` subclass, `kernels/saqqara_run.py` and
`kernels/saqqara_index.py`, and the K0.3 gate pins that list. The registry discovers kernels by
walking this package, so a third would register as an independent kernel, and adding one amends
the pin.

## Tests

Tests live in `tests/saqqara/` and are named `test_k<N>…_<subject>.py`, one or more files per gate. The
numbering is this family's own and does not follow any other convention in this repository.

```bash
conda run -n ragix-env python -m pytest tests/saqqara -q
```

**Fixtures are generated by code** (`tests/saqqara/generators.py`). No document is committed to
this repository — not one. That is a design rule with a practical consequence: because every
character of every fixture comes from the generator, fixture content is proved by construction,
and there is no list of forbidden strings to maintain and no way for a stray document to arrive by
accident. A binary file appearing under `tests/saqqara/` is refused by the pre-commit guard.

## Guard

`tools/check_forbidden.py` refuses content that must not enter this repository. It carries a
`--selftest` that plants a violation and proves the guard fires — a guard never observed to fire is
not a guard.

```bash
python3 tools/check_forbidden.py --selftest
python3 tools/check_forbidden.py --staged
```

It runs in two places, and only one of them counts. The `.git/hooks/pre-commit` launcher is local
to a single clone: it covers nobody else's, and `--no-verify` steps over it. `.github/workflows/guard.yml`
runs the same checks where neither is possible.

## Installation

```bash
pip install -e ".[saqqara]"
```

## Licence

MIT, as the rest of this repository.
