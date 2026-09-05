# saqqara

**Document substrate kernels: typed trees with provenance on every node.**

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

A document is an object before it is a fragment. `saqqara` reads pdf, word-processing,
spreadsheet, presentation and markdown files into one typed tree, keeps an exact citation on every
node, and recognises the structures a reader sees — sections, tables, header bands, label tilings —
using ordered hard rules that abstain rather than guess.

It computes. There is no model inside it, no retrieval, no store: those live elsewhere in this
repository and none of them is a dependency here.

## Status

Complete and gated. `SPEC.md` holds 148 falsifiable propositions and every one is carried by a test.

| layer | what it does | gate |
|---|---|---|
| `model.py` | nodes, kinds, locators, provenance, stable JSON | K1 |
| `adapters/` | one reader per format — pdf, docx, xlsx, pptx, md | K2 |
| `builder.py` | observations into a tree, with the accounting reconciled | K3.j |
| `analyzers/` | tables, header bands, islands, chains, sections, outline | K3.a–K3.i |
| `services.py` | title cascade, page policy, lookup | K3.g |
| `assets.py`, `render/`, object analyzers | figures, captions and drawn regions — what a document shows rather than says | K6 |
| `kernel.py` | the envelope, the roots, the MCP surface | K4 |
| `store/`, `kernels/saqqara_index.py` | one SQLite file: trees, chunks, objects, edges and their vectors | K7 |

## Using it

```python
from ragix_kernels.base import KernelInput
from ragix_kernels.saqqara.kernel import SaqqaraKernel

result = SaqqaraKernel().run(
    KernelInput(workspace="/tmp/work", config={"source": {"path": "corpus/"}})
)
print(result.summary)
```

Through MCP: `koas_saqqara_run(source, workspace, formats, promote_outline)` and
`koas_saqqara_status(workspace)`.

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
├── SPEC.md        the specification: 148 propositions, gates K1-K4, K6 and K7
├── kernel.py      the single KOAS kernel facade  <- only Kernel subclass here
├── model.py       Node, KindRegistry, Locator, Provenance, Tree
├── adapters/      one reader per format; each emits Mastaba raw facts
├── analyzers/     tables, header_bands, islands, sections
└── views.py       lazy projections: blocks, anchors, chains, signature
```

Exactly one module exposes a `Kernel` subclass. The registry discovers kernels by walking this
package, so a second one would register as an independent kernel.

## Tests

Tests live in `tests/saqqara/` and are named `test_k<N>_<subject>.py`, one file per gate. The
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
