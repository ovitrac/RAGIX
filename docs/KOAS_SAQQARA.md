# KOAS Saqqara — Document Substrate: Typed Trees with Provenance

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

**Version:** 1.0 (2026-08-30)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [CLI Reference (saqqaractl)](#4-cli-reference-saqqaractl)
5. [MCP Tools](#5-mcp-tools)
6. [Design Rules](#6-design-rules)
7. [Gates and Specification](#7-gates-and-specification)
8. [Licence Discipline](#8-licence-discipline)

---

## 1. Overview

### What KOAS Saqqara Does

Reads pdf, word-processing, spreadsheet, presentation and markdown files into **one typed tree
each**, keeps an exact citation on every node, and recognises the structures a reader actually
sees: sections, tables, header bands, label tilings, figures and their captions.

It computes. There is no model inside it, no retrieval, no store — those live elsewhere in this
repository and none of them is a dependency of this package.

### Key Differentiators

- **A document is an object before it is a fragment.** The tree is built from what the file
  says, then interpreted; it is not a window over a byte stream.
- **Every node carries its citation.** Source document, format, sheet/page/cell, extraction
  kernel, confidence. A projection that loses provenance is a defect, not a simplification.
- **Abstention is an object.** When the evidence does not decide, the output says so with a
  reason from a closed vocabulary, and that reason travels onward as a fact.
- **Two roots, and the difference is the point.** `merkle_root` covers the trees and is stable
  across runs; `source_root` covers the bytes and excludes the paths.

### Use Cases

- Turning a heterogeneous document set into citable structure before any retrieval exists.
- Recovering the questions and answer regions of forms and questionnaires.
- Auditing what a reader can and cannot claim about a corpus, with the refusals counted.

---

## 2. Architecture

### Reader and analyzer, deliberately separated

```
adapters/          read the file          -> observations (origin=read, confidence 1.0)
builder.py         place them in a tree   -> nodes, with every refusal counted
analyzers/         interpret the tree     -> NEW nodes (origin=inferred, confidence < 1.0)
services.py        title cascade, lookup
views.py           lazy projections
kernels/           the KOAS kernel facade
```

An adapter emits what the file **says**. An analyzer decides what it **means**, and does so by
adding nodes rather than rewriting them — so an observation is never silently replaced by an
inference.

### Analyzer order

`tables` → `docx_tables` → `header_bands` → `islands` → `sections`.

`outline` is opt-in and runs **after** `sections`, never before: run first it would hand the
section channels its own conclusions to read back.

`chains` is not a pass. It computes the ancestry of one position on demand and stores nothing —
making it a pipeline stage would mean writing a derived value into the tree.

---

## 3. Quick Start

### Prerequisites

Python 3.12. No model server, no vector store, no network.

### Installation

```bash
pip install -e ".[saqqara]"
```

**What a plain install can do.** `.[saqqara]` gives you the readers, the analyzers,
the store and **both retrieval lanes** — lexical over FTS5 and dense over the
vectors in the database. `DocumentStore.search` is part of the protocol, so the
extra declares what the code imports rather than leaving a method that raises on a
clean install.

What it does not give you is a way to *produce* vectors. An embedder is a separate
choice, and each comes through its own extra:

| you want | install | then set |
|---|---|---|
| lexical only (the default) | `.[saqqara]` | `embedder.provider: none` |
| local sentence-transformers | `.[saqqara,retrieval]` | `embedder.provider: sentence-transformers` |
| a local Ollama server | `.[saqqara]` | `embedder.provider: ollama` |
| FAISS instead of numpy | `.[saqqara,retrieval]` | `index.backend: faiss` |

With `provider: none` the store is lexical-only and says so — `dense: disabled (no
embedder)`. It never writes a zero vector to make the column look populated.

### Basic Workflow

```bash
# read a directory into typed trees
python -m ragix_kernels.saqqara.cli.saqqaractl run ./documents -w ./work

# read back what that run found, without re-reading the documents
python -m ragix_kernels.saqqara.cli.saqqaractl status ./work
```

From Python:

```python
from ragix_kernels.base import KernelInput
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel

result = SaqqaraKernel().run(
    KernelInput(workspace="./work", config={"source": {"path": "./documents"}})
)
print(result.summary)
```

---

## 4. CLI Reference (saqqaractl)

### `run` — read a file or directory into typed trees

```bash
python -m ragix_kernels.saqqara.cli.saqqaractl run SOURCE [options]
```

| option | meaning |
|---|---|
| `-w, --workspace` | where to write the result (default: beside the source; created if absent) |
| `--formats` | comma-separated extensions to narrow the scan, e.g. `.docx,.xlsx` |
| `--promote-outline` | also run the opt-in typed-outline pass, which **adds** inferred headings; off by default because a promotion is an inference |
| `--json` | emit the raw result as JSON |
| `-v, --verbose` | list every abstention rather than the first ten |

Reports what was read, what was refused and why, what each analyzer abstained on, any counted
drops, and both roots. A report key the CLI does not recognise is printed raw rather than
skipped — a reporter that quietly drops an unknown key is how a new count becomes invisible.

### `status` — read back a previous run

```bash
python -m ragix_kernels.saqqara.cli.saqqaractl status WORKSPACE [--json] [-v]
```

Reads `WORKSPACE/stage1/saqqara.json`. A workspace with no result is an error, not an empty
answer: the two must not look alike.

---

## 5. MCP Tools

Registered by `ragix_kernels.saqqara.mcp.tools.register_saqqara_tools(server)`.

### `koas_saqqara_run`

```
koas_saqqara_run(source, workspace="", formats="", promote_outline=False)
```

Returns `{success, summary, output_file, documents, merkle_root, source_root, report, errors}`.
`documents` carries path, format, sha256 and node count per document.

### `koas_saqqara_status`

```
koas_saqqara_status(workspace)
```

Returns `{documents, merkle_root, source_root, report, abstentions}` — enough to check a
citation without re-reading the corpus.

The CLI and these tools answer identically by construction: same configuration keys, same stored
result, same roots. A surface that answered differently would be a second implementation of the
envelope, and the first divergence would surface as a citation nobody could reproduce.

---

## 6. Design Rules

Each is carried by a proposition in `SPEC.md` and by a test, not by convention:

- **No node without provenance.** A citation is part of constructing a node.
- **Abstention is an object**, never a default value and never a silent fallback.
- **Ordered hard rules, never a blended score.** Recognition applies declared rules in a
  declared order and exposes a decomposed trace.
- **What is dropped is counted.** Every classification ends in a branch that counts what it did
  not recognise, and the vocabulary test reads the fixture that produces that reason.
- **Facts before interpretation.** A new fact can be added without disturbing any rule that does
  not use it.
- **Every advanced method ships with its baseline.** Ties are recorded, not omitted.

---

## 7. Gates and Specification

`ragix_kernels/saqqara/SPEC.md` holds **144 falsifiable propositions**, each with the fixture
that exercises it and what would falsify it.

| gate | layer | propositions |
|---|---|---:|
| K1 | `model.py` — nodes, kinds, locators, provenance, stable JSON | 10 |
| K2 | `adapters/` — one reader per format | 25 |
| K3 | `analyzers/` + `builder.py` + `services.py` | 72 |
| K4 | `kernels/` — the envelope and the roots | 2 |
| K6 | objects — figures, captions, drawn regions | 19 |
| K7 | the store — one SQLite file, embeddings beside the chunks | 16 |
| | **total** | **144** |

K0 is the gate that lets the others mean something. It proves the specification parses and is
self-consistent, that specification and fixtures agree **in both directions**, that the package
defines exactly the kernels it declares, that nothing trips the repository guard — and that the
prose describing all of this states the same counts the tests pin.

**Fixtures are generated by code** (`tests/saqqara/generators.py`). No document is committed to
this repository, so fixture content is proved by construction rather than by review.

```bash
python3 -m pytest tests/saqqara -q
```

---

## 8. Licence Discipline

The package default is permissive: `pip install -e ".[saqqara]"` pulls a pdf renderer under
Apache-2.0 and nothing under a copyleft licence.

`pymupdf` is AGPL-3.0. It is reachable only through the opt-in `saqqara-mupdf` extra, which is
**not** referenced by `all`. The quarantine is enforced rather than documented, by
`ragix_kernels/saqqara/render/guard.py`:

- `scan_sources()` refuses an AGPL import anywhere in the package except the one module that
  exists to hold it;
- `loaded_agpl_modules()` proves at runtime that a default install never imported it.

Both are asserted in `tests/saqqara/test_k6_regions.py`.
