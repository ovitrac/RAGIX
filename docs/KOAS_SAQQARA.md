# KOAS Saqqara — Document Substrate: Typed Trees with Provenance, and the Store Built on Them

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

**Version:** 2.0 (2026-09-10) — describes RAGIX 0.74.0 at commit `75b0fe1`

**Audience:** users. Modules, schema, records, refusal code paths and extension points are in the
developer reference, [KOAS_SAQQARA_DEV.md](KOAS_SAQQARA_DEV.md).

---

## Table of Contents

1. [What saqqara is, and what it is not](#1-what-saqqara-is-and-what-it-is-not)
2. [Install and environment](#2-install-and-environment)
3. [The pipeline as a user sees it](#3-the-pipeline-as-a-user-sees-it)
4. [The command line: saqqaractl](#4-the-command-line-saqqaractl)
5. [The MCP tools](#5-the-mcp-tools)
6. [From Python, and through KOAS](#6-from-python-and-through-koas)
7. [The store on disk](#7-the-store-on-disk)
8. [Querying](#8-querying)
9. [Refusals, abstentions and the register](#9-refusals-abstentions-and-the-register)
10. [Sovereignty](#10-sovereignty)
11. [Limits and known behaviours](#11-limits-and-known-behaviours)
12. [Configuration reference](#12-configuration-reference)
13. [Gates and specification](#13-gates-and-specification)
14. [Licence discipline](#14-licence-discipline)

---

## 1. What saqqara is, and what it is not

saqqara reads pdf, word-processing, spreadsheet, presentation and markdown files into **one typed
tree per document**, keeps an exact citation on every node, and recognises structure by ordered
rules that abstain rather than guess: blocks and tables, header bands, label tilings, and the section
names each format declares. What it could not decide is listed, with the reason. It can then cut
those trees into chunks
and keep them in **one SQLite file** that answers queries on a lexical lane and, when an embedder
is configured, on a dense lane. Every hit leads back to the nodes it was cut from.

It is two KOAS kernels:

| kernel | stage | requires | provides | what it does |
|---|---|---|---|---|
| `saqqara` | 1 | nothing | `document_tree`, `traces`, `merkle_root` | reads files into trees and runs the structure analyzers |
| `saqqara_index` | 2 | `document_tree` | `document_store` | chunks the trees into the store and embeds what is missing |

It has four surfaces over those kernels: the `saqqaractl` command line (`run`, `status`, `index`,
`search`), four MCP tools, the Python API, and the KOAS orchestrator, which finds both kernels in
its registry like any other.

**What it is not.**

- **No language model runs anywhere in it.** Reading, analysing, chunking, storing and retrieving
  are computation. The only model it can call is an embedder you configure, and only from `index`
  and `search`.
- **It does not answer questions.** `search` returns ranked chunks and their citations. It writes no
  text.
- **It does not invent headings.** A heading node exists only where a reader read one, or where the
  opt-in `outline` pass promoted a numbered line. A document without either has no heading path
  (§3.4, §11).
- **It does not read scanned pages.** A page with no text layer is reported and listed in the
  abstention register; nothing is recognised on it. OCR is declared out of scope in `SPEC.md`.
- **It has one store provider, `sqlite`.** Authority, freshness and quality boosts are not
  implemented: the `boosts` field of a hit exists and is always empty.
- **It has no query filters** (§8.5).

---

## 2. Install and environment

The repository declares `requires-python >= 3.10`. Continuous integration runs the saqqara gates on
Python 3.12, and the baseline in §13 was measured on Python 3.12.12.

```bash
pip install -e ".[saqqara]"
```

| extra | installs | needed for |
|---|---|---|
| (base) | `requests`, `pyyaml` | the configuration loader, and the HTTP transport of the `ollama` embedder |
| `saqqara` | `ragix[docs]` (`python-docx`, `python-pptx`, `openpyxl`, `odfpy`), `pypdf`, `pypdfium2`, `numpy` | the readers, the analyzers, the store, both retrieval lanes, the default renderer |
| `retrieval` | `sentence-transformers`, `numpy` | the `sentence-transformers` embedder |
| `saqqara-mupdf` | `pymupdf` (AGPL-3.0) | the opt-in alternative renderer (§14); saqqara never needs it |
| `mcp` | `mcp[cli]` | the MCP server that registers the four tools |
| `dev` | `pytest`, `black`, `ruff` | the gates and the demo |

`odfpy` arrives with `ragix[docs]`, but no saqqara reader claims an OpenDocument extension.

**Choosing an embedder.** An embedder is a separate choice from the install:

| you want | install | set in your `saqqara.yaml` |
|---|---|---|
| lexical only (the default) | `.[saqqara]` | `embedder.provider: none` |
| a local sentence-transformers model | `.[saqqara,retrieval]` | `embedder.provider: sentence-transformers`, optionally `embedder.model` |
| a local Ollama server | `.[saqqara]` | `embedder.provider: ollama`, `embedder.model`, and `embedder.base_url` if the server is not the default one |
| FAISS rather than numpy for the dense index | `.[saqqara]`, plus a FAISS package that no extra of this repository declares | `index.backend: faiss` |

With `provider: none` the store is lexical-only and says so: `dense: disabled (no embedder)`. No
zero vector is ever written to make the column look populated.

**The demo.** Four scripts build five documents with the same generators the gates use, read them,
index them and query them. No network, no model download, no document committed.

```bash
pip install -e ".[saqqara,dev]"
bash examples/saqqara/run_demo.sh ./demo-work     # or with no argument: a temporary directory
```

| script | what it shows |
|---|---|
| `examples/saqqara/01_read_tree.py` | five formats read into trees; one node with its citation chain; the two roots |
| `examples/saqqara/02_index_store.py` | the trees chunked into one SQLite file, with `embedder.provider: none` |
| `examples/saqqara/03_search_with_trace.py` | both lanes when `sentence-transformers` is importable, the lexical lane alone otherwise, every hit walked back to its nodes |
| `examples/saqqara/04_lexical_only.py` | no vector written without an embedder, and a configuration typo refused with its path |

The demo configuration is `examples/saqqara/saqqara.yaml`: it sets `store.path: saqqara.db`,
`store.corpus: demo`, `embedder.provider: none` and `retrieval.top_k: 5`. The examples in §4 run
against the workspace the demo leaves behind.

---

## 3. The pipeline as a user sees it

```
files ──read──▶ observations ──build──▶ tree ──analyze──▶ tree + traces           saqqara (stage 1)
                                                                │
                                                     stage1/saqqara.json
                                                                │
tree ──chunk──▶ level-0 units + level-1 roll-ups ──store──▶ one SQLite file       saqqara_index (stage 2)
                                                                │
                                                        embed what is missing
                                                                │
                                   refine (opt-in): split oversized or refused texts into parts
```

### 3.1 Reading

A source is a file or a directory. A directory is walked recursively and **every file found is
offered to the readers**, not only the extensions saqqara knows. A file no reader claims is refused
as `unsupported-format`, and a file its reader fails on is refused as `unreadable-file`. Both are
listed with their path in `report.refusals`. The `formats` option narrows the walk **before** any
reading, so a file it excludes appears nowhere in the report.

| extension | reader | reader version |
|---|---|---|
| `.pdf` | `pdf` | 0.8.0 |
| `.docx` | `docx` | 0.7.0 |
| `.xlsx`, `.xlsm` | `xlsx` | 0.6.0 |
| `.pptx` | `pptx` | 0.5.0 |
| `.md`, `.markdown` | `md` | 0.2.0 |

Files are compared by the sha256 of their bytes. A second byte-identical file is listed under
`report.duplicates` and is not read again: two copies of one document are one document.

### 3.2 The tree

A reader emits **observations**: what the file says, never what it means. The builder places them
in a tree. Every node carries these fields:

| field | meaning |
|---|---|
| `kind` | one registered kind (below); an unregistered kind cannot be constructed |
| `text` | the text the node carries, if any |
| `level` | a heading or outline level, where one exists |
| `span` | the extent the node covers, in its source's own terms |
| `facts` | what the reader observed, uninterpreted (a data type, a boldness, a font size) |
| `children` | the nodes below it, in document order |
| `provenance` | source path, source format, a chain of locators broadest first, the producing kernel and its version |
| `origin` | `read` (confidence exactly 1.0) or `inferred` (confidence below 1.0) |
| `confidence` | 1.0 for what was read; lower for an inference, never 1.0 |

**Kinds.** The standard kinds are `document`, `section`, `heading`, `paragraph`, `list`,
`list_item`, `table`, `figure` and `caption`. The builder registers `cell`, `marker`, `note`,
`shape`, `page`, `slide`, `sheet` and `vector_region`; the analyzers register `block`.

**What each format becomes**, as the builder's format plans declare it:

| format | containers | nodes | attached rather than made into nodes |
|---|---|---|---|
| xlsx | a sheet becomes a `section` | `cell`, `figure` | a border, as a fact of its cell |
| docx | a table becomes a `table` | `paragraph`, `marker`, `cell`, `figure` | — |
| pdf | a page becomes a `page` | a text placement becomes a `paragraph`; a declared outline entry becomes a `heading`; `figure` | a drawing, as a fact of its page |
| pptx | a slide becomes a `slide`; a table becomes a `table` | `shape`, a speaker note becomes a `note`, `cell`, `figure` | — |
| md | — | `heading`, `paragraph` | front matter, as the tree's metadata |

**Locators.** A citation is only meaningful in the coordinate system of the thing cited, so each
format has its own locator:

| format | locator fields |
|---|---|
| document | none: the coordinate a tree root cites |
| pdf | `page`, `bbox`, `xobject` |
| xlsx | `sheet`, `sheet_index`, `cell`, `row`, `col`, `merged_range`, `anchor` |
| docx | `flow` (`body`, a page header, a nested table), `paragraph`, `run`, `relationship`, `table_index`, `row`, `col` |
| pptx | `slide` (one-based), `shape`, `notes`, `shape_id`, `row`, `col` |
| md | `line` (one-based) |

**Two roots come back from every run.** `merkle_root` covers the trees and is stable across runs of
the same input, so a claim about a document can name the structure it was read from. `source_root`
covers the bytes of the files read and deliberately excludes their paths: moving a document does
not change it, editing one does.

### 3.3 The analyzers a run applies

A run applies five analyzers in this order, and each keeps its trace under its own name in
`documents[].traces`:

| analyzer | what it adds |
|---|---|
| `tables` | on spreadsheets, the blocks of each sheet, typed table, text or list by ordered rules, as new nodes |
| `grid_tables` | on word-processing and presentation tables, a type: `data-form`, `layout` or `table_uncertain` |
| `header_bands` | the header band, label columns, title and section rows of each table, or an abstention |
| `islands` | a flag on a block that holds several disconnected value regions; the block is never re-segmented |
| `sections` | section names collected per channel, a gauntlet of refusals, and a `section_ancestry` fact on each node |

**`outline` is opt-in** (`--promote-outline` on the command line, `promote_outline` elsewhere). It
runs after `sections`, and it **adds** an inferred `heading` node beneath each numbered paragraph it
promotes, never rewriting the paragraph.

**Three analyzers exist only as a library.** `format_headings` (headings shown by size or weight),
`caption_binding` (which line captions which figure) and `vector_regions` (when drawn ink amounts to
a figure) are not in the run's pipeline, and no surface of the kernel calls them. The readers also
extract pictures only when they are handed an asset store, and the kernel hands them none. A
`saqqaractl run` therefore produces no `figure`, `caption` or `vector_region` node.

**Options.** An analyzer that declares options takes them at construction, and records the options
it ran with, defaults included, in its trace. At this version only `header_bands` declares any:
`max_header_rows` (default 3) and `max_label_cols` (default 3). They are passed as
`analyzers: {header_bands: {...}}` in the kernel's configuration, which the command line and the
MCP tools do not expose. An unknown analyzer name, or an option an analyzer does not declare, is
refused before anything is read.

### 3.4 Chunks

`saqqara_index` cuts each tree along its own structure. It never reads a source file again.

**Level 0.** One chunk per node of a chunkable kind that carries text, in document order. The
chunkable kinds are `paragraph`, `heading`, `list`, `list_item`, `table`, `caption`, `block`,
`slide`, `shape`, `note`, `cell` and `marker`. A `table` chunk holds its cells' text joined by
` | `, and its cells are then not chunked a second time. `section`, `page` and `figure` are never
chunked, each for a declared reason.

**Level 1.** One roll-up per section: the texts of its level-0 chunks joined by a newline. Each
level-0 chunk names its roll-up in `parent_id`. A section opens at a `section` or `heading` node,
and text before the first one belongs to an unnamed section. **A document with no `section` or
`heading` node therefore gets exactly one roll-up, holding its whole text, and an empty section
path.** On the demo corpus:

| format | level-0 chunks carry a section path | roll-ups | chunks carry pages |
|---|---|---|---|
| md | yes | one per heading | no |
| xlsx | yes | one per sheet | no |
| pdf without a declared outline | no | one for the document | yes |
| docx | no | one for the document | no |
| pptx | no | one for the deck | no |

A word-processing document's heading styles feed the `sections` analyzer, which writes
`section_ancestry` facts. The chunker reads section and heading nodes, not those facts.

**Oversized units.** A level-0 text longer than `chunker.unit_max_chars` (1200) is kept whole and
flagged `meta.oversize`. Only a single node longer than `chunker.window_fallback_chars` (4000) is
cut: into windows of that many characters overlapping by 200, each its own level-0 chunk naming the
same node and flagged `meta.fallback: window`.

**Identity.** A chunk's id digests its document, level, node addresses and text. Indexing an
unchanged tree again writes nothing. A chunk the tree no longer produces is deleted with its
vectors, and the deletion is counted.

### 3.5 Objects and edges

Nodes of kind `table`, `figure` and `vector_region` are also kept as **objects** beside the tree,
with their page, box, caption, asset reference and cells. A store built through `index` holds
tables only, since the run produces no figure or region (§3.3).

**Edges** have a closed vocabulary: `binds`, `refers_to`, `continues`, `supersedes`. The feed writes
only `binds`, copied from a caption's `caption_of` fact, and only the library-only caption analyzer
writes that fact. A store built through `index` therefore holds no edge. `refers_to`, `continues`
and `supersedes` are declared, and nothing writes them.

### 3.6 Embeddings

With an embedder configured, every chunk of both levels is embedded unless this model already holds
a vector for it. Vectors are keyed by chunk and model, so two models coexist. The model name
recorded is `embedder.model`, or the provider's name when that is empty.

A text the embedder rejects is **a refusal, not an error**: it is recorded with its reason and
signals, the other texts are embedded, and the run reports embedded, already present and refused as
three numbers. The `ollama` backend never lets the server truncate a text, so an oversized text is a
refusal on every server rather than a silently shortened vector. A run whose lane ends with
refusals and **no vector at all** fails, because that is a missing lane, not a lane with gaps.

### 3.7 Refinement: parts (opt-in)

A roll-up of a long section may be refused by the embedder, or accepted and diluted into one
vector. Refinement restores a usable vector without losing text. It runs only when
`refine.budgets` lists at least one character budget **and** an embedder is configured.

- Each budget is one **pass**, in the order listed. A pass splits every level-1 text that has no
  parts yet and is over that budget, or refused, into **windows**: contiguous runs of its own
  children, joined as the parent joins them.
- Only the parts are embedded. Once they are, the parent's own vector is removed, and the removal
  is counted. A split text is then represented by its parts.
- A text of a single unit is never cut, and keeps its refusal.
- A pass that makes no part ends the passes. Each pass records the digest of the store it read.
- A part is a level-1 chunk whose `parent_id` names the text it was split from, with a `meta.part`
  block: `kind`, `pass`, `budget`, `index`, `span`, `parent_chars`.
- Parts are **not** added to the lexical index: their parent is there already and carries the same
  words.
- A failure stops the pass with the document and the pass named, and every text still holds the
  vectors it had.

---

## 4. The command line: saqqaractl

```bash
python -m ragix_kernels.saqqara.cli.saqqaractl <command> [options]
```

No console script is installed for it. Colour is used only when the output is a terminal.

| exit code | meaning |
|---|---|
| 0 | done |
| 1 | the kernel failed (`run`, `index`), the output is on standard error; or an uncaught configuration error in `search` |
| 2 | the input does not exist: no such source, no stored result, or no store |

### 4.1 `run` — read documents into typed trees

```bash
saqqaractl run SOURCE [-w WORKSPACE] [--formats EXTS] [--promote-outline] [--json] [-v]
```

| option | default | meaning |
|---|---|---|
| `SOURCE` | required | the file or directory to read |
| `-w`, `--workspace` | the parent directory of the resolved source | where `stage1/saqqara.json` is written; created if absent |
| `--formats` | every file | comma-separated extensions, each with its leading dot and in lower case, e.g. `.docx,.xlsx` |
| `--promote-outline` | off | also run the `outline` pass (§3.3) |
| `--json` | off | print the kernel's result as JSON, keys sorted; exit 0 on success and 1 on failure |
| `-v`, `--verbose` | off | list every abstention rather than the first ten |

The report prints the summary line, one line per document, then refusals, duplicates, counted
drops and abstentions where there are any, then both roots. A report key this command does not know
is printed raw rather than skipped.

```text
$ saqqaractl run demo-work/corpus -w demo-work/read2 --formats .md,.docx
2 document(s), 21 nodes. Refused 0, duplicate 0. 0 abstention(s), 0 drop(s) counted. root b28fd3a5e9b7.
read 2  refused 0  duplicate 0
  md          5 nodes  demo-work/corpus/notes.md
  docx       16 nodes  demo-work/corpus/report.docx

merkle_root  <64 hexadecimal characters>
source_root  <64 hexadecimal characters>

written to demo-work/read2/stage1/saqqara.json
```

The demo generators build new files on each run, so your roots will differ.

### 4.2 `status` — read back a previous run

```bash
saqqaractl status WORKSPACE [--json] [-v]
```

| option | default | meaning |
|---|---|---|
| `WORKSPACE` | required | a workspace a previous `run` wrote to |
| `--json` | off | print the stored result as JSON |
| `-v`, `--verbose` | off | list every abstention rather than the first ten |

It reads `WORKSPACE/stage1/saqqara.json` without reading any document. A workspace with no result
is an error (exit 2), never an empty answer.

```text
$ saqqaractl status demo-work
read 5  refused 0  duplicate 0
  pptx        9 nodes  demo-work/corpus/deck.pptx
  pdf         5 nodes  demo-work/corpus/figures.pdf
  md          5 nodes  demo-work/corpus/notes.md
  docx       16 nodes  demo-work/corpus/report.docx
  xlsx       47 nodes  demo-work/corpus/workbook.xlsx

abstentions  1
  header_bands: uniform-block  (demo-work/corpus/workbook.xlsx)
```

### 4.3 `index` — chunk what was read into a store

```bash
saqqaractl index WORKSPACE [-c CONFIG] [--json] [-v]
```

| option | default | meaning |
|---|---|---|
| `WORKSPACE` | required | a workspace holding `stage1/saqqara.json` |
| `-c`, `--config` | the packaged defaults | a `saqqara.yaml` overlay (§12) |
| `--json` | off | print the kernel's result as JSON |
| `-v`, `--verbose` | off | accepted, and has no effect |

The store is written at `store.path`, resolved against the workspace when relative: by default
`WORKSPACE/.ragix/saqqara.db`. The command prints the summary line, the store's counts and the
store's drops.

```text
$ saqqaractl index demo-work -c examples/saqqara/saqqara.yaml
5 document(s), 35 chunk(s). Embedded 0, already present 35, refused 0. Dense: disabled (no embedder).
  documents          5
  chunks             35
  objects            4
  edges              0
  embeddings         0
  embeddings_parked  0
  dense              disabled (no embedder)
```

With `embedder.provider: none`, "already present" counts every chunk: nothing was embedded, and
nothing was needed.

### 4.4 `search` — query the store

```bash
saqqaractl search WORKSPACE QUERY [-c CONFIG] [-k N] [--json] [-v]
```

| option | default | meaning |
|---|---|---|
| `WORKSPACE` | required | the workspace holding the store |
| `QUERY` | required | what to look for |
| `-c`, `--config` | the packaged defaults | the same overlay `index` used, so that `store.path` and the embedder agree |
| `-k`, `--top-k` | `retrieval.top_k` (10) | how many hits |
| `--json` | off | print the hits as JSON, the same shape the MCP tool returns |
| `-v`, `--verbose` | off | accepted, and has no effect |

Each hit shows its three ranks, the first 160 characters of its text, and the citation of every
node it was cut from.

```text
$ saqqaractl search demo-work Moyens -c examples/saqqara/saqqara.yaml -k 2
dense: disabled (no embedder) — lexical lane only

dense=None lexical=1 final=1
  Q1 Organisation | Q2 Moyens | Q3 Delais | Q4 Suivi
    table      demo-work/corpus/workbook.xlsx [{'format': 'xlsx', 'cell': 'B4:D8', 'col': 2, 'row': 4, 'sheet': 'Questionnaire', 'sheet_index': 1}]

dense=None lexical=2 final=2
  Formulaire
Reference | R-1 | R-2
Q1 Organisation | Q2 Moyens | Q3 Delais | Q4 Suivi
    cell       demo-work/corpus/workbook.xlsx [{'format': 'xlsx', 'cell': 'B1', 'col': 2, 'merged_range': 'B1:D2', 'row': 1, 'sheet': 'Questionnaire', 'sheet_index': 1}]
    table      demo-work/corpus/workbook.xlsx [{'format': 'xlsx', 'cell': 'F1:F3', 'col': 6, 'row': 1, 'sheet': 'Questionnaire', 'sheet_index': 1}]
    table      demo-work/corpus/workbook.xlsx [{'format': 'xlsx', 'cell': 'B4:D8', 'col': 2, 'row': 4, 'sheet': 'Questionnaire', 'sheet_index': 1}]
```

The second hit is the sheet's level-1 roll-up, which contains the first: both levels are searchable
(§8.5). Without `-c`, the command looks for `demo-work/.ragix/saqqara.db`, finds nothing, and exits
with 2.

---

## 5. The MCP tools

The RAGIX MCP server registers the four tools at start-up by calling
`register_saqqara_tools(server)` from `ragix_kernels/saqqara/mcp/tools.py`. If registration fails,
the server logs a warning and continues without them. Every tool returns `{"error": message}` when
it raises.

| tool | parameters | returns |
|---|---|---|
| `koas_saqqara_run` | `source`, `workspace=""`, `formats=""`, `promote_outline=False` | `success`, `summary`, `output_file`, `documents` (path, format, sha256, node count), `merkle_root`, `source_root`, `report`, `errors` |
| `koas_saqqara_status` | `workspace` | `meta` (the envelope's `_meta`), `documents` (path, format, sha256), `merkle_root`, `source_root`, `report`, `abstentions` |
| `koas_saqqara_index` | `workspace`, `config=""` | `success`, `summary`, `output_file`, `documents`, `status`, `embedded`, `skipped`, `errors` |
| `koas_saqqara_search` | `workspace`, `query`, `k=10`, `config=""` | `hits`: exactly what `saqqaractl search --json` prints |

Behaviours worth knowing, each read from the code:

- `koas_saqqara_run` with an empty `workspace` writes beside the source, in its parent directory,
  as the command line does. It does not create the workspace, and the envelope refuses one that
  does not exist.
- `koas_saqqara_status` builds its `abstentions` field from each trace's own `abstentions` key,
  where only `header_bands` keeps its records. The complete register is `report.abstentions` in
  the same answer.
- `koas_saqqara_index` creates the workspace, and refuses when no `stage1/saqqara.json` is in it.
- `koas_saqqara_search` runs the command line's search function and returns its hits. A search that
  exits non-zero, such as one with no store, returns `{"error": "search failed with code 2"}`.

---

## 6. From Python, and through KOAS

**Reading** (from `examples/saqqara/01_read_tree.py`):

```python
from ragix_kernels.base import KernelInput
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel

output = SaqqaraKernel().run(
    KernelInput(workspace="./demo-work", config={"source": {"path": "./demo-work/corpus"}}))
print(output.summary)
```

The kernel's configuration keys are `source.path` (required), `formats` (a list of extensions),
`promote_outline` (a boolean) and `analyzers` (per-analyzer options, §3.3).
`ragix_kernels.saqqara.kernel` re-exports the same class for older callers.

**Indexing** (from `examples/saqqara/02_index_store.py`):

```python
from pathlib import Path
from ragix_kernels.base import KernelInput
from ragix_kernels.saqqara.kernels.saqqara_index import SaqqaraIndexKernel

read = Path("./demo-work/stage1/saqqara.json")
output = SaqqaraIndexKernel().run(KernelInput(
    workspace="./demo-work", config={"config": "examples/saqqara/saqqara.yaml"},
    dependencies={"document_tree": read}))
```

Its configuration keys are `config` (a `saqqara.yaml` path), `overrides` (dotted keys applied after
the file, validated by the same rule, e.g. `{"refine.budgets": [4000]}`) and `result` (a stage-1
result to use instead of the declared dependency).

**Searching, with the `any` combinator** (from `examples/saqqara/03_search_with_trace.py`, with the
combinator added; the command line and the MCP tool always use `all`):

```python
from ragix_kernels.saqqara.store.config import load_config
from ragix_kernels.saqqara.store.ports import build_store
from ragix_kernels.saqqara.store.retrieve import Retriever, provenance_of

config = load_config("examples/saqqara/saqqara.yaml")
store = build_store({**config.section("store"), "path": "demo-work/saqqara.db"})
retriever = Retriever(store)
hits = retriever.search("Moyens Suivi", vector=None, top_k=5, combinator="any")
print(retriever.last_lanes)          # what each lane did, and why the lexical lane refused if it did
trees = {d.doc_id: d.tree for d in store.list_documents()}
for hit in hits:
    print(hit.final_rank, provenance_of(trees[hit.chunk.doc_id], hit.chunk))
```

**Through the KOAS orchestrator.** Both kernels are registered: `saqqara` at stage 1 and
`saqqara_index` at stage 2. The orchestrator passes a kernel's `options` mapping from the manifest
as its configuration, so the keys above go under `stage1.saqqara.options` and
`stage2.saqqara_index.options`. The saqqara gates do not exercise this route.

---

## 7. The store on disk

| path, relative to the workspace | written by | holds |
|---|---|---|
| `stage1/saqqara.json` | `run` | the envelope: `_meta` (kernel name and version, execution time, input hash, wall-clock timestamp, success) and `data` (documents, roots, report) |
| `stage1/saqqara.summary.txt` | `run` | the one-line summary |
| `stage2/saqqara_index.json` | `index` | the envelope around the index result: per-document counts, the store's status, embedding counts and refusals, refinement, the configuration used |
| `stage2/saqqara_index.summary.txt` | `index` | the one-line summary |
| `store.path`, by default `.ragix/saqqara.db` | `index` | the store: one SQLite file in WAL journal mode, so `-wal` and `-shm` files may sit beside it |

**What the file holds.** Tables `meta` (the schema version, 1), `documents`, `document_paths`,
`objects`, `edges`, `chunks`, `embeddings`, `embedding_refusals`, `embeddings_trash`, and the FTS5
table `chunks_fts` with its shadow tables. The developer reference gives every field. A document's
id is the sha256 of its bytes: the same bytes at two paths are one document, and `document_paths`
records every path it was seen at.

**Read-only mode: there is none in saqqara.** The store opens its file for writing and applies its
schema statements every time it is opened, `search` included. To inspect a store with no chance of
writing to it, open it with SQLite's read-only URI mode, or work on a copy:

```python
import sqlite3
con = sqlite3.connect("file:demo-work/saqqara.db?mode=ro", uri=True)
```

**Trash, restore and purge** exist in the Python API only; no command-line or MCP surface calls
them.

| call | effect | returns |
|---|---|---|
| `store.delete_document(doc_id)` | marks the document trashed and moves its vectors to `embeddings_trash` | `{documents, chunks, embeddings, parked}` |
| `store.restore_document(doc_id)` | clears the mark and puts **the same vectors** back; nothing is recomputed | `True`, or `False` if it was not trashed |
| `store.delete_document(doc_id, purge=True)` | removes the document, its chunks, lexical rows, vectors, parked vectors, objects, edges and paths | the counts removed |

A trashed document leaves both lanes and the `documents` count at once. Every trash and purge is
also recorded in the store's drop trace.

**Orphans.** Re-indexing a document replaces its chunk set: a chunk the tree no longer produces is
deleted with its lexical row and its vectors, counted as `chunk-superseded`. A part survives a
re-index while its parent is still produced, and goes when its parent goes. A vector whose chunk is
missing is never returned as a hit.

**The store's status**, printed by `index` and returned in its result:

| key | meaning |
|---|---|
| `path`, `corpus`, `schema_version` | where the file is, the store's corpus label, the schema version |
| `documents`, `trashed` | live and trashed documents |
| `objects`, `edges`, `chunks` | rows held |
| `embeddings`, `embeddings_parked` | live vectors, and vectors parked with trashed documents |
| `embedding_refusals` | texts the model refused, as of the last run over each document |
| `models` | every model that holds a vector |
| `drops` | what this process removed, and why (§9.8) |
| `dense` | added by `index`: `disabled (no embedder)`, or the model with its vector and refusal counts |

---

## 8. Querying

### 8.1 The lexical lane

The lexical lane is SQLite FTS5 over every stored chunk except parts, ranked by `bm25`. Its
tokenizer is `unicode61 remove_diacritics 2`, so a query typed without accents finds text written
with them.

The query string never reaches FTS5 as written. Each whitespace-separated word holding at least
one letter or digit is quoted and matched as a phrase by the table's own tokenizer, so apostrophes
and question marks are harmless. A word with no letter or digit is dropped.

| combinator | how the terms combine | where you can choose it |
|---|---|---|
| `all` (default) | every term must match | everywhere; the command line and the MCP tool always use it |
| `any` | at least one term must match | the Python API only: `Retriever.search(..., combinator="any")` or `store.lexical_search(query, k, combinator="any")` |

A query left with no searchable term is **refused with a reason**, `empty query` or `no term the
index can search`. The refusal is recorded on the store with the tokenizer in force, and reported by
`Retriever.last_lanes`. The command line and the MCP tool show such a query only as "no hit".

### 8.2 The dense lane

The dense lane loads the vectors of one model from the database into a `ragix_core` vector index,
`numpy` or `faiss` according to `index.backend`. The index is a cache rebuilt from the database
whenever the rows change, never a second store.

**The unit is the text, never a part.** A text split into parts has one vector per part. They are
collapsed to one hit at the best part, which is the maximum cosine, and the hit's `part` field says
which part won: `{"whole_text": true}`, or the part's `chunk_id`, `node_ids`, `index`, `span`,
`pass` and `budget`. To fill `top_k` texts despite that collapse, the lane over-fetches, doubling up
to a factor of 16.

### 8.3 The fused result

Both lanes are fused by **reciprocal rank fusion**. A hit scores `1 / (rrf_k + rank)` for each lane
that returned it, with `retrieval.rrf_k` = 60. Hits are ordered by that sum, ties broken by chunk
id. The fused rank never replaces the lane ranks: a hit carries `dense_rank`, `lexical_rank` and
`final_rank`, and `null` for a lane that did not return it, never a worst-case number. The dense
lane is asked for `retrieval.dense_k` (40) texts and the lexical lane for `retrieval.lexical_k` (40)
chunks before fusion.

### 8.4 What a hit carries

| field | meaning |
|---|---|
| `chunk` | `chunk_id`, `doc_id`, `seq`, `text`, `level`, `node_ids`, `parent_id`, `section_path`, `pages`, `lang`, `object_refs`, `meta` |
| `dense_rank`, `lexical_rank`, `final_rank` | the ranks; `null` for a lane that did not return the hit |
| `part` | the part the dense lane matched, or `null` for a lexical-only hit |
| `boosts` | always empty at this version |
| `provenance` | added by the command line and the MCP tool: for each node the chunk names, `node_id`, `kind`, `origin`, `confidence`, `source_path`, `source_format` and the locator `chain` |

### 8.5 Filters, and what is excluded without asking

There are **no query filters** at this version: no document, format, page, level or corpus filter on
any surface. What every query excludes without being asked:

- the chunks of a trashed document, on both lanes;
- parts, on the lexical lane, because their parent carries the same words;
- the lower-scoring parts of a text, on the dense lane.

Both levels are searchable on both lanes, so a unit and the roll-up that contains it can both be
returned, as in the example of §4.4.

---

## 9. Refusals, abstentions and the register

Nothing saqqara declines is silent. Each family below says where it appears and what to do.

### 9.1 Files refused at reading — `report.refusals`

| reason | meaning | what you do |
|---|---|---|
| `unsupported-format` | no reader claims the file's extension | nothing, if the file is not a document; `--formats` keeps it out of the report |
| `unreadable-file` | the file could not be read, or its reader failed on it; `detail` carries the message | open the file: it is damaged, protected, or not what its extension says |

**Duplicates**, `report.duplicates`, list the paths of byte-identical copies that were not read
again. They are not refusals.

### 9.2 The abstention register — `report.abstentions`

One record per abstention, never only a count: `path`, `analyzer`, `locator` (`null` where the
producer kept only a count), `reason`, the `signals` the rules read, and `count`.

| analyzer | reasons | meaning | what you do |
|---|---|---|---|
| `header_bands` | `no-populated-region`, `no-body-rows`, `uniform-block`, `band-too-deep`, `non-laminar-band-merges`, `no-column-evidence-within-cap` | the header band of a table could not be read; `band-too-deep` carries the depth found and the cap in force | read the table by hand; for `band-too-deep`, the signals say how far over the cap it went |
| `grid_tables` | `D4-unstyled-single-row`, `D5-no-positive-evidence` | a word-processing or slide table was typed `layout` (D4: unstyled, one row) or `table_uncertain` (D5: no positive evidence either way); the record carries the type, and neither is read by the header rules | check whether the table holds data |
| `saqqara.text_layer` | `no-text-layer` | a PDF page has no text layer; the locator names the page and the signals carry its image count and size | the page is unread: it needs OCR, which saqqara does not do |

Two further producers exist in the library, and a run does not call them: `format_headings`
(`no-size-contrast`, `no-weight-contrast`, `declared-outline`) and `saqqara.caption_binding`
(`ambiguous-candidates`, `no-candidate-within-gap`, `candidate-already-bound`).

### 9.3 Decisions carried on nodes and in traces

| where | reasons | meaning |
|---|---|---|
| a block's `block_uncertain` fact (`tables`) | `single-column-with-header-evidence`, `no-value-evidence` | the block could not be typed table, text or list |
| the `sections` trace, rejected candidates | `printed-contents`, `orphan-number`, `page-furniture` | a printed contents line, a number alone on its line, or a title repeated on three or more pages |
| the `sections` trace, a blind document | `no-channel-for-format`, `no-candidate-found` | the document yielded no section, and says why |
| the `outline` trace (opt-in) | `illegal-step`, `chain-too-short`, `flat-uncorroborated` | a numbered label that was not promoted |
| the `islands` trace | `feedback` records, and `segmentation_feedback` on the block | a block holding several disconnected value regions |

### 9.4 Drops the builder and analyzers count — `documents[].traces.*.dropped`

| trace | reason | meaning |
|---|---|---|
| `builder` | `unmapped-observation` | a reader emitted an observation no format plan maps to a node |
| `builder` | `drawing-without-a-page` | a drawing arrived with no page to attach it to |
| `builder` | `border-matches-no-cell` | a spreadsheet border surrounds no cell record |
| `tables` | `bordered-region-holds-no-value` | a ruled region with no value anywhere, counted with its range |

### 9.5 Chunker refusals — counted per document in the index result

| reason | meaning |
|---|---|
| `absorbed-by-an-aggregating-node` | a cell whose text is already in its table's chunk |
| `section-is-context` | a section node, which names a roll-up and is never a chunk |
| `not-a-chunkable-kind` | a `page` or a `figure`, each excluded with a declared reason |
| `no-text` | a node of a chunkable kind with no text, such as an empty cell |
| `binding-target-not-found` | a caption binding whose figure cannot be found in the tree |
| `binding-target-ambiguous` | a caption binding that resolves to more than one node, refused rather than guessed |

`documents[].refused` in the index result is this count. Embedding refusals are counted apart
(§9.6).

### 9.6 Embedding refusals — the store's `embedding_refusals` table

| reason | meaning | what you do |
|---|---|---|
| `input-rejected` | the server rejected this text alone; the signals carry the HTTP status, the server's message, the model, the batch size, the length in characters, the chunk's level and its sequence number | declare `refine.budgets` so the text is split into parts, or use a model with a longer context |

A refused text is found by halving the request until one text refuses alone, not by reading the
server's message. A refusal is **not a mark on the chunk**: the next run asks again. The index
result lists them under `embedder_refusals`, and the store keeps them per document until the next
run over that document replaces them.

### 9.7 Lexical refusals — `store.lexical_refusals`, `Retriever.last_lanes`

| reason | meaning |
|---|---|
| `empty query` | the query was empty or only whitespace |
| `no term the index can search` | no word of the query holds a letter or a digit |

### 9.8 Store drops — `status.drops`

| reason | meaning |
|---|---|
| `document-trashed` | a document was trashed; its vectors were parked |
| `document-purged` | a document was purged, with the counts removed |
| `chunk-superseded` | chunks a re-index no longer produced were deleted with their vectors |
| `vector-superseded-by-parts` | a split text's own vector was removed once its parts had theirs |

The drop trace lives in the store object, for the process that made the drops.

### 9.9 Configuration and run failures

These are raised, and a kernel run reports them as `success: false` with the message.

| message, in substance | cause |
|---|---|
| `unknown configuration key 'embedder.privider'; known here: …` | a key the defaults do not declare, reported with its path |
| `configuration key … is a mapping in the defaults but a value here` | a value where a mapping belongs, or the reverse |
| `no configuration file at …` | a `-c` path that is not a file |
| `embedder.provider must be one of …` | a provider outside `none`, `sentence-transformers`, `ollama`, `dummy` |
| `index.backend must be 'numpy' or 'faiss'` | any other backend |
| `the manifest configures analyzers this kernel does not run` / `… does not take …` | an unknown analyzer name, or an undeclared option |
| `source.path is required` | `run` with no source |
| `no stage-1 result at …` | `index` before `run` |
| `the dense lane holds no vector and refused N chunk(s) …` | every text was refused: check the model and the server |
| `refinement stopped on document … in pass … (budget …)` | a pass failed; completed documents are named, and every text still holds the vectors it had |
| `embedder returned N vectors for M chunks` / `empty vector for chunk …` / `… is both refused and embedded` | a backend answer that cannot be matched to its inputs |

---

## 10. Sovereignty

- **Local by construction.** Reading, analysing, chunking, storing and querying make no network
  call. The only network traffic saqqara can cause is to the embedder you configure: the `ollama`
  backend posts to `embedder.base_url`, or to the core layer's default `http://localhost:11434`. The
  `sentence-transformers` backend loads its model through that library.
- **No model reads your documents.** No language model is called at any point (§1).
- **Secrets travel by reference.** `embedder.api_key_ref` accepts `env:VAR` or `file:/path#Label`,
  is resolved only at the moment of use, and fails closed. A loaded configuration never contains a
  secret's value, so it is always safe to write down. No embedder of this version reads the key
  (§11).
- **Activity events.** RAGIX writes `koas.event/1.0` activity events, each carrying a sovereignty
  attestation, only when an activity writer has been initialised for the process. At this version
  the document-summary runner initialises one; no saqqara surface does. `saqqaractl`, the MCP tools
  and a direct `Kernel.run` therefore leave no activity event. Run through the KOAS orchestrator in
  a process whose caller has called `ragix_kernels.activity.init_activity_writer(workspace)`, each
  kernel leaves a start and an end event under scope `docs.kernel`, each with
  `sovereignty.local_only: true`, in `WORKSPACE/.KOAS/activity/events.jsonl`. Either way, a run's
  own evidence is its output: the two roots, the envelope's input hash, and the registers of §9.
- **The licence quarantine** of §14 keeps the AGPL renderer off every default path.

---

## 11. Limits and known behaviours

1. **A PDF line is often several nodes.** The PDF reader emits one observation per text-showing
   operation, and each becomes a `paragraph` node, so one printed line can be several nodes and
   several level-0 chunks. A value split across two placements is split across two chunks. Text
   observations carry the point where the text was placed, and no bounding box. Line assembly exists
   only in the library analyzer `format_headings`, which a run does not call.
2. **A PDF without a declared outline has no heading nodes.** Its chunks carry no section path, and
   the whole document is one level-1 roll-up. The same holds for every word-processing document and
   every presentation (§3.4).
3. **A page without a text layer is unread.** The page node records `has_text: false`,
   `needs_ocr: true` and its image count, the register lists it under `no-text-layer` with its page,
   and no chunk comes from it.
4. **Oversized texts.** A unit over 1200 characters is kept whole and flagged; a single node over
   4000 characters is cut into overlapping windows. A long section's roll-up can be refused by the
   embedder: it stays in the lexical lane, the refusal is recorded, and refinement (§3.7) restores
   its vector through parts.
5. **Only PDF chunks carry pages.** The word-processing reader derives each node's page from the
   document's own break marks, as a `derived` fact block (`page`, `pages`, `spans_break`, `source`,
   `declared_pages`, `derived_pages`, `consistent`). A document with no mark gets no page, never page
   1 by default. The chunker reads pages only from locators, so these derived pages do not reach
   `chunks.pages`.
6. **No figure, caption or drawn region from a run** (§3.3), so no edge in a store built by `index`
   (§3.5).
7. **Keys declared and not read.** In `defaults.yaml`, nothing reads `source.path`,
   `source.formats` or `source.promote_outline`: the run kernel takes its own configuration.
   `index.use_gpu` and `retrieval.fusion` are read by nothing; fusion is always reciprocal rank.
   `embedder.api_key_ref` resolves through `StoreConfig.api_key()`, and no embedder of this version
   calls it.
8. **The store's corpus label.** `store.corpus` is reported by the store's status. Documents written
   by the index kernel are recorded under the corpus `default`.
9. **`search` ignores `embedder.base_url` and `embedder.batch_size`.** It builds its embedder from
   the provider and model only, so an `ollama` server at another address is reached by `index` and
   not by `search`.
10. **Options accepted and not used.** `-v` on `index` and `search` has no effect. The name `outline`
    is accepted under `analyzers`, and its options are not passed to it.
11. **`--formats` compares exact suffixes.** Give each extension with its dot, in lower case.
12. **`search` does not catch configuration errors.** An unknown key in `-c` ends `search` with a
    Python traceback; `index` reports the same error as a failed run.
13. **Output files carry wall-clock time.** The envelope stamps `_meta.timestamp` and
    `execution_time_ms`, and vectors and refusals carry their time of writing. Reproducibility is
    stated on the trees and the roots, which carry none.
14. **The envelope truncates a summary line over 500 characters** and records a warning.

---

## 12. Configuration reference

The packaged `ragix_kernels/saqqara/store/defaults.yaml` is the shape of every configuration. A
`saqqara.yaml` is a partial overlay naming only what it changes; every key it names must exist in
the defaults, or loading refuses it with its path. This file configures `index` and `search`. The
`run` kernel is configured by its own keys (§6).

| key | default | read by | effect |
|---|---|---|---|
| `source.path` | `./docs` | nothing | declared, not read (§11) |
| `source.formats` | `[pdf, docx, xlsx, pptx, md]` | nothing | declared, not read |
| `source.promote_outline` | `false` | nothing | declared, not read |
| `store.provider` | `sqlite` | `build_store` | the only registered provider |
| `store.path` | `.ragix/saqqara.db` | `index`, `search` | the store file, relative to the workspace |
| `store.corpus` | `default` | the store | the label in `status` (§11) |
| `refine.budgets` | `[]` | `index` | character budgets, one refinement pass each, in order; empty means none |
| `refine.overlap_children` | `0` | `index` | overlap between consecutive windows, counted in children |
| `embedder.provider` | `none` | `index`, `search` | `none`, `sentence-transformers`, `ollama` or `dummy`; `dummy` is the core layer's test backend |
| `embedder.model` | `""` | `index`, `search` | the model; empty uses the backend's default and records the provider's name |
| `embedder.base_url` | `""` | `index` | the Ollama server; empty uses the core layer's default |
| `embedder.api_key_ref` | `""` | `StoreConfig.api_key()` | a secret by reference; no embedder uses it (§11) |
| `embedder.batch_size` | `128` | `index` | texts per HTTP request for `ollama`; `1` sends one text per request |
| `index.backend` | `numpy` | `search` | `numpy` or `faiss` |
| `index.use_gpu` | `false` | nothing | declared, not read |
| `chunker.unit_max_chars` | `1200` | `index` | a level-0 text over this is flagged `oversize` |
| `chunker.rollup_levels` | `1` | `index` | `0` switches level-1 roll-ups off |
| `chunker.window_fallback_chars` | `4000` | `index` | a single node over this is cut into windows |
| `retrieval.fusion` | `rrf` | nothing | declared, not read; fusion is always reciprocal rank |
| `retrieval.rrf_k` | `60` | `search` | the fusion constant |
| `retrieval.dense_k` | `40` | `search` | texts asked of the dense lane before fusion |
| `retrieval.lexical_k` | `40` | `search` | chunks asked of the lexical lane before fusion |
| `retrieval.top_k` | `10` | `search` | hits returned when `-k` is not given |

---

## 13. Gates and specification

`ragix_kernels/saqqara/SPEC.md` holds **153 falsifiable propositions**, each with the fixture that
exercises it and what would falsify it.

| gate | layer | propositions |
|---|---|---:|
| K1 | `model.py`: nodes, kinds, locators, provenance, stable JSON | 10 |
| K2 | `adapters/`: one reader per format | 25 |
| K3 | `analyzers/`, `builder.py`, `services.py` | 73 |
| K4 | the kernel envelope, the roots and the abstention register | 3 |
| K6 | objects: figures, captions, drawn regions | 20 |
| K7 | the store: one SQLite file, embeddings beside the chunks | 22 |
| | **total** | **153** |

K0 is the gate that lets the others mean something. It proves that the specification parses and is
self-consistent, that specification and fixtures agree in both directions, that the package defines
exactly the kernels it declares, that nothing trips the repository guard, and that the prose of this
page, the package README and the specification states the counts the tests pin.

**Fixtures are generated by code** (`tests/saqqara/generators.py`). No document is committed to this
repository.

```bash
python -m pytest tests/saqqara -q
```

At `75b0fe1`, on Python 3.12.12, the suite collects 783 tests: 775 pass and 8 skip. Three of the
skips are live embedding tests, run only with `OLLAMA_LIVE=1`. The developer reference lists all
eight.

---

## 14. Licence discipline

The package default is permissive: `pip install -e ".[saqqara]"` pulls the `pypdfium2` renderer,
under Apache-2.0, and nothing under a copyleft licence.

`pymupdf` is AGPL-3.0. It is reachable only through the opt-in `saqqara-mupdf` extra, which the
`all` extra does **not** include. The quarantine is enforced, not only documented, by
`ragix_kernels/saqqara/render/guard.py`:

- `scan_sources()` refuses an import of `pymupdf`, `fitz` or `pymupdf4llm` anywhere in the package
  except `render/mupdf.py`, the one module that exists to hold it;
- `loaded_agpl_modules()` proves at run time that a default route never loaded it.

Both are asserted in `tests/saqqara/test_k6_regions.py`.
