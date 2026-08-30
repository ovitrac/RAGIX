# KOAS Tender — Structured Response Preparation over Document Stores

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

**Version:** 0.1 (2026-08-30)

---

## 1. Overview

A family of kernels that work over a document store built by
[saqqara](KOAS_SAQQARA.md): they read what a store already holds and prepare
structured responses from it.

Three things it does not do, and they are the shape of the family rather than
temporary limitations:

- **it never builds a store** — the store is whatever `saqqara_index` wrote;
- **it never reads a source file** — everything comes from the stored trees, which
  are the only things carrying provenance;
- **it never writes to what it reads** — asserted on the database file's sha256
  across a run, not on intent.

### What this family will hold

Kernels that turn a store into a prepared response: recovering the questions a
document asks and the regions that answer them, pairing a new question against
prior evidence, and drafting only from evidence that was selected first. Every one
of those is a decision that must be traceable back to a node, which is why the
family sits on a store that keeps provenance rather than on a corpus.

Today it holds one kernel that computes almost nothing — see below.

## 2. Installation

```bash
pip install -e ".[saqqara]"
```

No extra of its own. The family imports `saqqara` and nothing else.

## 3. The one kernel

`tender_probe` (stage 3) opens a store, asks what it holds, and reports that.

| | |
|---|---|
| `requires` | `["document_store"]` |
| `provides` | `["tender_probe"]` |
| writes | nothing |

It exists so that discovery, registration, the dependency declaration, both
surfaces, the configuration refusals and the guard scope are proved on something
that cannot be wrong on substance. Every one of those is easier to get wrong than
the arithmetic they will later carry, and a skeleton is the only thing that
isolates them.

```bash
python -m ragix_kernels.tender.cli.tenderctl probe -c tender.yaml
python -m ragix_kernels.tender.cli.tenderctl status -c tender.yaml
```

`probe --json` prints exactly what the MCP tool returns.

## 4. MCP tools

Registered by `ragix_kernels.tender.mcp.tools.register_tender_tools(server)`.

### `koas_tender_probe(config="")`

Returns `{"probe": {"store_path", "documents", "chunks", "dense_enabled", "dense"}}`.

`dense_enabled` is a field rather than something inferred from a count: a store
with no vectors because nothing was embedded, and one with none because embedding
failed, are different situations and only the store can tell them apart.

### `koas_tender_status(config="")`

Returns `{"store_path", "exists", "config"}` without opening the store — it answers
"which store did you mean", which is the question a refusal from `probe` raises.

## 5. Configuration

```yaml
store:
  path: .ragix/saqqara.db
probe: {}
```

The packaged `defaults.yaml` is the shape; a user file is a partial overlay naming
only what it changes. An unknown key is refused **with its path**, because a typo
accepted in silence is the user's instruction discarded. The rules are imported
from the store's configuration rather than copied: one rule in one place, since
the copy that drifts is whichever nobody is watching.

## 6. Gates

`tests/tender/test_t0_family.py`, gate **T0 — 7 propositions**:

| id | claim |
|---|---|
| T0.1 | the registry finds exactly the kernels this family declares |
| T0.2 | the dependency declaration is enforced by the envelope, and ordered after the store |
| T0.3 | the probe reports the store's own counts, and states whether dense is enabled |
| T0.4 | the probe does not modify the store — checked on its sha256 |
| T0.5 | the CLI and the MCP surface return the same result, and the server registers the family |
| T0.6 | an unknown configuration key is refused with its path; a missing store is refused, never created |
| T0.7 | adding this family moves none of saqqara's pinned lists or frozen counts |

The fixture is a real store, built at test time from the same generators the
saqqara gates use. No document is committed to this repository.
