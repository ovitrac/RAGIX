# RAGIX-Sealed

Sovereign processing of **sensitive, confidential documents** with a `human ↔ LLM`
protection boundary: the original is sealed (AES-256-GCM + AAD), sensitive values are
replaced by role-aware placeholders, the result is leak-scanned, and **only cooled
content may reach an LLM**. Confidential content stays fully visible to the authorized
human; it is hidden only from the eyes of any LLM.

Domain-neutral: the same machinery applies wherever confidentiality matters.

Full design & doctrine: [`docs/RAGIX_SEALED.md`](../docs/RAGIX_SEALED.md).

> **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Module map

| Package | Role |
|---|---|
| `ragix_core/crypto/sealed_aead.py` | AES-256-GCM + AAD primitive (`seal_bytes`/`open_bytes`) |
| `vault/` | RAGIX-owned reversible-redaction vault; human-authorized re-identification only |
| `contracts/` | 8 declarative YAML contracts + loader/validator |
| `ingest/` | Warm ingestion pipeline (state machine → `COOLED_INDEXABLE`), detection v0, leak scanner v0 |
| `multimodal/` | Source-class detection + derivative provenance graph |
| `model_router/` | Policy-first model cascade with refusal normalization |
| `workers/` | Sealed job protocol (remote AAD, cleanup attestation); in-process worker |
| `corpus/` | BM25 cooled-corpus index over placeholderized chunks; safe search |
| `kernels/` | Level-1 inventory + Level-2 analysis (v0) kernels |
| `reporting/` | Sanitized reports + the four §10.1 export modes |

## Boundary rules (normative)

- Operate only on cooled documents; placeholders + roles only; opaque IDs only.
- Citations are `doc_id / page / chunk_id` — never filenames or paths.
- Re-identification is a controlled, human-authorized export — never toward an LLM.
- All outputs pass the leak scanner; deny by default.

## Install

```bash
pip install -e .[sealed]   # adds pypdf, pdfminer.six, python-docx for PDF/DOCX extraction
```
(Core ingestion of TXT/MD is dependency-free.)

## Quickstart

```python
from ragix_sealed.contracts import load_contracts
from ragix_sealed.ingest import SealedIngestor, new_case_context
from ragix_sealed.corpus import CooledCorpusIndex
from ragix_sealed.corpus.chunking import chunk_cooled
from ragix_sealed.kernels import run_inventory, run_analysis, CorpusView
from ragix_sealed.reporting import build_sanitized_memo, render, ExportMode, render_reidentified
from ragix_sealed.vault import RAGIXSealedVaultBackend
from ragix_sealed.vault.backend import AuthorizationToken, ReidentificationPurpose

contracts = load_contracts()
vault = RAGIXSealedVaultBackend()
ing = SealedIngestor(contracts, vault)
ctx = new_case_context("case_001")

# 1. Ingest (warm -> cooled). Raw bytes are sealed; values are placeholderized.
status = ing.ingest(ctx, b"Contact jane.synthetic@example.com on 2024-03-15.", "txt", 1)
cooled = ing.get_cooled(status.doc_id)          # placeholderized, leak-scanned

# 2. Index + safe search.
idx = CooledCorpusIndex(contracts)
idx.add_document(ctx, status.doc_id, cooled.text)
hits = idx.search("contact")                    # SafeSearchResult (placeholders + citations)

# 3. Inventory + analysis (metrics / placeholders only).
inventory = run_inventory(CorpusView.from_statuses([status], contracts, idx.chunk_count))
analysis = run_analysis(chunk_cooled(ctx, status.doc_id, cooled.text), contracts.placeholder_schema)

# 4. Report — sanitized by default (safe for an LLM context).
memo = build_sanitized_memo(inventory, analysis)
print(render(memo, ExportMode.SANITIZED_LLM_SAFE, schema=contracts.placeholder_schema))

# 5. Re-identify — HUMAN path only, gated + watermarked, never toward an LLM.
auth = AuthorizationToken("human::reviewer-1", ReidentificationPurpose.REPORT_EXPORT_HUMAN, token="grant")
print(render_reidentified(memo, vault, ctx.case_id, auth))
```

## Tests

```bash
python -m pytest tests/sealed/ -q
```
Includes a red-team leakage suite (`tests/sealed/test_redteam.py`) that seeds canary
secrets and asserts no leakage across every public surface.

## Status & production gaps

Full roadmap implemented (vault, contracts, ingestion, multimodal/workers/router, cooled
corpus, inventory/analysis/reporting kernels, red-team suite). Remaining before production:

- **NER** — detection v0 is deterministic (regex) only; names/orgs/locations rely on the
  known-entity scanner + human review.
- **Persistence** — vault/ingest/corpus state is in-memory; no `/var/lib/ragix-sealed`
  store or envelope/KMS key wrapping yet.
- **`AuthorizationToken`** is a stub (needs signed, expiring, identity-bound grants).
- **Live transport** — model-router→Ollama and the SSH sealed-worker transport are
  pluggable stubs; the contradiction kernel is deferred to a model-router-backed kernel.
