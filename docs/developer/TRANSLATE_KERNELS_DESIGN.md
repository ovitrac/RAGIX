# KOAS-Translate — Design Proposal for a Translation Kernel Family

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
**Status:** DRAFT proposal (for review before any implementation)
**Scope:** Promote the punctual EN→FR scientific translation pipeline
(`~/Documents/Adservio/draft/translation/`, ~1,955 LOC) into a registered KOAS
kernel family `translate`, reusing RAGIX infrastructure.

---

## 1. Motivation & current state

A protocol-driven (`translation.md`) pipeline already translated a full book
EN→FR, locally, on the GPU box. It is **not** throwaway scripting — it already
embodies the KOAS contract:

| Property | Where (current pipeline) |
|---|---|
| Deterministic decoding | `modelfiles/granite4.1-translate.Modelfile` (temp 0, top_p 1, top_k 1) |
| Idempotent + resumable | `out/tm.sqlite`, keyed by `segment_id` + `source_hash` |
| Auditable | per-segment `model`, `prompt_version`, `created_at/updated_at` |
| Protected spans | `⟦P0001⟧` codec in `pipeline/segment.py` / `pipeline/rebuild.py` |
| Terminology control | `glossary/glossary.csv` (EN, FR, rule) |
| Local-first | Ollama + conda env, no cloud |

Six stages: **extract → segment → draft (Pass A) → qa (Pass B) → harmonize →
rebuild**, orchestrated by a `Makefile`.

**Goal:** make this a first-class, reusable, multi-language RAGIX capability
rather than a single-book one-off.

## 2. Goals / non-goals

**Goals**
- Register the six stages as KOAS kernels (orchestrator-driven, provenance-tracked).
- Reuse `ragix_core/llm_backends.py`, the orchestrator, activity logging, MCP/CLI.
- Generalize beyond EN→FR (language-pair parameter) and beyond one book.
- Unify the protected-span codec (3rd incarnation: presenter, sealed, translate).
- Add a test suite (the current pipeline has none).

**Non-goals (for v1)**
- Replacing the SQLite translation memory with the generic kernel cache (keep the TM).
- Cloud translation backends (stays local/Ollama; commercial APIs only via explicit opt-in).
- A GUI/CAT editor.

## 3. The generative-vs-compute kernel distinction (must settle first)

KOAS's headline guarantee is *"kernels compute deterministically; LLMs only plan
and reason, never produce metrics — no hallucinated numbers."* Translation
kernels **generate content with the LLM**, so they are a **generative** kernel
class, not deterministic-compute.

This is already true of `reviewer` and `summary` (LLM-authored edits/summaries),
so the family is consistent — but the README guarantee should be reworded to make
the boundary explicit:

> *Compute* kernels (audit, security, parts of docs/summary) never put an LLM in
> the numeric path. *Generative* kernels (reviewer, summary authoring, **translate**)
> use the LLM for content, and guarantee **reproducibility** instead of
> determinism via greedy decoding (temperature 0), pinned model digests, and a
> recorded `prompt_version`.

Action: add a "Kernel classes" subsection to `README.md` / `docs/KOAS.md`.

## 4. The `translate` kernel family

Stage maps onto KOAS's coarse `stage` (1=collection, 2=analysis, 3=reporting);
fine ordering is the `requires` DAG.

```mermaid
graph LR
  E[translate_extract\nstage 1] --> S[translate_segment\nstage 1]
  S --> D[translate_draft\nstage 2 · Pass A]
  D --> Q[translate_qa\nstage 2 · Pass B]
  Q --> H[translate_harmonize\nstage 2]
  H --> R[translate_rebuild\nstage 3]
```

| Kernel | stage | requires | config (from manifest) | reads → writes (workspace) |
|---|---|---|---|---|
| `translate_extract` | 1 | — | `src_glob`, `max_pages?` | `src/*.pdf` → `out/source.md` |
| `translate_segment` | 1 | extract | `lang_pair`, `protect_rules` | `source.md` → `chunks.jsonl`, `tm.sqlite` |
| `translate_draft` | 2 | segment | `lang_pair`, `model`, `glossary`, `prompt_version`, `limit?` | TM.source → TM.raw_translation |
| `translate_qa` | 2 | draft | `model`, `prompt_version` | TM.raw → TM.qa_report, TM.final (when ok) |
| `translate_harmonize` | 2 | qa | `model`, `prompt_version` | TM.final → `chapter_revisions` |
| `translate_rebuild` | 3 | harmonize | `only_translated?` | chapters → `out/final.md` (+ unprotect) |

Each kernel implements `compute(self, input: KernelInput) -> Dict[str, Any]`
where `input.workspace` is the translation project dir, `input.config` the manifest
block, `input.dependencies` the upstream artifacts. `data` returns counts +
artifact paths; `summary` is the `make status` line (segments / translated / qa /
final / chapter_revisions). The base wraps it into `KernelOutput` with
`input_hash` for provenance — complementing (not replacing) the TM's per-segment
hashing.

**Resumability contract:** these kernels are long-running and externally stateful
(the TM). A kernel re-run is a no-op for segments whose `source_hash` and
`prompt_version` are unchanged — identical to today's behavior, but now reported
through `KernelOutput.summary` and the activity log.

## 5. Shared assets

### 5.1 Translation Memory store
Keep `pipeline/store.py` (218 LOC) as a family-local store
`ragix_kernels/translate/tm_store.py`. Schema is already clean (`segments`,
`chapter_revisions`); add `lang_pair` and `glossary_version` columns for
multi-pair reuse. **Do not** fold into the generic orchestrator cache — the TM is
a domain artifact (CAT-tool pattern), inspectable and portable.

### 5.2 Protected-span codec (unification)
Promote the `⟦P####⟧` mechanism to `ragix_kernels/shared/protected_spans.py`
with a tested API:

```python
def protect(text: str, rules: list[str]) -> tuple[str, dict[str,str]]   # → masked, map
def restore(text: str, mapping: dict[str,str]) -> tuple[str, Report]    # → text, missing/hallucinated
```

Then refactor presenter marp-protection and the sealed placeholder masking to use
it. This is the highest-leverage consolidation (one tested codec, three callers).

### 5.3 Glossary & model registry
- `glossary/<lang_pair>.csv` (EN,FR,rule today) → resolved by `lang_pair`.
- A small registry mapping `lang_pair → {ollama_model, modelfile, decoding}` so the
  pinned-determinism settings live in one declarative place (mirrors the sealed
  contracts pattern).

## 6. Infrastructure reuse
- Replace `pipeline/ollama_client.py` (198 LOC) with `ragix_core/llm_backends.py`;
  bake the deterministic params via the modelfile **and** assert them at call time.
- Emit per-stage events through the existing activity logger (`ragix_kernels/activity.py`).
- Optional `[translate]` extra in `pyproject.toml` (`pymupdf4llm`, etc.) — off the
  core install, exactly like `[sealed]`.

## 7. Exposure
- Orchestrator manifest template `templates/translate_manifest.yaml`.
- MCP tools (`koas_translate_run`, `koas_translate_status`) + `ragix-koas translate …`.
- `docs/KOAS_TRANSLATE.md` (mirror `KOAS_PRESENTER.md` structure).

## 8. Testing plan (mirror `tests/test_marp_postprocess.py`)
Pure-ish, no-LLM targets first:
- `protected_spans`: protect/restore round-trip, missing/hallucinated detection, idempotency.
- `segment`: chunk boundaries, hash stability, protected-map correctness.
- `rebuild`: stitch order, unprotect, missing-token report.
- TM store: idempotent upsert, `source_hash` invalidation, resume.
- LLM stages: mock backend (deterministic stub) to test gating logic (`qa` status=ok → final).

## 9. Module → kernel migration map

| Current | LOC | Becomes |
|---|---|---|
| `pipeline/extract.py` | 178 | `translate/extract.py` (kernel) — consider reusing `docs` extraction |
| `pipeline/segment.py` | 316 | `translate/segment.py` + `shared/protected_spans.py` |
| `pipeline/translate.py` | 194 | `translate/draft.py` |
| `pipeline/qa.py` | 151 | `translate/qa.py` |
| `pipeline/harmonize.py` | 170 | `translate/harmonize.py` |
| `pipeline/rebuild.py` | 132 | `translate/rebuild.py` |
| `pipeline/store.py` | 218 | `translate/tm_store.py` (+lang_pair) |
| `pipeline/ollama_client.py` | 198 | **drop** → `llm_backends.py` |
| `pipeline/glossary.py` / `config.py` | 49 / 127 | `translate/glossary.py` / merge into KOAS config |

## 10. Phasing
- **P1 — kernelize (no behavior change):** wrap the six stages as registered
  kernels reusing llm_backends + orchestrator + activity; TM unchanged; manifest +
  `koas` status. Acceptance: `ragix-koas translate` reproduces `make all` on the
  30-page snapshot byte-for-byte.
- **P2 — consolidate + test:** extract `shared/protected_spans.py`, refactor
  presenter/sealed onto it; full test suite green.
- **P3 — generalize:** language-pair parameter + glossary/model registry; MCP +
  CLI + `docs/KOAS_TRANSLATE.md`; README "kernel classes" wording.

## 11. Decisions (resolved 2026-06-27)
1. **Family name:** `translate`. ✓
2. **Extraction:** keep the dedicated `pymupdf4llm` extractor for fidelity on
   math/figures/protected spans; revisit converging with the `docs` family later. ✓
3. **TM location:** per-project `out/tm.sqlite` (portable, inspectable, CAT-tool layout). ✓
4. **Repo placement:** public `ovitrac/RAGIX`, consistent with sealed/pentest;
   only kernel code ships — book/source content always stays out. ✓
5. **Counts:** the 6-kernel `translate` family takes the registry 88 → 94 (noted).

## 12. Risks
- Long single-call latency (~40 min/chunk) — kernels must stream progress and be
  killable/resumable; the orchestrator timeout model needs a long-running mode.
- Determinism caveat: greedy decoding is reproducible *per model digest*; pin the
  Ollama image/digest or record it in provenance.
- Scope creep into a CAT GUI — explicitly out of scope for v1.
