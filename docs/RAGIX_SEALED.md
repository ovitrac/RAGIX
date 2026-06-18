# RAGIX-Sealed — Sovereign Processing of Sensitive Documents

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
**Status:** DRAFT v0.1 — sealed vault, contracts, and warm-ingestion MVP implemented; kernels pending

---

## 1. What RAGIX-Sealed is

RAGIX-Sealed extends RAGIX to operate on **sensitive, confidential documents** — any corpus
whose raw content (identities, personal data, references, attachments) must never reach an
LLM in clear. It lets local LLM-driven kernels *inventory, analyse, and report* on such a
corpus **without any raw content ever reaching the eyes of an LLM**.

It is domain-neutral: the same machinery applies wherever confidentiality matters. It builds
on RAGIX's existing principles — local-first, auditable, deterministic — and adds a sealed
subsystem with much-higher protection at the LLM boundary.

> **The reasoning layer begins only after documents have "cooled":** the original is
> encrypted, sensitive values are replaced by role-aware placeholders, the result is
> leak-scanned, and human review is applied where required.

---

## 2. Threat model (the keystone)

Everything in RAGIX-Sealed follows from one decision:

> **Confidential content stays fully visible to the authorized human (the operator). It is
> hidden only from the eyes of any LLM. The protection boundary is `human ↔ LLM`, not
> `human ↔ system`.**

Consequences:

| Principle | Meaning |
|---|---|
| **We never ask an LLM to keep secrets** | We prevent secrets from reaching it. LLM discretion is never part of the security boundary. |
| **Re-identification is a normal human path** | Resolving a placeholder back to a real value is a routine, *authorized + audited* operation for the operator — never a path that feeds an LLM. |
| **The leak scanner is an egress filter** | Its only job is to stop raw values crossing *toward* an LLM. It never gates the human. |
| **Protection is defense-in-depth at the boundary** | AAD-bound ciphertext + opaque IDs + boundary deny-list + multi-stage leak scan — all gating egress toward an LLM. |

### 2.1 The sealed zone

The category that matters is **inside vs outside the sealed zone** — not "local vs cloud"
and not "LLM vs non-LLM":

- **Inside (INTERNAL):** local sealed-zone models, invoked through a *sealed execution
  profile* that disables prompt echoing, raw transcript persistence, external telemetry,
  and developer-facing trace output. Only these may touch raw or semi-raw content.
- **Outside:** Claude Code, cloud LLMs, public orchestrators, external MCP servers,
  developer terminals — unless explicitly running under a sealed local profile.
- **Cooled kernels** operate on placeholderized content only.

---

## 3. Architecture overview

Two parts: **warm ingestion** (sealing ceremony) → **cooled kernels** (reasoning).

```text
              WARM (volatile, pre-cooling)                COOLED (safe to reason over)
   ┌───────────────────────────────────┐      ┌──────────────────────────────────────┐
   │ Zone 0  Drop box (bytes only)      │      │ Zone 2  Cooled operational corpus      │
   │ Zone 1  Sealed ingestion enclave   │ ───▶ │  - placeholderized text                │
   │   encrypt original immediately     │      │  - opaque doc/chunk IDs                │
   │   extract / scrub metadata         │      │  - sanitized metadata                  │
   │   detect entities → placeholderize │      │  - BM25 / vector indexes               │
   │   leak-scan → human review         │      │  Inventory / Analysis / Reporting      │
   │   (tmpfs working airlock)          │      │  kernels (placeholders only)           │
   └───────────────────────────────────┘      └──────────────────────────────────────┘
            /mnt/ragix-warm (volatile)                /var/lib/ragix-sealed (persistent)
```

### 3.1 Ingestion state machine

```text
RECEIVED → QUARANTINED → FINGERPRINTED → ORIGINAL_ENCRYPTED → NORMALIZED_INTERNAL
→ METADATA_SCRUBBED → ENTITY_DETECTED → PLACEHOLDERIZED → LEAK_SCANNED
→ HUMAN_REVIEWED_IF_REQUIRED → COOLED_INDEXABLE → REASONABLE
```

Only `COOLED_INDEXABLE` / `REASONABLE` documents are visible to kernels. The public
orchestrator sees **state transitions and metrics, never content**.

---

## 4. The sealed vault (implemented)

The cryptographic foundation and the reversible-redaction vault. It is **RAGIX-owned and
independent of CloakMCP** (different lifecycle — RAGIX-Sealed never restores a value into an
LLM workflow, only to the authorized human).

### 4.1 Modules

| Module | Role |
|---|---|
| `ragix_core/crypto/sealed_aead.py` | AES-256-GCM + AAD primitive: `SealingAAD`, `KeyRef`, `SealedBlob`, `seal_bytes` / `open_bytes` |
| `ragix_sealed/vault/backend.py` | RAGIX-owned `SealedVaultBackend` Protocol + types |
| `ragix_sealed/vault/native.py` | `RAGIXSealedVaultBackend` — the native default backend |

### 4.2 Cryptographic sealing (AES-256-GCM + AAD)

The associated data (AAD) is **load-bearing**: it binds each ciphertext to its case,
document, content hash, policy version, schema version, and ingestion state. Moving
ciphertext between cases, rolling back the policy, or substituting a document all fail
authentication — no plaintext is returned.

```python
from ragix_core.crypto.sealed_aead import KeyRef, SealingAAD, seal_bytes, open_bytes

key = KeyRef.generate("case-key::case_001")
aad = SealingAAD(
    case_id="case_001",
    doc_id="doc_8f3a91c0b72e",
    raw_sha256="…64 hex…",
    policy_version="seal-policy-0.1",
    placeholder_schema_version="placeholder-schema-0.1",
    ingestion_state="PLACEHOLDERIZED",
)

blob = seal_bytes(b"sensitive bytes", aad, key)   # fresh 96-bit nonce per call
plain = open_bytes(blob, aad, key)                # raises SealedAEADError on any mismatch
```

`SealedBlob` carries only ciphertext + non-secret metadata; it is safe to persist and to
expose as an opaque object. A wrong key, mismatched AAD, or tampered ciphertext raises
`SealedAEADError` and yields no plaintext.

### 4.3 The vault — placeholders & human-authorized re-identification

```python
from ragix_sealed.vault import (
    RAGIXSealedVaultBackend, PlaceholderPolicy,
    AuthorizationToken, ReidentificationPurpose,
)

vault = RAGIXSealedVaultBackend()
person = PlaceholderPolicy(entity_type="PERSON")

# 1. Replace a raw value with a stable, role-aware placeholder.
ref = vault.create_placeholder("case_001", "PERSON",
                               raw_value="Jane Synthetic", role="primary_party", policy=person)
ref.placeholder          # -> "[PERSON_001 | primary_party]"   (role visible, identity sealed)
ref.to_public_dict()     # public-facing: NO raw value

# 2. Re-identify — HUMAN path only, requires authorization + a human purpose.
auth = AuthorizationToken(subject="human::reviewer-1",
                          purpose=ReidentificationPurpose.AUTHORIZED_REVIEW,
                          token="…grant…")
raw = vault.resolve_placeholder("case_001", ref.placeholder,
                                ReidentificationPurpose.AUTHORIZED_REVIEW, auth)
# -> "Jane Synthetic"   (returned to the authorized human; never to an LLM)
```

There is **no API path** that returns a raw value into an LLM prompt. `resolve_placeholder`
raises `VaultAuthorizationError` without a valid, purpose-matched authorization.

### 4.4 Security properties (locked by tests)

The `tests/sealed/` suite verifies (vault portion):

- **AAD binding** rejects cross-case / policy-rollback / state-confusion at authentication.
- **Tamper & wrong-key detection** — flipping a ciphertext byte or using different key
  material fails; no plaintext returned.
- **Fresh nonce per seal** — identical plaintext yields different blobs.
- **Placeholder stability / dedup** — same raw → same placeholder; counters are per
  `(case, entity_type)`; the dedup index stores an HMAC, never the raw value.
- **Human-authorized re-identification only** — denied without a token or on purpose
  mismatch.
- **No-raw boundary** — public-facing objects carry no raw values and no leak-prone keys
  (`raw_value`, `original_filename`, `plaintext`, `decrypted`, `vault_mapping`,
  `prompt_text`, `source_path`); secret-bearing reprs are masked.
- **Case isolation** — per-case keys + AAD `case_id` binding prevent cross-case opening.

---

## 5. Placeholder vocabulary

Role-aware placeholders preserve the abstractions reasoning needs while leaking nothing
identifying. The role label is visible to the reasoning layer; the real value stays in the
encrypted vault.

```text
[PERSON_001 | primary_party]   [ORG_001 | organization]    [LOCATION_008]
[REFERENCE_004]                [DATE_017 | month external]  [ID_NUMBER_002]
```

Entity classes (initial): PERSON, ORG, CONTRACT, DATE, AMOUNT, ADDRESS, EMAIL, PHONE,
REFERENCE, ID_NUMBER, DOCUMENT, EVENT, LOCATION. The same vocabulary applies to text and to
multimodal-derived text (OCR, captions, transcripts).

---

## 6. Kernel boundary rules (normative)

All cooled kernels must obey:

- **K1** Operate only on cooled documents.
- **K2** No raw entity values — placeholders + roles only.
- **K3** No raw filenames — opaque IDs only.
- **K4** Citations are `doc_id / page / chunk_id` only — never filenames or paths.
- **K5** Re-identification is a controlled human export, not a reasoning kernel.
- **K6** All outputs pass the leak scanner before release.
- **K7** Deny by default — uncertain ⇒ block + request review.

### 6.1 Export modes

| Mode | Audience | Content |
|---|---|---|
| `SANITIZED_LLM_SAFE` | Any LLM context (default) | Placeholders + roles + opaque IDs; passes leak scan |
| `HUMAN_AUTHORIZED` | Authorized human, audited | May contain re-identified raw values; never enters an LLM |
| `AUDIT_ONLY` | Audit trail | Provenance, hashes, metrics, versions; no document content |
| `ORCHESTRATOR_METRICS` | Public orchestrator (e.g. Claude Code) | Counts, states, confidences; never text |

---

## 7. Contracts and warm ingestion (implemented)

**Contracts** (`ragix_sealed/contracts/`): eight declarative YAML files plus a
loader/validator — `policy.yaml` (trust invariant, isolation levels, export modes),
`placeholder_schema.yaml`, `ingestion_state_machine.yaml`, `model_registry.yaml`,
`worker_config.schema.yaml`, `tool_matrix.yaml`, `audit_event.schema.yaml`,
`provenance.schema.yaml`. The validator enforces internal consistency (state-machine
reachability, tool-matrix disjointness, no-raw field discipline, etc.).

**Warm ingestion** (`ragix_sealed/ingest/`): a dependency-free pipeline that drives the
state machine from RECEIVED to COOLED_INDEXABLE — opaque-id derivation, immediate sealing of
the original, TXT/MD extraction (PDF/DOCX via the optional `sealed` extra), metadata scrub,
regex detection v0 (EMAIL/PHONE/DATE/AMOUNT/REFERENCE), vault placeholderization, and leak
scanner v0. A leak FAIL routes the document to BLOCKED; the public `IngestStatus` and the
cooled document carry no raw content.

```bash
pip install -e .[sealed]   # enables PDF/DOCX extraction (pypdf / pdfminer.six / python-docx)
```

---

## 8. Status & roadmap

| Sprint | Deliverable | Status |
|---|---|---|
| **0 — Vault spike** | Vault interface + native AES-GCM+AAD backend, synthetic tests | ✅ **Done** |
| **1 — Contracts** | 8 declarative YAML contracts + loader/validator + tests | ✅ **Done** |
| **2 — Warm Ingestion MVP** | Opaque IDs, immediate encryption, TXT/MD extraction (PDF/DOCX via `sealed` extra), metadata scrub, regex detection v0, placeholderization, leak scanner v0, state-machine pipeline → COOLED_INDEXABLE | ✅ **Done** |
| **2bis — Multimodal & Workers** | Source-class detection, derivative provenance graph, sealed worker job protocol (remote AAD + cleanup attestation; SSH transport deferred), policy-first model cascade | ✅ **Done** |
| **3 — Cooled Corpus Index** | Cooled-text chunker (opaque chunk IDs), BM25 over placeholderized chunks, defensive leak re-check, safe search API (`doc_id/page/chunk_id` citations); vector index deferred | ✅ **Done** |
| **4 — Inventory Kernels** | Corpus metrics, typology, entity inventory, quality, review queue | ⏳ Next |
| **5 — Analysis Kernels** | Timeline, entity-role graph, contradiction, commitment extraction v0, gap detection v0 | Planned |
| **6 — Reporting Kernels** | Sanitized memo, contradiction report, commitment matrix, audit attestation, re-identification design | Planned |
| **7 — Red-Team & Leakage Tests** | Seeded fake secrets; log / prompt / metadata / OCR / vector / report leakage tests | Planned |

### Current limitations

- Vault and ingestion state are **in-memory** — persistence (`/var/lib/ragix-sealed/`) and
  envelope encryption of per-document keys by a master/KMS key are upcoming work.
- **No NER yet** — detection v0 covers deterministic patterns only; names/orgs/locations
  require the NER layer (later sprint) and human review.
- `AuthorizationToken` is a stub — production needs signed, expiring, identity-bound grants.

---

## 9. Doctrine

> We do not ask a public or semi-public LLM to keep secrets. We prevent secrets from
> reaching it. Confidential content stays visible to the authorized human and is hidden
> only from the eyes of any LLM. RAGIX-Sealed is the sealed intermediary between sensitive
> material and external orchestration.

*Some design ideas were refined with the help of large language models, used strictly as
tooling.*
