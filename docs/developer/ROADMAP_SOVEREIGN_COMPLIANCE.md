# Roadmap: Sovereign Compliance Implementation

**Project:** RAGIX — KOAS Document Pipeline
**Reference:** `docs/SOVEREIGN_LLM_OPERATIONS.md` (v1.2.0)
**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Date:** 2026-01-29
**Status:** Gap Analysis & Implementation Plan

---

## Executive Summary

The `SOVEREIGN_LLM_OPERATIONS.md` document specifies normative requirements (MUST/MUST NOT) for sovereign LLM operations. This roadmap documents the **compliance gap** between the specification and current implementation, and provides a **prioritized implementation plan**.

| Category | Implemented | Partial | Not Implemented |
|----------|-------------|---------|-----------------|
| **MUST** (6 items) | 0 | 3 | 3 |
| **MUST NOT** (5 items) | 1 | 1 | 3 |
| **Overall Compliance** | **18%** | — | — |

**Target:** 100% compliance for v0.65.0 release.

**Risk Summary:**
| Phase | Risk Level | Breaking Changes |
|-------|------------|------------------|
| Phase 1 | HIGH | Output format, CLI defaults |
| Phase 2 | MEDIUM | Cache schema |
| Phase 3 | LOW | None |
| Phase 4 | LOW | None |

---

## Table of Contents

1. [Gap Analysis Report](#1-gap-analysis-report)
2. [Implementation Plan](#2-implementation-plan)
3. [Phase 1: Output Isolation (Priority: Critical)](#3-phase-1-output-isolation)
4. [Phase 2: Provenance & Merkle Roots (Priority: High)](#4-phase-2-provenance--merkle-roots)
5. [Phase 3: Code Protection & Mode Awareness (Priority: Medium)](#5-phase-3-code-protection--mode-awareness)
6. [Phase 4: Audit Trail Enhancement (Priority: Medium)](#6-phase-4-audit-trail-enhancement)
7. [Verification Checklist](#7-verification-checklist)
8. [Impact & Risk of Regression](#8-impact--risk-of-regression)
9. [References](#9-references)

---

## 1. Gap Analysis Report

### 1.1 MUST Requirements — Compliance Status

| # | Requirement | Status | Current State | Target |
|---|-------------|--------|---------------|--------|
| M1 | Strip Markdown metadata in external outputs | ❌ **NOT IMPL** | No `--output-level` flag | CLI flag + build-time validation |
| M2 | Enforce output-level contract via CLI | ❌ **NOT IMPL** | No denylist validation | `--output-level=external\|orchestrator\|compliance` |
| M3 | Forced caching with `call_hash`, `inputs_merkle_root` | ⚠️ **PARTIAL** | `LLMCache` exists, no Merkle | Add Merkle root computation |
| M4 | Protect fenced/inline code from boilerplate | ❌ **NOT IMPL** | No code fence detection | Extract/restore code blocks |
| M5 | Canonical ordering for reproducibility | ⚠️ **PARTIAL** | Seeds exist, no ordering | Sort children by `(path, chunk_index)` |
| M6 | Seed logging in audit trail | ⚠️ **PARTIAL** | Seeds in config | Log to `kernel_execution[].seed` |

### 1.2 MUST NOT Requirements — Compliance Status

| # | Requirement | Status | Current State | Target |
|---|-------------|--------|---------------|--------|
| N1 | Apply boilerplate to Code/Docs+Code mode | ⚠️ **UNCLEAR** | No mode check | Skip boilerplate for `--type code\|mixed` |
| N2 | Include provenance markers in external outputs | ❌ **NOT IMPL** | No denylist scan | `DENYLIST_KEYS` check in report assembly |
| N3 | Modify representative content in `doc_extract` | ✅ **CORRECT** | Scoring only, no modification | *(maintained)* |
| N4 | Allow orchestrator to see excerpts | ❌ **NOT IMPL** | No metrics-only mode | `--output-level=orchestrator` |
| N5 | Trust implicit defaults for isolation | ❌ **NOT IMPL** | Implicit defaults | Explicit `--isolation-mode` required |

### 1.3 Evidence Matrix

| File | Location | Finding |
|------|----------|---------|
| `run_doc_koas.py` | Lines 941-964 | No `--output-level` argument |
| `cache.py` | Lines 109-336 | `LLMCache` exists, no Merkle |
| `doc_extract.py` | Lines 380-399 | `_compile_boilerplate_pattern()` — no code fence protection |
| `doc_extract.py` | Lines 406-482 | `_score_sentence_quality()` — correct (scoring only) |
| `config.py` | Line 352 | `seed: int = 42` in `ClusteringConfig` |
| `doc_cluster_leiden.py` | Line 58 | `seed: int = 42` in `LeidenConfig` |

---

## 2. Implementation Plan

### 2.1 Priority Matrix

| Phase | Scope | Priority | Effort | Dependencies |
|-------|-------|----------|--------|--------------|
| **Phase 1** | Output Isolation | **CRITICAL** | 3 days | None |
| **Phase 2** | Provenance & Merkle | **HIGH** | 4 days | Phase 1 |
| **Phase 3** | Code Protection | **MEDIUM** | 2 days | None |
| **Phase 4** | Audit Trail | **MEDIUM** | 2 days | Phase 2 |

### 2.2 Milestone Schedule

```
Week 1: Phase 1 (Output Isolation) + Phase 3 (Code Protection)
Week 2: Phase 2 (Provenance) + Phase 4 (Audit Trail)
Week 3: Integration testing + Documentation update
```

---

## 3. Phase 1: Output Isolation

**Goal:** Implement `--output-level` CLI flag with build-time validation.

### 3.1 New CLI Arguments

**File:** `ragix_kernels/run_doc_koas.py`

```python
# Add to run_parser (~line 962)
run_parser.add_argument(
    "--output-level",
    choices=["internal", "external", "orchestrator", "compliance"],
    default="internal",
    help="Output isolation level: internal (full), external (redacted), orchestrator (metrics only), compliance (full + attestation)"
)
run_parser.add_argument(
    "--redact-paths", action="store_true",
    help="Redact file paths in output (implied by --output-level=external)"
)
run_parser.add_argument(
    "--anonymize-ids", action="store_true",
    help="Anonymize internal identifiers (implied by --output-level=external)"
)
run_parser.add_argument(
    "--strip-metadata", action="store_true",
    help="Strip internal metadata (implied by --output-level=external)"
)
```

### 3.2 Denylist Validation

**New File:** `ragix_kernels/output_sanitizer.py`

```python
"""
Output sanitizer for sovereign compliance.
Enforces MUST requirements M1, M2 and MUST NOT requirement N2.
"""

import re
from pathlib import Path
from typing import List, Set
from dataclasses import dataclass
from enum import Enum

class OutputLevel(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    ORCHESTRATOR = "orchestrator"
    COMPLIANCE = "compliance"

# Denylist keys that MUST NOT appear in external outputs
DENYLIST_KEYS = [
    "llm_trace", "call_hash", "inputs_merkle_root",
    "run_id", "endpoint", "model", "cache_status",
    "prompt_tokens", "completion_tokens", "digest",
    "prompt_hash", "response_hash", "sovereignty"
]

# Path patterns to redact
PATH_PATTERNS = [
    r"/home/[^/]+/",           # Unix home directories
    r"/Users/[^/]+/",          # macOS home directories
    r"C:\\Users\\[^\\]+\\",    # Windows home directories
    r"/tmp/[^/]+/",            # Temp directories
]

# ID patterns to anonymize
ID_PATTERNS = [
    r"F\d{6}",                 # File IDs (F000123)
    r"run_\d{8}_\d{6}_[a-f0-9]+",  # Run IDs
    r"[a-f0-9]{16,}",          # Long hashes
]

class SecurityViolation(Exception):
    """Raised when denylist keys appear in external output."""
    pass

def validate_external_report(report_path: Path, level: OutputLevel) -> bool:
    """
    Validate report for external delivery.

    Raises SecurityViolation if denylist keys detected.
    """
    if level == OutputLevel.INTERNAL:
        return True  # No validation for internal

    content = report_path.read_text()

    if level in (OutputLevel.EXTERNAL, OutputLevel.ORCHESTRATOR):
        for key in DENYLIST_KEYS:
            if key in content:
                raise SecurityViolation(
                    f"Denylist key '{key}' found in {level.value} report. "
                    f"Build FAILED. Remove or redact before delivery."
                )

    return True

def redact_paths(content: str) -> str:
    """Redact file system paths."""
    result = content
    for pattern in PATH_PATTERNS:
        result = re.sub(pattern, "[PATH-REDACTED]/", result)
    return result

def anonymize_ids(content: str) -> str:
    """Anonymize internal identifiers."""
    result = content
    # Replace file IDs with sequential anonymous IDs
    file_ids = set(re.findall(r"F\d{6}", result))
    for i, fid in enumerate(sorted(file_ids)):
        result = result.replace(fid, f"[DOC-{chr(65+i)}]")

    # Replace run IDs
    result = re.sub(r"run_\d{8}_\d{6}_[a-f0-9]+", "[RUN-ID]", result)

    return result

def strip_metadata_blocks(content: str) -> str:
    """Strip YAML/TOML front-matter and provenance comments."""
    # YAML front-matter
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    # TOML front-matter
    content = re.sub(r"^\+\+\+\n.*?\n\+\+\+\n", "", content, flags=re.DOTALL)
    # HTML provenance comments
    content = re.sub(r"<!-- PROVENANCE.*?-->", "", content, flags=re.DOTALL)
    # Fenced metadata blocks
    content = re.sub(r"```metadata\n.*?\n```", "", content, flags=re.DOTALL)

    return content

def sanitize_for_level(content: str, level: OutputLevel) -> str:
    """Apply sanitization based on output level."""
    if level == OutputLevel.INTERNAL:
        return content

    if level == OutputLevel.ORCHESTRATOR:
        # Return metrics only - strip all text content
        # This requires structured extraction, not simple regex
        raise NotImplementedError("Orchestrator mode requires structured output")

    if level in (OutputLevel.EXTERNAL, OutputLevel.COMPLIANCE):
        content = strip_metadata_blocks(content)
        content = redact_paths(content)
        content = anonymize_ids(content)

    return content
```

### 3.3 Integration Points

**File:** `ragix_kernels/docs/doc_final_report.py`

Add sanitization call before writing final report:

```python
from ragix_kernels.output_sanitizer import (
    OutputLevel, sanitize_for_level, validate_external_report
)

# In compute() method, before writing report:
output_level = OutputLevel(input.config.get("output_level", "internal"))

# Sanitize content
report_content = sanitize_for_level(report_content, output_level)

# Write report
report_path.write_text(report_content)

# Validate (raises SecurityViolation if denylist detected)
validate_external_report(report_path, output_level)
```

### 3.4 Acceptance Criteria

- [ ] `--output-level=external` removes all denylist keys
- [ ] `--output-level=external` redacts file paths
- [ ] `--output-level=external` anonymizes internal IDs
- [ ] Build fails if denylist key detected in external output
- [ ] Unit tests for sanitization functions

---

## 4. Phase 2: Provenance & Merkle Roots

**Goal:** Implement `call_hash` and `inputs_merkle_root` for all LLM calls.

### 4.1 Canonical Request Computation

**File:** `ragix_kernels/llm_wrapper.py` (extend existing)

```python
import hashlib
import json
import re
from typing import List, Dict, Any

def canonicalize_llm_request(request: dict) -> str:
    """
    Produce stable JSON for cache key computation.

    Implements MUST M3: Forced caching with call_hash.
    """
    canonical = {
        "model": request["model"],
        "temperature": request.get("temperature", 0.0),
        "template_id": request.get("template_id"),
        "template_version": request.get("template_version"),
        "messages": _canonicalize_messages(request.get("messages", [])),
    }
    # Sorted keys, no whitespace variance
    return json.dumps(canonical, sort_keys=True, separators=(',', ':'))

def _canonicalize_messages(messages: List[dict]) -> List[dict]:
    """Normalize message content."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        # Remove volatile fields (timestamps, run IDs)
        content = re.sub(r'run_\d{8}_\d{6}_[a-f0-9]+', 'RUN_ID', content)
        result.append({"role": msg.get("role", "user"), "content": content})
    return result

def compute_call_hash(request: dict) -> str:
    """Compute SHA256 hash of canonical request."""
    canonical = canonicalize_llm_request(request)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

### 4.2 Merkle Root for Pyramidal Synthesis

**File:** `ragix_kernels/merkle.py` (new file)

```python
"""
Merkle tree computation for pyramidal provenance.

Implements MUST M3: inputs_merkle_root
Implements MUST M5: Canonical ordering for reproducibility
"""

import hashlib
from typing import List, Dict, Any

def sha256(data: str) -> str:
    """Compute SHA256 hash."""
    return hashlib.sha256(data.encode()).hexdigest()

def compute_inputs_merkle_root(children: List[dict]) -> str:
    """
    Compute Merkle root of child inputs for cache key.

    MUST M5: Children ordered by (file_path, chunk_index) for determinism.
    """
    if not children:
        return sha256("")

    # Step 1: Sort children deterministically (MUST M5)
    sorted_children = sorted(
        children,
        key=lambda c: (c.get("file_path", ""), c.get("chunk_index", 0))
    )

    # Step 2: Hash each child's content
    child_hashes = []
    for c in sorted_children:
        content_hash = c.get("content_hash") or sha256(c.get("content", ""))
        child_hashes.append(content_hash)

    # Step 3: Build Merkle tree
    while len(child_hashes) > 1:
        if len(child_hashes) % 2 == 1:
            child_hashes.append(child_hashes[-1])  # Duplicate last if odd
        child_hashes = [
            sha256(child_hashes[i] + child_hashes[i+1])
            for i in range(0, len(child_hashes), 2)
        ]

    return child_hashes[0]

def build_node_ref(
    level: str,
    node_id: str,
    parents: List[str],
    children: List[dict]
) -> dict:
    """
    Build a NodeRef for synthesis provenance.

    Used for tracking what an LLM call is summarizing.
    """
    return {
        "level": level,  # chunk | doc | group | domain | corpus
        "node_id": node_id,
        "parents": parents,
        "children_count": len(children),
        "inputs_merkle_root": compute_inputs_merkle_root(children),
    }
```

### 4.3 Enhanced LLM Call Logging

**File:** `ragix_kernels/cache.py`

Extend `CacheEntry` to include provenance fields:

```python
@dataclass
class CacheEntry:
    """A cached LLM response with full provenance."""
    cache_key: str
    call_hash: str  # NEW: SHA256 of canonical request
    model: str
    model_digest: str
    prompt_hash: str
    temperature: float
    created_at: str
    response: str
    response_hash: str
    sovereignty: Dict[str, Any]
    usage: Dict[str, int] = field(default_factory=dict)
    # NEW: Provenance fields
    template_id: str = ""
    template_version: str = ""
    inputs_merkle_root: str = ""  # For pyramidal synthesis
    node_ref: Dict[str, Any] = field(default_factory=dict)
```

### 4.4 Acceptance Criteria

- [ ] `call_hash` computed for every LLM call
- [ ] `inputs_merkle_root` computed for pyramidal synthesis
- [ ] Children sorted by `(file_path, chunk_index)` before Merkle
- [ ] Provenance fields stored in cache entries
- [ ] Same inputs → same Merkle root (reproducibility test)

---

## 5. Phase 3: Code Protection & Mode Awareness

**Goal:** Protect code blocks from boilerplate filtering; respect analysis mode.

### 5.1 Code Block Protection

**File:** `ragix_kernels/docs/doc_extract.py`

Add code fence extraction before boilerplate matching:

```python
# Add to class attributes
CODE_FENCE_PATTERN = re.compile(r'```[\s\S]*?```')
INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')

def _protect_code_blocks(self, text: str) -> tuple[str, list[str], list[str]]:
    """
    Extract and protect code blocks from boilerplate matching.

    Implements MUST M4: Protect fenced/inline code.

    Returns:
        (protected_text, fenced_blocks, inline_codes)
    """
    # Extract fenced code blocks
    fenced_blocks = self.CODE_FENCE_PATTERN.findall(text)
    protected = text
    for i, block in enumerate(fenced_blocks):
        protected = protected.replace(block, f'__FENCED_CODE_{i}__', 1)

    # Extract inline code
    inline_codes = self.INLINE_CODE_PATTERN.findall(protected)
    for i, code in enumerate(inline_codes):
        protected = protected.replace(code, f'__INLINE_CODE_{i}__', 1)

    return protected, fenced_blocks, inline_codes

def _restore_code_blocks(
    self,
    text: str,
    fenced_blocks: list[str],
    inline_codes: list[str]
) -> str:
    """Restore code blocks after boilerplate matching."""
    result = text
    for i, block in enumerate(fenced_blocks):
        result = result.replace(f'__FENCED_CODE_{i}__', block)
    for i, code in enumerate(inline_codes):
        result = result.replace(f'__INLINE_CODE_{i}__', code)
    return result
```

Update `_score_sentence_quality()` to use protection:

```python
def _score_sentence_quality(self, sentence: str) -> float:
    """Score sentence quality, protecting code blocks."""
    config = getattr(self, '_quality_config', None) or QualityConfig()

    # MUST M4: Protect code blocks from boilerplate matching
    protected, fenced, inline = self._protect_code_blocks(sentence)

    score = config.base_score

    # ... existing scoring logic, but operate on 'protected' text
    # for boilerplate pattern matching only

    # Boilerplate detection on protected text (code excluded)
    if self._boilerplate_vocab_pattern is None:
        self._boilerplate_vocab_pattern = self._compile_boilerplate_pattern(config)

    if self._boilerplate_vocab_pattern.search(protected):  # Use protected
        score -= config.boilerplate_penalty

    # ... rest of scoring

    return max(0, min(1, score))
```

### 5.2 Analysis Mode Awareness

**File:** `ragix_kernels/docs/doc_extract.py`

Check analysis type before applying boilerplate filtering:

```python
def compute(self, input: KernelInput) -> Dict[str, Any]:
    """Extract key sentences from documents."""
    # ...

    # MUST NOT N1: Don't apply boilerplate to code/mixed mode
    analysis_type = input.config.get("analysis_type", "docs")
    skip_boilerplate = analysis_type in ("code", "mixed")

    if skip_boilerplate:
        logger.info("[doc_extract] Skipping boilerplate filtering (code/mixed mode)")
        # Set quality config to disable boilerplate penalty
        quality_config.boilerplate_penalty = 0.0
```

### 5.3 Acceptance Criteria

- [ ] Code blocks within ` ``` ` fences are not matched by boilerplate patterns
- [ ] Inline code within `` ` `` is not matched by boilerplate patterns
- [ ] Boilerplate filtering disabled for `--type code` and `--type mixed`
- [ ] Unit test: code block with "Table of Contents" inside is NOT penalized

---

## 6. Phase 4: Audit Trail Enhancement

**Goal:** Log seeds and provenance to audit trail.

### 6.1 Seed Logging

**File:** `ragix_kernels/docs/doc_cluster_leiden.py`

Log seed in kernel output:

```python
def compute(self, input: KernelInput) -> Dict[str, Any]:
    """Execute Leiden community detection."""
    # ...
    config = LeidenConfig(...)

    # MUST M6: Log seed for reproducibility
    logger.info(f"[doc_cluster_leiden] Using seed={config.seed}")

    # Include in output for audit trail
    return {
        "communities": communities,
        "resolution_levels": resolution_data,
        "_audit": {
            "seed": config.seed,
            "algorithm": "leiden",
            "resolutions": config.resolutions,
        }
    }
```

### 6.2 Audit Trail Schema Update

**File:** `ragix_kernels/orchestrator.py` (or audit module)

Extend audit trail to capture seeds:

```python
# Kernel execution entry
kernel_entry = {
    "name": kernel.name,
    "stage": kernel.stage,
    "success": True,
    "execution_time_s": elapsed,
    "input_hash": input_hash,
    "output_hash": output_hash,
    "llm_calls": llm_call_count,
    # NEW: Seed for partial-determinism kernels
    "seed": output.get("_audit", {}).get("seed"),
}
```

### 6.3 Acceptance Criteria

- [ ] `doc_cluster` logs seed in output
- [ ] `doc_cluster_leiden` logs seed in output
- [ ] `partition` kernel logs seed in output
- [ ] Seeds appear in `kernel_execution[].seed` in audit trail
- [ ] Replay test: same seed → same clusters

---

## 7. Verification Checklist

### 7.1 Pre-Release Checklist

| Requirement | Test Command | Expected Result |
|-------------|--------------|-----------------|
| M1: Metadata strip | `ragix-koas run --output-level=external && grep -E "call_hash\|merkle" report.md` | No matches |
| M2: Output-level CLI | `ragix-koas run --help \| grep output-level` | Flag documented |
| M3: call_hash | `jq '.llm_calls[0].call_hash' audit_trail.json` | SHA256 hash |
| M4: Code protection | Manual: code block with "Table of Contents" | Not penalized |
| M5: Canonical ordering | Run twice with same input | Identical Merkle root |
| M6: Seed logging | `jq '.kernel_execution[] \| select(.name=="doc_cluster") \| .seed' audit_trail.json` | Integer (42) |
| N1: No boilerplate for code | `ragix-koas run --type code` | No boilerplate penalty applied |
| N2: No provenance in external | `ragix-koas run --output-level=external && grep "PROVENANCE" report.md` | No matches |
| N3: No content modification | Diff input vs extracted sentence | Identical text |
| N4: Metrics only | `ragix-koas run --output-level=orchestrator` | No excerpts in output |
| N5: Explicit isolation | `ragix-koas run` (no flags) | Warning about implicit mode |

### 7.2 Regression Tests

Create test suite in `tests/test_sovereign_compliance.py`:

```python
import pytest
from ragix_kernels.output_sanitizer import (
    validate_external_report, sanitize_for_level, OutputLevel
)
from ragix_kernels.merkle import compute_inputs_merkle_root

def test_denylist_validation():
    """M2: Denylist keys trigger build failure."""
    # ... test code

def test_merkle_determinism():
    """M5: Same inputs → same Merkle root."""
    children = [
        {"file_path": "b.txt", "chunk_index": 0, "content": "B"},
        {"file_path": "a.txt", "chunk_index": 0, "content": "A"},
    ]
    root1 = compute_inputs_merkle_root(children)
    root2 = compute_inputs_merkle_root(children)
    assert root1 == root2

def test_code_fence_protection():
    """M4: Code blocks protected from boilerplate."""
    # ... test code
```

---

## 8. Impact & Risk of Regression

This section documents the potential impact and regression risks for each implementation phase. Use this for change review and rollback planning.

### 8.1 Phase 1: Output Isolation — HIGH RISK

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Breaking existing workflows** | HIGH | Consumers expecting current output format will fail. Add `--output-level=internal` as explicit default with deprecation warning for implicit behavior. |
| **Over-redaction** | MEDIUM | Regex patterns may redact legitimate content (e.g., file paths in code examples). Requires allowlist for code blocks. |
| **Build failures on denylist** | HIGH | Strict enforcement may block valid builds if internal metadata leaks unintentionally. Add `--strict` vs `--warn` modes. |

**Affected Components:**
- All report generation (`doc_final_report.py`)
- CLI interface (`run_doc_koas.py`)
- Any downstream scripts parsing KOAS output

**Regression Test Priority:**
```bash
# Must pass before merge
ragix-koas run --output-level=internal  # Existing behavior preserved
ragix-koas run --output-level=external  # New: no denylist keys
diff <(ragix-koas run 2>/dev/null) <(ragix-koas run --output-level=internal 2>/dev/null)  # Identical
```

---

### 8.2 Phase 2: Merkle & Provenance — MEDIUM RISK

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Cache invalidation** | MEDIUM | New `call_hash` scheme may invalidate existing caches. Implement cache migration or versioned cache schema. |
| **Performance overhead** | LOW | SHA256 and Merkle computation add ~10-50ms per LLM call. Acceptable for typical corpus sizes. |
| **Non-determinism bugs** | HIGH | If child ordering is not strictly canonical, Merkle roots will vary between runs. Test with shuffled inputs. |

**Affected Components:**
- `ragix_kernels/cache.py` — CacheEntry schema change
- `ragix_kernels/llm_wrapper.py` — Request canonicalization
- All `doc_*` kernels using LLM calls

**Regression Test Priority:**
```bash
# Determinism check
SEED=42 ragix-koas run corpus1/ --cache-dir /tmp/c1
SEED=42 ragix-koas run corpus1/ --cache-dir /tmp/c2
diff /tmp/c1/cache.json /tmp/c2/cache.json  # Must be identical
```

---

### 8.3 Phase 3: Code Protection — LOW RISK

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Regex edge cases** | LOW | Nested code fences, escaped backticks may confuse extractor. Use proper parser or state machine. |
| **False positives in mixed mode** | MEDIUM | Prose inside docstrings may be incorrectly protected. Accept as conservative behavior. |

**Affected Components:**
- `ragix_kernels/docs/doc_extract.py` — `_protect_code_blocks()`
- Boilerplate pattern matching

**Regression Test Priority:**
```bash
# Code block preservation
echo '```python\n# Table of Contents\nprint("hello")\n```' | ragix-extract --stdin
# Should NOT penalize "Table of Contents" inside code fence
```

---

### 8.4 Phase 4: Audit Trail — LOW RISK

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Schema change** | LOW | Adding `seed` field to `kernel_execution[]` is additive. Existing consumers ignore unknown fields. |
| **Missing seeds** | LOW | Kernels without randomness return `seed: null`. Document this. |

**Affected Components:**
- `ragix_kernels/orchestrator.py` — Audit trail generation
- `doc_cluster*.py` kernels

---

### 8.5 Cumulative Regression Risk Matrix

| Change | Files Modified | Breaking? | Rollback Complexity |
|--------|----------------|-----------|---------------------|
| `--output-level` CLI | 2 | **YES** (if default changes) | Low (remove arg) |
| `output_sanitizer.py` | 1 (new) | No | Low (delete file) |
| `merkle.py` | 1 (new) | No | Low (delete file) |
| `CacheEntry` schema | 1 | **YES** (cache compat) | Medium (migration) |
| Code fence protection | 1 | No | Low (revert function) |
| Audit trail seeds | 2 | No | Low (remove field) |

### 8.6 Recommended Rollout Strategy

1. **Feature flags** — Gate new behavior behind `RAGIX_SOVEREIGN_V2=1` env var
2. **Parallel caches** — Store v1 and v2 caches side-by-side during transition
3. **Canary runs** — Test on DOCSET corpus before enabling globally
4. **Version bump** — v0.65.0 enables new behavior, v0.64.x remains stable

### 8.7 Rollback Procedures

**Phase 1 Rollback:**
```bash
# Revert CLI changes
git checkout HEAD~1 -- ragix_kernels/run_doc_koas.py
# Remove sanitizer (if standalone file)
rm ragix_kernels/output_sanitizer.py
```

**Phase 2 Rollback:**
```bash
# Revert cache schema (requires cache wipe or migration)
git checkout HEAD~1 -- ragix_kernels/cache.py
rm -rf ~/.ragix/cache/  # Warning: loses cached LLM calls
```

**Phase 3/4 Rollback:**
```bash
# Simple function reverts, no data impact
git checkout HEAD~1 -- ragix_kernels/docs/doc_extract.py
git checkout HEAD~1 -- ragix_kernels/docs/doc_cluster_leiden.py
```

---

## 9. References

| Document | Description |
|----------|-------------|
| `docs/SOVEREIGN_LLM_OPERATIONS.md` | Normative specification |
| `docs/developer/ROADMAP_DOCS_TRACEABILITY.md` | Related: LLM call traceability |
| `ragix_kernels/cache.py` | Current LLM cache implementation |
| `ragix_kernels/docs/doc_extract.py` | Boilerplate filtering implementation |
| `ragix_kernels/run_doc_koas.py` | CLI argument parsing |

---

## Appendix A: Implementation Diff Summary

| File | Lines Changed | Type |
|------|---------------|------|
| `ragix_kernels/run_doc_koas.py` | +15 | CLI arguments |
| `ragix_kernels/output_sanitizer.py` | +150 | **NEW** |
| `ragix_kernels/merkle.py` | +80 | **NEW** |
| `ragix_kernels/llm_wrapper.py` | +50 | Canonicalization |
| `ragix_kernels/cache.py` | +20 | CacheEntry fields |
| `ragix_kernels/docs/doc_extract.py` | +40 | Code protection |
| `ragix_kernels/docs/doc_cluster_leiden.py` | +10 | Seed logging |
| `ragix_kernels/docs/doc_cluster.py` | +10 | Seed logging |
| `tests/test_sovereign_compliance.py` | +200 | **NEW** |

**Total:** ~575 lines of new/modified code

---

## Appendix B: Priority Justification

**Phase 1 (Critical):** Output isolation is the most visible gap. External deliveries currently risk leaking internal metadata and paths. This is a compliance blocker for client-facing reports.

**Phase 2 (High):** Merkle roots enable true reproducibility and verification. Without them, "LLM-free replay" claims in the documentation are unsupported.

**Phase 3 (Medium):** Code protection affects mixed-mode audits. Important but lower impact than Phases 1-2 since most current use cases are pure-docs mode.

**Phase 4 (Medium):** Audit trail enhancement completes the provenance story but doesn't block core functionality.

---

*RAGIX KOAS-Docs | Adservio Innovation Lab | 2026-01-29*
