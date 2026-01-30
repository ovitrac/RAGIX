# TODO: Implement N1 — Mode-Aware Boilerplate Filtering

**Requirement:** MUST NOT N1 from `docs/SOVEREIGN_LLM_OPERATIONS.md`
**Status:** ⚠️ Partial — Integration pending
**Priority:** Medium
**Target:** v0.65.1

---

## Requirement

> **MUST NOT apply boilerplate cleaning to code or Code+Docs mode** — Boilerplate detection is for **pure document corpora only**. In mixed mode, code files bypass boilerplate filtering entirely.

## Current State

- Code fence protection is implemented (`doc_extract.py:_protect_code_blocks()`)
- But the **analysis mode check** is not yet integrated
- Currently, boilerplate filtering applies to all files regardless of `--type` flag

## Implementation Plan

### 1. Add `--type` CLI argument

**File:** `ragix_kernels/run_doc_koas.py`

```python
run_parser.add_argument(
    "--type",
    choices=["docs", "code", "mixed"],
    default="docs",
    help="Analysis type: docs (pure documents), code (source code), mixed (both)"
)
```

### 2. Pass analysis_type to kernel config

**File:** `ragix_kernels/run_doc_koas.py` (in `cmd_run`)

```python
config["analysis_type"] = args.type
```

### 3. Check mode in doc_extract

**File:** `ragix_kernels/docs/doc_extract.py` (in `compute()`)

```python
# After loading quality_config
analysis_type = input.config.get("analysis_type", "docs")
skip_boilerplate = analysis_type in ("code", "mixed")

if skip_boilerplate:
    logger.info("[doc_extract] Skipping boilerplate filtering (code/mixed mode)")
    quality_config.boilerplate_penalty = 0.0
    quality_config.formatting_penalty = 0.0
```

### 4. Add file-level detection (optional enhancement)

For mixed mode, detect file type and skip boilerplate for code files:

```python
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h"}

def _is_code_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in CODE_EXTENSIONS
```

## Testing

```bash
# Pure docs mode (boilerplate filtering ON)
python -m ragix_kernels.run_doc_koas run -w ./docs_only --all --type docs

# Code mode (boilerplate filtering OFF)
python -m ragix_kernels.run_doc_koas run -w ./code_repo --all --type code

# Mixed mode (boilerplate filtering OFF)
python -m ragix_kernels.run_doc_koas run -w ./mixed_project --all --type mixed
```

## Acceptance Criteria

- [ ] `--type` CLI argument added
- [ ] `--type docs` applies boilerplate penalties (current behavior)
- [ ] `--type code` disables boilerplate penalties
- [ ] `--type mixed` disables boilerplate penalties
- [ ] Log message indicates when boilerplate filtering is skipped
- [ ] Unit test verifies mode-aware behavior

## Files to Modify

| File | Change |
|------|--------|
| `ragix_kernels/run_doc_koas.py` | Add `--type` argument, pass to config |
| `ragix_kernels/docs/doc_extract.py` | Check `analysis_type`, conditionally disable penalties |
| `tests/test_sovereign_compliance.py` | Add mode-aware test cases |

---

*Created: 2026-01-30*
*Related: ROADMAP_SOVEREIGN_COMPLIANCE.md, SOVEREIGN_LLM_OPERATIONS.md*
