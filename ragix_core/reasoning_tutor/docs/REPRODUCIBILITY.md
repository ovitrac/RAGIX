# Reproducibility Protocol

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.1.0 (2025-12-27)
**Status:** Specification (implementation pending)

---

## Overview

This document defines the reproducibility protocol for the Interpreter-Tutor system, ensuring that:

1. Every run produces an auditable **run manifest**
2. Baseline results can be **exactly reproduced**
3. Envelope configurations are **fully traceable**
4. Golden trace tests **guarantee backward compatibility**

---

## Run Manifest

### Purpose

Every benchmark run writes a JSON manifest file containing all information needed to reproduce the exact run conditions.

### Manifest Schema

```json
{
  "manifest_version": "1.0.0",
  "run_id": "uuid-v4",

  "environment": {
    "timestamp_start": "2025-12-27T10:00:00Z",
    "timestamp_end": "2025-12-27T10:05:30Z",
    "duration_seconds": 330.5,
    "python_version": "3.12.0",
    "platform": "linux-x86_64",
    "hostname": "workstation-01"
  },

  "repository": {
    "git_commit": "abc123def456...",
    "git_branch": "main",
    "git_dirty": false,
    "git_tags": ["v0.62.0"]
  },

  "tutor": {
    "api_version": "0.1.0",
    "tutor_file_hash": "sha256:...",
    "envelope_enabled": false,
    "envelope_config": null
  },

  "model": {
    "model_id": "granite3.1-moe:3b",
    "provider": "ollama",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
    "seed": null
  },

  "benchmark": {
    "benchmark_ids": ["01", "02", "03", "04", "05", "06"],
    "benchmark_version": "1.0.0",
    "benchmark_files_hash": "sha256:...",
    "round": 4
  },

  "policy": {
    "bundle_enabled": false,
    "bundle_path": null,
    "bundle_hash": null,
    "compiled_by": null,
    "compiled_at": null
  },

  "outputs": {
    "jsonl_log": "results/olympics_round4.jsonl",
    "csv_features": "results/olympics_round4_features.csv",
    "manifest_file": "results/olympics_round4_manifest.json"
  },

  "checksums": {
    "jsonl_log_hash": "sha256:...",
    "csv_features_hash": "sha256:..."
  }
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `manifest_version` | string | Manifest schema version |
| `run_id` | string | Unique run identifier (UUID v4) |
| `git_commit` | string | Full commit hash |
| `git_dirty` | bool | True if uncommitted changes |
| `tutor.api_version` | string | `TUTOR_API_VERSION` constant |
| `model.temperature` | float | Must be 0.0 for determinism |
| `benchmark_files_hash` | string | Hash of benchmark YAML files |

### Optional Fields (Envelope Mode)

When `TutorEnvelope` is enabled:

```json
{
  "tutor": {
    "envelope_enabled": true,
    "envelope_config": {
      "layer1_schema": "strict",
      "layer2_repair": "bestofn",
      "repair_max_attempts": 3,
      "repair_trigger_verdicts": ["ILL_TYPED", "UNDECIDABLE"],
      "repair_temperature": 0.0,
      "layer3_policy": "bundle",
      "policy_bundle_path": "policies/round4_optimized.yaml"
    }
  },
  "repair_stats": {
    "total_retries": 12,
    "retries_per_turn": [0, 0, 2, 1, 0, 3, ...]
  }
}
```

---

## Deterministic Execution

### Temperature Control

For reproducibility, all runs must use `temperature=0.0`:

```python
# In benchmark runner
response = ollama.chat(
    model=model_id,
    messages=messages,
    options={
        "temperature": 0.0,  # REQUIRED for reproducibility
        "top_p": 1.0,
        "seed": None  # Ollama handles determinism internally
    }
)
```

### Verified Determinism

From `REPRODUCIBILITY_REPORT.md`:

| Benchmark | Model | Run 1 | Run 2 | Run 3 | Std Dev |
|-----------|-------|-------|-------|-------|---------|
| B01 | qwen2.5-coder:14b | 1 turn | 1 turn | 1 turn | 0.0 |
| B02 | granite3.1-moe:3b | 1 turn | 1 turn | 1 turn | 0.0 |
| B02 | qwen2.5-coder:7b | 1 turn | 1 turn | 1 turn | 0.0 |

**Conclusion:** With `temperature=0.0`, Ollama produces identical outputs across runs.

### Seed Policy

- **Default:** No explicit seed (Ollama's deterministic mode)
- **If temp > 0:** Require explicit seed in manifest
- **Layer 2 repair:** Must use `repair_temperature=0.0` by default

---

## Golden Trace Tests

### Definition

A **golden trace** is a per-turn deterministic record including:

| Field | Source | Purpose |
|-------|--------|---------|
| `prompt_text` | LLM input | Verify prompt generation |
| `model_output` | LLM response | Verify parsing input |
| `parsed_move` | `parse_move()` | Verify parsing logic |
| `check_verdict` | `Tutor.check()` | Verify CHECK semantics |
| `move_verdict` | `Tutor.execute_move()` | Verify move validation |
| `score_delta` | `GameState.score` | Verify scoring rules |
| `state_hash` | `PCG` serialization | Verify state consistency |
| `routing_decision` | `StrategicAdvisor` | Verify routing logic |

### Golden Trace Format

```jsonl
{"turn": 1, "prompt_hash": "sha256:abc...", "output_hash": "sha256:def...", "move_type": "PROPOSE", "check_verdict": "PROVABLE", "move_verdict": "LEGAL", "score_delta": 50, "state_hash": "sha256:ghi..."}
{"turn": 2, "prompt_hash": "sha256:jkl...", "output_hash": "sha256:mno...", "move_type": "ASSERT", "check_verdict": "UNDECIDABLE", "move_verdict": "LEGAL", "score_delta": 0, "state_hash": "sha256:pqr..."}
```

### Golden Trace Generation

```bash
# Generate golden traces for baseline
cd ragix_core/reasoning_tutor
python -m benchmarks.golden_trace_generator \
    --benchmarks "01,02,03" \
    --model "granite3.1-moe:3b" \
    --output tests/golden/
```

### Compatibility Test

```python
# tests/test_compatibility_golden_trace.py

def test_envelope_off_equals_baseline():
    """TutorEnvelope with all layers off must match TutorV1 exactly."""

    # Load golden trace
    golden = load_golden_trace("tests/golden/baseline_granite3b.jsonl")

    # Run with baseline Tutor
    baseline_trace = run_benchmark(tutor=Tutor(), ...)

    # Run with envelope (all off)
    envelope_trace = run_benchmark(
        tutor=TutorEnvelope(
            base_tutor=Tutor(),
            layer1_schema="off",
            layer2_repair="off",
            layer3_policy="off"
        ),
        ...
    )

    # Assert equivalence
    assert baseline_trace == golden, "Baseline changed from golden"
    assert envelope_trace == golden, "Envelope(off) differs from baseline"
```

### CI Integration

```yaml
# .github/workflows/compatibility.yml
name: Golden Trace Compatibility

on: [push, pull_request]

jobs:
  compatibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run golden trace tests
        run: |
          pytest tests/test_compatibility_golden_trace.py -v
```

---

## Reproducing Paper Results

### Round 4 Reproduction

```bash
# 1. Checkout exact commit
git checkout v0.62.0

# 2. Install dependencies
pip install -r requirements.txt
pip install -r ragix_core/reasoning_tutor/requirements.txt

# 3. Verify Ollama models
ollama list | grep -E "granite3.1-moe|llama3.2|mistral|deepseek"

# 4. Run Round 4 benchmark
cd ragix_core/reasoning_tutor
python -m benchmarks.scored_mode \
    --round 4 \
    --models "granite3.1-moe:3b,llama3.2:3b,mistral:latest,deepseek-r1:14b" \
    --temperature 0.0 \
    --output results/olympics_round4_reproduction.jsonl

# 5. Compare with published results
python -m analysis.compare_traces \
    --golden results/olympics_round4_features.csv \
    --reproduced results/olympics_round4_reproduction_features.csv
```

### Expected Output

```
Reproduction Check:
  - Win rates: MATCH
  - Score distributions: MATCH
  - Failure counts: MATCH
  - Turn counts: MATCH (±0 with temp=0.0)

Reproduction status: SUCCESS
```

---

## Version Tracking

### TUTOR_API_VERSION

Add to `tutor.py`:

```python
# At module level
TUTOR_API_VERSION = "0.1.0"

# In Tutor class
class Tutor:
    VERSION = TUTOR_API_VERSION
    ...
```

### Version Semantics

| Version Change | Meaning |
|----------------|---------|
| Patch (0.1.x) | Bug fixes, no behavior change |
| Minor (0.x.0) | New features, backward compatible |
| Major (x.0.0) | Breaking changes, new golden traces required |

### Deprecation Policy

When `Tutor` behavior changes:

1. Increment `TUTOR_API_VERSION`
2. Generate new golden traces
3. Document changes in CHANGELOG
4. Update manifest schema if needed

---

## Manifest Writer Implementation

```python
# manifest.py

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import platform

@dataclass
class RunManifest:
    """Run manifest for reproducibility."""

    manifest_version: str = "1.0.0"
    run_id: str = ""

    # Environment
    timestamp_start: str = ""
    timestamp_end: str = ""
    python_version: str = ""
    platform_info: str = ""

    # Repository
    git_commit: str = ""
    git_dirty: bool = False

    # Tutor
    tutor_api_version: str = ""
    envelope_enabled: bool = False
    envelope_config: Optional[dict] = None

    # Model
    model_id: str = ""
    temperature: float = 0.0

    # Benchmark
    benchmark_ids: list = None
    round_number: int = 0

    # Policy
    policy_bundle_path: Optional[str] = None
    policy_bundle_hash: Optional[str] = None

    @classmethod
    def create(cls, **kwargs) -> "RunManifest":
        """Create manifest with auto-populated environment info."""
        manifest = cls(**kwargs)
        manifest.timestamp_start = datetime.utcnow().isoformat() + "Z"
        manifest.python_version = platform.python_version()
        manifest.platform_info = platform.platform()
        manifest.git_commit = cls._get_git_commit()
        manifest.git_dirty = cls._is_git_dirty()
        return manifest

    @staticmethod
    def _get_git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            return "unknown"

    @staticmethod
    def _is_git_dirty() -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except:
            return True

    def finalize(self, output_files: list[str]) -> None:
        """Finalize manifest with end time and output checksums."""
        self.timestamp_end = datetime.utcnow().isoformat() + "Z"
        # Add checksums for output files...

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
```

---

## References

- `SAFETY_ENVELOPE.md` — Envelope architecture
- `DSPY_INTEGRATION.md` — DSPy integration guide
- `benchmarks/scored_mode.py` — Main benchmark runner
- `results/` — Published results directory

---

*RAGIX Interpreter-Tutor Reproducibility Protocol v0.1.0*
