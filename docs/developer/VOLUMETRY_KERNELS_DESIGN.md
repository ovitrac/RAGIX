# Design: Volumetry-Aware Audit Kernels

**Date:** 2025-12-16
**Context:** ACME-ERP audit revealed need for production workload analysis
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr

---

## 1. Problem Statement

The current KOAS kernels analyze **code structure** (LOC, complexity, coupling) but don't consider **production workload**. The ACME-ERP analysis showed that:

- A module with 683 LOC can be **more critical** than one with 15,260 LOC if it processes 4M messages/day
- Risk assessment without volumetry misses operational hotspots
- Architecture recommendations require workload patterns (peaks, async opportunities)

**Goal:** Add kernels that integrate operational data with code metrics.

---

## 2. Proposed Kernels

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEW VOLUMETRY-AWARE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: Collection          STAGE 2: Analysis         STAGE 3: Report │
│  ══════════════════          ═════════════════         ════════════════ │
│                                                                          │
│  ┌──────────────┐           ┌──────────────┐          ┌──────────────┐  │
│  │  volumetry   │──────────▶│ risk_matrix  │─────────▶│ section_arch │  │
│  │  (NEW)       │           │ (NEW)        │          │ (NEW)        │  │
│  └──────────────┘           └──────────────┘          └──────────────┘  │
│         │                          │                         │          │
│         │                          │                         │          │
│  ┌──────────────┐           ┌──────────────┐                │          │
│  │ module_group │──────────▶│ code_pattern │────────────────┘          │
│  │  (NEW)       │           │ (NEW)        │                            │
│  └──────────────┘           └──────────────┘                            │
│         │                          │                                    │
│         │                   ┌──────────────┐                            │
│  ┌──────────────┐           │  load_model  │                            │
│  │  ast_scan    │──────────▶│  (NEW)       │                            │
│  │  (existing)  │           └──────────────┘                            │
│  └──────────────┘                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Kernel Specifications

### 3.1 Volumetry Kernel (Stage 1)

**Purpose:** Ingest and normalize operational volumetry data.

```python
class VolumetryKernel(Kernel):
    """
    Ingest operational volumetry data from various sources.

    Inputs:
        - volumetry.yaml/json in workspace/data/
        - Or manual config in manifest

    Outputs:
        - Normalized volumetry per module/flow
        - Peak patterns (hour, day)
        - Throughput requirements (msg/sec)
    """

    name = "volumetry"
    version = "1.0.0"
    category = "audit"
    stage = 1

    requires = []  # No dependencies
    provides = ["volumetry", "peak_patterns", "throughput"]
```

**Input Format (volumetry.yaml):**

```yaml
# Operational data from client
flows:
  - name: "MSG-HUB"
    volume_day: 4_000_000
    unit: "messages"
    peak_hour: 5
    peak_window: "00:00-10:00"
    peak_multiplier: 10  # vs average

  - name: "RGDS"
    volume_day: 10_000
    unit: "files"
    peak_hour: null

modules:
  - name: "acme-msg-hub"
    flows: ["MSG-HUB"]
    role: "entry_point"

  - name: "acme-core-sk04"
    flows: ["RGDS"]
    role: "processor"

incidents:
  - date: "2025-01"
    type: "saturation"
    module: "acme-msg-hub"
    cause: "peak_overflow"
```

**Output:**

```json
{
  "flows": {
    "MSG-HUB": {
      "volume_day": 4000000,
      "volume_sec_avg": 46,
      "volume_sec_peak": 500,
      "peak_hour": 5,
      "normalized_score": 10.0
    }
  },
  "module_volumetry": {
    "acme-msg-hub": {
      "flow": "MSG-HUB",
      "volume_day": 4000000,
      "criticality": "HIGH"
    }
  }
}
```

---

### 3.2 Module Grouper Kernel (Stage 1)

**Purpose:** Group files into functional modules based on path/naming patterns.

```python
class ModuleGrouperKernel(Kernel):
    """
    Group files into functional modules.

    Uses:
        - Path-based patterns (acme-*, iog-*, etc.)
        - Package structure
        - Maven/Gradle module detection

    Dependencies:
        - ast_scan (file list)

    Outputs:
        - Module definitions with file lists
        - Module metrics aggregates (LOC, classes, methods)
    """

    name = "module_group"
    version = "1.0.0"
    category = "audit"
    stage = 1

    requires = ["ast_scan"]
    provides = ["modules", "module_metrics"]
```

**Configuration:**

```yaml
module_group:
  enabled: true
  options:
    patterns:
      # Regex patterns to extract module name from path
      - regex: ".*/([a-z]+-[a-z]+-[a-z0-9]+)/.*"
        group: 1
      - regex: ".*/([a-z]+-[a-z]+-[a-z]+-[a-z0-9]+)/.*"
        group: 1
    # Or explicit mapping
    mapping:
      "acme-msg-hub": "Exchange MSG-HUB"
      "acme-core-sk01": "SC01 Processing"
```

**Output:**

```json
{
  "modules": {
    "acme-msg-hub": {
      "display_name": "Exchange MSG-HUB",
      "files": 20,
      "loc": 2374,
      "classes": 17,
      "methods": 54,
      "file_list": ["MsgHubJmsListener.java", ...]
    }
  },
  "summary": {
    "total_modules": 19,
    "total_files": 806,
    "unassigned_files": 12
  }
}
```

---

### 3.3 Risk Matrix Kernel (Stage 2)

**Purpose:** Combine code metrics + volumetry into weighted risk scores.

```python
class RiskMatrixKernel(Kernel):
    """
    Compute volumetry-aware risk scores.

    Formula:
        Risk = (LOC_norm × w1) + (Complexity_norm × w2) + (Volumetry_norm × w3)

    Dependencies:
        - module_group (module metrics)
        - volumetry (operational data)
        - metrics (complexity data)

    Outputs:
        - Risk matrix (module × risk factors)
        - Risk ranking
        - Critical path identification
    """

    name = "risk_matrix"
    version = "1.0.0"
    category = "audit"
    stage = 2

    requires = ["module_group", "volumetry", "metrics"]
    provides = ["risk_matrix", "risk_ranking", "critical_path"]
```

**Configuration:**

```yaml
risk_matrix:
  enabled: true
  options:
    weights:
      loc: 0.25
      complexity: 0.25
      volumetry: 0.50

    normalization:
      loc_max: 15000      # LOC = 15K → score 10
      complexity_max: 15   # methods/class = 15 → score 10
      volumetry_max: 4_000_000  # 4M/day → score 10

    thresholds:
      critical: 7.0
      high: 5.0
      medium: 3.0
```

**Output:**

```json
{
  "matrix": [
    {
      "module": "acme-support-commons",
      "loc": 11523,
      "loc_norm": 7.7,
      "complexity_norm": 5.8,
      "volumetry": 4000000,
      "volumetry_norm": 10.0,
      "risk_score": 7.9,
      "risk_level": "CRITICAL"
    }
  ],
  "ranking": [
    {"module": "acme-support-commons", "risk": 7.9, "level": "CRITICAL"},
    {"module": "acme-iog-models", "risk": 6.6, "level": "HIGH"}
  ],
  "critical_path": ["acme-msg-hub", "acme-core-sk01", "acme-support-commons"],
  "summary": {
    "critical": 1,
    "high": 4,
    "medium": 1,
    "low": 4
  }
}
```

---

### 3.4 Code Pattern Kernel (Stage 2)

**Purpose:** Categorize code by functional pattern (DTO, Listener, Factory, etc.).

```python
class CodePatternKernel(Kernel):
    """
    Identify code patterns by analyzing naming and structure.

    Patterns detected:
        - DTO/Model: *DTO.java, *Model.java, *Entity.java
        - Listener: *Listener.java, @JmsListener
        - Factory: *Factory.java
        - Utils: *Utils.java, *Helper.java
        - Service: *Service.java, *Impl.java

    Dependencies:
        - ast_scan (class names, annotations)
        - module_group (grouping)

    Outputs:
        - Pattern distribution per module
        - Pattern-specific metrics
    """

    name = "code_pattern"
    version = "1.0.0"
    category = "audit"
    stage = 2

    requires = ["ast_scan", "module_group"]
    provides = ["patterns", "pattern_metrics"]
```

**Pattern Rules:**

```yaml
code_pattern:
  enabled: true
  options:
    patterns:
      dto:
        name_patterns: ["*DTO", "*Model", "*Entity", "*Request", "*Response"]
        annotations: ["@Entity", "@Data", "@Value"]
        expected_methods: ["get*", "set*", "toString", "equals", "hashCode"]
        weight: "light"  # Typically fast to process

      listener:
        name_patterns: ["*Listener", "*Handler", "*Consumer"]
        annotations: ["@JmsListener", "@KafkaListener", "@RabbitListener"]
        weight: "critical"  # Entry points, throughput-sensitive

      factory:
        name_patterns: ["*Factory", "*Builder", "*Creator"]
        weight: "medium"

      utils:
        name_patterns: ["*Utils", "*Helper", "*Util", "*Commons"]
        weight: "heavy"  # Often contains CPU-intensive code

      service:
        name_patterns: ["*Service", "*ServiceImpl", "*Manager"]
        annotations: ["@Service", "@Component"]
        weight: "medium"
```

**Output:**

```json
{
  "by_module": {
    "acme-support-commons": {
      "dto": {"files": 45, "loc": 5200, "percent": 45},
      "utils": {"files": 12, "loc": 2800, "percent": 24},
      "factory": {"files": 8, "loc": 1500, "percent": 13}
    },
    "acme-msg-hub": {
      "listener": {"files": 7, "loc": 1050, "percent": 44},
      "dto": {"files": 5, "loc": 600, "percent": 25}
    }
  },
  "critical_patterns": [
    {"module": "acme-msg-hub", "pattern": "listener", "count": 7, "throughput_risk": "HIGH"},
    {"module": "acme-support-commons", "pattern": "utils", "count": 12, "cpu_risk": "HIGH"}
  ]
}
```

---

### 3.5 Load Model Kernel (Stage 2)

**Purpose:** Estimate processing capacity and identify async opportunities.

```python
class LoadModelKernel(Kernel):
    """
    Model processing load and capacity.

    Estimates:
        - Processing time per message/file
        - Capacity (msg/sec) per component
        - Async vs sync suitability
        - Off-peak processing window

    Dependencies:
        - volumetry (workload data)
        - code_pattern (component types)
        - risk_matrix (critical path)

    Outputs:
        - Load model per module
        - Capacity estimation
        - Async recommendations
    """

    name = "load_model"
    version = "1.0.0"
    category = "audit"
    stage = 2

    requires = ["volumetry", "code_pattern", "risk_matrix"]
    provides = ["load_model", "capacity", "async_recommendations"]
```

**Processing Time Estimates (configurable):**

```yaml
load_model:
  enabled: true
  options:
    # Time estimates by pattern (milliseconds)
    pattern_times:
      jms_receive: 0.2
      deserialize: 0.5
      dto_create: 0.1
      xml_validate: 8.0    # CPU-intensive!
      transform: 4.0
      persist: 2.0
      jms_ack: 0.5

    # Peak configuration
    peak:
      window_hours: 10     # 00:00-10:00
      multiplier: 10       # Peak vs average

    # Off-peak window
    off_peak:
      start_hour: 10
      end_hour: 24
      duration_seconds: 50400  # 14 hours
```

**Output:**

```json
{
  "modules": {
    "acme-msg-hub": {
      "sync_time_ms": 15.2,
      "async_time_ms": 0.9,
      "capacity_sync": 66,
      "capacity_async": 1111,
      "bottleneck": "xml_validate",
      "async_suitable": true
    }
  },
  "recommendations": {
    "async_candidates": [
      {
        "module": "acme-msg-hub",
        "current_capacity": 66,
        "required_capacity": 1000,
        "gap": 934,
        "solution": "async_queue",
        "gain_factor": 16.8
      }
    ],
    "off_peak_processing": {
      "available_seconds": 50400,
      "deferred_volume": 2500000,
      "required_rate": 50,
      "workers_needed": 1
    }
  }
}
```

---

### 3.6 Section Architecture Kernel (Stage 3)

**Purpose:** Generate architecture recommendations section for report.

```python
class SectionArchitectureKernel(Kernel):
    """
    Generate architecture recommendations for audit report.

    Content:
        - Async processing recommendations
        - Circuit breaker implementation
        - Scaling strategy
        - Implementation roadmap

    Dependencies:
        - load_model (capacity analysis)
        - risk_matrix (critical modules)
        - code_pattern (entry points)

    Outputs:
        - Markdown section for report
        - Architecture diagrams (Mermaid)
    """

    name = "section_architecture"
    version = "1.0.0"
    category = "audit"
    stage = 3

    requires = ["load_model", "risk_matrix", "code_pattern"]
    provides = ["architecture_section", "architecture_diagrams"]
```

---

## 4. Manifest Integration

```yaml
# manifest.yaml additions

# Stage 1: Collection
stage1:
  ast_scan:
    enabled: true

  volumetry:
    enabled: true
    options:
      data_file: "data/volumetry.yaml"

  module_group:
    enabled: true
    options:
      patterns:
        - regex: ".*/([a-z]+-[a-z]+-[a-z0-9]+)/.*"

# Stage 2: Analysis
stage2:
  metrics:
    enabled: true

  risk_matrix:
    enabled: true
    options:
      weights:
        loc: 0.25
        complexity: 0.25
        volumetry: 0.50

  code_pattern:
    enabled: true

  load_model:
    enabled: true

# Stage 3: Reporting
stage3:
  sections:
    - id: "architecture"
      title: "Architecture Recommendations"
      kernel: "section_architecture"
```

---

## 5. Implementation Plan

### Phase 1: Core Kernels (Week 1)

| Kernel | Effort | Priority |
|--------|--------|----------|
| `volumetry` | 2d | P1 |
| `module_group` | 2d | P1 |
| `risk_matrix` | 2d | P1 |

### Phase 2: Pattern Analysis (Week 2)

| Kernel | Effort | Priority |
|--------|--------|----------|
| `code_pattern` | 3d | P2 |
| `load_model` | 3d | P2 |

### Phase 3: Reporting (Week 3)

| Kernel | Effort | Priority |
|--------|--------|----------|
| `section_architecture` | 2d | P2 |
| Integration tests | 2d | P2 |
| Documentation | 1d | P3 |

**Total estimated effort:** 3 weeks / 1 FTE

---

## 6. Usage Example

```bash
# 1. Initialize workspace with volumetry
ragix-koas init \
  --workspace ./audit/acme-erp \
  --project /path/to/acme-erp \
  --name "ACME-ERP"

# 2. Add volumetry data
cat > ./audit/acme-erp/data/volumetry.yaml << 'EOF'
flows:
  - name: MSG-HUB
    volume_day: 4000000
    peak_hour: 5
    peak_window: "00:00-10:00"
modules:
  - name: acme-msg-hub
    flows: [MSG-HUB]
EOF

# 3. Run analysis
ragix-koas run --workspace ./audit/acme-erp --all --parallel

# 4. View results
cat ./audit/acme-erp/stage2/risk_matrix.json
cat ./audit/acme-erp/stage2/load_model.json
```

---

## 7. Benefits

### For ACME-ERP-like Audits

- **Automated risk matrix** with volumetry weighting
- **Capacity estimation** without manual calculation
- **Async recommendations** generated from analysis
- **Reproducible** — same input → same recommendations

### For KOAS Philosophy

- Kernels remain **pure computation** (no LLM inside)
- Clear **dependency chain** (volumetry → risk_matrix → load_model → section)
- **Composable** — can run individual kernels or full pipeline
- **Auditable** — full trace of calculations

---

## 8. Future Extensions

### 8.1 Git Integration

```yaml
volumetry:
  sources:
    - type: "git_commits"
      metric: "churn_rate"
    - type: "ci_logs"
      metric: "build_frequency"
```

### 8.2 Monitoring Integration

```yaml
volumetry:
  sources:
    - type: "prometheus"
      endpoint: "http://prometheus:9090"
      query: "sum(rate(http_requests_total[5m]))"
```

### 8.3 Architecture Patterns Library

```yaml
load_model:
  patterns:
    async_queue:
      template: "patterns/async_queue.md"
      diagrams: "patterns/async_queue.mermaid"
    circuit_breaker:
      template: "patterns/circuit_breaker.md"
```

---

*Design document for RAGIX v0.7x — Volumetry-Aware Audit Kernels*
