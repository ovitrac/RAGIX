# Code Audit Kernel Examples

This directory contains example workspaces demonstrating the RAGIX Code Audit kernels,
based on real-world patterns from IOWIZME/SIAS enterprise architecture.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17

---

## Architecture Context: IOWIZME/SIAS

The examples are based on a real enterprise integration platform:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IOWIZME Platform                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │ iow-ech  │───▶│ iow-ioc  │───▶│ iow-iok  │───▶│ iog-support│    │
│   │  -sias   │    │  -sc02   │    │  -sk01   │    │  -commons  │    │
│   │          │    │          │    │          │    │            │    │
│   │ Gateway  │    │ Orchestr │    │ Business │    │  Shared    │    │
│   │ 4M/day   │    │ 4M/day   │    │ 4M/day   │    │  Libs      │    │
│   └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│        │                                                ▲            │
│        │         ┌──────────┐                          │            │
│        └────────▶│ iow-iog  │──────────────────────────┘            │
│                  │ -models  │                                        │
│                  │  DTOs    │                                        │
│                  └──────────┘                                        │
│                                                                      │
│   Peak: 05:00 UTC | 500-1000 msg/sec | 4M SIAS messages/day         │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Metrics:**
- **4,000,000** SIAS messages/day through gateway
- **Peak hours:** 05:00-07:00 UTC
- **Processing rate:** 500-1000 messages/second at peak
- **Critical path:** iow-ech-sias → iow-ioc-sc02 → iow-iok-sk01

---

## Available Examples

| Example | Description | Kernels Used |
|---------|-------------|--------------|
| `java_monolith/` | Large Java codebase analysis | file_inventory, code_metrics |
| `microservices/` | Multi-module microservices | module_group, dependency analysis |
| `volumetry_analysis/` | Traffic-weighted risk assessment | volumetry, risk_matrix |
| `full_audit/` | Complete IOWIZME-style audit | All audit kernels |

---

## Quick Start

### Interactive Demo

```bash
cd examples/audit
./run_audit_demo.sh

# The interactive menu offers:
# 1) Volumetry Analysis    - Risk weighted by traffic volume
# 2) Microservices         - Service catalog & dependencies
# 3) Java Monolith         - Complexity & refactoring
# 4) Full Audit            - Comprehensive system analysis
# 5) Custom Options        - Configuration parameters
```

### Command Line Options

```bash
# Run specific demos directly
./run_audit_demo.sh --volumetry      # or -1
./run_audit_demo.sh --microservices  # or -2
./run_audit_demo.sh --monolith       # or -3
./run_audit_demo.sh --full           # or -4
./run_audit_demo.sh --custom         # or -5

# Run all demos sequentially
./run_audit_demo.sh --all            # or -a

# Check prerequisites
./run_audit_demo.sh --check          # or -c

# Show help
./run_audit_demo.sh --help           # or -h
```

---

## Example 1: Volumetry Analysis

Analyze code risk weighted by operational traffic volume.

```bash
cd examples/audit/volumetry_analysis

# View sample volumetry data
cat data/volumetry.yaml

# Run analysis
python -m ragix_kernels.orchestrator run -w . -s 1
python -m ragix_kernels.orchestrator run -w . -s 2

# View risk matrix
cat stage2/risk_matrix.json | python -m json.tool
```

**Sample Output:**
```
Module                Risk   Level    LOC    Vol/day   Primary Factor
──────────────────────────────────────────────────────────────────────
iog-support-commons   8.2    CRIT    13,290  4,011,500  volumetry
iow-ech-sias          6.8    HIGH     1,430  4,000,000  volumetry
iow-iog-models        6.5    HIGH     4,500  4,011,500  volumetry
iow-ioc-sc02          6.2    HIGH     1,570  4,000,000  volumetry
iow-iok-sk01          5.7    HIGH       990  4,000,000  volumetry
```

---

## Example 2: Microservices Analysis

Analyze a microservices architecture with module grouping.

```bash
cd examples/audit/microservices

# Run module grouping
python -m ragix_kernels.orchestrator run -w . -s 1 -k module_group

# View grouped metrics
cat stage1/module_group.json | python -m json.tool
```

---

## Example 3: Full Audit Pipeline

Complete audit with all stages.

```bash
cd examples/audit/full_audit

# Run all stages
python -m ragix_kernels.orchestrator run -w .

# View final report
cat stage3/architecture_report.md
```

---

## Volumetry Data Format

The `volumetry.yaml` file defines operational traffic:

```yaml
# Flow definitions
flows:
  - name: "SIAS Gateway"
    source: "external"
    target: "iow-ech-sias"
    volume_per_day: 4000000
    peak_hour: 5
    peak_multiplier: 3.5
    description: "SIAS message ingestion"

# Batch processing
batches:
  - name: "Daily Report Generation"
    module: "iow-iok-sk04"
    schedule: "0 2 * * *"
    volume_per_run: 50000
    duration_minutes: 45

# Incidents (affect risk scoring)
incidents:
  - date: "2024-11-15"
    module: "iow-ech-sias"
    severity: "high"
    description: "Gateway timeout under peak load"
```

---

## Risk Matrix Calculation

Risk is calculated using weighted factors:

```
Risk Score = (LOC × W_loc) + (Complexity × W_cx) + (Volumetry × W_vol)

Where:
- W_loc = 0.25 (code size weight)
- W_cx  = 0.25 (complexity weight)
- W_vol = 0.50 (volumetry weight - primary factor)
```

**Risk Levels:**
| Score | Level | Action |
|-------|-------|--------|
| ≥ 8.0 | CRITICAL | Immediate attention required |
| ≥ 6.0 | HIGH | Priority refactoring |
| ≥ 4.0 | MEDIUM | Monitor closely |
| < 4.0 | LOW | Standard maintenance |

---

## Integration with Code Analysis

For real codebase analysis, point to your source directory:

```yaml
# In manifest.yaml
file_inventory:
  options:
    source_dirs:
      - "/path/to/your/java/project"
    include_patterns:
      - "**/*.java"
    exclude_patterns:
      - "**/test/**"
      - "**/target/**"
```

---

## Output Files

```
audit_workspace/
├── data/
│   ├── volumetry.yaml       # Operational traffic data
│   └── modules.yaml         # Module definitions (optional)
├── manifest.yaml            # Kernel configuration
├── stage1/
│   ├── file_inventory.json  # File listing
│   ├── volumetry.json       # Parsed volumetry
│   └── module_group.json    # Module grouping
├── stage2/
│   ├── code_metrics.json    # LOC, complexity
│   └── risk_matrix.json     # Weighted risk scores
└── stage3/
    └── architecture_report.md
```

---

## Extending the Analysis

### Adding Custom Volumetry

Create `data/volumetry.yaml` with your traffic patterns:

```yaml
flows:
  - name: "API Gateway"
    source: "clients"
    target: "api-service"
    volume_per_day: 1000000
    peak_hour: 14
    peak_multiplier: 2.5

  - name: "Database Writes"
    source: "api-service"
    target: "db-writer"
    volume_per_day: 500000
```

### Custom Module Patterns

For non-standard directory structures:

```yaml
# In manifest.yaml
module_group:
  options:
    patterns:
      - "src/main/java/com/company/([^/]+)/.*"
      - "modules/([^/]+)/src/.*"
```

---

## References

- [KOAS Documentation](../../docs/KOAS.md)
- [Volumetry Kernels Design](../../docs/developer/VOLUMETRY_KERNELS_DESIGN.md)
- [Risk Matrix Methodology](../../docs/developer/RISK_MATRIX_METHODOLOGY.md)
