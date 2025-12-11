# TODO — RAGIX v0.5 Professional Code Audit

**Version:** v0.5.0
**Updated:** 2025-12-10
**Reference:** `FORREVIEW_AUDIT.md`, `PLAN_v0.40_AUDIT.md`
**Target:** Industrial-grade audit for legacy Java applications (IOWIZME, SIAS)

---

## Executive Summary

RAGIX v0.5 focuses on **quantitative code audit** for enterprise Java applications with 15+ years of evolution. The primary goals are:

1. **Dead code detection** — Identify unreachable code across multiple entry points
2. **Descriptive statistics** — Full distributional analysis (quartiles, entropy, skewness)
3. **Coupling metrics** — Martin's instability/abstractness model (Ca, Ce, I, A, D)
4. **MCO estimation** — Maintenance cost model per component (SK/SC/SG)
5. **Multi-layer analysis** — Support for presentation/service/domain/infra architectures

**Business Context:**
Companies like GRDF spend significant budget on code reviews by consultants (Cap Gemini, etc.). RAGIX v0.5 aims to provide automated, reproducible analysis that engineers can run independently.

---

## Phase 1: Descriptive Statistics Engine (~12h)

**Rationale:** Current reports show only averages. A physicist/engineer needs distributions, not point estimates.

### 1.1 Complexity Distribution Statistics (4h)

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Quartiles** | Q1, Q2 (median), Q3 | Identify spread and outliers |
| **IQR** | Q3 - Q1 | Detect high-variance components |
| **Standard Deviation** | $\sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{n}}$ | Measure dispersion |
| **Coefficient of Variation** | $CV = \sigma / \mu$ | Normalized dispersion (compare across components) |
| **Skewness** | $\gamma_1 = \frac{\sum(x_i - \mu)^3}{n \sigma^3}$ | Positive = long tail of complex methods |
| **Kurtosis** | $\gamma_2 = \frac{\sum(x_i - \mu)^4}{n \sigma^4} - 3$ | Heavy tails = extreme outliers |

**Implementation:**
```python
# ragix_audit/statistics.py
@dataclass
class DistributionStats:
    count: int
    mean: float
    std: float
    min: float
    q1: float      # 25th percentile
    median: float  # 50th percentile
    q3: float      # 75th percentile
    max: float
    iqr: float     # Interquartile range
    cv: float      # Coefficient of variation
    skewness: float
    kurtosis: float

    @classmethod
    def from_values(cls, values: List[float]) -> "DistributionStats":
        """Compute all statistics from raw values."""
        ...
```

**Output:** JSON + histogram data for D3.js visualization

### 1.2 Per-Component Statistics (4h)

For each SK/SC/SG component:
- LOC distribution (min, Q1, median, Q3, max)
- Complexity distribution (same)
- Method count distribution
- File size variance

**Report Section:**
```
┌────────────────────────────────────────────────────────────┐
│ Component Statistics — SK04 (194 files)                    │
├────────────────────────────────────────────────────────────┤
│ Metric         │  Min │   Q1 │  Med │   Q3 │  Max │   σ   │
├────────────────┼──────┼──────┼──────┼──────┼──────┼───────┤
│ LOC/file       │   12 │   85 │  142 │  287 │ 1,847│  234  │
│ CC/method      │    1 │    1 │    2 │    4 │   47 │  4.2  │
│ Methods/class  │    1 │    4 │    8 │   15 │   89 │ 12.3  │
└────────────────────────────────────────────────────────────┘
```

### 1.3 Global Codebase Statistics (4h)

- **File size distribution** — Identify monolithic files
- **Package size distribution** — Identify bloated packages
- **Dependency count distribution** — Spot hub classes
- **Age distribution** (from file mtimes) — Correlate age with complexity

**Box Plot Data Export:**
```json
{
  "complexity": {
    "whisker_low": 1,
    "q1": 2,
    "median": 3,
    "q3": 6,
    "whisker_high": 15,
    "outliers": [23, 31, 47]
  }
}
```

---

## Phase 2: Entropy Metrics (~10h)

**Rationale:** Entropy measures uniformity of distribution. Low entropy = concentrated risk. High entropy = well-distributed (or fragmented).

### 2.1 Structural Entropy (4h)

**Definition:**
$$H_s = -\sum_{i=1}^{n} p_i \log_2 p_i$$

Where $p_i$ = proportion of code (LOC or files) in component $i$.

| Entropy Level | Interpretation |
|---------------|----------------|
| $H_s < 2.0$ | Highly concentrated (monolith risk) |
| $2.0 ≤ H_s ≤ 4.0$ | Moderate distribution |
| $H_s > 4.0$ | Well-distributed (check for over-fragmentation) |

**Implementation:**
```python
# ragix_audit/entropy.py
def structural_entropy(component_sizes: Dict[str, int]) -> float:
    """
    Compute Shannon entropy of code distribution across components.

    Args:
        component_sizes: {component_id: LOC or file_count}

    Returns:
        Entropy in bits (0 = all in one component, log2(n) = uniform)
    """
    total = sum(component_sizes.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for size in component_sizes.values():
        if size > 0:
            p = size / total
            entropy -= p * math.log2(p)

    return entropy
```

### 2.2 Complexity Entropy (3h)

**Definition:** How uniformly is complexity distributed across files?

$$H_c = -\sum_{i=1}^{n} \frac{CC_i}{CC_{total}} \log_2 \frac{CC_i}{CC_{total}}$$

- **Low $H_c$** → Complexity concentrated in few files (hotspots)
- **High $H_c$** → Complexity spread evenly (good)

### 2.3 Coupling Entropy (3h)

**Definition:** How uniformly are dependencies distributed?

$$H_d = -\sum_{i=1}^{n} \frac{deg_i}{deg_{total}} \log_2 \frac{deg_i}{deg_{total}}$$

Where $deg_i$ = in-degree + out-degree of node $i$.

- **Low $H_d$** → Hub-and-spoke architecture (fragile)
- **High $H_d$** → Mesh architecture (resilient but complex)

**Report Section:**
```
┌─────────────────────────────────────────────────────────┐
│ Entropy Analysis                                        │
├─────────────────────────────────────────────────────────┤
│ Structural Entropy (LOC):     3.42 bits (moderate)      │
│ Complexity Entropy:           2.87 bits (concentrated)  │
│ Coupling Entropy:             4.12 bits (distributed)   │
│ Max possible entropy:         4.09 bits (17 components) │
│                                                         │
│ Interpretation:                                         │
│ - Code is moderately distributed across components      │
│ - Complexity is concentrated in fewer files             │
│ - Dependencies are well-distributed (no major hubs)     │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 3: Coupling & Instability Metrics (~14h)

**Reference:** Robert C. Martin's *Agile Software Development* (2002)

### 3.1 Afferent/Efferent Coupling (4h)

| Metric | Definition | Formula |
|--------|------------|---------|
| **Ca (Afferent)** | Incoming dependencies | # classes outside that depend on this package |
| **Ce (Efferent)** | Outgoing dependencies | # classes outside that this package depends on |

**Implementation:**
```python
# ragix_audit/coupling.py
@dataclass
class PackageCoupling:
    package: str
    ca: int          # Afferent coupling (incoming)
    ce: int          # Efferent coupling (outgoing)
    internal: int    # Internal dependencies (within package)

    @property
    def total_coupling(self) -> int:
        return self.ca + self.ce
```

### 3.2 Instability Index (3h)

**Definition:**
$$I = \frac{C_e}{C_a + C_e}$$

| Instability | Interpretation |
|-------------|----------------|
| $I = 0$ | Maximally stable (many dependents, no dependencies) |
| $I = 1$ | Maximally unstable (no dependents, many dependencies) |
| $I ≈ 0.5$ | Balanced |

**Stable Dependency Principle (SDP):**
Dependencies should flow *toward* stability. A package with $I = 0.2$ should not depend on a package with $I = 0.8$.

### 3.3 Abstractness (3h)

**Definition:**
$$A = \frac{N_{abstract}}{N_{total}}$$

Where:
- $N_{abstract}$ = # interfaces + # abstract classes
- $N_{total}$ = total # classes in package

| Abstractness | Interpretation |
|--------------|----------------|
| $A = 0$ | Concrete package (implementation) |
| $A = 1$ | Fully abstract (interfaces only) |

### 3.4 Distance from Main Sequence (4h)

**Definition:**
$$D = |A + I - 1|$$

The "Main Sequence" is the line $A + I = 1$ in the A-I plane.

| Zone | Condition | Risk |
|------|-----------|------|
| **Zone of Pain** | $A ≈ 0, I ≈ 0$ | Concrete + stable = rigid, hard to extend |
| **Zone of Uselessness** | $A ≈ 1, I ≈ 1$ | Abstract + unstable = unused abstractions |
| **Main Sequence** | $D ≈ 0$ | Balanced design |

**Visualization:** Scatter plot with packages as dots, color by D value.

**Report Section:**
```
┌───────────────────────────────────────────────────────────────────┐
│ Coupling Analysis — Top 10 Packages                               │
├───────────────────────────────────────────────────────────────────┤
│ Package              │  Ca │  Ce │   I   │   A   │   D   │ Zone  │
├──────────────────────┼─────┼─────┼───────┼───────┼───────┼───────┤
│ fr.iowizmi.iok.sk04  │  12 │  47 │  0.80 │  0.05 │  0.15 │ OK    │
│ fr.iowizmi.iok.sk02  │   8 │  23 │  0.74 │  0.12 │  0.14 │ OK    │
│ fr.iowizmi.domain    │  34 │   5 │  0.13 │  0.08 │  0.79 │ PAIN  │
│ fr.iowizmi.api       │  45 │   2 │  0.04 │  0.85 │  0.11 │ OK    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Phase 4: Dead Code Detection (~12h)

**Target:** Legacy applications with 15+ years of evolution have significant dead code.

### 4.1 Entry Point Discovery (4h)

Detect entry points automatically:
- **Spring annotations:** `@RestController`, `@Controller`, `@Service`, `@Component`
- **Main methods:** `public static void main(String[])`
- **JMS listeners:** `@JmsListener`, `MessageListener` implementations
- **Scheduled tasks:** `@Scheduled`, `Quartz` jobs
- **Servlet mappings:** `web.xml`, `@WebServlet`

**Configuration override:**
```yaml
# ragix_audit_config.yaml
entry_points:
  include:
    - "fr.iowizmi.api.*Controller"
    - "fr.iowizmi.batch.*Job"
  exclude:
    - "*Test*"
    - "*Mock*"
```

### 4.2 Reachability Analysis (4h)

**Algorithm:** BFS/DFS from all entry points.

```python
# ragix_audit/dead_code.py
def find_unreachable_code(
    graph: DependencyGraph,
    entry_points: Set[str]
) -> Set[str]:
    """
    Find classes/methods not reachable from any entry point.

    Returns:
        Set of unreachable symbol qualified names
    """
    reachable = set()
    queue = deque(entry_points)

    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)

        # Add all callees
        for callee in graph.get_callees(node):
            if callee not in reachable:
                queue.append(callee)

    all_nodes = graph.get_all_nodes()
    return all_nodes - reachable
```

### 4.3 Dead Code Report (4h)

**Categories:**
1. **Unreachable classes** — No path from any entry point
2. **Unused methods** — Methods never called (within analyzed codebase)
3. **Unused fields** — Private fields never read
4. **Orphan packages** — Packages with no incoming dependencies

**Report Section:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ Dead Code Analysis                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Entry points discovered:          47                                │
│ Total classes analyzed:        1,032                                │
│ Reachable classes:               892 (86.4%)                        │
│ Potentially dead classes:        140 (13.6%)                        │
│                                                                     │
│ Estimated dead code:          18,450 LOC                            │
│ Potential cleanup savings:     ~92 hours (at 200 LOC/hour)          │
├─────────────────────────────────────────────────────────────────────┤
│ Top Orphan Packages (no incoming deps):                             │
│ - fr.iowizmi.legacy.util (23 classes, 4,200 LOC)                   │
│ - fr.iowizmi.deprecated.v1 (12 classes, 1,800 LOC)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Propagation Impact Analysis (~8h)

### 5.1 Propagation Factor (4h)

**Definition:**
$$PF(n) = \frac{|reachable\_downstream(n)|}{|V|}$$

Where $V$ = all nodes in the dependency graph.

| PF Range | Risk Level |
|----------|------------|
| $PF > 0.3$ | Critical — changes here affect >30% of codebase |
| $0.1 < PF ≤ 0.3$ | High — significant downstream impact |
| $PF ≤ 0.1$ | Normal — localized impact |

**Implementation:**
```python
def propagation_factor(graph: DependencyGraph, node: str) -> float:
    """Compute PF using BFS on downstream dependencies."""
    downstream = set()
    queue = deque([node])

    while queue:
        current = queue.popleft()
        for dependent in graph.get_dependents(current):
            if dependent not in downstream:
                downstream.add(dependent)
                queue.append(dependent)

    return len(downstream) / graph.node_count()
```

### 5.2 Change Impact Simulation (4h)

Given a list of files to change, compute:
- **Direct impact:** Files that import changed files
- **Transitive impact:** All downstream files
- **Test coverage gap:** Impacted files without corresponding tests

**Interactive Query:**
```
> ragix-audit impact --files src/fr/iowizmi/iok/sk04/SK04Service.java

Impact Analysis for SK04Service.java:
├── Direct dependents: 12 classes
├── Transitive dependents: 47 classes (PF = 0.046)
├── Test coverage: 8/47 (17%) — CRITICAL GAP
└── Estimated regression risk: HIGH
```

---

## Phase 6: MCO Estimation Model (~10h)

**Target:** Provide actionable maintenance cost estimates.

### 6.1 Effort Model (6h)

**Formula:**
$$Effort(p) = \alpha_1 \cdot CC_{density} + \alpha_2 \cdot D + \alpha_3 \cdot PF + \alpha_4 \cdot V + \alpha_5 \cdot Size$$

Where:
- $CC_{density}$ = average CC / LOC (normalized)
- $D$ = Distance from main sequence
- $PF$ = Propagation factor
- $V$ = Volatility (change frequency if git available, else from file mtimes)
- $Size$ = LOC (log-normalized)

**Default weights (calibrate on historical data):**
```yaml
mco_weights:
  complexity_density: 0.25
  distance: 0.15
  propagation: 0.25
  volatility: 0.20
  size: 0.15
```

### 6.2 MCO Report (4h)

**Per-Component Breakdown:**
```
┌─────────────────────────────────────────────────────────────────┐
│ MCO Estimation — Annual Maintenance Effort                      │
├─────────────────────────────────────────────────────────────────┤
│ Component │  LOC   │ CC_d  │   D   │  PF   │ Effort │ % Total  │
├───────────┼────────┼───────┼───────┼───────┼────────┼──────────┤
│ SK04      │ 28,450 │ 0.032 │ 0.15  │ 0.12  │  340h  │   28.3%  │
│ SK13      │  6,200 │ 0.041 │ 0.22  │ 0.08  │  120h  │   10.0%  │
│ SC04      │  5,800 │ 0.028 │ 0.18  │ 0.05  │   85h  │    7.1%  │
│ ...       │        │       │       │       │        │          │
├───────────┴────────┴───────┴───────┴───────┼────────┼──────────┤
│ TOTAL ESTIMATED MCO EFFORT                 │ 1,200h │  100.0%  │
└────────────────────────────────────────────┴────────┴──────────┘

Quick Wins (highest ROI refactoring targets):
1. SK04.DataProcessor — CC=47, PF=0.12 → Refactor saves ~45h/year
2. SC04.ScreenManager — CC=31, PF=0.08 → Refactor saves ~28h/year
```

---

## Phase 7: Architecture Layer Analysis (~8h)

### 7.1 Layer Definition & Detection (4h)

**Default layers (configurable):**
```yaml
layers:
  - name: presentation
    patterns: ["*.ui.*", "*.web.*", "*.controller.*", "*.rest.*"]
    allowed_deps: [service]

  - name: service
    patterns: ["*.service.*", "*.application.*"]
    allowed_deps: [domain, infra]

  - name: domain
    patterns: ["*.domain.*", "*.model.*", "*.entity.*"]
    allowed_deps: []  # Domain should have no dependencies

  - name: infra
    patterns: ["*.infra.*", "*.repository.*", "*.persistence.*", "*.adapter.*"]
    allowed_deps: [domain]
```

### 7.2 Layer Violation Detection (4h)

**Violations:**
1. **Skip-layer:** Presentation → Infra (skipping service)
2. **Reverse dependency:** Domain → Service (domain should be independent)
3. **Circular layer:** Service → Presentation → Service

**Report:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Architecture Layer Violations                                   │
├─────────────────────────────────────────────────────────────────┤
│ Total violations: 23                                            │
│                                                                 │
│ Skip-layer violations (Presentation → Infra): 8                │
│ - SC04Controller → DatabaseRepository (line 45)                │
│ - SC05View → JdbcTemplate (line 123)                           │
│                                                                 │
│ Reverse dependencies (Domain → Service): 5                     │
│ - User.java → UserService (line 78)                            │
│                                                                 │
│ Recommended refactoring effort: ~35 hours                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 8: Report Consolidation (~6h)

### 8.1 Enhanced Technical Audit Report (4h)

Add sections (already partially done in v0.40):
- [x] Component Analysis (SK/SC/SG breakdown)
- [x] Service Life Profiles
- [ ] Descriptive Statistics (quartiles, entropy)
- [ ] Coupling Metrics (Ca, Ce, I, A, D)
- [ ] Dead Code Summary
- [ ] MCO Estimation

### 8.2 Executive Dashboard (2h)

**Key metrics at a glance:**
```
┌─────────────────────────────────────────────────────────────────┐
│ IOWIZME Code Health Dashboard                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Components: 17          Structural Entropy: 3.42               │
│  ┌──────────────┐        ████████░░░░░░░░░ (moderate)          │
│  │ SK: 9  (53%) │                                              │
│  │ SC: 7  (41%) │        Complexity Entropy: 2.87              │
│  │ SG: 1  ( 6%) │        ██████░░░░░░░░░░░ (concentrated)      │
│  └──────────────┘                                              │
│                                                                 │
│  High Risk:  2 components     Dead Code: ~14%                  │
│  MCO Effort: 1,200h/year      Tech Debt: ~280h                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Sprint 1: Statistics Foundation (Week 1)
- [ ] Create `ragix_audit/statistics.py` — DistributionStats class
- [ ] Add quartile/skewness/kurtosis computation
- [ ] Integrate into `report_engine.py` — new statistics section
- [ ] Box plot data export for D3.js

### Sprint 2: Entropy & Coupling (Week 2)
- [ ] Create `ragix_audit/entropy.py` — structural/complexity/coupling entropy
- [ ] Create `ragix_audit/coupling.py` — Ca, Ce, I, A, D computation
- [ ] Add A-I scatter plot to report (identify zones of pain/uselessness)
- [ ] SDP violation detection

### Sprint 3: Dead Code & Impact (Week 3)
- [ ] Create `ragix_audit/dead_code.py` — entry point discovery + reachability
- [ ] Create `ragix_audit/impact.py` — PF computation + change simulation
- [ ] Dead code report section
- [ ] CLI command: `ragix-audit impact --files ...`

### Sprint 4: MCO & Layers (Week 4)
- [ ] Create `ragix_audit/mco.py` — effort model + cost estimation
- [ ] Create `ragix_audit/layers.py` — layer definition + violation detection
- [ ] MCO breakdown per component
- [ ] Layer violation report

### Sprint 5: Integration & Validation (Week 5)
- [ ] Full integration test on IOWIZME
- [ ] Full integration test on SIAS
- [ ] Calibrate MCO weights against historical audit data
- [ ] Update all HTML report templates
- [ ] Documentation and examples

---

## File Structure

```
ragix_audit/
├── __init__.py           # Existing - public API
├── timeline.py           # Existing - service life profiles
├── component_mapper.py   # Existing - SK/SC/SG detection
├── risk.py               # Existing - risk scoring
├── drift.py              # Existing - spec-code drift
├── service_detector.py   # Existing - service detection
│
├── statistics.py         # NEW — DistributionStats, quartiles, skewness
├── entropy.py            # NEW — structural/complexity/coupling entropy
├── coupling.py           # NEW — Ca, Ce, I, A, D, SDP violations
├── dead_code.py          # NEW — entry points, reachability, orphan detection
├── impact.py             # NEW — propagation factor, change simulation
├── mco.py                # NEW — MCO effort model, cost estimation
├── layers.py             # NEW — architecture layers, violation detection
└── reports.py            # Existing - enhanced with new sections
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Dead code detection accuracy** | >90% | Validated against manual review |
| **MCO estimation accuracy** | ±20% | Compared to actual maintenance hours |
| **Layer violation coverage** | >95% | All skip-layer/reverse deps detected |
| **Report generation time** | <5 min | For 1000-file codebase |
| **Entropy correlation** | >0.7 | With maintenance effort (Pearson) |

---

## Dependencies

**Python packages (add to pyproject.toml):**
```toml
[tool.poetry.dependencies]
scipy = "^1.11"       # For statistics (skewness, kurtosis)
numpy = "^1.24"       # For numerical computation
networkx = "^3.0"     # For graph algorithms (already present)
```

---

## References

1. Martin, R.C. (2002). *Agile Software Development, Principles, Patterns, and Practices*. Prentice Hall. — Coupling metrics (Ca, Ce, I, A, D)
2. McCabe, T.J. (1976). "A Complexity Measure". *IEEE TSE*, 2(4). — Cyclomatic complexity
3. Shannon, C.E. (1948). "A Mathematical Theory of Communication". *Bell System Technical Journal*. — Entropy
4. Chidamber & Kemerer (1994). "A Metrics Suite for OO Design". *IEEE TSE*, 20(6). — CK metrics

---

*"What gets measured gets managed." — Peter Drucker*

*"Entropy is the price of structure." — Ilya Prigogine*
