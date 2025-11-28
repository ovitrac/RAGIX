# RAGIX AST Guide

**Version:** 0.20.0 | **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Updated:** 2025-11-28

---

The RAGIX Abstract Syntax Tree (`ragix-ast`) tool is a powerful command-line interface for deep code analysis, architectural exploration, and software metrics calculation. It works by parsing source code into an Abstract Syntax Tree (AST) and building a comprehensive dependency graph of your entire project.

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Quick Start](#2-quick-start)
3. [Subcommand Reference](#3-subcommand-reference)
4. [Query Language](#4-query-language)
5. [Output Formats](#5-output-formats)
6. [Integration Examples](#6-integration-examples)

---

## 1. Core Concepts

### Abstract Syntax Tree (AST)

An AST is a tree representation of source code structure. `ragix-ast` uses language-specific parsers to build these trees:
- **Python:** Built-in `ast` module
- **Java:** `javalang` parser

### Dependency Graph

A graph where:
- **Nodes** are symbols (classes, functions, methods, interfaces)
- **Edges** are relationships (imports, calls, inheritance, implementation)

This graph powers all analysis features.

### Supported Languages

| Language | File Extensions | Parser |
|----------|-----------------|--------|
| Python | `.py` | Built-in `ast` |
| Java | `.java` | `javalang` |

---

## 2. Quick Start

```bash
# Scan a project and get summary
ragix-ast scan ./src

# Calculate code metrics
ragix-ast metrics ./src

# Find the most complex code
ragix-ast hotspots ./src --top 10

# Generate interactive dependency visualization
ragix-ast graph ./src --format html --output deps.html

# Search for patterns
ragix-ast search ./src "type:class name:*Service*"
```

---

## 3. Subcommand Reference

### `scan`

Performs a full analysis of a project directory.

**Usage:**
```bash
ragix-ast scan <path> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--lang LANG` | Force language (`python`, `java`, `auto`) | `auto` |
| `--exclude PATTERN` | Exclude paths matching pattern | None |
| `--json` | Output raw JSON | Disabled |
| `--verbose` | Show detailed progress | Disabled |

**Example:**
```bash
# Scan Java project
ragix-ast scan ./src --lang java

# Scan with exclusions
ragix-ast scan . --exclude "test/*" --exclude "vendor/*"

# Get JSON output for scripting
ragix-ast scan ./src --json > analysis.json
```

**Output:**
```
Project Analysis: ./src
━━━━━━━━━━━━━━━━━━━━━━━━
Files analyzed:     150
Symbols extracted:  1,234
Dependencies found: 3,456
Classes:           89
Functions:         456
Methods:           689
```

---

### `parse`

Parses a single file and shows its AST. Useful for debugging and understanding code structure.

**Usage:**
```bash
ragix-ast parse <file> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--json` | Full AST as JSON | Disabled |
| `--symbols` | List symbols only | Disabled |
| `--depth N` | Limit tree depth | Unlimited |

**Examples:**
```bash
# Show AST summary
ragix-ast parse src/main.py --depth 5

# List all symbols in a file
ragix-ast parse src/api/user.java --symbols

# Get full AST as JSON
ragix-ast parse src/handler.py --json
```

---

### `search`

Search for symbols using the RAGIX query language. This is one of the most powerful features.

**Usage:**
```bash
ragix-ast search <path> "<query>" [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--limit N` | Maximum results | `50` |
| `--json` | Output as JSON | Disabled |
| `--context N` | Show N lines of context | `0` |

**Examples:**
```bash
# Find all classes with "Service" in name
ragix-ast search . "type:class name:*Service*"

# Find all public methods calling database
ragix-ast search . "type:method visibility:public calls:database.query"

# Find all test methods
ragix-ast search . "type:method @Test"

# Find classes extending BaseRepository
ragix-ast search . "type:class extends:BaseRepository"

# Find non-abstract service classes
ragix-ast search . "type:class extends:BaseService ! name:*Abstract*"
```

> See [Query Language](#4-query-language) for full syntax.

---

### `graph`

Generate dependency graph visualizations.

**Usage:**
```bash
ragix-ast graph <path> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--format FMT` | Output format (`html`, `dot`, `mermaid`, `json`) | `html` |
| `--output FILE` | Output file path | stdout |
| `--cluster` | Group by file/package | Enabled |
| `--max-nodes N` | Limit nodes for performance | `1000` |

**Examples:**
```bash
# Interactive HTML visualization
ragix-ast graph ./src --format html --output deps.html

# Graphviz DOT format (pipe to dot)
ragix-ast graph ./src --format dot | dot -Tpng -o graph.png

# Mermaid for documentation
ragix-ast graph ./src --format mermaid > deps.md

# JSON for custom processing
ragix-ast graph ./src --format json > graph.json
```

---

### `metrics`

Calculate professional code metrics.

**Usage:**
```bash
ragix-ast metrics <path> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--json` | Output as JSON | Disabled |
| `--csv` | Output as CSV | Disabled |

**Metrics Reported:**

| Category | Metrics |
|----------|---------|
| **Overview** | Files, Lines of Code, Code Lines, Comment Lines |
| **Complexity** | Cyclomatic Complexity (total, average, max) |
| **Maintainability** | Maintainability Index (0-100) |
| **Technical Debt** | Estimated hours to fix issues |
| **Coupling** | Afferent (Ca), Efferent (Ce), Instability (I) |
| **Cycles** | Number of circular dependencies |

**Example:**
```bash
ragix-ast metrics ./src
```

**Output:**
```
Code Metrics Report: ./src
━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Overview
   Files:              150
   Lines of Code:      25,432
   Code Lines:         18,567
   Comment Lines:      3,245

🧮 Complexity
   Total:              1,234
   Average:            3.5
   Maximum:            45 (UserService.processOrder)

🛠️ Maintainability
   Index:              72.5 (Good)

⏱️ Technical Debt
   Estimated:          45.5 hours

🔗 Coupling
   Afferent (Ca):      234
   Efferent (Ce):      567
   Instability:        0.71

⚠️ Circular Dependencies: 3 detected
   - UserService ↔ OrderService
   - AuthModule ↔ UserModule
   - Cache ↔ Database
```

---

### `matrix`

Generate a Dependency Structure Matrix (DSM) - a compact representation for analyzing architecture.

**Usage:**
```bash
ragix-ast matrix <path> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--level LEVEL` | Aggregation (`class`, `package`, `file`) | `package` |
| `--output FILE` | Output HTML file | stdout |

**Example:**
```bash
# Package-level DSM
ragix-ast matrix ./src --level package --output matrix.html

# Class-level for detailed analysis
ragix-ast matrix ./src --level class --output class_matrix.html
```

**How to Read:**
- Rows: Dependent modules (who depends on others)
- Columns: Depended-upon modules (who is depended upon)
- Cell values: Number of dependencies
- Colors: Intensity indicates coupling strength
- Diagonal: Self-dependencies (cycles if non-zero)

---

### `radial`

Generate an ego-centric visualization showing a central symbol and its dependencies radiating outward.

**Usage:**
```bash
ragix-ast radial <path> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--focal SYMBOL` | Central symbol to focus on | Most connected |
| `--levels N` | Dependency depth levels | `3` |
| `--output FILE` | Output HTML file | stdout |

**Example:**
```bash
# Explore UserService dependencies
ragix-ast radial ./src --focal UserService --output user_deps.html

# Auto-select most connected class
ragix-ast radial ./src --levels 2 --output core_deps.html
```

---

### `hotspots`

Find the most complex and problematic code.

**Usage:**
```bash
ragix-ast hotspots <path> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--top N` | Number of hotspots to show | `20` |
| `--metric METRIC` | Sort by metric (`complexity`, `size`, `coupling`) | `complexity` |

**Example:**
```bash
ragix-ast hotspots ./src --top 10
```

**Output:**
```
Top 10 Complexity Hotspots
━━━━━━━━━━━━━━━━━━━━━━━━━

1. UserService.processOrder (CC: 45)
   📁 src/services/user.py:234

2. PaymentHandler.validate (CC: 38)
   📁 src/handlers/payment.py:89

3. OrderValidator.checkRules (CC: 32)
   📁 src/validators/order.py:156
...
```

---

### `cycles`

Detect and list circular dependencies.

**Usage:**
```bash
ragix-ast cycles <path>
```

**Example:**
```bash
ragix-ast cycles ./src
```

**Output:**
```
Circular Dependencies Detected: 3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cycle 1 (length: 2):
  UserService → OrderService → UserService

Cycle 2 (length: 3):
  AuthModule → UserModule → SessionModule → AuthModule

Cycle 3 (length: 2):
  Cache → Database → Cache
```

---

### `maven`

Parse Maven POM files for Java project analysis.

**Usage:**
```bash
ragix-ast maven <path>
```

**Example:**
```bash
ragix-ast maven ./pom.xml
ragix-ast maven ./  # Scans for pom.xml files
```

---

### `sonar`

Query a SonarQube/SonarCloud instance.

**Usage:**
```bash
ragix-ast sonar <project_key> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--url URL` | SonarQube server URL | `SONAR_URL` env |
| `--token TOKEN` | Authentication token | `SONAR_TOKEN` env |

**Example:**
```bash
export SONAR_URL="https://sonarcloud.io"
export SONAR_TOKEN="your_token"
ragix-ast sonar my-project-key
```

---

## 4. Query Language

The RAGIX query language enables powerful symbol searches.

### Predicates

| Predicate | Description | Example |
|-----------|-------------|---------|
| `type:<type>` | Symbol type | `type:class`, `type:method` |
| `name:<pattern>` | Name pattern (supports `*`) | `name:*Service*` |
| `extends:<pattern>` | Parent class | `extends:BaseEntity` |
| `implements:<pattern>` | Interface | `implements:Serializable` |
| `calls:<pattern>` | Called function | `calls:database.*` |
| `@<annotation>` | Has annotation | `@Test`, `@Override` |
| `visibility:<vis>` | Access level | `visibility:public` |
| `! <predicate>` | Negation | `! type:method` |

### Symbol Types

- `class` - Classes
- `interface` - Interfaces
- `method` - Class methods
- `function` - Standalone functions
- `field` - Class fields/attributes
- `enum` - Enumerations

### Combining Predicates

Predicates are ANDed together by default:

```bash
# Find public methods in Service classes
ragix-ast search . "type:method visibility:public name:*Service*"

# Find abstract classes that extend BaseEntity
ragix-ast search . "type:class extends:BaseEntity name:Abstract*"
```

---

## 5. Output Formats

### HTML (Interactive)

Self-contained HTML files with D3.js visualizations. Features:
- Zoom and pan
- Node filtering
- Search highlighting
- Tooltip details
- Package grouping

### DOT (Graphviz)

Standard DOT format for custom rendering:
```bash
ragix-ast graph ./src --format dot | dot -Tsvg -o graph.svg
```

### Mermaid

Markdown-embeddable diagrams:
```bash
ragix-ast graph ./src --format mermaid >> README.md
```

### JSON

Structured data for custom processing:
```bash
ragix-ast graph ./src --format json | jq '.nodes | length'
```

---

## 6. Integration Examples

### CI/CD Quality Gate

```bash
#!/bin/bash
# quality_check.sh

# Check for high complexity
HOTSPOTS=$(ragix-ast hotspots ./src --top 5 --metric complexity)
MAX_CC=$(echo "$HOTSPOTS" | grep -oP 'CC: \K\d+' | head -1)

if [ "$MAX_CC" -gt 30 ]; then
    echo "FAIL: Maximum complexity ($MAX_CC) exceeds threshold (30)"
    exit 1
fi

# Check for cycles
CYCLES=$(ragix-ast cycles ./src 2>&1)
if echo "$CYCLES" | grep -q "Detected:"; then
    echo "FAIL: Circular dependencies found"
    exit 1
fi

echo "PASS: Code quality checks passed"
```

### Generate Architecture Documentation

```bash
# Generate all visualizations for documentation
ragix-ast graph ./src --format html --output docs/dependency_graph.html
ragix-ast matrix ./src --level package --output docs/dsm.html
ragix-ast metrics ./src --json > docs/metrics.json
ragix-ast hotspots ./src --top 20 > docs/hotspots.txt
```

### Custom Analysis Script

```python
import subprocess
import json

# Get metrics as JSON
result = subprocess.run(
    ["ragix-ast", "metrics", "./src", "--json"],
    capture_output=True, text=True
)
metrics = json.loads(result.stdout)

# Check maintainability
if metrics["maintainability"]["index"] < 50:
    print("Warning: Low maintainability score!")
```

---

*For more information, see the [CLI Guide](CLI_GUIDE.md) or run `ragix-ast --help`.*
