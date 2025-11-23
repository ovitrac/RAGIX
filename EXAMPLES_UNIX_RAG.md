# Unix-RAG Examples for RAGIX v0.4

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab | olivier.vitrac@adservio.fr
**Date:** 2025-11-23

This document provides practical examples of using RAGIX in **pure Unix-RAG style** — using classic Unix tools (`grep`, `sed`, `head`, `tail`, `awk`, etc.) instead of SWE commands. This approach emphasizes composability, pipelines, and traditional Unix philosophy.

**Note:** These examples complement `EXAMPLES_SWE.md`. SWE tools are recommended for systematic navigation and editing, but Unix-RAG style remains powerful for quick exploration and one-off operations.

---

## 1. Bug Localization Workflow (Unix-RAG Style)

### Scenario
You encounter an error message: `"safety_margin calculation failed"` and need to locate and fix it.

### Steps

```bash
# Step 1: Find all files mentioning "safety_margin" (recursive grep)
grep -R -n "safety_margin" src/

# Output:
# src/sim/gas_flow_model.py:123: def compute_safety_margin(flow_rate, pressure):
# src/sim/gas_flow_model.py:156:     margin = compute_safety_margin(rate, p)
# src/tests/test_gas_flow.py:45:     result = compute_safety_margin(10.0, 15.0)

# Step 2: View context around the definition (±10 lines)
sed -n '113,133p' src/sim/gas_flow_model.py

# Step 3: Count occurrences to understand scope
grep -c "safety_margin" src/sim/gas_flow_model.py

# Step 4: Search for error handling in that file
grep -n "calculation failed" src/sim/gas_flow_model.py

# Step 5: View the problematic function with line numbers
awk 'NR>=123 && NR<=150' src/sim/gas_flow_model.py | cat -n

# Step 6: Extract just the function signature
sed -n '123p' src/sim/gas_flow_model.py

# Step 7: Check if there are tests for this function
grep -R "test_.*safety_margin" tests/ | cut -d: -f1 | uniq
```

**Unix Philosophy:** Small tools chained together for precise queries.

---

## 2. Feature Addition Workflow (Unix-RAG Style)

### Scenario
Add a new logging function to a module.

### Steps

```bash
# Step 1: Find the module
find src/ -name "gas_flow_model.py"

# Step 2: Check existing imports (first 30 lines)
head -30 src/sim/gas_flow_model.py

# Step 3: Find where imports end (look for first function def)
grep -n "^def " src/sim/gas_flow_model.py | head -1

# Step 4: Count total lines to understand file size
wc -l src/sim/gas_flow_model.py

# Step 5: Check if logging is already imported
grep -n "^import logging" src/sim/gas_flow_model.py || echo "Not found"

# Step 6: View the function we want to modify
sed -n '123,128p' src/sim/gas_flow_model.py

# Step 7: Check function calls to understand usage
grep -n "compute_safety_margin(" src/sim/gas_flow_model.py

# Step 8: Find all docstrings to understand documentation style
grep -A2 '"""' src/sim/gas_flow_model.py | head -20

# Step 9: Use Python to analyze function structure
python3 << 'PYEOF'
import ast
import sys

with open("src/sim/gas_flow_model.py") as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        print(f"Function: {node.name} at line {node.lineno}")
PYEOF
```

**Unix Philosophy:** Use the right tool for each task — shell for text, Python for AST.

---

## 3. Refactoring Workflow (Unix-RAG Style)

### Scenario
Rename a function across multiple files.

### Steps

```bash
# Step 1: Find all occurrences with context
grep -R -n -B2 -A2 "old_function_name" src/

# Step 2: Count occurrences per file
grep -R "old_function_name" src/ | cut -d: -f1 | sort | uniq -c

# Step 3: List just the files (for scripting)
grep -R -l "old_function_name" src/

# Step 4: Extract function definition
grep -A10 "^def old_function_name" src/module_a.py

# Step 5: Show all call sites with surrounding code
grep -B3 -A3 "old_function_name(" src/*.py

# Step 6: Generate a replacement plan (dry-run)
grep -R -n "old_function_name" src/ | \
  awk -F: '{print "Edit " $1 " line " $2}'

# Step 7: Verify no occurrences in comments only
grep -R "old_function_name" src/ | grep -v "^[[:space:]]*#"

# Step 8: Check if function name appears in documentation
find . -name "*.md" -exec grep -l "old_function_name" {} \;

# Step 9: Use sed for preview (no actual change)
sed -n 's/old_function_name/new_function_name/p' src/module_a.py

# Step 10: Count lines that would change
grep -R "old_function_name" src/ | wc -l
```

**Unix Philosophy:** Compose pipelines to analyze before modifying.

---

## 4. Large File Navigation (Unix-RAG Style)

### Scenario
Navigate a 5000-line configuration file to find specific settings.

### Steps

```bash
# Step 1: Check file size first
wc -l config/settings.py

# Step 2: Get a bird's-eye view (every 100th line)
sed -n '1~100p' config/settings.py | cat -n

# Step 3: Find section markers (usually uppercase constants)
grep -n "^[A-Z_]*\s*=" config/settings.py | head -20

# Step 4: Search for specific setting
grep -n "DATABASE_SETTINGS" config/settings.py

# Output: 2341: DATABASE_SETTINGS = {

# Step 5: View that section (50 lines after)
sed -n '2341,2391p' config/settings.py

# Step 6: Find where the section ends (closing brace)
awk 'NR>=2341 && /^}/ {print NR; exit}' config/settings.py

# Step 7: Extract just that section
sed -n '2341,2450p' config/settings.py > /tmp/db_settings.txt

# Step 8: Search within extracted section
grep "connection_timeout" /tmp/db_settings.txt

# Step 9: View with line numbers relative to file
grep -n "connection_timeout" config/settings.py

# Step 10: Get context (±5 lines)
grep -n -C5 "connection_timeout" config/settings.py

# Step 11: Use awk for structured extraction
awk '/DATABASE_SETTINGS/,/^}/' config/settings.py
```

**Unix Philosophy:** Use sed/awk for range operations, grep for patterns.

---

## 5. Code Review Workflow (Unix-RAG Style)

### Scenario
Review recent changes to understand modifications.

### Steps

```bash
# Step 1: List recently modified files
find src/ -type f -name "*.py" -mtime -7 | sort

# Step 2: Get git status summary
git status -sb

# Step 3: Show files changed in last commit
git diff --name-only HEAD~1

# Step 4: Show line changes summary
git diff --stat HEAD~1

# Step 5: View specific change hunks
git diff HEAD~1 src/module.py

# Step 6: Extract just added lines
git diff HEAD~1 src/module.py | grep "^+"

# Step 7: Extract just removed lines
git diff HEAD~1 src/module.py | grep "^-"

# Step 8: Count changes by type
git diff HEAD~1 --numstat

# Step 9: Find function definitions that changed
git diff HEAD~1 src/module.py | grep "^[+-]def "

# Step 10: Show commit message and changes
git show HEAD~1 --stat

# Step 11: View specific function context
git show HEAD~1:src/module.py | sed -n '40,60p'

# Step 12: Compare old vs new side-by-side
diff -y <(git show HEAD~1:src/module.py | sed -n '40,50p') \
        <(sed -n '40,50p' src/module.py)

# Step 13: Use git grep to find related code
git grep "process_data" src/
```

**Unix Philosophy:** Leverage git as a database, combine with text tools.

---

## 6. Multi-File Pattern Analysis (Unix-RAG Style)

### Scenario
Find all copyright headers and analyze their format.

### Steps

```bash
# Step 1: Find all Python files
find . -name "*.py" -type f | wc -l

# Step 2: Extract first 10 lines of each file
find . -name "*.py" -exec head -10 {} \; | head -100

# Step 3: Find copyright statements
grep -R "Copyright" . --include="*.py" | head -20

# Step 4: Count copyright formats
grep -R "Copyright" . --include="*.py" | \
  sed 's/.*Copyright/Copyright/' | sort | uniq -c

# Step 5: Find files WITHOUT copyright
find . -name "*.py" -exec sh -c \
  'grep -q "Copyright" "$1" || echo "$1"' _ {} \;

# Step 6: Extract year patterns
grep -R "Copyright.*[0-9]\{4\}" . --include="*.py" | \
  sed 's/.*\([0-9]\{4\}\).*/\1/' | sort | uniq -c

# Step 7: Find oldest copyright year
grep -R "Copyright.*[0-9]\{4\}" . --include="*.py" | \
  sed 's/.*\([0-9]\{4\}\).*/\1/' | sort -n | head -1

# Step 8: Generate summary by author
grep -R "Copyright.*©" . --include="*.py" | \
  sed 's/.*©\s*\([^0-9]*\).*/\1/' | sort | uniq -c

# Step 9: List files by copyright year
grep -R -l "Copyright.*2024" . --include="*.py"

# Step 10: Create mapping of file->year
grep -R "Copyright" . --include="*.py" | \
  awk -F: '{print $1}' | while read file; do
    year=$(grep "Copyright" "$file" | sed 's/.*\([0-9]\{4\}\).*/\1/' | head -1)
    echo "$file: $year"
  done | column -t
```

**Unix Philosophy:** Combine find, grep, sed, awk for data extraction and analysis.

---

## 7. Documentation Update Workflow (Unix-RAG Style)

### Scenario
Add docstrings to undocumented functions.

### Steps

```bash
# Step 1: Find all function definitions
grep -R -n "^def " src/ --include="*.py"

# Step 2: Check if next line is docstring
grep -R -A1 "^def " src/ --include="*.py" | grep -v '"""' | grep "def "

# Step 3: Count functions vs docstrings
FUNCS=$(grep -R "^def " src/ --include="*.py" | wc -l)
DOCS=$(grep -R -A1 "^def " src/ --include="*.py" | grep '"""' | wc -l)
echo "Functions: $FUNCS, Documented: $DOCS, Missing: $((FUNCS - DOCS))"

# Step 4: Find undocumented functions with line numbers
grep -R -n "^def " src/ --include="*.py" | while read line; do
  file=$(echo "$line" | cut -d: -f1)
  linenum=$(echo "$line" | cut -d: -f2)
  nextline=$((linenum + 1))
  if ! sed -n "${nextline}p" "$file" | grep -q '"""'; then
    echo "$file:$linenum"
  fi
done | head -20

# Step 5: Extract function signatures for analysis
grep -R "^def " src/ --include="*.py" | \
  sed 's/.*def \([^(]*\).*/\1/' | sort | uniq -c | sort -rn

# Step 6: Find functions with type hints
grep -R "^def.*->" src/ --include="*.py"

# Step 7: Find functions WITHOUT type hints
grep -R "^def " src/ --include="*.py" | grep -v " -> "

# Step 8: Check existing docstring styles
grep -R -A5 '"""' src/ --include="*.py" | head -50

# Step 9: Find Google-style vs NumPy-style docstrings
grep -R "Args:" src/ --include="*.py" | wc -l  # Google
grep -R "Parameters" src/ --include="*.py" | wc -l  # NumPy

# Step 10: Generate template for undocumented function
cat << 'TEMPLATE'
    """
    Brief description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description of return value
    """
TEMPLATE
```

**Unix Philosophy:** Use text processing to audit and generate templates.

---

## 8. Testing Integration (Unix-RAG Style)

### Scenario
Add test cases for a newly added function.

### Steps

```bash
# Step 1: Find existing test files
find tests/ -name "test_*.py" | sort

# Step 2: Check test structure
head -50 tests/test_gas_flow.py

# Step 3: Count test methods
grep -c "def test_" tests/test_gas_flow.py

# Step 4: List all test method names
grep "def test_" tests/test_gas_flow.py | \
  sed 's/.*def \(test_[^(]*\).*/\1/'

# Step 5: Find test class structure
grep -n "^class.*Test" tests/test_gas_flow.py

# Step 6: Check coverage of function
grep "compute_safety_margin" tests/test_gas_flow.py

# Step 7: Find assertion patterns used
grep "self.assert" tests/test_gas_flow.py | \
  sed 's/.*self\.\(assert[^(]*\).*/\1/' | sort | uniq -c

# Step 8: Check for setUp/tearDown methods
grep -n "def setUp\|def tearDown" tests/test_gas_flow.py

# Step 9: Find mock usage
grep -n "mock\|patch" tests/test_gas_flow.py

# Step 10: Extract test method template
sed -n '/def test_/,/^[[:space:]]*def \|^class /p' \
  tests/test_gas_flow.py | head -20

# Step 11: Count lines per test (complexity check)
awk '/def test_/ {start=NR} /^[[:space:]]*def / && NR>start {
  print NR-start; start=0
}' tests/test_gas_flow.py | \
  awk '{sum+=$1; count++} END {print "Avg lines per test:", sum/count}'

# Step 12: Find untested functions
comm -23 \
  <(grep "^def " src/sim/gas_flow_model.py | sed 's/def \([^(]*\).*/\1/' | sort) \
  <(grep "test_" tests/test_gas_flow.py | sed 's/.*test_\([^(]*\).*/\1/' | sort)
```

**Unix Philosophy:** Use awk for counting, comm for set operations.

---

## 9. Dependency Analysis (Unix-RAG Style)

### Scenario
Understand module dependencies and imports.

### Steps

```bash
# Step 1: Extract all imports from a file
grep "^import \|^from " src/sim/gas_flow_model.py

# Step 2: Count import statements
grep -c "^import \|^from " src/sim/gas_flow_model.py

# Step 3: List unique imported modules
grep "^import \|^from " src/sim/gas_flow_model.py | \
  sed 's/^from \([^ ]*\).*/\1/; s/^import \(.*\)/\1/' | \
  cut -d. -f1 | sort | uniq

# Step 4: Find standard library vs third-party imports
grep "^import \|^from " src/sim/gas_flow_model.py | \
  awk '{print $2}' | cut -d. -f1 | sort | uniq | \
  while read mod; do
    python3 -c "import $mod" 2>/dev/null && echo "stdlib: $mod" || echo "external: $mod"
  done

# Step 5: Find circular dependencies (within src/)
for file in src/**/*.py; do
  imports=$(grep "^from src\." "$file" | sed 's/from \(src\.[^ ]*\).*/\1/')
  for imp in $imports; do
    echo "$file -> $imp"
  done
done | tee /tmp/deps.txt

# Step 6: Build dependency graph (DOT format)
cat << 'DOTEOF' > /tmp/deps.dot
digraph deps {
DOTEOF
grep "^from src\|^import src" src/**/*.py | \
  awk -F: '{print $1}' | while read file; do
    imports=$(grep "^from src\." "$file" | sed 's/from \(src\.[^ ]*\).*/\1/')
    for imp in $imports; do
      echo "  \"$file\" -> \"$imp\";" >> /tmp/deps.dot
    done
  done
echo "}" >> /tmp/deps.dot

# Step 7: Find files with most imports (high coupling)
find src/ -name "*.py" -exec sh -c \
  'echo "$(grep -c "^import \|^from " "$1") $1"' _ {} \; | \
  sort -rn | head -10

# Step 8: Find most imported modules (high fan-in)
grep -R "^from src\." src/ --include="*.py" | \
  sed 's/.*from \(src\.[^ ]*\).*/\1/' | \
  sort | uniq -c | sort -rn | head -10

# Step 9: Check for unused imports (Python help)
python3 << 'PYEOF'
import ast
import sys

file = "src/sim/gas_flow_model.py"
with open(file) as f:
    tree = ast.parse(f.read())

imports = set()
used = set()

for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            imports.add(node.module.split('.')[0])
    elif isinstance(node, ast.Name):
        used.add(node.id)

unused = imports - used
if unused:
    print("Potentially unused imports:", unused)
PYEOF
```

**Unix Philosophy:** Generate dependency graphs, use Python for AST when needed.

---

## 10. Performance Profiling (Unix-RAG Style)

### Scenario
Find performance bottlenecks in code.

### Steps

```bash
# Step 1: Find loops (potential bottlenecks)
grep -R -n "for \|while " src/ --include="*.py" | head -20

# Step 2: Find nested loops
grep -R -B1 "for \|while " src/ --include="*.py" | \
  grep -A1 "for \|while " | head -20

# Step 3: Count algorithmic complexity indicators
echo "Nested loops:"
grep -R "for.*for\|for.*while" src/ --include="*.py" | wc -l

# Step 4: Find list comprehensions (good) vs loops
echo "List comprehensions:"
grep -R "\[.*for.*in.*\]" src/ --include="*.py" | wc -l

# Step 5: Find potential N+1 queries (in loops)
grep -R -A5 "for .* in " src/ --include="*.py" | \
  grep -B5 "query\|find\|get" | head -30

# Step 6: Check for global variables (bad for performance)
grep -R "^[A-Z_][A-Z_]*\s*=" src/ --include="*.py"

# Step 7: Find function call depth (call within call)
grep -R "[a-z_]*([^)]*([^)]*(" src/ --include="*.py" | head -10

# Step 8: Measure function sizes (lines per function)
awk '/^def / {if (start) print NR-start; start=NR}
     END {if (start) print NR-start}' \
  src/sim/gas_flow_model.py | \
  awk '{sum+=$1; if ($1>max) max=$1; count++}
       END {print "Avg:", sum/count, "Max:", max}'

# Step 9: Find recursive functions
for file in src/**/*.py; do
  grep "^def " "$file" | sed 's/def \([^(]*\).*/\1/' | while read func; do
    if grep -q "$func(" "$file"; then
      count=$(grep -c "$func(" "$file")
      if [ "$count" -gt 1 ]; then
        echo "$file: $func (possibly recursive)"
      fi
    fi
  done
done

# Step 10: Use cProfile for actual profiling
python3 -m cProfile -s cumtime src/sim/gas_flow_model.py 2>&1 | head -30
```

**Unix Philosophy:** Combine pattern matching with Python profiling tools.

---

## 11. Security Audit (Unix-RAG Style)

### Scenario
Check for common security issues.

### Steps

```bash
# Step 1: Find potential SQL injection points
grep -R "execute.*%\|execute.*format\|execute.*\+" src/ --include="*.py"

# Step 2: Find eval/exec usage (dangerous)
grep -R -n "eval(\|exec(" src/ --include="*.py"

# Step 3: Find shell command execution
grep -R "subprocess\|os.system\|os.popen" src/ --include="*.py"

# Step 4: Check for hardcoded secrets
grep -R -i "password\s*=\|api_key\s*=\|secret\s*=" src/ --include="*.py"

# Step 5: Find pickle usage (unsafe deserialization)
grep -R "pickle\|cPickle" src/ --include="*.py"

# Step 6: Check for open() without encoding
grep -R "open(" src/ --include="*.py" | grep -v "encoding="

# Step 7: Find potential path traversal
grep -R "os.path.join.*input\|open(.*input" src/ --include="*.py"

# Step 8: Check for weak random usage
grep -R "random\." src/ --include="*.py" | grep -v "secrets\."

# Step 9: Find unvalidated redirects
grep -R "redirect.*request\|redirect.*input" src/ --include="*.py"

# Step 10: Generate security report
cat << 'REPORT'
Security Audit Report
=====================
REPORT
echo "Eval/exec calls: $(grep -R 'eval(\|exec(' src/ --include='*.py' | wc -l)"
echo "Shell commands: $(grep -R 'subprocess\|os.system' src/ --include='*.py' | wc -l)"
echo "Hardcoded secrets: $(grep -R -i 'password\s*=\|api_key\s*=' src/ --include='*.py' | wc -l)"
echo "Pickle usage: $(grep -R 'pickle' src/ --include='*.py' | wc -l)"
```

**Unix Philosophy:** Pattern matching for known vulnerabilities.

---

## 12. Code Metrics Collection (Unix-RAG Style)

### Scenario
Generate comprehensive code metrics.

### Steps

```bash
# Step 1: Count total lines of code
find src/ -name "*.py" -exec wc -l {} + | tail -1

# Step 2: Count files by type
find src/ -name "*.py" | wc -l

# Step 3: Lines per file distribution
find src/ -name "*.py" -exec wc -l {} + | \
  awk '{print $1}' | sort -n | \
  awk '{sum+=$1; count++; values[count]=$1}
       END {
         print "Total:", sum;
         print "Mean:", sum/count;
         print "Median:", values[int(count/2)];
         print "Max:", values[count]
       }'

# Step 4: Function count
grep -R "^def " src/ --include="*.py" | wc -l

# Step 5: Class count
grep -R "^class " src/ --include="*.py" | wc -l

# Step 6: Comment density
TOTAL=$(find src/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
COMMENTS=$(grep -R "^\s*#" src/ --include="*.py" | wc -l)
echo "scale=2; $COMMENTS * 100 / $TOTAL" | bc

# Step 7: Docstring coverage
FUNCS=$(grep -R "^def " src/ --include="*.py" | wc -l)
DOCS=$(grep -R -A1 "^def " src/ --include="*.py" | grep '"""' | wc -l)
echo "scale=2; $DOCS * 100 / $FUNCS" | bc

# Step 8: Average function length
awk '/^def / {if (start) {print NR-start}; start=NR}
     END {if (start) print NR-start}' \
  src/**/*.py | \
  awk '{sum+=$1; count++} END {print sum/count}'

# Step 9: Complexity estimation (rough)
echo "Conditional branches:"
grep -R "if \|elif \|else:" src/ --include="*.py" | wc -l
echo "Loops:"
grep -R "for \|while " src/ --include="*.py" | wc -l
echo "Try/except blocks:"
grep -R "try:\|except " src/ --include="*.py" | wc -l

# Step 10: Generate summary report
cat << 'METRICS' > /tmp/metrics.txt
Code Metrics Summary
====================
Files: $(find src/ -name "*.py" | wc -l)
Total LOC: $(find src/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
Functions: $(grep -R "^def " src/ --include="*.py" | wc -l)
Classes: $(grep -R "^class " src/ --include="*.py" | wc -l)
Comments: $(grep -R "^\s*#" src/ --include="*.py" | wc -l)
Docstrings: $(grep -R '"""' src/ --include="*.py" | wc -l)
METRICS
cat /tmp/metrics.txt
```

**Unix Philosophy:** Aggregate statistics using wc, awk, bc.

---

## Comparison: Unix-RAG vs SWE Tools

| Task | Unix-RAG Approach | SWE Approach | Best For |
|------|------------------|--------------|----------|
| **Quick search** | `grep -R -n "pattern" src/` | `rt grep "pattern" src/` | Unix-RAG (faster) |
| **View context** | `sed -n '100,150p' file` | `rt open file:125` | SWE (cleaner) |
| **Systematic nav** | Multiple sed/head/tail calls | `rt open`, `rt scroll` | SWE (stateful) |
| **Line-based edit** | Manual sed/temp file | `rt edit file 10 20` | SWE (safer) |
| **Analysis** | awk/grep pipelines | Same tools | Unix-RAG (flexible) |
| **Automation** | Shell scripts | Either | Unix-RAG (composable) |

---

## Best Practices: Unix-RAG Style

### DO:
- ✅ Use grep for discovery
- ✅ Use sed/awk for extraction
- ✅ Pipe commands for composition
- ✅ Test patterns before modifying
- ✅ Use `-n` flag for line numbers
- ✅ Leverage git as a database
- ✅ Combine with Python for AST work

### DON'T:
- ❌ Don't modify files without backups
- ❌ Don't use sed -i without testing
- ❌ Don't forget to escape special regex characters
- ❌ Don't pipe grep to grep repeatedly (use single regex)
- ❌ Don't ignore error codes in scripts

---

## When to Use Each Approach

### Use Unix-RAG when:
- 🔍 Doing exploratory analysis
- 🔍 Building complex pipelines
- 🔍 Working on remote systems (no RAGIX)
- 🔍 Automating batch operations
- 🔍 Generating reports

### Use SWE tools when:
- 🎯 Systematically navigating large files
- 🎯 Making precise line-based edits
- 🎯 Need state tracking across commands
- 🎯 Want safety guarantees (backups, profiles)
- 🎯 Working with LLM agents

---

**For SWE-style examples, see:** `EXAMPLES_SWE.md`
**For tool reference, see:** `README_RAGIX_TOOLS.md`
**For specification, see:** `SWE_TOOLING.md`
