# SWE Tooling Examples for RAGIX v0.5

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab | olivier.vitrac@adservio.fr
**Date:** 2025-11-24
**Updated for:** RAGIX v0.5 modular package structure

This document provides practical examples of using RAGIX SWE tools for systematic codebase exploration and editing, following the conventions defined in `SWE_TOOLING.md`.

## Installation and Setup (v0.5)

```bash
# Install RAGIX modular package
cd /path/to/RAGIX
pip install -e .

# Start agent with SWE tools (rt commands available)
ragix-unix-agent --profile dev --sandbox-root ~/my-project

# Or run directly without install
python3 -m ragix_unix.cli --profile dev --sandbox-root ~/my-project

# With debug logging
python3 -m ragix_unix.cli --profile dev --debug --sandbox-root ~/project
```

---

## 1. Bug Localization Workflow

### Scenario
You encounter an error message: `"safety_margin calculation failed"` and need to locate and fix it.

### Steps

```bash
# Step 1: Find all files mentioning "safety_margin"
rt grep "safety_margin" src/

# Output:
# src/sim/gas_flow_model.py:123: def compute_safety_margin(flow_rate, pressure):
# src/sim/gas_flow_model.py:156:     margin = compute_safety_margin(rate, p)
# src/tests/test_gas_flow.py:45:     result = compute_safety_margin(10.0, 15.0)

# Step 2: Jump to the definition
rt open src/sim/gas_flow_model.py:123

# Output shows 100 lines centered around line 123

# Step 3: Search within this specific file for error handling
rt grep-file "calculation failed" src/sim/gas_flow_model.py

# Step 4: Edit the problematic range
rt edit src/sim/gas_flow_model.py 145 150 << 'EOF'
def compute_safety_margin(flow_rate: float, pressure: float) -> float:
    """Calculate safety margin with proper error handling."""
    if flow_rate <= 0 or pressure <= 0:
        raise ValueError("safety_margin calculation failed: invalid input")
    return max(0.0, pressure - 1.1 * flow_rate)
EOF

# Step 5: Verify changes
git diff -- src/sim/gas_flow_model.py
```

---

## 2. Feature Addition Workflow

### Scenario
Add a new logging function to a module.

### Steps

```bash
# Step 1: Find the module
rt find "gas_flow_model.py" src/

# Step 2: Check the import section
rt open src/sim/gas_flow_model.py:1-30

# Step 3: Insert new import at appropriate location
rt insert src/sim/gas_flow_model.py 10 << 'EOF'
import logging
logger = logging.getLogger(__name__)
EOF

# Step 4: Find where to add logging call
rt grep-file "def compute_safety_margin" src/sim/gas_flow_model.py

# Step 5: Navigate to function
rt open src/sim/gas_flow_model.py:123

# Step 6: Edit function to add logging
rt edit src/sim/gas_flow_model.py 123 128 << 'EOF'
def compute_safety_margin(flow_rate: float, pressure: float) -> float:
    """Calculate safety margin with logging."""
    logger.debug(f"Computing safety margin: flow_rate={flow_rate}, pressure={pressure}")
    result = max(0.0, pressure - 1.1 * flow_rate)
    logger.debug(f"Safety margin result: {result}")
    return result
EOF

# Step 7: Verify with diff
git diff src/sim/gas_flow_model.py
```

---

## 3. Refactoring Workflow

### Scenario
Rename a function across multiple files.

### Steps

```bash
# Step 1: Find all occurrences
rt grep "old_function_name" src/ --json > matches.json

# Step 2: Review all matches
rt grep "old_function_name" src/

# Output:
# src/module_a.py:42: def old_function_name(x):
# src/module_b.py:89:     result = old_function_name(data)
# src/module_c.py:120:     value = old_function_name(input)

# Step 3: Edit definition in module_a
rt open src/module_a.py:42
rt edit src/module_a.py 42 44 << 'EOF'
def new_function_name(x):
    """Renamed for clarity."""
    return x * 2
EOF

# Step 4: Edit call site in module_b
rt open src/module_b.py:89
rt edit src/module_b.py 89 89 << 'EOF'
    result = new_function_name(data)
EOF

# Step 5: Edit call site in module_c
rt open src/module_c.py:120
rt edit src/module_c.py 120 120 << 'EOF'
    value = new_function_name(input)
EOF

# Step 6: Verify no remaining occurrences
rt grep "old_function_name" src/

# Step 7: Review all changes
git diff src/
```

---

## 4. Large File Navigation

### Scenario
Navigate a 5000-line configuration file to find specific settings.

### Steps

```bash
# Step 1: Check file size
wc -l config/settings.py

# Step 2: Search for relevant section
rt grep-file "DATABASE_SETTINGS" config/settings.py

# Output:
#   2341: DATABASE_SETTINGS = {

# Step 3: Jump to that section
rt open config/settings.py:2341

# Step 4: Scroll down to see more settings
rt scroll config/settings.py +
rt scroll config/settings.py +

# Step 5: Find specific sub-setting
rt grep-file "connection_timeout" config/settings.py

# Step 6: Jump to that line
rt open config/settings.py:2367

# Step 7: Edit the setting
rt edit config/settings.py 2367 2367 << 'EOF'
    "connection_timeout": 30,  # Increased from 10
EOF
```

---

## 5. Code Review Workflow

### Scenario
Review recent changes to understand modifications.

### Steps

```bash
# Step 1: Get list of recently modified files
git diff --name-only HEAD~1

# Step 2: For each file, find what changed
git diff HEAD~1 -- src/module.py | grep "^@@"

# Output:
# @@ -42,5 +42,8 @@ def process_data(input):

# Step 3: Jump to changed region
rt open src/module.py:42

# Step 4: Read the context
rt scroll src/module.py +

# Step 5: Search for related functions
rt grep-file "process_data" src/module.py

# Step 6: Check test coverage
rt grep "test_process_data" tests/
```

---

## 6. Multi-File Pattern Replacement

### Scenario
Update all copyright headers from 2024 to 2025.

### Steps

```bash
# Step 1: Find all files with copyright
rt grep "Copyright.*2024" . --ext py

# Step 2: For each file, open and edit
FILES=$(rt grep "Copyright.*2024" . --ext py -l)

for file in $FILES; do
    echo "Processing: $file"

    # Find exact line
    LINE=$(rt grep-file "Copyright.*2024" "$file" | cut -d: -f1 | head -1)

    # Open around that line
    rt open "$file:$LINE"

    # Edit the line
    rt edit "$file" "$LINE" "$LINE" << 'EOF'
# Copyright (c) 2025 Adservio Innovation Lab
EOF
done

# Step 3: Verify all changes
git diff
```

---

## 7. Documentation Update Workflow

### Scenario
Add docstrings to undocumented functions.

### Steps

```bash
# Step 1: Find functions without docstrings
rt grep -e "^def " src/ | while read match; do
    file=$(echo "$match" | cut -d: -f1)
    line=$(echo "$match" | cut -d: -f2)

    # Check if next line is docstring
    rt open "$file:$line" --chunk-size 5 | grep -q '"""' || echo "$file:$line"
done

# Step 2: For each undocumented function
rt open src/module.py:145

# Step 3: Insert docstring
rt insert src/module.py 146 << 'EOF'
    """
    Brief description of function.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description of return value
    """
EOF
```

---

## 8. Testing Integration

### Scenario
Add test cases for a newly added function.

### Steps

```bash
# Step 1: Find the test file
rt find "test_gas_flow" tests/

# Step 2: Find where to add new tests
rt grep-file "class.*Test" tests/test_gas_flow.py

# Step 3: Navigate to end of test class
rt open tests/test_gas_flow.py:250

# Step 4: Insert new test method
rt insert tests/test_gas_flow.py 251 << 'EOF'

    def test_safety_margin_edge_cases(self):
        """Test safety margin with edge cases."""
        # Test zero flow rate
        with self.assertRaises(ValueError):
            compute_safety_margin(0, 15.0)

        # Test negative pressure
        with self.assertRaises(ValueError):
            compute_safety_margin(10.0, -5.0)
EOF

# Step 5: Run tests
pytest tests/test_gas_flow.py::TestGasFlow::test_safety_margin_edge_cases
```

---

## 9. Environment-Aware Workflows

### Safe Read-Only Mode (Audit/Review)

```bash
# Enable safe mode
export UNIX_RAG_PROFILE=safe-read-only

# Navigation and search work normally
rt open src/module.py:100
rt grep-file "pattern" src/module.py

# Edit operations are blocked
rt edit src/module.py 100 105 << 'EOF'
...
EOF
# Error: Edit operations blocked in safe-read-only profile
```

### Development Mode (Default)

```bash
# Full access to all tools
export UNIX_RAG_PROFILE=dev

# All operations allowed
rt open src/module.py:100
rt edit src/module.py 100 105 << 'EOF'
...
EOF
```

### Disable SWE Tools Globally

```bash
# Temporarily disable all SWE tools
export RAGIX_ENABLE_SWE=0

rt open src/module.py
# Error: SWE tools are disabled (RAGIX_ENABLE_SWE=0)
```

---

## 10. Best Practices Summary

### DO:
- ✅ Use `rt open path:line` to jump directly to grep results
- ✅ Verify edits with `git diff` before committing
- ✅ Keep edit ranges small and focused
- ✅ Use `rt grep-file` for single-file searches (faster than recursive grep)
- ✅ Check file size with `wc -l` before opening large files
- ✅ Use `--show-diff` flag when you want immediate feedback

### DON'T:
- ❌ Don't repeatedly scroll when you know the line number
- ❌ Don't edit without first inspecting the region
- ❌ Don't skip the verification step after edits
- ❌ Don't edit binary files (they're automatically rejected)
- ❌ Don't use SWE tools on files outside the sandbox

---

## 11. Integration with Git

### Pre-commit Workflow

```bash
# Review all staged changes using SWE tools
for file in $(git diff --cached --name-only); do
    echo "=== $file ==="

    # Get changed line ranges
    git diff --cached -U0 "$file" | grep "^@@" | while read range; do
        # Extract line numbers
        LINE=$(echo "$range" | sed 's/.*+\([0-9]*\).*/\1/')

        # Review context
        rt open "$file:$LINE" --chunk-size 20
    done
done
```

### Post-commit Review

```bash
# Review last commit using SWE tools
git show --name-only --format="" HEAD | while read file; do
    rt open "$file:1" --chunk-size 50
done
```

---

## 12. Automation Examples

### Batch Operations

```bash
#!/bin/bash
# Update version strings across codebase

VERSION_FILES=$(rt grep "__version__" src/ -l)

for file in $VERSION_FILES; do
    LINE=$(rt grep-file "__version__" "$file" | cut -d: -f1 | head -1)
    rt edit "$file" "$LINE" "$LINE" << EOF
__version__ = "0.4.0"
EOF
done
```

### CI/CD Integration

```bash
# In CI pipeline: Use safe-read-only mode for static analysis
export UNIX_RAG_PROFILE=safe-read-only

# Generate code review report
rt grep "TODO\|FIXME\|XXX" src/ > review_report.txt
```

---

**For more information, see:**
- `SWE_TOOLING.md` — Full specification
- `README_RAGIX_TOOLS.md` — Command reference
- `QUICKSTART_CLAUDE_CODE.md` — Integration with Claude Code
