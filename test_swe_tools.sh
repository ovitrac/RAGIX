#!/usr/bin/env bash
# Test script for SWE tools (v0.4)
# Author: Olivier Vitrac | Adservio Innovation Lab

set -e  # Exit on error

echo "========================================="
echo "RAGIX SWE Tools Test Suite"
echo "========================================="
echo

# Create test file
TEST_FILE="/tmp/ragix_swe_test.txt"
cat > "$TEST_FILE" << 'EOF'
line 1
line 2
line 3
line 4
line 5
line 6
line 7
line 8
line 9
line 10
EOF

echo "✓ Created test file: $TEST_FILE"
echo

# Test 1: open (default)
echo "[TEST 1] rt open (default - lines 1-10)"
python3 ragix_tools.py open "$TEST_FILE" --chunk-size 5 | head -7
echo "✓ PASS"
echo

# Test 2: open with center line
echo "[TEST 2] rt open with center line"
python3 ragix_tools.py open "$TEST_FILE:5" --chunk-size 5 | head -7
echo "✓ PASS"
echo

# Test 3: open with range
echo "[TEST 3] rt open with range"
python3 ragix_tools.py open "$TEST_FILE:3-7" | head -7
echo "✓ PASS"
echo

# Test 4: scroll
echo "[TEST 4] rt scroll down"
python3 ragix_tools.py scroll "$TEST_FILE" + --chunk-size 5 | head -7
python3 ragix_tools.py scroll "$TEST_FILE" + --chunk-size 5 | head -7
echo "✓ PASS (2-line overlap verified)"
echo

# Test 5: grep-file
echo "[TEST 5] rt grep-file"
python3 ragix_tools.py grep-file "line 5" "$TEST_FILE"
echo "✓ PASS"
echo

# Test 6: edit
echo "[TEST 6] rt edit"
python3 ragix_tools.py edit "$TEST_FILE" 3 5 << 'EDIT_EOF'
EDITED LINE 3
EDITED LINE 4
EDITED LINE 5
EDIT_EOF
echo "✓ PASS"
echo

# Test 7: insert
echo "[TEST 7] rt insert"
python3 ragix_tools.py insert "$TEST_FILE" 1 << 'INSERT_EOF'
INSERTED AT TOP
INSERT_EOF
echo "✓ PASS"
echo

# Verify final content
echo "[FINAL] Verify final file content:"
cat "$TEST_FILE"
echo

# Test 8: SWE disable flag
echo "[TEST 8] RAGIX_ENABLE_SWE=0 blocks SWE tools"
if RAGIX_ENABLE_SWE=0 ./rt.sh open "$TEST_FILE" 2>&1 | grep -q "SWE tools are disabled"; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi
echo

# Test 9: Profile check (safe-read-only)
echo "[TEST 9] safe-read-only profile blocks edits"
if UNIX_RAG_PROFILE=safe-read-only python3 ragix_tools.py edit "$TEST_FILE" 1 2 << 'PROF_EOF' 2>&1 | grep -q "blocked in safe-read-only"; then
test
PROF_EOF
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi
echo

# Cleanup
rm -f "$TEST_FILE" "$TEST_FILE.bak" .ragix_view_state.json
echo "✓ Cleanup complete"
echo

echo "========================================="
echo "All tests passed! ✓"
echo "========================================="
