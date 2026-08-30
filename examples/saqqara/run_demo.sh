#!/usr/bin/env bash
# The whole demo, on a clean install of .[saqqara,dev] and nothing else.
#
# The dense lane is optional: 03 skips it with a printed reason when no embedder
# is importable. Every other step must succeed — this script is what CI runs.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${1:-$(mktemp -d)}"
PY="${PYTHON:-python3}"

echo "workspace: $WORKSPACE"
for step in 01_read_tree 02_index_store 03_search_with_trace 04_lexical_only; do
    echo
    echo "=============================================================="
    echo "  $step"
    echo "=============================================================="
    "$PY" "$HERE/$step.py" "$WORKSPACE"
done

echo
echo "done. Nothing was downloaded, no document was committed, and every"
echo "hit above can name the node it came from."
