#!/usr/bin/env bash
# Generic wrapper for ragix_tools.py (v0.4 with SWE extensions)
# Usage: rt find ..., rt grep ..., rt open ..., rt edit ..., etc.

# Author: Olivier Vitrac | Adservio Innovation Lab | olivier.vitrac@adservio.fr
# Contact: olivier.vitrac@adservio.fr

RAGIX_TOOLS="${RAGIX_TOOLS:-ragix_tools.py}"
RAGIX_ENABLE_SWE="${RAGIX_ENABLE_SWE:-1}"

if [ ! -f "$RAGIX_TOOLS" ]; then
  echo "Error: RAGIX_TOOLS not found at '$RAGIX_TOOLS'." >&2
  echo "Set RAGIX_TOOLS=/path/to/ragix_tools.py" >&2
  exit 1
fi

# Check if SWE tools are disabled
if [ "$1" = "open" ] || [ "$1" = "scroll" ] || [ "$1" = "grep-file" ] || [ "$1" = "edit" ] || [ "$1" = "insert" ]; then
  if [ "$RAGIX_ENABLE_SWE" = "0" ]; then
    echo "Error: SWE tools are disabled (RAGIX_ENABLE_SWE=0)" >&2
    echo "Set RAGIX_ENABLE_SWE=1 to enable SWE navigation and editing tools" >&2
    exit 1
  fi
fi

python3 "$RAGIX_TOOLS" "$@"
