#!/usr/bin/env bash
# Generic wrapper for RAGIX_tools.py
# Usage: rt find ..., rt grep ..., rt stats ..., etc.

# Author: Olivier Vitrac | Adservio Innovation Lab | olivier.vitrac@adservio.fr
# Contact: olivier.vitrac@adservio.fr

RAGIX_TOOLS="${RAGIX_TOOLS:-RAGIX_tools.py}"

if [ ! -f "$RAGIX_TOOLS" ]; then
  echo "Error: RAGIX_TOOLS not found at '$RAGIX_TOOLS'." >&2
  echo "Set RAGIX_TOOLS=/path/to/RAGIX_tools.py" >&2
  exit 1
fi

python3 "$RAGIX_TOOLS" "$@"
