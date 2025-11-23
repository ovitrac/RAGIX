#!/usr/bin/env bash
# Wrapper for ragix_tools.py find command
# Author: Olivier Vitrac | Adservio Innovation Lab | olivier.vitrac@adservio.fr

RAGIX_TOOLS="${RAGIX_TOOLS:-ragix_tools.py}"
python3 "$RAGIX_TOOLS" find "$@"

