#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# release.sh
#
# Helper to create a tagged GitHub release for RAGIX.
#
# - Reads the version from ragix_core/version.py (single source of truth)
# - Optionally takes a Markdown notes file for the release body
# - Uses gh CLI to create the release on GitHub
#
# Usage:
#   ./release.sh                 # auto-detect version, ask confirmation, generate notes
#   ./release.sh -n NOTES.md     # same, but use NOTES.md as release body
#   ./release.sh -y -n NOTES.md  # no confirmation, use NOTES.md
#
# Environment:
#   PYTHON   Python interpreter to use (default: python3)
#
#
# Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

NOTES_FILE=""
AUTO_YES=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -n FILE   Use FILE as Markdown release notes (gh --notes-file FILE)
  -y        Do not ask for interactive confirmation (non-interactive / CI)
  -h        Show this help message and exit

Behavior:
  - Reads RAGIX version from ragix_core/version.py (__version__).
  - Creates a git tag "v<version>" if it doesn't already exist.
  - Pushes branch and tag to origin.
  - Creates a GitHub release via gh:
      - If -n is given, uses --notes-file FILE
      - Otherwise, uses --generate-notes

Examples:
  $(basename "$0")
  $(basename "$0") -n NOTES-v0.33.0.md
  $(basename "$0") -y -n NOTES-v0.33.0.md

EOF
}

while getopts ":n:yh" opt; do
    case "$opt" in
        n)
            NOTES_FILE="$OPTARG"
            ;;
        y)
            AUTO_YES=1
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Error: invalid option -$OPTARG" >&2
            usage
            exit 1
            ;;
        :)
            echo "Error: option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

# ---------------------------------------------------------------------------
# 1. Read version from ragix_core/version.py via Python
# ---------------------------------------------------------------------------

VERSION="$("$PYTHON" - <<'EOF'
import os, sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

try:
    from ragix_core.version import __version__
except Exception as e:
    sys.stderr.write(f"Error importing ragix_core.version: {e}\n")
    sys.exit(1)

print(__version__)
EOF
)"

if [ -z "$VERSION" ]; then
    echo "Error: could not read version from ragix_core/version.py" >&2
    exit 1
fi

TAG="v${VERSION}"

echo "Detected RAGIX version: ${VERSION}"
echo "Will use git tag: ${TAG}"

# ---------------------------------------------------------------------------
# 2. Check notes file if provided
# ---------------------------------------------------------------------------

if [ -n "$NOTES_FILE" ]; then
    if [ ! -f "$NOTES_FILE" ]; then
        echo "Error: notes file not found: $NOTES_FILE" >&2
        exit 1
    fi
    echo "Using notes file: $NOTES_FILE"
fi

# ---------------------------------------------------------------------------
# 3. Safety checks: working tree clean, gh present, etc.
# ---------------------------------------------------------------------------

if ! git diff --quiet; then
    echo "Error: working tree is not clean. Commit or stash changes first." >&2
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: gh CLI is not installed or not in PATH." >&2
    exit 1
fi

# Optional: check that we're on main / master or a release branch (soft check)
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Current branch: $CURRENT_BRANCH"

# ---------------------------------------------------------------------------
# 4. Show recent commits and ask for confirmation (unless -y)
# ---------------------------------------------------------------------------

echo
echo "Last commits:"
git --no-pager log -5 --oneline
echo

if [ "$AUTO_YES" -eq 0 ]; then
    read -rp "Proceed to create tag ${TAG} and GitHub release? [y/N] " ans
    if [[ "${ans:-N}" != [yY] ]]; then
        echo "Aborted by user."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 5. Create tag if it does not exist
# ---------------------------------------------------------------------------

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Tag ${TAG} already exists (will reuse it)."
else
    git tag -a "$TAG" -m "RAGIX ${VERSION}"
    echo "Created tag ${TAG}."
fi

# ---------------------------------------------------------------------------
# 6. Push branch and tag
# ---------------------------------------------------------------------------

echo "Pushing current branch and tag ${TAG} to origin..."
git push
git push origin "$TAG"

# ---------------------------------------------------------------------------
# 7. Create GitHub release via gh
# ---------------------------------------------------------------------------

echo "Creating GitHub release for ${TAG}..."

if [ -n "$NOTES_FILE" ]; then
    gh release create "$TAG" \
        --title "RAGIX ${VERSION} – Contractive Reasoning & Agentic Profiles" \
        --notes-file "$NOTES_FILE"
else
    gh release create "$TAG" \
        --title "RAGIX ${VERSION} – Contractive Reasoning & Agentic Profiles" \
        --generate-notes
fi

echo "Release ${TAG} created."

