#!/bin/bash
# =============================================================================
# KOAS MCP Demo Launcher
# =============================================================================
#
# Starts the KOAS MCP demo server and opens it in the default browser.
#
# Usage:
#   ./run_demo.sh              # Start on default port 8080
#   ./run_demo.sh --port 9000  # Start on custom port
#   ./run_demo.sh --no-browser # Don't open browser automatically
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PORT=8080
HOST="127.0.0.1"
OPEN_BROWSER=true
RELOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --host|-h)
            HOST="$2"
            shift 2
            ;;
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        --reload|-r)
            RELOAD=true
            shift
            ;;
        --help)
            echo "KOAS MCP Demo Launcher"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port, -p PORT     Port to bind to (default: 8080)"
            echo "  --host, -h HOST     Host to bind to (default: 127.0.0.1)"
            echo "  --no-browser        Don't open browser automatically"
            echo "  --reload, -r        Enable auto-reload for development"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    KOAS MCP Demo Launcher                     ║"
echo "║                       Version 0.62.0                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

MISSING_DEPS=()

if ! check_package "fastapi"; then
    MISSING_DEPS+=("fastapi")
fi

if ! check_package "uvicorn"; then
    MISSING_DEPS+=("uvicorn")
fi

if ! check_package "httpx"; then
    MISSING_DEPS+=("httpx")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing dependencies: ${MISSING_DEPS[*]}${NC}"
    pip install "${MISSING_DEPS[@]}" --quiet
fi

echo -e "${GREEN}✓ All dependencies available${NC}"

# Check Ollama
echo -e "${YELLOW}Checking Ollama...${NC}"
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${YELLOW}⚠ Ollama not running (chat feature will be limited)${NC}"
    echo -e "${YELLOW}  Start with: ollama serve${NC}"
fi

# URL
URL="http://${HOST}:${PORT}"

echo ""
echo -e "${GREEN}Starting server on ${URL}${NC}"
echo ""

# Open browser after a delay (in background)
if [ "$OPEN_BROWSER" = true ]; then
    (
        sleep 2
        if command -v xdg-open &> /dev/null; then
            xdg-open "$URL" 2>/dev/null || true
        elif command -v open &> /dev/null; then
            open "$URL" 2>/dev/null || true
        fi
    ) &
fi

# Build uvicorn command
cd "$SCRIPT_DIR"

UVICORN_CMD="python3 -m uvicorn server:app --host $HOST --port $PORT"

if [ "$RELOAD" = true ]; then
    UVICORN_CMD="$UVICORN_CMD --reload"
fi

# Run the server
exec $UVICORN_CMD
