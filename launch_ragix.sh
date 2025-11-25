#!/bin/bash
# =============================================================================
# RAGIX Launcher
# =============================================================================
#
# Retrieval-Augmented Generative Interactive eXecution Agent
# A sovereign, local-first development assistant using Unix-RAG patterns.
#
# This launcher:
#   1. Initializes Conda
#   2. Creates/activates ragix-env environment
#   3. Checks and installs dependencies
#   4. Verifies Ollama and available LLM models
#   5. Launches the RAGIX GUI or selected component
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
#
# Usage:
#   ./launch_ragix.sh           # Interactive menu
#   ./launch_ragix.sh gui       # Launch GUI directly
#   ./launch_ragix.sh demo      # Run Claude demo
#   ./launch_ragix.sh mcp       # Start MCP server
#   ./launch_ragix.sh test      # Run LLM backend test
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

ENV_NAME="ragix-env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# Banner
# =============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${BOLD}${GREEN}RAGIX${NC} v0.7 — Retrieval-Augmented Generative IX Agent           ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${MAGENTA}Sovereign • Local-First • Unix-RAG Patterns${NC}                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${BLUE}Adservio Innovation Lab${NC}                                        ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
# Conda Initialization
# =============================================================================

condainit() {
    # Try common conda locations using ~ for portability
    local CONDA_PATHS=(
        ~/anaconda3
        ~/miniconda3
        ~/miniforge3
        /opt/conda
        /opt/anaconda3
    )

    local CONDA_BASE=""
    for path in "${CONDA_PATHS[@]}"; do
        if [ -d "$path" ]; then
            CONDA_BASE="$path"
            break
        fi
    done

    if [ -z "$CONDA_BASE" ]; then
        echo -e "${RED}Error: Could not find Conda installation.${NC}"
        echo "Searched: ${CONDA_PATHS[*]}"
        return 1
    fi

    __conda_setup="$("$CONDA_BASE/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            . "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            export PATH="$CONDA_BASE/bin:$PATH"
        fi
    fi
    unset __conda_setup

    export CONDA_BASE
}

# =============================================================================
# Environment Management
# =============================================================================

check_env_exists() {
    conda env list | grep -q "^${ENV_NAME} "
    return $?
}

create_environment() {
    echo -e "${YELLOW}Creating new conda environment: ${ENV_NAME}...${NC}"

    if [ -f "$SCRIPT_DIR/environment.yaml" ]; then
        conda env create -f "$SCRIPT_DIR/environment.yaml"
    else
        echo -e "${YELLOW}No environment.yaml found, creating minimal environment...${NC}"
        conda create -n "$ENV_NAME" python=3.11 pip -y
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Environment created successfully"
        return 0
    else
        echo -e "${RED}Error: Failed to create environment${NC}"
        return 1
    fi
}

install_dependencies() {
    echo -e "${YELLOW}Checking/installing dependencies...${NC}"

    # Install from requirements.txt
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Dependencies installed"
        else
            echo -e "${YELLOW}⚠${NC}  Some dependencies may have failed to install"
        fi
    fi

    # Install ragix_core in development mode
    if [ -f "$SCRIPT_DIR/pyproject.toml" ] || [ -d "$SCRIPT_DIR/ragix_core" ]; then
        pip install -e "$SCRIPT_DIR" --quiet 2>/dev/null || true
    fi
}

# =============================================================================
# Ollama Checks
# =============================================================================

check_ollama() {
    echo -e "${BLUE}[Ollama]${NC} Checking status..."

    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Ollama is running"

        # Get model information
        local MODELS_JSON=$(curl -s http://localhost:11434/api/tags)
        local MODEL_COUNT=$(echo "$MODELS_JSON" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('models', [])))" 2>/dev/null)

        if [ ! -z "$MODEL_COUNT" ] && [ "$MODEL_COUNT" -gt 0 ]; then
            echo -e "   Available models: ${GREEN}${MODEL_COUNT}${NC}"
            echo ""
            echo -e "   ${BOLD}Models:${NC}"
            echo "$MODELS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', [])[:10]:  # Show max 10
    name = m['name']
    size_gb = m.get('size', 0) / 1e9
    # Sovereignty indicator
    print(f'     🟢 {name} ({size_gb:.1f} GB)')
" 2>/dev/null

            if [ "$MODEL_COUNT" -gt 10 ]; then
                echo -e "     ... and $((MODEL_COUNT - 10)) more"
            fi
        else
            echo -e "${YELLOW}⚠${NC}  No models found. Install one with:"
            echo "     ollama pull mistral"
            echo "     ollama pull granite3.1-moe:3b"
        fi

        return 0
    else
        echo -e "${YELLOW}⚠${NC}  Ollama not detected at localhost:11434"
        echo ""
        echo -e "   To start Ollama:"
        echo "     ollama serve"
        echo ""
        echo -e "   To install a model:"
        echo "     ollama pull mistral"
        echo ""
        return 1
    fi
}

# =============================================================================
# Component Launchers
# =============================================================================

launch_gui() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  Launching RAGIX Web Interface                                  ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  URL: ${CYAN}http://localhost:8501${NC}                                     ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""

    streamlit run "$SCRIPT_DIR/ragix_app.py"
}

launch_demo() {
    echo ""
    echo -e "${BLUE}Running Claude Demo...${NC}"
    echo ""
    python3 "$SCRIPT_DIR/examples/claude_demo.py"
}

launch_mcp() {
    echo ""
    echo -e "${BLUE}Starting MCP Server...${NC}"
    echo ""
    python3 "$SCRIPT_DIR/MCP/ragix_mcp_server.py"
}

launch_test() {
    echo ""
    echo -e "${BLUE}Running LLM Backend Test...${NC}"
    echo ""
    bash "$SCRIPT_DIR/examples/test_llm_backends.sh"
}

# =============================================================================
# Interactive Menu
# =============================================================================

show_menu() {
    echo ""
    echo -e "${BOLD}What would you like to do?${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} Launch Web GUI         ${CYAN}(Streamlit interface)${NC}"
    echo -e "  ${GREEN}2)${NC} Run Claude Demo        ${CYAN}(Feature demonstration)${NC}"
    echo -e "  ${GREEN}3)${NC} Start MCP Server       ${CYAN}(For Claude Desktop/Code)${NC}"
    echo -e "  ${GREEN}4)${NC} Test LLM Backends      ${CYAN}(Compare Ollama models)${NC}"
    echo -e "  ${GREEN}5)${NC} Python Shell           ${CYAN}(Interactive RAGIX)${NC}"
    echo -e "  ${GREEN}6)${NC} Check Ollama Status    ${CYAN}(View available models)${NC}"
    echo -e "  ${GREEN}q)${NC} Quit"
    echo ""
    read -p "Select option [1-6, q]: " choice

    case $choice in
        1) launch_gui ;;
        2) launch_demo ;;
        3) launch_mcp ;;
        4) launch_test ;;
        5)
            echo ""
            echo -e "${BLUE}Starting Python shell with RAGIX...${NC}"
            echo -e "${YELLOW}Try: from ragix_core import *${NC}"
            echo ""
            python3
            ;;
        6)
            echo ""
            check_ollama
            echo ""
            show_menu
            ;;
        q|Q)
            echo ""
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            show_menu
            ;;
    esac
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    print_banner

    # Step 1: Initialize Conda
    echo -e "${YELLOW}[1/4]${NC} Initializing Conda..."
    condainit

    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: Conda not found after initialization.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Conda initialized ($(conda --version))"
    echo ""

    # Step 2: Check/Create Environment
    echo -e "${YELLOW}[2/4]${NC} Checking environment: ${ENV_NAME}..."

    if ! check_env_exists; then
        echo -e "${YELLOW}Environment not found. Creating...${NC}"
        create_environment
    fi

    # Activate environment
    conda activate "$ENV_NAME"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to activate ${ENV_NAME} environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Environment activated: ${ENV_NAME}"
    echo -e "   Python: $(python3 --version 2>&1 | head -1)"
    echo ""

    # Step 3: Check Dependencies
    echo -e "${YELLOW}[3/4]${NC} Checking dependencies..."
    install_dependencies
    echo ""

    # Step 4: Check Ollama
    echo -e "${YELLOW}[4/4]${NC} Checking Ollama..."
    check_ollama
    echo ""

    # Handle command-line arguments or show menu
    case "${1:-}" in
        gui)
            launch_gui
            ;;
        demo)
            launch_demo
            ;;
        mcp)
            launch_mcp
            ;;
        test)
            launch_test
            ;;
        "")
            show_menu
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [gui|demo|mcp|test]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

# Cleanup message (shown after Ctrl+C or exit)
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  RAGIX session ended.                                            ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Thank you for using RAGIX!                                      ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  ${BLUE}https://github.com/ovitrac/RAGIX${NC}                               ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
