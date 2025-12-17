#!/bin/bash
# Security Kernels Demo Script
# Demonstrates RAGIX security network audit capabilities
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
#
# Usage:
#   ./run_security_demo.sh                  # Interactive menu
#   ./run_security_demo.sh local_network    # Run specific demo
#   ./run_security_demo.sh web_audit
#   ./run_security_demo.sh compliance
#   ./run_security_demo.sh config

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAGIX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          RAGIX Security Network Audit Demo                    ║"
echo "║                                                               ║"
echo "║  Kernel-Orchestrated Audit System (KOAS)                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    local missing=0

    # Check nmap
    if command -v nmap &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} nmap installed"
    else
        echo -e "  ${RED}✗${NC} nmap not found (sudo apt install nmap)"
        missing=1
    fi

    # Check nikto
    if command -v nikto &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} nikto installed"
    else
        echo -e "  ${YELLOW}○${NC} nikto not found (optional: sudo apt install nikto)"
    fi

    # Check nuclei
    if command -v nuclei &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} nuclei installed"
    else
        echo -e "  ${YELLOW}○${NC} nuclei not found (optional for vuln scanning)"
    fi

    # Check Python packages
    if python3 -c "import nmap" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} python-nmap installed"
    else
        echo -e "  ${RED}✗${NC} python-nmap not found (pip install python-nmap)"
        missing=1
    fi

    echo ""

    if [ $missing -eq 1 ]; then
        echo -e "${RED}Some required tools are missing. Install them and try again.${NC}"
        exit 1
    fi
}

# Function to run local network scan
run_local_network() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Demo: Local Network Scan${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This demo scans localhost to discover services."
    echo "For a real network scan, edit: examples/security/local_network/data/targets.yaml"
    echo ""

    cd "$SCRIPT_DIR/local_network"

    echo -e "${YELLOW}Running Stage 1: Discovery...${NC}"
    python -m ragix_kernels.orchestrator run -w . -s 1

    echo ""
    echo -e "${GREEN}Results:${NC}"

    if [ -f "stage1/net_discover.json" ]; then
        echo -e "\n${YELLOW}Discovered Hosts:${NC}"
        python3 -c "
import json
with open('stage1/net_discover.json') as f:
    data = json.load(f).get('data', {})
    hosts = data.get('hosts', [])
    print(f'  Found {len(hosts)} host(s)')
    for h in hosts[:5]:
        print(f'    - {h.get(\"ip\")} ({h.get(\"hostname\", \"unknown\")})')
"
    fi

    if [ -f "stage1/port_scan.json" ]; then
        echo -e "\n${YELLOW}Discovered Services:${NC}"
        python3 -c "
import json
with open('stage1/port_scan.json') as f:
    data = json.load(f).get('data', {})
    services = data.get('services', [])
    print(f'  Found {len(services)} service(s)')
    for s in services[:10]:
        print(f'    - {s.get(\"host\")}:{s.get(\"port\")} ({s.get(\"service\", \"unknown\")})')
"
    fi

    echo ""
    echo -e "${GREEN}Output files:${NC}"
    ls -la stage1/*.json 2>/dev/null || echo "  (no output files)"
}

# Function to run config audit
run_config_audit() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Demo: Firewall Configuration Audit${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This demo analyzes sample firewall configurations for security issues."
    echo ""

    cd "$SCRIPT_DIR/config_audit"

    echo -e "${YELLOW}Sample configs in data/configs/:${NC}"
    ls -la data/configs/
    echo ""

    echo -e "${YELLOW}Running config_parse kernel...${NC}"
    python -m ragix_kernels.orchestrator run -w . -s 1 -k config_parse

    echo ""
    echo -e "${GREEN}Results:${NC}"

    if [ -f "stage1/config_parse.json" ]; then
        echo -e "\n${YELLOW}Configuration Analysis:${NC}"
        python3 -c "
import json
with open('stage1/config_parse.json') as f:
    data = json.load(f).get('data', {})
    devices = data.get('devices', [])
    rules = data.get('rules', [])
    findings = data.get('findings', [])

    print(f'  Devices parsed: {len(devices)}')
    print(f'  Rules analyzed: {len(rules)}')
    print(f'  Security findings: {len(findings)}')

    print()
    print('  Findings by severity:')
    severities = {}
    for f in findings:
        sev = f.get('severity', 'unknown')
        severities[sev] = severities.get(sev, 0) + 1
    for sev, count in sorted(severities.items()):
        print(f'    - {sev}: {count}')

    if findings:
        print()
        print('  Top issues:')
        for f in findings[:5]:
            print(f'    [{f.get(\"severity\", \"?\")}] {f.get(\"description\", \"Unknown\")}')
"
    fi
}

# Function to run web audit
run_web_audit() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Demo: Web Application Audit${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This demo performs a web security assessment."
    echo "Edit examples/security/web_audit/data/targets.yaml to scan real targets."
    echo ""
    echo -e "${YELLOW}WARNING: Only scan websites you are authorized to test!${NC}"
    echo ""

    cd "$SCRIPT_DIR/web_audit"

    echo -e "${YELLOW}Running Stages 1-3...${NC}"
    python -m ragix_kernels.orchestrator run -w . || true

    echo ""
    echo -e "${GREEN}Output files:${NC}"
    find . -name "*.json" -o -name "*.md" 2>/dev/null | grep -E "stage[123]" | head -20

    if [ -f "stage3/security_report.md" ]; then
        echo ""
        echo -e "${YELLOW}Security Report Preview:${NC}"
        head -50 stage3/security_report.md
    fi
}

# Function to run compliance check
run_compliance() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Demo: Compliance Assessment (ANSSI/NIST/CIS)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This demo evaluates security compliance against:"
    echo "  - ANSSI (French National Cybersecurity Agency)"
    echo "  - NIST Cybersecurity Framework"
    echo "  - CIS Controls v8"
    echo ""

    cd "$SCRIPT_DIR/compliance_check"

    echo -e "${YELLOW}Running Stages 1-2...${NC}"
    python -m ragix_kernels.orchestrator run -w . -s 1 || true
    python -m ragix_kernels.orchestrator run -w . -s 2 || true

    echo ""
    echo -e "${GREEN}Results:${NC}"

    if [ -f "stage2/compliance.json" ]; then
        echo -e "\n${YELLOW}Compliance Scores:${NC}"
        python3 -c "
import json
with open('stage2/compliance.json') as f:
    data = json.load(f).get('data', {})
    scores = data.get('compliance_scores', {})
    stats = data.get('statistics', {})

    print(f'  Overall Score: {stats.get(\"overall_score\", 0)}%')
    print()
    print('  By Framework:')
    for fw, result in scores.items():
        name = result.get('framework', fw)
        score = result.get('score', 0)
        compliant = result.get('compliant', 0)
        total = result.get('total_controls', 0)
        print(f'    - {name}: {score}% ({compliant}/{total} controls)')

    findings = data.get('findings', [])
    if findings:
        print()
        print('  Non-compliant items:')
        for f in findings[:5]:
            print(f'    [{f.get(\"framework\")}] {f.get(\"title\", \"Unknown\")}')
"
    fi
}

# Interactive menu
show_menu() {
    echo ""
    echo -e "${YELLOW}Select a demo to run:${NC}"
    echo ""
    echo "  1) Local Network Scan     - Discover hosts and services"
    echo "  2) Config Audit           - Analyze firewall configurations"
    echo "  3) Web Application Audit  - Web security assessment"
    echo "  4) Compliance Check       - ANSSI/NIST/CIS evaluation"
    echo "  5) Run All Demos"
    echo "  q) Quit"
    echo ""
    read -p "Enter choice [1-5, q]: " choice

    case $choice in
        1) run_local_network ;;
        2) run_config_audit ;;
        3) run_web_audit ;;
        4) run_compliance ;;
        5)
            run_local_network
            echo ""
            run_config_audit
            echo ""
            # Skip web_audit and compliance by default (need real targets)
            echo -e "${YELLOW}Skipping web_audit and compliance (edit targets.yaml first)${NC}"
            ;;
        q|Q) exit 0 ;;
        *) echo -e "${RED}Invalid choice${NC}"; show_menu ;;
    esac
}

# Main
cd "$RAGIX_ROOT"

# Handle command line argument
if [ $# -gt 0 ]; then
    case $1 in
        local_network|local|network)
            check_prerequisites
            run_local_network
            ;;
        config|config_audit|firewall)
            run_config_audit
            ;;
        web|web_audit)
            check_prerequisites
            run_web_audit
            ;;
        compliance|compliance_check)
            check_prerequisites
            run_compliance
            ;;
        all)
            check_prerequisites
            run_local_network
            run_config_audit
            ;;
        *)
            echo "Usage: $0 [local_network|config|web_audit|compliance|all]"
            exit 1
            ;;
    esac
else
    check_prerequisites
    show_menu
fi

echo ""
echo -e "${GREEN}Demo complete!${NC}"
echo -e "See the ${YELLOW}examples/security/README.md${NC} for more details."
