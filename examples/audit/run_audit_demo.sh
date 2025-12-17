#!/bin/bash
# =============================================================================
# RAGIX Code Audit Demo Script
# Interactive demonstration of KOAS audit kernels
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAGIX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}$1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${CYAN}┌──────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│${NC}  ${BOLD}$1${NC}"
    echo -e "${CYAN}└──────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC}  $1"
}

print_item() {
    echo -e "   ${MAGENTA}•${NC} $1"
}

# =============================================================================
# Banner
# =============================================================================

show_banner() {
    clear
    echo -e "${CYAN}"
    echo "  ╦═╗╔═╗╔═╗╦═╗ ╦   ╔═╗╦ ╦╔╦╗╦╔╦╗   ╔╦╗╔═╗╔╦╗╔═╗"
    echo "  ╠╦╝╠═╣║ ╦║╔╩╦╝   ╠═╣║ ║ ║║║ ║     ║║║╠╣ ║║║║ ║"
    echo "  ╩╚═╩ ╩╚═╝╩╩ ╚    ╩ ╩╚═╝═╩╝╩ ╩    ═╩╝╚═╝╩ ╩╚═╝"
    echo -e "${NC}"
    echo -e "  ${BOLD}KOAS - Kernel-Orchestrated Audit System${NC}"
    echo -e "  ${CYAN}Code Analysis & Risk Assessment Demos${NC}"
    echo ""
    echo -e "  ${YELLOW}Based on IOWIZME/SIAS Architecture${NC}"
    echo ""
}

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    print_section "Checking Prerequisites"

    local all_ok=true

    # Check Python
    if command -v python3 &> /dev/null; then
        local py_ver=$(python3 --version 2>&1)
        print_success "Python: $py_ver"
    else
        print_error "Python 3 not found"
        all_ok=false
    fi

    # Check RAGIX installation
    if python3 -c "import ragix_kernels" &> /dev/null 2>&1; then
        print_success "ragix_kernels module available"
    else
        print_warning "ragix_kernels not installed - will use local path"
        export PYTHONPATH="$RAGIX_ROOT:$PYTHONPATH"
    fi

    # Check data files
    local data_files=(
        "$SCRIPT_DIR/volumetry_analysis/data/volumetry.yaml"
        "$SCRIPT_DIR/volumetry_analysis/data/code_metrics.yaml"
        "$SCRIPT_DIR/microservices/data/modules.yaml"
        "$SCRIPT_DIR/java_monolith/data/structure.yaml"
        "$SCRIPT_DIR/full_audit/data/system_inventory.yaml"
    )

    local data_ok=true
    for f in "${data_files[@]}"; do
        if [[ -f "$f" ]]; then
            print_success "Found: $(basename "$f")"
        else
            print_error "Missing: $f"
            data_ok=false
        fi
    done

    if [[ "$all_ok" == "false" ]]; then
        print_error "Prerequisites not met. Please install missing components."
        exit 1
    fi

    print_success "All prerequisites satisfied"
}

# =============================================================================
# Demo Functions
# =============================================================================

# Demo 1: Volumetry Analysis
demo_volumetry() {
    print_header "Demo 1: Volumetry-Weighted Risk Analysis"

    print_info "This demo shows risk assessment weighted by operational traffic volume"
    print_info "Using IOWIZME production data: 4M messages/day, peak at 05:00 UTC"
    echo ""

    print_section "IOWIZME Traffic Patterns"
    echo -e "  ${BOLD}Daily Flow Summary:${NC}"
    print_item "SIAS Message Ingestion: 4,000,000 msg/day"
    print_item "Internal Routing: 4,000,000 msg/day"
    print_item "Business Processing: 4,000,000 msg/day"
    print_item "Batch Processing: 10,000 records/day"
    echo ""
    echo -e "  ${BOLD}Peak Characteristics:${NC}"
    print_item "Peak Hour: 05:00 UTC (overnight processing)"
    print_item "Peak Rate: ~1,000 msg/sec"
    print_item "Peak Multiplier: 3.5x average"
    echo ""

    print_section "Risk Calculation Formula"
    echo -e "  ${CYAN}Risk = (LOC × 0.25) + (Complexity × 0.25) + (Volumetry × 0.50)${NC}"
    echo ""
    print_item "LOC: Lines of code (normalized 0-10)"
    print_item "Complexity: Cyclomatic complexity (normalized 0-10)"
    print_item "Volumetry: Traffic volume impact (normalized 0-10)"
    echo ""

    print_section "Module Risk Scores (Simulated)"
    echo ""
    printf "  ${BOLD}%-25s %8s %10s %10s %8s %8s${NC}\n" "Module" "LOC" "Complexity" "Volumetry" "Risk" "Level"
    echo "  ─────────────────────────────────────────────────────────────────────"
    printf "  %-25s %8d %10.1f %10d %8.1f ${RED}%8s${NC}\n" "iog-support-commons" 13290 8.5 4000000 "8.7" "CRITICAL"
    printf "  %-25s %8d %10.1f %10d %8.1f ${RED}%8s${NC}\n" "iow-ech-sias" 1430 12.3 4000000 "7.2" "HIGH"
    printf "  %-25s %8d %10.1f %10d %8.1f ${YELLOW}%8s${NC}\n" "iow-ioc-sc02" 1570 15.2 4000000 "6.8" "HIGH"
    printf "  %-25s %8d %10.1f %10d %8.1f ${YELLOW}%8s${NC}\n" "iow-iok-sk01" 990 18.5 4000000 "6.5" "HIGH"
    printf "  %-25s %8d %10.1f %10d %8.1f ${GREEN}%8s${NC}\n" "iow-iok-sk04" 14350 6.2 10000 "3.2" "MEDIUM"
    printf "  %-25s %8d %10.1f %10d %8.1f ${GREEN}%8s${NC}\n" "iow-iog-models" 4500 2.1 0 "2.1" "LOW"
    echo ""

    print_section "Critical Path Identified"
    echo -e "  ${RED}iow-ech-sias${NC} → ${YELLOW}iow-ioc-sc02${NC} → ${YELLOW}iow-iok-sk01${NC} → ${RED}iog-support-commons${NC}"
    echo ""
    print_info "This path handles 100% of production traffic"
    print_info "Any failure impacts 4M daily messages"
    echo ""

    print_success "Volumetry analysis demo complete"
}

# Demo 2: Microservices Analysis
demo_microservices() {
    print_header "Demo 2: Microservices Architecture Analysis"

    print_info "Analyzing IOWIZME as a microservices decomposition"
    echo ""

    print_section "Service Catalog"
    echo ""
    printf "  ${BOLD}%-20s %-15s %-12s %8s %10s${NC}\n" "Service" "Layer" "Technology" "LOC" "Port"
    echo "  ────────────────────────────────────────────────────────────────────"
    printf "  %-20s %-15s %-12s %8d %10d\n" "api-gateway" "edge" "Spring Cloud" 850 8080
    printf "  %-20s %-15s %-12s %8d %10d\n" "auth-service" "security" "Spring Sec" 2200 8081
    printf "  %-20s %-15s %-12s %8d %10d\n" "message-gateway" "integration" "Spring Int" 1430 8082
    printf "  %-20s %-15s %-12s %8d %10d\n" "message-router" "orchestration" "Apache Camel" 1570 8083
    printf "  %-20s %-15s %-12s %8d %10d\n" "business-processor" "business" "Spring Boot" 990 8084
    printf "  %-20s %-15s %-12s %8d %10d\n" "batch-processor" "batch" "Spring Batch" 14350 8085
    printf "  %-20s %-15s %-12s %8d %10d\n" "data-service" "persistence" "Spring Data" 3200 8086
    printf "  %-20s %-15s %-12s %8d %10s\n" "shared-libs" "shared" "Java Lib" 13290 "-"
    printf "  %-20s %-15s %-12s %8d %10s\n" "shared-models" "shared" "Java+Lombok" 4500 "-"
    echo ""

    print_section "Dependency Graph"
    echo ""
    echo "  ┌─────────────────┐"
    echo "  │   api-gateway   │"
    echo "  └────────┬────────┘"
    echo "           │"
    echo "           ▼"
    echo "  ┌─────────────────┐     ┌─────────────────┐"
    echo "  │  auth-service   │────▶│   redis-cache   │"
    echo "  └────────┬────────┘     └─────────────────┘"
    echo "           │"
    echo "           ▼"
    echo "  ┌─────────────────┐"
    echo "  │ message-gateway │◀──── External Partners (SIAS)"
    echo "  └────────┬────────┘"
    echo "           │"
    echo "           ▼"
    echo "  ┌─────────────────┐     ┌─────────────────┐"
    echo "  │ message-router  │────▶│    event-bus    │ (Kafka)"
    echo "  └────────┬────────┘     └─────────────────┘"
    echo "           │"
    echo "      ┌────┴────┐"
    echo "      ▼         ▼"
    echo "  ┌────────┐ ┌────────────┐"
    echo "  │business│ │   batch    │"
    echo "  │processor│ │ processor │"
    echo "  └───┬────┘ └─────┬──────┘"
    echo "      │            │"
    echo "      └─────┬──────┘"
    echo "            ▼"
    echo "  ┌─────────────────┐     ┌─────────────────┐"
    echo "  │  data-service   │────▶│   postgresql    │"
    echo "  └─────────────────┘     └─────────────────┘"
    echo ""

    print_section "Shared Library Impact"
    echo ""
    echo -e "  ${BOLD}shared-libs${NC} (13,290 LOC) is used by:"
    print_item "message-gateway"
    print_item "message-router"
    print_item "business-processor"
    print_item "batch-processor"
    print_item "data-service"
    echo ""
    print_warning "High fan-out: changes affect 5 services"
    print_info "Recommendation: Consider breaking into smaller, focused libraries"
    echo ""

    print_success "Microservices analysis demo complete"
}

# Demo 3: Java Monolith Analysis
demo_java_monolith() {
    print_header "Demo 3: Java Monolith Complexity Analysis"

    print_info "Analyzing a large Java codebase for refactoring candidates"
    echo ""

    print_section "Codebase Overview"
    echo ""
    echo -e "  ${BOLD}Total Statistics:${NC}"
    print_item "Total Files: 185"
    print_item "Total LOC: 28,500"
    print_item "Total Classes: 185"
    print_item "Total Methods: 1,450"
    echo ""

    print_section "Layer Distribution"
    echo ""
    printf "  ${BOLD}%-15s %8s %10s %12s${NC}\n" "Layer" "Files" "LOC" "Avg Complexity"
    echo "  ──────────────────────────────────────────────────"
    printf "  %-15s %8d %10d %12d\n" "Controller" 15 4200 16
    printf "  %-15s %8d %10d %12d\n" "Service" 25 12500 32
    printf "  %-15s %8d %10d %12d\n" "Repository" 12 2100 8
    printf "  %-15s %8d %10d %12d\n" "Model" 35 3200 2
    printf "  %-15s %8d %10d %12d\n" "DTO" 45 2800 1
    printf "  %-15s %8d %10d %12d\n" "Utility" 28 2450 10
    printf "  %-15s %8d %10d %12d\n" "Config" 12 1200 5
    printf "  %-15s %8d %10d %12d\n" "Exception" 18 450 1
    echo ""

    print_section "Hot Spots (High Complexity)"
    echo ""
    printf "  ${BOLD}%-25s %8s %12s %-30s${NC}\n" "File" "LOC" "Complexity" "Issue"
    echo "  ─────────────────────────────────────────────────────────────────────────"
    printf "  ${RED}%-25s %8d %12d${NC} %-30s\n" "ReportService.java" 3200 52 "Complex report generation"
    printf "  ${RED}%-25s %8d %12d${NC} %-30s\n" "PaymentService.java" 1800 42 "Multiple payment gateways"
    printf "  ${YELLOW}%-25s %8d %12d${NC} %-30s\n" "OrderService.java" 2100 35 "Order lifecycle mgmt"
    printf "  ${YELLOW}%-25s %8d %12d${NC} %-30s\n" "CustomQueryRepo.java" 890 22 "Dynamic queries"
    echo ""

    print_section "Refactoring Recommendations"
    echo ""
    echo -e "  ${BOLD}1. ReportService.java${NC} (${RED}CRITICAL${NC})"
    print_item "Issue: 3,200 LOC, complexity 52"
    print_item "Action: Split into ReportGenerator, ReportFormatter, ReportExporter"
    print_item "Pattern: Strategy pattern for report types"
    echo ""
    echo -e "  ${BOLD}2. PaymentService.java${NC} (${RED}CRITICAL${NC})"
    print_item "Issue: Multiple payment integrations in one class"
    print_item "Action: Extract PaymentStrategy interface"
    print_item "Pattern: Strategy + Factory for payment gateways"
    echo ""
    echo -e "  ${BOLD}3. OrderService.java${NC} (${YELLOW}HIGH${NC})"
    print_item "Issue: Complex state transitions"
    print_item "Action: Implement State Machine pattern"
    print_item "Pattern: State pattern for order lifecycle"
    echo ""

    print_section "Quality Metrics"
    echo ""
    printf "  ${BOLD}%-20s %10s %10s %10s${NC}\n" "Metric" "Current" "Target" "Status"
    echo "  ──────────────────────────────────────────────────"
    printf "  %-20s %10s %10s ${YELLOW}%10s${NC}\n" "Test Coverage" "58%" "80%" "BELOW"
    printf "  %-20s %10s %10s ${YELLOW}%10s${NC}\n" "Javadoc Coverage" "42%" "70%" "BELOW"
    printf "  %-20s %10s %10s ${YELLOW}%10s${NC}\n" "Code Duplication" "12%" "5%" "ABOVE"
    echo ""

    print_success "Java monolith analysis demo complete"
}

# Demo 4: Full Audit
demo_full_audit() {
    print_header "Demo 4: Comprehensive System Audit"

    print_info "Full audit combining all analysis capabilities"
    print_info "Based on IOWIZME production system"
    echo ""

    print_section "System Inventory"
    echo ""
    echo -e "  ${BOLD}Applications:${NC}"
    print_item "IOWIZME Integration Platform (Critical)"
    print_item "Configuration Management (High)"
    print_item "Monitoring Stack (Medium)"
    echo ""
    echo -e "  ${BOLD}Infrastructure:${NC}"
    print_item "Kubernetes: 12 nodes, v1.28"
    print_item "PostgreSQL: 850 GB, streaming replication"
    print_item "Kafka: 5 brokers, 42 topics, 15M msg/day"
    print_item "Redis: 6 nodes, 64 GB cluster"
    echo ""

    print_section "Audit Findings Summary"
    echo ""
    printf "  ${BOLD}%-10s %-40s %10s${NC}\n" "Severity" "Finding" "Impact"
    echo "  ──────────────────────────────────────────────────────────────────"
    printf "  ${RED}%-10s${NC} %-40s %10s\n" "CRITICAL" "shared-libs high fan-out (5 services)" "System-wide"
    printf "  ${RED}%-10s${NC} %-40s %10s\n" "CRITICAL" "Single point of failure: message-gateway" "4M msg/day"
    printf "  ${YELLOW}%-10s${NC} %-40s %10s\n" "HIGH" "Test coverage below target (58% vs 80%)" "Reliability"
    printf "  ${YELLOW}%-10s${NC} %-40s %10s\n" "HIGH" "batch-processor complexity (14K LOC)" "Maintenance"
    printf "  ${CYAN}%-10s${NC} %-40s %10s\n" "MEDIUM" "Javadoc coverage insufficient (42%)" "Onboarding"
    printf "  ${CYAN}%-10s${NC} %-40s %10s\n" "MEDIUM" "Code duplication at 12%" "Tech debt"
    echo ""

    print_section "Risk Matrix"
    echo ""
    echo "                    LOW COMPLEXITY ──────────────────▶ HIGH COMPLEXITY"
    echo ""
    echo "      HIGH    │  iow-iog-models     │  iow-ech-sias    │"
    echo "    TRAFFIC   │  (Safe zone)        │  (CRITICAL)      │"
    echo "              │                     │  iow-ioc-sc02    │"
    echo "       │      ├─────────────────────┼──────────────────┤"
    echo "       │      │                     │  iog-support-    │"
    echo "       │      │                     │  commons         │"
    echo "       ▼      │                     │  (CRITICAL)      │"
    echo "              ├─────────────────────┼──────────────────┤"
    echo "      LOW     │                     │  iow-iok-sk04    │"
    echo "    TRAFFIC   │                     │  (Low risk -     │"
    echo "              │                     │   batch only)    │"
    echo ""

    print_section "Recommendations (Prioritized)"
    echo ""
    echo -e "  ${BOLD}${RED}Priority 1: Critical Path Hardening${NC}"
    print_item "Add circuit breakers to message-gateway"
    print_item "Implement retry logic with exponential backoff"
    print_item "Deploy redundant instances (current: 1 → target: 3)"
    echo ""
    echo -e "  ${BOLD}${YELLOW}Priority 2: Shared Library Decomposition${NC}"
    print_item "Split iog-support-commons into domain-specific modules"
    print_item "Extract: commons-validation, commons-logging, commons-security"
    print_item "Reduce blast radius of changes"
    echo ""
    echo -e "  ${BOLD}${CYAN}Priority 3: Quality Improvements${NC}"
    print_item "Increase test coverage: focus on critical path modules"
    print_item "Add integration tests for message flow"
    print_item "Implement mutation testing for core business logic"
    echo ""

    print_section "Team Assignments"
    echo ""
    printf "  ${BOLD}%-25s %-30s %15s${NC}\n" "Team" "Action Item" "Effort"
    echo "  ─────────────────────────────────────────────────────────────────────"
    printf "  %-25s %-30s %15s\n" "Platform Engineering" "Deploy HA for gateway" "2 sprints"
    printf "  %-25s %-30s %15s\n" "Platform Engineering" "Split shared-libs" "4 sprints"
    printf "  %-25s %-30s %15s\n" "Integration Team" "Add circuit breakers" "1 sprint"
    printf "  %-25s %-30s %15s\n" "Business Logic Team" "Increase test coverage" "3 sprints"
    printf "  %-25s %-30s %15s\n" "Batch Processing Team" "Refactor batch-processor" "5 sprints"
    echo ""

    print_success "Full audit demo complete"
}

# Demo 5: Custom Analysis
demo_custom() {
    print_header "Demo 5: Custom Analysis Options"

    print_info "Configure your own analysis parameters"
    echo ""

    print_section "Available Kernels"
    echo ""
    echo -e "  ${BOLD}Stage 1 - Collection:${NC}"
    print_item "volumetry    - Parse traffic and volume data"
    print_item "module_group - Group code by modules/services"
    print_item "inventory    - System inventory parsing"
    echo ""
    echo -e "  ${BOLD}Stage 2 - Analysis:${NC}"
    print_item "risk_matrix  - Calculate risk scores"
    print_item "dependency   - Analyze dependencies"
    print_item "tech_debt    - Assess technical debt"
    echo ""
    echo -e "  ${BOLD}Stage 3 - Reporting:${NC}"
    print_item "executive    - Executive summary"
    print_item "technical    - Detailed technical report"
    print_item "action_plan  - Prioritized action items"
    echo ""

    print_section "Risk Weight Presets"
    echo ""
    printf "  ${BOLD}%-20s %-15s %-15s %-15s %-15s${NC}\n" "Preset" "LOC" "Complexity" "Volumetry" "Dependencies"
    echo "  ─────────────────────────────────────────────────────────────────────────"
    printf "  %-20s %-15s %-15s %-15s %-15s\n" "volumetry_focused" "20%" "25%" "40%" "15%"
    printf "  %-20s %-15s %-15s %-15s %-15s\n" "complexity_focused" "25%" "45%" "15%" "15%"
    printf "  %-20s %-15s %-15s %-15s %-15s\n" "size_focused" "40%" "25%" "20%" "15%"
    printf "  %-20s %-15s %-15s %-15s %-15s\n" "balanced" "25%" "25%" "25%" "25%"
    echo ""

    print_section "Threshold Configurations"
    echo ""
    echo -e "  ${BOLD}Risk Level Thresholds:${NC}"
    print_item "Critical: score >= 8.0"
    print_item "High:     score >= 6.0"
    print_item "Medium:   score >= 4.0"
    print_item "Low:      score >= 2.0"
    echo ""
    echo -e "  ${BOLD}God Class Detection:${NC}"
    print_item "LOC threshold: 1,000 lines"
    print_item "Methods threshold: 50 methods"
    print_item "Complexity threshold: 30"
    echo ""

    print_section "Output Formats"
    echo ""
    print_item "Markdown (.md) - Human-readable reports"
    print_item "JSON (.json) - Machine-processable data"
    print_item "YAML (.yaml) - Configuration-friendly"
    print_item "HTML (.html) - Web-ready visualization"
    echo ""

    print_success "Custom analysis options displayed"
}

# =============================================================================
# Interactive Menu
# =============================================================================

show_menu() {
    echo ""
    echo -e "${BOLD}Select a demo to run:${NC}"
    echo ""
    echo -e "  ${CYAN}1)${NC}  Volumetry Analysis    - Risk weighted by traffic volume"
    echo -e "  ${CYAN}2)${NC}  Microservices        - Service catalog & dependencies"
    echo -e "  ${CYAN}3)${NC}  Java Monolith        - Complexity & refactoring"
    echo -e "  ${CYAN}4)${NC}  Full Audit           - Comprehensive system analysis"
    echo -e "  ${CYAN}5)${NC}  Custom Options       - Configuration parameters"
    echo ""
    echo -e "  ${MAGENTA}a)${NC}  Run All Demos"
    echo -e "  ${MAGENTA}c)${NC}  Check Prerequisites"
    echo -e "  ${MAGENTA}h)${NC}  Show Help"
    echo ""
    echo -e "  ${RED}q)${NC}  Quit"
    echo ""
    echo -n -e "${BOLD}Enter choice [1-5/a/c/h/q]: ${NC}"
}

show_help() {
    print_header "RAGIX Audit Demo Help"

    print_section "About KOAS"
    echo "  KOAS (Kernel-Orchestrated Audit System) is a framework for"
    echo "  systematic code analysis and risk assessment."
    echo ""
    echo "  Key concepts:"
    print_item "Volumetry: Traffic/usage patterns weighted in risk scoring"
    print_item "Risk Matrix: Multi-factor risk calculation"
    print_item "Critical Path: High-traffic code paths requiring attention"
    echo ""

    print_section "Demo Descriptions"
    echo ""
    echo -e "  ${BOLD}1. Volumetry Analysis${NC}"
    echo "     Demonstrates risk calculation where operational traffic volume"
    echo "     is the primary weighting factor. Based on IOWIZME production"
    echo "     data with 4M messages/day."
    echo ""
    echo -e "  ${BOLD}2. Microservices Analysis${NC}"
    echo "     Shows service catalog, dependency graphs, and impact analysis"
    echo "     for distributed architectures."
    echo ""
    echo -e "  ${BOLD}3. Java Monolith Analysis${NC}"
    echo "     Analyzes large codebases for complexity hotspots, god classes,"
    echo "     and refactoring candidates."
    echo ""
    echo -e "  ${BOLD}4. Full Audit${NC}"
    echo "     Comprehensive analysis combining all kernels with prioritized"
    echo "     recommendations and team assignments."
    echo ""
    echo -e "  ${BOLD}5. Custom Options${NC}"
    echo "     Shows available configuration parameters for custom analysis."
    echo ""

    print_section "Running Actual Analysis"
    echo "  To run actual KOAS analysis (not just demos):"
    echo ""
    echo -e "  ${CYAN}cd examples/audit/volumetry_analysis${NC}"
    echo -e "  ${CYAN}python -m ragix_kernels.orchestrator run -w .${NC}"
    echo ""
}

# =============================================================================
# Main Loop
# =============================================================================

main() {
    show_banner

    # Parse command line arguments
    case "${1:-}" in
        --check|-c)
            check_prerequisites
            exit 0
            ;;
        --volumetry|-1)
            check_prerequisites
            demo_volumetry
            exit 0
            ;;
        --microservices|-2)
            check_prerequisites
            demo_microservices
            exit 0
            ;;
        --monolith|-3)
            check_prerequisites
            demo_java_monolith
            exit 0
            ;;
        --full|-4)
            check_prerequisites
            demo_full_audit
            exit 0
            ;;
        --custom|-5)
            check_prerequisites
            demo_custom
            exit 0
            ;;
        --all|-a)
            check_prerequisites
            demo_volumetry
            demo_microservices
            demo_java_monolith
            demo_full_audit
            demo_custom
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
    esac

    # Interactive mode
    while true; do
        show_menu
        read -r choice

        case "$choice" in
            1)
                demo_volumetry
                ;;
            2)
                demo_microservices
                ;;
            3)
                demo_java_monolith
                ;;
            4)
                demo_full_audit
                ;;
            5)
                demo_custom
                ;;
            a|A)
                check_prerequisites
                demo_volumetry
                read -p "Press Enter to continue..."
                demo_microservices
                read -p "Press Enter to continue..."
                demo_java_monolith
                read -p "Press Enter to continue..."
                demo_full_audit
                read -p "Press Enter to continue..."
                demo_custom
                ;;
            c|C)
                check_prerequisites
                ;;
            h|H)
                show_help
                ;;
            q|Q)
                echo ""
                print_info "Thank you for using RAGIX Audit Demo!"
                echo ""
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please select 1-5, a, c, h, or q."
                ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
        show_banner
    done
}

# Run main
main "$@"
