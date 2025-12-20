#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOAS MCP Tools Test Suite
=========================

Tests all 16 KOAS MCP tools to verify proper execution.

Usage:
    python test_koas_tools.py              # Run all tests
    python test_koas_tools.py --security   # Run security tests only
    python test_koas_tools.py --audit      # Run audit tests only

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-19
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MCP"))

# Import test utilities
import importlib.util

def load_mcp_server():
    """Dynamically load the MCP server module."""
    mcp_server_path = PROJECT_ROOT / "MCP" / "ragix_mcp_server.py"
    spec = importlib.util.spec_from_file_location("ragix_mcp_server", mcp_server_path)
    ragix_mcp = importlib.util.module_from_spec(spec)
    sys.modules['ragix_mcp_server'] = ragix_mcp
    spec.loader.exec_module(ragix_mcp)
    return ragix_mcp


def test_tool(tool_func, params, test_name):
    """Execute a tool and report results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Parameters: {json.dumps(params, indent=2)[:200]}...")

    start = time.time()
    try:
        result = tool_func(**params)
        duration = time.time() - start

        if isinstance(result, dict):
            status = result.get("status", "unknown")
            summary = result.get("summary", "No summary")
            error = result.get("error")

            if error:
                print(f"❌ FAILED ({duration:.2f}s): {error}")
                return False
            else:
                print(f"✓ PASSED ({duration:.2f}s)")
                print(f"  Summary: {summary[:200]}...")
                if "workspace" in result:
                    print(f"  Workspace: {result['workspace']}")
                return True
        else:
            print(f"✓ PASSED ({duration:.2f}s)")
            print(f"  Result: {str(result)[:200]}...")
            return True

    except Exception as e:
        duration = time.time() - start
        print(f"❌ EXCEPTION ({duration:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return False


def run_security_tests(mcp):
    """Run security tool tests."""
    print("\n" + "="*70)
    print("SECURITY TOOLS TESTS")
    print("="*70)

    results = []

    # Test 1: Network Discovery
    results.append(test_tool(
        mcp.koas_security_discover,
        {"target": "127.0.0.1", "method": "ping", "timeout": 10},
        "koas_security_discover (localhost)"
    ))

    # Get workspace from previous test for chaining
    discover_result = mcp.koas_security_discover(target="127.0.0.1", method="ping", timeout=10)
    workspace = discover_result.get("workspace", "")

    # Test 2: Port Scan (using discovered targets)
    results.append(test_tool(
        mcp.koas_security_scan_ports,
        {"target": "127.0.0.1", "ports": "web", "detect_services": True, "workspace": workspace},
        "koas_security_scan_ports (web ports)"
    ))

    # Test 3: SSL Check
    results.append(test_tool(
        mcp.koas_security_ssl_check,
        {"target": "127.0.0.1:8080", "check_ciphers": False, "workspace": workspace},
        "koas_security_ssl_check (localhost)"
    ))

    # Test 4: DNS Check
    results.append(test_tool(
        mcp.koas_security_dns_check,
        {"domain": "localhost", "check_security": True, "workspace": workspace},
        "koas_security_dns_check (localhost)"
    ))

    # Test 5: Vulnerability Scan (dry-run mode by default)
    results.append(test_tool(
        mcp.koas_security_vuln_scan,
        {"target": "127.0.0.1", "severity": "high", "templates": "default", "workspace": workspace},
        "koas_security_vuln_scan (high severity)"
    ))

    # Test 6: Compliance Check
    results.append(test_tool(
        mcp.koas_security_compliance,
        {"workspace": workspace, "framework": "anssi", "level": "standard"},
        "koas_security_compliance (ANSSI)"
    ))

    # Test 7: Risk Assessment
    results.append(test_tool(
        mcp.koas_security_risk,
        {"workspace": workspace, "top_hosts": 5},
        "koas_security_risk (top 5)"
    ))

    # Test 8: Security Report
    results.append(test_tool(
        mcp.koas_security_report,
        {"workspace": workspace, "format": "summary", "language": "en"},
        "koas_security_report (summary)"
    ))

    return results


def run_audit_tests(mcp):
    """Run audit tool tests."""
    print("\n" + "="*70)
    print("AUDIT TOOLS TESTS")
    print("="*70)

    results = []

    # Use the project itself as test target
    project_path = str(PROJECT_ROOT / "MCP")

    # First, get workspace from scan (runs before tests to ensure deps exist)
    print("\n[Setup] Running initial scan to create workspace and dependencies...")
    scan_result = mcp.koas_audit_scan(project_path=project_path, language="python")
    workspace = scan_result.get("workspace", "")
    print(f"[Setup] Workspace: {workspace}")

    # Run metrics to create stage1 dependencies
    print("[Setup] Running metrics to create dependencies...")
    mcp.koas_audit_metrics(workspace=workspace, threshold_cc=10, threshold_loc=300)

    # Run dependencies to create module graph
    print("[Setup] Running dependencies analysis...")
    mcp.koas_audit_dependencies(workspace=workspace, detect_cycles=True)

    print("[Setup] Setup complete, running tests...")

    # Test 1: AST Scan (already ran in setup, just verify it works)
    results.append(test_tool(
        mcp.koas_audit_scan,
        {"project_path": project_path, "language": "python", "include_tests": False},
        "koas_audit_scan (MCP directory)"
    ))

    # Test 2: Metrics
    results.append(test_tool(
        mcp.koas_audit_metrics,
        {"workspace": workspace, "threshold_cc": 10, "threshold_loc": 300},
        "koas_audit_metrics (thresholds)"
    ))

    # Test 3: Hotspots
    results.append(test_tool(
        mcp.koas_audit_hotspots,
        {"workspace": workspace, "top_n": 10},
        "koas_audit_hotspots (top 10)"
    ))

    # Test 4: Dependencies
    results.append(test_tool(
        mcp.koas_audit_dependencies,
        {"workspace": workspace, "detect_cycles": True},
        "koas_audit_dependencies (cycles)"
    ))

    # Test 5: Dead Code
    results.append(test_tool(
        mcp.koas_audit_dead_code,
        {"workspace": workspace},
        "koas_audit_dead_code"
    ))

    # Test 6: Risk Assessment
    results.append(test_tool(
        mcp.koas_audit_risk,
        {"workspace": workspace, "include_volumetry": False},
        "koas_audit_risk"
    ))

    # Test 7: Compliance Check
    results.append(test_tool(
        mcp.koas_audit_compliance,
        {"workspace": workspace, "standard": "maintainability"},
        "koas_audit_compliance (maintainability)"
    ))

    # Test 8: Audit Report
    results.append(test_tool(
        mcp.koas_audit_report,
        {"workspace": workspace, "format": "executive", "language": "en"},
        "koas_audit_report (executive)"
    ))

    return results


def main():
    """Main test runner."""
    print("="*70)
    print("KOAS MCP TOOLS TEST SUITE")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse arguments
    run_security = "--security" in sys.argv or len(sys.argv) == 1
    run_audit = "--audit" in sys.argv or len(sys.argv) == 1

    # Load MCP server
    print("\nLoading MCP server...")
    try:
        mcp = load_mcp_server()
        print("✓ MCP server loaded")
    except Exception as e:
        print(f"❌ Failed to load MCP server: {e}")
        return 1

    all_results = []

    # Run tests
    if run_security:
        all_results.extend(run_security_tests(mcp))

    if run_audit:
        all_results.extend(run_audit_tests(mcp))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in all_results if r)
    failed = len(all_results) - passed

    print(f"Total tests: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
