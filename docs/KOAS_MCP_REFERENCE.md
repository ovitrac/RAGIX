# KOAS MCP Reference

**Version:** 0.62.0
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-12-19

---

## Overview

KOAS (Kernel-Orchestrated Audit System) exposes 31 kernels (21 audit + 10 security) via MCP with simplified interfaces designed for LLM consumption. This reference documents the 16 new simplified tools added in v0.62.0.

## Design Principles

1. **Single values** instead of arrays - use `"discovered"` keyword for chaining
2. **Preset strings** instead of complex configurations
3. **Mandatory summaries** (<300 chars) in every response
4. **Explicit action items** in risk/compliance outputs
5. **Details in files** - full JSON referenced via `details_file`
6. **Auto-workspace** - `/tmp/koas_{uuid}` created automatically if not specified

---

## Security Tools (8)

### `koas_security_discover`

Discover hosts on a network segment.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target` | string | *required* | Network/host (e.g., "192.168.1.0/24") |
| `method` | string | "ping" | Discovery method: "ping", "arp", "list" |
| `timeout` | integer | 120 | Scan timeout in seconds |
| `workspace` | string | "" | Path to workspace (auto-created if empty) |

**Returns:**

```json
{
  "summary": "Found 12 hosts on 192.168.1.0/24.",
  "hosts": [{"ip": "192.168.1.1", "mac": "aa:bb:cc:dd:ee:ff", "hostname": "router"}],
  "hosts_total": 12,
  "workspace": "/tmp/koas_security_20251219_123456_abc12345",
  "details_file": "/tmp/koas_.../stage1/net_discover.json"
}
```

---

### `koas_security_scan_ports`

Scan ports on target hosts.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target` | string | "discovered" | Target or "discovered" for prior results |
| `ports` | string | "common" | Preset: "common", "web", "database", "admin", "top100", "full" |
| `detect_services` | boolean | true | Enable service/version detection |
| `workspace` | string | *required if target="discovered"* | Workspace path |

**Port Presets:**

| Preset | Ports |
|--------|-------|
| `common` | 21,22,23,25,53,80,110,135,139,143,443,445,993,995,1723,3306,3389,5432,5900,8080,8443 |
| `web` | 80,443,8000,8080,8443,8888,9000,9090 |
| `database` | 1433,1521,3306,5432,5984,6379,27017,28017 |
| `admin` | 22,23,3389,5900,5901 |
| `top100` | Top 100 most common ports |
| `full` | 1-65535 |

**Returns:**

```json
{
  "summary": "Port scan on 5 host(s). Found 23 ports, 15 services.",
  "ports": [{"port": 22, "protocol": "tcp", "state": "open", "service": "ssh"}],
  "ports_total": 23,
  "services_found": ["ssh", "http", "https", "mysql"],
  "workspace": "...",
  "details_file": "..."
}
```

---

### `koas_security_ssl_check`

Analyze SSL/TLS configuration and certificates.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target` | string | "discovered" | Target or "discovered" |
| `check_ciphers` | boolean | true | Check cipher suite security |
| `check_vulnerabilities` | boolean | true | Check for TLS vulnerabilities |
| `workspace` | string | "" | Workspace path |

**Returns:**

```json
{
  "summary": "SSL/TLS analysis completed. 2 weak ciphers found.",
  "certificates": [{"subject": "example.com", "issuer": "Let's Encrypt", "expires": "2025-06-15", "valid": true}],
  "weak_ciphers": ["TLS_RSA_WITH_3DES_EDE_CBC_SHA"],
  "vulnerabilities": ["POODLE"],
  "action_items": [{"priority": "high", "action": "Disable weak ciphers"}],
  "details_file": "..."
}
```

---

### `koas_security_vuln_scan`

Scan for known vulnerabilities.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `target` | string | "discovered" | Target or "discovered" |
| `severity` | string | "medium" | Minimum severity: "info", "low", "medium", "high", "critical" |
| `templates` | string | "default" | Template set: "default", "cves", "misconfigs", "exposures", "all" |
| `workspace` | string | "" | Workspace path |

**Returns:**

```json
{
  "summary": "Vulnerability scan on 3 target(s). Found 5 vulnerabilities, 1 critical.",
  "vulnerabilities": [{"id": "CVE-2024-1234", "severity": "critical", "title": "OpenSSL RCE", "host": "192.168.1.10"}],
  "vulnerabilities_total": 5,
  "critical_count": 1,
  "high_count": 2,
  "action_items": [{"priority": "critical", "action": "Patch CVE-2024-1234"}],
  "details_file": "..."
}
```

---

### `koas_security_dns_check`

Analyze DNS configuration and security records.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `domain` | string | *required* | Domain to analyze |
| `check_security` | boolean | true | Check SPF, DKIM, DMARC, DNSSEC |
| `workspace` | string | "" | Workspace path |

**Returns:**

```json
{
  "summary": "DNS analysis for example.com. SPF: OK DMARC: missing DNSSEC: enabled",
  "records": {"A": ["93.184.216.34"], "MX": ["mail.example.com"], "NS": ["ns1.example.com"]},
  "security": {"spf": true, "dkim": false, "dmarc": false, "dnssec": true},
  "subdomains": ["www", "mail", "api"],
  "action_items": [{"priority": "medium", "action": "Configure DMARC"}],
  "details_file": "..."
}
```

---

### `koas_security_compliance`

Check compliance against security frameworks.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with prior scan results |
| `framework` | string | "anssi" | Framework: "anssi", "nist", "cis" |
| `level` | string | "standard" | ANSSI: "essential", "standard", "reinforced"; CIS: "IG1", "IG2", "IG3" |

**Frameworks:**

| Framework | Description |
|-----------|-------------|
| `anssi` | ANSSI Guide d'hygiène informatique (42 rules) |
| `nist` | NIST Cybersecurity Framework (5 functions) |
| `cis` | CIS Controls v8 (18 controls) |

**Returns:**

```json
{
  "summary": "ANSSI compliance: 75%. 30/40 controls passed.",
  "compliance_score": 75.0,
  "framework": "ANSSI",
  "passed": 30,
  "failed": 10,
  "findings": [{"rule_id": "ANSSI-07", "status": "fail", "recommendation": "Restrict exposed services"}],
  "action_items": [...],
  "details_file": "..."
}
```

---

### `koas_security_risk`

Calculate network security risk scores.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with scan results |
| `top_hosts` | integer | 5 | Number of highest-risk hosts to return |

**Returns:**

```json
{
  "summary": "Network risk: HIGH (7.2/10). Top risk: 192.168.1.50.",
  "risk_score": 7.2,
  "risk_level": "HIGH",
  "top_risks": [{"host": "192.168.1.50", "score": 8.5, "factors": ["exposed_ssh", "outdated_openssl"]}],
  "risk_breakdown": {"vulnerabilities": 3.5, "exposure": 2.5, "compliance": 1.2},
  "action_items": [{"priority": "high", "action": "Patch 192.168.1.50"}],
  "details_file": "..."
}
```

**Risk Levels:**

| Score | Level |
|-------|-------|
| >= 8.0 | CRITICAL |
| >= 6.0 | HIGH |
| >= 4.0 | MEDIUM |
| < 4.0 | LOW |

---

### `koas_security_report`

Generate security assessment report.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with scan results |
| `format` | string | "summary" | Report format: "summary", "detailed", "executive" |
| `language` | string | "en" | Language: "en" (English), "fr" (French) |

**Returns:**

```json
{
  "summary": "Security assessment complete. 5 findings, 8 recommendations. Report: security_report.md",
  "report_file": "/tmp/koas_.../stage3/security_report.md",
  "key_findings": ["Exposed SSH on 3 hosts", "Weak TLS configuration", "Missing DMARC"],
  "recommendations": ["Restrict SSH access", "Upgrade TLS to 1.3", "Configure email security"],
  "scores": {"compliance": 75, "risk": 7.2}
}
```

---

## Audit Tools (8)

### `koas_audit_scan`

Scan a codebase and extract AST symbols.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `project_path` | string | *required* | Path to project directory |
| `language` | string | "auto" | Language: "auto", "python", "java", "typescript" |
| `include_tests` | boolean | false | Include test files |
| `workspace` | string | "" | Workspace path (auto-created if empty) |

**Returns:**

```json
{
  "summary": "Scanned 45 files. Found 120 classes, 580 methods, 95 functions.",
  "symbols_total": 795,
  "classes": 120,
  "methods": 580,
  "functions": 95,
  "files_scanned": 45,
  "top_files": [{"file": "src/main.py", "symbols": 85}],
  "workspace": "...",
  "details_file": "..."
}
```

---

### `koas_audit_metrics`

Compute code metrics (complexity, LOC, maintainability).

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with scan results |
| `threshold_cc` | integer | 10 | Cyclomatic complexity threshold |
| `threshold_loc` | integer | 300 | Lines of code threshold |

**Returns:**

```json
{
  "summary": "Total: 15,230 LOC, avg CC: 4.2. 8 high-complexity items flagged.",
  "total_loc": 15230,
  "avg_complexity": 4.2,
  "maintainability_index": 68.5,
  "high_complexity": [{"name": "process_data", "cc": 25, "file": "src/processor.py"}],
  "large_files": [{"file": "src/utils.py", "loc": 1200}],
  "details_file": "..."
}
```

---

### `koas_audit_hotspots`

Identify complexity and risk hotspots.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with analysis results |
| `top_n` | integer | 20 | Number of hotspots to return |

**Returns:**

```json
{
  "summary": "10 hotspots identified. Top: PaymentService.process (score: 8.5).",
  "hotspots": [{"name": "PaymentService.process", "file": "src/payment.py", "score": 8.5, "factors": ["high_cc", "many_dependencies"]}],
  "risk_distribution": {"high": 3, "medium": 5, "low": 2},
  "action_items": [{"priority": "high", "action": "Refactor PaymentService.process"}],
  "details_file": "..."
}
```

---

### `koas_audit_dependencies`

Analyze code dependencies and coupling.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with scan results |
| `detect_cycles` | boolean | true | Detect circular dependencies |

**Returns:**

```json
{
  "summary": "45 modules, 120 dependencies. 2 circular dependencies detected.",
  "modules": 45,
  "dependencies": 120,
  "cycles": [["module_a", "module_b", "module_c", "module_a"]],
  "cycles_count": 2,
  "high_coupling": [{"module": "utils", "fan_in": 35, "fan_out": 12}],
  "details_file": "..."
}
```

---

### `koas_audit_dead_code`

Detect potentially dead or unused code.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with analysis results |

**Returns:**

```json
{
  "summary": "Found 15 potentially dead code items: 8 functions, 5 methods, 2 classes.",
  "dead_code": [{"name": "old_handler", "type": "function", "file": "src/handlers.py", "reason": "no_references"}],
  "dead_code_total": 15,
  "by_type": {"function": 8, "method": 5, "class": 2},
  "action_items": [{"priority": "low", "action": "Review old_handler for removal"}],
  "details_file": "..."
}
```

---

### `koas_audit_risk`

Calculate code risk scores.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with analysis results |
| `include_volumetry` | boolean | false | Include volumetry data in calculation |

**Returns:**

```json
{
  "summary": "Code risk: MEDIUM (5.2/10). Top risk: PaymentModule.",
  "risk_score": 5.2,
  "risk_level": "MEDIUM",
  "top_risks": [{"module": "PaymentModule", "score": 7.8, "factors": ["high_complexity", "tight_coupling"]}],
  "risk_breakdown": {"complexity": 2.5, "coupling": 1.5, "debt": 1.2},
  "action_items": [{"priority": "medium", "action": "Reduce complexity in PaymentModule"}],
  "details_file": "..."
}
```

---

### `koas_audit_compliance`

Check code quality compliance.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with analysis results |
| `standard` | string | "maintainability" | Standard: "maintainability", "testability", "documentation" |

**Standards:**

| Standard | Checks |
|----------|--------|
| `maintainability` | Maintainability index, complexity thresholds |
| `testability` | Test coverage, test-to-code ratio |
| `documentation` | Docstring/Javadoc coverage |

**Returns:**

```json
{
  "summary": "Maintainability compliance: 72%. 15 violations found.",
  "compliance_score": 72.0,
  "standard": "maintainability",
  "violations": [{"rule": "CC_HIGH", "severity": "medium", "location": "src/processor.py"}],
  "action_items": [{"priority": "medium", "action": "Fix CC_HIGH in src/processor.py"}],
  "details_file": "..."
}
```

---

### `koas_audit_report`

Generate code audit report.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `workspace` | string | *required* | Workspace with analysis results |
| `format` | string | "executive" | Format: "executive", "detailed", "technical" |
| `language` | string | "en" | Language: "en", "fr" |

**Returns:**

```json
{
  "summary": "Audit complete. 8 findings, 12 recommendations. Report: audit_report.md",
  "report_file": "/tmp/koas_.../stage3/audit_report.md",
  "key_findings": ["High complexity in payment module", "Circular dependency detected", "Low test coverage"],
  "recommendations": ["Refactor payment processing", "Break dependency cycle", "Increase test coverage to 80%"],
  "scores": {"risk": 5.2, "compliance": 72}
}
```

---

## Workflow Patterns

### Security Assessment Flow

```
1. koas_security_discover(target="192.168.1.0/24")
   → Returns workspace path

2. koas_security_scan_ports(target="discovered", workspace=ws)
   → Uses hosts from step 1

3. koas_security_ssl_check(target="discovered", workspace=ws)
   → Analyzes HTTPS services

4. koas_security_vuln_scan(target="discovered", workspace=ws)
   → Scans for CVEs

5. koas_security_compliance(workspace=ws, framework="anssi")
   → Checks against ANSSI rules

6. koas_security_risk(workspace=ws)
   → Calculates overall risk

7. koas_security_report(workspace=ws, format="executive")
   → Generates final report
```

### Code Audit Flow

```
1. koas_audit_scan(project_path="/path/to/project")
   → Returns workspace path

2. koas_audit_metrics(workspace=ws)
   → Computes complexity, LOC, MI

3. koas_audit_dependencies(workspace=ws)
   → Analyzes coupling, cycles

4. koas_audit_hotspots(workspace=ws)
   → Identifies risk areas

5. koas_audit_dead_code(workspace=ws)
   → Finds unused code

6. koas_audit_risk(workspace=ws)
   → Calculates overall risk

7. koas_audit_compliance(workspace=ws, standard="maintainability")
   → Checks quality standards

8. koas_audit_report(workspace=ws, format="executive")
   → Generates final report
```

---

## Error Handling

All tools return errors in a consistent format:

```json
{
  "status": "error",
  "error": "Error message",
  "kernel": "kernel_name",
  "summary": "Kernel kernel_name failed: Error message",
  "action_items": [{"priority": "high", "action": "Investigate kernel_name failure"}]
}
```

---

## LLM Usage Notes

### For System Prompts

```
You have access to KOAS tools for security scanning and code auditing.

When the user asks to scan a network:
1. Use koas_security_discover to find hosts
2. Use koas_security_scan_ports to check open ports
3. Use koas_security_risk to assess overall risk

When the user asks to audit code:
1. Use koas_audit_scan to analyze the codebase
2. Use koas_audit_metrics for complexity metrics
3. Use koas_audit_risk to assess code quality

Always provide the workspace path when chaining tools.
Use "discovered" as target to reference previous scan results.
```

### Tool Selection Hints

| User Intent | Recommended Tool |
|------------|------------------|
| "Check my network" | `koas_security_discover` |
| "What ports are open?" | `koas_security_scan_ports` |
| "Is my SSL secure?" | `koas_security_ssl_check` |
| "Find vulnerabilities" | `koas_security_vuln_scan` |
| "ANSSI compliance" | `koas_security_compliance` |
| "Scan my code" | `koas_audit_scan` |
| "Code complexity" | `koas_audit_metrics` |
| "Find hotspots" | `koas_audit_hotspots` |
| "Dead code" | `koas_audit_dead_code` |

---

## Version History

- **v0.62.0** (2025-12-19): Added 16 simplified KOAS tools for LLM consumption
- **v0.61.0** (2025-12-17): Initial KOAS security kernels (10 kernels)
