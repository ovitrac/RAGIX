# Security Network Audit Kernels Guide

**RAGIX Version:** 0.61.0+
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17

---

## Overview

The Security Network Audit Kernels provide automated network security assessment
following the KOAS (Kernel-Orchestrated Audit System) pipeline:

| Stage | Kernel | Purpose |
|-------|--------|---------|
| 1 | `net_discover` | Host discovery (nmap, arp-scan) |
| 1 | `port_scan` | Service detection (nmap) |
| 1 | `dns_enum` | DNS enumeration (dnspython, dig) |
| 1 | `config_parse` | Firewall/router config analysis |
| 2 | `ssl_analysis` | SSL/TLS analysis (sslyze) |
| 2 | `vuln_assess` | Vulnerability scanning (nuclei) |
| 2 | `web_scan` | Web app scanning (nikto, ZAP) |
| 2 | `compliance` | Compliance (ANSSI, NIST, CIS) |
| 2 | `risk_network` | Risk aggregation |
| 3 | `section_security` | Report generation |

---

## Prerequisites

### Required Tools

**System packages:**
```bash
# Ubuntu/Debian
sudo apt install nmap dnsutils nikto

# Web scanning (OWASP ZAP)
sudo snap install zaproxy --classic

# Optional but recommended
sudo apt install arp-scan testssl dnsrecon
```

**Python packages:**
```bash
pip install python-nmap dnspython sslyze
```

**Nuclei (vulnerability scanner):**
```bash
# Download from GitHub releases
wget https://github.com/projectdiscovery/nuclei/releases/download/v3.3.7/nuclei_3.3.7_linux_amd64.zip
unzip nuclei_3.3.7_linux_amd64.zip
sudo mv nuclei /usr/local/bin/

# Download templates
git clone https://github.com/projectdiscovery/nuclei-templates ~/nuclei-templates
```

### Verification

```bash
# Check installation
python scripts/check_security_tools.py
```

---

## Quick Start

### 1. Create Workspace

```bash
mkdir -p ./audit/network/data
cd ./audit/network
```

### 2. Define Targets

Create `data/targets.yaml`:
```yaml
targets:
  - "192.168.1.0/24"
  - "10.0.0.1-10"

exclude:
  - "192.168.1.1"  # Gateway

methods:
  - ping
  - arp
```

Or for domain scanning, create `data/domains.yaml`:
```yaml
domains:
  - "example.com"
  - "test.example.com"
```

### 3. Create Manifest

Create `manifest.yaml`:
```yaml
# Security Network Audit Manifest
name: "Network Security Assessment"
description: "Automated security scan"

stages:
  1:
    net_discover:
      enabled: true
      options:
        methods: ["ping"]
        timeout: 300

    port_scan:
      enabled: true
      options:
        ports: "top100"
        scan_type: "connect"
        service_detection: true

    dns_enum:
      enabled: true
      options:
        record_types: ["A", "AAAA", "MX", "NS", "TXT", "SOA"]
        check_zone_transfer: true
        check_dnssec: true

  2:
    ssl_analysis:
      enabled: true
      options:
        check_certificates: true
        check_ciphers: true
        check_vulnerabilities: true

    vuln_assess:
      enabled: true
      options:
        templates: ["cves", "misconfiguration", "exposed-panels"]
        severity: "low"
        rate_limit: 50

    risk_network:
      enabled: true
      options:
        weights:
          vulnerabilities: 0.40
          ssl_issues: 0.20
          exposure: 0.25
          dns_issues: 0.15

  3:
    section_security:
      enabled: true
      options:
        output_format: "markdown"
        include_executive_summary: true
        include_remediation_plan: true
```

### 4. Run the Audit

```bash
# Full pipeline (all stages)
python -m ragix_kernels.orchestrator run -w ./audit/network

# Specific stage
python -m ragix_kernels.orchestrator run -w ./audit/network -s 1

# Specific kernel
python -m ragix_kernels.orchestrator run -w ./audit/network -s 1 -k port_scan
```

---

## Kernel Reference

### Stage 1: Discovery

#### net_discover

Discovers live hosts on the network.

**Input:**
```yaml
net_discover:
  options:
    targets: ["192.168.1.0/24"]
    exclude: ["192.168.1.1"]
    methods: ["ping", "arp", "list"]
    timeout: 300
```

**Output:**
```json
{
  "hosts": [
    {
      "ip": "192.168.1.10",
      "hostname": "server1.local",
      "status": "up",
      "mac": "AA:BB:CC:DD:EE:FF",
      "vendor": "Dell Inc.",
      "discovery_method": "nmap_ping"
    }
  ],
  "statistics": {
    "hosts_up": 15,
    "scan_time_sec": 12.5
  }
}
```

**Methods:**
- `ping`: ICMP/TCP ping sweep (nmap -sn)
- `arp`: ARP scan for local network (requires root)
- `list`: DNS resolution only, no probing

---

#### port_scan

Discovers services and versions on target hosts.

**Input:**
```yaml
port_scan:
  options:
    targets: ["192.168.1.10"]  # Optional if net_discover ran
    ports: "top100"            # Or "22,80,443" or "1-1024"
    scan_type: "connect"       # "syn" requires root, "udp" for UDP
    service_detection: true
    scripts: ["banner", "http-title"]
    timeout: 300
```

**Port presets:**
- `top20`: Common 20 ports
- `top100`: Extended common ports
- `web`: HTTP/HTTPS ports (80, 443, 8080, etc.)
- `database`: Database ports (MySQL, PostgreSQL, etc.)
- `common`: Mix of common services

**Output:**
```json
{
  "services": [
    {
      "host": "192.168.1.10",
      "port": 443,
      "protocol": "tcp",
      "state": "open",
      "service": "https",
      "version": "nginx 1.18.0",
      "product": "nginx",
      "cpe": "cpe:/a:nginx:nginx:1.18.0"
    }
  ],
  "by_host": {
    "192.168.1.10": [...]
  },
  "statistics": {
    "hosts_scanned": 1,
    "total_open_ports": 5
  }
}
```

---

#### dns_enum

DNS enumeration and security analysis.

**Input:**
```yaml
dns_enum:
  options:
    domains: ["example.com"]
    record_types: ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME"]
    check_zone_transfer: true
    check_dnssec: true
    subdomain_wordlist: "/path/to/wordlist.txt"  # Optional
```

**Output:**
```json
{
  "records": {
    "example.com": {
      "A": [{"type": "A", "value": "93.184.216.34", "ttl": 300}],
      "MX": [{"type": "MX", "value": "mail.example.com", "preference": 10}],
      "TXT": [{"type": "TXT", "value": "v=spf1 include:_spf.google.com ~all"}]
    }
  },
  "misconfigurations": [
    {
      "domain": "example.com",
      "type": "zone_transfer_allowed",
      "detail": "Zone transfer allowed on ns1.example.com",
      "severity": "high"
    }
  ],
  "subdomains": [
    {"subdomain": "www.example.com", "ips": ["93.184.216.34"]}
  ]
}
```

**Security checks:**
- SPF record present
- DMARC record present
- DNSSEC enabled
- Zone transfer restricted

---

#### config_parse

Parses and analyzes firewall/router configurations.

**Input:**
```yaml
config_parse:
  options:
    config_dir: "data/configs"     # Directory with config files
    config_type: "auto"            # Or: iptables, cisco_ios, cisco_asa, pfsense, fortigate
    analyze_rules: true
    check_best_practices: true
```

**Supported formats:**
- `iptables` / `nftables` (Linux)
- `cisco_ios` / `cisco_asa` (Cisco)
- `pfsense` / `opnsense` (XML)
- `fortigate` (Fortinet)

**Output:**
```json
{
  "devices": [
    {
      "name": "firewall1",
      "type": "linux_firewall",
      "config_type": "iptables"
    }
  ],
  "rules": [
    {
      "type": "rule",
      "chain": "INPUT",
      "action": "permit",
      "source": "192.168.1.0/24",
      "destination": "any",
      "port": "22"
    }
  ],
  "findings": [
    {
      "type": "any_to_any",
      "severity": "high",
      "description": "Rule permits any-to-any traffic"
    }
  ]
}
```

**Best practice checks:**
- Default deny policy
- No any-to-any rules
- Telnet/rsh blocked
- Dangerous ports restricted

---

### Stage 2: Analysis

#### ssl_analysis

SSL/TLS security assessment.

**Input:**
```yaml
ssl_analysis:
  options:
    targets: ["example.com:443"]  # Optional if port_scan ran
    check_certificates: true
    check_ciphers: true
    check_vulnerabilities: true
    timeout: 30
```

**Output:**
```json
{
  "certificates": [
    {
      "host": "example.com",
      "port": 443,
      "subject": "CN=example.com",
      "issuer": "CN=DigiCert TLS RSA SHA256 2020 CA1",
      "not_after": "2024-12-15T23:59:59",
      "days_until_expiry": 363,
      "valid": true,
      "san": ["example.com", "www.example.com"]
    }
  ],
  "cipher_suites": {
    "example.com:443": [
      {"name": "TLS_AES_256_GCM_SHA384", "version": "tls13", "weak": false},
      {"name": "TLS_RSA_WITH_3DES_EDE_CBC_SHA", "version": "tls12", "weak": true}
    ]
  },
  "vulnerabilities": [
    {
      "host": "example.com",
      "port": 443,
      "type": "weak_cipher",
      "detail": "Weak cipher TLS_RSA_WITH_3DES_EDE_CBC_SHA (tls12)",
      "severity": "medium"
    },
    {
      "host": "example.com",
      "port": 443,
      "type": "deprecated_protocol",
      "detail": "Deprecated protocol TLS10 supported",
      "severity": "medium"
    }
  ]
}
```

**Vulnerability checks:**
- Heartbleed (CVE-2014-0160)
- ROBOT attack
- Deprecated protocols (SSL2, SSL3, TLS 1.0, TLS 1.1)
- Weak ciphers (RC4, 3DES, NULL, EXPORT)
- Certificate validity and expiration

---

#### vuln_assess

Automated vulnerability scanning with nuclei.

**Input:**
```yaml
vuln_assess:
  options:
    targets: ["https://example.com"]  # Optional if port_scan ran
    templates:
      - cves
      - misconfiguration
      - exposed-panels
      - default-logins
      - takeovers
    severity: "low"           # Minimum severity to report
    rate_limit: 100           # Requests per second
    timeout: 600              # Total scan timeout
    template_path: "~/nuclei-templates"
    exclude_templates: []
```

**Output:**
```json
{
  "vulnerabilities": [
    {
      "template_id": "CVE-2021-44228",
      "name": "Apache Log4j RCE",
      "severity": "critical",
      "host": "https://example.com",
      "matched_at": "https://example.com/api/login",
      "cve": "CVE-2021-44228",
      "cvss": 10.0,
      "description": "Remote code execution in Apache Log4j..."
    }
  ],
  "by_severity": {
    "critical": [...],
    "high": [...],
    "medium": [...],
    "low": [...],
    "info": [...]
  },
  "statistics": {
    "total_findings": 15,
    "critical": 1,
    "high": 3,
    "unique_cves": 2
  }
}
```

---

#### web_scan

Web application vulnerability scanning with nikto and OWASP ZAP.

**Input:**
```yaml
web_scan:
  options:
    targets: ["https://example.com"]  # Optional if port_scan ran
    scanners: ["nikto", "zap"]
    scan_mode: "standard"             # quick, standard, full
    timeout: 600
```

**Output:**
```json
{
  "findings": [
    {
      "scanner": "nikto",
      "target": "https://example.com",
      "id": "OSVDB-3092",
      "severity": "medium",
      "description": "/admin/: This might be interesting..."
    },
    {
      "scanner": "zap",
      "target": "https://example.com",
      "id": "10021",
      "severity": "low",
      "name": "X-Content-Type-Options Header Missing"
    }
  ],
  "statistics": {
    "targets_scanned": 1,
    "total_findings": 15,
    "scanners_used": ["nikto", "zap"]
  }
}
```

**Scanners:**
- `nikto`: CGI scanner, misconfigurations, outdated software
- `zap`: OWASP ZAP baseline scan (active scanning requires daemon mode)

---

#### compliance

Compliance assessment against security frameworks.

**Supported Frameworks:**
- **ANSSI** (Primary): French National Cybersecurity Agency
  - Guide d'hygiène informatique (42 measures)
  - TLS recommendations
- **NIST CSF**: Cybersecurity Framework
- **CIS Controls v8**: Critical Security Controls

**Input:**
```yaml
compliance:
  options:
    frameworks: ["anssi", "nist", "cis"]
    anssi_level: "standard"           # essential, standard, reinforced
    include_recommendations: true
```

**Output:**
```json
{
  "compliance_scores": {
    "anssi": {
      "framework": "ANSSI",
      "level": "standard",
      "total_controls": 25,
      "compliant": 18,
      "non_compliant": 7,
      "score": 72.0
    },
    "nist": {
      "framework": "NIST CSF",
      "score": 68.5
    },
    "cis": {
      "framework": "CIS Controls v8",
      "score": 75.0
    }
  },
  "findings": [
    {
      "framework": "anssi",
      "rule_id": "ANSSI-TLS-01",
      "title": "Utiliser TLS 1.2 au minimum",
      "status": "non_compliant",
      "failed_checks": ["tls_1_2_minimum"]
    }
  ],
  "recommendations": [
    {
      "priority": "high",
      "action": "Configure TLS 1.2 minimum on all services"
    }
  ]
}
```

**ANSSI levels:**
- `essential`: Core security measures (must-have)
- `standard`: Recommended practices (should-have)
- `reinforced`: Advanced security (nice-to-have)

---

#### risk_network

Aggregates all findings into a unified risk matrix.

**Input:**
```yaml
risk_network:
  options:
    weights:
      vulnerabilities: 0.40
      ssl_issues: 0.20
      exposure: 0.25
      dns_issues: 0.15
    thresholds:
      critical: 8.0
      high: 6.0
      medium: 4.0
      low: 2.0
```

**Output:**
```json
{
  "risk_matrix": [
    {
      "host": "192.168.1.10",
      "total_score": 8.5,
      "risk_level": "CRITICAL",
      "component_scores": {
        "vulnerabilities": 9.5,
        "ssl_issues": 7.0,
        "exposure": 8.0,
        "dns_issues": 2.0
      },
      "services_count": 5,
      "vulnerabilities_count": 3,
      "top_issues": [
        {"type": "vulnerability", "severity": "critical", "name": "Log4j RCE"}
      ]
    }
  ],
  "prioritized_hosts": [
    {"host": "192.168.1.10", "risk_score": 8.5, "risk_level": "CRITICAL"}
  ],
  "remediation_priorities": [
    {
      "priority": 1,
      "host": "192.168.1.10",
      "issue": "Apache Log4j RCE",
      "severity": "critical",
      "recommendation": "Apply security patch for CVE-2021-44228..."
    }
  ]
}
```

**Risk factors:**
- **Vulnerabilities (40%)**: CVEs and security findings from nuclei
- **SSL Issues (20%)**: TLS misconfigurations, weak ciphers, expired certs
- **Exposure (25%)**: Dangerous services (telnet, FTP), database exposure
- **DNS Issues (15%)**: Zone transfers, missing SPF/DMARC

---

### Stage 3: Reporting

#### section_security

Generates the security section of the audit report.

**Input:**
```yaml
section_security:
  options:
    output_format: "markdown"  # or "json"
    include_executive_summary: true
    include_technical_details: true
    include_remediation_plan: true
    language: "en"            # or "fr"
    max_findings: 50
```

**Output:**
- `stage3/security_report.md` - Full security report
- JSON metadata with generation timestamp

**Report sections:**
1. Executive Summary (risk distribution, key findings)
2. Detailed Findings (by severity)
3. Host Risk Details
4. Remediation Roadmap (by urgency)
5. Appendix (scan statistics)

---

## Usage Examples

### Example 1: Internal Network Scan

```bash
# Setup
mkdir -p audit/internal/data
cat > audit/internal/data/targets.yaml << 'EOF'
targets:
  - "10.0.0.0/24"
exclude:
  - "10.0.0.1"
methods:
  - ping
EOF

# Run discovery
python -m ragix_kernels.orchestrator run -w audit/internal -s 1

# Run analysis
python -m ragix_kernels.orchestrator run -w audit/internal -s 2

# Generate report
python -m ragix_kernels.orchestrator run -w audit/internal -s 3
```

### Example 2: Web Application Scan

```bash
# Setup
mkdir -p audit/webapp/data
cat > audit/webapp/manifest.yaml << 'EOF'
name: "Web Application Security Scan"

stages:
  1:
    port_scan:
      enabled: true
      options:
        targets: ["app.example.com"]
        ports: "web"
        service_detection: true

    dns_enum:
      enabled: true
      options:
        domains: ["example.com"]
        check_zone_transfer: true

  2:
    ssl_analysis:
      enabled: true

    vuln_assess:
      enabled: true
      options:
        templates: ["cves", "exposed-panels", "takeovers"]
        severity: "medium"

    risk_network:
      enabled: true

  3:
    section_security:
      enabled: true
EOF

# Run full audit
python -m ragix_kernels.orchestrator run -w audit/webapp
```

### Example 3: SSL/TLS Only Assessment

```bash
# Direct SSL scan of specific hosts
cat > audit/ssl/manifest.yaml << 'EOF'
stages:
  2:
    ssl_analysis:
      enabled: true
      options:
        targets:
          - "mail.example.com:993"
          - "mail.example.com:465"
          - "web.example.com:443"
        check_certificates: true
        check_ciphers: true
        check_vulnerabilities: true
EOF

python -m ragix_kernels.orchestrator run -w audit/ssl -s 2 -k ssl_analysis
```

### Example 4: Vulnerability-Focused Scan

```bash
# Focus on vulnerability detection
cat > audit/vuln/manifest.yaml << 'EOF'
stages:
  1:
    port_scan:
      enabled: true
      options:
        targets: ["target.example.com"]
        ports: "top100"

  2:
    vuln_assess:
      enabled: true
      options:
        templates:
          - cves
          - vulnerabilities
          - default-logins
          - exposed-panels
        severity: "low"
        rate_limit: 100
        template_path: "~/nuclei-templates"

    risk_network:
      enabled: true
EOF

python -m ragix_kernels.orchestrator run -w audit/vuln
```

---

## Output Structure

```
audit/network/
├── data/
│   ├── targets.yaml
│   └── domains.yaml
├── manifest.yaml
├── stage1/
│   ├── net_discover.json
│   ├── port_scan.json
│   └── dns_enum.json
├── stage2/
│   ├── ssl_analysis.json
│   ├── vuln_assess.json
│   ├── risk_network.json
│   └── nuclei_output.json
└── stage3/
    └── security_report.md
```

---

## Security Considerations

### Authorization

**IMPORTANT:** Only run security scans on systems you are authorized to test.

- Always obtain written permission before scanning
- Respect scope limitations
- Do not scan production systems without proper change management

### Rate Limiting

Default rate limits are conservative:
- nuclei: 100 requests/second
- nmap: T4 timing (aggressive but not disruptive)

Adjust for sensitive environments:
```yaml
vuln_assess:
  options:
    rate_limit: 10  # Slower for production systems
```

### Logging

All scan activities are logged:
- Workspace: `stage*/` directories contain full output
- Orchestrator: Standard logging to stderr

---

## Troubleshooting

### nmap permission errors

For SYN scans (`scan_type: "syn"`), root privileges are required:
```bash
sudo python -m ragix_kernels.orchestrator run -w audit/network -s 1
```

Or use TCP connect scans (no root needed):
```yaml
port_scan:
  options:
    scan_type: "connect"
```

### nuclei template errors

Update templates:
```bash
cd ~/nuclei-templates
git pull
```

### sslyze connection errors

For targets behind firewalls, increase timeout:
```yaml
ssl_analysis:
  options:
    timeout: 60
```

### Missing python-nmap

If nmap Python bindings fail, the kernels fall back to subprocess calls:
```bash
pip install python-nmap
```

---

## References

- [nmap Documentation](https://nmap.org/docs.html)
- [nuclei Documentation](https://nuclei.projectdiscovery.io/)
- [sslyze Documentation](https://nabla-c0d3.github.io/sslyze/documentation/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)

---

*RAGIX Security Kernels - Automated Network Security Assessment*
