# Design: Security Network Audit Kernels

**Date:** 2025-12-16
**Status:** DRAFT — For Discussion
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr

---

## 1. Context

A new kernel collection is requested for **security network audits** — distinct from code audits. These kernels must:

- Follow KOAS philosophy (pure computation, no LLM inside)
- Be **sovereign** (local tools, no cloud dependencies)
- Be **auditable** (full trace of scans and findings)
- Wrap established security tools (nmap, testssl, etc.)

---

## 2. Domain Analysis

### What Security Network Audit Covers

| Domain | Description | Tools |
|--------|-------------|-------|
| **Discovery** | Asset enumeration, network mapping | nmap, masscan, arp-scan |
| **Port Analysis** | Service detection, version fingerprinting | nmap, netcat |
| **Vulnerability** | CVE scanning, weakness detection | OpenVAS, Nessus (API), nuclei |
| **SSL/TLS** | Certificate analysis, cipher suites | testssl.sh, sslyze |
| **DNS** | Zone analysis, misconfigurations | dig, dnsenum, dnsrecon |
| **Web** | HTTP headers, CORS, security headers | curl, nikto, httpx |
| **Configuration** | Firewall rules, router config | Custom parsers |
| **Compliance** | CIS, NIST, PCI-DSS checks | Custom or OpenSCAP |

### What's NOT Covered (Active Exploitation)

These kernels are for **audit and assessment**, NOT:
- Penetration testing (active exploitation)
- Credential attacks
- DoS/DDoS testing
- Malware deployment

---

## 3. Proposed Kernel Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SECURITY NETWORK AUDIT PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: Discovery           STAGE 2: Analysis        STAGE 3: Report  │
│  ══════════════════          ═════════════════        ═══════════════   │
│                                                                          │
│  ┌──────────────┐           ┌──────────────┐         ┌──────────────┐   │
│  │ net_discover │──────────▶│ vuln_assess  │────────▶│section_security│ │
│  └──────────────┘           └──────────────┘         └──────────────┘   │
│         │                          │                        │           │
│  ┌──────────────┐           ┌──────────────┐               │           │
│  │ port_scan    │──────────▶│ ssl_analysis │───────────────┤           │
│  └──────────────┘           └──────────────┘               │           │
│         │                          │                        │           │
│  ┌──────────────┐           ┌──────────────┐               │           │
│  │ dns_enum     │──────────▶│ compliance   │───────────────┘           │
│  └──────────────┘           └──────────────┘                            │
│         │                          │                                    │
│  ┌──────────────┐           ┌──────────────┐                            │
│  │ config_parse │──────────▶│ risk_network │                            │
│  └──────────────┘           └──────────────┘                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1: Discovery Kernels

### 4.1 Network Discovery (`net_discover`)

**Purpose:** Asset enumeration and network mapping.

```python
class NetDiscoverKernel(Kernel):
    name = "net_discover"
    stage = 1
    requires = []
    provides = ["hosts", "network_map"]
```

**Wraps:**
- `nmap -sn` (ping sweep)
- `arp-scan` (local LAN)
- `masscan` (large networks)

**Input:**
```yaml
net_discover:
  targets:
    - "192.168.1.0/24"
    - "10.0.0.0/16"
  exclude:
    - "192.168.1.1"  # Gateway
  methods: ["ping", "arp"]
```

**Output:**
```json
{
  "hosts": [
    {"ip": "192.168.1.10", "mac": "aa:bb:cc:dd:ee:ff", "hostname": "server1", "status": "up"},
    {"ip": "192.168.1.20", "mac": "11:22:33:44:55:66", "hostname": "workstation", "status": "up"}
  ],
  "statistics": {
    "total_scanned": 254,
    "hosts_up": 15,
    "scan_time_sec": 12.3
  }
}
```

---

### 4.2 Port Scanner (`port_scan`)

**Purpose:** Service detection and version fingerprinting.

```python
class PortScanKernel(Kernel):
    name = "port_scan"
    stage = 1
    requires = ["net_discover"]
    provides = ["services", "open_ports"]
```

**Wraps:**
- `nmap -sV` (service version)
- `nmap -sS` (SYN scan)
- `nmap --script=banner`

**Input:**
```yaml
port_scan:
  ports: "1-1024,3306,5432,8080,8443"
  scan_type: "syn"  # syn, connect, udp
  service_detection: true
  os_detection: false  # Requires root
```

**Output:**
```json
{
  "services": [
    {
      "host": "192.168.1.10",
      "port": 22,
      "protocol": "tcp",
      "state": "open",
      "service": "ssh",
      "version": "OpenSSH 8.9p1",
      "banner": "SSH-2.0-OpenSSH_8.9p1"
    },
    {
      "host": "192.168.1.10",
      "port": 443,
      "protocol": "tcp",
      "state": "open",
      "service": "https",
      "version": "nginx 1.18.0"
    }
  ]
}
```

---

### 4.3 DNS Enumeration (`dns_enum`)

**Purpose:** DNS zone analysis and misconfiguration detection.

```python
class DnsEnumKernel(Kernel):
    name = "dns_enum"
    stage = 1
    requires = []
    provides = ["dns_records", "dns_issues"]
```

**Wraps:**
- `dig` (DNS queries)
- `host` (lookups)
- `dnsrecon` (enumeration)

**Input:**
```yaml
dns_enum:
  domains:
    - "example.com"
    - "internal.example.com"
  record_types: ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME"]
  zone_transfer: true  # Attempt AXFR
```

**Output:**
```json
{
  "records": [
    {"domain": "example.com", "type": "A", "value": "93.184.216.34", "ttl": 300},
    {"domain": "example.com", "type": "MX", "value": "mail.example.com", "priority": 10}
  ],
  "issues": [
    {"domain": "example.com", "type": "zone_transfer_allowed", "severity": "HIGH", "ns": "ns1.example.com"}
  ]
}
```

---

### 4.4 Configuration Parser (`config_parse`)

**Purpose:** Parse and normalize network device configurations.

```python
class ConfigParseKernel(Kernel):
    name = "config_parse"
    stage = 1
    requires = []
    provides = ["firewall_rules", "router_config", "switch_config"]
```

**Supported formats:**
- Cisco IOS/NX-OS
- Juniper JunOS
- Palo Alto PAN-OS
- iptables/nftables
- pfSense
- FortiGate

**Input:**
```yaml
config_parse:
  configs:
    - path: "configs/firewall.conf"
      type: "iptables"
    - path: "configs/router.cfg"
      type: "cisco_ios"
```

**Output:**
```json
{
  "firewall_rules": [
    {
      "source": "config/firewall.conf",
      "chain": "INPUT",
      "rule_num": 1,
      "action": "ACCEPT",
      "protocol": "tcp",
      "dport": 22,
      "source_ip": "192.168.1.0/24"
    }
  ],
  "issues": [
    {"type": "any_any_rule", "severity": "CRITICAL", "location": "rule 15"}
  ]
}
```

---

## 5. Stage 2: Analysis Kernels

### 5.1 Vulnerability Assessment (`vuln_assess`)

**Purpose:** CVE mapping and vulnerability scoring.

```python
class VulnAssessKernel(Kernel):
    name = "vuln_assess"
    stage = 2
    requires = ["port_scan"]
    provides = ["vulnerabilities", "cve_mapping"]
```

**Wraps:**
- `nuclei` (template-based scanning)
- CVE database lookup (local NVD mirror)
- `vulners` nmap script

**Output:**
```json
{
  "vulnerabilities": [
    {
      "host": "192.168.1.10",
      "port": 22,
      "cve": "CVE-2023-48795",
      "title": "SSH Terrapin Attack",
      "cvss": 5.9,
      "severity": "MEDIUM",
      "description": "Prefix truncation attack on SSH",
      "remediation": "Upgrade to OpenSSH 9.6+"
    }
  ],
  "statistics": {
    "total_vulns": 12,
    "critical": 1,
    "high": 3,
    "medium": 5,
    "low": 3
  }
}
```

---

### 5.2 SSL/TLS Analysis (`ssl_analysis`)

**Purpose:** Certificate validation and cipher suite analysis.

```python
class SslAnalysisKernel(Kernel):
    name = "ssl_analysis"
    stage = 2
    requires = ["port_scan"]
    provides = ["certificates", "cipher_suites", "tls_issues"]
```

**Wraps:**
- `testssl.sh` (comprehensive TLS testing)
- `sslyze` (Python-native)
- `openssl s_client`

**Output:**
```json
{
  "certificates": [
    {
      "host": "192.168.1.10:443",
      "subject": "CN=server.example.com",
      "issuer": "CN=Let's Encrypt Authority X3",
      "valid_from": "2024-01-01",
      "valid_until": "2024-04-01",
      "days_remaining": 45,
      "san": ["server.example.com", "www.example.com"],
      "key_type": "RSA",
      "key_size": 2048,
      "signature": "SHA256withRSA"
    }
  ],
  "cipher_suites": [
    {"host": "192.168.1.10:443", "protocol": "TLSv1.3", "cipher": "TLS_AES_256_GCM_SHA384", "strength": "strong"}
  ],
  "issues": [
    {"host": "192.168.1.10:443", "type": "weak_cipher", "cipher": "TLS_RSA_WITH_AES_128_CBC_SHA", "severity": "MEDIUM"},
    {"host": "192.168.1.10:443", "type": "certificate_expiring_soon", "days": 45, "severity": "LOW"}
  ]
}
```

---

### 5.3 Compliance Checker (`compliance`)

**Purpose:** Check against security standards.

```python
class ComplianceKernel(Kernel):
    name = "compliance"
    stage = 2
    requires = ["port_scan", "ssl_analysis", "config_parse"]
    provides = ["compliance_results", "gaps"]
```

**Standards supported:**
- CIS Benchmarks
- NIST 800-53
- PCI-DSS
- ISO 27001
- ANSSI (French)

**Output:**
```json
{
  "framework": "CIS",
  "version": "1.0",
  "controls": [
    {
      "id": "CIS-5.1.1",
      "title": "Ensure SSH root login is disabled",
      "status": "FAIL",
      "evidence": "PermitRootLogin yes in /etc/ssh/sshd_config",
      "severity": "HIGH",
      "remediation": "Set PermitRootLogin no"
    }
  ],
  "summary": {
    "total_controls": 150,
    "passed": 120,
    "failed": 25,
    "not_applicable": 5,
    "compliance_percent": 80.0
  }
}
```

---

### 5.4 Network Risk Assessment (`risk_network`)

**Purpose:** Aggregate findings into risk scores.

```python
class RiskNetworkKernel(Kernel):
    name = "risk_network"
    stage = 2
    requires = ["vuln_assess", "ssl_analysis", "compliance"]
    provides = ["network_risk", "risk_by_host", "attack_surface"]
```

**Risk formula:**
```
Risk = (Vuln_score × 0.4) + (Exposure_score × 0.3) + (Compliance_gap × 0.3)
```

**Output:**
```json
{
  "overall_risk": 6.2,
  "risk_level": "HIGH",
  "risk_by_host": [
    {"host": "192.168.1.10", "risk": 7.5, "level": "HIGH", "primary_factor": "vulnerabilities"},
    {"host": "192.168.1.20", "risk": 3.2, "level": "MEDIUM", "primary_factor": "exposure"}
  ],
  "attack_surface": {
    "external_services": 12,
    "deprecated_protocols": 3,
    "weak_ciphers": 5,
    "expired_certs": 0,
    "open_admin_ports": 2
  }
}
```

---

## 6. Stage 3: Reporting Kernel

### Section Security (`section_security`)

**Purpose:** Generate security report section.

```python
class SectionSecurityKernel(Kernel):
    name = "section_security"
    stage = 3
    requires = ["risk_network", "vuln_assess", "compliance"]
    provides = ["security_section"]
```

**Output:** Markdown report with:
- Executive summary
- Risk heatmap (by host/service)
- Vulnerability breakdown (by severity)
- Compliance gaps
- Prioritized remediation roadmap

---

## 7. Manifest Example

```yaml
# manifest.yaml for security network audit

audit:
  name: "Network Security Audit"
  type: "security_network"
  date: "2025-12-16"

targets:
  networks:
    - "192.168.1.0/24"
  domains:
    - "example.com"

# Stage 1: Discovery
stage1:
  net_discover:
    enabled: true
    options:
      methods: ["ping", "arp"]

  port_scan:
    enabled: true
    options:
      ports: "1-1024,3306,5432,8080,8443"
      service_detection: true

  dns_enum:
    enabled: true
    options:
      zone_transfer: true

  config_parse:
    enabled: true
    options:
      configs:
        - path: "configs/firewall.conf"
          type: "iptables"

# Stage 2: Analysis
stage2:
  vuln_assess:
    enabled: true
    options:
      cvss_threshold: 4.0

  ssl_analysis:
    enabled: true

  compliance:
    enabled: true
    options:
      framework: "CIS"
      skip_not_applicable: true

  risk_network:
    enabled: true

# Stage 3: Reporting
stage3:
  section_security:
    enabled: true
    options:
      include_remediation: true
      executive_summary: true

# Output
output:
  format: "markdown"
  language: "en"
```

---

## 8. Tool Dependencies

### Required Tools (System)

| Tool | Purpose | Package |
|------|---------|---------|
| `nmap` | Port scanning, service detection | `apt install nmap` |
| `masscan` | Fast network scanning | `apt install masscan` |
| `testssl.sh` | TLS analysis | GitHub clone |
| `nuclei` | Vulnerability scanning | `go install` |
| `dig` | DNS queries | `apt install dnsutils` |

### Python Dependencies

```
python-nmap>=0.7.1
dnspython>=2.4.0
sslyze>=6.0.0
```

### Local Databases

- **NVD Mirror:** Local copy of CVE database for offline use
- **Nuclei Templates:** Local template repository

---

## 9. Sovereignty Considerations

### Local-First Design

- All scans run locally (no cloud APIs)
- CVE database mirrored locally
- No data exfiltration

### Authorization Tracking

```yaml
authorization:
  scope: "internal_network"
  authorized_by: "John Doe, CISO"
  authorization_date: "2025-12-15"
  scope_limitations:
    - "No active exploitation"
    - "Business hours only"
    - "Exclude production database servers"
```

### Audit Trail

Every scan action logged:
```json
{
  "timestamp": "2025-12-16T10:30:00Z",
  "kernel": "port_scan",
  "targets": ["192.168.1.10"],
  "action": "nmap -sV -p 1-1024",
  "authorization_ref": "AUTH-2025-001"
}
```

---

## 10. Implementation Plan

### Phase 1: Foundation (Week 1-2)

| Kernel | Priority | Effort |
|--------|----------|--------|
| `net_discover` | P1 | 2d |
| `port_scan` | P1 | 3d |
| `dns_enum` | P2 | 2d |

### Phase 2: Analysis (Week 3-4)

| Kernel | Priority | Effort |
|--------|----------|--------|
| `ssl_analysis` | P1 | 3d |
| `vuln_assess` | P1 | 4d |
| `compliance` | P2 | 3d |

### Phase 3: Integration (Week 5)

| Kernel | Priority | Effort |
|--------|----------|--------|
| `config_parse` | P2 | 3d |
| `risk_network` | P1 | 2d |
| `section_security` | P2 | 2d |

**Total estimated effort:** 5 weeks / 1 FTE

---

## 11. Package Structure

```
ragix_kernels/
├── audit/           # Code audit kernels (existing)
│   ├── ast_scan.py
│   ├── volumetry.py
│   └── ...
│
├── security/        # Security network kernels (NEW)
│   ├── __init__.py
│   ├── net_discover.py
│   ├── port_scan.py
│   ├── dns_enum.py
│   ├── config_parse.py
│   ├── ssl_analysis.py
│   ├── vuln_assess.py
│   ├── compliance.py
│   ├── risk_network.py
│   └── section_security.py
│
└── tools/           # Tool wrappers (shared)
    ├── nmap_wrapper.py
    ├── testssl_wrapper.py
    └── nuclei_wrapper.py
```

---

## 12. Questions for Discussion

1. **Scope:** Should we include web application scanning (nikto, OWASP ZAP)?
2. **Active vs Passive:** Should kernels support active probing or passive-only?
3. **Tool selection:** Which vulnerability scanner (nuclei vs OpenVAS)?
4. **Compliance frameworks:** Which standards to prioritize first?
5. **Integration:** Should security kernels integrate with code audit (e.g., dependency vulnerabilities)?

---

*Draft design for RAGIX Security Network Audit Kernels*
