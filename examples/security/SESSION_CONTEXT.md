# Session Context — RAGIX Security Examples

**Session Date:** 2025-12-17
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Summary

This session completed the implementation of security kernels and examples for RAGIX KOAS.

## Completed Work

### 1. Security Kernels (10 implemented)

| Kernel | Stage | File | Purpose |
|--------|-------|------|---------|
| `net_discover` | 1 | `net_discover.py` | Network asset enumeration |
| `port_scan` | 1 | `port_scan.py` | Service/port detection |
| `dns_enum` | 1 | `dns_enum.py` | DNS analysis |
| `config_parse` | 1 | `config_parse.py` | Firewall config parsing |
| `ssl_analysis` | 2 | `ssl_analysis.py` | TLS/certificate audit |
| `vuln_assess` | 2 | `vuln_assess.py` | Vulnerability assessment |
| `web_scan` | 2 | `web_scan.py` | Web application scanning |
| `compliance` | 2 | `compliance.py` | Compliance checking |
| `risk_network` | 2 | `risk_network.py` | Network risk scoring |
| `section_security` | 3 | `section_security.py` | Security report |

### 2. Compliance Framework

**ANSSI (Primary):**
- `ANSSI_HYGIENE_RULES`: 42 rules from Guide d'hygiène
- `ANSSI_TLS_RULES`: TLS best practices
- French-language recommendations

**NIST CSF:**
- 5 functions: Identify, Protect, Detect, Respond, Recover
- Control mapping to findings

**CIS Controls v8:**
- 18 controls
- Implementation Groups (IG1, IG2, IG3)

### 3. Security Examples Structure

```
examples/security/
├── README.md                    # Documentation
├── run_security_demo.sh        # Interactive demo
├── local_network/
│   ├── manifest.yaml
│   └── data/targets.yaml
├── web_audit/
│   ├── manifest.yaml
│   └── data/targets.yaml
├── compliance_check/
│   ├── manifest.yaml
│   └── data/targets.yaml
└── config_audit/
    ├── manifest.yaml
    └── data/
        └── configs/
            ├── sample_iptables.rules
            └── sample_cisco.conf
```

### 4. Tools Integration

**Installed:**
- nikto (web scanner)
- OWASP ZAP 2.17.0 (via snap)

**Supported:**
- nmap (network scanning)
- testssl.sh (TLS analysis)
- nuclei (vulnerability scanning)
- arp-scan (network discovery)
- dig (DNS queries)

### 5. Config Parsing Support

Firewall formats:
- iptables
- nftables
- Cisco IOS
- Cisco ASA
- pfSense
- FortiGate

Security checks:
- Default allow policies
- Wide open rules
- Insecure services (telnet, FTP)
- Missing egress filtering
- Exposed management interfaces

## Demo Test Results

Config audit demo successfully found:
- 13 security issues in sample configs
- 2 critical (iptables)
- 10 high (iptables + Cisco)
- 1 medium (Cisco)

## Files Created

**Kernels:**
- `ragix_kernels/security/web_scan.py`
- `ragix_kernels/security/compliance.py`
- `ragix_kernels/security/config_parse.py`
- `ragix_kernels/security/__init__.py` (updated)

**Examples:**
- `examples/security/run_security_demo.sh`
- `examples/security/README.md`
- All workspace manifests and data files

## Next Steps (Potential)

1. Add nuclei templates integration
2. Implement CVSS scoring in vuln_assess
3. Add more firewall format parsers
4. Create compliance report PDF export
5. Add real network scan capabilities (requires authorization)

---

*Context saved for session continuity.*
