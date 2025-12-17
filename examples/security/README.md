# Security Kernel Examples

This directory contains example workspaces demonstrating the RAGIX Security Network Audit kernels.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17

---

## Available Examples

| Example | Description | Kernels Used |
|---------|-------------|--------------|
| `local_network/` | Scan local network for hosts and services | net_discover, port_scan, risk_network |
| `web_audit/` | Web application security audit | port_scan, ssl_analysis, vuln_assess, web_scan |
| `compliance_check/` | Compliance assessment against ANSSI/NIST/CIS | All Stage 1+2, compliance |
| `config_audit/` | Firewall configuration analysis | config_parse |

---

## Prerequisites

Before running examples, ensure tools are installed:

```bash
# Check tool availability
python scripts/check_security_tools.py

# Required tools
sudo apt install nmap dnsutils nikto
sudo snap install zaproxy --classic
pip install python-nmap dnspython sslyze
```

---

## Quick Start

### Example 1: Local Network Scan

Scan your local network for hosts and open services.

```bash
cd examples/security/local_network

# Edit targets (adjust to your network)
nano data/targets.yaml

# Run discovery (Stage 1)
python -m ragix_kernels.orchestrator run -w . -s 1

# View results
cat stage1/net_discover.json | python -m json.tool
cat stage1/port_scan.json | python -m json.tool
```

### Example 2: Web Application Audit

Comprehensive web security assessment.

```bash
cd examples/security/web_audit

# Edit targets
nano data/targets.yaml

# Run full audit
python -m ragix_kernels.orchestrator run -w .

# View report
cat stage3/security_report.md
```

### Example 3: Compliance Check

Evaluate against ANSSI, NIST, and CIS frameworks.

```bash
cd examples/security/compliance_check

# Run compliance assessment
python -m ragix_kernels.orchestrator run -w .

# View compliance scores
cat stage2/compliance.json | python -m json.tool
```

### Example 4: Firewall Config Audit

Analyze firewall rules for misconfigurations.

```bash
cd examples/security/config_audit

# Place your config files in data/configs/
cp /path/to/iptables.rules data/configs/

# Run analysis
python -m ragix_kernels.orchestrator run -w . -s 1 -k config_parse

# View findings
cat stage1/config_parse.json | python -m json.tool
```

---

## Running the Demo Script

A comprehensive demo script is provided:

```bash
# Run interactive demo
./examples/security/run_security_demo.sh

# Or run specific example
./examples/security/run_security_demo.sh local_network
./examples/security/run_security_demo.sh web_audit
./examples/security/run_security_demo.sh compliance
./examples/security/run_security_demo.sh config
```

---

## Output Structure

After running an example, outputs are organized by stage:

```
example_workspace/
├── data/
│   ├── targets.yaml      # Input targets
│   └── domains.yaml      # Input domains (optional)
├── manifest.yaml         # Kernel configuration
├── stage1/
│   ├── net_discover.json
│   ├── port_scan.json
│   └── dns_enum.json
├── stage2/
│   ├── ssl_analysis.json
│   ├── vuln_assess.json
│   ├── web_scan.json
│   ├── compliance.json
│   └── risk_network.json
└── stage3/
    └── security_report.md
```

---

## Customizing Examples

### Adjusting Targets

Edit `data/targets.yaml`:

```yaml
targets:
  - "192.168.1.0/24"      # CIDR notation
  - "10.0.0.1-10"         # IP range
  - "server.example.com"  # Hostname

exclude:
  - "192.168.1.1"         # Skip gateway

methods:
  - ping                  # ICMP ping sweep
  - arp                   # ARP scan (local only)
```

### Adjusting Scan Intensity

Edit `manifest.yaml`:

```yaml
port_scan:
  options:
    ports: "top20"        # Faster: top20, thorough: top100
    scan_type: "connect"  # No root: connect, with root: syn
```

### Selecting Compliance Frameworks

```yaml
compliance:
  options:
    frameworks:
      - anssi             # French ANSSI (primary)
      - nist              # NIST CSF
      - cis               # CIS Controls v8
    anssi_level: "standard"  # essential, standard, reinforced
```

---

## Security Considerations

**IMPORTANT:** Only scan networks and systems you are authorized to test.

- Obtain written permission before scanning
- Respect scope limitations
- Do not scan production systems without change management approval
- Use rate limiting for sensitive environments

---

## Troubleshooting

### Permission Errors

```bash
# For SYN scans, run with sudo
sudo python -m ragix_kernels.orchestrator run -w . -s 1
```

### Missing Tools

```bash
# Check what's installed
python scripts/check_security_tools.py --json

# Install missing tools
sudo apt install nmap nikto
pip install python-nmap sslyze
```

### Slow Scans

Reduce scan intensity in manifest.yaml:

```yaml
port_scan:
  options:
    ports: "top20"
    timeout: 120

vuln_assess:
  options:
    rate_limit: 50
    templates: ["cves"]
```
