# KOAS MCP Demo

Interactive demonstration of KOAS (Kernel-Orchestrated Audit System) tools via MCP.

## Features

- **16 Simplified Tools**: 8 security + 8 audit tools optimized for LLM consumption
- **Model Selection**: Choose from available Ollama models
- **Real-time Tool Trace**: Visualize tool calls as they execute
- **Scenario Browser**: Pre-built security and audit workflows
- **Dry Run Mode**: Test without actual network scans
- **Auto-workspace Management**: Temporary workspaces created automatically

## Quick Start

```bash
# From the RAGIX root directory
cd demos/koas_mcp_demo

# Install dependencies (if not already installed)
pip install fastapi uvicorn httpx

# Start the demo server
python server.py

# Open in browser
# http://127.0.0.1:8080
```

## Requirements

- Python 3.10+
- FastAPI + Uvicorn
- Ollama (for chat functionality)
- RAGIX with KOAS kernels

## Available Tools

### Security Tools

| Tool | Stage | Description |
|------|-------|-------------|
| `koas_security_discover` | 1 | Network host discovery |
| `koas_security_scan_ports` | 1 | Port scanning |
| `koas_security_dns_check` | 1 | DNS analysis |
| `koas_security_ssl_check` | 2 | SSL/TLS analysis |
| `koas_security_vuln_scan` | 2 | Vulnerability scanning |
| `koas_security_compliance` | 2 | Compliance checking |
| `koas_security_risk` | 2 | Risk scoring |
| `koas_security_report` | 3 | Report generation |

### Audit Tools

| Tool | Stage | Description |
|------|-------|-------------|
| `koas_audit_scan` | 1 | AST scanning |
| `koas_audit_metrics` | 1 | Code metrics |
| `koas_audit_dependencies` | 1 | Dependency analysis |
| `koas_audit_hotspots` | 2 | Hotspot identification |
| `koas_audit_dead_code` | 2 | Dead code detection |
| `koas_audit_risk` | 2 | Risk scoring |
| `koas_audit_compliance` | 2 | Quality compliance |
| `koas_audit_report` | 3 | Report generation |

## Pre-built Scenarios

1. **Quick Security Scan**: Fast network discovery and port scan
2. **Full Security Assessment**: Complete audit with compliance
3. **Quick Code Audit**: Fast codebase scan and metrics
4. **Full Code Audit**: Complete code quality assessment

## API Endpoints

- `GET /` - Main demo UI
- `GET /api/tools` - List available tools
- `GET /api/scenarios` - List scenarios
- `GET /api/models` - List Ollama models
- `POST /api/tool` - Execute a tool
- `POST /api/scenario` - Run a scenario
- `POST /api/chat` - Chat with LLM
- `GET /api/health` - Health check
- `WS /ws` - WebSocket for real-time updates

## Usage

### Execute a Tool

```python
import httpx

response = httpx.post("http://localhost:8080/api/tool", json={
    "tool_name": "koas_security_discover",
    "parameters": {"target": "192.168.1.0/24"},
    "dry_run": True
})
print(response.json())
```

### Run a Scenario

```python
response = httpx.post("http://localhost:8080/api/scenario", json={
    "scenario_id": "security_quick",
    "dry_run": True
})
print(response.json())
```

## For LLM Testing

This demo is designed to test KOAS tool usage with various LLM sizes:

| Model | Expected Tool Accuracy |
|-------|------------------------|
| Mistral 7B | 70% (baseline) |
| Llama 3.1 70B | 90%+ |
| Qwen 2.5 72B | 90%+ |
| DeepSeek-V2 236B | 95%+ |

## Author

Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-19
