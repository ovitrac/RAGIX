"""
RAGIX Security Network Audit Kernels

Stage 1: Discovery
- net_discover: Network/host discovery (nmap, arp-scan)
- port_scan: Port scanning and service detection (nmap)
- dns_enum: DNS enumeration and analysis (dig, dnspython)

Stage 2: Analysis
- ssl_analysis: SSL/TLS certificate and cipher analysis (sslyze, testssl)
- vuln_assess: Vulnerability assessment (nuclei, CVE mapping)
- risk_network: Network risk scoring and aggregation

Stage 3: Reporting
- section_security: Security report section generation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

from ragix_kernels.security.net_discover import NetDiscoverKernel
from ragix_kernels.security.port_scan import PortScanKernel
from ragix_kernels.security.dns_enum import DNSEnumKernel
from ragix_kernels.security.config_parse import ConfigParseKernel
from ragix_kernels.security.ssl_analysis import SSLAnalysisKernel
from ragix_kernels.security.vuln_assess import VulnAssessKernel
from ragix_kernels.security.web_scan import WebScanKernel
from ragix_kernels.security.compliance import ComplianceKernel
from ragix_kernels.security.risk_network import RiskNetworkKernel
from ragix_kernels.security.section_security import SectionSecurityKernel

__all__ = [
    # Stage 1: Discovery
    "NetDiscoverKernel",
    "PortScanKernel",
    "DNSEnumKernel",
    "ConfigParseKernel",
    # Stage 2: Analysis
    "SSLAnalysisKernel",
    "VulnAssessKernel",
    "WebScanKernel",
    "ComplianceKernel",
    "RiskNetworkKernel",
    # Stage 3: Reporting
    "SectionSecurityKernel",
]
