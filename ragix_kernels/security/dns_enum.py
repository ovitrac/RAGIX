"""
Kernel: DNS Enumeration
Stage: 1 (Discovery)
Category: security

DNS enumeration and analysis for security audits.
Discovers subdomains, zone transfers, DNS records, and misconfigurations.

Wraps:
- dig (DNS queries)
- dnspython (Python DNS library)
- host (simple DNS lookups)

Input:
    domains: List of domains to enumerate
    record_types: DNS record types to query (A, AAAA, MX, NS, TXT, SOA, CNAME)
    check_zone_transfer: Attempt zone transfers (default: true)
    check_dnssec: Check DNSSEC configuration (default: true)
    subdomain_wordlist: Path to subdomain wordlist (optional)

Output:
    records: DNS records by domain and type
    subdomains: Discovered subdomains
    zone_transfers: Zone transfer results
    misconfigurations: Detected DNS issues

Example:
    python -m ragix_kernels.orchestrator run -w ./audit/network -s 1 -k dns_enum

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-17
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import dns.resolver
    import dns.zone
    import dns.query
    import dns.rdatatype
    import dns.exception
    DNSPYTHON_AVAILABLE = True
except ImportError:
    DNSPYTHON_AVAILABLE = False

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


# Common DNS record types for security assessment
RECORD_TYPES = ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME", "PTR", "SRV"]

# Security-relevant TXT record patterns
SECURITY_TXT_PATTERNS = {
    "spf": "v=spf1",
    "dmarc": "v=DMARC1",
    "dkim": "v=DKIM1",
    "google_verification": "google-site-verification",
    "ms_verification": "MS=",
    "facebook_verification": "facebook-domain-verification",
}


class DNSEnumKernel(Kernel):
    """
    DNS enumeration and analysis kernel.

    Configuration options:
        domains: List of domains to enumerate
        record_types: Record types to query (default: all common types)
        check_zone_transfer: Try zone transfers (default: true)
        check_dnssec: Check DNSSEC (default: true)
        subdomain_wordlist: Path to wordlist for subdomain brute-force
        timeout: Query timeout in seconds (default: 5)

    Example manifest:
        dns_enum:
          enabled: true
          options:
            domains:
              - "example.com"
            record_types: ["A", "MX", "NS", "TXT"]
            check_zone_transfer: true
    """

    name = "dns_enum"
    version = "1.0.0"
    category = "security"
    stage = 1
    description = "DNS enumeration and analysis"

    requires = []
    provides = ["dns_records", "subdomains", "dns_misconfigurations"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Enumerate DNS records and detect misconfigurations."""

        # Get configuration
        domains = self._get_domains(input)
        if not domains:
            return {
                "records": {},
                "subdomains": [],
                "zone_transfers": {},
                "misconfigurations": [],
                "statistics": {"error": "No domains specified"},
            }

        record_types = input.config.get("record_types", RECORD_TYPES)
        check_zone_transfer = input.config.get("check_zone_transfer", True)
        check_dnssec = input.config.get("check_dnssec", True)
        timeout = input.config.get("timeout", 5)
        subdomain_wordlist = input.config.get("subdomain_wordlist")

        logger.info(f"[dns_enum] Enumerating {len(domains)} domain(s)")

        all_records = {}
        all_subdomains = []
        zone_transfers = {}
        misconfigurations = []
        statistics = {
            "domains_scanned": 0,
            "total_records": 0,
            "subdomains_found": 0,
            "zone_transfers_successful": 0,
            "misconfigurations_found": 0,
        }

        for domain in domains:
            logger.info(f"[dns_enum] Processing {domain}")
            domain_records = {}

            # Query each record type
            for rtype in record_types:
                records = self._query_records(domain, rtype, timeout)
                if records:
                    domain_records[rtype] = records
                    statistics["total_records"] += len(records)

            all_records[domain] = domain_records

            # Check for security TXT records
            txt_analysis = self._analyze_txt_records(domain_records.get("TXT", []))
            if txt_analysis.get("missing"):
                for missing in txt_analysis["missing"]:
                    misconfigurations.append({
                        "domain": domain,
                        "type": "missing_security_record",
                        "detail": f"Missing {missing} record",
                        "severity": "medium" if missing in ["spf", "dmarc"] else "low",
                    })

            # Attempt zone transfer
            if check_zone_transfer:
                ns_servers = [r["value"] for r in domain_records.get("NS", [])]
                zt_result = self._try_zone_transfer(domain, ns_servers, timeout)
                zone_transfers[domain] = zt_result
                if zt_result.get("success"):
                    statistics["zone_transfers_successful"] += 1
                    misconfigurations.append({
                        "domain": domain,
                        "type": "zone_transfer_allowed",
                        "detail": f"Zone transfer allowed on {zt_result.get('server')}",
                        "severity": "high",
                    })

            # Check DNSSEC
            if check_dnssec:
                dnssec_status = self._check_dnssec(domain, timeout)
                if not dnssec_status.get("signed"):
                    misconfigurations.append({
                        "domain": domain,
                        "type": "dnssec_not_enabled",
                        "detail": "DNSSEC not enabled for domain",
                        "severity": "low",
                    })

            # Subdomain enumeration (if wordlist provided)
            if subdomain_wordlist:
                wordlist_path = Path(subdomain_wordlist)
                if wordlist_path.exists():
                    found_subdomains = self._enumerate_subdomains(
                        domain, wordlist_path, timeout
                    )
                    all_subdomains.extend(found_subdomains)
                    statistics["subdomains_found"] += len(found_subdomains)

            statistics["domains_scanned"] += 1

        statistics["misconfigurations_found"] = len(misconfigurations)

        return {
            "records": all_records,
            "subdomains": all_subdomains,
            "zone_transfers": zone_transfers,
            "misconfigurations": misconfigurations,
            "statistics": statistics,
        }

    def _get_domains(self, input: KernelInput) -> List[str]:
        """Get domains from config or targets file."""

        # Check targets file
        for filename in ["domains.yaml", "domains.yml", "domains.txt", "targets.yaml"]:
            filepath = input.workspace / "data" / filename
            if filepath.exists():
                content = filepath.read_text()
                if filename.endswith(".txt"):
                    return [line.strip() for line in content.split("\n") if line.strip()]
                elif filename.endswith((".yaml", ".yml")):
                    try:
                        import yaml
                        data = yaml.safe_load(content) or {}
                        return data.get("domains", [])
                    except ImportError:
                        pass

        return input.config.get("domains", [])

    def _query_records(
        self,
        domain: str,
        record_type: str,
        timeout: int
    ) -> List[Dict[str, Any]]:
        """Query DNS records for a domain."""
        records = []

        if DNSPYTHON_AVAILABLE:
            try:
                resolver = dns.resolver.Resolver()
                resolver.timeout = timeout
                resolver.lifetime = timeout

                answers = resolver.resolve(domain, record_type)
                for rdata in answers:
                    record = {
                        "type": record_type,
                        "value": str(rdata),
                        "ttl": answers.rrset.ttl,
                    }

                    # Add parsed fields for specific types
                    if record_type == "MX":
                        record["preference"] = rdata.preference
                        record["exchange"] = str(rdata.exchange)
                    elif record_type == "SOA":
                        record["mname"] = str(rdata.mname)
                        record["rname"] = str(rdata.rname)
                        record["serial"] = rdata.serial

                    records.append(record)

            except dns.resolver.NXDOMAIN:
                logger.debug(f"[dns_enum] {domain}: NXDOMAIN")
            except dns.resolver.NoAnswer:
                logger.debug(f"[dns_enum] {domain}: No {record_type} records")
            except dns.exception.Timeout:
                logger.warning(f"[dns_enum] {domain}: Timeout querying {record_type}")
            except Exception as e:
                logger.error(f"[dns_enum] {domain} {record_type} error: {e}")

        else:
            # Fallback to dig
            records = self._dig_query(domain, record_type, timeout)

        return records

    def _dig_query(
        self,
        domain: str,
        record_type: str,
        timeout: int
    ) -> List[Dict[str, Any]]:
        """Query DNS using dig command."""
        records = []

        if not shutil.which("dig"):
            return records

        try:
            result = subprocess.run(
                ["dig", "+short", "+time=" + str(timeout), domain, record_type],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )

            for line in result.stdout.strip().split("\n"):
                if line:
                    records.append({
                        "type": record_type,
                        "value": line.strip(),
                        "ttl": None,  # dig +short doesn't show TTL
                    })

        except subprocess.TimeoutExpired:
            logger.warning(f"[dns_enum] dig timeout for {domain} {record_type}")
        except Exception as e:
            logger.error(f"[dns_enum] dig error: {e}")

        return records

    def _analyze_txt_records(self, txt_records: List[Dict]) -> Dict[str, Any]:
        """Analyze TXT records for security configurations."""
        found = {}
        missing = []

        txt_values = " ".join(r.get("value", "") for r in txt_records)

        for name, pattern in SECURITY_TXT_PATTERNS.items():
            if pattern.lower() in txt_values.lower():
                found[name] = True
            else:
                found[name] = False

        # Critical security records
        if not found.get("spf"):
            missing.append("spf")
        if not found.get("dmarc"):
            missing.append("dmarc")

        return {
            "found": found,
            "missing": missing,
        }

    def _try_zone_transfer(
        self,
        domain: str,
        nameservers: List[str],
        timeout: int
    ) -> Dict[str, Any]:
        """Attempt zone transfer from nameservers."""
        result = {
            "success": False,
            "server": None,
            "records_count": 0,
        }

        if DNSPYTHON_AVAILABLE:
            for ns in nameservers:
                ns = ns.rstrip(".")
                try:
                    zone = dns.zone.from_xfr(
                        dns.query.xfr(ns, domain, timeout=timeout)
                    )
                    result["success"] = True
                    result["server"] = ns
                    result["records_count"] = len(list(zone.iterate_names()))
                    logger.warning(f"[dns_enum] Zone transfer SUCCESS on {ns}!")
                    return result
                except Exception as e:
                    logger.debug(f"[dns_enum] Zone transfer failed on {ns}: {e}")

        else:
            # Fallback to dig
            for ns in nameservers:
                ns = ns.rstrip(".")
                if shutil.which("dig"):
                    try:
                        res = subprocess.run(
                            ["dig", f"@{ns}", domain, "AXFR", "+time=" + str(timeout)],
                            capture_output=True,
                            text=True,
                            timeout=timeout + 5,
                        )
                        if "Transfer failed" not in res.stdout and "XFR" in res.stdout:
                            lines = [l for l in res.stdout.split("\n") if l and not l.startswith(";")]
                            if len(lines) > 2:
                                result["success"] = True
                                result["server"] = ns
                                result["records_count"] = len(lines)
                                return result
                    except Exception:
                        pass

        return result

    def _check_dnssec(self, domain: str, timeout: int) -> Dict[str, Any]:
        """Check if DNSSEC is enabled for domain."""
        result = {
            "signed": False,
            "valid": False,
        }

        if DNSPYTHON_AVAILABLE:
            try:
                resolver = dns.resolver.Resolver()
                resolver.timeout = timeout

                # Check for DNSKEY records
                try:
                    answers = resolver.resolve(domain, "DNSKEY")
                    if answers:
                        result["signed"] = True
                except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
                    pass

                # Check for DS records at parent
                try:
                    answers = resolver.resolve(domain, "DS")
                    if answers:
                        result["valid"] = True
                except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
                    pass

            except Exception as e:
                logger.debug(f"[dns_enum] DNSSEC check error: {e}")

        return result

    def _enumerate_subdomains(
        self,
        domain: str,
        wordlist: Path,
        timeout: int
    ) -> List[Dict[str, Any]]:
        """Brute-force subdomain enumeration."""
        found = []

        try:
            words = wordlist.read_text().strip().split("\n")
        except Exception as e:
            logger.error(f"[dns_enum] Cannot read wordlist: {e}")
            return found

        logger.info(f"[dns_enum] Testing {len(words)} subdomains for {domain}")

        if DNSPYTHON_AVAILABLE:
            resolver = dns.resolver.Resolver()
            resolver.timeout = timeout
            resolver.lifetime = timeout

            for word in words[:1000]:  # Limit to prevent long scans
                subdomain = f"{word.strip()}.{domain}"
                try:
                    answers = resolver.resolve(subdomain, "A")
                    ips = [str(r) for r in answers]
                    found.append({
                        "subdomain": subdomain,
                        "ips": ips,
                        "source": "brute_force",
                    })
                    logger.info(f"[dns_enum] Found: {subdomain} -> {ips}")
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
                    pass
                except Exception:
                    pass

        return found

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        records = data.get("records", {})
        misconfigs = data.get("misconfigurations", [])

        # Count record types
        type_counts = {}
        for domain_records in records.values():
            for rtype, recs in domain_records.items():
                type_counts[rtype] = type_counts.get(rtype, 0) + len(recs)

        type_str = ", ".join(f"{t}({c})" for t, c in sorted(type_counts.items()))

        # Severity breakdown
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for m in misconfigs:
            sev = m.get("severity", "low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return (
            f"DNS Enum: {stats.get('domains_scanned', 0)} domains, "
            f"{stats.get('total_records', 0)} records. "
            f"Types: {type_str}. "
            f"Subdomains: {stats.get('subdomains_found', 0)}. "
            f"Zone transfers: {stats.get('zone_transfers_successful', 0)}. "
            f"Issues: {severity_counts['high']} high, "
            f"{severity_counts['medium']} medium, {severity_counts['low']} low."
        )
