"""
Tests for budgeted_recall.py — domain extraction, aliases, coverage entropy.

Focuses on deterministic logic (no LLM, no store required for most tests).
"""

import math

import pytest

from ragix_core.memory.budgeted_recall import (
    _extract_domain_from_filename,
    _canonicalize_domain,
    _DOMAIN_ALIASES,
    coverage_entropy,
)


# ── _extract_domain_from_filename ────────────────────────────────────────

class TestExtractDomainFromFilename:
    """V2.3 dynamic domain extraction."""

    # --- GRDF RIE corpus (27 documents) ---

    @pytest.mark.parametrize("filename,expected", [
        ("RIE - RHEL.pdf", "rhel"),
        ("RIE - Oracle 19c.pdf", "oracle"),
        ("RIE - Java 21.pdf", "java"),
        ("RIE - TOMCAT.pdf", "tomcat"),
        ("RIE - K8S.pdf", "kubernetes"),
        ("RIE - PostgreSQL 13.pdf", "postgresql"),
        ("RIE - Ansible Tower.pdf", "ansible"),
        ("RIE - Active Directory.pdf", "active-directory"),
        ("RIE - Crowdstrike.pdf", "crowdstrike"),
        ("RIE - DNS Applicatif.pdf", "dns"),
        ("RIE - OpenSSH.pdf", "openssh"),
        ("RIE - PHP 8.pdf", "php"),
        ("RIE - PROFTPD.pdf", "proftpd"),
        ("RIE - SAN.pdf", "san"),
        ("RIE - NAS.pdf", "nas"),
        ("RIE - SQL Server 2019.pdf", "sql-server"),
        ("RIE - VSPHERE7.pdf", "vsphere"),
        ("RIE - Weblogic.pdf", "weblogic"),
        ("RIE - Windows 2016.pdf", "windows"),
        ("RIE - WINDOWS SERVEUR.pdf", "windows"),
        ("RIE - Sauvegarde.pdf", "sauvegarde"),
        ("RIE - Ordonnancement.pdf", "ordonnancement"),
        ("RIE - PORT APPLICATIF.pdf", "port-applicatif"),
        ("RIE - STK S3 EDGAR.pdf", "stk-s3-edgar"),
        ("RIE - Argo Workflows.pdf", "argo-workflows"),
        ("RIE-ANGULAR.pdf", "angular"),
        ("FAQ - RHEL.pdf", "rhel"),
    ])
    def test_grdf_rie_corpus(self, filename, expected):
        result = _extract_domain_from_filename(filename)
        assert result == expected, f"{filename} → {result}, expected {expected}"

    # --- Edge cases ---

    def test_empty_filename(self):
        assert _extract_domain_from_filename("") == "unknown"

    def test_no_prefix(self):
        # Files without RIE/FAQ prefix — full name becomes domain
        result = _extract_domain_from_filename("Kubernetes Guide.pdf")
        assert result == "kubernetes-guide"  # no version to strip, both words kept

    def test_no_extension(self):
        result = _extract_domain_from_filename("RIE - Oracle 19c")
        assert result == "oracle"

    def test_complex_versioned(self):
        # Version stripping should work with multi-part versions
        result = _extract_domain_from_filename("RIE - Java 21.pdf")
        assert result == "java"

    def test_preserves_compound_names(self):
        result = _extract_domain_from_filename("RIE - Active Directory.pdf")
        assert result == "active-directory"

    def test_alias_applied(self):
        # K8S should be aliased to kubernetes
        result = _extract_domain_from_filename("RIE - K8S.pdf")
        assert result == "kubernetes"

    def test_vsphere_alias(self):
        result = _extract_domain_from_filename("RIE - VSPHERE7.pdf")
        assert result == "vsphere"


# ── _canonicalize_domain ─────────────────────────────────────────────────

class TestCanonicalizeDomain:
    def test_known_alias(self):
        assert _canonicalize_domain("k8s") == "kubernetes"
        assert _canonicalize_domain("pg") == "postgresql"
        assert _canonicalize_domain("ssh") == "openssh"

    def test_passthrough(self):
        assert _canonicalize_domain("rhel") == "rhel"
        assert _canonicalize_domain("tomcat") == "tomcat"

    def test_all_aliases_resolve(self):
        """Every alias must resolve to a non-alias value."""
        for alias, canonical in _DOMAIN_ALIASES.items():
            assert canonical not in _DOMAIN_ALIASES or canonical == alias, (
                f"Alias chain: {alias} → {canonical} → ..."
            )


# ── coverage_entropy ─────────────────────────────────────────────────────

class TestCoverageEntropy:
    """Shannon entropy of domain distribution, normalized to [0, 1]."""

    def test_perfect_balance(self):
        # All domains have equal counts → entropy = 1.0
        counts = {"a": 10, "b": 10, "c": 10, "d": 10}
        result = coverage_entropy(counts)
        assert result == 1.0

    def test_single_domain(self):
        # Only 1 domain → entropy = 0.0 (by definition, <2 domains)
        assert coverage_entropy({"a": 100}) == 0.0

    def test_empty(self):
        assert coverage_entropy({}) == 0.0

    def test_highly_skewed(self):
        # 97 items in one domain, 1 in 3 others → low entropy
        counts = {"rhel": 97, "java": 1, "tomcat": 1, "oracle": 1}
        result = coverage_entropy(counts)
        assert result < 0.5  # very skewed

    def test_moderate_balance(self):
        # Reasonable distribution
        counts = {"a": 20, "b": 15, "c": 12, "d": 10, "e": 8}
        result = coverage_entropy(counts)
        assert 0.85 < result < 1.0  # near-balanced

    def test_two_domains_equal(self):
        counts = {"a": 50, "b": 50}
        assert coverage_entropy(counts) == 1.0

    def test_two_domains_skewed(self):
        counts = {"a": 99, "b": 1}
        result = coverage_entropy(counts)
        assert result < 0.15  # very skewed

    def test_returns_float(self):
        result = coverage_entropy({"a": 5, "b": 10, "c": 15})
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
