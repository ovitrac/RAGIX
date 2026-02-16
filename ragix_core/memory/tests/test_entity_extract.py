"""
Tests for entity_extract.py — deterministic regex + vocabulary extraction.
"""

import pytest
from ragix_core.memory.entity_extract import extract_entities


class TestExtractEntities:
    """Test suite for extract_entities()."""

    def test_cve_extraction(self):
        text = "Patch CVE-2024-1234 required on all servers"
        entities = extract_entities(text)
        assert "CVE-2024-1234" in entities

    def test_product_vocabulary(self):
        text = "Deploy RHEL 8.4 with PostgreSQL and Ansible Tower"
        entities = extract_entities(text)
        assert "rhel" in entities
        assert "postgresql" in entities
        assert "ansible" in entities

    def test_version_strict_3component(self):
        text = "Upgrade to version 9.1.2 immediately"
        entities = extract_entities(text)
        assert "9.1.2" in entities

    def test_version_2component_rejected_without_product(self):
        """2-component versions alone are NOT extracted (too noisy)."""
        text = "See section 2.3 for details"
        entities = extract_entities(text)
        assert "2.3" not in entities

    def test_version_2component_product_adjacent(self):
        """2-component versions ARE extracted when adjacent to a product."""
        text = "RHEL 8.4 must be patched"
        entities = extract_entities(text)
        assert "8.4" in entities
        assert "rhel" in entities

    def test_version_prefixed(self):
        text = "Use version 1.2.3 for compatibility"
        entities = extract_entities(text)
        assert "1.2.3" in entities

    def test_config_paths(self):
        text = "Edit /etc/ssh/sshd_config to disable root login"
        entities = extract_entities(text)
        assert "/etc/ssh/sshd_config" in entities

    def test_port_numbers(self):
        text = "Open port 22/tcp and 443/tcp on the firewall"
        entities = extract_entities(text)
        assert "22/tcp" in entities
        assert "443/tcp" in entities

    def test_compliance_markers(self):
        text = "Root login MUST be disabled. Password auth SHOULD use MFA."
        entities = extract_entities(text)
        assert "MUST" in entities
        assert "SHOULD" in entities

    def test_empty_input(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        text = "This is a plain sentence with no technical content."
        entities = extract_entities(text)
        assert len(entities) == 0

    def test_deduplication(self):
        text = "Oracle Oracle Oracle 19c 19c"
        entities = extract_entities(text)
        assert entities.count("oracle") == 1

    def test_sorted_output(self):
        text = "Use RHEL with Ansible and Kubernetes for CVE-2024-0001"
        entities = extract_entities(text)
        assert entities == sorted(entities)

    def test_multiword_product(self):
        text = "Configure Active Directory with SQL Server"
        entities = extract_entities(text)
        assert "active directory" in entities
        assert "sql server" in entities

    def test_custom_vocabulary(self):
        custom = {"myproduct", "specialdb"}
        text = "Deploy MyProduct with SpecialDB for production"
        entities = extract_entities(text, product_vocabulary=custom)
        assert "myproduct" in entities
        assert "specialdb" in entities
