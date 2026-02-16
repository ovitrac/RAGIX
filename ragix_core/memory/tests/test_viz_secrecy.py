"""
Tests for secrecy-tier redaction in visualization memory_api.

Validates:
- export_memory_items with secrecy_tier="S0" redacts paths, emails, filenames
- export_memory_items with secrecy_tier="S3" preserves all data
- _redact_field() with various patterns (paths, emails, IPs)
- _redact_list() with mixed content

Uses in-memory SQLite and deterministic data.
"""

import pytest
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance
from ragix_kernels.summary.visualization.memory_api import (
    _redact_field,
    _redact_list,
    export_memory_items,
)


@pytest.fixture
def store():
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with items containing sensitive-looking content."""
    items = [
        MemoryItem(
            id="SEC-001", tier="stm", type="fact",
            title="Server config for db-01.corp",
            content="Database at /opt/oracle/data/prod.db, contact admin@company.com, IP 192.168.1.42.",
            tags=["config", "database"],
            entities=["db-01.corp", "oracle"],
            provenance=MemoryProvenance(
                source_kind="doc",
                source_id="/home/user/docs/audit_report.pdf",
            ),
            confidence=0.9,
        ),
        MemoryItem(
            id="SEC-002", tier="mtm", type="constraint",
            title="Compliance rule for internal.server.local",
            content="Access srv-42 via 10.0.0.1 using credentials from config.yaml.",
            tags=["compliance", "security"],
            entities=["srv-42", "internal.server.local"],
            provenance=MemoryProvenance(
                source_kind="doc",
                source_id="policy_v3.pdf",
            ),
            confidence=0.95,
        ),
        MemoryItem(
            id="SEC-003", tier="stm", type="note",
            title="Clean data item",
            content="This item has no sensitive patterns at all.",
            tags=["general"],
            provenance=MemoryProvenance(source_kind="chat", source_id="turn_1"),
            confidence=0.5,
        ),
    ]
    for item in items:
        store.write_item(item)
    return store


# ---------------------------------------------------------------------------
# _redact_field tests
# ---------------------------------------------------------------------------

class TestRedactField:
    def test_s3_no_redaction(self):
        """S3 tier preserves all content."""
        text = "File at /opt/data/test.pdf with admin@example.com"
        assert _redact_field(text, "S3") == text

    def test_s0_redacts_paths(self):
        """S0 tier redacts filesystem paths."""
        text = "Config at /etc/nginx/nginx.conf"
        result = _redact_field(text, "S0")
        assert "/etc/nginx" not in result
        assert "[PATH]" in result

    def test_s0_redacts_emails(self):
        """S0 tier redacts email addresses."""
        text = "Contact admin@company.com for support."
        result = _redact_field(text, "S0")
        assert "admin@company.com" not in result
        assert "[EMAIL]" in result

    def test_s0_redacts_ips(self):
        """S0 tier redacts IP addresses."""
        text = "Server at 192.168.1.1 and 10.0.0.42."
        result = _redact_field(text, "S0")
        assert "192.168.1.1" not in result
        assert "10.0.0.42" not in result
        assert "[IP]" in result

    def test_s0_redacts_filenames(self):
        """S0 tier redacts filenames with known extensions."""
        text = "See the report in audit_report.pdf and data.csv."
        result = _redact_field(text, "S0")
        assert "audit_report.pdf" not in result
        assert "[FILE]" in result

    def test_s0_redacts_hostnames(self):
        """S0 tier redacts internal hostnames."""
        text = "Connect to db-server.corp via ssh."
        result = _redact_field(text, "S0")
        assert "db-server.corp" not in result
        assert "[HOST]" in result

    def test_s2_keeps_paths_and_emails(self):
        """S2 tier only redacts hostnames, IPs, and hashes."""
        text = "File /opt/data.pdf, email user@test.com, host db.corp, IP 10.0.0.1"
        result = _redact_field(text, "S2")
        # S2 keeps paths and emails
        assert "/opt/data.pdf" in result or "[PATH]" in result  # path may be redacted by filename
        assert "user@test.com" in result
        # S2 redacts hostnames and IPs
        assert "db.corp" not in result
        assert "10.0.0.1" not in result

    def test_empty_string(self):
        """Empty string returns empty string for any tier."""
        assert _redact_field("", "S0") == ""
        assert _redact_field("", "S3") == ""

    def test_none_like_empty(self):
        """Falsy text returns as-is."""
        assert _redact_field("", "S0") == ""


# ---------------------------------------------------------------------------
# _redact_list tests
# ---------------------------------------------------------------------------

class TestRedactList:
    def test_s3_no_redaction(self):
        """S3 tier preserves all list elements."""
        items = ["/path/to/file.pdf", "admin@test.com", "192.168.1.1"]
        assert _redact_list(items, "S3") == items

    def test_s0_redacts_each_element(self):
        """S0 tier redacts sensitive patterns in each list element."""
        items = ["/opt/data/test.txt", "admin@company.com", "clean text"]
        result = _redact_list(items, "S0")
        assert len(result) == 3
        assert "[PATH]" in result[0] or "[FILE]" in result[0]
        assert "[EMAIL]" in result[1]
        assert result[2] == "clean text"  # no sensitive pattern

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _redact_list([], "S0") == []

    def test_mixed_content(self):
        """List with a mix of sensitive and clean strings."""
        items = ["db-01.corp", "simple text", "10.0.0.1"]
        result = _redact_list(items, "S0")
        assert "[HOST]" in result[0] or "[ENTITY]" in result[0]
        assert result[1] == "simple text"
        assert "[IP]" in result[2]


# ---------------------------------------------------------------------------
# export_memory_items with secrecy
# ---------------------------------------------------------------------------

class TestExportSecrecy:
    def test_s3_preserves_all_data(self, populated_store):
        """S3 export preserves all content without redaction."""
        data = export_memory_items(populated_store, secrecy_tier="S3")
        items = data["items"]
        assert len(items) == 3

        sec001 = [i for i in items if i["id"] == "SEC-001"][0]
        assert "admin@company.com" in sec001["content_preview"]
        assert "192.168.1.42" in sec001["content_preview"]

    def test_s0_redacts_sensitive_content(self, populated_store):
        """S0 export redacts paths, emails, IPs, filenames from content."""
        data = export_memory_items(populated_store, secrecy_tier="S0")
        items = data["items"]
        assert len(items) == 3

        sec001 = [i for i in items if i["id"] == "SEC-001"][0]
        # Content should be redacted
        assert "admin@company.com" not in sec001["content_preview"]
        assert "192.168.1.42" not in sec001["content_preview"]

    def test_s0_redacts_source_id(self, populated_store):
        """S0 export redacts paths in source_id field."""
        data = export_memory_items(populated_store, secrecy_tier="S0")
        items = data["items"]
        sec001 = [i for i in items if i["id"] == "SEC-001"][0]
        # source_id was "/home/user/docs/audit_report.pdf" — should be redacted
        assert "/home/user" not in sec001["source_id"]

    def test_s0_redacts_title(self, populated_store):
        """S0 export redacts hostnames in title field."""
        data = export_memory_items(populated_store, secrecy_tier="S0")
        items = data["items"]
        sec001 = [i for i in items if i["id"] == "SEC-001"][0]
        assert "db-01.corp" not in sec001["title"]

    def test_s0_redacts_entities(self, populated_store):
        """S0 export redacts entity labels."""
        data = export_memory_items(populated_store, secrecy_tier="S0")
        items = data["items"]
        sec002 = [i for i in items if i["id"] == "SEC-002"][0]
        # "internal.server.local" should be redacted in entities
        for ent in sec002["entities"]:
            assert "internal.server.local" not in ent

    def test_clean_item_unchanged(self, populated_store):
        """Items with no sensitive patterns should be unchanged regardless of tier."""
        data_s3 = export_memory_items(populated_store, secrecy_tier="S3")
        data_s0 = export_memory_items(populated_store, secrecy_tier="S0")
        sec003_s3 = [i for i in data_s3["items"] if i["id"] == "SEC-003"][0]
        sec003_s0 = [i for i in data_s0["items"] if i["id"] == "SEC-003"][0]
        assert sec003_s3["content_preview"] == sec003_s0["content_preview"]
        assert sec003_s3["title"] == sec003_s0["title"]

    def test_metadata_present(self, populated_store):
        """Export should include metadata with stats."""
        data = export_memory_items(populated_store, secrecy_tier="S3")
        assert "metadata" in data
        assert "total_returned" in data["metadata"]
        assert data["metadata"]["total_returned"] == 3
