"""
Graph Store — SQLite-backed graph for consolidation assist (Graph-RAG)

Tables (added to existing memory DB):
    graph_nodes  - Typed nodes (doc, chunk, item, entity)
    graph_edges  - Typed weighted edges (contains, adjacent, extracted_from, mentions, similar)
    graph_kv     - Key-value metadata

The graph is NOT a retrieval platform. It constrains merge candidates
during consolidation so that only items sharing provenance neighborhoods
can be clustered together — preventing cross-corpus mega-chains.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-15
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id    TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    item_id    TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    src_id        TEXT NOT NULL,
    dst_id        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, kind)
);

CREATE TABLE IF NOT EXISTS graph_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON graph_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON graph_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON graph_edges(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON graph_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_item ON graph_nodes(item_id);
"""


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Entity vocabulary filter
# ---------------------------------------------------------------------------

# Controlled vocabulary patterns — only these create entity nodes
_ENTITY_PATTERNS = [
    re.compile(r"^CVE-\d{4}-\d+$", re.I),                    # CVE IDs
    re.compile(r"^\d+(?:\.\d+){1,3}[a-z]?$"),                 # version tokens
    re.compile(r"^RVW-\d+$"),                                  # review rule IDs
    re.compile(r"^MEM-[0-9a-f]+$", re.I),                     # memory item IDs
    re.compile(r"^(?:MUST|SHALL|SHOULD|PROHIBITED|REQUIRED)$", re.I),  # compliance markers
    re.compile(r"^(?:/etc/|/var/|/opt/).+|.*\.(?:conf|cfg|yaml|yml|ini|xml|json)$"),  # config paths
    re.compile(r"^\d{1,5}(?:/tcp|/udp)$"),                     # port numbers
]

# Known product names (lower-cased for matching)
_PRODUCT_VOCABULARY = {
    "windows", "linux", "rhel", "centos", "debian", "ubuntu",
    "kubernetes", "k8s", "docker", "podman",
    "nginx", "apache", "tomcat", "haproxy",
    "postgresql", "mysql", "mariadb", "oracle", "mongodb", "redis",
    "ansible", "terraform", "puppet", "chef",
    "grafana", "prometheus", "zabbix", "nagios", "elk", "splunk",
    "kafka", "rabbitmq", "activemq",
    "openssl", "tls", "ssh", "ldap", "kerberos", "saml",
    "java", "python", "node", "dotnet", "golang",
    "git", "gitlab", "jenkins", "sonarqube", "nexus",
    "vmware", "esxi", "vsphere", "hyper-v", "kvm", "xen",
    "active directory", "ad", "dns", "dhcp", "ntp",
    "firewall", "iptables", "nftables", "pfsense",
    # V2.2: corpus-driven additions (CORP-ENERGY RIE)
    "crowdstrike", "sql server", "proftpd", "argo", "weblogic",
    "openssh", "php", "nas", "san", "angular",
}


def _is_valid_entity(name: str) -> bool:
    """
    Check if an entity name is from the controlled vocabulary.

    Rejects free-text noise like generic words. Only typed entities
    (versions, CVEs, products, config tokens, compliance markers)
    create graph nodes.
    """
    # Skip very short or very long strings
    if len(name) < 2 or len(name) > 60:
        return False

    # Check product vocabulary (case-insensitive)
    if name.lower() in _PRODUCT_VOCABULARY:
        return True

    # Check regex patterns
    for pat in _ENTITY_PATTERNS:
        if pat.match(name):
            return True

    return False


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """
    SQLite-backed graph store sharing the same DB as MemoryStore.

    Node kinds: doc, chunk, item, entity
    Edge kinds: contains, adjacent, extracted_from, mentions, similar

    The graph is cheap to rebuild (all edges are deterministic except
    'similar' which requires embeddings). Typical rebuild: <30s for
    ~1,500 nodes and ~3,000 edges.
    """

    def __init__(self, db_path: str):
        """Initialize graph store sharing the given SQLite database."""
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()
        logger.debug(f"GraphStore initialized: {db_path}")

    def _ensure_tables(self) -> None:
        """Create graph tables if they don't exist (safe on existing DB)."""
        self._conn.executescript(_GRAPH_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    # -- Node CRUD ---------------------------------------------------------

    def add_node(
        self, node_id: str, kind: str, label: str = "", item_id: Optional[str] = None,
    ) -> None:
        """Insert or update a graph node."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO graph_nodes
                   (node_id, kind, label, item_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (node_id, kind, label, item_id, _now_iso()),
            )
            self._conn.commit()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Read a single node."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM graph_nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if row is None:
                return None
            return {
                "node_id": row["node_id"],
                "kind": row["kind"],
                "label": row["label"],
                "item_id": row["item_id"],
                "created_at": row["created_at"],
            }

    # -- Edge CRUD ---------------------------------------------------------

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        kind: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a graph edge."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO graph_edges
                   (src_id, dst_id, kind, weight, metadata_json, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    src_id, dst_id, kind, weight,
                    json.dumps(metadata or {}), _now_iso(),
                ),
            )
            self._conn.commit()

    # -- Traversal ---------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        depth: int = 1,
        edge_kinds: Optional[List[str]] = None,
        max_size: int = 500,
    ) -> Set[str]:
        """
        BFS to `depth`, optionally filtering by edge kind.
        Returns set of reachable node_ids (excludes start node).
        """
        visited: Set[str] = set()
        queue: deque = deque([(node_id, 0)])
        visited.add(node_id)

        with self._lock:
            while queue and len(visited) < max_size:
                current, d = queue.popleft()
                if d >= depth:
                    continue

                # Outgoing edges
                if edge_kinds:
                    placeholders = ",".join("?" for _ in edge_kinds)
                    rows = self._conn.execute(
                        f"SELECT dst_id FROM graph_edges WHERE src_id=? AND kind IN ({placeholders})",
                        [current] + edge_kinds,
                    ).fetchall()
                    rows += self._conn.execute(
                        f"SELECT src_id FROM graph_edges WHERE dst_id=? AND kind IN ({placeholders})",
                        [current] + edge_kinds,
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT dst_id FROM graph_edges WHERE src_id=?", (current,)
                    ).fetchall()
                    rows += self._conn.execute(
                        "SELECT src_id FROM graph_edges WHERE dst_id=?", (current,)
                    ).fetchall()

                for row in rows:
                    nid = row[0]
                    if nid not in visited:
                        visited.add(nid)
                        queue.append((nid, d + 1))

        visited.discard(node_id)  # exclude start
        return visited

    def neighborhood_items(
        self,
        item_id: str,
        depth: int = 2,
        max_size: int = 50,
    ) -> List[str]:
        """
        Return memory item IDs reachable from item_id within depth.

        This is the key method for consolidation: only items sharing
        a neighborhood are merge candidates. Filters to 'item' nodes only.
        """
        node_id = f"item:{item_id}"
        all_reachable = self.neighbors(node_id, depth=depth, max_size=max_size * 5)

        # Filter to item nodes only
        item_ids = []
        for nid in all_reachable:
            if nid.startswith("item:"):
                mid = nid[5:]  # strip "item:" prefix
                if mid != item_id:
                    item_ids.append(mid)
                    if len(item_ids) >= max_size:
                        break
        return item_ids

    # -- Statistics --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Node/edge counts by kind."""
        with self._lock:
            node_counts = {}
            for row in self._conn.execute(
                "SELECT kind, COUNT(*) as cnt FROM graph_nodes GROUP BY kind"
            ).fetchall():
                node_counts[row["kind"]] = row["cnt"]

            edge_counts = {}
            for row in self._conn.execute(
                "SELECT kind, COUNT(*) as cnt FROM graph_edges GROUP BY kind"
            ).fetchall():
                edge_counts[row["kind"]] = row["cnt"]

            total_nodes = sum(node_counts.values())
            total_edges = sum(edge_counts.values())

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "nodes_by_kind": node_counts,
            "edges_by_kind": edge_counts,
        }

    # -- Maintenance -------------------------------------------------------

    def clear(self) -> None:
        """Drop all graph data (for rebuild)."""
        with self._lock:
            self._conn.execute("DELETE FROM graph_edges")
            self._conn.execute("DELETE FROM graph_nodes")
            self._conn.execute("DELETE FROM graph_kv")
            self._conn.commit()
        logger.info("GraphStore cleared")

    def compact(self, store) -> Dict[str, int]:
        """
        Post-consolidation graph compaction:
          1. Find superseded item nodes (item archived + superseded_by set)
          2. Rewire edges from superseded → canonical node
          3. Remove superseded item nodes
          4. Remove orphaned similar edges pointing to archived items

        Prevents graph pollution after repeated consolidations.
        Returns compaction stats.
        """
        stats = {"rewired": 0, "nodes_removed": 0, "edges_removed": 0}

        with self._lock:
            # Find all item nodes in graph
            item_nodes = self._conn.execute(
                "SELECT node_id, item_id FROM graph_nodes WHERE kind='item' AND item_id IS NOT NULL"
            ).fetchall()

        for row in item_nodes:
            node_id = row["node_id"]
            item_id = row["item_id"]

            # Check if item is superseded in the store
            item = store.read_item(item_id)
            if item is None or not item.superseded_by:
                continue

            canonical_id = item.superseded_by
            canonical_node = f"item:{canonical_id}"

            with self._lock:
                # Ensure canonical node exists
                exists = self._conn.execute(
                    "SELECT 1 FROM graph_nodes WHERE node_id=?", (canonical_node,)
                ).fetchone()
                if not exists:
                    # Canonical not in graph yet — skip (will be added on rebuild)
                    continue

                # Rewire edges: replace superseded node with canonical
                # Outgoing edges (src_id = superseded)
                out_edges = self._conn.execute(
                    "SELECT dst_id, kind, weight, metadata_json FROM graph_edges WHERE src_id=?",
                    (node_id,),
                ).fetchall()
                for edge in out_edges:
                    # Skip similar edges (don't rewire — they become stale)
                    if edge["kind"] == "similar":
                        self._conn.execute(
                            "DELETE FROM graph_edges WHERE src_id=? AND dst_id=? AND kind=?",
                            (node_id, edge["dst_id"], edge["kind"]),
                        )
                        stats["edges_removed"] += 1
                        continue
                    # Insert redirected edge (ignore if already exists)
                    try:
                        self._conn.execute(
                            """INSERT OR IGNORE INTO graph_edges
                               (src_id, dst_id, kind, weight, metadata_json, created_at)
                               VALUES (?,?,?,?,?,?)""",
                            (canonical_node, edge["dst_id"], edge["kind"],
                             edge["weight"], edge["metadata_json"], _now_iso()),
                        )
                        stats["rewired"] += 1
                    except Exception:
                        pass

                # Incoming edges (dst_id = superseded)
                in_edges = self._conn.execute(
                    "SELECT src_id, kind, weight, metadata_json FROM graph_edges WHERE dst_id=?",
                    (node_id,),
                ).fetchall()
                for edge in in_edges:
                    if edge["kind"] == "similar":
                        self._conn.execute(
                            "DELETE FROM graph_edges WHERE src_id=? AND dst_id=? AND kind=?",
                            (edge["src_id"], node_id, edge["kind"]),
                        )
                        stats["edges_removed"] += 1
                        continue
                    try:
                        self._conn.execute(
                            """INSERT OR IGNORE INTO graph_edges
                               (src_id, dst_id, kind, weight, metadata_json, created_at)
                               VALUES (?,?,?,?,?,?)""",
                            (edge["src_id"], canonical_node, edge["kind"],
                             edge["weight"], edge["metadata_json"], _now_iso()),
                        )
                        stats["rewired"] += 1
                    except Exception:
                        pass

                # Delete all edges for superseded node
                self._conn.execute(
                    "DELETE FROM graph_edges WHERE src_id=? OR dst_id=?",
                    (node_id, node_id),
                )
                # Delete the superseded node
                self._conn.execute(
                    "DELETE FROM graph_nodes WHERE node_id=?", (node_id,),
                )
                stats["nodes_removed"] += 1
                self._conn.commit()

        logger.info(
            f"Graph compacted: {stats['nodes_removed']} nodes removed, "
            f"{stats['rewired']} edges rewired, "
            f"{stats['edges_removed']} stale edges removed"
        )
        return stats

    # -- Bulk Build --------------------------------------------------------

    def build_from_store(
        self,
        store,
        scope: str,
        embedder=None,
        sim_threshold: float = 0.85,
        similarity_top_k: int = 0,
        max_edges_per_node: int = 0,
    ) -> Dict[str, int]:
        """
        Bulk build deterministic edges from existing MemoryStore data.

        4 edge kinds built deterministically:
          1. contains:       doc_node → chunk_node
          2. adjacent:       chunk_node → chunk_node (consecutive in same doc)
          3. extracted_from: item_node → chunk_node
          4. mentions:       item_node → entity_node

        Optional 5th kind (if embedder provided):
          5. similar: item_node → item_node (cosine > threshold)

        Returns counts by edge kind.
        """
        counts = {
            "contains": 0, "adjacent": 0,
            "extracted_from": 0, "mentions": 0, "similar": 0,
        }

        items = store.list_items(scope=scope, exclude_archived=True, limit=10000)
        if not items:
            logger.warning("No items found for graph build")
            return counts

        logger.info(f"Building graph from {len(items)} items (scope={scope})")

        # Collect all unique doc+chunk references from provenance
        doc_chunks: Dict[str, List[str]] = {}  # doc_name -> sorted chunk indices
        chunk_to_node: Dict[str, str] = {}     # "filename:N" -> chunk_node_id

        for item in items:
            source_id = item.provenance.source_id or ""
            # Parse source_id format: "filename.pdf" or "filename.pdf:chunk_3"
            if ":" in source_id and "chunk" in source_id:
                parts = source_id.rsplit(":", 1)
                doc_name = parts[0]
                chunk_ref = parts[1]
            else:
                doc_name = source_id
                chunk_ref = None

            if doc_name:
                if doc_name not in doc_chunks:
                    doc_chunks[doc_name] = []

            # Also parse chunk_ids from provenance
            for cid in item.provenance.chunk_ids:
                # chunk_ids may be "filename:chunk_N" or just "chunk_N"
                if ":" in cid:
                    cparts = cid.rsplit(":", 1)
                    cdoc = cparts[0]
                    cidx = cparts[1]
                else:
                    cdoc = doc_name
                    cidx = cid
                if cdoc and cidx:
                    if cdoc not in doc_chunks:
                        doc_chunks[cdoc] = []
                    if cidx not in doc_chunks[cdoc]:
                        doc_chunks[cdoc].append(cidx)

        # Create doc nodes + chunk nodes
        for doc_name, chunk_refs in doc_chunks.items():
            doc_node = f"doc:{doc_name}"
            self.add_node(doc_node, "doc", label=doc_name)

            # Sort chunk refs numerically if possible
            def _chunk_sort_key(c):
                import re as _re
                m = _re.search(r"(\d+)", c)
                return int(m.group(1)) if m else 0

            sorted_chunks = sorted(set(chunk_refs), key=_chunk_sort_key)

            prev_chunk_node = None
            for cref in sorted_chunks:
                chunk_node = f"chunk:{doc_name}:{cref}"
                self.add_node(chunk_node, "chunk", label=f"{doc_name}:{cref}")
                chunk_to_node[f"{doc_name}:{cref}"] = chunk_node

                # Edge 1: contains (doc → chunk)
                self.add_edge(doc_node, chunk_node, "contains")
                counts["contains"] += 1

                # Edge 2: adjacent (consecutive chunks)
                if prev_chunk_node is not None:
                    self.add_edge(prev_chunk_node, chunk_node, "adjacent")
                    counts["adjacent"] += 1
                prev_chunk_node = chunk_node

        # Create item nodes + extracted_from + mentions edges
        entity_nodes_created: Set[str] = set()

        for item in items:
            item_node = f"item:{item.id}"
            self.add_node(item_node, "item", label=item.title[:80], item_id=item.id)

            # Edge 3: extracted_from (item → chunk)
            source_id = item.provenance.source_id or ""
            linked_chunks: Set[str] = set()

            for cid in item.provenance.chunk_ids:
                if ":" in cid:
                    key = cid
                else:
                    # Try to pair with source_id doc
                    doc_name = source_id.rsplit(":", 1)[0] if ":" in source_id else source_id
                    key = f"{doc_name}:{cid}" if doc_name else cid

                if key in chunk_to_node:
                    linked_chunks.add(key)

            # If no chunk_ids matched, try to link via source_id
            if not linked_chunks and source_id:
                for ckey in chunk_to_node:
                    if ckey.startswith(source_id.split(":")[0]):
                        linked_chunks.add(ckey)
                        break

            for ckey in linked_chunks:
                self.add_edge(item_node, chunk_to_node[ckey], "extracted_from")
                counts["extracted_from"] += 1

            # Edge 4: mentions (item → entity, filtered by controlled vocabulary)
            for entity in item.entities:
                if not _is_valid_entity(entity):
                    continue
                entity_node = f"entity:{entity.lower()}"
                if entity_node not in entity_nodes_created:
                    self.add_node(entity_node, "entity", label=entity)
                    entity_nodes_created.add(entity_node)
                self.add_edge(item_node, entity_node, "mentions")
                counts["mentions"] += 1

        # Edge 5: similar (item → item, requires embeddings)
        if embedder is not None:
            counts["similar"] = self._build_similarity_edges(
                store, items, embedder, sim_threshold, top_k=similarity_top_k,
            )

        # V2.4: Enforce edge cap if specified
        if max_edges_per_node > 0:
            removed = self.enforce_edge_cap(max_edges_per_node)
            if removed > 0:
                logger.info(f"Edge cap removed {removed} edges")

        logger.info(
            f"Graph built: {self.stats()['total_nodes']} nodes, "
            f"{self.stats()['total_edges']} edges"
        )
        return counts

    def _build_similarity_edges(
        self,
        store,
        items,
        embedder,
        threshold: float,
        top_k: int = 0,
    ) -> int:
        """
        Build 'similar' edges between items with cosine > threshold.

        V2.4: If top_k > 0, only keep the top-k most similar edges per item
        (reduces O(n^2) to O(n*k) effective edges).
        """
        from ragix_core.memory.embedder import cosine_similarity

        count = 0
        # Load all embeddings
        embs: Dict[str, List[float]] = {}
        for item in items:
            data = store.read_embedding(item.id)
            if data is not None:
                embs[item.id] = data[0]

        if len(embs) < 2:
            return 0

        item_ids = list(embs.keys())
        logger.info(
            f"Computing similarity edges for {len(item_ids)} items "
            f"(threshold={threshold}, top_k={top_k or 'unlimited'})"
        )

        if top_k > 0:
            # V2.4: Per-item top-k constraint
            for i, iid in enumerate(item_ids):
                # Compute similarity with all other items
                candidates = []
                for j, jid in enumerate(item_ids):
                    if j <= i:
                        continue
                    sim = cosine_similarity(embs[iid], embs[jid])
                    if sim >= threshold:
                        candidates.append((jid, sim))
                # Keep top-k by similarity (descending)
                candidates.sort(key=lambda x: x[1], reverse=True)
                for jid, sim in candidates[:top_k]:
                    self.add_edge(
                        f"item:{iid}", f"item:{jid}", "similar", weight=sim,
                    )
                    count += 1
        else:
            # Original all-pairs (backward compatible)
            for i in range(len(item_ids)):
                for j in range(i + 1, len(item_ids)):
                    sim = cosine_similarity(embs[item_ids[i]], embs[item_ids[j]])
                    if sim >= threshold:
                        self.add_edge(
                            f"item:{item_ids[i]}",
                            f"item:{item_ids[j]}",
                            "similar",
                            weight=sim,
                        )
                        count += 1

        logger.info(f"Similarity edges: {count}")
        return count

    # -- Export (secrecy-aware) ------------------------------------------------

    def export_graph(self, tier: str = "S3") -> Dict[str, Any]:
        """
        Export full graph as JSON dict, applying secrecy-tier redaction.

        At S0: entity labels, filenames, paths, IPs, hostnames redacted.
        At S2: hostnames, IPs, hashes redacted.
        At S3: no redaction (full detail).
        """
        from ragix_kernels.summary.kernels.summary_redact import (
            redact_for_storage,
        )

        with self._lock:
            nodes_raw = self._conn.execute(
                "SELECT node_id, kind, label, item_id, created_at FROM graph_nodes"
            ).fetchall()
            edges_raw = self._conn.execute(
                "SELECT src_id, dst_id, kind, weight, metadata_json, created_at FROM graph_edges"
            ).fetchall()

        nodes = []
        for row in nodes_raw:
            label = row["label"]
            node_id = row["node_id"]
            if tier != "S3":
                label = redact_for_storage(label, tier)
                # Redact entity node IDs at S0 (they carry infra names)
                if tier == "S0" and row["kind"] == "entity":
                    node_id = f"entity:{redact_for_storage(row['node_id'][7:], tier)}"
            nodes.append({
                "id": node_id,
                "kind": row["kind"],
                "label": label,
                "item_id": row["item_id"],
            })

        edges = []
        for row in edges_raw:
            src_id = row["src_id"]
            dst_id = row["dst_id"]
            meta = row["metadata_json"]
            if tier != "S3":
                meta = redact_for_storage(meta, tier)
                if tier == "S0":
                    # Redact entity references in edge endpoints
                    if src_id.startswith("entity:"):
                        src_id = f"entity:{redact_for_storage(src_id[7:], tier)}"
                    if dst_id.startswith("entity:"):
                        dst_id = f"entity:{redact_for_storage(dst_id[7:], tier)}"
            edges.append({
                "src": src_id,
                "dst": dst_id,
                "kind": row["kind"],
                "weight": row["weight"],
                "metadata": json.loads(meta) if meta else {},
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.stats(),
            "tier": tier,
        }

    # -- V3.0: Cross-corpus edges ────────────────────────────────────────────

    def build_cross_corpus_edges(
        self,
        store,
        corpus_a: str,
        corpus_b: str,
        scope: str = "project",
    ) -> Dict[str, int]:
        """
        Build cross-corpus edges between items in two corpora.

        Edge kinds:
          - same_rule:     matching rule_id, same content_hash
          - evolved_from:  matching rule_id, different content_hash

        Returns counts by edge kind.
        """
        counts = {"same_rule": 0, "evolved_from": 0}

        items_a = store.list_items(
            scope=scope, corpus_id=corpus_a, exclude_archived=True, limit=10000,
        )
        items_b = store.list_items(
            scope=scope, corpus_id=corpus_b, exclude_archived=True, limit=10000,
        )

        # Index A by rule_id
        a_by_rule: Dict[str, Any] = {}
        for item in items_a:
            if item.rule_id:
                a_by_rule[item.rule_id] = item

        for item_b in items_b:
            if not item_b.rule_id or item_b.rule_id not in a_by_rule:
                continue
            item_a = a_by_rule[item_b.rule_id]
            src = f"item:{item_a.id}"
            dst = f"item:{item_b.id}"

            # Ensure both nodes exist
            self.add_node(src, "item", label=item_a.title[:80], item_id=item_a.id)
            self.add_node(dst, "item", label=item_b.title[:80], item_id=item_b.id)

            if item_a.content_hash == item_b.content_hash:
                self.add_edge(src, dst, "same_rule", weight=1.0, metadata={
                    "rule_id": item_b.rule_id,
                    "corpus_a": corpus_a,
                    "corpus_b": corpus_b,
                })
                counts["same_rule"] += 1
            else:
                self.add_edge(dst, src, "evolved_from", weight=0.8, metadata={
                    "rule_id": item_b.rule_id,
                    "corpus_a": corpus_a,
                    "corpus_b": corpus_b,
                })
                counts["evolved_from"] += 1

        logger.info(
            f"Cross-corpus edges: {counts['same_rule']} same_rule, "
            f"{counts['evolved_from']} evolved_from"
        )
        return counts

    def enforce_edge_cap(self, max_edges_per_node: int = 100) -> int:
        """
        V2.4: Enforce max edges per node to prevent hub explosion.

        For nodes exceeding the cap, keeps edges with highest weight
        and removes the rest. Returns total edges removed.
        """
        removed = 0
        with self._lock:
            # Find nodes with too many edges
            rows = self._conn.execute(
                """SELECT node_id, cnt FROM (
                    SELECT src_id AS node_id, COUNT(*) AS cnt FROM graph_edges GROUP BY src_id
                    UNION ALL
                    SELECT dst_id AS node_id, COUNT(*) AS cnt FROM graph_edges GROUP BY dst_id
                ) GROUP BY node_id HAVING SUM(cnt) > ?""",
                (max_edges_per_node,),
            ).fetchall()

        for row in rows:
            node_id = row["node_id"]
            with self._lock:
                # Get all edges for this node, sorted by weight descending
                edges = self._conn.execute(
                    """SELECT rowid, src_id, dst_id, kind, weight FROM graph_edges
                       WHERE src_id=? OR dst_id=?
                       ORDER BY weight DESC""",
                    (node_id, node_id),
                ).fetchall()

            if len(edges) <= max_edges_per_node:
                continue

            # Remove excess edges (lowest weight first)
            to_remove = edges[max_edges_per_node:]
            with self._lock:
                for edge in to_remove:
                    self._conn.execute(
                        "DELETE FROM graph_edges WHERE src_id=? AND dst_id=? AND kind=?",
                        (edge["src_id"], edge["dst_id"], edge["kind"]),
                    )
                    removed += 1
                self._conn.commit()

        if removed > 0:
            logger.info(f"Edge cap enforced: {removed} edges removed (cap={max_edges_per_node})")
        return removed
