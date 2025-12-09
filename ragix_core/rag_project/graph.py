"""
RAGIX Project RAG - Knowledge Graph

Maintains relationships between files, chunks, and concepts.

Graph structure:
    - File nodes: One per indexed file
    - Chunk nodes: One per text chunk
    - Concept nodes: Extracted or user-defined concepts

Edge types:
    - FILE_CONTAINS_CHUNK: File → Chunk
    - CHUNK_MENTIONS_CONCEPT: Chunk → Concept
    - CHUNK_RELATED_TO: Chunk ↔ Chunk (similarity)
    - FILE_IMPORTS_FILE: File → File (future AST integration)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Iterator
import json
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Node and Edge Types
# =============================================================================

class NodeType(str, Enum):
    """Graph node types."""
    FILE = "file"
    CHUNK = "chunk"
    CONCEPT = "concept"
    # Future: AST_CLASS, AST_METHOD, AST_FUNCTION


class EdgeType(str, Enum):
    """Graph edge types."""
    CONTAINS = "contains"           # File → Chunk
    MENTIONS = "mentions"           # Chunk → Concept
    RELATED_TO = "related_to"       # Chunk ↔ Chunk (semantic similarity)
    SIMILAR_TO = "similar_to"       # Concept ↔ Concept
    # Future AST edges:
    IMPORTS = "imports"             # File → File
    DEPENDS_ON = "depends_on"       # Class → Class
    CALLS = "calls"                 # Method → Method


# =============================================================================
# Graph Nodes
# =============================================================================

@dataclass
class GraphNode:
    """
    A node in the knowledge graph.

    Represents a file, chunk, or concept.
    """
    id: str
    type: str                       # NodeType value
    label: str                      # Display label
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(
            id=data["id"],
            type=data["type"],
            label=data["label"],
            data=data.get("data", {}),
        )

    @staticmethod
    def file_node(
        file_id: str,
        path: str,
        kind: str,
        **extra_data,
    ) -> "GraphNode":
        """Create a file node."""
        filename = Path(path).name
        return GraphNode(
            id=file_id,
            type=NodeType.FILE.value,
            label=filename,
            data={
                "path": path,
                "kind": kind,
                **extra_data,
            },
        )

    @staticmethod
    def chunk_node(
        chunk_id: str,
        file_id: str,
        file_path: str,
        line_start: int,
        line_end: int,
        kind: str,
        **extra_data,
    ) -> "GraphNode":
        """Create a chunk node."""
        filename = Path(file_path).name
        if line_start == line_end:
            label = f"{filename}:{line_start}"
        else:
            label = f"{filename}:{line_start}-{line_end}"

        return GraphNode(
            id=chunk_id,
            type=NodeType.CHUNK.value,
            label=label,
            data={
                "file_id": file_id,
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "kind": kind,
                **extra_data,
            },
        )

    @staticmethod
    def concept_node(
        concept_id: str,
        label: str,
        origin: str = "extracted",  # "extracted" or "user"
        description: str = "",
        **extra_data,
    ) -> "GraphNode":
        """Create a concept node."""
        return GraphNode(
            id=concept_id,
            type=NodeType.CONCEPT.value,
            label=label,
            data={
                "origin": origin,
                "description": description,
                **extra_data,
            },
        )


# =============================================================================
# Graph Edges
# =============================================================================

@dataclass
class GraphEdge:
    """
    An edge in the knowledge graph.

    Connects two nodes with a typed relationship.
    """
    source: str                     # Source node ID
    target: str                     # Target node ID
    type: str                       # EdgeType value
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.source,
            "to": self.target,
            "type": self.type,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(
            source=data["from"],
            target=data["to"],
            type=data["type"],
            data=data.get("data", {}),
        )

    @staticmethod
    def contains(file_id: str, chunk_id: str) -> "GraphEdge":
        """Create a FILE_CONTAINS_CHUNK edge."""
        return GraphEdge(
            source=file_id,
            target=chunk_id,
            type=EdgeType.CONTAINS.value,
        )

    @staticmethod
    def mentions(chunk_id: str, concept_id: str, score: float = 1.0) -> "GraphEdge":
        """Create a CHUNK_MENTIONS_CONCEPT edge."""
        return GraphEdge(
            source=chunk_id,
            target=concept_id,
            type=EdgeType.MENTIONS.value,
            data={"score": score},
        )

    @staticmethod
    def related_to(chunk_id_1: str, chunk_id_2: str, similarity: float) -> "GraphEdge":
        """Create a CHUNK_RELATED_TO edge."""
        return GraphEdge(
            source=chunk_id_1,
            target=chunk_id_2,
            type=EdgeType.RELATED_TO.value,
            data={"similarity": similarity},
        )


# =============================================================================
# Knowledge Graph
# =============================================================================

class KnowledgeGraph:
    """
    Knowledge graph for project RAG.

    Stores relationships between files, chunks, and concepts.
    Persisted to .RAG/metadata/graph.json
    """

    def __init__(self, project_root: Path):
        """
        Initialize knowledge graph.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.graph_path = project_root / ".RAG" / "metadata" / "graph.json"

        # In-memory storage
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []

        # Indices for fast lookup
        self._edges_by_source: Dict[str, List[GraphEdge]] = {}
        self._edges_by_target: Dict[str, List[GraphEdge]] = {}
        self._nodes_by_type: Dict[str, Set[str]] = {
            NodeType.FILE.value: set(),
            NodeType.CHUNK.value: set(),
            NodeType.CONCEPT.value: set(),
        }

        # Concept label to ID mapping
        self._concept_by_label: Dict[str, str] = {}

        # Track if loaded
        self._loaded = False

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load graph from disk.

        Returns:
            True if loaded successfully
        """
        if not self.graph_path.exists():
            logger.debug(f"No graph file at {self.graph_path}")
            self._loaded = True
            return False

        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load nodes
            for node_data in data.get("nodes", []):
                node = GraphNode.from_dict(node_data)
                self._add_node_to_index(node)

            # Load edges
            for edge_data in data.get("edges", []):
                edge = GraphEdge.from_dict(edge_data)
                self._add_edge_to_index(edge)

            self._loaded = True
            logger.info(f"Loaded graph: {len(self._nodes)} nodes, {len(self._edges)} edges")
            return True

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            self._loaded = True
            return False

    def save(self) -> bool:
        """
        Save graph to disk.

        Returns:
            True if saved successfully
        """
        # Ensure directory exists
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges],
            }

            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved graph: {len(self._nodes)} nodes, {len(self._edges)} edges")
            return True

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False

    def _ensure_loaded(self) -> None:
        """Ensure graph is loaded from disk."""
        if not self._loaded:
            self.load()

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self._ensure_loaded()
        self._add_node_to_index(node)

    def _add_node_to_index(self, node: GraphNode) -> None:
        """Add node to internal indices."""
        self._nodes[node.id] = node
        self._nodes_by_type[node.type].add(node.id)

        # Track concepts by label
        if node.type == NodeType.CONCEPT.value:
            self._concept_by_label[node.label.lower()] = node.id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        self._ensure_loaded()
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a given type."""
        self._ensure_loaded()
        node_ids = self._nodes_by_type.get(node_type.value, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        self._ensure_loaded()
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and its edges.

        Returns:
            True if node was removed
        """
        self._ensure_loaded()

        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]

        # Remove from type index
        self._nodes_by_type[node.type].discard(node_id)

        # Remove from concept index
        if node.type == NodeType.CONCEPT.value:
            self._concept_by_label.pop(node.label.lower(), None)

        # Remove edges
        self._edges = [e for e in self._edges if e.source != node_id and e.target != node_id]
        self._edges_by_source.pop(node_id, None)
        self._edges_by_target.pop(node_id, None)

        # Update edge indices
        for edges in self._edges_by_source.values():
            edges[:] = [e for e in edges if e.target != node_id]
        for edges in self._edges_by_target.values():
            edges[:] = [e for e in edges if e.source != node_id]

        # Remove node
        del self._nodes[node_id]
        return True

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self._ensure_loaded()
        self._add_edge_to_index(edge)

    def _add_edge_to_index(self, edge: GraphEdge) -> None:
        """Add edge to internal indices."""
        self._edges.append(edge)

        if edge.source not in self._edges_by_source:
            self._edges_by_source[edge.source] = []
        self._edges_by_source[edge.source].append(edge)

        if edge.target not in self._edges_by_target:
            self._edges_by_target[edge.target] = []
        self._edges_by_target[edge.target].append(edge)

    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Get all edges originating from a node."""
        self._ensure_loaded()
        return self._edges_by_source.get(node_id, [])

    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """Get all edges pointing to a node."""
        self._ensure_loaded()
        return self._edges_by_target.get(node_id, [])

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbor node IDs (both directions)."""
        self._ensure_loaded()
        neighbors = set()

        for edge in self.get_edges_from(node_id):
            neighbors.add(edge.target)
        for edge in self.get_edges_to(node_id):
            neighbors.add(edge.source)

        return list(neighbors)

    # -------------------------------------------------------------------------
    # Concept Operations
    # -------------------------------------------------------------------------

    def get_or_create_concept(
        self,
        label: str,
        origin: str = "extracted",
        description: str = "",
    ) -> GraphNode:
        """
        Get existing concept or create new one.

        Args:
            label: Concept label
            origin: "extracted" or "user"
            description: Optional description

        Returns:
            Concept node
        """
        self._ensure_loaded()

        label_lower = label.lower()
        if label_lower in self._concept_by_label:
            concept_id = self._concept_by_label[label_lower]
            return self._nodes[concept_id]

        # Create new concept
        concept_id = f"K_{label.upper().replace(' ', '_')}"
        concept = GraphNode.concept_node(
            concept_id=concept_id,
            label=label,
            origin=origin,
            description=description,
        )
        self.add_node(concept)
        return concept

    def get_concept_by_label(self, label: str) -> Optional[GraphNode]:
        """Get concept by label."""
        self._ensure_loaded()
        concept_id = self._concept_by_label.get(label.lower())
        if concept_id:
            return self._nodes.get(concept_id)
        return None

    def get_chunks_for_concept(self, concept_id: str) -> List[GraphNode]:
        """Get all chunks that mention a concept."""
        self._ensure_loaded()
        chunk_ids = []

        for edge in self.get_edges_to(concept_id):
            if edge.type == EdgeType.MENTIONS.value:
                chunk_ids.append(edge.source)

        return [self._nodes[cid] for cid in chunk_ids if cid in self._nodes]

    def get_files_for_concept(self, concept_id: str) -> List[GraphNode]:
        """Get all files containing chunks that mention a concept."""
        self._ensure_loaded()
        file_ids = set()

        for chunk in self.get_chunks_for_concept(concept_id):
            file_id = chunk.data.get("file_id")
            if file_id:
                file_ids.add(file_id)

        return [self._nodes[fid] for fid in file_ids if fid in self._nodes]

    # -------------------------------------------------------------------------
    # File/Chunk Operations
    # -------------------------------------------------------------------------

    def get_chunks_for_file(self, file_id: str) -> List[GraphNode]:
        """Get all chunks belonging to a file."""
        self._ensure_loaded()
        chunk_ids = []

        for edge in self.get_edges_from(file_id):
            if edge.type == EdgeType.CONTAINS.value:
                chunk_ids.append(edge.target)

        return [self._nodes[cid] for cid in chunk_ids if cid in self._nodes]

    def remove_file(self, file_id: str) -> int:
        """
        Remove a file and all its chunks from the graph.

        Returns:
            Number of nodes removed
        """
        self._ensure_loaded()
        removed = 0

        # Get chunks for this file
        chunks = self.get_chunks_for_file(file_id)

        # Remove chunks
        for chunk in chunks:
            if self.remove_node(chunk.id):
                removed += 1

        # Remove file node
        if self.remove_node(file_id):
            removed += 1

        return removed

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        self._ensure_loaded()

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "file_count": len(self._nodes_by_type.get(NodeType.FILE.value, set())),
            "chunk_count": len(self._nodes_by_type.get(NodeType.CHUNK.value, set())),
            "concept_count": len(self._nodes_by_type.get(NodeType.CONCEPT.value, set())),
            "edges_by_type": self._count_edges_by_type(),
        }

    def _count_edges_by_type(self) -> Dict[str, int]:
        """Count edges by type."""
        counts = {}
        for edge in self._edges:
            counts[edge.type] = counts.get(edge.type, 0) + 1
        return counts

    # -------------------------------------------------------------------------
    # Clear/Reset
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()
        self._concept_by_label.clear()
        for node_set in self._nodes_by_type.values():
            node_set.clear()

    def delete_graph_file(self) -> bool:
        """Delete the graph file from disk."""
        if self.graph_path.exists():
            try:
                self.graph_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to delete graph file: {e}")
        return False
