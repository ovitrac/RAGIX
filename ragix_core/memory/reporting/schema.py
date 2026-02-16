"""
Config schema — dataclasses for scenario configuration sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryItem:
    """A single audit query."""
    question: str = ""
    query: str = ""
    type_filter: Optional[str] = None
    tags: Optional[str] = None
    k: Optional[int] = None
    read_full: bool = True


@dataclass
class CrossrefFacet:
    label: str = ""
    query: str = ""
    type_filter: Optional[str] = None
    read_full: bool = True


@dataclass
class CrossrefChain:
    name: str = ""
    facets: List[CrossrefFacet] = field(default_factory=list)


@dataclass
class RecallRun:
    query: str = ""
    budget_tokens: int = 1500


@dataclass
class LinkSpec:
    src_pattern: str = ""
    dst_pattern: str = ""
    relation: str = "supports"


@dataclass
class ThresholdsConfig:
    min_items_total: int = 1
    max_avg_search_ms: float = 50.0
    max_avg_recall_ms: float = 200.0
    format_version: int = 1


def parse_query_items(raw: List[Dict[str, Any]]) -> List[QueryItem]:
    """Parse query items from YAML config."""
    return [QueryItem(**{k: v for k, v in item.items()}) for item in raw]


def parse_crossref_chains(raw: List[Dict[str, Any]]) -> List[CrossrefChain]:
    """Parse crossref chains from YAML config."""
    chains = []
    for chain_data in raw:
        facets = [
            CrossrefFacet(**f) for f in chain_data.get("facets", [])
        ]
        chains.append(CrossrefChain(name=chain_data.get("name", ""), facets=facets))
    return chains


def parse_recall_runs(raw: List[Dict[str, Any]]) -> List[RecallRun]:
    return [RecallRun(**r) for r in raw]


def parse_link_specs(raw: List[Dict[str, Any]]) -> List[LinkSpec]:
    return [LinkSpec(**ls) for ls in raw]
