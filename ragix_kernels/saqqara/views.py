"""
saqqara.views — lazy projections over the tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K1.5-K1.8 (structure signature) and K3 (anchors, chains) in SPEC.md.
The signature lands in P1; anchors and chains in P3.

The tree is the source. A view computes what a consumer needs and never becomes a second source of
truth. A projection that loses provenance is a defect, not a trade-off.

The structure signature answers one question — *what shape is this document?* — in a form that can
be compared, stored and hashed. It is built to satisfy three constraints that pull against each
other:

  it must not depend on serialisation (K1.5), so it is computed from the tree, never from text;
  it must not depend on where the file sits (K1.7), so the comparable part excludes every path;
  it must never disagree with the tree it describes (K1.6), so the totals are derived from one
  walk and reconciled rather than accumulated in separate passes that can drift apart.

`flat_mass_ratio` is the one number here that carries judgement rather than counting. It measures
how much of a document's text sits under no heading at all. A well-structured report approaches
zero; a wall of paragraphs with a title on top approaches one. It separates documents whose
structure can be trusted from documents where it must be inferred (K1.8).
"""

from __future__ import annotations

from typing import Any, Iterator

from .model import Node, Tree

__all__ = ["OUTLINE_CAP", "STRUCTURING_KINDS", "signature_core", "structure_signature"]

#: Headings kept in the outline. The total is always reported alongside, so the
#: cap can never be mistaken for the count (K1.6).
OUTLINE_CAP = 50

#: Kinds that make their descendants structured rather than flat.
STRUCTURING_KINDS = frozenset({"heading", "section"})


def _walk_with_structure(node: Node, structured: bool = False) -> Iterator[tuple[Node, bool]]:
    """Every node, paired with whether a structuring ancestor covers it."""
    yield node, structured
    below = structured or node.kind in STRUCTURING_KINDS
    for child in node.children:
        yield from _walk_with_structure(child, below)


def signature_core(tree: Tree) -> dict[str, Any]:
    """The shape of a document, independent of where it is stored.

    Everything here comes from a single walk, so the totals cannot drift apart
    from each other or from the tree.
    """
    by_kind: dict[str, int] = {}
    headings_by_level: dict[str, int] = {}
    outline: list[str] = []
    outline_total = 0
    nodes = 0
    flat_chars = 0
    total_chars = 0

    for node, structured in _walk_with_structure(tree.root):
        nodes += 1
        by_kind[node.kind] = by_kind.get(node.kind, 0) + 1

        if node.kind == "heading":
            level = str(node.level if node.level is not None else 0)
            headings_by_level[level] = headings_by_level.get(level, 0) + 1
            outline_total += 1
            if len(outline) < OUTLINE_CAP:
                outline.append(node.text or "")

        if node.text:
            total_chars += len(node.text)
            if not structured and node.kind not in STRUCTURING_KINDS:
                flat_chars += len(node.text)

    return {
        "nodes": nodes,
        "by_kind": dict(sorted(by_kind.items())),
        "kinds": sorted(by_kind),
        "headings_by_level": dict(sorted(headings_by_level.items())),
        "outline_total": outline_total,
        "outline": outline,
        "outline_capped": outline_total > len(outline),
        "text_chars": total_chars,
        "flat_mass_ratio": round(flat_chars / total_chars, 6) if total_chars else 0.0,
    }


def structure_signature(tree: Tree) -> dict[str, Any]:
    """The core shape, plus the source facts that are deliberately outside it.

    Splitting the two is what lets K1.7 be a real claim: `core` is what two
    copies of one document share, `source` is what tells them apart.
    """
    root_prov = tree.root.provenance
    return {
        "core": signature_core(tree),
        "source": {
            "path": root_prov.source_path,
            "format": root_prov.source_format,
            "sha256": root_prov.source_sha256,
            "kernel": root_prov.kernel,
            "kernel_version": root_prov.kernel_version,
        },
    }
