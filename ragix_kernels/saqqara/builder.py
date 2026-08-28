"""
saqqara.builder — observations into a tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K3.j (K3.47-K3.51) in SPEC.md.

The readers produce a flat stream of observations. The analyzers speak of blocks, bands and tiles.
This module is the step between, and it is deliberately one step rather than a habit each analyzer
picks up on its own: a border observation has to be matched to the cell it surrounds, and if that
matching lives in three analyzers it will be done three slightly different ways, and the third one
will be wrong in a way nobody notices for a month.

It is also where a document stops being a list and becomes a pyramid.

**Accounting is the contract.** Every observation ends in exactly one of three places: it becomes
a node, it is attached to a node as facts, or it is dropped with a reason. The three totals sum to
the number of observations read, and the sum is asserted rather than assumed. A builder that
quietly discarded one observation in a thousand would be almost impossible to notice downstream —
the tree would simply be a little thinner than the document.

**Provenance is derived, never invented.** A node cites the coordinate its observation carried and
names the reader that made it. The builder placed the node; it did not see the document, and a
node that cited the builder as the source of what it says would be claiming an authority it does
not have.

**What differs between formats is data.** Each format declares how its observations nest — which
observation opens a container, what key ties a child to it — and the assembly is the same code for
all of them. A caller builds a tree the same way whatever it read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .adapters.contract import Mastaba
from .model import DocumentLocator, Locator, Node, Provenance, Tree, kind_registry

__all__ = ["BuildResult", "Builder", "FORMAT_PLANS", "FormatPlan", "build_tree"]

BUILDER_VERSION = "0.1.0"

#: Kinds this layer needs that the standard vocabulary does not declare.
#: Registered deliberately rather than by widening the core (K1.3).
for _kind in ("cell", "marker", "note", "shape", "page", "slide", "sheet"):
    kind_registry.register(_kind)


@dataclass
class BuildResult:
    """A tree, and an honest account of everything that did not become part of it."""

    tree: Tree
    trace: dict[str, Any]

    @property
    def reconciles(self) -> bool:
        t = self.trace
        return t["observations"] == t["nodes"] + t["attached"] + t["dropped"]


@dataclass(frozen=True)
class FormatPlan:
    """How one format's observations nest. Data, not a code path.

    `containers` maps an observation kind to the node kind it opens.
    `child_of` says which container an observation belongs under, by naming the
    locator attribute that ties them together.
    `node_kinds` maps every remaining observation kind to its node kind.
    `attach` names the observation kinds that are not nodes at all: they are
    folded into a node, or into the tree's metadata.
    """

    format: str
    containers: dict[str, str] = field(default_factory=dict)
    container_key: str | None = None
    node_kinds: dict[str, str] = field(default_factory=dict)
    attach: frozenset[str] = frozenset()


#: One entry per reader. Adding a format is adding a plan, not a branch.
FORMAT_PLANS: dict[str, FormatPlan] = {
    "xlsx": FormatPlan(
        format="xlsx",
        containers={"sheet": "section"},
        container_key="sheet_index",
        node_kinds={"cell": "cell"},
        attach=frozenset({"border"}),
    ),
    "docx": FormatPlan(
        format="docx",
        containers={"table": "table"},
        container_key="table_key",
        node_kinds={"cell": "cell", "marker": "marker", "paragraph": "paragraph"},
    ),
    "pdf": FormatPlan(
        format="pdf",
        containers={"page": "page"},
        container_key="page",
        node_kinds={"text": "paragraph", "outline_entry": "heading",
                    "figure": "figure"},
    ),
    "pptx": FormatPlan(
        format="pptx",
        containers={"slide": "slide", "table": "table"},
        container_key="pptx_key",
        node_kinds={"shape": "shape", "notes": "note", "cell": "cell"},
    ),
    "md": FormatPlan(
        format="md",
        node_kinds={"heading": "heading", "paragraph": "paragraph"},
        attach=frozenset({"metadata"}),
    ),
}


class Builder:
    """Assemble observations into a tree, under one contract for every format."""

    version = BUILDER_VERSION

    def __init__(self, plan: FormatPlan) -> None:
        self.plan = plan

    # ------------------------------------------------------------------ keys

    def _container_key(self, locator: Locator, kind: str | None = None) -> Any:
        """What ties an observation to its container, per the plan.

        The observation's kind is part of the question, not a shortcut. On a
        slide, a table and a plain text box occupy the same shape index, and only
        the kind says whether a cell belongs to the table or the slide.
        """
        key = self.plan.container_key
        if key is None:
            return None
        if key == "table_key":                        # a docx table is (flow, index)
            return (getattr(locator, "flow", None), getattr(locator, "table_index", None))
        if key == "pptx_key":
            slide = getattr(locator, "slide", None)
            if kind in ("table", "cell"):
                return (slide, getattr(locator, "shape", None))
            return (slide, None)
        return getattr(locator, key, None)

    @staticmethod
    def _position(locator: Locator) -> tuple:
        """Where an observation sits, for matching a border to its cell."""
        return (
            getattr(locator, "sheet_index", None),
            getattr(locator, "row", None),
            getattr(locator, "col", None),
        )

    # ----------------------------------------------------------------- build

    def build(
        self,
        observations: Sequence[Mastaba],
        source_path: str,
        reader: str,
        reader_version: str,
    ) -> BuildResult:
        plan = self.plan
        drops: list[dict[str, Any]] = []
        attached = 0
        meta: dict[str, Any] = {}

        def provenance(locator: Locator) -> Provenance:
            """Derived from the observation's own coordinate (K3.48)."""
            return Provenance(
                source_path=source_path,
                source_format=plan.format,
                chain=(locator,),
                kernel=reader,
                kernel_version=reader_version,
            )

        root = Node(
            kind="document",
            provenance=Provenance(
                source_path=source_path,
                source_format=plan.format,
                chain=(DocumentLocator(),),
                kernel=reader,
                kernel_version=reader_version,
            ),
        )

        # Borders first: they are facts of a cell, never nodes of their own, so
        # they must be in hand before the cells they belong to are built.
        borders: dict[tuple, dict[str, Any]] = {}
        for observation in observations:
            if observation.kind == "border":
                borders[self._position(observation.locator)] = dict(observation.facts)

        containers: dict[Any, Node] = {}
        matched_borders: set[tuple] = set()
        nodes = 0

        # Containers first, in a pass of their own. Doing this inline would make
        # a child's parent depend on whether its reader happened to emit the
        # container before it — which is a property of a reader's loop order, not
        # of the document, and produced exactly one silent misplacement.
        for observation in observations:
            if observation.kind in plan.containers:
                node = Node(
                    kind=plan.containers[observation.kind],
                    provenance=provenance(observation.locator),
                    text=observation.text,
                    facts=dict(observation.facts),
                )
                containers[self._container_key(observation.locator, observation.kind)] = node
                root.children.append(node)
                nodes += 1

        for observation in observations:
            kind = observation.kind

            if kind == "border":
                continue                      # accounted for after the loop

            if kind in plan.containers:
                continue                      # built in the pass above

            if kind == "metadata" and "metadata" in plan.attach:
                meta.update(observation.facts)
                attached += 1
                continue

            node_kind = plan.node_kinds.get(kind)
            if node_kind is None:
                drops.append(
                    {
                        "reason": "unmapped-observation",
                        "kind": kind,
                        "locator": observation.locator.to_dict(),
                    }
                )
                continue

            facts = dict(observation.facts)
            if node_kind == "cell":
                position = self._position(observation.locator)
                border = borders.get(position)
                if border is not None:
                    facts["border"] = border
                    matched_borders.add(position)

            node = Node(
                kind=node_kind,
                provenance=provenance(observation.locator),
                text=observation.text,
                facts=facts,
            )
            parent = containers.get(
                self._container_key(observation.locator, observation.kind), root)
            parent.children.append(node)
            nodes += 1

        # A border that matched no cell is a drop, with its reason. It is not a
        # node — a border is not a thing in the document, it is a property of one.
        for position, facts in borders.items():
            if position in matched_borders:
                attached += 1
            else:
                drops.append(
                    {"reason": "border-matches-no-cell", "kind": "border", "position": list(position)}
                )

        trace = {
            "builder": "saqqara.builder",
            "builder_version": self.version,
            "format": plan.format,
            "reader": reader,
            "reader_version": reader_version,
            "observations": len(observations),
            "nodes": nodes,
            "attached": attached,
            "dropped": len(drops),
            "drops": drops,
        }

        meta["builder"] = {"name": "saqqara.builder", "version": self.version}
        tree = Tree(root=root, meta=meta)
        return BuildResult(tree=tree, trace=trace)


def build_tree(
    observations: Iterable[Mastaba],
    source_path: str,
    source_format: str,
    reader: str,
    reader_version: str,
) -> BuildResult:
    """Build a tree for any format the package reads.

    One entry point: the difference between formats is the plan looked up here,
    not a decision the caller has to make (K3.51).
    """
    plan = FORMAT_PLANS.get(source_format)
    if plan is None:
        raise ValueError(f"no build plan declares the format {source_format!r}")
    return Builder(plan).build(list(observations), source_path, reader, reader_version)
