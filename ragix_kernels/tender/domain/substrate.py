"""tender.substrate — the lab's one way to obtain a saqqara tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The public kernel offers readers that return raw observations and a builder that
turns observations into a tree; obtaining an *analyzed* tree is three calls plus
two opt-ins, and the opt-ins are the part that is easy to get wrong:

  - the **object store** is opt-in at the call site, because a reader given no
    store reads no images: it has nowhere to put them (K6.2 forbids putting the
    bytes in the tree). A caller that wants only structure pays nothing for
    pictures it will not read, and a caller that wants figures and forgets the
    store gets a tree with none, silently;
  - the **promoting analyzers** (outline, format headings) stay out of the
    default pipeline for the reasons their own modules give. The old substrate's
    structure kernel promotes, so any reading meant to be comparable with it has
    to ask for them. The vector-region lane is *not* offered here: it needs the
    same store the reader was given plus a render policy, and that pair of
    decisions belongs to the caller that has them — today, the parity harness.

Eight lab instruments need the same tree. Written out eight times, the two
opt-ins are eight places to forget one, and the one that drifts is whichever
nobody is watching — so the idiom lives here once. The parity harness is the
deliberate exception: it states its calls in full, because it is the instrument
that measures the difference between the two substrates and must not read one
of them through a convenience.

What the builder refused to place is returned with the tree rather than
discarded. A drop nobody reads is a silent drop.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["TreeRead", "read_tree"]


@dataclass
class TreeRead:
    """An analyzed tree, and the account of what did not reach it."""

    tree: Any
    source_format: str
    reader_version: str
    drops: Counter = field(default_factory=Counter)
    promoted: bool = False

    @property
    def dropped(self) -> int:
        """Total observations the builder refused to place."""
        return sum(self.drops.values())


def read_tree(
    path: Path | str,
    *,
    store: Any = None,
    promote: bool = False,
) -> TreeRead:
    """Read one file into an analyzed saqqara tree.

    Args:
        path: the file to read; the reader is chosen by suffix and raises
            `UnsupportedFormat` when no reader claims it (fail closed).
        store: an `AssetStore` if the caller wants objects read; `None` means
            structure only, and no figure is reported as missing because none
            was ever asked for.
        promote: run the opt-in promoting analyzers (outline, format headings).
            Required for any reading meant to be comparable with the old
            substrate's structure kernel.

    Returns:
        A `TreeRead` carrying the tree, the format and reader version that
        produced it, and the builder's drops by (reason, kind).
    """
    from ragix_kernels.saqqara.adapters import adapter_for, read_path
    from ragix_kernels.saqqara.analyzers import pipeline
    from ragix_kernels.saqqara.builder import build_tree

    path = Path(path)
    adapter = adapter_for(path)
    if adapter is None:
        from ragix_kernels.saqqara.adapters import UnsupportedFormat

        raise UnsupportedFormat(f"no reader claims {path.suffix!r}: {path.name}")

    built = build_tree(
        read_path(path, store=store),
        str(path),
        adapter.format,
        adapter.format,
        adapter.version,
    )

    drops: Counter = Counter()
    for drop in built.trace.get("drops", ()):
        drops[(drop.get("reason", "?"), drop.get("kind", "?"))] += 1

    tree = pipeline(built.tree)[0]

    if promote:
        from ragix_kernels.saqqara.analyzers.format_headings import FormatHeadingsAnalyzer
        from ragix_kernels.saqqara.analyzers.outline import OutlineAnalyzer

        tree = OutlineAnalyzer().run(tree).tree
        tree = FormatHeadingsAnalyzer().run(tree).tree

    return TreeRead(
        tree=tree,
        source_format=adapter.format,
        reader_version=adapter.version,
        drops=drops,
        promoted=promote,
    )
