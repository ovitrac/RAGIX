"""
saqqara.analyzers — chained recognisers, each with its own decomposed trace.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K3 in SPEC.md. Implemented in P3; the section recogniser in P3'.

Each analyzer takes a tree and returns a tree plus a trace. They chain — the chain is called the
`pipeline` — and each one is testable alone.

  grid_tables   is this a declared table, or a page layout drawn with one
  outline       a typed label promotes only where the walk supports it
  format_headings  a line set larger than the body is a heading where a tier supports it
  sections      section names per channel, with the gauntlet and the ancestry
  tables        segmentation cascade: declared table objects, then bordered boxes, then connected
                regions, then typing
  header_bands  merged header bands, laminar-or-abstain; header chains read broadest first; an
                empty merged tile is an addressable rung
  islands       a block holding several disconnected value regions is reported, never
                re-segmented on its own authority
  chains        the column and row ancestry of one position, with the self-reference guard
  sections      section names and the instructions attached to them (P3')

An analyzer that cannot decide says so. The abstention travels with the object, carries a reason
from a closed vocabulary, and is never a default value.
"""

from .chains import BLANK_RUNG, SelfReferenceError, anchors  # noqa: F401
from .grid_tables import (  # noqa: F401
    DOCX_TYPES,
    GRID_TABLE_FORMATS,
    GridTablesAnalyzer,
    type_table,
)
from .contract import (  # noqa: F401
    ABSTENTION_REASONS,
    TYPING_REASONS,
    Abstention,
    Analyzer,
    AnalyzerResult,
)
from .format_headings import (  # noqa: F401
    FORMAT_ABSTENTIONS,
    FORMAT_CHANNEL,
    SHAPE_RULES,
    TIER_DROPS,
    FormatHeadingsAnalyzer,
    assemble_lines,
    baseline_format_headings,
    shape_refusal,
)
from .header_bands import HeaderBandsAnalyzer, analyze_block  # noqa: F401
from .islands import IslandsAnalyzer, find_islands  # noqa: F401
from .outline import MIN_CHAIN, OUTLINE_DROPS, OutlineAnalyzer, parse_label  # noqa: F401
from .sections import (  # noqa: F401
    CHANNELS,
    GAUNTLET,
    REJECTION_REASONS,
    SectionsAnalyzer,
    baseline_sections,
)
from .tables import TablesAnalyzer  # noqa: F401

#: The standard order. Segmentation first — nothing else can run without blocks;
#: then the header band, because chains read from it; islands last, because what
#: it reports is a disagreement with the segmentation that opened the chain.
#: `sections` runs last of the default chain because on a spreadsheet it reads
#: block titles and section rows that the header analysis produced.
#:
#: `grid_tables` sits second because a table a file DECLARES — in a document or
#: on a slide — has to be typed before the header rules can know whether to read
#: it at all. A spreadsheet needs no such pass: there a table is found, not declared.
PIPELINE = (
    TablesAnalyzer,
    GridTablesAnalyzer,
    HeaderBandsAnalyzer,
    IslandsAnalyzer,
    SectionsAnalyzer,
)

#: `outline` is OPT-IN and runs AFTER `sections`, never before.
#:
#: Before, it would turn every numbered line into a heading, and the eight
#: reader-fed channels would then be reading this package's own conclusions back
#: to themselves. It adds nodes rather than rewriting them, under the channel
#: `outline-promotion`, so an inference is never filed among the observations. A
#: caller who wants promotions inside the ancestry runs `sections` again after it;
#: both passes appear in the trace.
#:
#: `chains` is not a pass and is deliberately absent from this list. It is a
#: projection: it computes an ancestry for one position on demand and stores
#: nothing. Making it a pipeline stage would mean writing a derived value into the
#: tree, which is the one thing a view must not do.

__all__ = [
    "FORMAT_ABSTENTIONS",
    "FORMAT_CHANNEL",
    "FormatHeadingsAnalyzer",
    "SHAPE_RULES",
    "TIER_DROPS",
    "assemble_lines",
    "baseline_format_headings",
    "shape_refusal",
    "parse_label",
    "baseline_sections",
    "SectionsAnalyzer",
    "REJECTION_REASONS",
    "OutlineAnalyzer",
    "OUTLINE_DROPS",
    "MIN_CHAIN",
    "GAUNTLET",
    "CHANNELS",
    "ABSTENTION_REASONS",
    "Abstention",
    "Analyzer",
    "AnalyzerResult",
    "BLANK_RUNG",
    "DOCX_TYPES",
    "GridTablesAnalyzer",
    "HeaderBandsAnalyzer",
    "IslandsAnalyzer",
    "PIPELINE",
    "SelfReferenceError",
    "TYPING_REASONS",
    "TablesAnalyzer",
    "analyze_block",
    "type_table",
    "anchors",
    "find_islands",
    "pipeline",
]


def pipeline(tree):
    """Run the standard chain, keeping every analyzer's trace under its own name."""
    traces = {}
    for analyzer_class in PIPELINE:
        analyzer = analyzer_class()
        result = analyzer.run(tree)
        tree = result.tree
        traces[analyzer.name] = result.trace
    return tree, traces
