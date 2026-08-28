"""
saqqara.analyzers.sections — the names a document gives to its own parts.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.40-K3.48 of SPEC.md.

A section name is the most useful structural fact a document offers and the least reliably
recorded. Some documents declare an outline. Some style their headings. Some number them in the
text and style nothing at all. Some do all three, inconsistently, in the same file. So names are
collected per **channel** — one named route from a file feature to a candidate — and every
candidate, whatever route it arrived by, then runs the same gauntlet.

Two traps deserve naming, because both produce output that looks entirely correct.

A **running header** repeats on every page. It is title-shaped, short, and sits at the top, which
is exactly what a section looks like. Read as one it squats at the head of every ancestry in the
document, and every answer underneath is then filed under a piece of page furniture. It is caught
by recurrence: the same name on three or more distinct pages is furniture, whatever it looks like.

A **printed table of contents** is a page of title-shaped lines with page numbers attached. Read
as sections it yields a complete, plausible, entirely duplicated outline of a document you already
have. It is caught by its dot leaders.

The gauntlet is ordered and every rejection is counted under its reason. A rejection nobody counts
is indistinguishable from a document that had nothing to find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..model import Node, Tree
from .contract import Analyzer, AnalyzerResult
from .outline import parse_label

__all__ = [
    "CHANNELS",
    "DERIVED_CHANNELS",
    "READER_CHANNELS",
    "FURNITURE_PAGES",
    "GAUNTLET",
    "REJECTION_REASONS",
    "SectionRecord",
    "SectionsAnalyzer",
    "baseline_sections",
]

#: The eight channels fed by readers: each is a route from something a reader
#: observed to a candidate. Closed.
READER_CHANNELS = (
    "pdf-numbered-heading",
    "pdf-native-toc",
    "docx-heading-style",
    "docx-numbering-property",
    "xlsx-sheet-name",
    "xlsx-section-row",
    "xlsx-block-title",
    "pptx-slide-title",
)

#: Channels fed by an analyzer rather than a reader. Kept separate on purpose:
#: a promotion is an inference about the document, and filing it beside the
#: observations would let it pass for one.
DERIVED_CHANNELS = ("outline-promotion", "format-promotion", "caption-binding")

#: Everything a record may be routed by.
CHANNELS = READER_CHANNELS + DERIVED_CHANNELS

#: The ordered tests every candidate runs, whatever channel it arrived by.
GAUNTLET = ("G1-printed-contents", "G2-orphan-number", "G3-page-furniture")

#: Why a candidate was refused. Closed, and one reason per test.
REJECTION_REASONS = ("printed-contents", "orphan-number", "page-furniture")

#: A name on this many distinct pages is furniture, whatever it looks like.
FURNITURE_PAGES = 3

_DOT_LEADER = re.compile(r"\.{3,}\s*\d+\s*$")
_BARE_NUMBER = re.compile(r"^\s*(\d+)\s*[.)]?\s*$")


@dataclass
class SectionRecord:
    """One accepted section name, with the route that found it."""

    channel: str
    name: str
    node: Node
    depth: int = 1
    page: Any = None
    trace: dict[str, Any] = field(default_factory=dict)


def baseline_sections(tree: Tree) -> list[str]:
    """The naive reading: anything heading-shaped is a section, nothing filtered.

    A styled heading, a declared outline entry, or a line that begins with a
    label — taken at face value, with no gauntlet. It is wrong in both
    directions at once, which is what makes it a useful comparison: it swallows
    page furniture and printed contents pages whole, and it misses every heading
    a document numbered as a property rather than typing the number out.
    """
    out = []
    for node in tree.walk():
        if not node.text:
            continue
        style = (node.facts.get("style") or "") if node.facts else ""
        if node.kind == "heading" or style.startswith("Heading"):
            out.append(node.text)
        elif parse_label(node.text) is not None:
            out.append(node.text)
    return out


class SectionsAnalyzer(Analyzer):
    """Collect section names per channel, run the gauntlet, attach the ancestry."""

    name = "sections"
    version = "0.1.0"

    def run(self, tree: Tree) -> AnalyzerResult:
        fmt = tree.root.provenance.source_format
        candidates, blind = self._collect(tree, fmt)

        counts = {reason: 0 for reason in REJECTION_REASONS}
        rejected: list[dict[str, Any]] = []
        accepted = self._gauntlet(candidates, counts, rejected)
        self._attach_ancestry(tree, accepted)

        trace: dict[str, Any] = {
            "analyzer": self.name,
            "version": self.version,
            "format": fmt,
            "gauntlet": list(GAUNTLET),
            "channels": {
                channel: [r.name for r in accepted if r.channel == channel]
                for channel in CHANNELS
                if any(r.channel == channel for r in accepted)
            },
            "candidates": len(candidates),
            "accepted": len(accepted),
            "rejected": counts,
            "rejections": rejected,
            "blind": blind,
            "baseline": {
                "rule": "every styled heading is a section, nothing filtered",
                "found": len(baseline_sections(tree)),
                "ours": len(accepted),
            },
        }
        return AnalyzerResult(tree=tree, trace=trace)

    # -------------------------------------------------------------- channels

    def _collect(self, tree: Tree, fmt: str):
        blind: dict[str, Any] = {}
        if fmt == "pdf":
            candidates = self._pdf(tree)
        elif fmt == "docx":
            candidates = self._docx(tree)
        elif fmt == "xlsx":
            candidates = self._xlsx(tree)
        elif fmt == "pptx":
            candidates = self._pptx(tree)
        else:
            # No channel applies to this format. Reported, not silently empty.
            return [], {"reason": "no-channel-for-format", "format": fmt}

        # A second pass, after a promotion analyzer has run, picks the inferred
        # headings up under their own channel — never under a reader's.
        candidates = candidates + self._promotions(tree)

        if not candidates:
            blind = {"reason": "no-candidate-found", "format": fmt}
        return candidates, blind

    @staticmethod
    def _promotions(tree: Tree) -> list[SectionRecord]:
        """Headings an analyzer inferred, if one has run. Never a reader's channel."""
        from .outline import OUTLINE_CHANNEL

        return [
            SectionRecord(
                channel=OUTLINE_CHANNEL,
                name=node.text,
                node=node,
                depth=int(node.level or 1),
                page=getattr(node.provenance.leaf, "page", None),
                trace={"inferred_by": node.provenance.kernel},
            )
            for node in tree.walk()
            if node.facts.get("channel") == OUTLINE_CHANNEL and node.text
        ]

    def _pdf(self, tree: Tree) -> list[SectionRecord]:
        out: list[SectionRecord] = []
        for page in tree.root.children:
            if page.kind != "page":
                continue
            number = page.provenance.leaf.page

            for node in page.children:
                if node.kind == "heading" and node.text and not node.facts.get("channel"):
                    out.append(
                        SectionRecord(
                            channel="pdf-native-toc",
                            name=node.text,
                            node=node,
                            depth=int(node.facts.get("level") or 1),
                            page=number,
                            trace={"declared": True},
                        )
                    )

            lines = [n for n in page.children if n.kind == "paragraph" and n.text]
            index = 0
            while index < len(lines):
                node = lines[index]
                text = node.text.strip()
                joined = False

                bare = _BARE_NUMBER.match(text)
                if bare:
                    following = lines[index + 1] if index + 1 < len(lines) else None
                    if (
                        following is not None
                        and not _BARE_NUMBER.match(following.text.strip())
                        and not _DOT_LEADER.search(following.text)
                        and parse_label(following.text) is None
                    ):
                        # A number alone, its title on the line beneath: one
                        # heading typeset across two lines.
                        text = f"{bare.group(1)}. {following.text.strip()}"
                        joined = True
                    else:
                        out.append(
                            SectionRecord(
                                channel="pdf-numbered-heading", name=text, node=node,
                                page=number, trace={"orphan_number": True},
                            )
                        )
                        index += 1
                        continue

                if _DOT_LEADER.search(text):
                    out.append(
                        SectionRecord(
                            channel="pdf-numbered-heading", name=text, node=node,
                            page=number, trace={"dot_leader": True},
                        )
                    )
                    index += 1
                    continue

                parsed = parse_label(text)
                if parsed is not None:
                    out.append(
                        SectionRecord(
                            channel="pdf-numbered-heading",
                            name=parsed[2],
                            node=node,
                            depth=len(parsed[1]),
                            page=number,
                            trace={"label": list(parsed[1]), "joined_lines": joined},
                        )
                    )
                index += 2 if joined else 1
        return out

    @staticmethod
    def _docx(tree: Tree) -> list[SectionRecord]:
        out = []
        for node in tree.walk():
            if node.kind != "paragraph" or not node.text:
                continue
            style = node.facts.get("style") or ""
            if style.startswith("Heading"):
                out.append(
                    SectionRecord(
                        channel="docx-heading-style", name=node.text, node=node,
                        depth=int(style.split()[-1]) if style.split()[-1].isdigit() else 1,
                        trace={"style": style},
                    )
                )
            elif node.facts.get("numbered"):
                # Numbered by property: the file stores the words, the screen
                # shows the number. Text-only reading misses this entirely.
                out.append(
                    SectionRecord(
                        channel="docx-numbering-property", name=node.text, node=node,
                        depth=int(node.facts.get("outline_level") or 0) + 1,
                        trace={"numbering_property": True},
                    )
                )
        return out

    @staticmethod
    def _xlsx(tree: Tree) -> list[SectionRecord]:
        out = []
        for section in tree.root.children:
            if section.kind != "section":
                continue
            out.append(
                SectionRecord(
                    channel="xlsx-sheet-name", name=section.text, node=section,
                    page=section.provenance.leaf.sheet_index,
                    trace={"sheet": section.text},
                )
            )
            for block in section.children:
                header = block.facts.get("header")
                if not header or header.get("uncertain"):
                    continue
                title = header.get("title")
                if title:
                    out.append(
                        SectionRecord(
                            channel="xlsx-block-title", name=title[1], node=block, depth=2,
                            page=section.provenance.leaf.sheet_index,
                            trace={"range": title[0]},
                        )
                    )
                for row in header.get("section_rows") or []:
                    cell = next(
                        (
                            c for c in block.children
                            if c.kind == "cell" and c.provenance.leaf.row == row and c.text
                        ),
                        None,
                    )
                    if cell is not None:
                        out.append(
                            SectionRecord(
                                channel="xlsx-section-row", name=cell.text, node=cell, depth=3,
                                page=section.provenance.leaf.sheet_index,
                                trace={"row": row},
                            )
                        )
        return out

    @staticmethod
    def _pptx(tree: Tree) -> list[SectionRecord]:
        out = []
        for node in tree.walk():
            if node.kind == "shape" and node.facts.get("is_title") and node.text:
                out.append(
                    SectionRecord(
                        channel="pptx-slide-title", name=node.text, node=node,
                        page=node.provenance.leaf.slide, trace={"title_placeholder": True},
                    )
                )
        return out

    # -------------------------------------------------------------- gauntlet

    @staticmethod
    def _gauntlet(candidates, counts, rejected):
        """G1 printed contents · G2 orphan number · G3 page furniture."""
        pages_of: dict[str, set] = {}
        for candidate in candidates:
            pages_of.setdefault(candidate.name, set()).add(candidate.page)

        accepted = []
        for candidate in candidates:
            if candidate.trace.get("dot_leader"):
                counts["printed-contents"] += 1
                rejected.append({"test": "G1-printed-contents", "reason": "printed-contents",
                                 "name": candidate.name, "channel": candidate.channel})
                continue
            if candidate.trace.get("orphan_number"):
                counts["orphan-number"] += 1
                rejected.append({"test": "G2-orphan-number", "reason": "orphan-number",
                                 "name": candidate.name, "channel": candidate.channel})
                continue
            if len(pages_of[candidate.name]) >= FURNITURE_PAGES:
                counts["page-furniture"] += 1
                rejected.append({"test": "G3-page-furniture", "reason": "page-furniture",
                                 "name": candidate.name, "channel": candidate.channel,
                                 "pages": sorted(p for p in pages_of[candidate.name]
                                                 if p is not None)})
                continue
            accepted.append(candidate)
        return accepted

    # -------------------------------------------------------------- ancestry

    @staticmethod
    def _attach_ancestry(tree: Tree, accepted: list[SectionRecord]) -> None:
        """Every node gets the ordered chain of sections covering it, broadest first."""
        order = {id(node): index for index, node in enumerate(tree.walk())}
        marks = sorted(
            ((order[id(r.node)], r) for r in accepted if id(r.node) in order),
            key=lambda pair: pair[0],
        )

        stack: list[SectionRecord] = []
        cursor = 0
        for index, node in enumerate(tree.walk()):
            while cursor < len(marks) and marks[cursor][0] <= index:
                record = marks[cursor][1]
                while stack and stack[-1].depth >= record.depth:
                    stack.pop()
                stack.append(record)
                cursor += 1
            # A node above the first section carries an empty chain, not a guess.
            node.facts["section_ancestry"] = [r.name for r in stack]
