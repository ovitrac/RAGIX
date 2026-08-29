"""
saqqara.adapters.md — the markdown reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K2.16 of SPEC.md.

Front matter is metadata, not prose. Reading it as a paragraph is the classic way to poison a
document's text with its own bookkeeping — a title, a date and a status arriving as the first
sentence of the body. It is emitted as a metadata observation instead, and the body starts where
the body starts.

Every block records the line it begins on. A citation into a text file that cannot name a line is
not a citation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..model import MdLocator
from .contract import Adapter, Mastaba, OpenVocabulary, register_adapter

FENCE = "---"

#: The declared vocabularies, one per record kind this reader emits (K2.19).
#:
#: A paragraph carries no facts at all — declared as the empty set, because a
#: kind emitted without a declaration is a failure and silence is not one.
#: Metadata is the one open vocabulary in this kernel: its fact names are the
#: front-matter keys the author wrote, so a closed set here would pin the
#: fixture rather than the reader (K2.21). `_unparsed` is the reader's own,
#: reserved for the front-matter lines that are not `key: value`.
HEADING_FACTS = ("level",)
PARAGRAPH_FACTS: tuple[str, ...] = ()
METADATA_FACTS = OpenVocabulary(reserved=("_unparsed",))


class MarkdownAdapter(Adapter):
    """Read a markdown file into metadata, heading and paragraph observations."""

    format = "md"
    # 0.2.0 declares a vocabulary per record kind, and drops `kind_hint`: it was
    # declared here and emitted by nothing, on any file — a name the pin had been
    # comparing only with itself.
    version = "0.2.0"
    extensions = (".md", ".markdown")
    fact_sets = {
        "metadata": METADATA_FACTS,
        "heading": HEADING_FACTS,
        "paragraph": PARAGRAPH_FACTS,
    }

    def read(self, path: Path) -> Iterator[Mastaba]:
        lines = path.read_text(encoding="utf-8").splitlines()
        start = 0

        if lines and lines[0].strip() == FENCE:
            end = next(
                (i for i in range(1, len(lines)) if lines[i].strip() == FENCE), None
            )
            if end is not None:
                yield Mastaba(
                    kind="metadata",
                    locator=MdLocator(line=1),
                    facts=self._front_matter(lines[1:end]),
                )
                start = end + 1

        block: list[str] = []
        block_line = start + 1

        for offset in range(start, len(lines)):
            line = lines[offset]
            stripped = line.strip()

            if stripped.startswith("#"):
                yield from self._flush(block, block_line)
                block = []
                level = len(stripped) - len(stripped.lstrip("#"))
                yield Mastaba(
                    kind="heading",
                    locator=MdLocator(line=offset + 1),
                    text=stripped[level:].strip(),
                    facts={"level": level},
                )
                block_line = offset + 2
                continue

            if not stripped:
                yield from self._flush(block, block_line)
                block = []
                block_line = offset + 2
                continue

            if not block:
                block_line = offset + 1
            block.append(line)

        yield from self._flush(block, block_line)

    @staticmethod
    def _flush(block: list[str], line: int) -> Iterator[Mastaba]:
        if block:
            yield Mastaba(
                kind="paragraph",
                locator=MdLocator(line=line),
                text="\n".join(block).strip(),
            )

    @staticmethod
    def _front_matter(lines: list[str]) -> dict:
        """Read `key: value` pairs. A line that is not a pair is kept, not dropped."""
        facts: dict[str, str] = {}
        unparsed: list[str] = []
        for line in lines:
            if ":" in line:
                key, _, value = line.partition(":")
                facts[key.strip()] = value.strip()
            elif line.strip():
                unparsed.append(line.strip())
        if unparsed:
            facts["_unparsed"] = unparsed
        return facts


register_adapter(MarkdownAdapter())
