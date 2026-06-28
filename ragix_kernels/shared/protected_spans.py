"""
Protected-span codec — mask/restore non-translatable spans behind opaque tokens.

A single, tested implementation of the ``⟦P####⟧`` placeholder mechanism used to
shield content that must survive an LLM pass verbatim — fenced code, display and
inline math, inline code, Markdown links/images, bare URLs, HTML comments, and
author–year / numeric citations.

The codec was originally inlined in the EN→FR translation pipeline
(``segment.py`` masks, ``rebuild.py`` restores); it is promoted here so the
presenter (marp protection) and sealed (placeholder masking) subsystems can
share one audited implementation.

Contract::

    masked, mapping = protect(text)               # hide spans → opaque tokens
    restored, report = restore(masked, mapping)   # tokens → originals + diagnostics

Tokens are ``⟦P0001⟧`` … ``⟦P9999⟧``. :func:`protect` is deterministic and
order-stable (more specific patterns win); :func:`restore` is exact and reports
``hallucinated`` (token in text, no mapping entry) and ``dropped`` (mapping entry
never seen in the text) placeholders.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# Ordered: longer / more specific patterns run first so they win; each match is
# replaced atomically with a token and never re-scanned by later patterns.
PROTECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    # HTML comments — must run first: their *topic* is otherwise read as a prompt.
    ("comment_html",   re.compile(r"<!--[\s\S]*?-->")),
    # Fenced code blocks (```…``` or ~~~…~~~), multiline.
    ("code_fenced",    re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)),
    # Display math $$…$$ (multiline allowed).
    ("math_display",   re.compile(r"\$\$[\s\S]+?\$\$")),
    # Inline math $…$ (single line; avoids currency like "$5").
    ("math_inline",    re.compile(r"\$(?![\s\d])[^\$\n]+?\$(?!\d)")),
    # Inline code `…`.
    ("code_inline",    re.compile(r"`[^`\n]+?`")),
    # Markdown image ![alt](url).
    ("image",          re.compile(r"!\[[^\]]*\]\([^)]+\)")),
    # Markdown link [label](url) — whole expression protected.
    ("link",           re.compile(r"\[[^\]]+\]\([^)]+\)")),
    # Bare URLs.
    ("url",            re.compile(r"https?://\S+")),
    # Author–year citations: [Smith 2020], [Smith et al. 2020a], [Smith & Jones 2020].
    ("cite_authoryear",
     re.compile(r"\[[A-Z][A-Za-zÀ-ÿ\-]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-zÀ-ÿ\-]+))?\s+\d{4}[a-z]?\]")),
    # Numeric citations: [1], [12,34], [1-3].
    ("cite_numeric",   re.compile(r"\[\d+(?:\s*[-,]\s*\d+)*\]")),
]

#: Canonical rule names, in application (specificity) order.
RULE_NAMES: tuple[str, ...] = tuple(name for name, _ in PROTECTION_RULES)

#: Matches an emitted placeholder token.
TOKEN_RE = re.compile(r"⟦P\d{4}⟧")

_TOKEN_FMT = "⟦P{:04d}⟧"


class SpanCounter:
    """Monotonic ``⟦P####⟧`` token allocator.

    Pass one shared instance to successive :func:`protect` calls (e.g. one per
    document segment) to keep token names globally unique across calls — required
    when per-segment mappings are later unioned for a whole chapter/document, so
    one segment's placeholders never clobber another's.
    """

    __slots__ = ("counter",)

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def next_token(self) -> str:
        self.counter += 1
        return _TOKEN_FMT.format(self.counter)


def _select_rules(rules: Iterable[str] | None) -> list[tuple[str, re.Pattern[str]]]:
    if rules is None:
        return PROTECTION_RULES
    wanted = set(rules)
    unknown = wanted - set(RULE_NAMES)
    if unknown:
        raise ValueError(
            f"unknown protection rule(s): {sorted(unknown)}; "
            f"valid rules: {list(RULE_NAMES)}"
        )
    # Preserve canonical specificity order, not the caller's argument order.
    return [(n, p) for n, p in PROTECTION_RULES if n in wanted]


def protect(
    text: str,
    rules: Iterable[str] | None = None,
    *,
    counter: SpanCounter | None = None,
) -> tuple[str, dict[str, str]]:
    """Replace each protected span in *text* with a fresh ``⟦P####⟧`` token.

    Args:
        text: input to mask.
        rules: subset of :data:`RULE_NAMES` to apply (default: all, in canonical
            specificity order). Unknown names raise :class:`ValueError`.
        counter: optional shared :class:`SpanCounter` for cross-call token
            uniqueness; a fresh one (starting at 0) is used when omitted.

    Returns:
        ``(masked_text, mapping)`` where *mapping* holds only the tokens created
        in this call (``token -> original span``).
    """
    state = counter if counter is not None else SpanCounter()
    mapping: dict[str, str] = {}
    for _name, pat in _select_rules(rules):
        def _sub(m: "re.Match[str]") -> str:
            tok = state.next_token()
            mapping[tok] = m.group(0)
            return tok
        text = pat.sub(_sub, text)
    return text, mapping


@dataclass
class RestoreReport:
    """Diagnostics from :func:`restore`.

    Attributes:
        restored: number of placeholders successfully replaced.
        hallucinated: tokens present in the text but absent from *mapping*
            (cannot be restored — likely model-invented or corrupted).
        dropped: tokens present in *mapping* that never appeared in the text
            (the protected span was deleted somewhere upstream).
    """

    restored: int = 0
    hallucinated: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every placeholder round-tripped (none invented or dropped)."""
        return not self.hallucinated and not self.dropped


def restore(text: str, mapping: dict[str, str]) -> tuple[str, RestoreReport]:
    """Reinstate ``⟦P####⟧`` tokens in *text* from *mapping*.

    Returns ``(restored_text, report)``. Tokens in the text with no mapping entry
    are left verbatim and recorded in ``report.hallucinated``; mapping entries
    never seen in the text are recorded in ``report.dropped``.
    """
    report = RestoreReport()
    seen: set[str] = set()

    def _sub(m: "re.Match[str]") -> str:
        tok = m.group(0)
        seen.add(tok)
        if tok in mapping:
            report.restored += 1
            return mapping[tok]
        report.hallucinated.append(tok)
        return tok

    restored_text = TOKEN_RE.sub(_sub, text)
    report.dropped = [tok for tok in mapping if tok not in seen]
    return restored_text, report
