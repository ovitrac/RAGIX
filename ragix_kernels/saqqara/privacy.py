"""Generic stamp shapes and explicit, presentation-only masking policies.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass
import re
import json
import unicodedata
from .field_views import stable_id, TextView

WORD = r"[^\W\d_](?:[^\W\d_]|[\u0300-\u036f])*(?:[-’'][^\W\d_](?:[^\W\d_]|[\u0300-\u036f])*)*"
DATES = re.compile(
    rf"(?<!\w)(?:\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}|\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}|"
    rf"\d{{1,2}}[ -]+{WORD}\.?[ ,/-]+\d{{4}}|{WORD}\.?\s+\d{{1,2}},?\s+\d{{4}}|[12]\d{{7}})(?!\d)"
)
TIMES = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3])[:h][0-5]\d(?::[0-5]\d)?(?:\s*[AP]M)?(?!\d)", re.I)
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")
USER = re.compile(r"(?<!\w)(?:[A-Za-z][A-Za-z0-9]*[._-][A-Za-z0-9]+|[A-Za-z]{1,4}\d{3,})(?!\w)")
WORDS = re.compile(rf"(?<!\w){WORD}\.?(?!\w)")


def stamp_matches(text, *, edge=False, recurring=False):
    """Flag shapes, not identities. No name list and no semantic name claim."""
    dates = list(DATES.finditer(text))
    times = list(TIMES.finditer(text))
    if not dates and not ((edge or recurring) and times):
        return ()
    excluded = [(m.start(), m.end()) for m in (*dates, *times)]
    identities = []
    for kind, pattern in (("email", EMAIL), ("user_id", USER)):
        for m in pattern.finditer(text):
            if not any(m.start() < b and m.end() > a for a, b in excluded):
                identities.append({"start": m.start(), "end": m.end(), "raw": m[0], "kind": kind})
    words = [
        m
        for m in WORDS.finditer(text)
        if m[0][0].isupper()
        and not any(m.start() < b and m.end() > a for a, b in excluded)
        and not any(m.start() < r["end"] and m.end() > r["start"] for r in identities)
    ]
    groups = []
    for word in words:
        if groups and re.fullmatch(r"[\s,]+", text[groups[-1][-1].end() : word.start()]):
            groups[-1].append(word)
        else:
            groups.append([word])
    for group in groups:
        if len(group) < 2:
            continue
        start, end = group[0].start(), group[-1].end()
        identities.append(
            {"start": start, "end": end, "raw": text[start:end], "kind": "name_shape"}
        )
    return tuple(sorted(identities, key=lambda r: (r["start"], r["end"])))


@dataclass(frozen=True)
class TableStampLine:
    table_id: str
    view: TextView
    matches: tuple[dict, ...]

    def __post_init__(self):
        if not self.table_id or not self.matches:
            raise ValueError("table stamp requires observed matches")
        for match in self.matches:
            if (
                set(match) != {"start", "end", "raw", "kind"}
                or match["kind"] not in {"email", "user_id", "name_shape"}
                or self.view.text[match["start"] : match["end"]] != match["raw"]
            ):
                raise ValueError("table stamp match must retain exact source text")
            self.view.source_refs(match["start"], match["end"])

    def as_record(self):
        view = self.view
        return {
            "candidate_id": stable_id("table-stamp/1", self.table_id, view.view_id, self.matches),
            "line_id": view.view_id,
            "source_id": view.source_id,
            "page": view.page,
            "text": view.text,
            "bbox": view.bbox,
            "personal_data_suspected": True,
            "matches": list(self.matches),
            "table_id": self.table_id,
            "mapping": [asdict(ref) if ref is not None else None for ref in view.mapping],
            "flags": view.flags,
        }


def table_row_stamps(document):
    """Join only cells in one observed row; never associate separate rows."""
    from .field_views import TextSpan, assemble

    stamps = []
    for page in document.pages:
        for table in page.tables:
            for row in table.cell_rows:
                cells = tuple(sorted((c for c in row if c.text), key=lambda c: c.bbox[0]))
                if len(cells) < 2 or any(stamp_matches(c.text) for c in cells):
                    continue  # Single-cell stamps already use the line path.
                view = assemble(
                    tuple(
                        TextSpan(
                            document.source_id, c.cell_id, page.page, c.text, c.bbox, flags=c.flags
                        )
                        for c in cells
                    )
                )
                matches = stamp_matches(view.text)
                if matches:
                    stamps.append(TableStampLine(table.table_id, view, matches))
    return tuple(stamps)


def table_stamp_from_dict(data):
    from .field_views import view_from_dict

    return TableStampLine(data["table_id"], view_from_dict(data["view"]), tuple(data["matches"]))


@dataclass(frozen=True)
class MaskPolicy:
    unmask: bool = False
    reason: str = ""
    extra_literals: tuple[str, ...] = ()
    extra_patterns: tuple[str, ...] = ()
    mask_token: str = "[MASKED]"
    version: str = "presentation-mask/0.1"

    def __post_init__(self):
        if type(self.unmask) is not bool or not isinstance(self.reason, str):
            raise ValueError("typed mask policy required")
        if self.unmask and not self.reason.strip():
            raise ValueError("unmasking requires an explicit recorded reason")
        if self.mask_token != "[MASKED]" or self.version != "presentation-mask/0.1":
            raise ValueError("invalid mask policy")
        if any(not isinstance(s, str) or not s for s in self.extra_literals):
            raise ValueError("nonempty mask literals required")
        for pattern in self.extra_patterns:
            if re.compile(pattern).search(""):
                raise ValueError("empty-match mask pattern forbidden")

    @property
    def digest(self):
        return stable_id(asdict(self))


DEFAULT_MASK_POLICY = MaskPolicy()


def profile_stamps(profile):
    fields = profile.fields if hasattr(profile, "fields") else profile["fields"]
    entry = fields["furniture"]
    value = entry.value if hasattr(entry, "value") else entry["value"]
    return tuple((value or {}).get("stamp_lines", ()))


class PresentationMasker:
    """Scrub copied literals too, including fragments in nested presentation data.

    Source observations are never edited. Hashes/coordinates remain addressable.
    Name tokens are scrubbed separately so a word-level view cannot bypass a mask.
    """

    def __init__(self, stamps, policy=None):
        if policy is not None and not isinstance(policy, MaskPolicy):
            raise TypeError("explicit MaskPolicy required")
        self.policy = policy or DEFAULT_MASK_POLICY
        self.stamps = tuple(stamps)
        terms = set(self.policy.extra_literals)
        for stamp in self.stamps:
            for match in stamp["matches"]:
                terms.add(match["raw"])
                terms.add(re.sub(r"\d+", "#", match["raw"]))
                if match["kind"] == "name_shape":
                    terms.update(m[0] for m in WORDS.finditer(match["raw"]))
        terms |= {unicodedata.normalize(form, t) for t in tuple(terms) for form in ("NFC", "NFD")}
        self.patterns = tuple(
            re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.I)
            for t in sorted(terms, key=lambda t: (-len(t), t))
        )
        self.patterns += tuple(re.compile(p) for p in self.policy.extra_patterns)

    def text(self, value):
        if self.policy.unmask:
            return value
        # Apply substitutions to original spans together; a replacement token
        # must not itself be matched by a later private policy.
        spans = sorted(
            (m.start(), m.end())
            for p in self.patterns
            for m in p.finditer(value)
            if m.end() > m.start()
        )
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        output = []
        position = 0
        for start, end in merged:
            output.extend((value[position:start], self.policy.mask_token))
            position = end
        return "".join((*output, value[position:]))

    def data(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, (list, tuple)):
            return [self.data(v) for v in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                masked_key = self.text(key)
                if masked_key in result:
                    raise ValueError("masked presentation keys collide")
                result[masked_key] = self.data(item)
            return result
        return value

    def provenance(self, lines):
        count = sum(
            self.text(line["text"]) != line["text"]
            for line in {r["line_id"]: r for r in lines}.values()
        )
        return {
            "masked_lines": count,
            "policy_digest": self.policy.digest,
            "masking_enabled": not self.policy.unmask,
            "unmask_reason": self.policy.reason if self.policy.unmask else None,
        }


def census_stamps(census):
    """The presentation guard reads upstream census facts, not editable reviews."""
    legacy = tuple(
        sorted(
            (
                {
                    "candidate_id": r.candidate_id,
                    "line_id": e.span_id,
                    "source_id": e.source_id,
                    "page": e.page,
                    "text": e.literal,
                    "bbox": e.bbox,
                    "personal_data_suspected": True,
                    "matches": json.loads(dict(r.attributes)["matched_spans"]),
                }
                for r in census.records
                if r.category == "stamp"
                for e in r.evidence
            ),
            key=lambda r: (r["page"], r["line_id"]),
        )
    )

    return tuple(
        sorted(
            (*legacy, *(line.as_record() for line in census.table_analysis.stamp_lines)),
            key=lambda record: (record["page"], record["line_id"]),
        )
    )
