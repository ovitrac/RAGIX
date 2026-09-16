"""Versioned evidence-backed notation profiles and append-only reviews.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace
import json
import re
import math
from .census import Census
from .field_views import stable_id
from ..harvest.report import replay_digest

VERSION = "document-profile/0.1"
FIELDS = (
    "language",
    "identifier_families",
    "numbering_style",
    "reference_fields",
    "id_row_tables",
    "furniture",
    "numeric_locale",
    "reading_coverage",
)
STATUSES = frozenset({"PROBED", "AMENDED", "CONFIRMED", "UNKNOWN"})


@dataclass(frozen=True)
class ProfileField:
    value: object
    confidence: float
    evidence: tuple[str, ...]
    rule_id: str
    status: str

    def __post_init__(self):
        if (
            self.status not in STATUSES
            or isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
            or not self.rule_id
        ):
            raise ValueError("invalid profile field")
        if self.status == "UNKNOWN":
            if self.value is not None or self.confidence != 0:
                raise ValueError("UNKNOWN has no value/confidence")
        elif self.value is None or not self.evidence:
            raise ValueError("supported field needs evidence")


@dataclass(frozen=True)
class ProfileReview:
    review_id: str
    source_id: str
    field: str
    replacement: ProfileField
    reviewer: str
    reason: str
    previous_review_id: str | None = None

    def __post_init__(self):
        if (
            not self.review_id
            or not self.source_id
            or self.field not in FIELDS
            or not self.reviewer
            or not self.reason
            or self.replacement.status not in {"AMENDED", "CONFIRMED", "UNKNOWN"}
        ):
            raise ValueError("invalid explicit review")


@dataclass(frozen=True)
class DocumentProfile:
    source_id: str
    census_id: str
    fields: dict[str, ProfileField]
    template_signature: str
    reviews: tuple[ProfileReview, ...] = ()
    version: str = VERSION

    def __post_init__(self):
        if (
            self.version != VERSION
            or set(self.fields) != set(FIELDS)
            or not self.source_id
            or not self.census_id
        ):
            raise ValueError("closed profile schema/version required")
        required = {
            "numbering_style": {
                "markers",
                "depths",
                "range_connectors",
                "list_connectors",
                "unknown_connectors",
            },
            "reference_fields": {
                "labels",
                "stop_labels",
                "continuation",
                "role_line",
                "max_gap_ratio",
            },
            "id_row_tables": {"tables"},
            "furniture": {
                "running_patterns",
                "mark_literals",
                "default",
                "edge_fraction",
                "large_points",
                "rotation_threshold",
            },
            "numeric_locale": {"decimal_separator", "grouping", "units", "comparators"},
            "reading_coverage": {"pages", "text_layer_pages"},
        }
        for name, entry in self.fields.items():
            if not isinstance(entry, ProfileField):
                raise ValueError("typed profile fields required")
            if entry.status == "UNKNOWN":
                continue
            value = entry.value
            if name in required and (not isinstance(value, dict) or set(value) != required[name]):
                raise ValueError("closed field value schema: " + name)
            if name == "language" and value not in {"fr", "en"}:
                raise ValueError("language schema")
            if name == "identifier_families":
                if not isinstance(value, list) or any(
                    not isinstance(f, dict) or set(f) != {"family_id", "shape", "revision_markers"}
                    for f in value
                ):
                    raise ValueError("family schema")
            if name == "numeric_locale" and value["decimal_separator"] not in {".", ","}:
                raise ValueError("locale schema")
            if name == "reference_fields" and (
                value["continuation"] != "same_page_column_until_boundary"
                or value["role_line"] != "undecidable"
            ):
                raise ValueError("unsupported field policy")
        if len({r.review_id for r in self.reviews}) != len(self.reviews):
            raise ValueError("duplicate profile review")


@dataclass(frozen=True)
class ProfileConfig:
    recurrence_fraction: float = 0.5
    reference_fraction: float = 0.8
    minimum_occurrences: int = 2
    continuation_gap_ratio: float = 2.5

    def __post_init__(self):
        if (
            not 0 < self.recurrence_fraction <= 1
            or not 0 < self.reference_fraction <= 1
            or self.minimum_occurrences < 2
            or self.continuation_gap_ratio <= 0
        ):
            raise ValueError("invalid profile thresholds")


def derive_profile(census: Census, config=ProfileConfig()) -> DocumentProfile:
    fields = {k: ProfileField(None, 0, (), "profile/unknown/1", "UNKNOWN") for k in FIELDS}

    def put(name, value, records, rule, confidence=1.0):
        ids = tuple(sorted({r.candidate_id for r in records}))
        if ids:
            fields[name] = ProfileField(value, confidence, ids, rule, "PROBED")

    by = {
        kind: [r for r in census.records if r.category == kind]
        for kind in {r.category for r in census.records}
    }
    langs = by.get("language_token", [])
    counts = {
        code: sum(r.count for r in langs if dict(r.attributes)["code"] == code)
        for code in ("fr", "en")
    }
    if sum(counts.values()) >= 2 and len(set(counts.values())) > 1:
        code = max(counts, key=counts.get)
        put(
            "language",
            code,
            langs,
            "language/function-words/1",
            counts[code] / sum(counts.values()),
        )
    ids = by.get("identifier", [])
    if ids:
        families = []
        for signature in sorted({dict(r.attributes)["shape"] for r in ids}):
            matching = [r for r in ids if dict(r.attributes)["shape"] == signature]
            families.append(
                {
                    "family_id": stable_id("family", signature),
                    "shape": signature,
                    "revision_markers": sorted(
                        {
                            m[1]
                            for r in matching
                            for m in [
                                re.match(
                                    r"(version|rev(?:ision)?|v)\s*[:.]?\s*\d",
                                    dict(r.attributes)["revision_tail"],
                                    re.I,
                                )
                            ]
                            if m
                        }
                    ),
                }
            )
        put("identifier_families", families, ids, "identifier/shape/1")
    numbering = by.get("numbering", [])
    connectors = by.get("connector", [])
    if numbering:
        # Only generic syntactic operators have deterministic direction. An unknown
        # connector remains in observations and is not relabelled as a range.
        observed = sorted({r.literal for r in connectors})
        ranges = [
            s for s in observed if s.casefold() in {"à", "to", "through", "–", "—", "-", "..", "…"}
        ]
        lists = [s for s in observed if s in {"", ",", ";", ", §", "; §", "§"}]
        markers = sorted({dict(r.attributes)["marker"] for r in numbering})
        put(
            "numbering_style",
            {
                "markers": markers,
                "depths": sorted({int(dict(r.attributes)["depth"]) for r in numbering}),
                "range_connectors": ranges,
                "list_connectors": lists,
                "unknown_connectors": [s for s in observed if s not in ranges + lists],
            },
            numbering + connectors,
            "numbering/literal-connectors/1",
        )
    labels = by.get("label", [])
    selected = []
    support = []
    for literal in sorted({r.literal for r in labels}):
        group = [r for r in labels if r.literal == literal]
        count = sum(r.count for r in group)
        positive = sum(
            r.count for r in group if dict(r.attributes)["follow"] == "identifier-bearing"
        )
        if count >= config.minimum_occurrences and positive / count >= config.reference_fraction:
            selected.append(literal)
            support.extend(group)
    if selected:
        put(
            "reference_fields",
            {
                "labels": selected,
                "stop_labels": sorted({r.literal for r in labels}),
                "continuation": "same_page_column_until_boundary",
                "role_line": "undecidable",
                "max_gap_ratio": config.continuation_gap_ratio,
            },
            support,
            "labels/identifier-follow/1",
        )
    tables = by.get("table_header", [])
    if tables:
        put(
            "id_row_tables",
            {
                "tables": [
                    {
                        "headers": json.loads(r.literal),
                        "id_columns": json.loads(dict(r.attributes)["id_columns"]),
                        "text_columns": json.loads(dict(r.attributes)["text_columns"]),
                        "empty_counts": json.loads(dict(r.attributes)["empty"]),
                        "unreadable_counts": json.loads(dict(r.attributes)["unreadable"]),
                        "row_count": int(dict(r.attributes)["rows"]),
                    }
                    for r in tables
                ]
            },
            tables,
            "tables/observed-topology/1",
        )
    recurring = [
        r
        for r in by.get("recurrence", [])
        if len({e.page for e in r.evidence}) >= max(2, census.pages * config.recurrence_fraction)
        and dict(r.attributes)["edge"] == "True"
    ]
    geometric = [
        r
        for r in by.get("geometry", [])
        if dict(r.attributes)["rotated"] == "True" and dict(r.attributes)["large"] == "True"
    ]
    # No lexical lifecycle assertion: a geometric mark is only a mark candidate.
    if recurring or geometric:
        put(
            "furniture",
            {
                "running_patterns": sorted({r.literal for r in recurring}),
                "mark_literals": sorted({r.literal for r in geometric}),
                "default": "UNKNOWN",
                **{
                    key: float(dict(by["recurrence"][0].attributes)[key])
                    for key in ("edge_fraction", "large_points", "rotation_threshold")
                },
            },
            recurring + geometric,
            "furniture/recurrence-and-geometry/1",
        )
    notation = by.get("notation", [])
    decimal = [r for r in notation if dict(r.attributes)["kind"] == "decimal"]
    separators = {s for r in decimal for s in (".", ",") if s in r.literal}
    if len(separators) == 1 and any(
        not re.fullmatch(r"[+−-]?[1-9]\d{0,2}[.,]\d{3}", r.literal) for r in decimal
    ):
        put(
            "numeric_locale",
            {
                "decimal_separator": next(iter(separators)),
                "grouping": [
                    r.literal for r in notation if dict(r.attributes)["kind"] == "grouping"
                ],
                "units": sorted(
                    {r.literal for r in notation if dict(r.attributes)["kind"] == "unit"}
                ),
                "comparators": sorted(
                    {r.literal for r in notation if dict(r.attributes)["kind"] == "operator"}
                ),
            },
            notation,
            "locale/consistent-separator/1",
        )
    pages = by.get("page", [])
    put(
        "reading_coverage",
        {
            "pages": census.pages,
            "text_layer_pages": sum(
                r.count for r in pages if dict(r.attributes)["text_layer"] == "True"
            ),
        },
        pages,
        "coverage/page-inventory/1",
    )
    signature = stable_id(
        VERSION,
        sorted({r.literal for r in labels}),
        sorted({r.literal for r in tables}),
        fields["numbering_style"].value,
        sorted({dict(r.attributes)["shape"] for r in ids}),
    )
    return DocumentProfile(census.source_id, derive_census_id(census), fields, signature)


def apply_reviews(profile, reviews, census):
    ids = {r.candidate_id for r in census.records}
    fields = dict(profile.fields)
    history = list(profile.reviews)
    if census.source_id != profile.source_id or derive_census_id(census) != profile.census_id:
        raise ValueError("stale census/profile")
    for review in reviews:
        if review.source_id != profile.source_id or review.review_id in {
            r.review_id for r in history
        }:
            raise ValueError("review identity conflict")
        previous = next((r.review_id for r in reversed(history) if r.field == review.field), None)
        if review.previous_review_id != previous or not set(review.replacement.evidence) <= ids:
            raise ValueError("review chain/evidence conflict")
        fields[review.field] = review.replacement
        history.append(review)
    return replace(profile, fields=fields, reviews=tuple(history))


def derive_census_id(census):
    return replay_digest([census])


def profile_from_dict(data):
    def field(f):
        return ProfileField(**{**f, "evidence": tuple(f["evidence"])})

    return DocumentProfile(
        **{
            **data,
            "fields": {k: field(v) for k, v in data["fields"].items()},
            "reviews": tuple(
                ProfileReview(**{**r, "replacement": field(r["replacement"])})
                for r in data.get("reviews", ())
            ),
        }
    )
