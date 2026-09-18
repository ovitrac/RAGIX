"""Versioned evidence-backed notation profiles and append-only reviews.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace, field
import json
import re
import math
from .census import Census, bare_connector
from .field_views import stable_id
from ..harvest.report import replay_digest

from .value_windows import ContinuationPolicy, DEFAULT_CONTINUATION, policy_from_dict

VERSION = "document-profile/0.6"
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
    diagnostics: dict = field(default_factory=dict)

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
        if (
            not isinstance(self.diagnostics, dict)
            or set(self.diagnostics)
            - {
                "ambiguous_observations",
                "observed_ratio",
                "reference_counts",
                "unresolved_occurrences",
                "label_in_prose",
            }
            or (
                "ambiguous_observations" in self.diagnostics
                and (
                    type(self.diagnostics["ambiguous_observations"]) is not int
                    or self.diagnostics["ambiguous_observations"] < 0
                )
            )
            or (
                "observed_ratio" in self.diagnostics
                and not 0 <= self.diagnostics["observed_ratio"] <= 1
            )
        ):
            raise ValueError("closed profile diagnostics required")
        if self.status == "UNKNOWN":
            if self.value is not None or (
                self.confidence != 0 and self.rule_id != "labels/identifier-plurality/2"
            ):
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
    continuation_policy: ContinuationPolicy = DEFAULT_CONTINUATION

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
            },
            "id_row_tables": {"tables", "reconstructed", "unresolved", "excluded_furniture"},
            "furniture": {
                "running_patterns",
                "mark_literals",
                "default",
                "edge_fraction",
                "large_points",
                "rotation_threshold",
                "stamp_lines",
                "stamp_count",
                "recurrence_fraction",
            },
            "numeric_locale": {
                "strategy",
                "decimal_separator",
                "prior_strength",
                "separator_counts",
                "n",
                "dominance_ratio",
                "minimum_n",
                "dominant_n",
                "grouping",
                "units",
                "comparators",
            },
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
            if name == "numeric_locale" and value["decimal_separator"] not in {None, ".", ","}:
                raise ValueError("locale schema")
            if name == "numeric_locale":
                counts = value["separator_counts"]
                if (
                    value["strategy"] != "per_token"
                    or value["prior_strength"] not in {"none", "weak", "dominant"}
                    or not isinstance(counts, dict)
                    or set(counts) != {".", ","}
                    or any(type(n) is not int or n < 0 for n in counts.values())
                    or type(value["n"]) is not int
                    or value["n"] != sum(counts.values())
                    or type(value["minimum_n"]) is not int
                    or value["minimum_n"] < 2
                    or type(value["dominant_n"]) is not int
                    or value["dominant_n"] < value["minimum_n"]
                    or not 0.5 < value["dominance_ratio"] <= 1
                    or (value["decimal_separator"] is None) != (value["prior_strength"] == "none")
                ):
                    raise ValueError("locale prior schema")
            if name == "reference_fields" and (
                value["continuation"] != "same_page_column_until_boundary"
                or value["role_line"] != "undecidable"
            ):
                raise ValueError("unsupported field policy")
        if len({r.review_id for r in self.reviews}) != len(self.reviews):
            raise ValueError("duplicate profile review")


@dataclass(frozen=True)
class ProfileConfig:
    recurrence_fraction: float | None = None
    minimum_occurrences: int = 2
    locale_dominance_ratio: float = 0.9
    locale_minimum_n: int = 2
    locale_dominant_n: int = 5

    def __post_init__(self):
        if (
            (self.recurrence_fraction is not None and not 0 < self.recurrence_fraction <= 1)
            or self.minimum_occurrences < 2
            or not 0.5 < self.locale_dominance_ratio <= 1
            or type(self.locale_minimum_n) is not int
            or self.locale_minimum_n < 2
            or type(self.locale_dominant_n) is not int
            or self.locale_dominant_n < self.locale_minimum_n
        ):
            raise ValueError("invalid profile thresholds")


def derive_profile(census: Census, config=ProfileConfig()) -> DocumentProfile:
    recurrence_fraction = census.geometry_policy["recurrence_fraction"]
    if config.recurrence_fraction is not None and config.recurrence_fraction != recurrence_fraction:
        raise ValueError("recurrence policy differs from census; rebuild census")
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
        markers = sorted({dict(r.attributes)["marker"] for r in numbering})
        # Connectors are classified bare: the marker of the next number is not theirs.
        observed = sorted({bare_connector(r.literal, markers) for r in connectors})
        ranges = [
            s for s in observed if s.casefold() in {"à", "to", "through", "–", "—", "-", "..", "…"}
        ]
        lists = [s for s in observed if s in {"", ",", ";"}]
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
    observed_labels = by.get("label", [])
    prose = [r for r in observed_labels if dict(r.attributes)["follow"] == "prose"]
    labels = [r for r in observed_labels if r not in prose]
    selected = []
    support = []
    stats = []
    for literal in sorted({r.literal for r in labels}):
        group = [r for r in labels if r.literal == literal]
        counts = {
            kind: sum(r.count for r in group if dict(r.attributes)["follow"] == kind)
            for kind in ("identifier-bearing", "number-bearing", "free-text", "empty")
        }
        positive = counts["identifier-bearing"]
        negative = counts["number-bearing"] + counts["free-text"]
        competitor = max(counts["number-bearing"], counts["free-text"])
        accepted = positive >= config.minimum_occurrences and positive > competitor
        stats.append(
            {
                "label": literal,
                "positives": positive,
                "non_empty_negatives": negative,
                "empty": counts["empty"],
                "classes": counts,
                "minimum_occurrences": config.minimum_occurrences,
                "confidence": positive / (positive + negative) if positive + negative else 0,
                "selected": accepted,
                "reason": (
                    None
                    if accepted
                    else (
                        "TOO_FEW_POSITIVES"
                        if positive < config.minimum_occurrences
                        else "COMPETING_PLURALITY"
                    )
                ),
            }
        )
        if accepted:
            selected.append(literal)
            support.extend(group)
    from .value_windows import unresolved_reason

    unresolved = [
        {
            "window_id": w.window_id,
            "label": w.label,
            "page": w.views[0].page,
            "span_id": w.views[0].view_id,
            "reason": unresolved_reason(w),
        }
        for w in census.windows
        if not w.following_text.strip()
    ]
    chosen = [s for s in stats if s["selected"]]
    confidence = (
        sum(s["positives"] for s in chosen)
        / sum(s["positives"] + s["non_empty_negatives"] for s in chosen)
        if chosen
        else max((s["confidence"] for s in stats), default=0)
    )
    if selected:
        put(
            "reference_fields",
            {
                "labels": selected,
                "stop_labels": sorted({r.literal for r in labels}),
                "continuation": "same_page_column_until_boundary",
                "role_line": "undecidable",
            },
            support,
            "labels/identifier-plurality/2",
            confidence,
        )
    fields["reference_fields"] = replace(
        fields["reference_fields"],
        confidence=confidence,
        rule_id="labels/identifier-plurality/2",
        evidence=tuple(sorted(r.candidate_id for r in observed_labels)),
        diagnostics={
            "reference_counts": stats,
            "unresolved_occurrences": unresolved,
            "label_in_prose": [
                {
                    "label": r.literal,
                    "page": e.page,
                    "span_id": e.span_id,
                    "reason": dict(r.attributes)["pattern"],
                }
                for r in prose
                for e in r.evidence
            ],
        },
    )
    tables = by.get("table_header", [])
    raw_ids = set(census.table_analysis.candidate_ids) | set(census.table_analysis.excluded)
    declared_tables = [r for r in tables if dict(r.attributes).get("table_id") not in raw_ids]
    if tables:
        put(
            "id_row_tables",
            {
                "reconstructed": [
                    {
                        "table_id": t.table_id,
                        "headers": list(t.headers),
                        "roles": list(t.roles),
                        "continued_on": t.continued_on,
                        "repetition_count": t.repetition_count,
                        "fragments": list(t.fragments),
                        "policy": t.policy,
                        "flags": t.flags,
                    }
                    for t in census.table_analysis.tables
                ],
                "unresolved": [asdict(f) for f in census.table_analysis.findings],
                "excluded_furniture": list(census.table_analysis.excluded),
                "tables": [
                    {
                        "headers": json.loads(r.literal),
                        "id_columns": json.loads(dict(r.attributes)["id_columns"]),
                        "text_columns": json.loads(dict(r.attributes)["text_columns"]),
                        "empty_counts": json.loads(dict(r.attributes)["empty"]),
                        "unreadable_counts": json.loads(dict(r.attributes)["unreadable"]),
                        "row_count": int(dict(r.attributes)["rows"]),
                    }
                    for r in declared_tables
                ],
            },
            tables,
            "tables/observed-topology/1",
        )
    recurring = [
        r
        for r in by.get("recurrence", [])
        if len({e.page for e in r.evidence}) >= max(2, census.pages * recurrence_fraction)
        and dict(r.attributes)["edge"] == "True"
    ]
    geometric = [
        r
        for r in by.get("geometry", [])
        if dict(r.attributes)["rotated"] == "True" and dict(r.attributes)["large"] == "True"
    ]
    stamp_records = by.get("stamp", [])
    from .privacy import census_stamps

    stamp_lines = list(census_stamps(census))
    # A binary flag on observed text, including a measured zero, is not UNKNOWN.
    if by.get("page"):
        put(
            "furniture",
            {
                "running_patterns": sorted({r.literal for r in recurring}),
                "mark_literals": sorted({r.literal for r in geometric}),
                "default": "UNKNOWN",
                **census.geometry_policy,
                "stamp_lines": stamp_lines,
                "stamp_count": len(stamp_lines),
            },
            recurring + geometric + stamp_records + by.get("page", []),
            "furniture/recurrence-geometry-and-stamps/1",
        )
    notation = by.get("notation", [])
    from ..harvest.numeric_locale import resolve_number

    physical = [
        r for r in notation if dict(r.attributes)["kind"] in {"decimal", "physical_integer"}
    ]
    counts = {".": 0, ",": 0}
    for record in physical:
        local = resolve_number(record.literal)
        if local.value is not None and local.separator is not None:
            counts[local.separator] += record.count
    n = sum(counts.values())
    winner = max(counts, key=counts.get)
    ratio = counts[winner] / n if n else 0
    separator = (
        winner if n >= config.locale_minimum_n and ratio >= config.locale_dominance_ratio else None
    )
    strength = ("dominant" if n >= config.locale_dominant_n else "weak") if separator else "none"
    if physical:
        put(
            "numeric_locale",
            {
                "strategy": "per_token",
                "decimal_separator": separator,
                "prior_strength": strength,
                "separator_counts": counts,
                "n": n,
                "dominance_ratio": config.locale_dominance_ratio,
                "minimum_n": config.locale_minimum_n,
                "dominant_n": config.locale_dominant_n,
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
            physical,
            "locale/token-first-prior/1",
            ratio if strength == "dominant" else min(ratio, 0.5),
        )
        fields["numeric_locale"] = replace(
            fields["numeric_locale"],
            diagnostics={
                "ambiguous_observations": sum(
                    r.count for r in physical if resolve_number(r.literal).value is None
                ),
                "observed_ratio": ratio,
            },
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
    return DocumentProfile(
        census.source_id,
        derive_census_id(census),
        fields,
        signature,
        continuation_policy=census.continuation_policy,
    )


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
            "continuation_policy": policy_from_dict(data["continuation_policy"]),
            "fields": {k: field(v) for k, v in data["fields"].items()},
            "reviews": tuple(
                ProfileReview(**{**r, "replacement": field(r["replacement"])})
                for r in data.get("reviews", ())
            ),
        }
    )
