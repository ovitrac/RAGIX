"""Read quantities from exact cell context without inventing text or semantics.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from collections import defaultdict
from dataclasses import asdict
from ..harvest.quantitative import harvest
from .census import page_lines


def read_cell_quantities(document, contexts, existing, locale):
    batches = []
    for context in contexts:
        candidates = harvest(
            context.value.text,
            source_id=document.source_id,
            node_id=context.value.cell_id,
            classification="UNKNOWN",
            uncertainty=context.value.flags,
            context=context,
            token_locale=True,
            locale_prior=locale,
            table_cell=True,
        )
        if any(c.quantitative for c in candidates):
            batches.append((context, candidates))
    # Replace only whole text-view batches whose character provenance is wholly
    # inside cells being re-read. This keeps composite member graphs closed and
    # does not erase observations or partially drop composite children.
    views = {v.view_id: v for p in document.pages for v in page_lines(p)}
    grouped = defaultdict(list)
    for q in existing:
        grouped[q["node_id"]].append(q)

    def located(q, context):
        view = views.get(q["node_id"])
        if view is None or view.page != context.value.page:
            return False
        refs = view.source_refs(q["start"], q["end"])
        if not refs:
            return False
        box = tuple(round(v, 3) for v in context.value.bbox)
        return all(
            box[0] <= round(r.bbox[0], 3)
            and round(r.bbox[2], 3) <= box[2]
            and box[1] <= round(r.bbox[1], 3)
            and round(r.bbox[3], 3) <= box[3]
            for r in refs
        )

    replaced = {
        node
        for node, qs in grouped.items()
        if qs and all(any(located(q, c) for c, _ in batches) for q in qs)
    }
    result = [q for q in existing if q["node_id"] not in replaced]
    result.extend(
        {**asdict(q), "needs_review": q.needs_review}
        for _, candidates in batches
        for q in candidates
    )
    return tuple(result)
