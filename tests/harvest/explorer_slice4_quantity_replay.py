"""Clean synthetic E11 replay of exact spans and member provenance.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
import json
from pathlib import Path
from ragix_kernels.saqqara.explorer import imported_provenance
from ragix_kernels.harvest.quantitative import harvest, roots
from ragix_kernels.harvest.report import replay_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(
        Path(__file__).with_name("fixtures_explorer_slice4_quantities.json").read_text()
    )["cases"]
    output = {"provenance": imported_provenance(require_clean=True), "cases": {}}
    for token_locale in (False, True):
        for case in cases:
            candidates = harvest(
                case["text"],
                source_id="synthetic-e11",
                node_id=case["id"],
                classification="CONTENT",
                token_locale=token_locale,
            )
            expected = (case["start"], case["end"], case["raw"], case["kind"])
            assert sum((c.start, c.end, c.raw, c.kind) == expected for c in roots(candidates)) == 1
            ids = {c.candidate_id for c in candidates}
            assert all(
                case["text"][c.start : c.end] == c.raw
                and all(m.candidate_id in ids for m in c.members)
                for c in candidates
            )
            output["cases"][f"{case['id']}:{token_locale}"] = replay_digest(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
