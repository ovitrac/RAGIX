#!/usr/bin/env python3
"""Inspect and remove isolated watermark instructions into a validated derivative.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragix_core.pdf_watermark import Limits, Refused, inspect_pdf, apply_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="Inventory exact, isolated, rotated/translucent text")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--text", required=True, help="Exact marker text; case preserved")
    inspect.add_argument("--plan", type=Path, required=True)
    inspect.add_argument("--dpi", type=int, default=72)
    apply = sub.add_parser("apply", help="Produce a separate derivative and audit after validation")
    apply.add_argument("input", type=Path)
    apply.add_argument("--plan", type=Path, required=True)
    selection = apply.add_mutually_exclusive_group(required=True)
    selection.add_argument("--select", nargs="+", metavar="CANDIDATE_ID")
    selection.add_argument("--all-candidates", action="store_true")
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "inspect":
            plan = inspect_pdf(args.input, args.text, Limits(dpi=args.dpi))
            with args.plan.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "status": plan["status"],
                        "candidates": len(plan["candidates"]),
                        "pages": plan["page_count"],
                        "refused_pages": len(plan["refusals"]),
                    }
                )
            )
        else:
            plan = json.loads(args.plan.read_text())
            ids = (
                [c["candidate_id"] for c in plan["candidates"]]
                if args.all_candidates
                else args.select
            )
            audit = apply_plan(args.input, plan, ids, args.output, args.report)
            print(
                json.dumps(
                    {
                        "status": audit["status"],
                        "removed": len(audit["selected"]),
                        "output_sha256": audit["output_sha256"],
                    }
                )
            )
        return 0
    except (Refused, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
