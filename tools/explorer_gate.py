#!/usr/bin/env python3
"""Synthetic Explorer replay and optional local classification measurement.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragix_kernels.saqqara.explorer import explore, digest_pdf, imported_provenance
from ragix_kernels.harvest.report import canonical_json, replay_digest


def fixtures():
    spec = importlib.util.spec_from_file_location(
        "explorer_synthetic", ROOT / "tests/saqqara/test_explorer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    provenance = imported_provenance(require_clean=args.require_clean)
    import platform

    platform_name = {"x86_64": "linux-64", "aarch64": "linux-aarch64"}.get(platform.machine())
    if platform_name is None:
        raise ValueError("unlocked platform")
    lock = json.loads((ROOT / "tools" / ("explorer-lock-" + platform_name + ".json")).read_text())
    for key, actual in (
        ("python", provenance["python"]),
        ("sqlite", provenance["sqlite"]),
        ("pymupdf", provenance["extractors"]["pymupdf"]),
    ):
        if lock[key] != actual:
            raise ValueError("platform lock mismatch: " + key)
    f = fixtures()
    records = {}
    for name, kwargs in [
        ("baseline", {}),
        ("label", {"label": "Link field"}),
        ("marker", {"marker": "Section"}),
        ("locale", {"separator": "."}),
        ("columns", {"order": (2, 0, 1)}),
        ("family", {"family": "LONG/FAMILY"}),
        ("furniture", {"mark": "RETIRED", "header": "Different header"}),
    ]:
        result = explore(f.fixture(**kwargs))
        counts = Counter()
        for record in result.census.records:
            counts[record.category] += record.count
        records[name] = {
            "census_counts": dict(sorted(counts.items())),
            "census_digest": replay_digest([result.census]),
            "profile_digest": replay_digest([result.profile]),
            "reading_digest": replay_digest([result.reading]),
            "report_digest": result.report.replay_digest,
        }
    import pymupdf

    with tempfile.TemporaryDirectory(prefix="explorer-synthetic-") as directory:
        path = Path(directory) / "synthetic.pdf"
        pdf = pymupdf.open()
        for number in range(2):
            page = pdf.new_page()
            for y, text in [
                (25, "Running header"),
                (100, "Link field: YZ-Q-582 Section 3.1; 3.2 to 3.5"),
                (116, "YZ-Q-849 Section 7.2"),
                (150, "Value: 9.25 V"),
            ]:
                page.insert_text((40, y), text, fontsize=10)
        pdf.save(path, no_new_id=True)
        pdf.close()
        document = digest_pdf(path, expected_pymupdf="1.27.2.2")
        result = explore(document)
        records["pdf"] = {
            "source_id": document.source_id,
            "census_digest": replay_digest([result.census]),
            "profile_digest": replay_digest([result.profile]),
            "reading_digest": replay_digest([result.reading]),
            "report_digest": result.report.replay_digest,
            "fields": len(result.reading.fields),
        }
    output = {"provenance": provenance, "synthetic": records, "models": []}
    if args.models:
        if not args.endpoint or not args.cache:
            parser.error("models require explicit --endpoint and --cache")
        from urllib.request import urlopen
        from ragix_kernels.cache import LLMCache
        from ragix_kernels.harvest.profile_classify import (
            ClassificationConfig,
            OllamaStructuredPort,
            classify_profile,
        )
        from ragix_kernels.saqqara.census import DocumentDigest, PageDigest

        with urlopen(args.endpoint.rstrip("/") + "/api/tags", timeout=10) as response:
            model_ids = {m["name"]: m["digest"] for m in json.load(response)["models"]}
        doc = DocumentDigest(
            "synthetic-model",
            tuple(
                PageDigest(
                    p,
                    600,
                    800,
                    (replace(f.line("Link field: ST-X-752", p, 0), source_id="synthetic-model"),),
                )
                for p in (1, 2)
            ),
            "synthetic",
            "1",
        )
        sample = explore(doc)
        expected = {
            r.candidate_id: (
                "reference_field"
                if r.category == "label"
                else "notation" if r.category == "identifier" else None
            )
            for r in sample.census.records
        }
        for model in args.models:
            if model not in model_ids:
                raise ValueError("model not installed: " + model)
            for thinking in (False, True):
                config = ClassificationConfig(model, model_ids[model], True, thinking, 4096)
                try:
                    result = classify_profile(
                        sample.census,
                        sample.profile,
                        config,
                        port=OllamaStructuredPort(args.endpoint, timeout=180),
                        cache=LLMCache(args.cache, endpoint=args.endpoint),
                    )
                    correct = sum(
                        d["role"] == expected[d["candidate_id"]] for d in result.decisions
                    )
                    category_by_id = {r.candidate_id: r.category for r in sample.census.records}
                    observed = {d["candidate_id"]: d["role"] for d in result.decisions}
                    per_field = {}
                    for candidate_id, role in expected.items():
                        field = category_by_id[candidate_id]
                        metrics = per_field.setdefault(field, {"correct": 0, "denominator": 0})
                        metrics["denominator"] += 1
                        metrics["correct"] += int(
                            candidate_id in observed and observed[candidate_id] == role
                        )
                    row = {
                        **result.manifest,
                        "correct": correct,
                        "denominator": len(expected),
                        "per_field": per_field,
                        "decisions": result.decisions,
                        "expected": expected,
                    }
                except Exception as exc:
                    row = {
                        "model": model,
                        "model_digest": model_ids[model],
                        "thinking": thinking,
                        "outcome": "transport_refused",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "denominator": len(expected),
                    }
                output["models"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
