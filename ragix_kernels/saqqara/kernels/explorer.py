"""Explicit deterministic adapters with counted per-document construct failures.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict
import json
from ...base import Kernel
from ..census import census, digest_from_dict, census_from_dict, CensusConfig
from ..profile import derive_profile, profile_from_dict, ProfileConfig
from ..profile_readers import read_document
from ..failures import CONSTRUCT_ERRORS, stage_failure, failure_report
from ...harvest.report import build_report, canonical_json


def dependency(input, name):
    data = json.loads(input.dependencies[name].read_text())
    if "_meta" in data and not data["_meta"].get("success"):
        raise ValueError("failed dependency")
    return data.get("data", data)


def attempt(row, stage, operation):
    if "failure" in row:
        return row
    try:
        return {**row, **operation()}
    except CONSTRUCT_ERRORS as error:
        return {**row, "failure": asdict(stage_failure(row.get("document"), stage, error))}


class CensusKernel(Kernel):
    name = "explorer_census"
    version = "0.7.0"
    stage = 1
    category = "docs"
    requires = []
    provides = ["document_census"]

    def compute(self, input):
        documents = input.config["documents"]
        if not documents:
            raise ValueError("nonempty document manifest required")
        from ..value_windows import policy_from_dict

        options = dict(input.config.get("census_options", {}))
        if "continuation_policy" in options:
            options["continuation_policy"] = policy_from_dict(options["continuation_policy"])
        if "table_policy" in options:
            from ..table_views import TablePolicy

            options["table_policy"] = TablePolicy(**options["table_policy"])
        config = CensusConfig(**options)
        results = []
        seen = set()
        for data in documents:

            def run():
                doc = digest_from_dict(data)
                if doc.source_id in seen:
                    raise ValueError("duplicate document identity")
                seen.add(doc.source_id)
                return {"document": asdict(doc), "census": asdict(census(doc, config))}

            results.append(attempt({"document": data}, "census", run))
        return {"documents": results}

    def summarize(self, data):
        return "Document census recorded; construct failures remain explicit."


class ProfileKernel(Kernel):
    name = "explorer_profile"
    version = "0.7.0"
    stage = 2
    category = "docs"
    requires = ["explorer_census"]
    provides = ["document_profiles"]

    def compute(self, input):
        data = dependency(input, "explorer_census")
        config = ProfileConfig(**input.config.get("profile_options", {}))
        return {
            "documents": [
                attempt(
                    row,
                    "profile",
                    lambda: {
                        "profile": asdict(derive_profile(census_from_dict(row["census"]), config))
                    },
                )
                for row in data["documents"]
            ]
        }

    def summarize(self, data):
        return "Profiles derived for readable documents; failures preserved."


class ReadKernel(Kernel):
    name = "explorer_read"
    version = "0.8.1"
    stage = 2
    category = "docs"
    requires = ["explorer_profile"]
    provides = ["document_readings"]

    def compute(self, input):
        return {
            "documents": [
                attempt(
                    row,
                    "read",
                    lambda: {
                        "reading": asdict(
                            read_document(
                                digest_from_dict(row["document"]),
                                census_from_dict(row["census"]),
                                profile_from_dict(row["profile"]),
                            )
                        )
                    },
                )
                for row in dependency(input, "explorer_profile")["documents"]
            ]
        }

    def summarize(self, data):
        return "Readings and construct failures recorded independently."


class ReportKernel(Kernel):
    name = "explorer_report"
    version = "0.8.1"
    stage = 3
    category = "docs"
    requires = ["explorer_read"]
    provides = ["document_reading_report"]

    def compute(self, input):
        results = []
        for row in dependency(input, "explorer_read")["documents"]:

            def run():
                doc = digest_from_dict(row["document"])
                observed = census_from_dict(row["census"])
                profile = profile_from_dict(row["profile"])
                reading = read_document(doc, observed, profile)
                if canonical_json(reading) != canonical_json(row["reading"]):
                    raise ValueError("reading replay mismatch")
                return {
                    "report": asdict(
                        build_report(
                            doc, observed, profile, reading, input.config.get("provenance")
                        )
                    )
                }

            outcome = attempt(row, "report", run)
            results.append(
                asdict(failure_report(outcome["failure"]))
                if "failure" in outcome
                else outcome["report"]
            )
        return {"reports": results}

    def summarize(self, data):
        return "Reports produced; failed documents remain visibly failed."


def register_explorer_kernels():
    from ...registry import KernelRegistry

    for kernel in (CensusKernel, ProfileKernel, ReadKernel, ReportKernel):
        KernelRegistry.register(kernel)
