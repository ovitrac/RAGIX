"""Explicitly registered deterministic explorer adapters; no model or network work.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict
import json
from ...base import Kernel
from ..census import census, digest_from_dict, census_from_dict
from ..profile import derive_profile, profile_from_dict
from ..profile_readers import read_document
from ...harvest.report import build_report


def dependency(input, name):
    data = json.loads(input.dependencies[name].read_text())
    if "_meta" in data and not data["_meta"].get("success"):
        raise ValueError("failed dependency")
    return data.get("data", data)


class CensusKernel(Kernel):
    name = "explorer_census"
    version = "0.1.0"
    stage = 1
    category = "docs"
    requires = []
    provides = ["document_census"]

    def compute(self, input):
        documents = input.config["documents"]
        if not documents:
            raise ValueError("nonempty document manifest required")
        results = []
        seen = set()
        for data in documents:
            document = digest_from_dict(data)
            if document.source_id in seen:
                raise ValueError("duplicate document identity in manifest")
            seen.add(document.source_id)
            results.append({"document": asdict(document), "census": asdict(census(document))})
        return {"documents": results}

    def summarize(self, data):
        return "Document notation census recorded."


class ProfileKernel(Kernel):
    name = "explorer_profile"
    version = "0.1.0"
    stage = 2
    category = "docs"
    requires = ["explorer_census"]
    provides = ["document_profiles"]

    def compute(self, input):
        data = dependency(input, "explorer_census")
        return {
            "documents": [
                {**row, "profile": asdict(derive_profile(census_from_dict(row["census"])))}
                for row in data["documents"]
            ]
        }

    def summarize(self, data):
        return "Evidence-backed document profiles derived."


class ReadKernel(Kernel):
    name = "explorer_read"
    version = "0.1.0"
    stage = 2
    category = "docs"
    requires = ["explorer_profile"]
    provides = ["document_readings"]

    def compute(self, input):
        data = dependency(input, "explorer_profile")
        return {
            "documents": [
                {
                    **row,
                    "reading": asdict(
                        read_document(
                            digest_from_dict(row["document"]),
                            census_from_dict(row["census"]),
                            profile_from_dict(row["profile"]),
                        )
                    ),
                }
                for row in data["documents"]
            ]
        }

    def summarize(self, data):
        return "Profile-driven readings and unknown-template findings recorded."


class ReportKernel(Kernel):
    name = "explorer_report"
    version = "0.1.0"
    stage = 3
    category = "docs"
    requires = ["explorer_read"]
    provides = ["document_reading_report"]

    def compute(self, input):
        data = dependency(input, "explorer_read")
        results = []
        for row in data["documents"]:
            doc = digest_from_dict(row["document"])
            observed = census_from_dict(row["census"])
            profile = profile_from_dict(row["profile"])
            # Decode by replaying the deterministic reader and checking its sealed
            # representation, rather than accepting a loose untyped payload.
            reading = read_document(doc, observed, profile)
            from ...harvest.report import canonical_json

            if canonical_json(reading) != canonical_json(row["reading"]):
                raise ValueError("reading replay mismatch")
            results.append(
                asdict(
                    build_report(doc, observed, profile, reading, input.config.get("provenance"))
                )
            )
        return {"reports": results}

    def summarize(self, data):
        return "Reading coverage and evidence-scoped reports recorded."


def register_explorer_kernels():
    from ...registry import KernelRegistry

    for kernel in (CensusKernel, ProfileKernel, ReadKernel, ReportKernel):
        KernelRegistry.register(kernel)
