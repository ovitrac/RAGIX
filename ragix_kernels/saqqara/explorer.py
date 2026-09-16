"""Library-first deterministic document exploration and explicit local file intake.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
from .census import DocumentDigest, PageDigest, TableObservation, Evidence, CensusConfig, census
from .table_views import TableCell
from .profile import ProfileConfig, derive_profile
from .profile_readers import read_document
from ..harvest.report import build_report, canonical_json, replay_digest


@dataclass(frozen=True)
class ExplorerResult:
    document: DocumentDigest
    census: object
    profile: object
    reading: object
    report: object


def explore(
    document: DocumentDigest,
    *,
    provenance=None,
    census_config=CensusConfig(),
    profile_config=ProfileConfig(),
) -> ExplorerResult:
    from .failures import CONSTRUCT_ERRORS, stage_failure, failure_report

    if profile_config.recurrence_fraction is not None:
        census_config = replace(
            census_config, recurrence_fraction=profile_config.recurrence_fraction
        )
    observed = profile = reading = None
    stage = "census"
    try:
        observed = census(document, census_config)
        stage = "profile"
        profile = derive_profile(observed, profile_config)
        stage = "read"
        reading = read_document(document, observed, profile)
        stage = "report"
        report = build_report(document, observed, profile, reading, provenance)
    except CONSTRUCT_ERRORS as error:
        report = failure_report(stage_failure(document, stage, error))
    return ExplorerResult(document, observed, profile, reading, report)


def imported_provenance(*, require_clean=False):
    """Hash the package actually imported, never a presumed sibling checkout.

    Untracked Python sources count in the development hash; caches do not. A gate
    requires a clean git checkout. Source hashes are not release pins.
    """
    import ragix_kernels

    root = Path(ragix_kernels.__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    hashes = [
        (p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files
    ]

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    if require_clean and (sha is None or status is None or status):
        raise ValueError("clean imported source required for gate")
    versions = {}
    for name in ("pymupdf", "pdfplumber", "pdfminer.six", "pypdf"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import platform, sqlite3

    return {
        "source_sha256": replay_digest(hashes),
        "git_sha": sha,
        "dirty": None if status is None else bool(status),
        "porcelain": status,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "extractors": versions,
    }


def digest_pdf(path: Path, *, expected_pymupdf=None) -> DocumentDigest:
    """Opt-in PyMuPDF intake. Empty text pages remain inventoried. No OCR guess."""
    from .adapters.pdf_mupdf import MuPdfTextReader

    reader = MuPdfTextReader()
    if expected_pymupdf and reader.version != expected_pymupdf:
        raise ValueError("extractor lock mismatch")
    path = Path(path)
    source_id = hashlib.sha256(path.read_bytes()).hexdigest()
    reader.open(path)
    pages = []
    try:
        for number in range(1, len(reader._document) + 1):
            geometry = reader.page_geometry(number, source_id=source_id)
            page = reader._document[number - 1]
            tables = []
            # Native table observations, including blank and unreadable cells, stay
            # separate from semantic id-row recognition.
            for index, table in enumerate(page.find_tables().tables):
                rows = table.extract()
                if not rows:
                    continue
                headers = tuple("" if h is None else h for h in rows[0])
                ident = f"table:{number}:{index}"
                evidence = tuple(
                    Evidence(
                        source_id,
                        number,
                        ident + f":{r}:{c}",
                        0,
                        len(cell or ""),
                        cell or "",
                        tuple(table.rows[r].cells[c] or table.bbox),
                    )
                    for r, row in enumerate(rows)
                    for c, cell in enumerate(row)
                    if table.rows[r].cells[c] is not None or cell
                )
                cell_rows = tuple(
                    tuple(
                        TableCell(
                            ident + f":{r}:{c}",
                            cell,
                            tuple(table.rows[r].cells[c] or table.bbox),
                            tuple(
                                s.span_id
                                for s in geometry["spans"]
                                if min(s.bbox[2], (table.rows[r].cells[c] or table.bbox)[2])
                                > max(s.bbox[0], (table.rows[r].cells[c] or table.bbox)[0])
                                and min(s.bbox[3], (table.rows[r].cells[c] or table.bbox)[3])
                                > max(s.bbox[1], (table.rows[r].cells[c] or table.bbox)[1])
                            ),
                            flags=(
                                ()
                                if table.rows[r].cells[c] is not None
                                else ("MISSING_CELL_GEOMETRY",)
                            ),
                            geometry_kind="cell_box",
                        )
                        for c, cell in enumerate(row)
                        if table.rows[r].cells[c] is not None or cell
                    )
                    for r, row in enumerate(rows)
                )
                tables.append(
                    TableObservation(
                        ident,
                        number,
                        headers,
                        tuple(tuple(r) for r in rows[1:]),
                        evidence,
                        cell_rows=cell_rows,
                    )
                )
            pages.append(
                PageDigest(
                    number,
                    geometry["width"],
                    geometry["height"],
                    tuple(geometry["spans"]),
                    tuple(geometry["vertical_rules"]),
                    tuple(tables),
                    len(page.get_drawings()),
                    tuple(geometry["horizontal_rules"]),
                )
            )
    finally:
        reader.close()
    return DocumentDigest(source_id, tuple(pages), reader.name, reader.version)
