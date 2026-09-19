"""Local, isolated legacy Office conversion into the existing OOXML readers.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tempfile
import zipfile

from .adapters.contract import ConversionUnavailable, UnreadableFile
from .model import ConversionLocator

FILTERS = {
    ".doc": ("docx", "Office Open XML Text", "word/document.xml"),
    ".xls": ("xlsx", "Calc MS Excel 2007 XML", "xl/workbook.xml"),
}
MIME = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
INSTALL_MESSAGE = "Install LibreOffice with Writer and Calc to read legacy .doc and .xls files."
PROFILE = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
 <item oor:path="/org.openoffice.Office.Common/Security/Scripting">
  <prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop>
 </item>
 <item oor:path="/org.openoffice.Office.Writer/Content/Update">
  <prop oor:name="Link" oor:op="fuse"><value>0</value></prop>
 </item>
 <item oor:path="/org.openoffice.Office.Calc/Content/Update">
  <prop oor:name="Link" oor:op="fuse"><value>1</value></prop>
 </item>
</oor:items>
"""


@dataclass(frozen=True)
class ConversionOptions:
    timeout: float = 120.0
    max_input_bytes: int = 128 * 1024 * 1024
    max_output_bytes: int = 128 * 1024 * 1024
    max_expanded_bytes: int = 512 * 1024 * 1024
    max_zip_entries: int = 20000

    def __post_init__(self):
        import math

        if (
            type(self.timeout) not in (int, float)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("positive finite conversion timeout required")
        if any(
            type(n) is not int or n <= 0
            for n in (
                self.max_input_bytes,
                self.max_output_bytes,
                self.max_expanded_bytes,
                self.max_zip_entries,
            )
        ):
            raise ValueError("positive conversion byte/entry limits required")


@dataclass(frozen=True)
class ConvertedOffice:
    path: Path
    provenance: ConversionLocator


def libreoffice_executable():
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise ConversionUnavailable(INSTALL_MESSAGE)
    return executable


def _run(arguments, *, timeout, env=None):
    """Keep subprocess output off user logs; timeouts kill only this process group."""
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.Popen(
                arguments,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            raise ConversionUnavailable(INSTALL_MESSAGE) from error
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise UnreadableFile(
                "LibreOffice conversion timed out; no partial result was accepted."
            ) from error
        output.seek(0)
        return process.returncode, output.read(16384).decode("utf-8", errors="replace")


def _version(executable, timeout):
    status, output = _run([executable, "--version"], timeout=min(timeout, 10))
    lines = [line.strip() for line in output.splitlines() if line.startswith("LibreOffice ")]
    if status or not lines:
        raise ConversionUnavailable("LibreOffice could not report its version. " + INSTALL_MESSAGE)
    return lines[0]


def _read_source(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise UnreadableFile("Legacy Office input exceeds the declared byte limit.")
    return data


def _validate_package(path, required, options):
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise UnreadableFile(
            "LibreOffice produced no usable converted file; check the document and Writer/Calc installation."
        )
    if path.stat().st_size > options.max_output_bytes:
        raise UnreadableFile("Converted Office file exceeds the declared byte limit.")
    try:
        with zipfile.ZipFile(path) as package:
            entries = package.infolist()
            names = [entry.filename for entry in entries]
            if (
                len(entries) > options.max_zip_entries
                or sum(e.file_size for e in entries) > options.max_expanded_bytes
            ):
                raise UnreadableFile("Converted Office package exceeds declared expansion limits.")
            if len(set(names)) != len(names) or any(
                PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names
            ):
                raise UnreadableFile("Converted Office package has invalid member paths.")
            if not {"[Content_Types].xml", required} <= set(names) or package.testzip() is not None:
                raise UnreadableFile("LibreOffice output is not the expected valid OOXML package.")
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise UnreadableFile(
            "LibreOffice output is not the expected valid OOXML package."
        ) from error


@contextmanager
def converted_office(path, *, store=None, options=ConversionOptions()):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FILTERS:
        raise ValueError("Only legacy .doc and .xls conversion is supported.")
    executable = libreoffice_executable()
    version = _version(executable, options.timeout)
    source = _read_source(path, options.max_input_bytes)
    source_hash = hashlib.sha256(source).hexdigest()
    target, filter_name, required = FILTERS[suffix]
    with tempfile.TemporaryDirectory(prefix="ragix-office-") as work:
        root = Path(work)
        profile = root / "profile"
        user = profile / "user"
        user.mkdir(parents=True)
        (user / "registrymodifications.xcu").write_text(PROFILE, encoding="utf-8")
        inputs = root / "input"
        outputs = root / "output"
        inputs.mkdir()
        outputs.mkdir()
        copied = inputs / ("source" + suffix)
        copied.write_bytes(source)
        env = dict(os.environ, SAL_USE_VCLPLUGIN="svp", LC_ALL="C")
        status, _ = _run(
            [
                executable,
                "-env:UserInstallation=" + profile.as_uri(),
                "--headless",
                "--nologo",
                "--nodefault",
                "--norestore",
                "--convert-to",
                target + ":" + filter_name,
                "--outdir",
                str(outputs),
                str(copied),
            ],
            timeout=options.timeout,
            env=env,
        )
        if status:
            raise UnreadableFile("LibreOffice conversion failed; no partial result was accepted.")
        converted = outputs / ("source." + target)
        _validate_package(converted, required, options)
        if _read_source(path, options.max_input_bytes) != source:
            raise UnreadableFile("Legacy Office source changed during conversion.")
        content = converted.read_bytes()
        derived_hash = hashlib.sha256(content).hexdigest()
        retained = store is not None
        if retained:
            held = store.put(
                content,
                MIME[target],
                reference={
                    "source_sha256": source_hash,
                    "input_format": suffix[1:],
                    "output_format": target,
                    "converter_version": version,
                    "rule": "office-conversion/1",
                },
            )
            if held != derived_hash:
                raise UnreadableFile("Converted artifact store returned an inconsistent hash.")
        provenance = ConversionLocator(
            suffix[1:],
            target,
            source_hash,
            derived_hash,
            "LibreOffice",
            version,
            filter_name,
            retained,
        )
        yield ConvertedOffice(converted, provenance)
        if _read_source(path, options.max_input_bytes) != source:
            raise UnreadableFile("Legacy Office source changed while its derivative was read.")


def read_legacy(path, reader, *, store=None, options=ConversionOptions()):
    """Conversion is a typed provenance bridge, never a cell/paragraph fact."""
    with converted_office(path, store=store, options=options) as converted:
        records = list(reader(converted.path))
        if not records:
            raise UnreadableFile("Converted Office document has no readable observations.")
        for record in records:
            yield replace(record, conversion=converted.provenance)
