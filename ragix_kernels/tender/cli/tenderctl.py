"""
tenderctl — CLI for the tender family.

Commands:
    probe   Report what a document store holds, without modifying it
    status  The same, as the configuration resolves it, without opening the store

Usage:
    python -m ragix_kernels.tender.cli.tenderctl probe -c tender.yaml
    python -m ragix_kernels.tender.cli.tenderctl status -c tender.yaml

`probe --json` prints exactly what the MCP tool returns. That is not a
convenience: two surfaces with two shapes disagree eventually, and the
disagreement is found by whoever trusted the wrong one.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from ragix_kernels.base import KernelInput

from ..config import load_config
from ..kernels.tender_probe import TenderProbeKernel

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return _c("1", text)


def _green(text: str) -> str:
    return _c("32", text)


def _red(text: str) -> str:
    return _c("31", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _resolved_store(config) -> Path:
    return Path(config.get("store.path"))


def cmd_probe(args) -> int:
    config = load_config(args.config or None)
    store = _resolved_store(config)

    if not store.is_file():
        print(_red(f"no document store at {store}"), file=sys.stderr)
        print("this family reads a store; build one with saqqaractl index",
              file=sys.stderr)
        return 2

    output = TenderProbeKernel().run(KernelInput(
        workspace=store.parent, config={"config": args.config} if args.config else {},
        dependencies={"document_store": store}))

    if not output.success:
        print(_red(output.summary), file=sys.stderr)
        for error in output.errors or []:
            print(_red(f"error: {error}"), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"probe": output.data["probe"]}, indent=2, sort_keys=True))
        return 0

    probe = output.data["probe"]
    print(_green(output.summary))
    print(f"  {'store':10s} {probe['store_path']}")
    print(f"  {'documents':10s} {probe['documents']}")
    print(f"  {'chunks':10s} {probe['chunks']}")
    line = f"  {'dense':10s} {probe['dense']}"
    print(line if probe["dense_enabled"] else _yellow(line))
    return 0


def cmd_status(args) -> int:
    """What the configuration resolves to, without opening anything.

    Useful precisely when the probe refuses: it answers "which store did you
    mean", which is the question a refusal raises.
    """
    config = load_config(args.config or None)
    store = _resolved_store(config)
    print(f"{_bold('store')}  {store}")
    print(f"{_bold('exists')} {store.is_file()}")
    if args.json:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tenderctl",
        description="Structured response preparation over document stores.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="report what a document store holds")
    p_probe.add_argument("-c", "--config", default=None, help="path to a tender.yaml")
    p_probe.add_argument("--json", action="store_true", help="emit the result as JSON")
    p_probe.add_argument("-v", "--verbose", action="store_true")
    p_probe.set_defaults(func=cmd_probe)

    p_status = sub.add_parser("status", help="what the configuration resolves to")
    p_status.add_argument("-c", "--config", default=None, help="path to a tender.yaml")
    p_status.add_argument("--json", action="store_true", help="emit the configuration")
    p_status.add_argument("-v", "--verbose", action="store_true")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
