#!/usr/bin/env python3
"""Pre-commit guard: refuse content that must not enter this public repository.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Two tiers, deliberately different in scope.

TIER 1 — repository-wide, hashed tokens
    A small set of identifiers must never appear anywhere in this repository.
    They are matched by SHA-256 of the lowercased token, not in clear text.

    Why hashed: a guard that lists the strings it forbids publishes them. This
    file is versioned in a public repository; writing the words here would be
    the very leak the guard exists to prevent, and would make the repository
    discoverable by a text search for them.

    This is obfuscation, NOT secrecy. A short token can be recovered by anyone
    who guesses it and hashes their guess. What hashing buys is real but narrow:
    the words are not published, not indexed, and not found by searching. That
    is the actual failure mode this guard addresses — accidental publication,
    not a determined adversary.

    Matching is by exact token: each line is lowercased and split on runs of
    non-alphanumeric characters, so `FOO_BAR` yields `foo` and `bar`. A token
    embedded inside a longer alphanumeric run is NOT matched.

TIER 2 — scoped patterns, clear text
    Patterns that are meaningless out of context and safe to publish, applied
    only under the paths in SCOPED_PREFIXES. They are scoped because the rest of
    this repository has legitimate uses for them (network kernels handle
    addresses; documentation discusses multi-tenancy).

An optional external denylist may be supplied through RAGIX_DENYLIST, one
regular expression per line, `#` for comments. It applies to TIER 2 scope only.
If the variable is set and the file cannot be read, the guard FAILS CLOSED.
If the variable is unset, the guard runs on its built-in rules alone: fixture
text is proved by a positive property (every string comes from the generator),
so no list of names is required for correctness.

Exemption: a line whose content is an `Author:` attribution header is exempt
from TIER 2 organisation matching. Attribution is required by the project
contract; without this exemption the guard reports dozens of false positives
per commit, is switched off within a week, and protects nothing.

Usage
    check_forbidden.py --staged        # what the pre-commit hook runs
    check_forbidden.py PATH [PATH ...] # explicit paths, for CI and tests
    check_forbidden.py --selftest      # prove the guard catches a planted hit

The hook is local to one clone: it does not cover a second clone, a contributor, or a pull
request, and `--no-verify` steps over it. `.github/workflows/guard.yml` runs the same checks where
none of that is possible, and that workflow — not the hook — is what makes the guard real.

Exit codes
    0  nothing found
    1  at least one finding (printed as file:line: rule <name>)
    2  the guard could not run (fail closed)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------- tier 1

#: SHA-256 of the lowercased tokens that must never appear anywhere.
#: Regenerate with:  python3 -c "import hashlib,sys;[print(hashlib.sha256(t.strip().lower().encode()).hexdigest()) for t in sys.stdin]"
FORBIDDEN_TOKEN_HASHES: dict[str, str] = {
    "f6f9d150ceae03aae95477223a9d0df395cb27755fef43971d0998977db4c298": "origin-identifier-1",
    "785757cb86310e56afe087c0bb461abcf865dcff8150a1999a64c8471df31be0": "origin-identifier-2",
    "fdd00fc797ddde5964c08550170de1fb1561cc525c01e063c9f5f82b39afdf4e": "origin-revision",
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# --------------------------------------------------------------------- tier 2

#: Paths under which the scoped rules apply.
SCOPED_PREFIXES = ("ragix_kernels/saqqara/", "tests/saqqara/")

#: Addresses that carry no deployment information.
_ALLOWED_IPS = re.compile(
    r"^(127\.\d{1,3}\.\d{1,3}\.\d{1,3}|0\.0\.0\.0|255\.255\.255\.255"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})$"
)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

SCOPED_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("multi-tenancy-vocabulary", re.compile(r"\btenants?\b", re.I)),
    ("ssh-tunnel-with-user", re.compile(r"ssh\s+.*-L\s+\S+\s+\S+@", re.I)),
    ("host-or-machine-name", re.compile(r"\b[a-z0-9-]+\.(?:local|internal|lan)\b", re.I)),
]

#: Attribution headers are exempt from the organisation-name rules of tier 2.
_AUTHOR_LINE = re.compile(r"^\s*(#|//|\*|\"\"\")?\s*Author\s*:", re.I)


def _tier1_hits(line: str) -> list[str]:
    """Token hashes of `line` that are on the forbidden list."""
    found = []
    for token in _TOKEN_SPLIT.split(line.lower()):
        if not token:
            continue
        digest = hashlib.sha256(token.encode()).hexdigest()
        name = FORBIDDEN_TOKEN_HASHES.get(digest)
        if name:
            found.append(name)
    return found


def _tier2_hits(line: str, extra: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    """Scoped-rule names matching `line`."""
    if _AUTHOR_LINE.match(line):
        return []
    found = [name for name, rx in (*SCOPED_RULES, *extra) if rx.search(line)]
    for addr in _IPV4.findall(line):
        if not _ALLOWED_IPS.match(addr):
            found.append("routable-address")
            break
    return found


def _load_extra_rules() -> list[tuple[str, re.Pattern[str]]]:
    """Read RAGIX_DENYLIST if declared. Fail closed when declared but unusable."""
    ref = os.environ.get("RAGIX_DENYLIST")
    if not ref:
        return []
    path = Path(ref).expanduser()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"check_forbidden: RAGIX_DENYLIST is set but unreadable: {exc}", file=sys.stderr)
        raise SystemExit(2)
    rules = []
    for n, entry in enumerate(raw.splitlines(), 1):
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            continue
        try:
            rules.append((f"denylist:{n}", re.compile(entry, re.I)))
        except re.error as exc:
            print(f"check_forbidden: bad expression at {path}:{n}: {exc}", file=sys.stderr)
            raise SystemExit(2)
    return rules


def _is_scoped(rel: str) -> bool:
    return rel.startswith(SCOPED_PREFIXES)


def scan(paths: list[str], root: Path) -> list[str]:
    """Return one finding per offending line, formatted file:line: rule <name>."""
    extra = _load_extra_rules()
    findings: list[str] = []

    for rel in paths:
        target = root / rel
        if not target.is_file():
            continue
        try:
            data = target.read_bytes()
        except OSError as exc:
            findings.append(f"{rel}:0: rule unreadable-file ({exc})")
            continue

        if b"\x00" in data:
            if _is_scoped(rel):
                findings.append(
                    f"{rel}:0: rule binary-fixture "
                    "(fixtures are generated by code, never committed)"
                )
            continue

        text = data.decode("utf-8", errors="replace")
        scoped = _is_scoped(rel)
        for n, line in enumerate(text.splitlines(), 1):
            for name in _tier1_hits(line):
                findings.append(f"{rel}:{n}: rule {name}")
            if scoped:
                for name in _tier2_hits(line, extra):
                    findings.append(f"{rel}:{n}: rule {name}")

    return findings


def staged_paths(root: Path) -> list[str]:
    """Paths staged for commit, added or modified."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(out)


def selftest() -> int:
    """Prove the guard catches a planted hit. A guard never seen to fire is not a guard."""
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "ragix_kernels" / "saqqara").mkdir(parents=True)

        # tier 1: the token is reconstructed here, never written as a literal.
        token = "".join(chr(c) for c in (105, 110, 103, 101, 110, 116, 105, 115))
        planted = root / "ragix_kernels" / "saqqara" / "planted.py"
        planted.write_text(f"# forked from {token}-rag\n", encoding="utf-8")
        hits = scan(["ragix_kernels/saqqara/planted.py"], root)
        print(f"selftest tier1: {len(hits)} finding(s) — {'PASS' if hits else 'FAIL'}")
        ok &= bool(hits)

        # tier 2: routable address inside the scope
        planted.write_text("endpoint = 'http://203.0.113.9:8080'\n", encoding="utf-8")
        hits = scan(["ragix_kernels/saqqara/planted.py"], root)
        print(f"selftest tier2: {len(hits)} finding(s) — {'PASS' if hits else 'FAIL'}")
        ok &= bool(hits)

        # loopback stays allowed
        planted.write_text("endpoint = 'http://127.0.0.1:11434'\n", encoding="utf-8")
        hits = scan(["ragix_kernels/saqqara/planted.py"], root)
        print(f"selftest loopback: {len(hits)} finding(s) — {'PASS' if not hits else 'FAIL'}")
        ok &= not hits

        # attribution header stays allowed
        planted.write_text("# Author: Someone | Adservio | 2026\n", encoding="utf-8")
        hits = scan(["ragix_kernels/saqqara/planted.py"], root)
        print(f"selftest author-exemption: {len(hits)} finding(s) — {'PASS' if not hits else 'FAIL'}")
        ok &= not hits

        # scope really is a scope: tier 2 must not fire outside it
        other = root / "elsewhere.py"
        other.write_text("endpoint = 'http://203.0.113.9:8080'\n", encoding="utf-8")
        hits = scan(["elsewhere.py"], root)
        print(f"selftest scope: {len(hits)} finding(s) — {'PASS' if not hits else 'FAIL'}")
        ok &= not hits

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="paths to scan, relative to the repository root")
    ap.add_argument("--staged", action="store_true", help="scan what is staged for commit")
    ap.add_argument("--selftest", action="store_true", help="prove the guard fires")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        root = repo_root()
    except subprocess.CalledProcessError:
        print("check_forbidden: not inside a git repository", file=sys.stderr)
        return 2

    paths = staged_paths(root) if args.staged else args.paths
    if not paths:
        return 0

    findings = scan(paths, root)
    if not findings:
        return 0

    print("check_forbidden: refusing this content\n", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    print(
        "\nRule names are deliberately opaque; see tools/check_forbidden.py."
        "\nFix the content. Do not bypass with --no-verify.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
