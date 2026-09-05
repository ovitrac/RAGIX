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
    check_forbidden.py --history A..B  # every blob revision and commit message in a range

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
    # The three rows below name an engagement whose material is processed elsewhere
    # and never reaches this repository: the reference it is known by, the archive
    # it arrived as, and the label of a corpus that is not public. They were added
    # after all three passed the guard inside commit messages of this repository —
    # the history job read those messages and found nothing, because the table did
    # not hold them. A table is only as good as the classes someone thought to put
    # in it, and the class that leaks is the one nobody anticipated.
    "14f63bedabb240ef1822019f6e552ec097f8693eca604fb28f794b6073556bbf": "engagement-reference",
    "41857d30178bd00704253e36d16d9e3df130eb03eff9e1bdec81aa334238a551": "engagement-archive-id",
    "302869a9b3c448b95fddb41fb28e5ea3953ab2b3aead7073a9828bc7026d5f15": "engagement-corpus-label",
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# --------------------------------------------------------------------- tier 2

#: Paths under which the scoped rules apply.
#:
#: The `tender` prefixes are here BEFORE the family exists. A scope extended after
#: the code it covers has already been reviewed against the old scope, and nobody
#: notices the gap because nothing fires either way. Extending it against an empty
#: directory costs nothing and cannot be argued with later.
SCOPED_PREFIXES = (
    "ragix_kernels/saqqara/",
    "tests/saqqara/",
    "ragix_kernels/tender/",
    "tests/tender/",
    "examples/saqqara/",
)

#: Addresses that carry no deployment information.
#:
#: The last three are the RFC 5737 documentation ranges — TEST-NET-1, TEST-NET-2
#: and TEST-NET-3. They exist to be written down: an address from them cannot
#: identify a deployment, because it is reserved from ever being one. Refusing
#: them made the guard fire on demo network configuration, which teaches people
#: to write a realistic address instead — the opposite of what the rule wants.
_ALLOWED_IPS = re.compile(
    r"^(127\.\d{1,3}\.\d{1,3}\.\d{1,3}|0\.0\.0\.0|255\.255\.255\.255"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.0\.2\.\d{1,3}|198\.51\.100\.\d{1,3}|203\.0\.113\.\d{1,3})$"
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


def _scan_blob(label: str, data: bytes, rel: str, extra) -> list[str]:
    """Findings for one blob's bytes, whatever produced them.

    Shared by the working-tree scan and the history scan on purpose: two copies of
    this logic would drift, and the mode that drifted would be the one nobody was
    watching. `label` names the finding's location (a path, or a commit and path);
    `rel` is the repository path that decides tier-2 scope.
    """
    findings: list[str] = []
    if b"\x00" in data:
        if _is_scoped(rel):
            findings.append(
                f"{label}:0: rule binary-under-scope "
                "(every document here is generated by code, never committed)"
            )
        return findings

    scoped = _is_scoped(rel)
    for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        for name in _tier1_hits(line):
            findings.append(f"{label}:{n}: rule {name}")
        if scoped:
            for name in _tier2_hits(line, extra):
                findings.append(f"{label}:{n}: rule {name}")
    return findings


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
        findings.extend(_scan_blob(rel, data, rel, extra))

    return findings


def _git(root: Path, *args: str) -> bytes:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if done.returncode != 0:
        print(
            f"check_forbidden: git {' '.join(args)} failed: "
            f"{done.stderr.decode('utf-8', 'replace').strip()}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return done.stdout


def scan_history(rev_range: str, root: Path) -> tuple[list[str], int, int]:
    """Scan every blob revision and every commit message in `rev_range`.

    The working-tree scan reads the tip and the hook reads what is staged. Neither
    sees a blob that existed only in an intermediate commit, and neither reads a
    commit message at all — a name removed in a later commit is still in the
    history, and a name written in a message was never in a file. Publishing a
    branch publishes both.

    Every (blob, path) pair reachable in the range is read once, so an unchanged
    file costs nothing per commit. Tier 1 applies everywhere; tier 2 applies under
    SCOPED_PREFIXES, judged on the path the blob had at that revision.

    Returns (findings, blob revisions scanned, messages scanned).
    """
    extra = _load_extra_rules()
    findings: list[str] = []

    revs = _git(root, "rev-list", rev_range).decode().split()

    seen: set[tuple[str, str]] = set()
    for rev in revs:
        listing = _git(root, "ls-tree", "-r", "-z", rev).decode("utf-8", "replace")
        for entry in listing.split("\0"):
            if not entry:
                continue
            meta, _, rel = entry.partition("\t")
            parts = meta.split()
            if len(parts) != 3 or parts[1] != "blob":
                continue
            sha = parts[2]
            if (sha, rel) in seen:
                continue
            seen.add((sha, rel))
            findings.extend(
                _scan_blob(f"{rev[:9]}:{rel}", _git(root, "cat-file", "blob", sha), rel, extra)
            )

    for rev in revs:
        body = _git(root, "log", "-1", "--format=%B", rev).decode("utf-8", "replace")
        for n, line in enumerate(body.splitlines(), 1):
            for name in _tier1_hits(line):
                findings.append(f"{rev[:9]}:message:{n}: rule {name}")

    return findings, len(seen), len(revs)


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
    """Prove the guard catches a planted hit. A guard never seen to fire is not a guard.

    Every prefix in SCOPED_PREFIXES is exercised, rather than one representative
    prefix: a scope is added by editing a tuple, and a scope nobody has watched
    fire is indistinguishable from a typo in that tuple. Adding a prefix without
    proving it therefore cannot happen — the loop below covers whatever is declared.
    """
    import tempfile

    ok = True

    def check(label: str, findings: list[str], want: bool) -> None:
        nonlocal ok
        good = bool(findings) is want
        print(f"selftest {label}: {len(findings)} finding(s) — {'PASS' if good else 'FAIL'}")
        ok &= good

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # The token is reconstructed here, never written as a literal: this file is
        # public, and spelling the word would be the leak the guard exists to prevent.
        token = "".join(chr(c) for c in (105, 110, 103, 101, 110, 116, 105, 115))

        # ---- tier 1 is repository-wide, so prove it fires OUTSIDE any scope too.
        # Nothing else in this selftest would notice if tier 1 silently became scoped.
        loose = root / "anywhere.py"
        loose.write_text(f"# forked from {token}-rag\n", encoding="utf-8")
        check("tier1-unscoped", scan(["anywhere.py"], root), True)

        # ---- every declared scope, exercised in turn
        for prefix in SCOPED_PREFIXES:
            rel = f"{prefix}planted.py"
            planted = root / rel
            planted.parent.mkdir(parents=True, exist_ok=True)
            short = prefix.rstrip("/").replace("/", ".")

            planted.write_text(f"# forked from {token}-rag\n", encoding="utf-8")
            check(f"tier1[{short}]", scan([rel], root), True)

            # The positive control for the routable-address rule.
            #
            # It used a documentation address, which the allowlist now admits, so
            # the control went silent — a check that cannot fail, guarding a rule
            # nobody would notice breaking. It uses shared address space (RFC 6598,
            # carrier-grade NAT) instead, and the distinction is the point: a
            # documentation address is reserved from ever identifying a deployment,
            # while CGNAT space is used in real infrastructure, so an address from
            # it appearing in source CAN be a deployment fact. Refusing it is what
            # the rule is for. Writing it here reveals nothing: the range is shared
            # and belongs to no one.
            planted.write_text("endpoint = 'http://100.64.0.1:8080'\n", encoding="utf-8")
            check(f"tier2[{short}]", scan([rel], root), True)

            planted.write_text("endpoint = 'http://127.0.0.1:11434'\n", encoding="utf-8")
            check(f"loopback[{short}]", scan([rel], root), False)

            # The other half of the change: what the allowlist now admits must be
            # observed being admitted, or "documentation ranges are allowed" is a
            # claim resting on one rule not firing for some other reason.
            planted.write_text("endpoint = 'http://203.0.113.9:8080'\n", encoding="utf-8")
            check(f"doc-range[{short}]", scan([rel], root), False)

            planted.write_text("# Author: Someone | Adservio | 2026\n", encoding="utf-8")
            check(f"author-exemption[{short}]", scan([rel], root), False)

            # A committed document is refused wherever documents are generated.
            # examples/saqqara/ joined this list before it exists, for the reason
            # tender/ did: a rule is cheaper to prove against an empty directory
            # than to argue about after the first binary has landed. The scope is
            # that directory and not examples/, which already holds demo data from
            # an earlier release that these rules were never applied to.
            planted.write_bytes(b"%PDF-1.4\n\x00 binary payload\n")
            check(f"binary[{short}]", scan([rel], root), True)

        # ---- scope really is a scope: tier 2 must not fire outside it
        other = root / "elsewhere.py"
        other.write_text("endpoint = 'http://100.64.0.1:8080'\n", encoding="utf-8")
        check("scope", scan(["elsewhere.py"], root), False)

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="paths to scan, relative to the repository root")
    ap.add_argument("--staged", action="store_true", help="scan what is staged for commit")
    ap.add_argument("--selftest", action="store_true", help="prove the guard fires")
    ap.add_argument(
        "--history", metavar="BASE..HEAD",
        help="scan every blob revision and every commit message in a range",
    )
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    try:
        root = repo_root()
    except subprocess.CalledProcessError:
        print("check_forbidden: not inside a git repository", file=sys.stderr)
        return 2

    if args.history:
        findings, blobs, messages = scan_history(args.history, root)
        print(
            f"check_forbidden: {args.history} — "
            f"{blobs} blob revisions, {messages} commit messages, {len(findings)} finding(s)"
        )
        if not findings:
            return 0
        sys.stdout.flush()  # the count is the first thing read; keep it above the list
        print("\ncheck_forbidden: refusing this history\n", file=sys.stderr)
        for f in findings:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nA finding here is in the history, not only in the tip: removing the line"
            "\nin a new commit does not remove it from what a push would publish.",
            file=sys.stderr,
        )
        return 1

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
