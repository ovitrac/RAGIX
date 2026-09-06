"""
Gate K0 — the specification is complete, self-consistent, and publishable.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

K0 is the gate that lets the later gates mean something. It proves four things
about the package before any of it computes anything:

  1. SPEC.md parses, its identifiers are well formed, unique and contiguous per
     gate, and every proposition states both what it claims and what would
     falsify it;
  2. the specification and the fixture generators agree IN BOTH DIRECTIONS — a
     fixture named by a proposition exists, and a generator nobody names is a
     finding, not a spare part;
  3. the package defines exactly the kernels it declares, named by their defining
     module, so an internal layer cannot register itself as an independent kernel
     by accident and every deliberate addition amends a reviewed list;
  4. nothing in the package or its tests trips the repository guard, and the
     guard is observed to fire on a planted violation — a guard that never
     fires proves nothing about the files it passed — including through the
     history mode, which reads what no other mode does: a commit message, and a
     blob that only an intermediate commit ever held.

The counts below are frozen deliberately. Adding a proposition without deciding
which gate carries it should fail here, loudly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ragix_kernels" / "saqqara"
SPEC = PACKAGE / "SPEC.md"
TESTS = ROOT / "tests" / "saqqara"
GUARD = ROOT / "tools" / "check_forbidden.py"

sys.path.insert(0, str(TESTS))
from generators import FIXTURES, FIXTURE_SUFFIX  # noqa: E402

#: gate -> number of propositions it must carry.
#:
#: Moved deliberately, never bypassed. Each change below shipped in the same edit
#: as the propositions it accounts for:
#:   K2 14 -> 16  the presentation and markdown readers (K2.15, K2.16), which P2
#:                implements and which no proposition previously justified;
#:   K3 41 -> 46  the typed-outline block K3.i (K3.42-K3.46), specified after the
#:                K0 crosswalk showed it was the one frozen, synthetic behaviour
#:                absent from this specification;
#:   K3 46 -> 51  the builder block K3.j (K3.47-K3.51). The analyzers speak of
#:                blocks and bands, the readers speak of observations, and the
#:                step between them was unspecified: unwritten, each analyzer
#:                would have re-invented it.
#:   K7 16 -> 19  K7.17 and K7.18, the two baseline comparisons: a fusion that
#:                reproduced a lane would be indistinguishable from one that
#:                fused nothing, and only a comparison can say otherwise. K7.19,
#:                the secret discipline — five tests already exercised it under
#:                K7.14, whose text is about the shape of the packaged defaults
#:                and says nothing about secrets;
#:   K3 71 -> 72  K3.72, the address a chain rung was read from. The analyzer
#:                already resolved the covering cell to find the rung and then
#:                discarded where it was: a chain could be quoted but not
#:                followed, and a citation that cannot be followed back to a
#:                position is a claim about a document rather than a reading of
#:                one;
#:   K1 8  -> 9   K1.9, added with the one deliberate reopening of the P1
#:                interface: a slide and its speaker notes shared a coordinate,
#:                so a locator was not a unique address inside its own format;
#:   K3 51 -> 58  K3.h grew from 2 propositions to 9. Two placeholders could not
#:                state eight channels, a recurrence filter, a printed-contents
#:                refusal, an ordered gauntlet with counted reasons, and an
#:                ancestry chain. The later blocks were renumbered rather than
#:                appended out of order: nothing is published yet, so the cost is
#:                one edit now against a broken numbering forever.
#:   K2 16 -> 17  K2.17, after the corpus parity showed a cell holding an empty
#:                string reading as content: it counted as a value in
#:                segmentation and typing, invisibly.
#:   K2 17 -> 18  K2.18, after the corpus parity found tables on slides being
#:                passed over entirely: a table shape has no text frame, and the
#:                reader walked text frames.
#:   K2 18 -> 22  the fact vocabularies (K2.19-K2.22). Measuring what the readers
#:                emit, kind by kind, found five readers declaring one vocabulary
#:                each and emitting sixteen: only `cell` was declared in the grid
#:                readers, only `page` in the laid-out one, and everything else --
#:                paragraphs, markers, tables, slides, shapes, notes, text runs,
#:                borders, sheets, front matter -- was guarded by nothing. Four
#:                propositions rather than one because four different things are
#:                claimed: that a vocabulary is declared per kind (K2.19); that
#:                declaration and emission are checked against each other in both
#:                directions (K2.20), which is what a single flat set could not do
#:                and which is how `kind_hint` survived, declared and produced by
#:                nothing; that a vocabulary whose names come from the document is
#:                declared open rather than left undeclared (K2.21); and that the
#:                grid kinds share one vocabulary across two readers (K2.22)
#:                instead of two copies free to drift.
#:   K3 58 -> 67  the K3.k block: headings a document only shows, by size. The
#:                gap it closes is the largest single row of the corpus parity,
#:                and measurement put the mass on the laid-out documents -- the
#:                old side promotes 3450 blocks there by size tier, against 164
#:                by typed label on flat documents, which K3.i already covers.
#:                Nine propositions because the method has nine independent ways
#:                to be wrong, and one of them -- line assembly (K3.59) -- is a
#:                step the old side never needed: it read blocks, a reader here
#:                emits one observation per text-showing operation, two and a
#:                half of them per line on the corpus. A rule applied to those
#:                fragments would cut a heading into pieces and then count each
#:                piece as separate evidence for its own tier.
#:   K2 22 -> 23  K2.23: the word-processing reader emits a bold FRACTION and a
#:                size, replacing a boolean. Measurement is why. The old kernel's
#:                bold-dominant rule reads a fraction >= 0.9 and has promoted
#:                nothing, ever, because the reader feeding it emitted no such
#:                fact -- a rule and its input written past each other. Re-
#:                expressing the rule faithfully meant re-expressing nothing, so
#:                the lead ruled to follow the intent instead, and the intent
#:                needs the fact. On the corpus, 229 heading-shaped paragraphs are
#:                partly bold: a boolean promotes every one of them.
#:   K3 67 -> 70  K3.68-K3.70: the weight branch of K3.k, and what separates it
#:                from the size branch. Three propositions because three distinct
#:                claims: that the two signals are tried in a declared order and
#:                the trace says which decided (K3.68); that weight is flat, since
#:                bold does not rank and cannot say how deep a heading sits
#:                (K3.69); and that weight means dominance rather than presence,
#:                measured against the boolean baseline in both directions (K3.70).
#:   K2 23 -> 24  K2.24, after the corpus parity was traced to its cause: the
#:                laid-out reader reported the `Tf` operand as the font size and
#:                ignored both matrices that scale it. Measured, 26 of the 32
#:                documents finding fewer headings than the old side disagreed
#:                with it about the body size -- 9 with every size collapsed onto
#:                one constant, 14 with every size inflated by a single factor.
#:                No analyzer proposition was false: the thresholds were applied
#:                to the wrong measurements, which is why this lands in K2 and
#:                K3 does not move.
#:   K3 70 -> 71  K3.71: a declared outline suppresses size inference. Measured on
#:                the reference corpus: five documents carry a native table of
#:                contents, both kernels read it identically -- 76 entries against
#:                76, 15 against 15, 1, 1, and 24 against 24 -- and the previous
#:                kernel routes those documents to its bookmark reader, which never
#:                infers by size. This analyzer inferred anyway and added 88
#:                headings beside a declaration that needed none. An inference set
#:                beside a declaration cannot corroborate it: it agrees redundantly
#:                or it disagrees, and the second is worse than the first is useful.
#:   K6 0  -> 6   the objects layer opens (K6.1-K6.6): the asset store, and images
#:                read per placement. `figure` and `caption` have been registered
#:                kinds with no producer since P1 -- measured, 119 documents where
#:                the previous kernel emits a figure and this one emits none.
#:
#:                The signed specification carries FIFTEEN propositions for this
#:                gate; six are declared here because a proposition is declared in
#:                the same edit as the fixture that exercises it and the code that
#:                answers it. K0 requires every fixture a proposition names to
#:                exist and to build, so declaring all fifteen now would mean
#:                writing nine fixtures for code that does not exist -- fixtures no
#:                failing test has ever refused, which is how a fixture comes to
#:                test nothing while appearing to pass. Two did exactly that
#:                earlier in this work. The remaining nine land with their steps:
#:                office readers, caption binding, vector regions, routing, skips.
#:   K6 6  -> 7   K6.7, after the first parity of the objects layer. The layer
#:                closed 108 of the 119 documents where the previous kernel found
#:                figures and this one found none -- and left a shortfall of 1 126
#:                placements over 24 documents that nothing could account for.
#:                Nothing could, because the reader dropped an image it failed to
#:                decode WITHOUT COUNTING IT: two bare handlers, written in the
#:                step before, made the reader's own failures invisible to the
#:                measurement meant to find them. Counted drops are doctrine here;
#:                this proposition is the doctrine made checkable for objects.
#:   K6 7  -> 8   K6.8, the office readers. Three of the five formats store their
#:                pictures as parts, so `figure` acquires a producer in three more
#:                readers with no renderer and no new dependency -- which is why
#:                this step comes before the one that rasterises anything. The
#:                proposition is about the SHARED vocabulary: one set held in the
#:                contract, four readers pointing at it, and the addressing that
#:                differs per format kept in the typed locator where it belongs.
#:   K4 2  -> 3   K4.3, after the demo corpus met the envelope. `summarize` cast
#:                every trace's `abstained` with `int(...)`; the analyzers publish
#:                it in four shapes, and `grid_tables` publishes a list. An empty
#:                list is falsy, so `or 0` hid the mismatch until a document
#:                actually abstained -- three files of 600 did, and the kernel
#:                lost 570 successfully read documents to a TypeError raised while
#:                writing one line of prose about them. The proposition is about
#:                the counting, not about the error handling: `Kernel.run`
#:                catching and reporting is fail-closed and stays.
#:   K7 19 -> 20  K7.20, measured before it was written. The ollama backend's
#:                `embed_batch` was a loop of single requests, and on the executor
#:                the round trip IS the cost: 300 chunks of a real corpus embed at
#:                4.3/s one by one, 75.5/s in slices of 32 and 93.7/s in slices of
#:                128 -- and the vectors are identical, 0.0 maximum absolute
#:                difference against the per-text ones for both models probed. The
#:                proposition is about the refusals as much as the transport: a
#:                short answer cannot be matched to its inputs, and retrying one by
#:                one would hide a server that cannot batch behind a run twenty
#:                times slower.
FROZEN_COUNTS = {"K1": 10, "K2": 25, "K3": 72, "K4": 3, "K6": 20, "K7": 22}

_ROW = re.compile(r"^\|\s*(K[1-7])\.(\d+)\s*\|(.+?)\|(.+?)\|(.+?)\|\s*$")
_TICKED = re.compile(r"`([a-z0-9_']+)`")


class Proposition:
    __slots__ = ("gate", "index", "id", "claim", "fixtures", "falsifier")

    def __init__(self, gate, index, claim, fixtures, falsifier):
        self.gate = gate
        self.index = index
        self.id = f"{gate}.{index}"
        self.claim = claim
        self.fixtures = fixtures
        self.falsifier = falsifier


def parse_spec() -> list[Proposition]:
    out = []
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        gate, index, claim, fixtures, falsifier = m.groups()
        out.append(
            Proposition(
                gate,
                int(index),
                claim.strip(),
                _TICKED.findall(fixtures),
                falsifier.strip(),
            )
        )
    return out


@pytest.fixture(scope="module")
def props() -> list[Proposition]:
    parsed = parse_spec()
    assert parsed, "SPEC.md yielded no propositions — the parser or the file is wrong"
    return parsed


# ------------------------------------------------------------ 1. the spec itself

def test_k0_1_spec_exists_and_parses(props):
    assert SPEC.is_file()
    assert len(props) == sum(FROZEN_COUNTS.values())


def test_k0_1_every_fixture_declares_its_suffix():
    """A fixture writes one format, and which one is part of registering it.

    Without this, a new fixture is written under the default suffix and handed to
    whichever reader claims it — which fails, loudly but unhelpfully, somewhere far
    from the omission.
    """
    undeclared = sorted(set(FIXTURES) - set(FIXTURE_SUFFIX))
    orphaned = sorted(set(FIXTURE_SUFFIX) - set(FIXTURES))
    assert not undeclared, f"registered with no declared suffix: {undeclared}"
    assert not orphaned, f"suffix declared for no fixture: {orphaned}"


def test_k0_1_identifiers_unique(props):
    ids = [p.id for p in props]
    assert len(ids) == len(set(ids)), "duplicate proposition identifier"


@pytest.mark.parametrize("gate", sorted(FROZEN_COUNTS))
def test_k0_1_gate_counts_are_frozen(props, gate):
    got = [p for p in props if p.gate == gate]
    assert len(got) == FROZEN_COUNTS[gate], (
        f"{gate} carries {len(got)} propositions, frozen at {FROZEN_COUNTS[gate]}: "
        "decide which gate owns the change before shipping it"
    )


@pytest.mark.parametrize("gate", sorted(FROZEN_COUNTS))
def test_k0_1_indices_are_contiguous(props, gate):
    got = sorted(p.index for p in props if p.gate == gate)
    assert got == list(range(1, len(got) + 1)), f"{gate} indices are not contiguous: {got}"


def test_k0_1_every_proposition_is_falsifiable(props):
    """A claim with no stated falsifier is a wish, not a specification."""
    for p in props:
        assert len(p.claim) > 30, f"{p.id}: claim too thin to test"
        assert len(p.falsifier) > 10, f"{p.id}: no stated falsifier"
        assert p.fixtures, f"{p.id}: names no fixture"


# ------------------------------------------- 2. spec and generators agree, both ways

def test_k0_2_every_named_fixture_has_a_generator(props):
    named = {f for p in props for f in p.fixtures}
    missing = sorted(named - set(FIXTURES))
    assert not missing, f"named in SPEC.md, absent from the generators: {missing}"


def test_k0_2_every_generator_is_named_by_the_spec(props):
    named = {f for p in props for f in p.fixtures}
    orphan = sorted(set(FIXTURES) - named)
    assert not orphan, f"generators no proposition names: {orphan}"


def test_k0_2_generators_either_build_or_refuse(tmp_path):
    """No generator may quietly return nothing.

    A stub must raise; an implemented generator must leave a non-empty file
    behind. The failure this forbids is the middle case — a generator that
    returns a path to an empty file, letting a later gate pass on no content
    at all.
    """
    for name, build in FIXTURES.items():
        target = tmp_path / name
        if getattr(build, "pending", False):
            with pytest.raises(NotImplementedError):
                build(target)
        else:
            out = build(target)
            assert out.is_file(), f"{name}: generator returned no file"
            assert out.stat().st_size > 0, f"{name}: generator produced an empty file"


# --------------------------------------------------- 3. the kernels are the pinned list

#: Every Kernel subclass this package defines, by the module that DEFINES it.
#:
#: This is an exact list and not a count, deliberately. The registry discovers
#: kernels by walking the package, so a module that grows a Kernel subclass becomes
#: an independently registered kernel with no other ceremony — which is exactly the
#: kind of change that should never arrive as a side effect. Pinning the paths means
#: every kernel added, moved or removed amends this line, in the commit that does it,
#: under review. The edit is the point; it is not an obstacle to the edit.
#:
#: A re-export does not appear here: the walk records a class only under the module
#: whose `__module__` it answers to, so `kernel.py` re-exporting SaqqaraKernel is
#: invisible to this gate, and compatibility shims cost nothing.
DECLARED_KERNELS = [
    "ragix_kernels.saqqara.kernels.saqqara_index.SaqqaraIndexKernel",
    "ragix_kernels.saqqara.kernels.saqqara_run.SaqqaraKernel",
]


def test_k0_3_package_defines_exactly_the_declared_kernels():
    from ragix_kernels.base import Kernel
    import importlib
    import pkgutil

    found = []
    package = importlib.import_module("ragix_kernels.saqqara")
    for _, module_name, _ in pkgutil.walk_packages(
        [str(PACKAGE)], prefix="ragix_kernels.saqqara."
    ):
        module = importlib.import_module(module_name)
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Kernel) and attr is not Kernel:
                if attr.__module__ == module_name:
                    found.append(f"{module_name}.{attr.__name__}")
    assert found == DECLARED_KERNELS, found
    assert package.__doc__, "the family package must document itself"


def test_k0_3_kernel_refuses_bad_input_rather_than_returning_nothing(tmp_path):
    """The failure to guard against is a plausible empty success.

    A kernel handed no source that answered with zero documents would be
    indistinguishable from one handed an empty folder. It has to refuse.
    """
    from ragix_kernels.base import KernelInput
    from ragix_kernels.saqqara.kernel import SaqqaraKernel

    kernel = SaqqaraKernel()
    errors = kernel.validate_input(KernelInput(workspace=tmp_path, config={}))
    assert any("source.path" in error for error in errors)

    output = kernel.run(KernelInput(workspace=tmp_path, config={}))
    assert output.success is False
    assert output.errors, "a refusal says why"


# ------------------------------------------------------- 4. the guard, observed

def _run_guard(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GUARD), *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_k0_4_guard_is_installed():
    assert GUARD.is_file(), "tools/check_forbidden.py is missing"
    hook = ROOT / ".git" / "hooks" / "pre-commit"
    if not hook.exists():
        pytest.skip("no .git/hooks in this checkout (packaged copy or CI export)")
    assert "check_forbidden.py" in hook.read_text(encoding="utf-8")


def test_k0_4_guard_fires_on_a_planted_violation():
    """A guard never seen to fire says nothing about the files it passed."""
    done = _run_guard("--selftest")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FAIL" not in done.stdout, done.stdout


def test_k0_4_package_and_tests_are_clean():
    targets = [
        str(p.relative_to(ROOT))
        for p in (*PACKAGE.rglob("*"), *TESTS.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    assert targets, "nothing to scan — the package is missing"
    done = _run_guard(*targets)
    assert done.returncode == 0, done.stderr


def test_k0_4_no_committed_binary_fixture():
    """Fixtures are generated by code; a binary under tests/saqqara is an accident."""
    offenders = [
        str(p.relative_to(ROOT))
        for p in TESTS.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and b"\x00" in p.read_bytes()
    ]
    assert not offenders, offenders


# ------------------------------------- 4b. the history mode, which reads what no other does

# The working-tree scan reads the tip and the hook reads what is staged. Neither
# reads a commit message at all, and neither sees a blob that existed only in an
# intermediate commit — yet publishing a branch publishes both. `scan_history`
# exists for that, CI runs it on every pull request, and until these tests it was
# the one part of the guard nothing exercised: a table that held the wrong tokens
# and a mechanism that had quietly stopped working looked identical from outside.
#
# The planted token is generated per test and injected into the table at run time. The
# real rows are not spelled anywhere — not in clear, not reconstructed from
# character codes — because writing them would be the leak the table exists to
# prevent; that each real row matches the token it means is proved by a control
# run against real history, recorded outside this repository.

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Gate", "GIT_AUTHOR_EMAIL": "gate@example.invalid",
    "GIT_COMMITTER_NAME": "Gate", "GIT_COMMITTER_EMAIL": "gate@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
}


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True,
        env={**os.environ, **_GIT_ENV},
    )
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    (root / "kept.py").write_text("value = 1\n", encoding="utf-8")
    return root


def _guard_module():
    """The guard imported, so the table can be extended for one test only."""
    spec = importlib.util.spec_from_file_location("check_forbidden_under_test", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plant_token(guard, monkeypatch) -> str:
    """A row that exists for one test, made the way a real row is made.

    The token is generated, never chosen: a word picked as a placeholder is a word
    that can turn out to be somebody's identifier, and it would then sit in a public
    test file forever. Thirty-two lowercase hex characters are one token under the
    split rule, and the digest goes in through the same hashing path the real rows
    use, so what the test exercises is the row format rather than a shortcut around it.
    """
    token = uuid.uuid4().hex
    digest = hashlib.sha256(token.lower().encode()).hexdigest()
    monkeypatch.setitem(guard.FORBIDDEN_TOKEN_HASHES, digest, "planted-token")
    return token


def test_k0_4_history_reads_a_commit_message_no_other_mode_reads(tmp_path, monkeypatch):
    """A name in a message was never in a file, and is published all the same."""
    guard = _guard_module()
    planted = _plant_token(guard, monkeypatch)

    repo = _repo(tmp_path / "messages")
    base = _commit(repo, "base")
    (repo / "kept.py").write_text("value = 2\n", encoding="utf-8")
    tip = _commit(repo, f"a change measured on {planted}")

    assert guard.scan(["kept.py"], repo) == [], "the tree is clean — that is the point"

    findings, blobs, messages = guard.scan_history(f"{base}..{tip}", repo)
    assert messages == 1, messages
    assert [f for f in findings if ":message:" in f and "planted-token" in f], findings


def test_k0_4_history_reads_a_blob_only_an_intermediate_commit_held(tmp_path, monkeypatch):
    """Removing the line in a later commit does not remove it from what a push publishes."""
    guard = _guard_module()
    planted = _plant_token(guard, monkeypatch)

    repo = _repo(tmp_path / "blobs")
    base = _commit(repo, "base")
    (repo / "notes.md").write_text(f"read from {planted}\n", encoding="utf-8")
    _commit(repo, "a note")
    (repo / "notes.md").write_text("read from the corpus\n", encoding="utf-8")
    tip = _commit(repo, "the note, without the name")

    assert guard.scan(["notes.md"], repo) == [], "the tip no longer carries it"

    findings, blobs, messages = guard.scan_history(f"{base}..{tip}", repo)
    assert [f for f in findings if "notes.md" in f and ":message:" not in f], findings


def test_k0_4_history_mode_exits_clean_on_a_clean_range_and_reports_what_it_read(tmp_path):
    """The count line is read before anything else, so it is part of the contract."""
    repo = _repo(tmp_path / "clean")
    base = _commit(repo, "base")
    (repo / "kept.py").write_text("value = 2\n", encoding="utf-8")
    tip = _commit(repo, "a change")

    done = subprocess.run(
        [sys.executable, str(GUARD), "--history", f"{base}..{tip}"],
        cwd=repo, capture_output=True, text=True, env={**os.environ, **_GIT_ENV},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "blob revisions" in done.stdout and "commit messages" in done.stdout, done.stdout


def test_k0_4_history_mode_refuses_through_the_command_line(tmp_path):
    """The whole path, exit code included — scoped rule, so no token is needed at all."""
    repo = _repo(tmp_path / "refused")
    base = _commit(repo, "base")
    scoped = repo / "ragix_kernels" / "saqqara" / "endpoint.py"
    scoped.parent.mkdir(parents=True)
    # Assembled, not written: this file lives under a scoped prefix, so the literal
    # would trip the very rule the test plants — and `test_k0_4_package_and_tests_are_clean`
    # is what would report it. Shared address space (RFC 6598) belongs to no one.
    address = ".".join(("100", "64", "0", "1"))
    scoped.write_text(f"endpoint = 'http://{address}:8080'\n", encoding="utf-8")
    _commit(repo, "an address under scope")
    scoped.write_text("endpoint = 'http://127.0.0.1:11434'\n", encoding="utf-8")
    tip = _commit(repo, "the address, made local")

    done = subprocess.run(
        [sys.executable, str(GUARD), "--history", f"{base}..{tip}"],
        cwd=repo, capture_output=True, text=True, env={**os.environ, **_GIT_ENV},
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "routable-address" in done.stderr, done.stderr


#: Rows on the tier-1 table. Frozen: a row is added with a control that observed it
#: fire against real history, and a row nobody has seen fire is a row nobody has
#: tested. Changing this number is the amendment; the control is the evidence.
FROZEN_TIER1_ROWS = 6


def test_k0_4_the_forbidden_table_is_frozen_and_well_formed():
    guard = _guard_module()
    table = guard.FORBIDDEN_TOKEN_HASHES
    assert len(table) == FROZEN_TIER1_ROWS, (
        f"the table carries {len(table)} rows, frozen at {FROZEN_TIER1_ROWS}: "
        "a row is added deliberately, with the control that saw it fire"
    )
    malformed = [k for k in table if not re.fullmatch(r"[0-9a-f]{64}", k)]
    assert not malformed, f"not SHA-256 digests of a lowercased token: {malformed}"
    assert len(set(table.values())) == len(table), "two rows share a name; a finding names one rule"



# --------------------------------------------------------------------------- k0.5
# The prose that describes the specification is not the specification, and nothing
# gated it. Three statements — two in README.md, one in SPEC.md — went on claiming
# 85 and 65 propositions across "K1-K4" while the frozen counts said 126 across
# five gates, because every check above reads the proposition TABLE and none reads
# the sentences around it. A reader opens the README first.

README = PACKAGE / "README.md"
KOAS_DOC = ROOT / "docs" / "KOAS_SAQQARA.md"

#: A count claimed about propositions: "85 falsifiable propositions", "65 propositions".
_CLAIMED_COUNT = re.compile(r"(\d+)\s+(?:[a-z-]+\s+){0,3}propositions?\b", re.I)

#: A gate range presented as the enumeration of gates: "K1-K4", "K1–K4".
_GATE_RANGE = re.compile(r"\bK(\d)\s*[-–]\s*K(\d)\b")

#: Any single gate identifier, so a range may be completed by naming the rest.
_GATE_TOKEN = re.compile(r"\bK(\d)\b")


def _prose_files() -> list[Path]:
    """Every file whose prose describes the specification and can contradict it.

    KOAS_SAQQARA.md joined this list the day it was written, not later: it states
    the gate table and the proposition count, which is exactly what had already
    gone stale twice in README.md and once in SPEC.md. A third unwatched copy of a
    number is a third chance to publish the wrong one.
    """
    return [p for p in (README, SPEC, KOAS_DOC) if p.is_file()]


def test_k0_5_prose_proposition_counts_match_the_frozen_total():
    """A number claimed about propositions is the frozen total, or it is wrong.

    Falsified by: any sentence in README.md or SPEC.md claiming a proposition
    count that differs from sum(FROZEN_COUNTS).
    """
    total = sum(FROZEN_COUNTS.values())
    wrong = []
    for path in _prose_files():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for claimed in _CLAIMED_COUNT.findall(line):
                if int(claimed) != total:
                    wrong.append(f"{path.name}:{n}: claims {claimed}, frozen total is {total}")
    assert not wrong, "prose contradicts FROZEN_COUNTS:\n  " + "\n  ".join(wrong)


def test_k0_5_prose_gate_ranges_name_every_gate():
    """A line that enumerates the gates enumerates ALL of them.

    A contiguous range cannot express the real set once a gate is skipped, so the
    line must complete the range by naming the remainder. Falsified by: a line
    mentioning a gate range whose gates, together with any other gate named on the
    same line, are not exactly the frozen gate set.
    """
    expected = {int(g[1:]) for g in FROZEN_COUNTS}
    wrong = []
    for path in _prose_files():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "gate" not in line.lower():
                continue
            ranges = _GATE_RANGE.findall(line)
            if not ranges:
                continue
            named = {int(g) for g in _GATE_TOKEN.findall(line)}
            for lo, hi in ranges:
                named |= set(range(int(lo), int(hi) + 1))
            if named != expected:
                wrong.append(
                    f"{path.name}:{n}: names {sorted(named)}, frozen gates are {sorted(expected)}"
                )
    assert not wrong, "prose contradicts the frozen gate set:\n  " + "\n  ".join(wrong)
