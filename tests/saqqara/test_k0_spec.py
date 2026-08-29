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
  3. the package presents exactly one kernel to the registry, so the internal
     layers cannot register themselves as independent kernels by accident;
  4. nothing in the package or its tests trips the repository guard, and the
     guard is observed to fire on a planted violation — a guard that never
     fires proves nothing about the files it passed.

The counts below are frozen deliberately. Adding a proposition without deciding
which gate carries it should fail here, loudly.
"""

from __future__ import annotations

import re
import subprocess
import sys
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
FROZEN_COUNTS = {"K1": 9, "K2": 24, "K3": 71, "K4": 2, "K6": 16}

_ROW = re.compile(r"^\|\s*(K[1-6])\.(\d+)\s*\|(.+?)\|(.+?)\|(.+?)\|\s*$")
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


# --------------------------------------------------- 3. one kernel, not several

def test_k0_3_package_presents_exactly_one_kernel():
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
    assert found == ["ragix_kernels.saqqara.kernel.SaqqaraKernel"], found
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
