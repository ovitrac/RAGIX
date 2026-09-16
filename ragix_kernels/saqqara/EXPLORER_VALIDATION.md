# Explorer v0 — synthetic validation, 2026-09-16

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The deterministic Explorer chain runs through the typed library, CLI, four KOAS
adapters and an Orchestrator manifest. The synthetic replay digests agree on
x86_64 and aarch64. Optional model classification is schema-safe on the measured
packet but semantically unreliable; it remains disabled by default.

Implementation source: `dcc1399f84445cd76c62735048ee2035c62e038e`.
Final regression additions: `cddd455`. Branch: `feat/document-explorer`.
No merge or public push is part of this validation.

## Measured gates

| Boundary | Evidence | Result and limit |
|---|---|---|
| E0 prerequisites | `tests/harvest/test_explorer_prerequisites.py`, existing quantity/binding/field-view suites | Interval, unit, grouping, label-number, baseline-spread and nested numeric-smuggling controls pass. Scientific alphanumeric names remain admissible. |
| E1 census | `test_census_all_planted_category_multisets`, exact-span and textless-page controls | Every planted occurrence and category multiset matches on the generated fixture: no missed or extra planted patterns. French and English controls pass. This is fixture precision/recall, not a population estimate. |
| E2 profile | Mutation, evidence, schema and append-only-review tests | X1–X6 change the corresponding semantic profile values. Unsupported notation stays UNKNOWN. Evidence and template hashes may change. |
| E3 classification | Generated two-page packet, four candidate ids; results below | Model-off parity, id-only output, exhaustive accounting, refusal caching and schema checks pass. No semantic accuracy threshold was imposed. |
| E4 readers | Mutation, range, revision, forged/stale-source and nondefault-geometry tests | Planted fields retain both targets and all section members. Ranges remain unexpanded relations. Unknown required notation produces a counted finding. |
| E5 report | Negative-record schema, HTML injection and source-column regressions | Missing scope/rule/positive count is refused. Numbers belong to addressable records; source strings are escaped. Source columns named like run metadata remain intact. |
| E6 adapters/replay | CLI/library equality, kernel chain, Orchestrator/cache test with socket guard; cross-architecture gate tool | Same semantic report; no deterministic-chain network call. Eight generated cases, including a PDF round trip, match across architectures. |

Full command on each host:

```bash
python -m pytest tests/harvest tests/saqqara -q -p no:cacheprovider
```

- x86_64: **1,200 passed, 32 skipped**, 17.27 seconds.
- aarch64: **1,199 passed, 33 skipped**, 12.01 seconds.
- The additional ARM skip is the local pre-commit-hook installation check: the
  isolated clone has no installed hook. The guard's content, planted-violation
  and history tests still run. No hook was bypassed.
- History scan through `cddd455`: 1,180 blob revisions and six commit messages,
  zero findings. All commits retain sole human authorship.

These are observed single-run durations, not a comparative performance claim.

## Replay identity

`tools/explorer_gate.py --require-clean` checks the actual imported checkout and
platform lock. Both machines used the declared Python and deterministic PDF
extractor versions. SQLite versions differ as recorded in the platform locks;
SQLite does not participate in this chain's semantic digest.

| Record | SHA-256 |
|---|---|
| Imported package source, both architectures | `ca2731c3cb9fc5a1d1395bbfebd330cb6a010153d10216b0483c8ea1b44aa628` |
| Baseline synthetic report, both architectures | `23b2e112f62fb89868ee8d953c862650727c4874e03d1de808b61818e9473337` |
| Generated-PDF report, both architectures | `a3d772c74b7eb24207bf70204500931cf92ff31d9a0c5f5653da709d586be5de` |

The other equal cases mutate the label, marker, locale, column order, identifier
family and furniture. Hashing includes character mapping boxes at millipoint
precision. It excludes run telemetry only in metadata contexts; a source cell
called `timestamp` is evidence and remains hashed.

## Optional local-model baseline

One sealed, generated document supplies four unique census candidate ids
(identifier, label, page observation, body recurrence). Each model receives the
same packet and closed schema at temperature zero, with a bounded output budget.
The expected labels were planted before generation. There is one call per
configuration; latency includes loading and warm-state effects.

| Model | Thinking requested | Schema/accounting | Correct / candidate ids | Latency (s) |
|---|---|---|---:|---:|
| Qwen3 14B | off | valid | 1 / 4 | 17.04 |
| Qwen3 14B | on | valid | 1 / 4 | 13.83 |
| Qwen3 32B | off | valid | 3 / 4 | 50.86 |
| Qwen3 32B | on | valid | 3 / 4 | 30.37 |

The 14B model labels only the reference-field candidate correctly; identifier,
page and recurrence decisions are wrong. The 32B model additionally abstains
correctly on page and recurrence, but mislabels the identifier. Thus `accepted`
in the job manifest means **schema and accounting accepted**, never semantically
confirmed. No classification is applied to the deterministic profile. Thinking
shows no accuracy improvement on this packet; the timing sample cannot establish
a speed effect. Four labels are far too few for a general accuracy claim.

Model digests:

- Qwen3 14B: `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.
- Qwen3 32B: `030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840`.

An earlier measurement of Qwen3 30B-A3B returned HTTP 500 in both thinking modes.
It is a recorded server failure, not a semantic score, and was not repaired by
changing the kernel or silently substituting a response. Its digest was
`ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226`.

Reproduce with an explicitly selected local endpoint and a separate cache:

```bash
python tools/explorer_gate.py --require-clean --output /tmp/explorer-gate.json
python tools/explorer_gate.py --require-clean \
  --endpoint "$EXPLORER_LOCAL_ENDPOINT" --models qwen3:14b qwen3:32b \
  --cache /tmp/explorer-model-cache --output /tmp/explorer-model-gate.json
```

## Refutations retained as regressions

- Copying adjacent identifier context into the profile coupled label and marker
  mutations to the family field. The context stays in the census; profile values
  retain only the family shape and revision-marker style.
- Dotted section keys are not evidence for decimal locale. Locale derives from
  physical-number observations, and ambiguous notation remains unknown.
- A recurring header followed much later by an identifier is not a field label.
  The adjacency rule now requires compatible local geometry.
- A global volatile-key filter erased legitimate source columns. Exclusions now
  apply only in metadata contexts.
- Geometry defaults cannot be duplicated independently in census and reader.
  They are recorded in the profile and exercised by a nondefault configuration.

## Acceptance boundary

This record establishes the generated fixtures' behavior and a local-model smoke
baseline. It does not establish recall on arbitrary documents, correctness of
model interpretations, a complete OCR route, or a consumer's domain acceptance.
Continuation distance is an explicit configurable bound. Unknown connectors,
unsupported language/locale and ambiguous layouts still require review.
Consumer verification against its own inventories and its own policy remains a
separate acceptance step. See `EXPLORER.md` for the library and adapter entrypoints.

## Post-merge correction — cross-family acceptance, 2026-09-16

The initial validation above covered Harvest and Saqqara only. It did **not**
establish repository-wide acceptance: from `206ed46`, the full suite failed at
`tests/tender/test_t0_family.py::test_t0_7_saqqara_pinned_lists_are_untouched_by_this_family`.
The Explorer change updated Saqqara's declared kernel list but omitted the
independent literal held by the tender family. The initial test-scope selection
missed this cross-family dependency; the existing guard correctly detected it.

Commit `f1f9954015de64670c2c57380a2ea2373903c50a` adds the four Explorer kernel
classes to that literal. It leaves `FROZEN_COUNTS` unchanged and preserves the
independent assertion. The fix is included in PR #25's merge,
`d3200776e316d0fc27cccfe296f49ce5cdea69c0`.

Independent verification of the merged implementation, using `ragix-env` on
x86_64:

```bash
conda run --no-capture-output -n ragix-env python -m pytest tests/ -q -p no:cacheprovider
```

**Result: 1,948 passed, 37 skipped, 140 warnings in 31.35 seconds.** The warnings
concern a test-class collection warning and deprecated naive UTC timestamp calls;
none is a test failure. This full-suite result supersedes any interpretation of
the earlier scoped counts as repository-wide acceptance. The earlier replay and
model measurements remain scoped as originally recorded; no new ARM full-suite
result is asserted here.

Prevention: `EXPLORER.md` now requires the full repository suite for acceptance,
plus inspection of consumers of `DECLARED_KERNELS`, `FROZEN_COUNTS` and pinned
lists whenever the kernel surface changes. Focused suites remain useful during
implementation. The synthetic gate tool and a clean source tree do not replace
repository-wide tests. CI already runs the tender family gates; this correction
does not weaken or replace that independent check.
