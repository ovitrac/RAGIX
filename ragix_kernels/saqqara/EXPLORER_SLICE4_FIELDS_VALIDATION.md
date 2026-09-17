# Explorer Slice 4 validation — reference fields

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Base: `e3da09991300c00b33117cc18077722d4a68ac63`. The demo freeze is unchanged. Branch
`feat/explorer-slice4-fields`. This record covers the label-value window and its readers
only; table recovery and the quantity grammar of the same slice are recorded separately.
It separates implementation checks from independent consumer acceptance, which is not
asserted here. No public push or merge is asserted by this record.

## Method

The gates were written before any repair, on synthetic fixtures, and committed first.
Each gate that the frozen reader failed carried a strict expected-failure mark naming the
defect measured on it; a mark that starts passing fails the suite, so every repair had to
remove its own marks, and a repair that did not move its gate was visible at once. One
repair was found incomplete that way, twice over, and completed before its commit (the
range relation below). Where a gate was still blocked by a defect scheduled for a later
commit, the blocked corner kept a mark naming that defect until its own commit.

Each commit was preceded by the full suite on the working tree and followed by the full
suite at the commit itself, in a detached worktree with `PYTHONPATH` pinned to it and the
resolved package path printed. One check of a corrected fixture against the frozen reader
first ran the branch code by mistake; the printed package path exposed it and the check
was repeated on the frozen tree.

| Commit | Scope | Detached suite at the commit |
|---|---|---|
| `e3da099` | base (frozen reader) | 2,377 passed, 38 skipped |
| `33861f5` | gates stated on synthetic geometry, 72 strict expected failures | 2,425 passed, 38 skipped, 72 xfailed |
| `3478ded` | painted rules read as edges | 2,467 passed, 38 skipped, 64 xfailed |
| `68504c7` | wrapped value kept when line boxes overlap | 2,474 passed, 38 skipped, 66 xfailed |
| `7075b39` | range kept one relation when its numbers carry a marker | 2,503 passed, 38 skipped, 52 xfailed |
| `360d84c` | a label must introduce a value | 2,538 passed, 38 skipped, 32 xfailed |
| `ca82f8e` | role-word lines listed, never read | 2,544 passed, 38 skipped, 29 xfailed |
| `25073ac` | an underline is emphasis, never an edge | 2,657 passed, 38 skipped, 5 xfailed |
| `876aa08` | clean-checkout replay of the fixtures | 2,657 passed, 38 skipped, 5 xfailed |

The expected failures rose by two at `68504c7` because one gate was split in two so that
each half names a single defect.

## Fixture geometry

Every label, identifier, type word and title in the fixtures is synthetic. Geometry enters
as ranges, never as a single value: line height 12–18 pt; label-to-value offset −0.5 to
+0.5 pt; an underline of one or two rules, 0.8–1.6 pt apart, 0–2.5 pt above the label's
bottom edge, the label's width ± 3 pt; gaps inside a field −2.5 to +7 pt; gap to the next
block 0.2–0.6 line heights; block gap 2.2 line heights. Each gate runs at four corners of
these ranges (the range ends and one interior point). A result that differs between
corners is treated as a hidden layout rule, not as a pass; after the repairs every gate
gives the same result at the four corners, except the one named under *Open*.

Two painting defects of the fixtures themselves were found and corrected. An ideal grid
failed at one corner only: at the largest line height the value was wider than its cell
and crossed the vertical rule. A gate did not move after its repair: a colon-less label
span lacked the space that extracted text carries, so label and value were glued. The
corrected paintings still fail on the frozen reader for the intended reason.

## Defects and repairs

**Painted rules are edges, not coordinates.** A producer paints each cell's own borders,
so a shared border arrives as two rules a fraction of a point apart. The cell lookup
compared coordinates by exact equality and the next-cell test required exactly one rule
between label and value. On such grids the label cell was not recognised: the value to
the right was lost as a column break, the value below was closed by the row rule. Now
parallel rules within `rule_tolerance` of an edge's first face are one edge that keeps
both faces (anchoring on the first face prevents a run of close rules from drifting into
one wide edge); a cell is bounded by inner faces, a neighbour shares the outer face; the
next-cell test counts edges.

**Overlapping line boxes are not one row.** Two boxes were one row as soon as they
overlapped vertically; the boxes of consecutive lines routinely overlap by a couple of
points, so the second line of a wrapped value became a column break and every later
member was lost. Two boxes now share a row when the vertical centre of the shorter lies
inside the taller. No number is introduced.

**A range stays one relation.** `§4.1 à §4.12` was read as two unrelated numbers. Three
observation defects combined, the first found only because the gate refused to move after
the second was repaired: the census recorded a number's marker only for the first number
of a line, which on a reference line is the revision, so the section sign was never a known
marker; the profile classified the connector with the marker inside it while the reader
stripped it later, against an empty range set; a connector ending a wrapped line, its
endpoint opening the next, was never observed. Now the section sign is observed on every
number, a word still only before the first (before a later number it is a connector);
`bare_connector` is the single definition used by profile and reader; a connector closing
a window line whose next line opens with a number is observed as an exact span of its own
line (`pattern: across_lines`).

**A label must introduce a value.** A label is known when the document shows it with a
colon anywhere, or when the consumer declares it; none is shipped. Without its colon, a
known label opening a line is a label only when reference content follows: an identifier,
a declared type word directly before one, a section sign or a dotted number, or a quoted
title. Otherwise the words are prose: observed (`follow: prose`), listed in the profile
diagnostics as `label_in_prose` with page, span and reason, and never a vote for or
against the label.

**Role-word lines are listed, never read.** Any letters before an identifier used to flag
a line and attach it all the same: a type word before the value made a found field
undecidable, the identifier of a quoted procedure became a target of the field, and a
role-word line with a colon became a label. A role-word line now opens with a declared
role word, optionally a number sign or a colon, then directly an identifier; it and every
later line up to the next structural stop are listed on the window (`undecidable`) and
never read; such a line is never a label nor the value of a plain label, which the
wrapped-statement fixture exercises (the wrapped line would otherwise become a label whose
value is the next role-word line).

**An underline is emphasis, never an edge.** On the label's bottom edge an underline lay
"between" the label line and the next and closed the window; inside a ruled cell it was
taken as the cell's bottom edge and the grid path was abandoned. A rule underlines a line
when it lies in the lower half of the line's box or on its bottom edge and runs no farther
than `underline_margin` line heights beyond the line's ends: extent tells an underline
from an edge. Underlines are set aside once per page. The trap holds at every corner: a
rule as wide as a table on the label's bottom edge still closes the window.

## Declared values and sensitivity

`ReferencePolicy` is passed through `CensusConfig`, sealed in the census and part of each
window identity. Nothing in it comes from a development document.

| Field | Default | Swept | Result on the fixtures |
|---|---|---|---|
| `rule_tolerance` (pt) | 1.0 | 0.5, 1.0, 2.0 | readings and cells identical; painted borders 0.3 pt apart; a column 12 pt wide stays a column at 2.0 |
| `underline_margin` (line heights) | 0.5 | 0.3, 0.5, 1.0 | readings and cells identical; the wide-rule trap holds at each value |
| fallback `max_gap_ratio` | 2.5 | 1.5, 2.5, 4.0 | readings identical on the twelve swept fixtures |
| fallback `max_lines` | 8 | 6, 8, 16 | readings identical on the twelve swept fixtures |
| `labels` | none | — | consumer data; a document also supplies the phrases it shows with a colon |
| `type_words` | generic document nouns, FR and EN | — | a domain's acronyms come from the consumer; fixtures use a synthetic acronym |
| `role_words` | generic FR and EN words | — | a word cannot be both a type word and a role word |

A fixture joins the sweep only once it reads correctly at every corner, so the sweep never
certifies an invariantly wrong reading.

## Replay on both architectures

`python -m tests.saqqara.explorer_slice4_fields_replay --output <file>` from a clean
checkout at `876aa08`: 19 fixture cases at four corners, 76 replay cases, each with
census, profile, reading and report digests and the counts of fields, targets, members,
stops and listed lines; the expected field count is asserted per case.

| Host | Interpreter, SQLite, PyMuPDF | source hash | cases digest | full suite |
|---|---|---|---|---|
| linux-64 | as `tools/explorer-lock-linux-64.json` | `911a7f935b5f6e3e` | `2f90d6d3e718985a` | 2,657 passed, 38 skipped, 5 xfailed |
| linux-aarch64 | as `tools/explorer-lock-linux-aarch64.json` | `911a7f935b5f6e3e` | `2f90d6d3e718985a` | 2,657 passed, 38 skipped, 5 xfailed |

The versions recorded in each replay's provenance equal the platform lock of its host.

The 76 cases are equal on the two architectures. On linux-64 the cases digest is identical
under two different hash seeds. The cases digest is the SHA-256 of the canonical JSON of
`cases`; the commit that adds this record adds no Python source, so the source hash is
unchanged by it.

## Record versions

`census/0.7` (reference policy sealed; symbol markers on every number; `across_lines`
connectors; `follow: prose` label records), `value-window/0.3` (policy in the identity;
`undecidable`), `document-profile/0.6` (bare connectors; `label_in_prose` diagnostics).
Stored records of earlier versions are refused and must be rebuilt. `EXPLORER.md` still
names the earlier versions and needs this note when the branch is merged.

## Open

- **Bounds derived from the gap histogram.** The frozen derivation accepts any two compact
  repeated gap values as two modes, whatever their scale: leading that alternates between
  0 and 0.2 line heights yields a derived bound of 0.1 and closes a window at ordinary
  leading. The window keeps its review flag, so the result is flagged, not silent. Whether
  a derived bound should stop as structure (without the flag), and under which scale
  guard, awaits a ruling; nothing was changed. Five expected failures remain and all name
  this: the role-word fixture at the interior corner, and the four corners of the gate
  that expects a derived bound to stop without the flag.
- **Platform locks** are unchanged. Aligning them with newer interpreter, SQLite and
  PyMuPDF versions requires the test environment to move first on both hosts, and
  `tools/explorer_gate.py` pins a PyMuPDF version of its own.

## Not exercised by this record

Derived forms of a label (plural and the like) are not folded: a consumer declares the
forms it uses. A line that is neither a stop, a role-word line nor a continuation is
still attached until a structural stop; the fixtures do not require an implicit end of
field. Range words beyond those already classified, list words, and the revision-marker
forms of the wider grammar list are outside this branch. The relay of table diagnostics
through the reader depends on the table branch.
