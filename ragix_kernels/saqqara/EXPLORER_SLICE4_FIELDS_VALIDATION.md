# Explorer Slice 4 validation — reference fields

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Base: `e3da09991300c00b33117cc18077722d4a68ac63`. The demo freeze is unchanged. Branch
`feat/explorer-slice4-fields`. This record covers the label-value window and its readers
only; table recovery and the quantity grammar of the same slice are recorded separately.
It reports implementation checks on synthetic fixtures. Independent confirmation on real
documents is the consumer's and is recorded on the consumer's side, figures included; what
it taught is stated here in generic terms. No public push or merge is asserted.

## What confirmation on real documents taught

The first hand-over of this branch passed every gate below and was **not confirmed**: on
real documents it recovered none of the fields the frozen reader missed, and it lost some
the frozen reader held. The work had been specified from a finding reported as `RULE_STOP`,
read as "a rule closes the window". `RULE_STOP` is a category of reasons (label, header,
caption, heading, rule, column break). On the pages concerned no window was closed by a
rule: the stop was a *label* (the value line itself taken as a label) or a *column break*
(label and value not recognised as neighbouring cells). The first seven commits repaired
defects that are real, each reproduced on the frozen reader and gated, but they were
defects of synthetic grids, not the causes at hand. One of them made things worse: "a
label must introduce a value" let a value that repeats a phrase seen elsewhere before a
colon become a label and close its own label's window.

The three real causes: a phrase made of a word, an identifier and a revision taken as a
label, with a colon after it since the frozen reader and without it since that commit; a
cell border a hundredth of a point inside the label's last glyph, which leaves label and
value unjoined and fails the next-cell test; and, as a loss of precision, an identifier
anywhere in wrapped prose voting for the one-word colon label before it.

Lessons kept in the method: a fixture is specified from the stop reason and the shape of
the stopping view, never from a reason category; a rule added for one case is checked
against what it newly admits; weaker evidence yields to stronger; a green synthetic suite
says nothing about real documents until an independent confirmation has run. The second
hand-over (the B gates and their repair) was confirmed: nothing the frozen reader held is
lost, and most of what it missed is recovered.

## Method

The gates were written before any repair, on synthetic fixtures, and committed first.
Each gate that the reader failed carried a strict expected-failure mark naming the defect
measured on it; a mark that starts passing fails the suite, so every repair had to remove
its own marks, and a repair that did not move its gate was visible at once. One repair was
found incomplete that way, twice over, and completed before its commit (the range relation
below). Where a gate was still blocked by a defect scheduled for a later commit, the
blocked corner kept a mark naming that defect until its own commit.

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
| `045b8b4` | first version of this record; first hand-over, not confirmed | 2,657 passed, 38 skipped, 5 xfailed |
| `2d6395e` | three defects found by confirmation, as strict expected failures | 2,661 passed, 38 skipped, 33 xfailed |
| `d7f34b1` | a value is never a label, a grazed border still separates cells; second hand-over, confirmed | 2,735 passed, 38 skipped, 5 xfailed |
| (this commit) | deferral recorded: four gates assert the kept behaviour; this record | 2,739 passed, 38 skipped, 1 xfailed |

The expected failures rose by two at `68504c7` because one gate was split in two so that
each half names a single defect.

## Fixture geometry

Every label, identifier, type word and title in the fixtures is synthetic. Geometry enters
as ranges, never as a single value: line height 12–18 pt; label-to-value offset −0.5 to
+0.5 pt; an underline of one or two rules, 0.8–1.6 pt apart, 0–2.5 pt above the label's
bottom edge, the label's width ± 3 pt; gaps inside a field −2.5 to +7 pt; gap to the next
block 0.2–0.6 line heights; block gap 2.2 line heights; a cell border 0 to 0.04 pt inside
the label's last glyph. Each gate runs at four corners of these ranges (the range ends and
one interior point). A result that differs between corners is treated as a hidden layout
rule, not as a pass; after the repairs every gate gives the same result at the four
corners, except the one named under *Open*.

Painting defects of the fixtures themselves were found and corrected. An ideal grid failed
at one corner only: at the largest line height the value was wider than its cell and
crossed the vertical rule. A gate did not move after its repair: a colon-less label span
lacked the space that extracted text carries, so label and value were glued. A trap
expected a rule on a space inside the label to be no border, whereas the text is
legitimately split there; the trap now places the rule through a letter. The corrected
paintings fail on the reader they were written against for the intended reason.

## Defects and repairs

**Painted rules are edges, not coordinates.** A producer paints each cell's own borders,
so a shared border arrives as two rules a fraction of a point apart. The cell lookup
compared coordinates by exact equality and the next-cell test required exactly one rule
between label and value. Now parallel rules within `rule_tolerance` of an edge's first
face are one edge that keeps both faces (anchoring on the first face prevents a run of
close rules from drifting into one wide edge); a cell is bounded by inner faces, a
neighbour shares the outer face; the next-cell test counts edges.

**A grazed border still separates two cells.** The border between a label's cell and its
value's cell is routinely a hundredth of a point inside the label's last glyph. The span
is then flagged as intersected and never joined with its neighbour, and the next-cell test
wanted the border at or beyond the label's right edge: the value was refused as a column
break. The next-cell test and cell containment now use `rule_tolerance`. A rule through a
letter, farther inside than the tolerance, is still no border; a rule on a space splits
the text there, as before.

**Overlapping line boxes are not one row.** Two boxes were one row as soon as they
overlapped vertically; the boxes of consecutive lines routinely overlap by a couple of
points, so the second line of a wrapped value became a column break. Two boxes now share a
row when the vertical centre of the shorter lies inside the taller. No number is
introduced.

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

**A label must introduce a value, and a value is never a label.** A label is known when
the document shows it with a colon anywhere, or when the consumer declares it; none is
shipped. A phrase that holds an identifier is reference content, with or without a colon
after it: the test is on the candidate label span, never on the line; such a phrase is no
colon label and never enters the document's own lexicon. Without its colon, a known label
opening a line is a label only when reference content follows: an identifier, a declared
type word directly before one, a section sign or a dotted number, or a quoted title.
Otherwise the words are prose: observed (`follow: prose`), listed in the profile
diagnostics as `label_in_prose` with page, span and reason, and never a vote. A colon-less
label met in the value position of a colon label that has no value yet is that label's
value and opens no window of its own. A window closed before any value is never silent:
the occurrence is reported with its page and span.

**A label's occurrence votes only when its value opens with reference content.** An
identifier met anywhere in the window used to make the occurrence identifier-bearing, so
prose after a one-word colon label voted for that label as soon as its wrapped lines were
kept, and the profile reported a reference field where none exists. The value must now
open with reference content, one word aside (the slot of a type word the policy may not
know). The criterion is the same at every corner, including ordinary leading, where the
frozen reader would itself report the label. No overlap fraction is introduced: on real
leading none separates wrapped prose from a wrapped value.

**Role-word lines are listed, never read.** Any letters before an identifier used to flag
a line and attach it all the same. A role-word line now opens with a declared role word,
optionally a number sign or a colon, then directly an identifier; it and every later line
up to the next structural stop are listed on the window (`undecidable`) and never read;
such a line is never a label nor the value of a plain label, which the wrapped-statement
fixture exercises.

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
| `rule_tolerance` (pt) | 1.0 | 0.5, 1.0, 2.0 | readings and cells identical: painted borders 0.3 pt apart; a border 0 and 0.04 pt inside the last glyph; a column 12 pt wide stays a column at 2.0 |
| `underline_margin` (line heights) | 0.5 | 0.3, 0.5, 1.0 | readings and cells identical; the wide-rule trap holds at each value |
| fallback `max_gap_ratio` | 2.5 | 1.5, 2.5, 4.0 | readings identical on the fourteen swept fixtures; prose never becomes a reference field |
| fallback `max_lines` | 8 | 6, 8, 16 | the same |
| `labels` | none | — | consumer data; a document also supplies the phrases it shows with a colon |
| `type_words` | generic document nouns, FR and EN | — | a domain's acronyms come from the consumer; fixtures use a synthetic acronym |
| `role_words` | generic FR and EN words | — | a word cannot be both a type word and a role word |

A fixture joins the sweep only once it reads correctly at every corner, so the sweep never
certifies an invariantly wrong reading.

## Replay on both architectures

`python -m tests.saqqara.explorer_slice4_fields_replay --output <file>` from a clean
checkout at `d7f34b1`: 23 fixture cases at four corners, 92 replay cases, each with census,
profile, reading and report digests and the counts of fields, targets, members, stops and
listed lines; the expected field count is asserted per case.

| Host | Interpreter, SQLite, PyMuPDF | source hash | cases, plain digest | cases, replay digest | full suite |
|---|---|---|---|---|---|
| linux-64 | as `tools/explorer-lock-linux-64.json` | `e73557c8ddda0a02` | `cbb2b2d2e6c89108` | `ff3367cb7983d5b6` | 2,735 passed, 38 skipped, 5 xfailed |
| linux-aarch64 | as `tools/explorer-lock-linux-aarch64.json` | `e73557c8ddda0a02` | `cbb2b2d2e6c89108` | `ff3367cb7983d5b6` | 2,735 passed, 38 skipped, 5 xfailed |

The 92 cases are equal on the two architectures, and the files were reproduced byte for
byte by an independent run. The plain digest is the SHA-256 of
`json.dumps(cases, sort_keys=True)` with default separators; the replay digest is
`ragix_kernels.harvest.report.replay_digest` of the same cases. The first version of this
record gave a plain digest and called it canonical. At `876aa08` the 76 cases of that
replay were equal on both architectures as well, under two hash seeds. The versions
recorded in each replay's provenance equal the platform lock of its host. The commit that
adds this record changes no Python source of the package, so the source hash is unchanged.

## Record versions

`census/0.7` (reference policy sealed; symbol markers on every number; `across_lines`
connectors; `follow: prose` label records; a vote needs a value that opens with reference
content), `value-window/0.3` (policy in the identity; `undecidable`), `document-profile/0.6`
(bare connectors; `label_in_prose` diagnostics). Stored records of earlier versions are
refused and must be rebuilt. Kernel cache versions are not changed by this branch and must
move with these records when it is merged, or cached outcomes of the earlier reader are
served; merging with another branch that moved them leaves no conflict to warn of it.

## Open

- **Bounds derived from the gap histogram: deferred by ruling.** The frozen behaviour is
  kept: a derived bound stops as a guard and every gap stop keeps its review flag; a gate
  asserts it at the four corners. Letting a derived bound stop without the flag was
  considered and set aside: measured stop gaps after a field and ordinary leading are of
  the same scale, so no scale guard separates them. A successor rule, a derived bound
  corroborated by structure at the stop line, is to be measured later. One defect of the
  frozen derivation remains, flagged and not silent: it accepts any two compact repeated
  gap values as two modes, so leading that alternates between 0 and 0.2 line heights
  yields a derived bound of 0.1 and closes a window at ordinary leading. One expected
  failure names it: the role-word fixture at the interior corner.
- **Revision written after a slash.** An identifier directly followed by a slash and a
  revision is read as one identifier whose last group is cut at the revision's first dot,
  with no revision: the field is found, its target is not segmented. This belongs to the
  explorer's revision grammar, outside this branch.
- **Windows that run to the line cap** are more frequent since wrapped lines are kept and
  underlines no longer stop a window; each is flagged. The rule that would end them, that
  a line which is no continuation ends the field, changes every window and needs its own
  fixture and ruling.
- **A label intersected by a rule whose value sits on another row** is not recovered.
- **Platform locks** are unchanged. Aligning them with newer interpreter, SQLite and
  PyMuPDF versions requires the test environment to move first on both hosts, and
  `tools/explorer_gate.py` pins a PyMuPDF version of its own.

## Not exercised by this record

Derived forms of a label (plural and the like) are not folded: a consumer declares the
forms it uses. Range words beyond those already classified, list words and the
revision-marker forms of the wider grammar list are outside this branch. The relay of
table diagnostics through the reader depends on the table branch.
