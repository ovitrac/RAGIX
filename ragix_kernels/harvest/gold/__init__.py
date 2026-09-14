"""Gold by blind reading: the reader's records, his agreement with himself, the selection, the gate.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

`read` is a read-only navigator over the store whose every command is logged, and the builder that
turns the reader's records into a verified gold; `agreement` compares two passes of the same reader;
`select_pass2` hands a later pass its questions shuffled and unlabelled; `coord_check` re-derives the
gate from the committed gold and the store alone. Paths hang from `HARVEST_LAB` (default: the current
directory) in the lab's layout, `demoE2E/...`.
"""
