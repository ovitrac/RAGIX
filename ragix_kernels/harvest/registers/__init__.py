"""The registers of a consultation's family of pieces: what each piece commits to, what it shares.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Deterministic, CPU only, the store read-only, every span re-read from the store byte for byte:
`family` (the family's skeleton, shared and unique sentences, the lot map and the orphans),
`template` (slot fills of near-identical sentences), `commitments` (each piece's own numeric
statements), `clauses` (what the contract and the consultation rules fix), `traps` (a piece
contradicting its family, for the human) and `render_fr` (the French rendering of the collection).
They run in that order in one output directory, each gating on what the one before wrote.
"""
