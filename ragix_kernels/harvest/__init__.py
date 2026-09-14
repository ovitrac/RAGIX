"""
The harvest family: knowledge read out of a document store, node by node, under a contract.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A family of kernels over a document store built by `saqqara`. They read what a store
holds and never write to it: everything they derive — abstracts, harvested forms, node
vectors, graph edges — goes to files in the workspace or to a sidecar derived store
(`derived.py`) keyed by the store's own identifiers and the hash of the source it was
computed from.

The layers, bottom up:

    fr/          the deterministic French readers: dates, typed values, cut values
    form         the harvest form: what a model may return, and its refusals
    runner       the jobs that call a model and hand its answer to the form
    recurrence   abstracts by recurrence, window to document to the whole, and its rules
    families     the family rung between the documents and the whole, sized
    derived      the sidecar derived store
    embed        node vectors, and the children-mean at document grain and above
    descent      question to document to window to leaf, measured
    graph        derived edges materialised from what already exists
    ask          composition over the descent and the registers, under a fail-closed checker
    rescore      recorded runs re-scored offline under the rules as they stand
    checks       the checker of a composed brief, and the cut-value scan
    registers/   commitments, clauses, template fills, traps and the French rendering
    gold/        gold by blind reading, its agreement and its re-derived gate
    provenance   the T0 manifest of a corpus by content hash
    models/      Ollama Modelfiles and the memory and concurrency monitors (not Python)

The model is never the authority: it classifies under a schema and writes prose; every
critical value is a grammar's reading, protected by substitution and verified, and a
refusal is counted, never repaired.

Kernels are named `harvest_<verb>`, one Kernel subclass per module under `kernels/`,
and their list is pinned in `tests/harvest/` so that adding one is a reviewed edit.
Nothing here is imported eagerly: the registry discovers kernels by walking the
package, and a kernel module imports its library only inside `compute`.
"""
