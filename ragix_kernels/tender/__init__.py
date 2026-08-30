"""
Structured response preparation over document stores.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

A family of kernels that work over a document store built by `saqqara`: they read
what a store already holds and prepare structured responses from it. They never
build a store, never read a source file, and never write to what they read.

This package holds one kernel today. That is on purpose — discovery, registration,
the dependency declaration, both surfaces and the guard scope are proved on
something that cannot be wrong on substance before anything with substance arrives.

Kernels are named `tender_<verb>`, one Kernel subclass per module, and the list of
them is pinned in `tests/tender/` so that adding one is a reviewed edit.

Nothing here is imported eagerly: the registry discovers kernels by walking the
package, and a family that imported its own kernels at package level would make
every consumer pay for all of them.
"""
