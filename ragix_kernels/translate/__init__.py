"""
KOAS-Translate — translation kernel family (P1, in progress).

A registered KOAS family that promotes the punctual EN→FR scientific translation
pipeline into stage-ordered kernels (extract → segment → draft → qa → harmonize
→ rebuild) over a SQLite translation memory, reusing the shared protected-span
codec (``ragix_kernels.shared.protected_spans``) and ``ragix_core.llm_backends``.

See ``docs/developer/TRANSLATE_KERNELS_DESIGN.md`` for the full design.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""
