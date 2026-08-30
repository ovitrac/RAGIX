"""tender kernels — one Kernel subclass per module.

The registry discovers them by walking this package, so every module here that
defines a Kernel registers independently. Adding one amends the pinned list in
`tests/tender/test_t0_family.py`: that edit is the point, not an obstacle to it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
