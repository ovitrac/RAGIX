"""saqqara kernels — one Kernel subclass per module.

The registry discovers kernels by walking the package, so every module here
that defines a Kernel registers independently. Adding one amends the pinned
list in the K0.3 gate: that edit is the point, not an obstacle to it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from .saqqara_run import SaqqaraKernel

__all__ = ["SaqqaraKernel"]
