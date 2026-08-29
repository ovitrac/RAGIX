"""Compatibility re-export: the kernel now lives in `kernels/saqqara_run.py`.

Kept because existing callers import `ragix_kernels.saqqara.kernel`. It defines
nothing, so the registry walk still finds exactly one definition site — the
K0.3 gate reads `attr.__module__`, and a re-export does not answer to it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel

__all__ = ["SaqqaraKernel"]
