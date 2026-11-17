"""saqqara kernels — six Kernel subclasses in three modules.

`saqqara_run` and `saqqara_index` hold one each, re-exported here; `explorer`
holds the four Explorer kernels. The registry discovers kernels by walking the
package, so every Kernel defined here registers independently. Adding one amends the pinned
list in the K0.3 gate: that edit is the point, not an obstacle to it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from .saqqara_index import SaqqaraIndexKernel
from .saqqara_run import SaqqaraKernel

__all__ = ["SaqqaraIndexKernel", "SaqqaraKernel"]
