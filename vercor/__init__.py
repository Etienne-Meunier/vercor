"""Primary VerCOR 4 assembly conveniences."""

import sys as _sys

from vercor.clock import Clock
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.grids import RectilinearGrid
from vercor.runtime import RuntimeOptions
from vercor.state import RunState

__all__ = [
    "Clock",
    "Coupler",
    "Exchange",
    "RectilinearGrid",
    "RunState",
    "RuntimeOptions",
]

# The assembly imports load private component runtime modules after the
# component facade has initialized.  Python attaches such children to their
# parent package; remove those accidental alternate access paths once root
# assembly is complete.
_component_facade = _sys.modules["vercor.components"]
for _module_name in ("runtime_execution", "setup_validation"):
    vars(_component_facade).pop(_module_name, None)
del _component_facade, _module_name
