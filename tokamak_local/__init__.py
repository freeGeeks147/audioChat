"""tokamak_local: Minimal local tokamak equilibrium and 1D transport.

Modules
- equilibrium: Analytic equilibrium profiles and geometry helpers
- sources: Source profile generators (ECH/NBI/ohmic)
- transport: 1D electron/ion transport solver (implicit)
- cli: Command-line interface to run an example locally
"""

from .equilibrium import generate_equilibrium
from .sources import gaussian_source, composite_sources
from .transport import run_transport
