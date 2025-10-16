from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GaussianParams:
    amplitude: float  # source strength (arbitrary units of dT/dt)
    center: float  # rho0 in [0, 1]
    width: float  # sigma of Gaussian in rho


def gaussian_source(rho: np.ndarray, p: GaussianParams) -> np.ndarray:
    """Gaussian source profile S(rho) >= 0 normalized by amplitude."""
    return p.amplitude * np.exp(-0.5 * ((rho - p.center) / max(p.width, 1e-6)) ** 2)


@dataclass(frozen=True)
class CompositeSources:
    electron_source: np.ndarray
    ion_source: np.ndarray


def composite_sources(
    rho: np.ndarray,
    ech: GaussianParams | None = None,
    nbi: GaussianParams | None = None,
    ohmic: float | None = None,
    electron_fraction_of_ech: float = 1.0,
    ion_fraction_of_nbi: float = 1.0,
) -> CompositeSources:
    """Compose simple source terms for electrons and ions.

    S_e = f_ech * ECH + f_ohm * Ohmic
    S_i = f_nbi * NBI

    All terms are in units of dT/dt when used directly in the transport model.
    """
    se = np.zeros_like(rho)
    si = np.zeros_like(rho)

    if ech is not None:
        se = se + electron_fraction_of_ech * gaussian_source(rho, ech)
    if nbi is not None:
        si = si + ion_fraction_of_nbi * gaussian_source(rho, nbi)
    if ohmic is not None and ohmic > 0.0:
        # Distribute ohmic as a slightly core-peaked profile
        ohm_prof = ohmic * (1.0 - 0.6 * rho**2)
        se = se + ohm_prof

    return CompositeSources(electron_source=se, ion_source=si)
