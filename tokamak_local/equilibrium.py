from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EquilibriumParams:
    R0: float = 3.0  # Major radius [m]
    a: float = 1.0   # Minor radius [m]
    kappa: float = 1.6  # Elongation [-]
    delta: float = 0.3  # Triangularity [-] (not used in 1D metrics here)
    B0: float = 5.3  # Toroidal field at R0 [T]
    q0: float = 1.0  # on-axis safety factor
    q95: float = 5.0  # q at rho=0.95
    nr: int = 129  # radial grid points


@dataclass(frozen=True)
class Equilibrium:
    rho: np.ndarray  # normalized radius in [0, 1]
    R0: float
    a: float
    kappa: float
    B0: float
    q: np.ndarray
    Vprime: np.ndarray  # dV/d(rho) [m^3]


def _q_profile(rho: np.ndarray, q0: float, q95: float) -> np.ndarray:
    """Simple monotonic q-profile using a quadratic form matched at rho=0.95.

    q(rho) = q0 + (q95 - q0) * (rho/0.95)^2 for rho<=0.95, then linear tail.
    """
    q = np.empty_like(rho)
    core_mask = rho <= 0.95
    tail_mask = ~core_mask
    q[core_mask] = q0 + (q95 - q0) * (rho[core_mask] / 0.95) ** 2
    if np.any(tail_mask):
        slope = (q95 - q0) * (2 * 0.95) / (0.95 ** 2)  # derivative at 0.95
        q[tail_mask] = q95 + slope * (rho[tail_mask] - 0.95)
    return q


def _Vprime(R0: float, a: float, kappa: float, rho: np.ndarray) -> np.ndarray:
    """Flux-surface volume derivative dV/d(rho) for simple elliptical cross-section.

    V(rho) = 2 * pi^2 * R0 * (a^2 * kappa) * rho^2  =>  V' = 4 * pi^2 * R0 * a^2 * kappa * rho
    """
    return 4.0 * np.pi**2 * R0 * (a**2) * kappa * np.clip(rho, 0.0, None)


def generate_equilibrium(params: EquilibriumParams) -> Equilibrium:
    """Generate a 1D analytic equilibrium useful for transport modeling.

    Returns arrays on a uniform rho-grid in [0, 1]. Geometry is simplified but
    consistent for volume weighting and basic metrics.
    """
    rho = np.linspace(0.0, 1.0, params.nr)
    q = _q_profile(rho, params.q0, params.q95)
    Vp = _Vprime(params.R0, params.a, params.kappa, rho)
    return Equilibrium(rho=rho, R0=params.R0, a=params.a, kappa=params.kappa, B0=params.B0, q=q, Vprime=Vp)
