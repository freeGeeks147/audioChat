from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .equilibrium import Equilibrium


@dataclass(frozen=True)
class TransportParams:
    nr: int = 129
    dt: float = 1e-3  # time step [s]
    nsteps: int = 5000
    Te_edge: float = 0.2  # keV
    Ti_edge: float = 0.2  # keV
    Te0: float = 2.5  # initial on-axis Te [keV]
    Ti0: float = 3.0  # initial on-axis Ti [keV]
    chi_e0: float = 0.5  # m^2/s core
    chi_e_edge: float = 2.5  # m^2/s edge
    chi_i0: float = 0.3  # m^2/s core
    chi_i_edge: float = 1.5  # m^2/s edge


@dataclass(frozen=True)
class TransportResult:
    time: np.ndarray  # [s]
    rho: np.ndarray
    Te: np.ndarray  # shape (nt, nr)
    Ti: np.ndarray  # shape (nt, nr)


def _build_chi_profile(rho: np.ndarray, chi0: float, chi_edge: float) -> np.ndarray:
    return chi0 + (chi_edge - chi0) * rho**2


def _implicit_step(
    Tn: np.ndarray,
    Vprime: np.ndarray,
    chi: np.ndarray,
    S: np.ndarray,
    dt: float,
    rho: np.ndarray,
    T_edge: float,
) -> np.ndarray:
    """Single backward-Euler implicit step for
        dT/dt = (1/V') d/drho [ V' chi dT/drho ] + S
    with Neumann at rho=0 and Dirichlet T(edge)=T_edge.
    """
    nr = Tn.size
    dr = rho[1] - rho[0]
    # Interface coefficients A = V' * chi
    Vp = Vprime
    A_plus = 0.5 * (Vp[1:] * chi[1:] + Vp[:-1] * chi[:-1])  # size nr-1
    A_minus = A_plus  # symmetric since central average

    diag = np.ones(nr)
    lower = np.zeros(nr-1)
    upper = np.zeros(nr-1)
    rhs = Tn + dt * S

    # Interior points 1..nr-2
    for j in range(1, nr-1):
        invV = 1.0 / max(Vp[j], 1e-12)
        aL = dt * invV * A_minus[j-1] / (dr * dr)
        aU = dt * invV * A_plus[j] / (dr * dr)
        lower[j-1] = -aL
        upper[j] = -aU
        diag[j] = 1.0 + aL + aU

    # Core (j=0): zero-gradient => no flux to j=-1
    j = 0
    invV0 = 1.0 / max(Vp[j], 1e-12)
    aU0 = dt * invV0 * A_plus[j] / (dr * dr)
    diag[j] = 1.0 + aU0
    upper[j] = -aU0

    # Edge (j=nr-1): Dirichlet T = T_edge
    j = nr - 1
    diag[j] = 1.0
    rhs[j] = T_edge

    # Solve tridiagonal system
    Tn1 = _solve_tridiagonal(lower, diag, upper, rhs)
    return Tn1


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal systems.
    a: lower diag (n-1)
    b: main diag (n)
    c: upper diag (n-1)
    d: rhs (n)
    """
    n = b.size
    ac, bc, cc, dc = a.copy(), b.copy(), c.copy(), d.copy()
    for i in range(1, n):
        mc = ac[i-1] / bc[i-1]
        bc[i] = bc[i] - mc * cc[i-1]
        dc[i] = dc[i] - mc * dc[i-1]
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def run_transport(
    equil: Equilibrium,
    tp: TransportParams,
    Se: np.ndarray,
    Si: np.ndarray,
) -> TransportResult:
    """Run coupled but separated e-/i- heat transport with given sources.

    Sources Se/Si are in units consistent with dT/dt. Geometry enters via V'.
    """
    rho = equil.rho
    assert rho.size == tp.nr == equil.Vprime.size

    # Initial profiles: smooth core-peaked to edge value
    Te = tp.Te_edge + (tp.Te0 - tp.Te_edge) * (1.0 - rho**2)**1.5
    Ti = tp.Ti_edge + (tp.Ti0 - tp.Ti_edge) * (1.0 - rho**2)**1.5

    chi_e = _build_chi_profile(rho, tp.chi_e0, tp.chi_e_edge)
    chi_i = _build_chi_profile(rho, tp.chi_i0, tp.chi_i_edge)

    nt = tp.nsteps + 1
    time = np.linspace(0.0, tp.dt * tp.nsteps, nt)
    Te_hist = np.empty((nt, tp.nr))
    Ti_hist = np.empty((nt, tp.nr))
    Te_hist[0] = Te
    Ti_hist[0] = Ti

    for k in range(tp.nsteps):
        Te = _implicit_step(Te, equil.Vprime, chi_e, Se, tp.dt, rho, tp.Te_edge)
        Ti = _implicit_step(Ti, equil.Vprime, chi_i, Si, tp.dt, rho, tp.Ti_edge)
        Te_hist[k + 1] = Te
        Ti_hist[k + 1] = Ti

    return TransportResult(time=time, rho=rho, Te=Te_hist, Ti=Ti_hist)
