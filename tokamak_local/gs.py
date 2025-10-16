from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np

MU0 = 4e-7 * np.pi  # [H/m]


@dataclass(frozen=True)
class GSProfiles:
    """Parameterization of p(psi) and F(psi) via their derivatives.

    We define dp/dpsi and d(F^2)/dpsi as functions of the normalized flux
    psi_norm in [0, 1], where psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis).

    The physical derivatives are obtained via chain rule:
      dp/dpsi = (d p / d psi_norm) * (1 / (psi_boundary - psi_axis))
      d(F^2)/dpsi = (d F^2 / d psi_norm) * (1 / (psi_boundary - psi_axis))

    The simplest (Solov'ev-like) case is constant derivatives in psi_norm.
    """

    # d p / d psi_norm = -alpha_p (constant gives linear p vs psi_norm)
    alpha_p: float = 0.0
    # d(F^2) / d psi_norm = -beta_F2 (constant gives linear F^2 vs psi_norm)
    beta_F2: float = 0.0
    # Boundary values for integration (p at boundary is often ~0; F at boundary from Bphi)
    p_boundary: float = 0.0
    F_boundary: float = 5.3 * 3.0  # [T*m] typical RBphi at boundary (e.g., B0*R0)

    # Optional custom derivative functions in psi_norm
    dp_dpsi_norm_fn: Callable[[np.ndarray], np.ndarray] | None = None
    dF2_dpsi_norm_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def dp_dpsi(self, psi: np.ndarray, psi_axis: float, psi_bdry: float) -> np.ndarray:
        scale = max(abs(psi_bdry - psi_axis), 1e-16)
        psi_norm = np.clip((psi - psi_axis) / (psi_bdry - psi_axis + 1e-30), 0.0, 1.0)
        if self.dp_dpsi_norm_fn is not None:
            g = self.dp_dpsi_norm_fn(psi_norm)
        else:
            g = -self.alpha_p * np.ones_like(psi_norm)
        return g / scale

    def dF2_dpsi(self, psi: np.ndarray, psi_axis: float, psi_bdry: float) -> np.ndarray:
        scale = max(abs(psi_bdry - psi_axis), 1e-16)
        psi_norm = np.clip((psi - psi_axis) / (psi_bdry - psi_axis + 1e-30), 0.0, 1.0)
        if self.dF2_dpsi_norm_fn is not None:
            g = self.dF2_dpsi_norm_fn(psi_norm)
        else:
            g = -self.beta_F2 * np.ones_like(psi_norm)
        return g / scale

    def integrate_p(self, psi: np.ndarray, psi_axis: float, psi_bdry: float) -> np.ndarray:
        # p(psi) = p_b + ∫_{psi_b}^{psi} dp/dpsi dpsi
        dpdpsi = self.dp_dpsi(psi, psi_axis, psi_bdry)
        # Approximate integral by assuming local derivative constant around psi
        # and anchoring p(psi_b) = p_boundary. A more accurate p(psi) requires
        # cumulative integration along sorted psi, which we keep simple here.
        return self.p_boundary + (psi - psi_bdry) * dpdpsi

    def integrate_F(self, psi: np.ndarray, psi_axis: float, psi_bdry: float) -> Tuple[np.ndarray, np.ndarray]:
        # F^2(psi) = F_b^2 + ∫_{psi_b}^{psi} dF2/dpsi dpsi
        F_b2 = self.F_boundary ** 2
        dF2dpsi = self.dF2_dpsi(psi, psi_axis, psi_bdry)
        F2 = F_b2 + (psi - psi_bdry) * dF2dpsi
        F2 = np.maximum(F2, 1e-12)
        return F2, np.sqrt(F2)


@dataclass(frozen=True)
class GSParams:
    # Grid
    R_min: float = 1.0
    R_max: float = 5.0
    Z_min: float = -2.0
    Z_max: float = 2.0
    nR: int = 257
    nZ: int = 257

    # Geometry guess (for initial psi and boundary function)
    R0: float = 3.0
    a: float = 1.0
    kappa: float = 1.7

    # Solver
    omega: float = 1.7  # SOR relaxation
    max_iters: int = 20000
    tol: float = 1e-6

    # Boundary value used for initial guess and rectangular Dirichlet boundary
    psi_boundary_value: float = 1.0


@dataclass(frozen=True)
class GSEquilibrium:
    R: np.ndarray  # shape (nR,)
    Z: np.ndarray  # shape (nZ,)
    psi: np.ndarray  # shape (nZ, nR) (Z index first for plotting with imshow/contour)
    p: np.ndarray  # same shape as psi
    F: np.ndarray  # same shape as psi (R*Bphi)
    B_R: np.ndarray  # poloidal field components
    B_Z: np.ndarray
    B_phi: np.ndarray
    psi_axis: float
    psi_boundary: float
    axis_index: Tuple[int, int]


def _boundary_psi_rectangular(R: np.ndarray, Z: np.ndarray, params: GSParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct grid and a smooth initial/boundary psi based on an ellipse guess.

    Dirichlet boundary is applied on the rectangle (edges) using this psi value.
    """
    Rg = np.linspace(params.R_min, params.R_max, params.nR)
    Zg = np.linspace(params.Z_min, params.Z_max, params.nZ)
    RR, ZZ = np.meshgrid(Rg, Zg)

    # Elliptical normalized radius s^2
    s2 = ((RR - params.R0) / params.a) ** 2 + (ZZ / (params.kappa * params.a)) ** 2
    s2_clamped = np.clip(s2, 0.0, 1.0)
    psi0 = params.psi_boundary_value * s2_clamped

    # Apply rectangular boundary equal to psi0 on edges
    psi0[0, :] = params.psi_boundary_value * np.clip(((Zg[0]) / (params.kappa * params.a)) ** 2 + ((Rg - params.R0) / params.a) ** 2, 0.0, 1.0)
    psi0[-1, :] = params.psi_boundary_value * np.clip(((Zg[-1]) / (params.kappa * params.a)) ** 2 + ((Rg - params.R0) / params.a) ** 2, 0.0, 1.0)
    psi0[:, 0] = params.psi_boundary_value * np.clip((Zg / (params.kappa * params.a)) ** 2 + ((Rg[0] - params.R0) / params.a) ** 2, 0.0, 1.0)
    psi0[:, -1] = params.psi_boundary_value * np.clip((Zg / (params.kappa * params.a)) ** 2 + ((Rg[-1] - params.R0) / params.a) ** 2, 0.0, 1.0)

    return Rg, Zg, psi0


def _sor_sweep(
    psi: np.ndarray,
    RR: np.ndarray,
    ZZ: np.ndarray,
    dr: float,
    dz: float,
    omega: float,
    rhs: np.ndarray,
) -> float:
    """Perform one SOR sweep, returning max update magnitude."""
    nZ, nR = psi.shape
    max_update = 0.0
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    for j in range(1, nZ - 1):
        for i in range(1, nR - 1):
            R = RR[j, i]
            a = inv_dr2 - 1.0 / (2.0 * R * dr)
            b = inv_dr2 + 1.0 / (2.0 * R * dr)
            c = inv_dz2
            d = inv_dz2
            e = -2.0 * (inv_dr2 + inv_dz2)

            rhs_ij = rhs[j, i]
            psi_new = (a * psi[j, i - 1] + b * psi[j, i + 1] + c * psi[j - 1, i] + d * psi[j + 1, i] + rhs_ij) / (-e)
            delta = psi_new - psi[j, i]
            psi[j, i] = psi[j, i] + omega * delta
            if abs(delta) > max_update:
                max_update = abs(delta)
    return max_update


def solve_gs(params: GSParams, profiles: GSProfiles) -> GSEquilibrium:
    """Solve the fixed-boundary Grad–Shafranov equation on a rectangular grid.

    Equation: Δ* psi = -mu0 R^2 dp/dpsi - 0.5 d(F^2)/dpsi
      with Δ* psi = ∂^2 psi/∂R^2 - (1/R) ∂psi/∂R + ∂^2 psi/∂Z^2

    Boundary: Dirichlet on the rectangle edges using an ellipse-based value.
    Nonlinearity is handled by Picard iteration (update RHS from latest psi).
    """
    Rg, Zg, psi = _boundary_psi_rectangular(np.array([]), np.array([]), params)
    RR, ZZ = np.meshgrid(Rg, Zg)
    dr = Rg[1] - Rg[0]
    dz = Zg[1] - Zg[0]

    # Ensure boundary is set (Dirichlet)
    psi[0, :] = psi[0, :]
    psi[-1, :] = psi[-1, :]
    psi[:, 0] = psi[:, 0]
    psi[:, -1] = psi[:, -1]

    for it in range(params.max_iters):
        # Determine axis and boundary psi for normalization
        axis_index = np.unravel_index(np.argmin(psi), psi.shape)
        psi_axis = float(psi[axis_index])
        # Take boundary values from edges (mean for robustness)
        edge_vals = np.concatenate([
            psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]
        ])
        psi_bdry = float(np.mean(edge_vals))

        # Build RHS from current psi
        dpdpsi = profiles.dp_dpsi(psi, psi_axis, psi_bdry)
        dF2dpsi = profiles.dF2_dpsi(psi, psi_axis, psi_bdry)
        rhs = -MU0 * (RR ** 2) * dpdpsi - 0.5 * dF2dpsi

        max_update = _sor_sweep(psi, RR, ZZ, dr, dz, params.omega, rhs)

        if it % 200 == 0:
            # Optional inexpensive residual estimate (max update)
            pass
        if max_update < params.tol:
            break

    # Post-process fields
    axis_index = np.unravel_index(np.argmin(psi), psi.shape)
    psi_axis = float(psi[axis_index])
    edge_vals = np.concatenate([psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
    psi_bdry = float(np.mean(edge_vals))

    # Compute p(psi) and F(psi)
    p = profiles.integrate_p(psi, psi_axis, psi_bdry)
    F2, F = profiles.integrate_F(psi, psi_axis, psi_bdry)

    # Poloidal field components
    # B_R = - (1/R) ∂psi/∂Z,  B_Z = (1/R) ∂psi/∂R,  B_phi = F / R
    dpsi_dR = np.zeros_like(psi)
    dpsi_dZ = np.zeros_like(psi)
    dpsi_dR[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dr)
    dpsi_dR[:, 0] = (psi[:, 1] - psi[:, 0]) / dr
    dpsi_dR[:, -1] = (psi[:, -1] - psi[:, -2]) / dr

    dpsi_dZ[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dz)
    dpsi_dZ[0, :] = (psi[1, :] - psi[0, :]) / dz
    dpsi_dZ[-1, :] = (psi[-1, :] - psi[-2, :]) / dz

    B_R = -dpsi_dZ / np.maximum(RR, 1e-9)
    B_Z = dpsi_dR / np.maximum(RR, 1e-9)
    B_phi = F / np.maximum(RR, 1e-9)

    return GSEquilibrium(
        R=Rg,
        Z=Zg,
        psi=psi,
        p=p,
        F=F,
        B_R=B_R,
        B_Z=B_Z,
        B_phi=B_phi,
        psi_axis=psi_axis,
        psi_boundary=psi_bdry,
        axis_index=axis_index,
    )
