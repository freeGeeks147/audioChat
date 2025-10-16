from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .equilibrium import EquilibriumParams, generate_equilibrium
from .sources import GaussianParams, composite_sources
from .transport import TransportParams, run_transport
from .gs import GSParams, GSProfiles, solve_gs


def _plot_profiles(outdir: Path, rho: np.ndarray, time: np.ndarray, Te: np.ndarray, Ti: np.ndarray) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Final profiles
    plt.figure(figsize=(6,4))
    plt.plot(rho, Te[-1], label="Te [keV]")
    plt.plot(rho, Ti[-1], label="Ti [keV]")
    plt.xlabel("rho")
    plt.ylabel("Temperature [keV]")
    plt.title("Final profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "final_profiles.png", dpi=160)
    plt.close()

    # Time traces at a few radii
    idxs = [0, len(rho)//3, 2*len(rho)//3, -1]
    plt.figure(figsize=(6,4))
    for idx in idxs:
        plt.plot(time, Te[:, idx], label=f"Te rho={rho[idx]:.2f}")
    plt.xlabel("t [s]")
    plt.ylabel("Te [keV]")
    plt.title("Electron temperature evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "Te_evolution.png", dpi=160)
    plt.close()


def run_example(output: str) -> None:
    outdir = Path(output)

    # Equilibrium
    eqp = EquilibriumParams(R0=3.0, a=1.0, kappa=1.7, B0=5.3, q0=1.0, q95=5.0, nr=129)
    equil = generate_equilibrium(eqp)

    # Sources: modest on-axis ECH and mid-radius NBI
    SeSi = composite_sources(
        equil.rho,
        ech=GaussianParams(amplitude=2.0, center=0.0, width=0.2),
        nbi=GaussianParams(amplitude=1.0, center=0.5, width=0.25),
        ohmic=0.3,
    )

    # Transport
    tp = TransportParams(nr=eqp.nr, dt=1e-3, nsteps=2000, Te_edge=0.2, Ti_edge=0.2, Te0=2.0, Ti0=2.5)
    result = run_transport(equil, tp, SeSi.electron_source, SeSi.ion_source)

    # Save arrays
    np.savez_compressed(
        outdir / "result.npz",
        rho=result.rho,
        time=result.time,
        Te=result.Te,
        Ti=result.Ti,
        q=equil.q,
        Vprime=equil.Vprime,
        meta=dict(equilibrium=asdict(eqp), transport=asdict(tp)),
    )

    _plot_profiles(outdir, result.rho, result.time, result.Te, result.Ti)
    print(f"Saved outputs and plots to {outdir}")


def _plot_gs(outdir: Path, R: np.ndarray, Z: np.ndarray, psi: np.ndarray) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    RR, ZZ = np.meshgrid(R, Z)
    plt.figure(figsize=(6,5))
    cs = plt.contour(RR, ZZ, psi, levels=25, colors='k', linewidths=0.6)
    plt.clabel(cs, inline=1, fontsize=8, fmt='%1.2f')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Poloidal flux contours (psi)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(outdir / 'psi_contours.png', dpi=160)
    plt.close()
def _estimate_q_midplane(R: np.ndarray, Z: np.ndarray, psi: np.ndarray, F: np.ndarray) -> dict:
    # Approx large-A formula on midplane: q ≈ r * F / |∂ψ/∂R| at Z=0
    j0 = int(np.argmin(np.abs(Z)))
    psi_row = psi[j0, :]
    dpsi_dR = np.gradient(psi_row, R)
    i_axis = int(np.argmin(psi_row))
    R_axis = R[i_axis]
    r = np.abs(R - R_axis)
    F_row = F[j0, :]
    # Avoid division by zero
    denom = np.maximum(np.abs(dpsi_dR), 1e-12)
    q_row = r * F_row / denom
    # Build psi_norm along this row to pick q0 and q95
    psi_axis = psi_row[i_axis]
    psi_edge = float(np.mean([psi[0, :].mean(), psi[-1, :].mean(), psi[:, 0].mean(), psi[:, -1].mean()]))
    psi_norm = (psi_row - psi_axis) / (psi_edge - psi_axis + 1e-30)
    # q0 ~ min-r value near axis
    near_axis_mask = r <= (0.05 * (R.max() - R.min()))
    q0_est = float(np.median(q_row[near_axis_mask])) if np.any(near_axis_mask) else float(q_row[i_axis])
    # q95 at psi_norm ~ 0.95 on midplane
    i95 = int(np.argmin(np.abs(psi_norm - 0.95)))
    q95_est = float(q_row[i95])
    return dict(q0=q0_est, q95=q95_est)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local tokamak equilibrium + 1D transport example")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("run-example", help="Run example and write outputs")
    ex.add_argument("--output", default=str(Path.cwd() / "examples" / "run1"), help="Output directory")

    gs = sub.add_parser("gs-solve", help="Run fixed-boundary Grad–Shafranov solver and write outputs")
    gs.add_argument("--output", default=str(Path.cwd() / "examples" / "gs1"), help="Output directory")
    gs.add_argument("--alpha_p", type=float, default=0.0, help="-d p / d psi_norm constant")
    gs.add_argument("--beta_F2", type=float, default=0.0, help="-d(F^2)/d psi_norm constant")
    gs.add_argument("--R0", type=float, default=3.0)
    gs.add_argument("--a", type=float, default=1.0)
    gs.add_argument("--kappa", type=float, default=1.7)
    gs.add_argument("--delta", type=float, default=0.0)
    gs.add_argument("--B0", type=float, default=5.3)
    gs.add_argument("--F", type=float, default=None, help="Toroidal field function F=R*Bphi at boundary")
    gs.add_argument("--nR", type=int, default=257)
    gs.add_argument("--nZ", type=int, default=257)
    gs.add_argument("--lcfs", type=str, default=None, help="Path to LCFS file (.npz with R,Z or .csv two columns)")

    args = parser.parse_args()

    if args.cmd == "run-example":
        run_example(args.output)
    elif args.cmd == "gs-solve":
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        params = GSParams(nR=args.nR, nZ=args.nZ)
        params = GSParams(
            R_min=1.0, R_max=5.0, Z_min=-2.0, Z_max=2.0,
            nR=args.nR, nZ=args.nZ,
            R0=args.R0, a=args.a, kappa=args.kappa, delta=args.delta,
            omega=1.7, max_iters=20000, tol=1e-6,
            psi_boundary_value=1.0,
        )
        F_boundary = args.F if args.F is not None else args.B0 * args.R0
        profiles = GSProfiles(alpha_p=args.alpha_p, beta_F2=args.beta_F2, F_boundary=F_boundary)
        lcfs = None
        if args.lcfs:
            p = Path(args.lcfs)
            if p.suffix == '.npz':
                data = np.load(p)
                lcfs = (data['R'], data['Z'])
            elif p.suffix == '.csv':
                arr = np.loadtxt(p, delimiter=',')
                lcfs = (arr[:,0], arr[:,1])
        eq = solve_gs(params, profiles, lcfs=lcfs)
        np.savez_compressed(
            outdir / "gs_result.npz",
            R=eq.R, Z=eq.Z, psi=eq.psi, p=eq.p, F=eq.F,
            B_R=eq.B_R, B_Z=eq.B_Z, B_phi=eq.B_phi, j_phi=eq.j_phi,
            psi_axis=eq.psi_axis, psi_boundary=eq.psi_boundary,
            axis_index=np.array(eq.axis_index),
            Ip=eq.Ip,
        )
        _plot_gs(outdir, eq.R, eq.Z, eq.psi)
        q_est = _estimate_q_midplane(eq.R, eq.Z, eq.psi, eq.F)
        print(f"Saved GS outputs and plots to {outdir}")
        print(f"Estimated Ip = {eq.Ip/1e6:.3f} MA, q0~{q_est['q0']:.2f}, q95~{q_est['q95']:.2f}")


if __name__ == "__main__":
    main()
