from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .equilibrium import EquilibriumParams, generate_equilibrium
from .sources import GaussianParams, composite_sources
from .transport import TransportParams, run_transport


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local tokamak equilibrium + 1D transport example")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("run-example", help="Run example and write outputs")
    ex.add_argument("--output", default=str(Path.cwd() / "examples" / "run1"), help="Output directory")

    args = parser.parse_args()

    if args.cmd == "run-example":
        run_example(args.output)


if __name__ == "__main__":
    main()
