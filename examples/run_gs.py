from tokamak_local.gs import GSParams, GSProfiles, solve_gs
import numpy as np
import os

if __name__ == "__main__":
    params = GSParams(nR=193, nZ=193)
    profiles = GSProfiles(alpha_p=0.0, beta_F2=0.0)
    eq = solve_gs(params, profiles)
    os.makedirs("examples/gs1", exist_ok=True)
    np.savez_compressed(
        "examples/gs1/gs_result.npz",
        R=eq.R, Z=eq.Z, psi=eq.psi, p=eq.p, F=eq.F,
        B_R=eq.B_R, B_Z=eq.B_Z, B_phi=eq.B_phi,
        psi_axis=eq.psi_axis, psi_boundary=eq.psi_boundary,
        axis_index=np.array(eq.axis_index),
    )
    print("Saved examples/gs1/gs_result.npz")
