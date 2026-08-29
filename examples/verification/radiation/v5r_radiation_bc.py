# SPDX-License-Identifier: MIT
"""V5r — Radiation boundary condition: steady-state slab.

Problem:
    Constant-property, non-ablating, non-decomposing 1-D slab of thickness L.
    Front face (z=0): far-field radiation BC.
    Back face (z=L): prescribed temperature T_back.

    Governing PDE at steady state:  d²T/dz² = 0  (no heat generation)

    Front BC:  −k · dT/dz|_{z=0} = σ·ε·(T_res⁴ − T_face⁴)
    Back BC:   T(L) = T_back

Analytical solution:
    At steady state, d²T/dz² = 0 → T(z) is linear:
        T(z) = T_face + (T_back − T_face) · z/L

    The surface temperature T_face satisfies the energy balance:
        σ·ε·(T_res⁴ − T_face⁴) = (k/L)·(T_face − T_back)

    This nonlinear equation is solved numerically (brentq) to obtain T_face_ss.

Parameters:
    k = 10 W/m/K, ε = 0.9, σ = 5.6704×10⁻⁸ W/m²/K⁴
    T_res = 1300 K, T_back = 300 K, L = 0.01 m

    Derived:  k/L = 1000 W/m²/K  (stiff conduction relative to radiation),
              T_face_ss ≈ 443 K

SCAM setup:
    ENERGY_BALANCE BC with alpha_conv=0, emissivity=0.9, T_rad_in=1300 K.
    PRESCRIBED_TEMP back BC at 300 K.  No B' table (no ablation).
    Inert material with k=10 W/m/K, N=50 nodes, L=0.01 m.
    Thermal time constant τ = L²·ρ·cp/k ≈ 18.5 s; run for t_end=200 s (≫ 10τ).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
SIGMA = 5.6704e-8   # [W/m²/K⁴] Stefan-Boltzmann constant
EPSILON = 0.9       # surface emissivity
T_RES = 1300.0      # [K] radiation reservoir temperature
T_BACK = 300.0      # [K] prescribed back-face temperature
K = 10.0            # [W/m/K] thermal conductivity
L = 0.01            # [m] slab thickness (10 mm)

RHO = 1850.0        # [kg/m³] density (arbitrary; does not affect steady state)
CP = 1000.0         # [J/kg/K] specific heat (arbitrary)
ALPHA = K / (RHO * CP)  # ≈ 5.41e-6 m²/s

T_INIT = 300.0      # [K] uniform initial temperature

N_NODES = 50        # nodes (linear profile → any resolution gives exact result)
T_END = 200.0       # [s] ≫ 10 τ ≈ 185 s; guarantees steady state


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def solve_T_face_ss() -> float:
    """Solve σ·ε·(T_res⁴ − T_face⁴) = k/L·(T_face − T_back) for T_face."""
    def balance(T_face):
        return EPSILON * SIGMA * (T_RES**4 - T_face**4) - K / L * (T_face - T_BACK)
    return float(brentq(balance, T_BACK + 1.0, T_RES - 1.0))


def T_analytical(z, T_face_ss: float) -> np.ndarray:
    """Steady-state linear temperature profile."""
    z = np.asarray(z, dtype=float)
    return T_face_ss + (T_BACK - T_face_ss) * z / L


# ---------------------------------------------------------------------------
# SCAM simulation
# ---------------------------------------------------------------------------

def _make_material() -> MaterialCard:
    T_pts = np.array([200.0, 5000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, K)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name="RadSlab", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0, decomposing=False,
    )


def run_scam():
    mat   = _make_material()
    geom  = GeometryConfig(geometry_type=GeometryType.SLAB)
    stack = StackConfig(layers=[LayerConfig("RadSlab", L, N_NODES, 1)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=0.0,
        T_aw=0.0,
        emissivity=EPSILON,
        T_rad_in=T_RES,
        view_factor=1.0,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    options = SolverOptions(
        t_end=T_END,
        dt_init=1e-2,
        dt_max=10.0,
        dt_min=1e-5,
        dt_max_dT=200.0,
        output_dt=10.0,
        allow_recession=False,
    )
    return run(stack, {"RadSlab": mat}, {"RadSlab": None},
               geom, surface_bc, back_bc, options, initial_T=T_INIT, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    T_face_ss = solve_T_face_ss()
    q_ss = K / L * (T_face_ss - T_BACK)
    print(f"Analytical T_face_ss = {T_face_ss:.4f} K")
    print(f"Steady-state heat flux q_ss = {q_ss:.2f} W/m²")
    print(f"  Radiation in:  ε·σ·T_res⁴ = {EPSILON*SIGMA*T_RES**4:.2f} W/m²")
    print(f"  Radiation out: ε·σ·T_face⁴ = {EPSILON*SIGMA*T_face_ss**4:.2f} W/m²")
    print(f"  Net radiation: {EPSILON*SIGMA*(T_RES**4 - T_face_ss**4):.2f} W/m²")

    results = run_scam()

    saved_times = np.array([s.time for s in results.snapshots])
    T_wall_hist = np.array([s.T_wall for s in results.snapshots])

    final_snap = results.snapshots[-1]
    z_final = final_snap.mesh.y_nodes   # no recession; y_nodes = z
    T_final = final_snap.T

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_hist, ax_prof, ax_err = axes

    # --- Panel 1: T_wall convergence ---
    ax_hist.plot(saved_times, T_wall_hist, 'b-', lw=1.5, label='SCAM T_wall')
    ax_hist.axhline(T_face_ss, color='k', ls='--', lw=1.5,
                    label=f'Analytical T_face = {T_face_ss:.2f} K')
    ax_hist.set_xlabel('Time [s]')
    ax_hist.set_ylabel('Surface temperature [K]')
    ax_hist.set_title('T_wall convergence to steady state')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # --- Panel 2: Temperature profile at steady state ---
    z_ana = np.linspace(0.0, L, 200)
    T_ana = T_analytical(z_ana, T_face_ss)
    ax_prof.plot(z_final * 1e3, T_final, 'b-', lw=1.5, label='SCAM (t=200 s)')
    ax_prof.plot(z_ana * 1e3, T_ana, 'k--', lw=1.5, label='Analytical (linear)')
    ax_prof.set_xlabel('Depth z [mm]')
    ax_prof.set_ylabel('Temperature [K]')
    ax_prof.set_title('Steady-state temperature profile')
    ax_prof.legend(fontsize=9)
    ax_prof.grid(True, alpha=0.3)

    # --- Panel 3: Absolute error vs time ---
    err_hist = np.abs(T_wall_hist - T_face_ss)
    ax_err.semilogy(saved_times, err_hist + 1e-3, 'r-', lw=1.5)
    ax_err.set_xlabel('Time [s]')
    ax_err.set_ylabel('|T_wall − T_face_ss| [K]')
    ax_err.set_title('Convergence of T_wall error')
    ax_err.grid(True, alpha=0.3)

    fig.suptitle(
        f'Radiation BC verification — steady-state slab\n'
        f'ε={EPSILON}, T_res={T_RES:.0f} K, T_back={T_BACK:.0f} K, '
        f'k={K} W/m/K, L={L*1e3:.0f} mm → T_face_ss={T_face_ss:.2f} K',
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(__file__).with_suffix('.png')
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    err_final = abs(T_final[0] - T_face_ss)
    print(f"\nFinal T_wall error: {err_final:.4f} K  (analytical: {T_face_ss:.4f} K)")
    T_lin = T_analytical(z_final, T_face_ss)
    max_profile_err = float(np.max(np.abs(T_final - T_lin)))
    print(f"Max profile deviation from linear: {max_profile_err:.4f} K")

    plt.show()


if __name__ == "__main__":
    main()
