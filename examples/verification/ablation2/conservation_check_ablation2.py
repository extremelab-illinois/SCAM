# SPDX-License-Identifier: MIT
"""Mass and energy conservation diagnostics for AblationTestCase_2.x.

Runs the base ablation2 case and plots:
  1. Solid mass ∫ρ dV and cumulative mass removed at surface vs time
  2. Mass residual: Δ(∫ρ dV) + cum_mass_out  (should be ≈ 0)
  3. Energy stored ∫ρh dV, cumulative q_cond·A·dt, and their difference
  4. Net internal energy source (decomposition + gas advection contribution)

For inert materials the energy residual is ≈ 0 (closed energy balance via
q_cond alone).  For charring TACOT, the difference reflects the net
chemical decomposition energy deposited into the solid.

Run:
    MPLBACKEND=Agg python3 examples/verification/ablation2/conservation_check_ablation2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pato2_common import (
    C_M, P_EDGE, RHOUE_CH, H_R, T_END, T_INIT,
    prepare_plot_output, save_plot_atomic,
)
from compare_pato_ablation2 import load_tacot_material, make_tacot_bprime_backend

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.diagnostics import conservation_check
from scam.solvers.material_response import run

OUT_PNG     = Path(__file__).parent / "conservation_check_ablation2.png"

THICKNESS = 0.05   # m
N_NODES   = 201

# ---------------------------------------------------------------------------
# BC time history (same as base ablation2)
# ---------------------------------------------------------------------------
def _alpha_conv(t):
    return RHOUE_CH if t <= 60.0 else 0.3e-2

def _h_r(t):
    return H_R if t <= 60.0 else 0.0

def _p_e(t):
    return P_EDGE


def run_scam() -> tuple:
    mat, bpt_yaml = load_tacot_material()
    bpt = make_tacot_bprime_backend(bpt_yaml, "table")

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=N_NODES, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=_alpha_conv,
        h_r=_h_r,
        p_e=_p_e,
        rho_e_u_e=RHOUE_CH,
        C_M=C_M,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=T_END,
        dt_init=1e-3,
        dt_max=1.0,
        dt_min=1e-5,
        dt_max_dT=20.0,
        output_dt=0.5,
        allow_recession=True,
        use_rho_old=True,
    )

    mat_cards  = {mat.name: mat}
    bpt_tables = {mat.name: bpt}
    mat_list   = [mat]

    print("Running SCAM ablation2 (TACOT 3.0 YAML + B' table, adiabatic back face)...")
    results = run(
        stack, mat_cards, bpt_tables,
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT, verbose=True,
    )
    return results, mat_list, stack


def main() -> None:
    prepare_plot_output(OUT_PNG)
    results, mat_list, stack = run_scam()

    diag = conservation_check(results, mat_list)

    t              = diag["times"]
    mass           = diag["mass"]             # ∫ρ dV [kg/m²]
    mass_exact_out = diag["cum_mass_exact"]   # exact domain mass loss [kg/m²]
    mass_model_out = diag["cum_mass_model"]   # CMA model: (m_dot_char+m_dot_pyro)*A*dt [kg/m²]
    mass_cma_res   = diag["mass_cma_residual"] # difference [kg/m²]

    energy  = diag["energy"]          # ∫ρh dV [J/m²]
    E_in    = diag["cum_energy_in"]   # cumulative q_cond·A·dt [J/m²]
    dE      = diag["dE_stored"]       # Δ(∫ρh dV) [J/m²]
    E_src   = diag["energy_source"]   # net internal source [J/m²]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "AblationTestCase_2.x — conservation diagnostics\n"
        "TACOT 3.0 YAML | B' table | adiabatic back | full ablation2 BC",
        fontsize=11,
    )

    # --- Panel 1: solid mass and mass removed ---
    ax = axes[0, 0]
    ax.plot(t, mass,            label=r"$\int\rho\,dV$ (solid)", color="steelblue")
    ax.plot(t, mass_exact_out,  label="exact domain mass out",   color="green",     ls="--")
    ax.plot(t, mass_model_out,  label=r"CMA model: $(ṁ_c+ṁ_g)A\,dt$", color="darkorange", ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass per unit area [kg/m²]")
    ax.set_title("Mass balance")
    ax.legend(fontsize=8)
    ax.axvline(60, color="gray", ls=":", lw=1)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: CMA mass residual ---
    ax = axes[0, 1]
    ax.plot(t, mass_cma_res * 1e3, color="crimson")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass [g/m²]")
    ax.set_title(
        r"CMA mass residual: $m^\mathrm{exact}_\mathrm{out} - m^\mathrm{model}_\mathrm{out}$"
        "\n(non-zero when $\\rho_\\mathrm{wall} \\neq \\rho_\\mathrm{char}$)"
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(60, color="gray", ls=":", lw=1)
    ax.grid(True, alpha=0.3)
    max_res = np.max(np.abs(mass_cma_res)) * 1e3
    ax.text(0.02, 0.96, f"max |residual| = {max_res:.3g} g/m²",
            transform=ax.transAxes, va="top", fontsize=9)

    # --- Panel 3: energy stored vs surface input ---
    ax = axes[1, 0]
    ax.plot(t, dE   / 1e6, label=r"$\Delta(\int\rho h\,dV)$ stored",  color="steelblue")
    ax.plot(t, E_in / 1e6, label=r"$\int q_\mathrm{cond}\,A\,dt$ input", color="darkorange", ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy per unit area [MJ/m²]")
    ax.set_title("Energy: stored vs conduction input")
    ax.legend(fontsize=8)
    ax.axvline(60, color="gray", ls=":", lw=1)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: net internal energy source ---
    ax = axes[1, 1]
    ax.plot(t, E_src / 1e6, color="purple")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(60, color="gray", ls=":", lw=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [MJ/m²]")
    ax.set_title(
        r"Net internal source: $\Delta E_\mathrm{stored} - \int q_\mathrm{cond}\,A\,dt$"
        "\n(decomposition + gas advection contribution)"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = save_plot_atomic(fig, OUT_PNG)
    print(f"Saved: {out}")

    # Print summary
    print(f"\nMass conservation summary:")
    print(f"  Initial solid mass      : {mass[0]:.4f} kg/m²")
    print(f"  Final   solid mass      : {mass[-1]:.4f} kg/m²")
    print(f"  Exact domain mass out   : {mass_exact_out[-1]:.4f} kg/m²")
    print(f"  CMA model mass out      : {mass_model_out[-1]:.4f} kg/m²")
    print(f"  CMA residual (max)      : {max_res:.3g} g/m²")
    print(f"\nEnergy balance summary:")
    print(f"  Energy stored ΔE        : {dE[-1]/1e6:.3f} MJ/m²")
    print(f"  Cumulative q_cond·A     : {E_in[-1]/1e6:.3f} MJ/m²")
    print(f"  Net internal source     : {E_src[-1]/1e6:.3f} MJ/m²")


if __name__ == "__main__":
    main()
