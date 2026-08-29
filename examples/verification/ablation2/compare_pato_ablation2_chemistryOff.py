# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_2.x_chemistryOff.

PATO setup (chemistry disabled — no B' char ablation):
  - 5 cm TACOT_v3 slab, T_init = 300 K
  - Convective BC: hconv = 300 W/m²/K, Tedge = 2800 K  (heating)
                   hconv = 1 W/m²/K,   Tedge = 300 K   (cooldown)
  - Tbackground = 300 K (radiation)
  - Heating t = 0.1 → 60 s; cooling t = 60.1 → 120 s
  - No char ablation; pyrolysis gas blowing only (no recession)
  - Adiabatic back face

SCAM BC: When chemistryOn=0 the PATO Bprime BC ignores rhoUeCH/h_r entirely
  and applies only q_conv = hconv*(Tedge - T).  Map directly:
    heating:  alpha_conv = 300 W/m²/K, T_aw = 2800 K
    cooldown: alpha_conv = 1 W/m²/K,   T_aw = 300 K
  - No B' table (b_prime_tables = None → no char erosion)
  - allow_recession = False

PATO reference data:
  pato-3.1/src/.../ref/1D/AblationTestCase_2.x_chemistryOff/output/

Run:
    MPLBACKEND=Agg python3 examples/ablation2/compare_pato_ablation2_chemistryOff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pato2_common import (
    PATO_REF, PATO_TUTS, T_END, T_INIT, TC_DEPTHS,
    load_pato_ta_plot, load_pato_ta_surfacepatch,
    prepare_plot_output, run_pato, save_plot_atomic, T_at_original_depths,
)
from compare_pato_ablation2 import load_tacot_material

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CASE_NAME  = "AblationTestCase_2.x_chemistryOff"
PATO_CASE  = PATO_TUTS / CASE_NAME
PATO_OUT   = PATO_REF  / CASE_NAME / "output/porousMat"
TA_PLOT    = PATO_OUT  / "scalar/Ta_plot"
TA_SURF    = PATO_OUT  / "scalar/Ta_surfacePatch"
OUT_PNG = Path(__file__).parent / "compare_pato_ablation2_chemistryOff.png"
THICKNESS = 0.05
N_NODES   = 201
TC_LABELS_MM = [1, 2, 4, 8, 12, 16, 24]

# ---------------------------------------------------------------------------
# BC: PATO Bprime BC with chemistryOn=0 uses ONLY hconv*(Tedge - T).
# The rhoUeCH/h_r columns in BoundaryConditions are ignored when chemistryOn=0.
# From BoundaryConditions:
#   t=0.1–60s:   hconv=300 W/m²/K, Tedge=2800 K  (heating)
#   t=60.1–120s: hconv=1 W/m²/K,   Tedge=300 K   (cooldown, essentially radiation only)
# ---------------------------------------------------------------------------
_BC_T = np.array([0.0, 0.099, 0.1,   60.0,  60.1, T_END])
_BC_A = np.array([1.0, 1.0,   300.0, 300.0,  1.0,  1.0])    # hconv [W/m²/K]
_BC_W = np.array([300, 300,   2800., 2800.,  300., 300.])    # Tedge [K]


def _alpha(t: float) -> float:
    return float(np.interp(t, _BC_T, _BC_A))


def _T_aw(t: float) -> float:
    return float(np.interp(t, _BC_T, _BC_W))


# ---------------------------------------------------------------------------
# Run SCAM
# ---------------------------------------------------------------------------

def run_scam():
    mat, _ = load_tacot_material()
    mat_cards      = {mat.name: mat}
    b_prime_tables = {mat.name: None}   # chemistry OFF

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=N_NODES, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=_alpha,
        T_aw=_T_aw,
        # emissivity < 0 → use the material's tables (TACOT_3.0.xls: virgin 0.8 →
        # char 0.9) blended by char fraction, instead of a hardcoded override.
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=300.0,   # PATO Tbackground=300 K (AblationTestCase_2.x_chemistryOff Ta BC)
        rho_e_u_e=0.0,   # no blowing correction when chemistry is off
        C_M=0.0,
        p_e=101325.0,
        lambda_blowing=0.5,
        blowing_model="lees",   # PATO-consistent (no-op here since rho_e_u_e=0)
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.5,
        dt_min=1e-5,
        dt_max_dT=30.0,
        output_dt=0.5,
        tc_positions=[],
        allow_recession=False,  # chemistry off → no char erosion
        use_rho_old=True,
    )

    print("Running SCAM (AblationTestCase_2.x_chemistryOff / TACOT 3.0 YAML) ...")
    return run(stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc,
               options, initial_T=T_INIT, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    prepare_plot_output(OUT_PNG)
    run_pato(PATO_CASE, TA_PLOT)

    print("Loading PATO reference data ...")
    t_p, T_p   = load_pato_ta_plot(TA_PLOT)
    t_ps, T_ps = load_pato_ta_surfacepatch(TA_SURF)

    results  = run_scam()
    t_s      = results.times_array()
    T_wall_s = results.T_wall_array()
    T_tc     = T_at_original_depths(results, TC_DEPTHS)  # (7, n_steps)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    colors = ["tab:blue", "tab:orange", "tab:green",
              "tab:red",  "tab:purple", "tab:brown", "tab:pink"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        "SCAM vs PATO AblationTestCase_2.x_chemistryOff\n"
        "TACOT 3.0 YAML | convective BC only (hconv=300 W/m²/K, T_edge=2800 K) | no char ablation",
        fontsize=11,
    )

    # Panel 1 — surface T
    ax = axes[0, 0]
    ax.plot(t_s,  T_wall_s,    "r-",  lw=2,   label="SCAM T_wall")
    ax.plot(t_ps, T_ps[:, 0],  "r--", lw=1.5, label="PATO T_surface")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface temperature"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2 — TC at 1, 4, 12 mm (full run)
    ax = axes[0, 1]
    for i, col in zip([0, 2, 4], colors):
        d_mm = TC_LABELS_MM[i]
        ax.plot(t_s, T_tc[i],        color=col, lw=2,   label=f"SCAM {d_mm} mm")
        ax.plot(t_p, T_p[:, i],      color=col, lw=1.5, ls="--", label=f"PATO {d_mm} mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC temperatures — 1, 4, 12 mm"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3 — T profiles at t=30 s and t=90 s
    ax = axes[0, 2]
    snap_30 = snap_90 = None
    for snap in results.snapshots:
        if abs(snap.time - 30) < 0.3 and snap_30 is None:
            snap_30 = snap
        if abs(snap.time - 90) < 0.3 and snap_90 is None:
            snap_90 = snap
    i30 = int(np.argmin(np.abs(t_p - 30)))
    i90 = int(np.argmin(np.abs(t_p - 90)))
    pato_d_mm = [d * 1000 for d in TC_DEPTHS]
    if snap_30:
        ax.plot(snap_30.mesh.y_nodes * 1000, snap_30.T, "b-", lw=2, label="SCAM t=30 s")
    if snap_90:
        ax.plot(snap_90.mesh.y_nodes * 1000, snap_90.T, "r-", lw=2, label="SCAM t=90 s")
    ax.plot(pato_d_mm, T_p[i30, :], "b--o", ms=4, lw=1.5, label=f"PATO t≈{t_p[i30]:.0f} s")
    ax.plot(pato_d_mm, T_p[i90, :], "r--o", ms=4, lw=1.5, label=f"PATO t≈{t_p[i90]:.0f} s")
    ax.set_xlabel("Depth from surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t≈30 s and t≈90 s")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4 — SCAM − PATO surface T difference (mask flux-cutoff sampling cliff)
    ax = axes[1, 0]
    T_ps_interp = np.interp(t_s, t_ps, T_ps[:, 0])
    diff_surf = T_wall_s - T_ps_interp
    diff_surf_masked = diff_surf.copy()
    diff_surf_masked[(t_s > 59.9) & (t_s < 61.1)] = np.nan
    ax.plot(t_s, diff_surf_masked, "r-", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("Surface T difference SCAM − PATO"); ax.grid(True, alpha=0.3)

    # Panel 5 — SCAM − PATO TC difference at 1, 4, 12 mm
    ax = axes[1, 1]
    for i, col in zip([0, 2, 4], colors):
        d_mm = TC_LABELS_MM[i]
        diff = T_tc[i] - np.interp(t_s, t_p, T_p[:, i])
        ax.plot(t_s, diff, color=col, lw=1.5, label=f"{d_mm} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("TC difference SCAM − PATO"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 6 — all 7 TC final T comparison (bar chart at t=60 s)
    ax = axes[1, 2]
    i60s = int(np.argmin(np.abs(t_s - 60)))
    i60p = int(np.argmin(np.abs(t_p - 60)))
    d_arr = np.array(TC_LABELS_MM, dtype=float)
    T_s60 = np.array([T_tc[i, i60s] for i in range(7)])
    T_p60 = T_p[i60p, :]
    x = np.arange(7)
    w = 0.35
    ax.bar(x - w/2, T_s60,  w, label="SCAM", color="steelblue",  alpha=0.8)
    ax.bar(x + w/2, T_p60,  w, label="PATO", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}" for d in d_arr])
    ax.set_xlabel("Depth from surface [mm]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC temperatures at t = 60 s (end of heating)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_plot_atomic(fig, OUT_PNG, dpi=150)
    print(f"\nPlot saved → {OUT_PNG}")

    # Phase masks — exclude flux-cutoff transition window [59.9, 61.1]s
    # (PATO 1-s output linearly interpolates the step; SCAM resolves it, creating
    #  a ~300K apparent gap that is purely a sampling artifact)
    mask_heat = (t_s >= 2.0)  & (t_s <= 59.9)
    mask_cool = (t_s >= 61.1) & (t_s <= T_END)

    T_ps_interp_all = np.interp(t_s, t_ps, T_ps[:, 0])
    surf_diff = T_wall_s - T_ps_interp_all

    print("\n--- SCAM vs PATO: max |ΔT| by phase (transition [59.9–61.1 s] excluded) ---")
    print(f"  Surface T — heating  (2–60 s):  {np.abs(surf_diff[mask_heat]).max():.1f} K")
    print(f"  Surface T — cooling (>61.1 s):  {np.abs(surf_diff[mask_cool]).max():.1f} K")
    print(f"  TC depths:")
    for i in range(7):
        diff = T_tc[i] - np.interp(t_s, t_p, T_p[:, i])
        print(f"    {TC_LABELS_MM[i]:2d} mm — heat {np.abs(diff[mask_heat]).max():.1f} K  cool {np.abs(diff[mask_cool]).max():.1f} K")


if __name__ == "__main__":
    main()
