# SPDX-License-Identifier: MIT
"""Compare SCAM vs Amar (2006) §8.8 — Thermochemical Ablation of a Decomposing Material.

Material: Carbon-phenolic composite (Appendix D) — decomposing, Darcy porous flow.
BC: Same ballistic reentry trajectory as §8.7 (Figs 8.14 and 8.15).

Note on the char specific heat:
    `carbon_phenolic.yaml` deliberately does NOT use Amar's Table D.4 cp column,
    which is erroneous (0.05 Btu/lbm-degR at 500 degR = 209 J/kg/K at 278 K —
    physically impossible for a carbonized char). It uses Sutton, NASA TN D-5930
    (1970), Table VI(b) instead. With D.4's values the back face overshoots the
    Fig 8.29 reference by +169 degR at t=50 s; with Sutton's it agrees to within
    +/-10 degR across the whole transient, with peak surface T and recession
    unchanged. See docs/verification/verification.md section 8.8.

Reference: Amar, Adam J. (2006), "Modeling of One-Dimensional Ablation with Porous
           Flow Using Finite Control Volume Procedure," PhD thesis, Chapter 8.8.
           Comparison against CMA code results (Figs 8.28–8.40).

Note on Darcy modeling:
    The Darcy energy source is activated automatically when mat.permeability > 0.
    The carbon_phenolic.yaml carries permeability = 1.003e-13 m² (char endpoint of
    Table D.5). Gas porosity eps_g_virgin/eps_g_char are both 0 (not populated in
    the material card from the table), so the Darcy pressure RHS is zero and the
    pressure field is uniform at p_surface. Pyrolysis gas flow is computed from
    mass conservation (standard SCAM approach), which is consistent with the limit
    of fast pressure equilibration via high-permeability Darcy flow. The impermeable
    back face is the default Neumann BC (dp/dy = 0) in the pressure_darcy solver.

Note on surface pressure for B' lookup and Darcy BC:
    The edge pressure p_e(t) from Fig 8.14 is passed both for the B' table lookup
    and as the Darcy solver's surface Dirichlet BC (p(y=0) = p_e(t), evaluated at
    the current time each step) — matching Amar's implementation. The effect on
    energy balance is small given the low permeability of the virgin material.

Note on Fig 8.15 files:
    figure8.15_heat_transfer_coefficient.csv → HTC ρ_e u_e C_H [lbm/ft²·s], peak ≈ 0.49
    figure8.15_edge_velocity.csv              → edge velocity [ft/s], start ≈ 22400 ft/s
Edge velocity is not directly used by SCAM; only the HTC file is loaded.

Run:
    MPLBACKEND=Agg python3 examples/verification/amar_thesis/compare_amar_cp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "digitized_data"   # digitized BC + reference curves
MAT_YAML = REPO / "scam/materials/ablative_organic/carbon_phenolic.yaml"
OUT_PNG = HERE / "compare_amar_cp.png"

# §8.8 uses the same trajectory as §8.7 and the same hot-wall / Kays blowing
# corrections.  That shared machinery lives in _amar_common so this case does
# NOT depend on the §8.7 script.
import _amar_common as CC

T_TRANSITION = CC.T_TRANSITION   # 23.5 s laminar -> turbulent
T_RAD_IN = 414.0 * (5.0 / 9.0)   # 414 °R far-field radiation source (§8.8.1) = 230 K

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
BTU_LBM_TO_J_KG = 2326.0          # h_r  [Btu/lbm → J/kg]
LBM_FT2S_TO_KG_M2S = 4.8824       # HTC  [lbm/ft²·s → kg/m²·s]
ATM_TO_PA = 101325.0               # p_e  [atm → Pa]
IN_S_TO_M_S = 0.0254               # ṡ    [in/s → m/s]
R_TO_K = 5.0 / 9.0                 # T    [°R → K]

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
THICKNESS = 0.0127       # [m]  0.5 inch (same geometry as §8.7)
N_NODES   = 100
T_INIT    = 297.04       # [K]  ~534°R — uniform initial temperature, Fig 8.32
T_END     = 50.0         # [s]

# Profile snapshot times for comparison with Figs 8.32–8.40
SNAP_TIMES = [10.0, 20.0, 25.0, 27.5, 30.0, 32.5, 35.0, 40.0, 50.0]

# ---------------------------------------------------------------------------
# Load BC from digitized CSVs
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load two-column CSV, sort by time, return (t, v)."""
    data = np.loadtxt(str(path), delimiter=",")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def _build_bc_callables():
    """Return (rhoUeCH_fn, h_r_fn, p_e_fn, u_e_fn) as callables f(t) -> float."""
    t_hr, hr_btu = _load_csv(DATA_DIR / "figure8.14_recovery_enthalpy.csv")
    hr_Jkg = np.maximum(hr_btu, 0.0) * BTU_LBM_TO_J_KG

    t_p, p_atm = _load_csv(DATA_DIR / "figure8.14_pressure.csv")
    p_Pa = np.maximum(p_atm, 0.0) * ATM_TO_PA

    t_htc, htc_lbm = _load_csv(DATA_DIR / "figure8.15_heat_transfer_coefficient.csv")
    htc_SI = np.maximum(htc_lbm, 0.0) * LBM_FT2S_TO_KG_M2S
    htc_SI = CC.smooth_laminar_htc(t_htc, htc_SI)

    t_ue, ue_ft_s = _load_csv(DATA_DIR / "figure8.15_edge_velocity.csv")
    ue_m_s = np.maximum(ue_ft_s, 0.0) * 0.3048

    def rhoUeCH_fn(t: float) -> float:
        return float(np.interp(t, t_htc, htc_SI, left=0.0, right=0.0))

    def h_r_fn(t: float) -> float:
        return float(np.interp(t, t_hr, hr_Jkg, left=float(hr_Jkg[0]), right=0.0))

    def p_e_fn(t: float) -> float:
        return float(np.interp(t, t_p, p_Pa, left=float(p_Pa[0]), right=float(p_Pa[-1])))

    def u_e_fn(t: float) -> float:
        return float(np.interp(t, t_ue, ue_m_s, left=float(ue_m_s[0]), right=float(ue_m_s[-1])))

    return rhoUeCH_fn, h_r_fn, p_e_fn, u_e_fn


# ---------------------------------------------------------------------------
# SCAM simulation
# ---------------------------------------------------------------------------

def run_scam() -> object:
    mat, b_prime = load_material(str(MAT_YAML))
    rhoUeCH_fn, h_r_fn, p_e_fn, u_e_fn = _build_bc_callables()
    # §8.8.1: "The Stanton number is corrected for both hot wall and blowing
    # effects."  Use the same faithful Amar et al. (2008) corrections as §8.7:
    # Cohen-Reshotko (laminar) / Eckert (turbulent) hot-wall factor applied to
    # BOTH heat and mass (Eqs 29-30), and the exact Kays blowing law.
    wall_correction = CC.AmarWallCorrection(h_r_fn, p_e_fn, u_e_fn)

    stack = StackConfig(layers=[LayerConfig(mat.name, THICKNESS, N_NODES, 4)])

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=rhoUeCH_fn,
        h_r=h_r_fn,
        p_e=p_e_fn,
        rho_e_u_e=rhoUeCH_fn,
        C_M=1.0,
        # Emissivity from material card (0.85); -1.0 means "use material card"
        emissivity=-1.0,
        T_rad_in=T_RAD_IN,   # 414 °R far-field radiation source (§8.8.1)
        view_factor=1.0,
        lambda_blowing=lambda t: 0.5 if t < T_TRANSITION else 0.4,
        blowing_model="kays",
        stanton_wall_correction=wall_correction,
        apply_wall_correction_to_heat=True,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    # Record profiles at the comparison snapshot times
    tc_positions: list[float] = []  # no fixed TC probes for CP (use full profiles)

    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.1,
        dt_min=1e-5,
        dt_max_dT=500.0,
        output_dt=0.25,
        allow_recession=True,
        continuous_remap=True,
        seb_tol=1.0,
    )

    return run(
        stack,
        {mat.name: mat},
        {mat.name: b_prime},
        geom,
        surface_bc,
        back_bc,
        options,
        initial_T=T_INIT,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_ref(filename: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = DATA_DIR / filename
    if p.exists():
        return _load_csv(p)
    return None


def _plot_ref(ax, basename: str, yscale: float = 1.0):
    """Overlay the digitized research-code (black) and CMA (red) curves.

    `basename` is the figure stem, e.g. 'figure8.28'.  Files follow the
    convention `{basename}_research_code.csv` / `{basename}_cma.csv` (with one
    legacy dot-typo, `figure8.30.research_code.csv`, handled here).
    """
    for suffix, color, label in (
        ("_research_code", "k", "Research code"),
        ("_cma", "r", "CMA"),
    ):
        ref = _try_load_ref(f"{basename}{suffix}.csv")
        if ref is None and suffix == "_research_code":
            ref = _try_load_ref(f"{basename}.research_code.csv")  # legacy typo
        if ref is not None:
            ax.plot(ref[0], ref[1] * yscale, color=color, ls="--", lw=1.5, label=label)


def _snap_at_time(snaps, t_target: float):
    """Return the snapshot closest to t_target."""
    times = [s.time for s in snaps]
    idx = int(np.argmin(np.abs(np.array(times) - t_target)))
    return snaps[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Running SCAM §8.8 carbon-phenolic ablation...")
    results = run_scam()

    snaps = results.snapshots
    times   = np.array([s.time         for s in snaps])
    T_wall  = np.array([s.T_wall       for s in snaps])
    s_total = np.array([s.mesh.s_total for s in snaps])
    s_dot   = np.array([s.s_dot        for s in snaps])
    T_back  = np.array([s.T[-1]        for s in snaps])

    T_wall_R  = T_wall  / R_TO_K
    T_back_R  = T_back  / R_TO_K
    s_total_in = s_total / IN_S_TO_M_S
    s_dot_ins  = s_dot   / IN_S_TO_M_S

    print(f"\n--- SCAM peak values (§8.8 carbon-phenolic) ---")
    idx = np.argmax(T_wall)
    print(f"T_surf peak:  {T_wall[idx]:.0f} K ({T_wall_R[idx]:.0f} °R)  at t={times[idx]:.1f} s")
    idx = np.argmax(T_back)
    print(f"T_back peak:  {T_back[idx]:.0f} K ({T_back_R[idx]:.0f} °R)  at t={times[idx]:.1f} s")
    idx = np.argmax(s_dot)
    print(f"ṡ_max:        {s_dot[idx]:.4e} m/s ({s_dot_ins[idx]:.4e} in/s) at t={times[idx]:.1f} s")
    print(f"s_total final:{s_total[-1]*1e3:.3f} mm  ({s_total_in[-1]:.4f} in)")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 4)

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_tb = fig.add_subplot(gs[0, 1])
    ax_s  = fig.add_subplot(gs[0, 2])
    ax_sd = fig.add_subplot(gs[0, 3])
    ax_prof = fig.add_subplot(gs[1, :])

    # T_surf vs time (Fig 8.28)
    ax_ts.plot(times, T_wall_R, 'b-', lw=2, label='SCAM')
    _plot_ref(ax_ts, "figure8.28")
    ax_ts.set_xlabel('Time [s]')
    ax_ts.set_ylabel('Surface temperature [°R]')
    ax_ts.set_title('T_surf — §8.8 C-Phenolic (Fig 8.28)')
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    # T_back vs time (Fig 8.29)
    ax_tb.plot(times, T_back_R, 'b-', lw=2, label='SCAM')
    _plot_ref(ax_tb, "figure8.29")
    ax_tb.set_xlabel('Time [s]')
    ax_tb.set_ylabel('Back face temperature [°R]')
    ax_tb.set_title('T_back — §8.8 C-Phenolic (Fig 8.29)')
    ax_tb.legend(fontsize=8)
    ax_tb.grid(True, alpha=0.3)

    # Total recession vs time (Fig 8.30, reference in inches)
    ax_s.plot(times, s_total_in * 1e3, 'b-', lw=2, label='SCAM')
    _plot_ref(ax_s, "figure8.30", yscale=1e3)
    ax_s.set_xlabel('Time [s]')
    ax_s.set_ylabel('Total recession [×10⁻³ in]')
    ax_s.set_title('Recession — §8.8 C-Phenolic (Fig 8.30)')
    ax_s.legend(fontsize=8)
    ax_s.grid(True, alpha=0.3)

    # Recession rate vs time (Fig 8.31, reference in in/s)
    ax_sd.plot(times, s_dot_ins * 1e3, 'b-', lw=2, label='SCAM')
    _plot_ref(ax_sd, "figure8.31", yscale=1e3)
    ax_sd.set_xlabel('Time [s]')
    ax_sd.set_ylabel('Recession rate [×10⁻³ in/s]')
    ax_sd.set_title('ṡ — §8.8 C-Phenolic (Fig 8.31)')
    ax_sd.legend(fontsize=8)
    ax_sd.grid(True, alpha=0.3)

    # Temperature profiles at snapshot times (Fig 8.32)
    #
    # Datum: the digitized Fig 8.32 curves are depth from the ORIGINAL front
    # face, not from the receded surface — each curve starts at that snapshot's
    # recession depth (e.g. the t=50 s curve starts at 0.139 in, matching
    # s_total=0.141 in). MeshState.y_nodes uses exactly that datum already
    # (y_nodes[0] == s_total), so SCAM is plotted as y_nodes with NO subtraction.
    # Subtracting y_nodes[0] here would shift SCAM left by the recession and
    # produce a spurious mismatch that grows with time.
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(SNAP_TIMES)))
    for i, t_snap in enumerate(SNAP_TIMES):
        snap = _snap_at_time(snaps, t_snap)
        depth_in = snap.mesh.y_nodes / IN_S_TO_M_S
        ax_prof.plot(depth_in, snap.T / R_TO_K, color=colors[i],
                     lw=1.5, label=f't={snap.time:.1f} s')
        # Digitized Fig 8.32 reference: figure8.32_{t}sec.csv, '.' -> 'p'
        stem = f"{t_snap:g}".replace(".", "p")
        ref = _try_load_ref(f"figure8.32_{stem}sec.csv")
        if ref:
            ax_prof.plot(ref[0], ref[1], color=colors[i], lw=1.5,
                        ls='--', alpha=0.7)

    ax_prof.set_xlabel('Depth from original front face [in]')
    ax_prof.set_ylabel('Temperature [°R]')
    ax_prof.set_title('Temperature profiles at snapshot times — §8.8 C-Phenolic\n'
                      '(dashed = CMA reference if available)')
    ax_prof.legend(fontsize=7, ncol=3)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(
        "SCAM vs Amar (2006) §8.8 — Carbon-Phenolic Thermochemical Ablation\n"
        "Ballistic reentry BC (Figs 8.14–8.15); "
        f"ρ_v={results.snapshots[0].rho[0]:.0f} kg/m³, "
        f"L={THICKNESS*1e3:.1f} mm, T₀={T_INIT:.0f} K, ε=0.85",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
