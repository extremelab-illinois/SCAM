# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_2.x_multiMat (3-layer stack).

PATO setup:
  - porousMat (TACOT_v3): 5.80 cm thick (7.21 → 1.41 cm from bottom), 70 cells graded
  - subMat1 (Fourier):    0.14 cm thick (1.41 → 1.27 cm),  2 cells
  - subMat2 (Fourier):    1.27 cm thick (1.27 → 0.00 cm), 10 cells
  - Fourier properties: k = 1 W/m/K, cp = 1000 J/kg/K, rho = 1000 kg/m³
  - Same BC as base 2.x: rhoUeCH = 0.3 kg/m²/s, h_r = 1.5e6 J/kg, p_e = 101325 Pa
  - Pyrolysis gas blowing, surface recession, adiabatic back face

PATO probe coordinates (y from mesh bottom):
  porousMat: y = 0.0721, 0.0711, 0.0701, 0.0681, 0.0641, 0.0601, 0.0561, 0.0481, 0.0268 m
  subMat1:   y = 0.0134 m  (midpoint of subMat1 layer)
  subMat2:   y = 0.0100 m  (midpoint of subMat2 layer)

SCAM configuration:
  - rhoUeCH = 0.3 kg/m²/s,  h_r = 1.5e6 J/kg  (exact PATO values)
  - Surface chemistry: live Cantera backend with SEB advective terms
  - Recession: s_dot = m_dot_char / rho_char  (CMA model)
  - TACOT mesh: 71 nodes, grading=10 (0.21 mm at surface → 2.11 mm at back)
    matches PATO's simpleGrading (1 0.1 1): last/first = 0.1 → 10× finer at surface

Sub-layer TC status:
  With grading=10 the TACOT cell adjacent to subMat1 is ~2.11 mm wide (centre 1.05 mm
  from interface) — identical to PATO's mesh.  The sub-layer temperature rise is only
  ~1.2 K (subMat1) and ~0.5 K (subMat2) over 120 s; SCAM shows a ~0.5 K / 0.3 K
  absolute discrepancy vs PATO at t=120 s, which is a ~40 % relative error in the
  temperature rise.  This residual gap is likely a true physics difference (interface
  conductance formulation or in-depth energy balance), not a mesh-mismatch artefact.

PATO reference data:
  pato-3.1/src/.../ref/1D/AblationTestCase_2.x_multiMat/output/

Run:
    MPLBACKEND=Agg python3 examples/ablation2/compare_pato_ablation2_multiMat.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pato2_common import (
    C_M, P_EDGE, PATO_REF, PATO_TUTS, RHOUE_CH, H_R, T_END,
    T_INIT,
    load_pato_ta_plot, load_pato_ta_surfacepatch,
    prepare_plot_output, run_pato, save_plot_atomic, T_at_original_depths,
)
from compare_pato_ablation2 import _DefaultZCBackend, load_tacot_material

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CASE_NAME  = "AblationTestCase_2.x_multiMat"
PATO_CASE  = PATO_TUTS / CASE_NAME
PATO_OUT   = PATO_REF  / CASE_NAME / "output"
TA_PLOT_PM = PATO_OUT  / "porousMat/scalar/Ta_plot"
TA_SURF_PM = PATO_OUT  / "porousMat/scalar/Ta_surfacePatch"
TA_PLOT_S1 = PATO_OUT  / "subMat1/scalar/Ta_plot"
TA_PLOT_S2 = PATO_OUT  / "subMat2/scalar/Ta_plot"
OUT_PNG = Path(__file__).parent / "compare_pato_ablation2_multiMat.png"
FOURIER_YAML = REPO / "scam/materials/subsurface/fourier.yaml"

# ---------------------------------------------------------------------------
# Geometry: TACOT top surface at y_bottom = 0 in SCAM mesh.
# Layers ordered hot-face first (SCAM convention: layers[0] = surface).
# ---------------------------------------------------------------------------
TACOT_THICKNESS  = 0.0580   # m  (7.21 - 1.41 cm)
SUBMAT1_THICKNESS = 0.0014  # m  (1.41 - 1.27 cm)
SUBMAT2_THICKNESS = 0.0127  # m  (1.27 - 0.00 cm)

TACOT_NODES  = 71    # 70 cells → 71 nodes
SUBMAT1_NODES = 3    #  2 cells →  3 nodes
SUBMAT2_NODES = 11   # 10 cells → 11 nodes

# PATO probe depths from TACOT original surface [m]:
#   porousMat probes: 0, 1, 2, 4, 8, 12, 16, 24, 45.3 mm
#   subMat1 probe:   58.7 mm  (y=0.0134m from bottom → 7.21-1.34=5.87cm depth)
#   subMat2 probe:   62.1 mm  (y=0.0100m from bottom → 7.21-1.00=6.21cm depth)
TC_TACOT_MM  = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 45.3]
TC_TACOT     = [d / 1000 for d in TC_TACOT_MM]
TC_SUBMAT1   = [0.0587]   # m  — midpoint of subMat1 (y=0.0134 m from global bottom)
TC_SUBMAT2   = [0.0621]   # m  — 0.0027 m below subMat1 interface (21% from top, NOT midpoint;
                            #      y=0.0100 m from global bottom per PATO plotDict)
TC_ALL       = TC_TACOT + TC_SUBMAT1 + TC_SUBMAT2   # 11 depths total

# ---------------------------------------------------------------------------
# BC — enthalpy-based (identical to base 2.x)
# ---------------------------------------------------------------------------
_T_STEPS    = np.array([0.0, 0.099, 0.1, 60.0, 60.1, T_END])
_RHOUE_VALS = np.array([0.0, 0.0, RHOUE_CH, RHOUE_CH, 0.0, 0.0])
_HR_VALS    = np.array([0.0, 0.0, H_R,      H_R,      0.0, 0.0])


def _rhoUeCH(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _RHOUE_VALS))


def _h_r(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _HR_VALS))


def _rhoue(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _RHOUE_VALS))


# ---------------------------------------------------------------------------
# Run SCAM
# ---------------------------------------------------------------------------

def _make_bprime():
    """Live Cantera backend with Cantera-derived fallback Z_C_pyro."""
    try:
        from scam.physics.bprime_evaluator import BprimeEvaluator
        ev = BprimeEvaluator.from_config(str(REPO / "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"))
        z_c_pyro = ev.pyrolysis_target_fraction("C")
        ev.lookup(1500.0, P_EDGE, 0.05, z_c_pyro)   # fail-fast probe
        print(f"  Surface chemistry: live Cantera (Cantera-derived Z_C_pyro={z_c_pyro:.6f}, + SEB advective terms)")
        return _DefaultZCBackend(ev, z_c_pyro)
    except Exception as e:
        raise RuntimeError(
            "Live Cantera is required for this validation example; "
            "table-only fallback produces a misleading cold surface comparison. "
            f"The current interpreter is {sys.executable!r}. Run with the "
            "Cantera-enabled Python used by this workspace or install "
            "Cantera into this interpreter."
        ) from e


def run_scam():
    tacot, _ = load_tacot_material()
    fourier, _ = load_material(str(FOURIER_YAML))
    four1 = dataclasses.replace(fourier, name="Fourier1")
    four2 = dataclasses.replace(fourier, name="Fourier2")

    bpt = _make_bprime()
    mat_cards      = {tacot.name: tacot, four1.name: four1, four2.name: four2}
    b_prime_tables = {tacot.name: bpt,   four1.name: None,   four2.name: None}

    # layers[0] = hot face (TACOT ablative),
    # layers[1] = subMat1 (Fourier), layers[2] = subMat2 (Fourier back)
    stack = StackConfig(layers=[
        # grading=10: cells 10× coarser at back face than at surface — matches PATO's
        # simpleGrading (1 0.1 1) which gives last/first = 0.1 (1/0.1 = 10 in SCAM convention).
        LayerConfig(tacot.name, thickness=TACOT_THICKNESS,   n_nodes=TACOT_NODES,  n_subcells=4, grading=10.0),
        LayerConfig(four1.name, thickness=SUBMAT1_THICKNESS, n_nodes=SUBMAT1_NODES, n_subcells=2),
        LayerConfig(four2.name, thickness=SUBMAT2_THICKNESS, n_nodes=SUBMAT2_NODES, n_subcells=2),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=_rhoUeCH,
        h_r=_h_r,
        # emissivity < 0 → use the material's tables (TACOT_3.0.xls: virgin 0.8 →
        # char 0.9) blended by char fraction, instead of a hardcoded override.
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=300.0,   # PATO Tbackground=300 K (AblationTestCase_2.x_multiMat Ta BC)
        rho_e_u_e=_rhoue,
        C_M=C_M,
        p_e=P_EDGE,
        lambda_blowing=0.5,
        blowing_model="lees",   # PATO-consistent: log(1+Phi)/Phi + blown B'_g
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.5,
        dt_min=1e-5,
        dt_max_dT=30.0,
        dt_max_drho_frac=0.05,
        output_dt=1.0,    # 1 s to match PATO ref resolution
        tc_positions=[],  # post-process at fixed original depths
        allow_recession=True,
        use_rho_old=True,
        continuous_remap=True,   # continuous moving-mesh (ALE) — no node-drop sawtooth
    )

    print("Running SCAM (AblationTestCase_2.x_multiMat / TACOT 3.0 YAML) ...")
    return run(stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc,
               options, initial_T=T_INIT, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    prepare_plot_output(OUT_PNG)
    run_pato(PATO_CASE, TA_PLOT_PM)

    print("Loading PATO reference data ...")
    t_pm, T_pm = load_pato_ta_plot(TA_PLOT_PM)    # (N,), (N, 9) — porousMat 9 probes
    t_ps, T_ps = load_pato_ta_surfacepatch(TA_SURF_PM)
    t_s1, T_s1 = load_pato_ta_plot(TA_PLOT_S1)    # subMat1 midpoint
    t_s2, T_s2 = load_pato_ta_plot(TA_PLOT_S2)    # subMat2 midpoint

    results  = run_scam()
    t_s      = results.times_array()
    T_wall_s = results.T_wall_array()
    rec_s    = results.s_array()
    T_tc     = T_at_original_depths(results, TC_ALL)  # (11, n_steps)

    # T_tc rows: [0..8] = TACOT probes, [9] = subMat1, [10] = subMat2
    T_tc_pm  = T_tc[:9, :]
    T_tc_s1  = T_tc[9,  :]
    T_tc_s2  = T_tc[10, :]

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

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        "SCAM vs PATO AblationTestCase_2.x_multiMat\n"
        "TACOT 3.0 YAML (5.8 cm) + Fourier sub-layer 1 (0.14 cm) + Fourier sub-layer 2 (1.27 cm)",
        fontsize=11,
    )

    # Panel 1 — surface T
    ax = axes[0, 0]
    ax.plot(t_s,  T_wall_s,    "r-",  lw=2,   label="SCAM T_wall")
    ax.plot(t_ps, T_ps[:, 0],  "r--", lw=1.5, label="PATO T_surface")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface temperature"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2 — TACOT TCs at 1, 8, 24 mm
    ax = axes[0, 1]
    idx_show = [1, 4, 7]   # 1 mm, 8 mm, 24 mm from TC_TACOT_MM
    for k, col in zip(idx_show, colors):
        d_mm = TC_TACOT_MM[k]
        ax.plot(t_s, T_tc_pm[k],       color=col, lw=2,   label=f"SCAM {d_mm:.0f} mm")
        ax.plot(t_pm, T_pm[:, k],      color=col, lw=1.5, ls="--", label=f"PATO {d_mm:.0f} mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TACOT TCs — 1, 8, 24 mm depth"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3 — sub-layer TCs
    ax = axes[0, 2]
    ax.plot(t_s,  T_tc_s1,      "b-",  lw=2,   label="SCAM subMat1 (58.7 mm)")
    ax.plot(t_s1, T_s1[:, 0],   "b--", lw=1.5, label="PATO subMat1")
    ax.plot(t_s,  T_tc_s2,      "g-",  lw=2,   label="SCAM subMat2 (62.1 mm)")
    ax.plot(t_s2, T_s2[:, 0],   "g--", lw=1.5, label="PATO subMat2")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Sub-layer TCs"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim([300, 305])
    # Panel 4 — T profile snapshots at t=30 s and t=90 s
    ax = axes[1, 0]
    snap_30 = snap_90 = None
    for snap in results.snapshots:
        if abs(snap.time - 30) < 0.6 and snap_30 is None:
            snap_30 = snap
        if abs(snap.time - 90) < 0.6 and snap_90 is None:
            snap_90 = snap
    i30_p = int(np.argmin(np.abs(t_pm - 30)))
    i90_p = int(np.argmin(np.abs(t_pm - 90)))
    pato_d_mm = TC_TACOT_MM
    if snap_30:
        # y_nodes is already measured from the original front face; adding
        # s_total would double-count recession and shift the profile too deep.
        y30 = snap_30.mesh.y_nodes * 1000
        ax.plot(y30, snap_30.T, "b-", lw=2, label=f"SCAM t≈30 s")
    if snap_90:
        y90 = snap_90.mesh.y_nodes * 1000
        ax.plot(y90, snap_90.T, "r-", lw=2, label=f"SCAM t≈90 s")
    valid_30 = T_pm[i30_p, :] > 0
    valid_90 = T_pm[i90_p, :] > 0
    ax.plot(np.array(pato_d_mm)[valid_30], T_pm[i30_p, valid_30],
            "b--o", ms=4, lw=1.5, label=f"PATO t≈{t_pm[i30_p]:.0f} s")
    ax.plot(np.array(pato_d_mm)[valid_90], T_pm[i90_p, valid_90],
            "r--o", ms=4, lw=1.5, label=f"PATO t≈{t_pm[i90_p]:.0f} s")
    # Add sub-layer probes at t=30 and t=90
    i30_s = int(np.argmin(np.abs(t_s1 - 30)))
    i90_s = int(np.argmin(np.abs(t_s1 - 90)))
    ax.plot([TC_SUBMAT1[0]*1000], [T_s1[i30_s, 0]], "b^", ms=7)
    ax.plot([TC_SUBMAT2[0]*1000], [T_s2[i30_s, 0]], "b^", ms=7)
    ax.plot([TC_SUBMAT1[0]*1000], [T_s1[i90_s, 0]], "r^", ms=7)
    ax.plot([TC_SUBMAT2[0]*1000], [T_s2[i90_s, 0]], "r^", ms=7)
    # Layer boundary markers
    ax.axvline(TACOT_THICKNESS * 1000, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.axvline((TACOT_THICKNESS + SUBMAT1_THICKNESS) * 1000, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Depth from original surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles (▲ = PATO sub-layer probes)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 5 — recession
    ax = axes[1, 1]
    ax.plot(t_s, rec_s * 1000, "b-", lw=2, label="SCAM recession")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Recession [mm]")
    ax.set_title("Surface recession"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 6 — SCAM − PATO difference at 1, 8, 24 mm (TACOT) + sub-layers
    ax = axes[1, 2]
    for k, col in zip(idx_show, colors):
        d_mm = TC_TACOT_MM[k]
        diff = T_tc_pm[k] - np.interp(t_s, t_pm, T_pm[:, k])
        diff[np.isnan(T_tc_pm[k])] = np.nan
        ax.plot(t_s, diff, color=col, lw=1.5, label=f"TACOT {d_mm:.0f} mm")
    diff_s1 = T_tc_s1 - np.interp(t_s, t_s1, T_s1[:, 0])
    diff_s2 = T_tc_s2 - np.interp(t_s, t_s2, T_s2[:, 0])
    ax.plot(t_s, diff_s1, "k-",  lw=1.5, label="subMat1 (58.7 mm)")
    ax.plot(t_s, diff_s2, "k--", lw=1.5, label="subMat2 (62.1 mm)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("Temperature difference SCAM − PATO")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_atomic(fig, OUT_PNG, dpi=150)
    print(f"\nPlot saved → {OUT_PNG}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n--- SCAM vs PATO: max |ΔT| ---")
    for k in range(len(TC_TACOT)):
        d_mm = TC_TACOT_MM[k]
        diff = T_tc_pm[k] - np.interp(t_s, t_pm, T_pm[:, k])
        valid = ~np.isnan(diff)
        if valid.any():
            print(f"  TACOT {d_mm:5.1f} mm : {np.abs(diff[valid]).max():.1f} K")
    diff_s1_v = T_tc_s1 - np.interp(t_s, t_s1, T_s1[:, 0])
    diff_s2_v = T_tc_s2 - np.interp(t_s, t_s2, T_s2[:, 0])
    print(f"  subMat1  58.7 mm : {np.abs(diff_s1_v).max():.1f} K")
    print(f"  subMat2  62.1 mm : {np.abs(diff_s2_v).max():.1f} K")
    print(f"\n  Recession at t=120 s: {rec_s[-1]*1000:.2f} mm")


if __name__ == "__main__":
    main()
