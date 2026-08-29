# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_2.x_equilibriumElementConservation.

PATO setup (same BC as 2.x but with element conservation in gas phase):
  - 5 cm TACOT_v3 slab, T_init = 300 K
  - Full SEB with B' chemistry + element conservation tracking
  - Convective BC: rhoUeCH = 0.3 kg/m²/s, h_r = 1.5e6 J/kg, p_e = 101325 Pa
  - Heating t = 0.1 → 60 s; cooling t = 60.1 → 120 s
  - Pyrolysis gas blowing, recession enabled

SCAM configuration:
  - rhoUeCH = 0.3 kg/m²/s,  h_r = 1.5e6 J/kg  (exact PATO values)
  - Element transport: backward-Euler upwind FVM for Z_i(y,t), enabled via
    SolverOptions(element_transport=True)
  - Surface chemistry: live Cantera backend with SEB advective terms
  - Recession: s_dot = m_dot_char / rho_char  (CMA model)

PATO reference data:
  pato-3.1/src/.../ref/1D/AblationTestCase_2.x_equilibriumElementConservation/output/

Run:
    MPLBACKEND=Agg python3 examples/ablation2/compare_pato_ablation2_equilibriumElementConservation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pato2_common import (
    C_M, P_EDGE, PATO_REF, PATO_TUTS, RHOUE_CH, H_R, T_END,
    T_INIT, TC_DEPTHS, equilibrium_hg_table,
    load_pato_ta_plot, load_pato_ta_surfacepatch,
    prepare_plot_output, run_pato, save_plot_atomic, T_at_original_depths,
)
from compare_pato_ablation2 import _DefaultZCBackend, load_tacot_material

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CASE_NAME  = "AblationTestCase_2.x_equilibriumElementConservation"
PATO_CASE  = PATO_TUTS / CASE_NAME
PATO_OUT   = PATO_REF  / CASE_NAME / "output/porousMat"
TA_PLOT    = PATO_OUT  / "scalar/Ta_plot"
TA_SURF    = PATO_OUT  / "scalar/Ta_surfacePatch"
OUT_PNG = Path(__file__).parent / "compare_pato_ablation2_equilibriumElementConservation.png"

THICKNESS = 0.05
N_NODES   = 201
TC_LABELS_MM = [1, 2, 4, 8, 12, 16, 24]

# ---------------------------------------------------------------------------
# BC time history — enthalpy-based (identical to base 2.x).  Cooldown keeps
# PATO's mapped rhoUeCH=0.3e-2 value, but chemistryOn=0 means the Bprime BC
# ignores rhoUeCH/h_r and the B' chemistry driver is zeroed.
# ---------------------------------------------------------------------------
_T_STEPS = np.array([0.0, 0.1, 60.0, 60.1, T_END])
_RHOUECH_VALS = np.array([
    0.01 * RHOUE_CH, RHOUE_CH, RHOUE_CH, 0.01 * RHOUE_CH, 0.01 * RHOUE_CH,
])
_CHEM_RHOUE_VALS = np.array([0.01 * RHOUE_CH, RHOUE_CH, RHOUE_CH, 0.0, 0.0])
_HR_VALS = np.array([0.0, H_R, H_R, 0.0, 0.0])


def _rhoUeCH(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _RHOUECH_VALS))


def _h_r(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _HR_VALS))


def _rhoue(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _CHEM_RHOUE_VALS))


def run_scam(mode="equilibrium", verbose=True):
    """Run SCAM in one of two surface-chemistry modes.

    mode="equilibrium" : EquilibriumElement — the in-depth element-conservation
        PDE tracks Z_C(y,t) and the surface carbon fraction Z_C_wall feeds the
        4-D B' table (the PATO-equivalent model).
    mode="cma" : classic CMA — no element conservation; the injected pyrolysis
        gas composition is fixed using Cantera's nominal pyrolysis-gas Z_C fallback
        when the live backend is available.

    Both modes use the continuous moving-mesh (ALE) recession scheme so T_wall
    is free of the node-drop sawtooth.
    """
    if mode not in {"equilibrium", "cma"}:
        raise ValueError(f"unknown run_scam mode: {mode!r}")

    mat, _ = load_tacot_material(use_pato_hg=False)
    # PATO's porousMatProperties for this case sets GasPropertiesType=Equilibrium,
    # so the in-depth pyrolysis-gas enthalpy is the chemical-equilibrium value, not
    # a fixed table.  Compute h_g(T) from live Cantera equilibrium element
    # conservation (falls back to the tabulated h_g if Cantera is missing).
    _eq_hg = equilibrium_hg_table()
    if _eq_hg is not None:
        mat.h_g_table = _eq_hg
        mat.h_g_abs_offset = None   # Cantera h_g is already on absolute reference
        # The card's gas_properties_pT (PATO *Tabulated* gasProperties) takes
        # priority over h_g_table in pyrolysis_gas_enthalpy_abs — clear it so
        # the equilibrium h_g actually gets used (PATO runs this case with
        # GasPropertiesType=Equilibrium, not Tabulated).
        mat.gas_properties_pT = None
        print("  Gas model: LIVE Cantera equilibrium element conservation "
              f"(re-speciated TACOT pyrolysis gas, {len(_eq_hg)} T-points)")
    else:
        print("  Gas model: tabulated h_g (Cantera unavailable)")

    element_transport = (mode == "equilibrium")

    # Surface chemistry: prefer the live Cantera equilibrium backend — it
    # exposes surface_enthalpies(), which activates the SEB advective terms
    # (qAdvPyro + qAdvChar) that the pre-computed B' table cannot provide.  In
    # EquilibriumElement mode the transported surface carbon fraction Z_C_pyro is
    # passed through to both lookup() and surface_enthalpies().  In CMA mode,
    # there is no transported Z_C, so Cantera's configured nominal pyrolysis-gas
    # carbon fraction is used as the fallback.  Cantera backend is required here because
    # table-only fallback omits the advective enthalpy terms and produces a
    # misleading cold surface plot.
    try:
        from scam.physics.bprime_evaluator import BprimeEvaluator
        bpt = BprimeEvaluator.from_config(str(REPO / "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"))
        if element_transport:
            bpt.lookup(1500.0, P_EDGE, 0.05, 0.3)   # probe: fail fast if broken
            table_dim = "live Cantera equilibrium backend (+ SEB advective terms)"
        else:
            z_c_pyro = bpt.pyrolysis_target_fraction("C")
            bpt.lookup(1500.0, P_EDGE, 0.05, z_c_pyro)   # probe: fail fast if broken
            bpt = _DefaultZCBackend(bpt, z_c_pyro)
            table_dim = f"live Cantera backend (Cantera-derived Z_C_pyro={z_c_pyro:.6f})"
        print(f"  Surface chemistry: {table_dim}")
    except Exception as _e:
        raise RuntimeError(
            "Live Cantera is required for this validation example; "
            "table-only fallback produces a misleading cold surface comparison. "
            f"The current interpreter is {sys.executable!r}. Run with the "
            "Cantera-enabled Python used by this workspace or install "
            "Cantera into this interpreter."
        ) from _e

    mat_cards      = {mat.name: mat}
    b_prime_tables = {mat.name: bpt}

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=N_NODES, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=_rhoUeCH,
        h_r=_h_r,
        # emissivity < 0 → use the material's blended value (virgin 0.8 → char 0.9,
        # PATO's TACOT_v3 values) based on the surface char fraction, rather than a
        # hardcoded override.  The surface is fully charred so this resolves to ~0.9.
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=300.0,   # PATO Tbackground=300 K (AblationTestCase_2.x Ta BC)
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
        output_dt=0.5,
        tc_positions=[],
        allow_recession=True,
        use_rho_old=True,
        element_transport=element_transport,
        continuous_remap=True,
    )

    label = "EquilibriumElement" if element_transport else "CMA (fixed pyrolysis gas)"
    print(f"Running SCAM [{label}] ...  surface chemistry: {table_dim}")
    print(f"  Element transport: {'ENABLED' if element_transport else 'DISABLED'}"
          f" | Recession: continuous ALE (no node-drop sawtooth)")
    return run(stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc,
               options, initial_T=T_INIT, verbose=verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _phase_max_dT(t_s, y_s, t_ref, y_ref):
    """Max |SCAM - PATO| over the heating and cooling phases separately.

    The flux is cut off (near-instantaneously) at t=60 s.  PATO reference data
    is only 1 s-resolved, so it cannot represent that sub-second cliff; comparing
    SCAM's resolved crash against a linear interpolation of PATO's coarse samples
    inside the transition produces a spurious ~300 K "error".  We therefore
    exclude the transition window [59.9, 61.1] s and report the two phases apart.
    """
    diff = y_s - np.interp(t_s, t_ref, y_ref)
    valid = ~np.isnan(diff)
    # Exclude the startup ramp (t<2 s): the surface shoots 300→~1270 K within
    # the first second, which PATO's 1 s-resolution data cannot represent, so a
    # direct instantaneous comparison there is a sampling artifact, not an error.
    heat = valid & (t_s >= 2.0) & (t_s <= 59.9)
    cool = valid & (t_s >= 61.1)
    mh = float(np.abs(diff[heat]).max()) if heat.any() else float("nan")
    mc = float(np.abs(diff[cool]).max()) if cool.any() else float("nan")
    return mh, mc


def main() -> None:
    prepare_plot_output(OUT_PNG)
    run_pato(PATO_CASE, TA_PLOT)

    print("Loading PATO reference data (1 s resolution) ...")
    t_p, T_p   = load_pato_ta_plot(TA_PLOT)
    t_ps, T_ps = load_pato_ta_surfacepatch(TA_SURF)

    # Run SCAM in both surface-chemistry modes for direct comparison to PATO.
    res_ee  = run_scam(mode="equilibrium", verbose=True)
    res_cma = run_scam(mode="cma", verbose=False)

    results  = res_ee   # primary (EquilibriumElement) for in-depth/Z_C panels
    t_s      = results.times_array()
    T_wall_s = results.T_wall_array()
    rec_s    = results.s_array()
    T_tc     = T_at_original_depths(results, TC_DEPTHS)

    t_cma     = res_cma.times_array()
    Twall_cma = res_cma.T_wall_array()
    rec_cma   = res_cma.s_array()

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
        "SCAM vs PATO AblationTestCase_2.x_equilibriumElementConservation\n"
        "SCAM: element transport + Cantera equilibrium gas + live Cantera surface chemistry  |  "
        "PATO: EquilibriumElement model",
        fontsize=10,
    )

    # Panel 1 — surface T (both SCAM modes vs PATO)
    ax = axes[0, 0]
    ax.plot(t_s,    T_wall_s,    "r-",  lw=2,   label="SCAM EquilibriumElement")
    ax.plot(t_cma,  Twall_cma,   "g-",  lw=1.5, label="SCAM CMA")
    ax.plot(t_ps,   T_ps[:, 0],  "k--", lw=1.5, label="PATO")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface temperature"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2 — TC at 1, 4, 12 mm
    ax = axes[0, 1]
    for i, col in zip([0, 2, 4], colors):
        d_mm = TC_LABELS_MM[i]
        ax.plot(t_s, T_tc[i],    color=col, lw=2,   label=f"SCAM {d_mm} mm")
        ax.plot(t_p, T_p[:, i],  color=col, lw=1.5, ls="--", label=f"PATO {d_mm} mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC temperatures — 1, 4, 12 mm"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3 — recession (both modes)
    ax = axes[0, 2]
    ax.plot(t_s,   rec_s * 1000,   "r-", lw=2,   label="SCAM EquilibriumElement")
    ax.plot(t_cma, rec_cma * 1000, "g-", lw=1.5, label="SCAM CMA")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Recession [mm]")
    ax.set_title("Surface recession (PATO mass file not in ref)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4 — T profiles at t=30 and t=90 s
    ax = axes[1, 0]
    snap_30 = snap_90 = None
    for snap in results.snapshots:
        if abs(snap.time - 30) < 0.3 and snap_30 is None:
            snap_30 = snap
        if abs(snap.time - 90) < 0.3 and snap_90 is None:
            snap_90 = snap
    i30 = int(np.argmin(np.abs(t_p - 30)))
    i90 = int(np.argmin(np.abs(t_p - 90)))
    pato_d_mm = [d * 1000 for d in TC_DEPTHS]
    # mesh.y_nodes is already the coordinate from the ORIGINAL front face
    # (surface node sits at y_nodes[0] == s_total); do NOT add s_total again.
    if snap_30:
        y30 = snap_30.mesh.y_nodes * 1000
        ax.plot(y30, snap_30.T, "b-", lw=2, label="SCAM t=30 s")
    if snap_90:
        y90 = snap_90.mesh.y_nodes * 1000
        ax.plot(y90, snap_90.T, "r-", lw=2, label="SCAM t=90 s")
    valid_30 = T_p[i30, :] > 0
    valid_90 = T_p[i90, :] > 0
    ax.plot(np.array(pato_d_mm)[valid_30], T_p[i30, valid_30],
            "b--o", ms=5, lw=1.5, label=f"PATO t≈{t_p[i30]:.0f} s")
    ax.plot(np.array(pato_d_mm)[valid_90], T_p[i90, valid_90],
            "r--o", ms=5, lw=1.5, label=f"PATO t≈{t_p[i90]:.0f} s")
    ax.set_xlabel("Depth from original surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t≈30 s and t≈90 s")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 5 — SCAM − PATO surface T (both modes; transition window masked)
    ax = axes[1, 1]
    def _masked_diff(ts, ys):
        d = ys - np.interp(ts, t_ps, T_ps[:, 0])
        d = d.copy()
        d[ts < 2.0] = np.nan                 # mask startup sampling artifact
        d[(ts > 59.9) & (ts < 61.1)] = np.nan   # mask flux-cutoff sampling cliff
        return d
    ax.plot(t_s,   _masked_diff(t_s, T_wall_s),    "r-", lw=1.5, label="EquilibriumElement")
    ax.plot(t_cma, _masked_diff(t_cma, Twall_cma), "g-", lw=1.5, label="CMA")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("Surface T difference SCAM − PATO"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6 — Z_C evolution at surface and select depths (from element transport)
    ax = axes[1, 2]
    Z_C_surf = []
    Z_C_5mm  = []
    Z_C_10mm = []
    for snap in results.snapshots:
        if snap.Z_elem is not None:
            Z_C_surf.append(float(snap.Z_elem[0, 0]))
            # Find node closest to 5 mm and 10 mm depth from original surface
            # (y_nodes is already measured from the original front face).
            y_mm = snap.mesh.y_nodes * 1000
            Z_C_5mm.append(float(snap.Z_elem[0, int(np.argmin(np.abs(y_mm - 5)))]))
            Z_C_10mm.append(float(snap.Z_elem[0, int(np.argmin(np.abs(y_mm - 10)))]))
        else:
            Z_C_surf.append(np.nan)
            Z_C_5mm.append(np.nan)
            Z_C_10mm.append(np.nan)
    Z_C_surf = np.array(Z_C_surf)
    Z_C_5mm  = np.array(Z_C_5mm)
    Z_C_10mm = np.array(Z_C_10mm)
    if not np.all(np.isnan(Z_C_surf)):
        ax.plot(t_s, Z_C_surf,  "r-",  lw=2,   label="Z_C surface")
        ax.plot(t_s, Z_C_5mm,   "b-",  lw=1.5, label="Z_C @ 5 mm")
        ax.plot(t_s, Z_C_10mm,  "g-",  lw=1.5, label="Z_C @ 10 mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Z_C (carbon mass fraction) [−]")
    ax.set_title("Carbon element fraction in porous gas phase")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_atomic(fig, OUT_PNG, dpi=150)
    print(f"\nPlot saved → {OUT_PNG}")

    # -----------------------------------------------------------------------
    # Quantitative comparison (heating / cooling phases reported separately;
    # the flux-cutoff transition window is excluded — see _phase_max_dT).
    # -----------------------------------------------------------------------
    def _Twall60(ts, tw):
        return float(np.interp(60.0, ts, tw))
    def _rec60(ts, rc):
        return float(np.interp(60.0, ts, rc) * 1000)
    pato_T60 = float(np.interp(60.0, t_ps, T_ps[:, 0]))

    print("\n=== SCAM vs PATO: surface temperature ===")
    print(f"  PATO  T_wall(60 s) = {pato_T60:.1f} K")
    for tag, ts, tw, rc in (("EquilibriumElement", t_s, T_wall_s, rec_s),
                            ("CMA               ", t_cma, Twall_cma, rec_cma)):
        mh, mc = _phase_max_dT(ts, tw, t_ps, T_ps[:, 0])
        print(f"  SCAM [{tag}] T_wall(60 s)={_Twall60(ts,tw):.1f} K "
              f"(Δ={_Twall60(ts,tw)-pato_T60:+.1f} K), recession(60 s)={_rec60(ts,rc):.2f} mm")
        print(f"        max|ΔT| heating={mh:.1f} K, cooling={mc:.1f} K")

    print("\n=== SCAM(EquilibriumElement) vs PATO: in-depth max |ΔT| (heating phase) ===")
    for i in range(len(TC_DEPTHS)):
        diff = T_tc[i] - np.interp(t_s, t_p, T_p[:, i])
        valid = (~np.isnan(diff)) & (t_s <= 59.9)
        if valid.any():
            print(f"  {TC_LABELS_MM[i]:2d} mm : {np.abs(diff[valid]).max():.1f} K")


if __name__ == "__main__":
    main()
