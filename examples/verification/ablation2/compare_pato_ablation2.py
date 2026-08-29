# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_2.x using the TACOT 3.0 YAML material.

PATO AblationTestCase_2.x setup:
  - 5 cm TACOT_v3 slab, T_init = 300 K
  - Full surface energy balance with B' char ablation chemistry
  - Convective BC: rhoUeCH = 0.3 kg/m²/s, h_r = 1.5e6 J/kg
  - Edge pressure: p_e = 101325 Pa
  - Heating from t = 0.1 s → 60 s; cooling t = 60.1 s → 120 s
  - Pyrolysis gas blowing + surface recession enabled
  - Adiabatic back face

SCAM configuration:
  - rhoUeCH = 0.3 kg/m²/s,  h_r = 1.5e6 J/kg  (exact PATO values)
  - q_conv = rhoUeCH_eff * (h_r - h_wall(T_w))  in the B' formulation
  - rho_e_u_e = 0.3 kg/m²/s,  C_M = 1.0  (for B' table blowing correction)
  - Material card loaded from scam/materials/ablative_organic/tacot_v3.0.yaml
  - Surface chemistry: TACOT 3.0 B' table or live Cantera
  - Gas properties: PATO base 2.x "Tabulated" gas model (no element transport)
  - Recession: s_dot = m_dot_char / rho_char  (CMA model)
  - Diagnostic chemistry modes are available through run_scam(chemistry_mode=...):
    "live_cantera_zc_default" (default), "live_cantera", "live_mpp",
    "table_bc_live_h", "table_hybrid", or "table";
    see docs/physics/surface_chemistry_modes.md

PATO reference data used:
  pato-3.1/src/.../ref/1D/AblationTestCase_2.x/output/

Run:
    MPLBACKEND=Agg python3 examples/verification/ablation2/compare_pato_ablation2.py
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
    T_INIT, TC_DEPTHS, _load_pato_hg,
    load_pato_ta_plot, load_pato_ta_surfacepatch, load_pato_mass,
    prepare_plot_output, run_pato, save_plot_atomic, T_at_original_depths,
)

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CASE_NAME  = "AblationTestCase_2.x"
PATO_CASE  = PATO_TUTS  / CASE_NAME
PATO_OUT   = PATO_REF   / CASE_NAME / "output/porousMat"
TA_PLOT    = PATO_OUT   / "scalar/Ta_plot"
TA_SURF    = PATO_OUT   / "scalar/Ta_surfacePatch"
MASS_FILE  = PATO_OUT   / "mass"
CHEMISTRY_MODE = "live_cantera"
OUT_PNG = Path(__file__).parent / "compare_pato_ablation2.png"
TACOT_YAML = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"
BPRIME_CONFIG = REPO / "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"
BPRIME_CONFIG_MPP = REPO / "scam/materials/ablative_organic/tacot_v3.0_mpp_config.yaml"

THICKNESS = 0.05    # m
N_NODES   = 201

# Depth labels matching PATO plotDict probe order
TC_LABELS_MM = [1, 2, 4, 8, 12, 16, 24]   # mm from original surface

# ---------------------------------------------------------------------------
# BC time history — enthalpy-based, matching PATO's BoundaryConditions table.
# PATO keeps a small rhoUeCH=0.3e-2 background before/after the heat pulse.
# At cooldown chemistryOn drops to 0, so the Bprime BC ignores rhoUeCH/h_r and
# the B' chemistry driver is zeroed.
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


class _BPrimeTableWithCanteraAdv:
    """PATO-table B' lookup plus Cantera enthalpy differences for qAdv terms."""

    def __init__(self, table, ev):
        self._table = table
        self._ev = ev
        self._last = None

    def lookup(self, T_wall, p_e, B_g, Z_C_pyro=None):
        B_c, h_wall = self._table.lookup(T_wall, p_e, B_g, Z_C_pyro)
        _B_c_ev, h_wall_ev = self._ev.lookup(T_wall, p_e, B_g, Z_C_pyro)
        self._last = (float(T_wall), float(p_e), Z_C_pyro, float(h_wall), float(h_wall_ev))
        return B_c, h_wall

    def surface_enthalpies(self, T_wall, p_e, Z_C_pyro=None):
        h_g_ev, h_c_ev = self._ev.surface_enthalpies(T_wall, p_e, Z_C_pyro)
        if self._last is None:
            return h_g_ev, h_c_ev
        _T_last, _p_last, _Z_last, h_wall_table, h_wall_ev = self._last
        return (
            h_wall_table + (h_g_ev - h_wall_ev),
            h_wall_table + (h_c_ev - h_wall_ev),
        )


class _DefaultZCBackend:
    """Supply a nominal pyrolysis-gas carbon fraction when none is passed.

    z_c_default is injected for lookup() only (B'_c depends on composition in
    element-transport cases).  surface_enthalpies() does NOT inject the default:
    it always passes Z_C_pyro as-is so the equilibrium h_g uses the nominal
    composition (Z_C_pyro=None → nominal + equilibrate) or the true element-
    transport value if provided.  Injecting z_c_default into surface_enthalpies
    would route through _pyro_x_from_z_c_molecular with the wrong elemental
    fractions, giving incorrect equilibrium h_g.
    """

    def __init__(self, backend, z_c_default):
        self._backend = backend
        self._z_c_default = z_c_default

    def _z(self, Z_C_pyro):
        return self._z_c_default if Z_C_pyro is None else Z_C_pyro

    def lookup(self, T_wall, p_e, B_g, Z_C_pyro=None):
        return self._backend.lookup(T_wall, p_e, B_g, self._z(Z_C_pyro))

    def surface_enthalpies(self, T_wall, p_e, Z_C_pyro=None):
        return self._backend.surface_enthalpies(T_wall, p_e, Z_C_pyro)


class _TableBCLiveEnthalpyBackend:
    """Use table B'c for mass loss and live Cantera enthalpies for the SEB."""

    def __init__(self, table, ev, z_c_default):
        self._table = table
        self._ev = ev
        self._z_c_default = z_c_default

    def _z(self, Z_C_pyro):
        return self._z_c_default if Z_C_pyro is None else Z_C_pyro

    def lookup(self, T_wall, p_e, B_g, Z_C_pyro=None):
        B_c, _h_wall_table = self._table.lookup(T_wall, p_e, B_g, Z_C_pyro)
        _B_c_ev, h_wall = self._ev.lookup(T_wall, p_e, B_g, self._z(Z_C_pyro))
        return B_c, h_wall

    def surface_enthalpies(self, T_wall, p_e, Z_C_pyro=None):
        return self._ev.surface_enthalpies(T_wall, p_e, self._z(Z_C_pyro))


def load_tacot_material(*, use_pato_hg: bool = True):
    """Load TACOT 3.0 from YAML and return ``(material, bprime_table)``.

    ``use_pato_hg`` replaces only the gas enthalpy table with the PATO
    gasProperties data used by the base 2.x reference.  The solid material,
    kinetics, emissivity, porosity, elemental fractions, and B' table all come
    from ``tacot.yaml``.
    """
    mat, bpt = load_material(str(TACOT_YAML))
    if use_pato_hg:
        hg_t, hg_val = _load_pato_hg()
        # The PATO gasProperties h_g is already on the absolute (Cantera/Mutation++)
        # reference — clear h_g_abs_offset so the assembly does not double-apply it.
        mat = dataclasses.replace(mat,
                                   h_g_table=np.column_stack([hg_t, hg_val]),
                                   h_g_abs_offset=None)
    return mat, bpt


def make_tacot_bprime_backend(bpt_from_yaml, mode=CHEMISTRY_MODE):
    """Return the surface-chemistry backend.

    mode="live_cantera":
        diagnostic raw Cantera nominal-composition mode.  Cantera evaluates
        B'c, h_wall, and qAdvPyro/qAdvChar enthalpies directly from the
        configured pyrolysis gas species string when no element-transport Z_C
        value is available.  Useful for isolating the B'c/h_wall lookup, but
        its Cantera pyrolysis-gas enthalpy leaves the surface too cold.
    mode="live_cantera_zc_default" (default):
        validation mode.  Cantera reports the carbon mass fraction of its configured
        nominal pyrolysis gas stream, and SCAM passes that value back as the
        fallback Z_C_pyro when no element-transport Z_C value is available.  The
        gas is then reconstructed through the same Z_C_pyro -> pyro_x path used
        by element transport.  This keeps B'c/h_wall close to tacot26 while
        giving the PATO-consistent qAdvPyro enthalpy for the base 2.x comparison.
    mode="table_hybrid":
        table B'c/h_wall plus Cantera enthalpy differences for the qAdv terms.
        Best recession diagnostic, but still cooler than PATO at the surface.
    mode="table_bc_live_h":
        table B'c with Cantera-derived nominal-Z_C live h_wall and qAdv enthalpies.
    mode="table":
        TACOT 3.0 B' table including its pretabulated h_g/h_c columns.  This mode
        has no live Cantera dependency.
    mode="live_mpp":
        Mutation++ warm-start B' grid for B'_c/h_wall, with Cantera providing
        reference-consistent h_g/h_c for the PATO advective terms.
    """
    table = bpt_from_yaml
    if mode == "table":
        print(
            f"  Surface chemistry: TACOT 3.0 B' table ({type(table).__name__}) "
            "with pretabulated advective enthalpies"
        )
        return table

    if mode == "live_mpp":
        from scam.physics.mpp_evaluator import MutationppEvaluator
        ev = MutationppEvaluator.from_config(str(BPRIME_CONFIG_MPP))
        print("  Surface chemistry: LIVE Mutation++ warm-start grid")
        return ev

    if mode in {"live_cantera", "live_cantera_zc_default", "table_hybrid", "table_bc_live_h"}:
        try:
            from scam.physics.bprime_evaluator import BprimeEvaluator
            ev = BprimeEvaluator.from_config(str(BPRIME_CONFIG))
            ev.lookup(1500.0, P_EDGE, 0.05, None)   # probe: fail fast if broken
        except Exception as e:
            raise RuntimeError(
                f"Cantera backend is required for chemistry_mode={mode!r}; "
                f"the current interpreter is {sys.executable!r}. Run with the "
                "Cantera-enabled Python used by this workspace (for example "
                "`python3 examples/ablation2/compare_pato_ablation2.py`) or "
                "install Cantera into this interpreter. Use "
                "chemistry_mode='table' only for the explicit table-only diagnostic."
            ) from e
        z_c_pyro = ev.pyrolysis_target_fraction("C")
        if mode == "table_hybrid":
            print("  Surface chemistry: TACOT 3.0 B' table + raw Cantera advective enthalpy differences")
            return _BPrimeTableWithCanteraAdv(table, ev)
        if mode == "table_bc_live_h":
            print(f"  Surface chemistry: table B'c + live Cantera enthalpies with Cantera-derived Z_C_pyro={z_c_pyro:.6f}")
            return _TableBCLiveEnthalpyBackend(table, ev, z_c_pyro)
        if mode == "live_cantera_zc_default":
            print(f"  Surface chemistry: LIVE Cantera with Cantera-derived Z_C_pyro={z_c_pyro:.6f}")
            return _DefaultZCBackend(ev, z_c_pyro)
        print("  Surface chemistry: LIVE Cantera raw nominal-composition backend")
        return ev
    else:
        raise ValueError(f"unknown surface chemistry mode: {mode!r}")

def run_scam(chemistry_mode=CHEMISTRY_MODE):
    if isinstance(chemistry_mode, bool):
        chemistry_mode = "live_cantera" if chemistry_mode else "table"
    mat, bpt_yaml = load_tacot_material()
    bpt = make_tacot_bprime_backend(bpt_yaml, chemistry_mode)
    mat_cards     = {mat.name: mat}
    b_prime_tables = {mat.name: bpt}

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=N_NODES, n_subcells=4),
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
        tc_positions=[],   # we post-process at fixed original depths ourselves
        allow_recession=True,
        use_rho_old=True,
        continuous_remap=True,   # continuous moving-mesh (ALE) — no node-drop sawtooth
    )

    print(f"Running SCAM (AblationTestCase_2.x / TACOT 3.0 YAML, mode={chemistry_mode}) ...")
    return run(stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc,
               options, initial_T=T_INIT, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _phase_max_dT(t_s, y_s, t_ref, y_ref):
    """Max |SCAM - PATO| outside startup and flux-cutoff sampling artifacts."""
    diff = y_s - np.interp(t_s, t_ref, y_ref)
    valid = ~np.isnan(diff)
    heat = valid & (t_s >= 2.0) & (t_s <= 59.9)
    cool = valid & (t_s >= 61.1)
    mh = float(np.abs(diff[heat]).max()) if heat.any() else float("nan")
    mc = float(np.abs(diff[cool]).max()) if cool.any() else float("nan")
    return mh, mc


def main() -> None:
    prepare_plot_output(OUT_PNG)
    run_pato(PATO_CASE, TA_PLOT)

    print("Loading PATO reference data ...")
    t_p, T_p   = load_pato_ta_plot(TA_PLOT)        # (N,), (N, 7)
    t_ps, T_ps = load_pato_ta_surfacepatch(TA_SURF) # surface + back face T
    t_m, mdg, mdc, rec_p = load_pato_mass(MASS_FILE)

    results = run_scam(CHEMISTRY_MODE)

    t_s = results.times_array()
    T_wall_s = results.T_wall_array()
    rec_s    = results.s_array()
    # Evaluate T at the same 7 probe depths as PATO (fixed in original space)
    T_tc = T_at_original_depths(results, TC_DEPTHS)  # (7, n_steps)

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
        "SCAM vs PATO AblationTestCase_2.x\n"
        f"TACOT 3.0 YAML | full SEB + B' chemistry | pyrolysis blowing | recession | "
        f"SCAM chemistry={CHEMISTRY_MODE}",
        fontsize=11,
    )

    # Panel 1 — surface T history
    ax = axes[0, 0]
    ax.plot(t_s,  T_wall_s,     "r-",  lw=2,   label="SCAM T_wall")
    ax.plot(t_ps, T_ps[:, 0],   "k--", lw=1.5, label="PATO T_surface")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface temperature"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2 — TC temperatures at 1, 4, 12 mm (heating phase)
    ax = axes[0, 1]
    plot_idx = [0, 2, 4]   # 1 mm, 4 mm, 12 mm
    for i, col in zip(plot_idx, colors):
        d_mm = TC_LABELS_MM[i]
        ax.plot(t_s,  T_tc[i],         color=col, lw=2,   label=f"SCAM {d_mm} mm")
        ax.plot(t_p,  T_p[:, i],       color=col, lw=1.5, ls="--", label=f"PATO {d_mm} mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC temperatures — 1, 4, 12 mm"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3 — recession history
    ax = axes[0, 2]
    ax.plot(t_s, rec_s * 1000, "b-",  lw=2,   label="SCAM recession")
    ax.plot(t_m, rec_p * 1000, "b--", lw=1.5, label="PATO recession")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Recession [mm]")
    ax.set_title("Surface recession"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4 — T profile snapshots
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
    # mesh.y_nodes is already measured from the original front face; the surface
    # node sits at y_nodes[0] == s_total.  Do not add s_total again.
    if snap_30 is not None:
        y_orig_30 = snap_30.mesh.y_nodes * 1000
        ax.plot(y_orig_30, snap_30.T, "b-", lw=2, label="SCAM t=30 s")
    if snap_90 is not None:
        y_orig_90 = snap_90.mesh.y_nodes * 1000
        ax.plot(y_orig_90, snap_90.T, "r-", lw=2, label="SCAM t=90 s")
    # PATO probe data as scatter
    valid_30 = T_p[i30, :] > 0
    valid_90 = T_p[i90, :] > 0
    ax.plot(np.array(pato_d_mm)[valid_30], T_p[i30, valid_30],
            "b--o", ms=5, lw=1.5, label=f"PATO t≈{t_p[i30]:.0f} s")
    ax.plot(np.array(pato_d_mm)[valid_90], T_p[i90, valid_90],
            "r--o", ms=5, lw=1.5, label=f"PATO t≈{t_p[i90]:.0f} s")
    ax.set_xlabel("Depth from original surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t≈30 s (heating) and t≈90 s (cooling)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 5 — SCAM − PATO surface T difference.
    # Match the EquilibriumElement diagnostic and mask the two sub-second
    # sampling artifacts handled by _phase_max_dT.
    ax = axes[1, 1]
    surf_diff = T_wall_s - np.interp(t_s, t_ps, T_ps[:, 0])
    surf_diff = surf_diff.copy()
    surf_diff[t_s < 2.0] = np.nan
    surf_diff[(t_s > 59.9) & (t_s < 61.1)] = np.nan
    ax.plot(t_s, surf_diff, "r-", lw=1.5, label="surface")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("Surface T difference SCAM − PATO"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6 — surface mass flux comparison
    ax = axes[1, 2]
    m_dot_g_s = np.array([snap.m_dot_pyro for snap in results.snapshots])
    m_dot_c_s = np.array([snap.m_dot_char for snap in results.snapshots])
    ax.plot(t_s, m_dot_g_s, "g-",  lw=2,   label="SCAM ṁ_gas")
    ax.plot(t_m, mdg,       "g--", lw=1.5, label="PATO ṁ_g_surf")
    ax.plot(t_s, m_dot_c_s, "r-",  lw=2,   label="SCAM ṁ_char")
    ax.plot(t_m, mdc,       "r--", lw=1.5, label="PATO ṁ_char")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Mass flux [kg/m²/s]")
    ax.set_title("Surface pyrolysis gas and char fluxes")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_atomic(fig, OUT_PNG, dpi=150)
    print(f"\nPlot saved → {OUT_PNG}  (chemistry_mode={CHEMISTRY_MODE})")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    surf_heat, surf_cool = _phase_max_dT(t_s, T_wall_s, t_ps, T_ps[:, 0])
    T60_s = float(np.interp(60.0, t_s, T_wall_s))
    T60_p = float(np.interp(60.0, t_ps, T_ps[:, 0]))
    print("\n--- SCAM vs PATO: surface temperature ---")
    print(f"  PATO T_wall(60 s): {T60_p:.1f} K")
    print(f"  SCAM T_wall(60 s): {T60_s:.1f} K  (Δ={T60_s - T60_p:+.1f} K)")
    print(f"  max |ΔT| heating:  {surf_heat:.1f} K  (t = 2.0–59.9 s)")
    print(f"  max |ΔT| cooling:  {surf_cool:.1f} K  (t >= 61.1 s)")
    print("  selected heating deltas:")
    for target in (2.0, 10.0, 30.0, 60.0):
        Ts = float(np.interp(target, t_s, T_wall_s))
        Tp = float(np.interp(target, t_ps, T_ps[:, 0]))
        print(f"    t={target:4.1f} s : SCAM={Ts:7.1f} K, PATO={Tp:7.1f} K, Δ={Ts - Tp:+6.1f} K")

    print("\n--- SCAM vs PATO: max |ΔT| at TC probes (full run) ---")
    for i in range(len(TC_DEPTHS)):
        d_mm = TC_LABELS_MM[i]
        T_p_interp = np.interp(t_s, t_p, T_p[:, i])
        diff = T_tc[i] - T_p_interp
        valid = ~np.isnan(diff)
        if valid.any():
            print(f"  {d_mm:2d} mm : max |SCAM−PATO| = {np.abs(diff[valid]).max():6.1f} K")

    print("\n--- Recession at t=60 s and t=120 s ---")
    i60 = int(np.argmin(np.abs(t_s - 60)))
    print(f"  SCAM t=60 s : {rec_s[i60]*1000:.2f} mm")
    i60p = int(np.argmin(np.abs(t_m - 60)))
    print(f"  PATO t=60 s : {rec_p[i60p]*1000:.2f} mm")
    print(f"  SCAM t=120 s: {rec_s[-1]*1000:.2f} mm")
    print(f"  PATO t=120 s: {rec_p[-1]*1000:.2f} mm")

    print("\n--- Surface mass flux at t=60 s ---")
    mgs60 = float(np.interp(60.0, t_s, m_dot_g_s))
    mcs60 = float(np.interp(60.0, t_s, m_dot_c_s))
    mgp60 = float(np.interp(60.0, t_m, mdg))
    mcp60 = float(np.interp(60.0, t_m, mdc))
    print(f"  pyrolysis gas: SCAM={mgs60:.5f} kg/m²/s, PATO={mgp60:.5f} kg/m²/s, Δ={mgs60 - mgp60:+.5f}")
    print(f"  char erosion:  SCAM={mcs60:.5f} kg/m²/s, PATO={mcp60:.5f} kg/m²/s, Δ={mcs60 - mcp60:+.5f}")


if __name__ == "__main__":
    main()
