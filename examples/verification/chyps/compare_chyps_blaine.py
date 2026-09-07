# SPDX-License-Identifier: MIT
"""Compare SCAM vs CHyPS BlaineTest reference data.

CHyPS reference data courtesy of Dr. Blaine Vollmer, University of Illinois at
Urbana-Champaign, redistributed with permission.  The probe files are
column-trimmed copies of the original CHyPS output (numerically identical for
this comparison); see the README in this directory.

CHyPS probe layout (fixed depths from original surface):
  T0 = surface (Lagrangian, tracks recession)
  T1 = 1 mm   T2 = 2 mm   T3 = 4 mm
  T4 = 8 mm   T5 = 12 mm  T6 = 16 mm
  T7 = 24 mm  T8 = back face (50 mm)

Probes T1–T5 are consumed by the receding surface; T6–T8 survive to t=100 s.

Run:
    MPLBACKEND=Agg python3 examples/verification/chyps/compare_chyps_blaine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO      = Path(__file__).resolve().parent.parent.parent.parent
CHYPS_DIR = Path(__file__).resolve().parent / "BlaineTest_chyps_results" / "probes"
OUT_PNG   = Path(__file__).resolve().parent / "compare_chyps_blaine.png"
TACOT_YAML  = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"
BPRIME_CONFIG = REPO / "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"

sys.path.insert(0, str(REPO))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# CHyPS probe depths [m] from original surface
# ---------------------------------------------------------------------------
PROBE_DEPTHS = {
    "T1": 0.001, "T2": 0.002, "T3": 0.004,
    "T4": 0.008, "T5": 0.012, "T6": 0.016,
    "T7": 0.024, "T8": 0.050,
}

# ---------------------------------------------------------------------------
# Boundary condition time profiles (match blaine_test.yaml exactly)
# ---------------------------------------------------------------------------
_T  = np.array([0.0,   0.1,   60.0,   60.1,   90.0,   90.1,   100.0])
_RH = np.array([0.03,  0.20,  0.30,   0.20,   0.20,   0.05,   0.05 ])  # rhoUeCH
_HR = np.array([0.0,   1.5e6, 1.5e6,  1.5e7,  1.5e7,  0.0,    0.0  ])  # h_r
_RE = np.array([0.03,  0.20,  0.30,   0.20,   0.20,   0.0,    0.0  ])  # rho_e_u_e


def _rhoUeCH(t): return float(np.interp(t, _T, _RH))
def _h_r(t):     return float(np.interp(t, _T, _HR))
def _rhoue(t):   return float(np.interp(t, _T, _RE))


# ---------------------------------------------------------------------------
# CHyPS loader
# ---------------------------------------------------------------------------
def load_chyps_probe(name: str):
    """Return (time, temperature) arrays for a CHyPS probe."""
    data = np.loadtxt(CHYPS_DIR / name, skiprows=1)
    # Two layouts are accepted: the trimmed 2-column set distributed with SCAM
    # (time, temperature), and raw CHyPS probe output (32+ columns), so an
    # original CHyPS run can be dropped in unchanged.
    if data.shape[1] >= 32:
        return data[:, 1], data[:, -2]
    return data[:, 0], data[:, -1]


def load_chyps_surface():
    """Return (time, T_surface, recession_m) from T0."""
    data = np.loadtxt(CHYPS_DIR / "T0", skiprows=1)
    if data.shape[1] >= 32:          # raw CHyPS output
        t, rec, temp = data[:, 1], -data[:, 3], data[:, -2]
    else:                            # trimmed: time, mesh_displacement, temperature
        t, rec, temp = data[:, 0], -data[:, 1], data[:, 2]
    return t, temp, rec


# ---------------------------------------------------------------------------
# T at fixed original-surface depths from SCAM results
# ---------------------------------------------------------------------------
def T_at_depths(results, depths_m):
    """Interpolate SCAM T snapshots to fixed original-surface depths.

    Returns array (n_depths, n_steps). Points at depths shallower than the
    current surface (s_total) are set to NaN (consumed by recession).
    """
    n = len(depths_m)
    n_steps = len(results.snapshots)
    out = np.full((n, n_steps), np.nan)
    for j, snap in enumerate(results.snapshots):
        y = snap.mesh.y_nodes        # [m] from original front face
        s = snap.mesh.s_total        # [m] total recession
        for i, d in enumerate(depths_m):
            if d < s:                # surface has receded past this probe
                continue
            out[i, j] = float(np.interp(d, y, snap.T))
    return out


# ---------------------------------------------------------------------------
# Run SCAM
# ---------------------------------------------------------------------------
def _make_bprime_backend(bpt_table):
    from scam.physics.bprime_evaluator import BprimeEvaluator

    class _DefaultZC:
        def __init__(self, ev, z_c):
            self._ev = ev
            self._z_c = z_c

        def _z(self, Z_C_pyro):
            return self._z_c if Z_C_pyro is None else Z_C_pyro

        def lookup(self, T_wall, p_e, B_g, Z_C_pyro=None):
            return self._ev.lookup(T_wall, p_e, B_g, self._z(Z_C_pyro))

        def surface_enthalpies(self, T_wall, p_e, Z_C_pyro=None):
            return self._ev.surface_enthalpies(T_wall, p_e, self._z(Z_C_pyro))

    ev = BprimeEvaluator.from_config(str(BPRIME_CONFIG))
    ev.lookup(1500.0, 101325.0, 0.05, None)
    z_c = ev.pyrolysis_target_fraction("C")
    print(f"  Using live Cantera backend (Z_C_pyro={z_c:.6f})")
    return _DefaultZC(ev, z_c)


def run_scam(use_live_bprime: bool = True):
    mat, bpt_yaml = load_material(str(TACOT_YAML))
    if use_live_bprime:
        bpt = _make_bprime_backend(bpt_yaml)
    else:
        bpt = bpt_yaml
        print("  Using B' table backend")
    mat_cards      = {mat.name: mat}
    b_prime_tables = {mat.name: bpt}

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=0.05, n_nodes=201, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=_rhoUeCH,
        h_r=_h_r,
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=300.0,
        rho_e_u_e=_rhoue,
        C_M=1.0,
        p_e=101325.0,
        lambda_blowing=0.5,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    options = SolverOptions(
        t_end=100.0,
        dt_init=0.01,
        dt_min=1e-5,
        dt_max=0.5,
        dt_max_dT=30.0,
        dt_max_drho_frac=0.05,
        output_dt=0.5,
        tc_positions=[],
        allow_recession=True,
        use_rho_old=True,
        continuous_remap=True,
    )

    mode = "live Cantera" if use_live_bprime else "B' table"
    print(f"Running SCAM (BlaineTest, {mode}) ...")
    return run(stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc,
               options, initial_T=300.0, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # -- Load CHyPS reference --
    print("Loading CHyPS reference data ...")
    tc, Tc, rec_c = load_chyps_surface()
    chyps = {name: load_chyps_probe(name) for name in PROBE_DEPTHS}

    # -- Run SCAM --
    results = run_scam(use_live_bprime=True)
    ts      = results.times_array()
    T_wall  = results.T_wall_array()
    rec_s   = results.s_array()

    depths  = list(PROBE_DEPTHS.values())
    names   = list(PROBE_DEPTHS.keys())
    T_tc    = T_at_depths(results, depths)   # (8, n_steps)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:gray"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        "SCAM vs CHyPS — BlaineTest\n"
        "TACOT v3.0 | full SEB + live Cantera + ALE recession | two-phase heating",
        fontsize=11,
    )

    # Panel 1 — surface T
    ax = axes[0, 0]
    ax.plot(ts, T_wall,    "r-",  lw=2,   label="SCAM T_wall")
    ax.plot(tc, Tc,        "k--", lw=1.5, label="CHyPS T0 (surface)")
    for xv in (60, 90): ax.axvline(xv, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface temperature"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2 — deep TC probes (survive full run): 16 mm and 24 mm
    ax = axes[0, 1]
    for idx, col in zip([5, 6], ["tab:blue", "tab:orange"]):  # T6=16mm, T7=24mm
        name = names[idx]; d_mm = depths[idx] * 1000
        scam_T = T_tc[idx]
        ct, cT = chyps[name]
        ax.plot(ts,  scam_T, color=col, lw=2,   label=f"SCAM {d_mm:.0f} mm")
        ax.plot(ct,  cT,     color=col, lw=1.5, ls="--", label=f"CHyPS {name} ({d_mm:.0f} mm)")
    for xv in (60, 90): ax.axvline(xv, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC probes — 16 mm and 24 mm"); ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3 — recession
    ax = axes[0, 2]
    ax.plot(ts, rec_s * 1000, "b-",  lw=2,   label="SCAM recession")
    ax.plot(tc, rec_c * 1000, "b--", lw=1.5, label="CHyPS recession")
    for xv in (60, 90): ax.axvline(xv, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Recession [mm]")
    ax.set_title("Surface recession"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4 — short-lived probes (T1=1mm, T3=4mm, T5=12mm until recession)
    ax = axes[1, 0]
    for idx, col in zip([0, 2, 4], ["tab:blue", "tab:green", "tab:red"]):
        name = names[idx]; d_mm = depths[idx] * 1000
        scam_T = T_tc[idx].copy()
        ct, cT = chyps[name]
        ax.plot(ts,  scam_T, color=col, lw=2,   label=f"SCAM {d_mm:.0f} mm")
        ax.plot(ct,  cT,     color=col, lw=1.5, ls="--", label=f"CHyPS {name} ({d_mm:.0f} mm)")
    for xv in (60, 90): ax.axvline(xv, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC probes consumed by recession — 1, 4, 12 mm")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Panel 5 — surface T difference
    ax = axes[1, 1]
    Tc_on_s = np.interp(ts, tc, Tc)
    diff = T_wall - Tc_on_s
    diff = diff.copy()
    diff[ts < 1.0] = np.nan
    diff[(ts > 59.5) & (ts < 60.5)] = np.nan
    diff[(ts > 89.5) & (ts < 90.5)] = np.nan
    ax.plot(ts, diff, "r-", lw=1.5, label="surface")
    ax.axhline(0, color="k", lw=0.8)
    for xv in (60, 90): ax.axvline(xv, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − CHyPS [K]")
    ax.set_title("Surface T difference SCAM − CHyPS"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6 — T profile snapshots at t≈30 s, 60 s, 90 s
    ax = axes[1, 2]
    snap_times = {30: None, 60: None, 90: None}
    for snap in results.snapshots:
        for t_tgt in list(snap_times):
            if snap_times[t_tgt] is None and abs(snap.time - t_tgt) < 0.4:
                snap_times[t_tgt] = snap
    palette = {"30": "tab:blue", "60": "tab:orange", "90": "tab:red"}
    for t_tgt, snap in snap_times.items():
        if snap is None: continue
        col = palette[str(t_tgt)]
        ax.plot(snap.mesh.y_nodes * 1000, snap.T, color=col, lw=2,
                label=f"SCAM t={t_tgt} s")
    # CHyPS scatter from probe data at same times
    chyps_depths_all = [0.0] + depths
    chyps_names_all  = ["T0"] + names
    for t_tgt, col in zip([30, 60, 90], [palette["30"], palette["60"], palette["90"]]):
        ys, Ts = [], []
        for nm, d in zip(chyps_names_all, chyps_depths_all):
            ct2, cT2 = (tc, Tc) if nm == "T0" else chyps[nm]
            if t_tgt > ct2[-1]: continue
            ys.append(d * 1000)
            Ts.append(float(np.interp(t_tgt, ct2, cT2)))
        ax.plot(ys, Ts, "o--", color=col, ms=5, lw=1.2, label=f"CHyPS t={t_tgt} s")
    ax.set_xlabel("Depth from original surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t = 30, 60, 90 s")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # atomic write
    tmp = OUT_PNG.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=150)
    tmp.replace(OUT_PNG)
    print(f"\nPlot saved → {OUT_PNG}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    valid = ~np.isnan(diff)
    ph1 = valid & (ts >= 1.0) & (ts <= 59.5)
    ph2 = valid & (ts >= 60.5) & (ts <= 89.5)
    cd  = valid & (ts >= 90.5)
    print("\n--- Surface temperature SCAM vs CHyPS ---")
    for t_tgt in (10, 30, 60, 75, 90, 100):
        Ts_v = float(np.interp(t_tgt, ts, T_wall))
        Tc_v = float(np.interp(t_tgt, tc, Tc))
        print(f"  t={t_tgt:5.0f} s: SCAM={Ts_v:7.1f} K  CHyPS={Tc_v:7.1f} K  Δ={Ts_v-Tc_v:+.1f} K")
    if ph1.any(): print(f"  max |Δ| phase 1 (1–59.5 s): {np.abs(diff[ph1]).max():.1f} K")
    if ph2.any(): print(f"  max |Δ| phase 2 (60.5–89.5 s): {np.abs(diff[ph2]).max():.1f} K")
    if cd.any():  print(f"  max |Δ| cooldown (>90.5 s): {np.abs(diff[cd]).max():.1f} K")

    print("\n--- Recession ---")
    for t_tgt in (30, 60, 90, 100):
        rs_v = float(np.interp(t_tgt, ts, rec_s)) * 1000
        rc_v = float(np.interp(t_tgt, tc, rec_c)) * 1000
        print(f"  t={t_tgt:3.0f} s: SCAM={rs_v:.2f} mm  CHyPS={rc_v:.2f} mm  Δ={rs_v-rc_v:+.3f} mm")

    print("\n--- Deep TC probes (16 mm, 24 mm) ---")
    for idx in [5, 6]:
        nm = names[idx]; d_mm = depths[idx] * 1000
        ct2, cT2 = chyps[nm]
        T_scam_i = T_tc[idx]
        for t_tgt in (30, 60, 90, 100):
            Ts_v = float(np.interp(t_tgt, ts, T_scam_i))
            Tc_v = float(np.interp(t_tgt, ct2, cT2))
            print(f"  {nm} ({d_mm:.0f} mm) t={t_tgt:3.0f} s: SCAM={Ts_v:.1f} K  CHyPS={Tc_v:.1f} K  Δ={Ts_v-Tc_v:+.1f} K")


if __name__ == "__main__":
    main()
