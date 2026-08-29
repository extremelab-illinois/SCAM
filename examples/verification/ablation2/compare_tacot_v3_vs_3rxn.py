# SPDX-License-Identifier: MIT
"""Compare TACOT v3.0 2-reaction vs 3-reaction kinetics on AblationTestCase_2.x.

Both variants use the same surface B' table (tacot_v3.0_bprime_air.yaml) and
identical material properties; only the in-depth Arrhenius decomposition
kinetics differ.  The expected result is near-identical outputs — any
divergence points to a kinetics-sensitive path in the in-depth solver.

Run:
    MPLBACKEND=Agg python3 examples/verification/ablation2/compare_tacot_v3_vs_3rxn.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(str(Path(__file__).resolve().parent)))

from _pato2_common import (
    C_M, P_EDGE, RHOUE_CH, H_R, T_END, T_INIT, TC_DEPTHS,
    _load_pato_hg, prepare_plot_output, save_plot_atomic,
    T_at_original_depths,
)

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

OUT_PNG     = Path(__file__).parent / "compare_tacot_v3_vs_3rxn.png"
YAML_2RXN   = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"
YAML_3RXN   = REPO / "scam/materials/ablative_organic/tacot_v3.0_3rxn.yaml"
BPRIME_CONFIG = REPO / "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"

THICKNESS = 0.05   # m
N_NODES   = 201

TC_LABELS_MM = [1, 2, 4, 8, 12, 16, 24]

_T_STEPS       = np.array([0.0, 0.1, 60.0, 60.1, T_END])
_RHOUECH_VALS  = np.array([0.01 * RHOUE_CH, RHOUE_CH, RHOUE_CH,
                            0.01 * RHOUE_CH, 0.01 * RHOUE_CH])
_CHEM_RHOUE_VALS = np.array([0.01 * RHOUE_CH, RHOUE_CH, RHOUE_CH, 0.0, 0.0])
_HR_VALS       = np.array([0.0, H_R, H_R, 0.0, 0.0])


def _rhoUeCH(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _RHOUECH_VALS))


def _h_r(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _HR_VALS))


def _rhoue(t: float) -> float:
    return float(np.interp(t, _T_STEPS, _CHEM_RHOUE_VALS))


def _make_bprime_backend():
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
    ev.lookup(1500.0, P_EDGE, 0.05, None)
    z_c = ev.pyrolysis_target_fraction("C")
    return _DefaultZC(ev, z_c)


def run_scam(yaml_path: Path, label: str):
    mat, bpt_yaml = load_material(str(yaml_path))
    hg_t, hg_val = _load_pato_hg()
    mat = dataclasses.replace(mat,
                              h_g_table=np.column_stack([hg_t, hg_val]),
                              h_g_abs_offset=None)

    try:
        bpt = _make_bprime_backend()
        print(f"  {label}: using live Cantera backend")
    except Exception:
        bpt = bpt_yaml
        print(f"  {label}: Cantera backend unavailable — using B' table")

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
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=300.0,
        rho_e_u_e=_rhoue,
        C_M=C_M,
        p_e=P_EDGE,
        lambda_blowing=0.5,
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
        continuous_remap=True,
    )

    print(f"Running SCAM — {label} ...")
    return run(stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc,
               options, initial_T=T_INIT, verbose=True)


def main() -> None:
    prepare_plot_output(OUT_PNG)

    res2 = run_scam(YAML_2RXN, "v3 2-rxn")
    res3 = run_scam(YAML_3RXN, "v3 3-rxn")

    t2 = res2.times_array();  T_wall2 = res2.T_wall_array();  rec2 = res2.s_array()
    t3 = res3.times_array();  T_wall3 = res3.T_wall_array();  rec3 = res3.s_array()

    T_tc2 = T_at_original_depths(res2, TC_DEPTHS)
    T_tc3 = T_at_original_depths(res3, TC_DEPTHS)

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
        "SCAM: TACOT v3.0 2-reaction vs 3-reaction kinetics\n"
        "AblationTestCase_2.x setup — full SEB + B' chemistry + ALE recession",
        fontsize=11,
    )

    # Panel 1 — surface T history
    ax = axes[0, 0]
    ax.plot(t2, T_wall2, "b-",  lw=2,   label="2-rxn")
    ax.plot(t3, T_wall3, "r--", lw=1.5, label="3-rxn")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Surface temperature"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2 — TC temperatures at 1, 4, 12 mm
    ax = axes[0, 1]
    plot_idx = [0, 2, 4]
    for i, col in zip(plot_idx, colors):
        d_mm = TC_LABELS_MM[i]
        ax.plot(t2, T_tc2[i], color=col, lw=2,   label=f"2-rxn {d_mm} mm")
        ax.plot(t3, T_tc3[i], color=col, lw=1.5, ls="--", label=f"3-rxn {d_mm} mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("TC temperatures — 1, 4, 12 mm"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3 — recession
    ax = axes[0, 2]
    ax.plot(t2, rec2 * 1000, "b-",  lw=2,   label="2-rxn")
    ax.plot(t3, rec3 * 1000, "r--", lw=1.5, label="3-rxn")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Recession [mm]")
    ax.set_title("Surface recession"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 4 — T profile snapshots at t≈30 s
    ax = axes[1, 0]
    snap2_30 = snap3_30 = snap2_90 = snap3_90 = None
    for snap in res2.snapshots:
        if abs(snap.time - 30) < 0.3 and snap2_30 is None:
            snap2_30 = snap
        if abs(snap.time - 90) < 0.3 and snap2_90 is None:
            snap2_90 = snap
    for snap in res3.snapshots:
        if abs(snap.time - 30) < 0.3 and snap3_30 is None:
            snap3_30 = snap
        if abs(snap.time - 90) < 0.3 and snap3_90 is None:
            snap3_90 = snap
    if snap2_30 is not None:
        ax.plot(snap2_30.mesh.y_nodes * 1000, snap2_30.T, "b-",  lw=2,   label="2-rxn t=30 s")
    if snap3_30 is not None:
        ax.plot(snap3_30.mesh.y_nodes * 1000, snap3_30.T, "b--", lw=1.5, label="3-rxn t=30 s")
    if snap2_90 is not None:
        ax.plot(snap2_90.mesh.y_nodes * 1000, snap2_90.T, "r-",  lw=2,   label="2-rxn t=90 s")
    if snap3_90 is not None:
        ax.plot(snap3_90.mesh.y_nodes * 1000, snap3_90.T, "r--", lw=1.5, label="3-rxn t=90 s")
    ax.set_xlabel("Depth from original surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t≈30 s and t≈90 s")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 5 — surface T difference (3-rxn − 2-rxn)
    ax = axes[1, 1]
    T_wall3_on2 = np.interp(t2, t3, T_wall3)
    diff_surf = T_wall3_on2 - T_wall2
    diff_surf = diff_surf.copy()
    diff_surf[t2 < 2.0] = np.nan
    diff_surf[(t2 > 59.9) & (t2 < 61.1)] = np.nan
    ax.plot(t2, diff_surf, "k-", lw=1.5, label="3-rxn − 2-rxn")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ΔT [K]")
    ax.set_title("Surface T difference: 3-rxn − 2-rxn"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6 — surface mass fluxes
    ax = axes[1, 2]
    m_dot_g2 = np.array([s.m_dot_pyro for s in res2.snapshots])
    m_dot_c2 = np.array([s.m_dot_char for s in res2.snapshots])
    m_dot_g3 = np.array([s.m_dot_pyro for s in res3.snapshots])
    m_dot_c3 = np.array([s.m_dot_char for s in res3.snapshots])
    ax.plot(t2, m_dot_g2, "g-",  lw=2,   label="2-rxn ṁ_gas")
    ax.plot(t3, m_dot_g3, "g--", lw=1.5, label="3-rxn ṁ_gas")
    ax.plot(t2, m_dot_c2, "r-",  lw=2,   label="2-rxn ṁ_char")
    ax.plot(t3, m_dot_c3, "r--", lw=1.5, label="3-rxn ṁ_char")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Mass flux [kg/m²/s]")
    ax.set_title("Surface pyrolysis gas and char fluxes")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_atomic(fig, OUT_PNG, dpi=150)
    print(f"\nPlot saved → {OUT_PNG}")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n--- Surface temperature comparison ---")
    for target in (10.0, 30.0, 60.0, 90.0, 120.0):
        Ts2 = float(np.interp(target, t2, T_wall2))
        Ts3 = float(np.interp(target, t3, T_wall3))
        print(f"  t={target:5.1f} s : 2-rxn={Ts2:7.1f} K, 3-rxn={Ts3:7.1f} K, Δ={Ts3 - Ts2:+6.2f} K")

    heat_valid = (t2 >= 2.0) & (t2 <= 59.9)
    cool_valid = t2 >= 61.1
    d_surf = T_wall3_on2 - T_wall2
    if heat_valid.any():
        print(f"  max |Δ| heating: {np.abs(d_surf[heat_valid & ~np.isnan(d_surf)]).max():.3f} K")
    if cool_valid.any():
        print(f"  max |Δ| cooling: {np.abs(d_surf[cool_valid & ~np.isnan(d_surf)]).max():.3f} K")

    print("\n--- Recession ---")
    for target in (60.0, 120.0):
        r2 = float(np.interp(target, t2, rec2)) * 1000
        r3 = float(np.interp(target, t3, rec3)) * 1000
        print(f"  t={target:.0f} s : 2-rxn={r2:.3f} mm, 3-rxn={r3:.3f} mm, Δ={r3 - r2:+.4f} mm")

    print("\n--- Surface mass fluxes at t=60 s ---")
    mg2 = float(np.interp(60.0, t2, m_dot_g2))
    mc2 = float(np.interp(60.0, t2, m_dot_c2))
    mg3 = float(np.interp(60.0, t3, m_dot_g3))
    mc3 = float(np.interp(60.0, t3, m_dot_c3))
    print(f"  ṁ_gas : 2-rxn={mg2:.5f}, 3-rxn={mg3:.5f}, Δ={mg3 - mg2:+.6f} kg/m²/s")
    print(f"  ṁ_char: 2-rxn={mc2:.5f}, 3-rxn={mc3:.5f}, Δ={mc3 - mc2:+.6f} kg/m²/s")


if __name__ == "__main__":
    main()
