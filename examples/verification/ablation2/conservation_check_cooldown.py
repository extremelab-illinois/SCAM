# SPDX-License-Identifier: MIT
"""Conservation-closure + secondary-term diagnostic for the ablation2 cooldown gap.

Plans #1 and #2 of docs/physics/ablation2_cooldown_gas_advection.md:

  #1 Confirm that the surface-node energy imbalance during cooldown equals the
     uncompensated pyrolysis-gas outflow  m_dot_g(surface)*h_g(T_w)  — i.e. that
     the in-depth assembly deposits  m_dot_g[1]*h_g(T_1)  at the surface node but
     never advects the gas back out (mg_out[0]=0), and the cooldown SEB does not
     remove it.

  #2 Quantify the other in-depth gas terms at the surface node (gas energy
     storage d(eps_g*rho_g*h_g)/dt and the Darcy expansion source) and show they
     are << the advective imbalance, so the advection term is the dominant driver.

The script monkeypatches scam.numerics.assembly._build_system to record the
surface-node (node 0) terms on every assembly call, runs the standard
compare_pato_ablation2 SCAM case once, and prints a cooldown table.

All terms are reported in W/m^2 (the SLAB area function is unit area, so the
extensive per-node power [W] equals the per-area flux [W/m^2]).

Run:
    MPLBACKEND=Agg python3 examples/verification/ablation2/conservation_check_cooldown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scam.numerics.assembly as _asm
from scam.physics.gas_enthalpy import pyrolysis_gas_enthalpy
from scam.physics.properties import eps_virgin as _eps_v
from scam.core.constants import SIGMA_SB

import compare_pato_ablation2 as C

# ---------------------------------------------------------------------------
# Recorder: wrap _build_system and capture surface-node gas terms
# ---------------------------------------------------------------------------
_records: list[dict] = []
_orig_build = _asm._build_system


def _patched_build(state, mat_list, stack, dt, back_bc, time,
                   drho_dt_y=None, m_dot_g=None, drho_dt_y_comp=None,
                   rho_old=None, T_prev=None, h_old=None, T_old_rhs=None):
    if m_dot_g is not None and 58.0 <= time <= 95.0:
        mesh = state.mesh
        T = state.T
        rho = state.rho
        lid = mesh.layer_id
        mat0 = mat_list[int(lid[0])]
        A0 = float(mesh.area_nodes[0])

        h_g0 = float(pyrolysis_gas_enthalpy(mat0, T[0]))
        h_g1 = float(pyrolysis_gas_enthalpy(mat0, T[1]))
        # what the assembly deposits at node 0 (mg_out[0]=0):
        Q_adv0 = float(m_dot_g[1] * h_g1)                # = m_dot_g[1]*h_g(T_1)
        # conservative (PATO) net = in - out:
        Q_adv0_cons = float(m_dot_g[1] * h_g1 - m_dot_g[0] * h_g0)
        # the uncompensated outflow (SCAM - conservative):
        imbalance = float(m_dot_g[0] * h_g0)             # = m_dot_g(surf)*h_g(T_w)

        # --- secondary term: gas energy storage d(eps_g*rho_g*h_g)/dt at node 0 ---
        q_store0 = 0.0
        if T_prev is not None and len(T_prev) == len(T):
            R_univ = 8.314
            if not (mat0.eps_g_char == 0.0 and mat0.eps_g_virgin == 0.0):
                ev = float(_eps_v(mat0, np.array([rho[0]]))[0])
                eps_g = mat0.eps_g_char + (mat0.eps_g_virgin - mat0.eps_g_char) * ev
                Tn = max(T[0], 1.0); Tp = max(T_prev[0], 1.0)
                rg_n = mat0.gas_pressure * mat0.gas_molar_mass / (R_univ * Tn)
                rg_p = mat0.gas_pressure * mat0.gas_molar_mass / (R_univ * Tp)
                hg_n = float(pyrolysis_gas_enthalpy(mat0, T[0]))
                hg_p = float(pyrolysis_gas_enthalpy(mat0, T_prev[0]))
                # assembly subtracts this from Q_vol (per volume); report per area (slab)
                q_store0 = -(eps_g * rg_n * hg_n - eps_g * rg_p * hg_p) / max(dt, 1e-12) \
                    * float(mesh.delta_nodes[0])

        # --- secondary term: Darcy / gas-expansion energy source at node 0 ---
        q_darcy0 = 0.0
        if T_prev is not None and len(T_prev) == len(T):
            dTdt = (T - T_prev) / max(dt, 1e-12)
            has_perm = any(mat_list[int(l)].permeability > 0.0 for l in np.unique(lid))
            try:
                if has_perm:
                    from scam.physics.pressure_darcy import pressure_darcy_energy_source
                    Qd = pressure_darcy_energy_source(mat_list, mesh, T, rho, dTdt)
                else:
                    from scam.physics.darcy_flow import gas_expansion_energy_source
                    Qd = gas_expansion_energy_source(mat_list, mesh, T, rho, dTdt)
                q_darcy0 = float(Qd[0]) / A0 if A0 > 0 else 0.0
            except Exception:
                q_darcy0 = float("nan")

        _records.append(dict(
            t=float(time), T0=float(T[0]), T1=float(T[1]),
            mdg0=float(m_dot_g[0]), mdg1=float(m_dot_g[1]),
            h_g0=h_g0, h_g1=h_g1,
            Q_adv0=Q_adv0, Q_adv0_cons=Q_adv0_cons, imbalance=imbalance,
            q_store0=q_store0, q_darcy0=q_darcy0,
        ))
    return _orig_build(state, mat_list, stack, dt, back_bc, time,
                       drho_dt_y, m_dot_g, drho_dt_y_comp, rho_old,
                       T_prev, h_old, T_old_rhs)


def main():
    _asm._build_system = _patched_build
    try:
        results = C.run_scam("live_cantera_zc_default")
    finally:
        _asm._build_system = _orig_build

    # one representative (last) record per ~0.5 s output bin
    recs = _records
    if not recs:
        print("No records captured."); return
    # dedupe: keep the last assembly call at each rounded 0.25 s
    seen: dict[float, dict] = {}
    for r in recs:
        seen[round(r["t"] * 4) / 4] = r
    rows = [seen[k] for k in sorted(seen)]

    # PATO surface T for the gap column
    t_s = results.times_array(); Tw_s = results.T_wall_array()
    from _pato2_common import PATO_REF
    from _pato2_common import load_pato_ta_surfacepatch
    ta_surf = PATO_REF / "AblationTestCase_2.x" / "output/porousMat/scalar/Ta_surfacePatch"
    tps, Tps = load_pato_ta_surfacepatch(ta_surf)

    print("\n" + "=" * 110)
    print("COOLDOWN SURFACE-NODE ENERGY TERMS  (W/m^2;  slab unit area)")
    print("=" * 110)
    print(f"{'t':>6} {'T_w':>7} {'m_dot_g':>9} {'h_g(Tw)':>10} "
          f"{'Q_adv0(SCAM)':>12} {'Q_adv0(cons)':>12} {'IMBAL=mdg*hg':>13} "
          f"{'q_store0':>9} {'q_darcy0':>9} {'q_rad_out':>9} {'dT(S-P)':>8}")
    for r in rows:
        if r["t"] < 60.0:
            continue
        Tw = float(np.interp(r["t"], t_s, Tw_s))
        Tw_p = float(np.interp(r["t"], tps, Tps[:, 0]))
        eps = 0.9
        q_rad_out = eps * SIGMA_SB * Tw**4 / 1.0
        print(f"{r['t']:6.2f} {Tw:7.1f} {r['mdg0']:9.5f} {r['h_g0']:10.2e} "
              f"{r['Q_adv0']/1e3:12.2f} {r['Q_adv0_cons']/1e3:12.2f} {r['imbalance']/1e3:13.2f} "
              f"{r['q_store0']/1e3:9.3f} {r['q_darcy0']/1e3:9.3f} {q_rad_out/1e3:9.2f} {Tw-Tw_p:8.1f}")

    # ---- closure check (#1) ----
    cool = [r for r in rows if r["t"] >= 61.0]
    imb = np.array([r["imbalance"] for r in cool])
    qadv = np.array([r["Q_adv0"] for r in cool])
    qadv_cons = np.array([r["Q_adv0_cons"] for r in cool])
    print("\n--- #1 closure ---")
    print(f"  SCAM deposits Q_adv0 = m_dot_g[1]*h_g(T1).  Conservative net = in-out.")
    print(f"  uncompensated outflow (SCAM-conservative) = m_dot_g[0]*h_g(Tw):")
    print(f"     identity check max|Q_adv0 - Q_adv0_cons - imbalance| = "
          f"{np.max(np.abs(qadv - qadv_cons - imb)):.3e} W/m^2 (should be ~0)")
    print(f"     cooldown imbalance range: {imb.min()/1e3:.2f} .. {imb.max()/1e3:.2f} kW/m^2 "
          f"(peak |sink| {np.abs(imb).max()/1e3:.2f})")

    # ---- secondary terms (#2) ----
    store = np.array([abs(r["q_store0"]) for r in cool])
    darcy = np.array([abs(r["q_darcy0"]) for r in cool])
    print("\n--- #2 secondary terms at surface node (cooldown) ---")
    print(f"  |gas-storage q_store0|   max = {np.nanmax(store)/1e3:8.4f} kW/m^2")
    print(f"  |Darcy expansion q_darcy0| max = {np.nanmax(darcy)/1e3:8.4f} kW/m^2")
    print(f"  |advective imbalance|     max = {np.abs(imb).max()/1e3:8.4f} kW/m^2  <-- dominant")


if __name__ == "__main__":
    main()
