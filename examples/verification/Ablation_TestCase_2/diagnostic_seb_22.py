#!/usr/bin/env python3
"""SEB diagnostic for Ablation Test Case 2.2.

Runs TC2.2 with the SPARSE h_g fallback (no dense PATO table loading),
then prints a full breakdown of SEB terms at t≈60s so we can compare
them against PATO v1.3.3 expected values term by term.

Run:
    MPLBACKEND=Agg python3 examples/verification/Ablation_TestCase_2/diagnostic_seb_22.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MATERIAL_FILE = REPO / "scam/materials/ablative_organic/tacot_v2.2.yaml"
BPRIME_CONFIG_MPP = REPO / "scam/materials/ablative_organic/tacot_v2.2_mpp_config.yaml"

sys.path.insert(0, str(REPO))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run
from scam.physics.gas_enthalpy import pyrolysis_gas_enthalpy

T_INIT = 300.0
T_END = 60.5   # only run to t=60s for diagnostic
THICKNESS = 0.05
N_NODES = 201
P_EDGE = 101325.0
RHOUE_CH = 0.3
H_RECOVERY = 1.5e6
SIGMA = 5.670374419e-8

# SPARSE h_g table — 16 representative points, absolute reference (h_g(300K)=-7.09 MJ/kg).
# This is the same fallback used by the main TC2.2 script when PATO gasProperties is absent.
_HG_SPARSE_T = np.array([200., 300., 500., 700., 800., 900., 1000., 1100., 1200.,
                          1300., 1400., 1500., 1644., 2000., 3000., 4000.])
_HG_SPARSE_H = np.array([-7.247e6, -7.090e6, -6.715e6, -6.005e6, -5.014e6, -3.335e6,
                          -2.170e6, -1.789e6, -1.199e6, -5.255e5, 1.299e5, 1.137e6,
                           2.625e6,  4.400e6,  1.100e7,  2.200e7])

_BC_TIMES    = np.array([0.0, 0.1, 60.0, 60.1, 120.0])
_RHOUECH_ARR = np.array([0.003, RHOUE_CH, RHOUE_CH, 0.003, 0.003])
_CHEM_ARR    = np.array([0.003, RHOUE_CH, RHOUE_CH, 0.0, 0.0])
_HR_ARR      = np.array([0.0, H_RECOVERY, H_RECOVERY, 0.0, 0.0])

def _rho_ue_ch(t): return float(np.interp(t, _BC_TIMES, _RHOUECH_ARR))
def _chem_rho(t):  return float(np.interp(t, _BC_TIMES, _CHEM_ARR))
def _h_r(t):       return float(np.interp(t, _BC_TIMES, _HR_ARR))


def main():
    material, bprime_table = load_material(str(MATERIAL_FILE))
    # Use material yaml as-is: sensible h_g for in-depth transport,
    # h_g_abs_offset applied in surface_energy.py for SEB q_adv_pyro.

    print("Loading Mutation++ backend ...")
    try:
        from scam.physics.mpp_evaluator import MutationppEvaluator
        bprime = MutationppEvaluator.from_config(str(BPRIME_CONFIG_MPP))
        bprime.lookup(1500.0, P_EDGE, 0.05)
        print("  Mpp loaded OK")
    except Exception as e:
        print(f"  Mpp failed: {e}, using table fallback")
        bprime = bprime_table

    stack = StackConfig(layers=[LayerConfig(material.name, thickness=THICKNESS,
                                            n_nodes=N_NODES, n_subcells=4)])
    geometry = GeometryConfig(geometry_type=GeometryType.SLAB)
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=_rho_ue_ch,
        h_r=_h_r,
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=T_INIT,
        rho_e_u_e=_chem_rho,
        C_M=1.0,
        p_e=P_EDGE,
        lambda_blowing=0.5,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.25,
        dt_min=1e-5,
        dt_max_dT=30.0,
        dt_max_drho_frac=0.05,
        output_dt=0.1,
        tc_positions=[],
        allow_recession=True,
        continuous_remap=True,
        use_rho_old=True,
    )

    print("Running SCAM TC2.2 (sparse h_g) to t=60.5s ...")
    results = run(
        stack, {material.name: material}, {material.name: bprime},
        geometry, surface_bc, back_bc, options,
        initial_T=T_INIT, verbose=False,
    )

    # --- Extract state at t≈60s ---
    times = results.times_array()
    Twall = results.T_wall_array()
    mdot_g_arr = np.array([s.m_dot_pyro for s in results.snapshots])
    mdot_c_arr = np.array([s.m_dot_char for s in results.snapshots])
    qcond_arr  = np.array([s.q_cond for s in results.snapshots])

    print(f"\nAll snap times near t=60: {times[(times>=59.5) & (times<=60.5)]}")
    print(f"All Twall near t=60: {Twall[(times>=59.5) & (times<=60.5)]}")

    # Use last snap at or before t=60.0 (peak heating phase)
    heating_mask = times <= 60.01
    idx60 = np.where(heating_mask)[0][-1]
    t60    = times[idx60]
    T_w    = Twall[idx60]
    m_dot_pyro = mdot_g_arr[idx60]
    m_dot_char = mdot_c_arr[idx60]
    q_cond = qcond_arr[idx60]

    print(f"\n=== SCAM state at t={t60:.3f} s ===")
    print(f"  T_wall      = {T_w:.2f} K   (PATO: 1568.36 K, gap = {T_w-1568.36:+.2f} K)")
    print(f"  m_dot_pyro  = {m_dot_pyro:.6f} kg/m²/s  (PATO: 0.012577)")
    print(f"  m_dot_char  = {m_dot_char:.6f} kg/m²/s  (PATO: 0.044794 — possibly blown)")
    print(f"  q_cond      = {q_cond/1e3:.3f} kW/m²")

    # --- Recompute SEB terms at T_w ---
    # B'_g (unblown basis)
    B_g_unblown = m_dot_pyro / (RHOUE_CH * 1.0)
    print(f"\n=== SEB recomputation at T_w={T_w:.2f} K ===")
    print(f"  B'_g (SCAM, unblown) = {B_g_unblown:.5f}")

    # B' lookup
    B_c_unblown, h_wall = bprime.lookup(T_w, P_EDGE, B_g_unblown)
    m_dot_char_unblown = B_c_unblown * RHOUE_CH
    B_c_blown_basis = B_c_unblown
    B_total = (m_dot_pyro + m_dot_char_unblown) / RHOUE_CH
    blow = 1.0 / (1.0 + 0.5 * B_total)  # rational
    m_dot_char_blown = m_dot_char_unblown * blow
    rhoUeCH_eff = RHOUE_CH * blow
    q_conv = rhoUeCH_eff * (H_RECOVERY - h_wall)

    print(f"  B'_c (unblown)      = {B_c_unblown:.5f}")
    print(f"  B_total (unblown)   = {B_total:.5f}")
    print(f"  blow_factor (ratio) = {blow:.5f}")
    print(f"  h_wall              = {h_wall/1e6:.5f} MJ/kg")
    print(f"  m_dot_char (unblown)= {m_dot_char_unblown:.6f} kg/m²/s")
    print(f"  m_dot_char (blown)  = {m_dot_char_blown:.6f} kg/m²/s")
    print(f"  q_conv              = {q_conv/1e3:.3f} kW/m²  [=rhoUeCH*blow*(h_r-h_wall)]")

    # Surface enthalpies for q_adv
    h_g_w_cantera, h_c_w = bprime.surface_enthalpies(T_w, P_EDGE)
    h_g_w_sensible = float(pyrolysis_gas_enthalpy(material, T_w))
    h_g_w_offset = material.h_g_abs_offset or 0.0
    h_g_w = h_g_w_sensible + h_g_w_offset  # same as seb_residual with h_g_abs_offset
    print(f"\n  h_g (sensible)      = {h_g_w_sensible/1e6:.5f} MJ/kg")
    print(f"  h_g_abs_offset      = {h_g_w_offset/1e6:.5f} MJ/kg")
    print(f"  h_g (abs=sens+off)  = {h_g_w/1e6:.5f} MJ/kg")
    q_adv_pyro = m_dot_pyro * (h_g_w - h_wall)
    q_adv_char = m_dot_char_blown * (h_c_w - h_wall)
    q_adv = q_adv_pyro + q_adv_char

    print(f"  h_g (Cantera unreacted) = {h_g_w_cantera/1e6:.5f} MJ/kg")
    print(f"  h_c (Cantera graphite)  = {h_c_w/1e6:.5f} MJ/kg")
    print(f"  q_adv_pyro = m_dot_pyro*(h_g_abs-h_wall)   = {q_adv_pyro/1e3:.3f} kW/m²")
    print(f"  q_adv_char = m_dot_c_blown*(h_c-h_wall)    = {q_adv_char/1e3:.3f} kW/m²")
    print(f"  q_adv (total)       = {q_adv/1e3:.3f} kW/m²")

    # Radiation (emissivity from material — char value at surface)
    eps = material.emissivity_char if hasattr(material, "emissivity_char") else material.emissivity
    T_rad_in = T_INIT  # 300 K
    q_rad_in  = eps * SIGMA * T_rad_in**4
    q_rad_out = eps * SIGMA * T_w**4
    q_rad_net = q_rad_in - q_rad_out

    print(f"\n  eps (char surface)  = {eps:.3f}")
    print(f"  q_rad_in            = {q_rad_in/1e3:.3f} kW/m²")
    print(f"  q_rad_out           = {q_rad_out/1e3:.3f} kW/m²")
    print(f"  q_rad_net           = {q_rad_net/1e3:.3f} kW/m²")

    q_net_surface = q_conv + q_adv + q_rad_net
    q_cond_check = q_net_surface  # SEB: q_cond = q_conv + q_adv + q_rad_net

    print(f"\n  SEB CHECK:")
    print(f"  q_conv + q_adv + q_rad_net  = {q_net_surface/1e3:.3f} kW/m²")
    print(f"  q_cond (stored by solver)   = {q_cond/1e3:.3f} kW/m²")
    print(f"  Residual                    = {(q_cond - q_net_surface)/1e3:.4f} kW/m²")

    # --- PATO values at T=1568.36 K for comparison ---
    T_pato = 1568.36
    B_c_pato, h_wall_pato = bprime.lookup(T_pato, P_EDGE, 0.012577 / RHOUE_CH)
    mdotg_pato = 0.012577
    mdotc_pato_unblown = B_c_pato * RHOUE_CH
    B_tot_pato = (mdotg_pato + mdotc_pato_unblown) / RHOUE_CH
    blow_pato = 1.0 / (1.0 + 0.5 * B_tot_pato)
    mdotc_pato_blown = mdotc_pato_unblown * blow_pato
    q_conv_pato = RHOUE_CH * blow_pato * (H_RECOVERY - h_wall_pato)
    h_g_w_pato, h_c_w_pato = bprime.surface_enthalpies(T_pato, P_EDGE)
    h_g_sens_pato = float(pyrolysis_gas_enthalpy(material, T_pato))
    h_g_abs_pato = h_g_sens_pato + (material.h_g_abs_offset or 0.0)
    q_adv_pyro_pato = mdotg_pato * (h_g_abs_pato - h_wall_pato)
    q_adv_char_pato = mdotc_pato_blown * (h_c_w_pato - h_wall_pato)
    q_rad_out_pato  = eps * SIGMA * T_pato**4
    q_rad_net_pato  = eps * SIGMA * T_INIT**4 - q_rad_out_pato
    q_net_pato = q_conv_pato + q_adv_pyro_pato + q_adv_char_pato + q_rad_net_pato

    print(f"\n=== PATO T_wall=1568.36 K SEB (computed with SCAM physics) ===")
    print(f"  h_wall      = {h_wall_pato/1e6:.5f} MJ/kg")
    print(f"  B'_c        = {B_c_pato:.5f}")
    print(f"  B_total     = {B_tot_pato:.5f}")
    print(f"  blow        = {blow_pato:.5f}")
    print(f"  q_conv      = {q_conv_pato/1e3:.3f} kW/m²")
    print(f"  q_adv_pyro  = {q_adv_pyro_pato/1e3:.3f} kW/m²")
    print(f"  q_adv_char  = {q_adv_char_pato/1e3:.3f} kW/m²")
    print(f"  q_rad_net   = {q_rad_net_pato/1e3:.3f} kW/m²")
    print(f"  q_cond_needed (=q_net) = {q_net_pato/1e3:.3f} kW/m²")
    print(f"\n  If PATO needs q_cond={q_net_pato/1e3:.3f} kW/m² at T=1568K")
    print(f"  and SCAM's q_cond at T_wall={T_w:.1f}K = {q_cond/1e3:.3f} kW/m²,")
    print(f"  then SCAM's F_cond relationship gives DIFFERENT q_cond for same T_wall")
    print(f"  → root cause is in q_cond(T_wall) alpha_F/beta_F being different\n")

    # Compare q_conv at PATO vs SCAM T_wall
    print(f"=== Term comparison at two T_wall values ===")
    print(f"{'Term':<25} {'PATO T=1568.4K':>18} {'SCAM T={:.1f}K'.format(T_w):>18} {'diff':>12}")
    print("-"*75)
    def row(name, v1, v2):
        print(f"{name:<25} {v1/1e3:>17.3f} {v2/1e3:>17.3f} {(v2-v1)/1e3:>+11.3f}")
    row("q_conv (kW/m²)", q_conv_pato, q_conv)
    row("q_adv_pyro (kW/m²)", q_adv_pyro_pato, q_adv_pyro)
    row("q_adv_char (kW/m²)", q_adv_char_pato, q_adv_char)
    row("q_rad_net (kW/m²)", q_rad_net_pato, q_rad_net)
    row("q_net→q_cond (kW/m²)", q_net_pato, q_net_surface)
    row("q_cond actual (kW/m²)", q_net_pato, q_cond)  # PATO q_cond from PATO's in-depth


if __name__ == "__main__":
    main()
