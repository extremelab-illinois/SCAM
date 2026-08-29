# SPDX-License-Identifier: MIT
"""V5 — Coupled surface mass & energy balance (full ablation).

Top rung: the full ENERGY_BALANCE path is active — the SEB Newton solve for
T_wall, B'-table char ablation, the blowing correction, surface recession and
node drop/merge — all on top of the in-depth response verified in V1–V4. This
is the TACOT 3.0 arcjet regime of ``tests/verification/test_tacot_benchmark.py``;
here we add *conservation/consistency* checks rather than only plausibility
bounds.

There is no closed-form reference, so the verification recomputes the governing
balances from each stored snapshot and confirms they close:

    * SEB closure  — the surface energy balance, rebuilt from the reported
      T_wall / q_cond / m_dot_char / m_dot_pyro, is ~0 (the Newton solve really
      converged). This is the energy-conservation check at the surface, applied
      every output step; it supersedes a separate volumetric audit, which on a
      receding, node-dropping mesh is dominated by bookkeeping artifacts.
    * recession consistency — s_dot equals the surface mass balance, and the
      time-integral of s_dot matches the reported total recession.
    * blowing correction — alpha_eff <= alpha_conv and decreases as B' rises.
    * stability/plausibility — finite fields; T_wall, s_total, s_dot in range.
"""
import numpy as np
import pytest

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.physics.blowing import blowing_corrected_alpha
from scam.physics.surface_energy import seb_residual

from .conftest import run_case


# Surface BC parameters (kept as module constants so the checks can rebuild the
# balances exactly as the solver saw them).
ALPHA_CONV = 5000.0
T_AW = 8000.0
EMISSIVITY = 0.85
VIEW_FACTOR = 1.0
RHO_E_U_E = 0.1
C_M = 0.01
P_E = 10000.0
LAMBDA_BLOWING = 0.5
SEB_TOL = 1.0


@pytest.fixture(scope="module")
def ablation_run():
    mat, bpt = load_material("scam/materials/ablative_organic/tacot_v3.0.yaml")
    stack = StackConfig(layers=[LayerConfig(mat.name, 0.05, 51, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=ALPHA_CONV, T_aw=T_AW,
        emissivity=EMISSIVITY, view_factor=VIEW_FACTOR, T_rad_in=0.0,
        rho_e_u_e=RHO_E_U_E, C_M=C_M, p_e=P_E, lambda_blowing=LAMBDA_BLOWING,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=60.0, dt_init=0.01, dt_max=2.0, dt_min=1e-5,
        dt_max_dT=50.0, dt_max_drho_frac=0.05, output_dt=2.0,
        seb_tol=SEB_TOL,
    )
    results = run_case(stack, {mat.name: mat}, surface_bc, back_bc, options,
                       b_prime_tables={mat.name: bpt}, initial_T=300.0)
    return mat, bpt, results


def _seb_residual_from_snapshot(snap, mat, bpt):
    """Rebuild the active SEB residual using the stored conductive flux."""
    bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=ALPHA_CONV, T_aw=T_AW,
        emissivity=EMISSIVITY, view_factor=VIEW_FACTOR, T_rad_in=0.0,
        rho_e_u_e=RHO_E_U_E, C_M=C_M, p_e=P_E, lambda_blowing=LAMBDA_BLOWING,
    )
    res, _q_cond, _m_dot_char, _h_wall = seb_residual(
        snap.T_wall,
        0.0,
        snap.q_cond,
        bc,
        snap.time,
        snap.m_dot_pyro,
        mat,
        bpt,
        emissivity_override=EMISSIVITY,
        rho_surface=snap.rho_wall,
    )
    return res


def test_v5_seb_closure(ablation_run):
    mat, bpt, results = ablation_run
    checked = 0
    for snap in results.snapshots:
        if snap.T_wall <= 300.01:   # initial snapshot, SEB not yet solved
            continue
        res = _seb_residual_from_snapshot(snap, mat, bpt)
        # Newton converges the residual below seb_tol; allow generous slack for
        # the table re-evaluation here.
        assert abs(res) < 100.0 * SEB_TOL, (
            f"t={snap.time:.1f}s: SEB does not close, residual={res:.3e} W/m^2 "
            f"(T_wall={snap.T_wall:.1f} K). Newton may not have converged."
        )
        checked += 1
    assert checked > 0, "no post-ignition snapshots to check."


def test_v5_recession_consistency(ablation_run):
    _mat, _bpt, results = ablation_run
    # 1. Per-step CMA surface mass balance: s_dot = m_dot_char / max(rho_wall, rho_char).
    #    Uses current surface density (PATO-consistent); clamps to rho_char once fully charred.
    #    Pyrolysis gas exits via Darcy flow and does not contribute to recession.
    from scam.io.material_loader import load_material
    mat, _ = load_material("scam/materials/ablative_organic/tacot_v3.0.yaml")
    rho_char = mat.rho_char
    for snap in results.snapshots:
        if snap.s_dot <= 0.0:
            continue
        rho_denom = max(snap.rho_wall, rho_char)
        expected = snap.m_dot_char / rho_denom
        rel = abs(snap.s_dot - expected) / max(expected, 1e-12)
        assert rel < 1e-6, (
            f"t={snap.time:.1f}s: s_dot={snap.s_dot:.3e} != m_dot_char/max(rho_wall,rho_char) {expected:.3e} m/s."
        )

    # 2. Reported total recession matches the time-integral of s_dot.
    t = results.times_array()
    s_dot = results.s_dot_array()
    s_total = results.s_array()[-1]
    s_integral = float(np.sum(0.5 * (s_dot[1:] + s_dot[:-1]) * np.diff(t)))
    rel = abs(s_total - s_integral) / max(s_total, 1e-12)
    assert rel < 0.05, (
        f"recession mismatch: reported s_total={s_total * 1e3:.3f} mm vs "
        f"∫s_dot dt={s_integral * 1e3:.3f} mm ({rel * 100:.2f}% > 5%)."
    )


def test_v5_blowing_correction(ablation_run):
    _mat, _bpt, results = ablation_run
    # At run conditions the effective alpha must never exceed the un-blown value.
    for snap in results.snapshots:
        alpha_eff = blowing_corrected_alpha(
            ALPHA_CONV, snap.m_dot_char, snap.m_dot_pyro,
            RHO_E_U_E, C_M, LAMBDA_BLOWING,
        )
        assert alpha_eff <= ALPHA_CONV + 1e-9, (
            f"t={snap.time:.1f}s: blowing raised alpha ({alpha_eff:.1f} > "
            f"{ALPHA_CONV})."
        )

    # The correction must be monotonically decreasing in total mass injection.
    B_vals = [0.0, 0.1, 0.5, 1.0, 2.0]
    alphas = [
        blowing_corrected_alpha(ALPHA_CONV, b * RHO_E_U_E * C_M, 0.0,
                                RHO_E_U_E, C_M, LAMBDA_BLOWING)
        for b in B_vals
    ]
    assert all(a2 < a1 for a1, a2 in zip(alphas, alphas[1:])), (
        f"blowing-corrected alpha not strictly decreasing in B': {alphas}"
    )


def test_v5_stability_and_plausibility(ablation_run):
    _mat, _bpt, results = ablation_run
    for snap in results.snapshots:
        assert np.all(np.isfinite(snap.T)), f"non-finite T at t={snap.time:.1f}s"
        assert np.all(np.isfinite(snap.rho)), f"non-finite rho at t={snap.time:.1f}s"

    T_wall_final = results.T_wall_array()[-1]
    s_final = results.s_array()[-1]
    s_dot_final = results.s_dot_array()[-1]
    assert 1500.0 < T_wall_final < 4000.0, (
        f"final T_wall {T_wall_final:.1f} K implausible (ref ~2800 K)."
    )
    # CMA model: s_dot = m_dot_char / rho_char.  At C_M=0.01 char flux is tiny.
    assert 1e-8 < s_final < 1e-3, (
        f"final recession {s_final * 1e6:.2f} µm implausible."
    )
    assert s_dot_final > 0.0, "recession rate non-positive at end of run."
