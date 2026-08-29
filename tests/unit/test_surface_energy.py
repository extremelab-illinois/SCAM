# SPDX-License-Identifier: MIT
"""Unit tests for physics/surface_energy.py and physics/blowing.py."""
import numpy as np
import pytest

from scam.core.constants import SIGMA_SB
from scam.config.boundary import SurfaceBCConfig
from scam.config.material import MaterialCard, KineticAblationCard
from scam.physics.surface_energy import (
    radiation_out, radiation_in, convective_flux, seb_residual,
)
from scam.physics.blowing import blowing_corrected_alpha
from scam.physics.stanton import (
    eckert_reference_enthalpy,
    kays_blowing_factor,
    laminar_wall_factor,
    solve_kays_blowing_factor,
    turbulent_eckert_wall_factor,
)


def test_radiation_out_sigma_T4():
    eps = 0.85
    T = 2000.0
    q = radiation_out(T, eps, view_factor=1.0)
    assert q == pytest.approx(eps * SIGMA_SB * T**4, rel=1e-10)


def test_radiation_out_zero_emissivity():
    q = radiation_out(2000.0, emissivity=0.0, view_factor=1.0)
    assert q == pytest.approx(0.0)


def test_radiation_in_zero_source():
    q = radiation_in(T_rad=0.0, emissivity=0.85, view_factor=1.0)
    assert q == pytest.approx(0.0)


def test_radiation_in_positive():
    q = radiation_in(T_rad=1000.0, emissivity=0.85, view_factor=1.0)
    assert q == pytest.approx(0.85 * SIGMA_SB * 1000.0**4, rel=1e-10)


def test_convective_flux_positive_heating():
    q = convective_flux(alpha_conv=5000.0, T_aw=8000.0, T_w=300.0)
    assert q > 0.0
    assert q == pytest.approx(5000.0 * (8000.0 - 300.0))


def test_convective_flux_zero_at_aw():
    q = convective_flux(alpha_conv=5000.0, T_aw=1000.0, T_w=1000.0)
    assert q == pytest.approx(0.0)


def test_blowing_no_blowing():
    alpha = blowing_corrected_alpha(
        alpha_conv=5000.0, m_dot_char=0.0, m_dot_pyro=0.0,
        rho_e_u_e=0.1, C_M=0.01, lambda_blowing=0.5,
    )
    assert alpha == pytest.approx(5000.0)


def test_blowing_reduces_alpha():
    alpha_no_blow = 5000.0
    alpha = blowing_corrected_alpha(
        alpha_conv=alpha_no_blow, m_dot_char=0.01, m_dot_pyro=0.005,
        rho_e_u_e=0.1, C_M=0.01, lambda_blowing=0.5,
    )
    assert alpha < alpha_no_blow


def test_blowing_zero_rho_e_ue():
    # With rho_e_u_e = 0, no blowing correction
    alpha = blowing_corrected_alpha(
        alpha_conv=5000.0, m_dot_char=0.01, m_dot_pyro=0.005,
        rho_e_u_e=0.0, C_M=0.01, lambda_blowing=0.5,
    )
    assert alpha == pytest.approx(5000.0)


def test_kays_blowing_factor_limit_and_value():
    assert kays_blowing_factor(0.0) == pytest.approx(1.0)
    assert kays_blowing_factor(0.2) == pytest.approx(
        0.2 / np.expm1(0.2)
    )


def test_kays_implicit_char_flux_is_self_consistent():
    omega = solve_kays_blowing_factor(
        m_dot_char_unblown=0.2,
        m_dot_pyro=0.0,
        rho_ue_ch0=1.0,
        lambda_blowing=0.4,
    )
    phi = 2.0 * 0.4 * (omega * 0.2)
    assert omega == pytest.approx(kays_blowing_factor(phi), rel=1e-11)


def test_amar_wall_correction_helpers():
    href = eckert_reference_enthalpy(1.0e6, 2.0e6, 1000.0, 0.9)
    assert href == pytest.approx(1.599e6)
    assert turbulent_eckert_wall_factor(2.0, 3.0, 1.0, 1.5) == pytest.approx(2.0)
    assert laminar_wall_factor(2.0, 3.0, 1.0, 1.5) == pytest.approx(4.0**0.1)


def test_surface_energy_applies_wall_and_kays_corrections():
    class Backend:
        def lookup(self, *_args, **_kwargs):
            return 0.2, 0.0

    mat = MaterialCard(
        name="test",
        rho_virgin=1000.0,
        rho_char=500.0,
        gamma_resin=0.0,
        emissivity=0.0,
    )
    bc = SurfaceBCConfig(
        rhoUeCH=1.0,
        h_r=1.0e6,
        rho_e_u_e=1.0,
        C_M=1.0,
        lambda_blowing=0.4,
        blowing_model="kays",
        stanton_wall_correction=lambda _T, _t, _hw: 0.5,
        T_rad_in=0.0,
    )
    residual, _q_cond, m_dot_char, _h_wall = seb_residual(
        T_w=1000.0,
        alpha_F=0.0,
        beta_F=0.0,
        bc=bc,
        time=0.0,
        m_dot_pyro=0.0,
        mat_surface=mat,
        b_prime_table=Backend(),
    )
    omega = solve_kays_blowing_factor(0.1, 0.0, 1.0, 0.4)
    assert m_dot_char == pytest.approx(0.1 * omega)
    assert residual == pytest.approx(0.5 * omega * 1.0e6)


def test_surface_energy_can_apply_wall_correction_to_mass_only():
    class Backend:
        def lookup(self, *_args, **_kwargs):
            return 0.2, 0.0

    mat = MaterialCard(
        name="test",
        rho_virgin=1000.0,
        rho_char=500.0,
        gamma_resin=0.0,
        emissivity=0.0,
    )
    bc = SurfaceBCConfig(
        rhoUeCH=1.0,
        h_r=1.0e6,
        rho_e_u_e=1.0,
        C_M=1.0,
        lambda_blowing=0.0,
        stanton_wall_correction=lambda _T, _t, _hw: 0.5,
        apply_wall_correction_to_heat=False,
        T_rad_in=0.0,
    )
    residual, _q_cond, m_dot_char, _h_wall = seb_residual(
        T_w=1000.0,
        alpha_F=0.0,
        beta_F=0.0,
        bc=bc,
        time=0.0,
        m_dot_pyro=0.0,
        mat_surface=mat,
        b_prime_table=Backend(),
    )
    assert m_dot_char == pytest.approx(0.1)
    assert residual == pytest.approx(1.0e6)


def test_lees_blowing_warm_start_reduces_lookup_count_with_same_solution():
    class Backend:
        def __init__(self):
            self.calls = []

        def lookup(self, T_w, _p_e, B_g, _Z_C_pyro=None):
            self.calls.append((float(T_w), float(B_g)))
            return 0.18 + 0.04 * B_g, 2.0e5 + 50.0 * T_w

    mat = MaterialCard(
        name="test",
        rho_virgin=1000.0,
        rho_char=500.0,
        gamma_resin=0.0,
        emissivity=0.0,
    )
    bc = SurfaceBCConfig(
        rhoUeCH=1.0,
        h_r=1.0e6,
        rho_e_u_e=1.0,
        C_M=1.0,
        lambda_blowing=0.5,
        blowing_model="lees",
        T_rad_in=0.0,
    )
    backend = Backend()

    cold = seb_residual(
        T_w=1200.0,
        alpha_F=0.0,
        beta_F=0.0,
        bc=bc,
        time=0.0,
        m_dot_pyro=0.08,
        mat_surface=mat,
        b_prime_table=backend,
    )
    cold_calls = len(backend.calls)
    cached_factor = backend._scam_lees_blow_factor

    backend.calls.clear()
    warm = seb_residual(
        T_w=1200.0,
        alpha_F=0.0,
        beta_F=0.0,
        bc=bc,
        time=0.0,
        m_dot_pyro=0.08,
        mat_surface=mat,
        b_prime_table=backend,
    )
    warm_calls = len(backend.calls)

    assert 0.0 < cached_factor <= 1.0
    assert warm_calls < cold_calls
    np.testing.assert_allclose(warm, cold, rtol=1e-10, atol=1e-8)


def test_bprime_backend_chemistry_off_uses_temperature_convection():
    class Backend:
        def lookup(self, *args, **kwargs):
            raise AssertionError("B' lookup should be skipped when rho_e_u_e=0")

    mat = MaterialCard(
        name="test",
        rho_virgin=1000.0,
        rho_char=500.0,
        gamma_resin=0.0,
        emissivity=0.0,
    )
    bc = SurfaceBCConfig(
        alpha_conv=10.0,
        T_aw=500.0,
        rhoUeCH=0.003,
        h_r=1.0e9,
        rho_e_u_e=0.0,
        C_M=1.0,
        T_rad_in=0.0,
    )

    residual, q_cond, m_dot_char, _h_wall = seb_residual(
        T_w=400.0,
        alpha_F=0.0,
        beta_F=0.0,
        bc=bc,
        time=0.0,
        m_dot_pyro=0.0,
        mat_surface=mat,
        b_prime_table=Backend(),
    )

    assert residual == pytest.approx(10.0 * (500.0 - 400.0))
    assert q_cond == pytest.approx(0.0)
    assert m_dot_char == pytest.approx(0.0)


def _kinetic_teflon_material():
    from scam.physics.kinetic_ablation import m_dot_kemp, h_ablation_total
    return MaterialCard(
        name="test_kinetic_ptfe",
        rho_virgin=2310.0,
        rho_char=2310.0,
        gamma_resin=1.0,
        decomposing=False,
        emissivity=0.85,
        k_virgin_table=np.array([[300.0, 0.27], [1200.0, 0.70]]),
        k_char_table=np.array([[300.0, 0.27], [1200.0, 0.70]]),
        kinetic_ablation=KineticAblationCard(
            B=3.0e19,
            E_a=347272.0,
            h_ablation_total_coeffs=(1468290.34, 364.77, 0.7028),
            h_ablation_reaction_coeffs=(1773661.27, -279.02),
        ),
    ), m_dot_kemp, h_ablation_total


def test_kinetic_ablation_m_dot_char_matches_closed_form():
    """seb_residual's m_dot_char must equal the standalone Kemp Eq. 12
    evaluation at the same T_w — this is independent of the SEB's energy
    accounting (PRESCRIBED_TEMP-style single evaluation, no Newton solve).
    """
    from scam.physics.properties import thermal_conductivity

    mat, m_dot_kemp, h_ablation_total = _kinetic_teflon_material()
    bc = SurfaceBCConfig(alpha_conv=200.0, T_aw=2000.0, blowing_model="rational")

    T_w = 1000.0
    _, _, m_dot_char, _ = seb_residual(
        T_w=T_w, alpha_F=0.0, beta_F=0.0, bc=bc, time=0.0, m_dot_pyro=0.0,
        mat_surface=mat, b_prime_table=None,
    )

    ka = mat.kinetic_ablation
    k_w = thermal_conductivity(mat, T_w, mat.rho_char)
    h_tot = h_ablation_total(T_w, ka.h_ablation_total_coeffs)
    expected = m_dot_kemp(T_w, mat.rho_char, k_w, ka.B, ka.E_a, h_tot)
    assert m_dot_char == pytest.approx(expected, rel=1e-12)


def test_kinetic_ablation_unaffected_by_blowing_but_residual_is():
    """Design decision: Kemp's mass flux is reaction-rate-limited, not
    mass-transfer-limited, so it must NOT be throttled by the Reynolds-
    analogy blow_factor — unlike the B'-table (CMA) path. Blowing must still
    affect q_conv/the residual (real transpiration cooling).
    """
    mat, _, _ = _kinetic_teflon_material()
    bc = SurfaceBCConfig(alpha_conv=200.0, T_aw=2000.0, blowing_model="rational")

    r_no_blow = seb_residual(
        T_w=1000.0, alpha_F=0.0, beta_F=0.0, bc=bc, time=0.0, m_dot_pyro=0.0,
        mat_surface=mat, b_prime_table=None,
    )
    r_heavy_blow = seb_residual(
        T_w=1000.0, alpha_F=0.0, beta_F=0.0, bc=bc, time=0.0, m_dot_pyro=5.0,
        mat_surface=mat, b_prime_table=None,
    )

    assert r_no_blow[2] == pytest.approx(r_heavy_blow[2], rel=1e-12)  # m_dot_char
    assert r_no_blow[0] != pytest.approx(r_heavy_blow[0])             # residual differs


def test_kinetic_ablation_precedence_over_bprime_table():
    """If a card somehow carries both (shouldn't happen post material_loader
    validation, but seb_residual itself must still be safe), kinetic_ablation
    takes unconditional precedence.
    """
    class Backend:
        def lookup(self, *args, **kwargs):
            raise AssertionError("B' lookup should be skipped when kinetic_ablation is set")

    mat, _, _ = _kinetic_teflon_material()
    bc = SurfaceBCConfig(
        alpha_conv=0.0, rho_e_u_e=1.0, C_M=0.5, rhoUeCH=0.5, h_r=1.0e7,
    )
    # Should not raise despite Backend.lookup asserting if called.
    _, _, m_dot_char, _ = seb_residual(
        T_w=1000.0, alpha_F=0.0, beta_F=0.0, bc=bc, time=0.0, m_dot_pyro=0.0,
        mat_surface=mat, b_prime_table=Backend(),
    )
    assert m_dot_char > 0.0
