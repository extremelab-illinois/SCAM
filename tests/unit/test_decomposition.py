# SPDX-License-Identifier: MIT
"""Unit tests for physics/decomposition.py.

Tests exact Arrhenius integration against scipy ODE solver and
validates the irreversibility constraint.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scam.config.material import ComponentCard, MaterialCard
from scam.core.constants import R_UNIVERSAL
from scam.physics.decomposition import arrhenius_k, _update_rho_i


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_comp(A: float = 1.2e4, E: float = 8.7e4,
               m: float = 3.0, rho_0: float = 60.0,
               rho_r: float = 0.0) -> ComponentCard:
    return ComponentCard(name="test", rho_0=rho_0, rho_r=rho_r,
                         A_rate=A, E_act=E, m_exp=m, h_decomp=0.0)


def _scipy_integrate(comp: ComponentCard, rho_init: float, T: float,
                     dt: float) -> float:
    """Integrate Arrhenius ODE with scipy for reference."""
    k = arrhenius_k(comp, T)

    def ode(t, y):
        rho = y[0]
        if rho <= comp.rho_r:
            return [0.0]
        xi = (rho - comp.rho_r) / (comp.rho_0 - comp.rho_r)
        xi = max(xi, 0.0)
        return [-k * (comp.rho_0 - comp.rho_r) * xi ** comp.m_exp]

    sol = solve_ivp(ode, [0, dt], [rho_init], method="RK45",
                    rtol=1e-10, atol=1e-12, dense_output=False)
    return float(sol.y[0, -1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_arrhenius_k_positive():
    comp = _make_comp()
    k = arrhenius_k(comp, 1000.0)
    assert k > 0.0


def test_arrhenius_k_increases_with_T():
    comp = _make_comp()
    k1 = arrhenius_k(comp, 500.0)
    k2 = arrhenius_k(comp, 1500.0)
    assert k2 > k1


def test_no_decomposition_below_rho_r():
    """Density should not decrease below residual density."""
    comp = _make_comp(rho_0=60.0, rho_r=10.0)
    T = 2000.0
    rho_init = 10.0  # already at rho_r
    rho_new = _update_rho_i(comp, rho_init, T, dt=100.0)
    assert rho_new >= comp.rho_r - 1e-12


def test_irreversibility():
    """Density must never increase (irreversibility)."""
    comp = _make_comp()
    T = 500.0   # low T → very slow decomposition
    rho_init = 50.0
    rho_new = _update_rho_i(comp, rho_init, T, dt=1e-6)
    assert rho_new <= rho_init + 1e-12


def test_m1_branch_exponential():
    """m=1 branch: rho decays exponentially → compare with direct formula."""
    comp = _make_comp(m=1.0, rho_0=60.0, rho_r=0.0)
    T = 1500.0
    dt = 0.5
    k = arrhenius_k(comp, T)
    rho_init = 60.0
    rho_exact = comp.rho_r + (rho_init - comp.rho_r) * np.exp(-k * dt)
    rho_scam = _update_rho_i(comp, rho_init, T, dt)
    assert abs(rho_scam - rho_exact) < 1e-10 * rho_exact


def test_m3_branch_matches_scipy():
    """m=3 branch: compare SCAM exact integration to scipy RK45."""
    comp = _make_comp(m=3.0, rho_0=60.0, rho_r=0.0,
                      A=1.2e4, E=8.7e4)
    T = 900.0
    dt = 1.0
    rho_init = 50.0

    rho_scam = _update_rho_i(comp, rho_init, T, dt)
    rho_ref  = _scipy_integrate(comp, rho_init, T, dt)

    assert abs(rho_scam - rho_ref) < 1e-4 * rho_ref, (
        f"SCAM={rho_scam:.8f}, ref={rho_ref:.8f}"
    )


def test_m3_branch_fast_decomp():
    """High-T: nearly complete decomposition, verify floor at rho_r."""
    comp = _make_comp(m=3.0, rho_0=60.0, rho_r=5.0, A=1e12, E=5e4)
    T = 2000.0
    dt = 10.0
    rho_new = _update_rho_i(comp, 60.0, T, dt)
    assert rho_new >= comp.rho_r - 1e-10
    assert rho_new < 10.0  # should be nearly fully charred
