# SPDX-License-Identifier: MIT
"""V2 — Conduction with temperature-dependent properties.

Adds temperature-dependent thermal conductivity k(T) (and the virgin/char
blending machinery in :mod:`scam.physics.properties`) on top of the bare
conduction kernel verified in V1. cp is held constant so the reference is a
pure steady-state conductivity check.

Reference: the **Kirchhoff transform**. Define theta(T) = ∫_{T_back}^{T} k dT'.
At steady state with constant cross-section and no volumetric source, the
conduction equation d/dy(k dT/dy) = 0 makes theta(y) *linear* in y, regardless
of how nonlinear k(T) is:

    theta(T(y)) = theta(T_front) * (1 - y/L)

Inverting this for a conductivity that is linear in T, k(T) = a + b T, gives a
closed-form quadratic for the reference profile T(y). The steady surface flux
is q = theta(T_front) / L.

The discrete profile approaches this reference at first order under grid
refinement (same half-cell boundary treatment as V1).
"""
import numpy as np

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import make_linear_k_material, run_case


# Linear conductivity k(T) = a + b T, anchored at the table endpoints.
K0, K1 = 0.5, 2.5          # W/m/K at T0_K and T1_K
T0_K, T1_K = 300.0, 3000.0
RHO, CP = 180.0, 710.0
T_FRONT, T_BACK = 2500.0, 400.0
L = 0.05

_B = (K1 - K0) / (T1_K - T0_K)
_A = K0 - _B * T0_K


def _theta(T):
    """Kirchhoff variable theta(T) = ∫_{T_BACK}^{T} (a + b T') dT'."""
    return _A * (T - T_BACK) + 0.5 * _B * (T * T - T_BACK * T_BACK)


def _theta_front():
    return _theta(T_FRONT)


def _T_reference(y):
    """Invert theta(T(y)) = theta_front * (1 - y/L) for the linear-k profile."""
    target = _theta_front() * (1.0 - np.asarray(y) / L)
    # 0.5 b T^2 + a T + c = 0  with  c = -(a*T_back + 0.5 b T_back^2 + target)
    out = np.empty_like(target)
    for i, tg in enumerate(target):
        c = -(_A * T_BACK + 0.5 * _B * T_BACK * T_BACK + tg)
        disc = _A * _A - 4.0 * (0.5 * _B) * c
        out[i] = (-_A + np.sqrt(disc)) / _B
    return out


def _run_steady(n_nodes: int):
    mat = make_linear_k_material("VarK", K0, K1, T0_K, T1_K, RHO, CP)
    t_diff = L**2 / (K0 / (RHO * CP))   # slowest diffusion time (smallest k)
    stack = StackConfig(layers=[LayerConfig("VarK", L, n_nodes, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_FRONT,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    options = SolverOptions(
        t_end=40.0 * t_diff, dt_init=0.1, dt_max=t_diff / 20.0, dt_min=1e-3,
        dt_max_dT=1e4, output_dt=40.0 * t_diff,
    )
    return run_case(stack, {"VarK": mat}, surface_bc, back_bc, options,
                    initial_T=T_BACK).snapshots[-1]


def test_v2_kirchhoff_profile_converges():
    snap_c = _run_steady(51)
    snap_f = _run_steady(101)

    err_c = float(np.abs(snap_c.T - _T_reference(snap_c.mesh.y_nodes)).max())
    err_f = float(np.abs(snap_f.T - _T_reference(snap_f.mesh.y_nodes)).max())

    span = T_FRONT - T_BACK
    assert err_f / span < 0.01, (
        f"Kirchhoff profile: fine-grid max err {err_f:.2f} K "
        f"({err_f / span * 100:.2f}% of span) too large."
    )
    ratio = err_c / err_f
    assert 1.7 < ratio < 2.3, (
        f"Kirchhoff profile: error ratio {ratio:.2f} on grid halving is not "
        f"first-order (expected ~2.0); coarse={err_c:.2f}, fine={err_f:.2f} K."
    )


def test_v2_steady_surface_flux():
    snap = _run_steady(101)
    q_ref = _theta_front() / L
    rel_err = abs(snap.q_cond - q_ref) / q_ref
    assert rel_err < 0.02, (
        f"variable-k steady flux: q_cond={snap.q_cond:.1f} vs Kirchhoff "
        f"q={q_ref:.1f} W/m^2 ({rel_err * 100:.2f}% > 2%)."
    )
