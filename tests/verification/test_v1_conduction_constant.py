# SPDX-License-Identifier: MIT
"""V1 — Conduction with constant properties (vs analytical).

Lowest rung of the verification ladder: nothing but the bare heat equation is
active (no decomposition, no recession, no surface energy balance). This
isolates FVM assembly + Thomas solve + time integration. Three closed-form
references are used:

    1. semi-infinite slab, prescribed surface flux   → erfc solution
    2. semi-infinite slab, prescribed surface temp    → erfc solution
    3. finite slab, both ends fixed, steady state      → linear profile

A failure here means the conduction kernel itself is wrong; every higher rung
builds on it.
"""
import numpy as np

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import (
    finite_slab_steady_linear, make_inert_material, run_case,
    semi_infinite_flux_T, semi_infinite_temp_T,
)


K_CONST = 1.0
RHO = 180.0
CP_CONST = 710.0
ALPHA = K_CONST / (RHO * CP_CONST)
T_INIT = 300.0


# ---------------------------------------------------------------------------
# 1. Semi-infinite slab, prescribed surface flux
# ---------------------------------------------------------------------------

def test_v1_semi_infinite_prescribed_flux():
    Q0 = 50_000.0
    t_end = 20.0
    thickness = 0.10  # thick enough that the back face stays near T_INIT

    mat = make_inert_material("Inert", k=K_CONST, rho=RHO, cp=CP_CONST)
    stack = StackConfig(layers=[LayerConfig("Inert", thickness, 101, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_FLUX, q_prescribed=lambda t: Q0,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=t_end, dt_init=0.05, dt_max=1.0, dt_min=1e-4,
        dt_max_dT=200.0, output_dt=t_end,
    )

    results = run_case(stack, {"Inert": mat}, surface_bc, back_bc, options,
                       initial_T=T_INIT)
    snap = results.snapshots[-1]
    y, T_num = snap.mesh.y_nodes, snap.T
    T_ana = semi_infinite_flux_T(y, t_end, Q0, K_CONST, ALPHA, T_INIT)

    # Compare only the well-resolved near-surface region (dT > 50 K). Deep in
    # the slab the backward-Euler scheme is numerically diffusive at large
    # Fourier number; that region is not the meaningful comparison zone.
    dT_ana = T_ana - T_INIT
    mask = dT_ana > 50.0
    assert mask.sum() > 0, "thermal front did not penetrate; nothing to compare"

    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ana[mask])
    max_err_pct = float(rel_err.max()) * 100.0
    assert max_err_pct < 4.0, (
        f"prescribed-flux conduction: max rel err {max_err_pct:.2f}% > 4% "
        f"(backward-Euler at Fo~8 is expected ~2-3%)."
    )


# ---------------------------------------------------------------------------
# 2. Semi-infinite slab, prescribed surface temperature
# ---------------------------------------------------------------------------

def test_v1_semi_infinite_prescribed_temp():
    T_wall = 1500.0
    t_end = 20.0
    thickness = 0.10

    mat = make_inert_material("Inert", k=K_CONST, rho=RHO, cp=CP_CONST)
    stack = StackConfig(layers=[LayerConfig("Inert", thickness, 101, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_wall,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=t_end, dt_init=0.02, dt_max=0.1, dt_min=1e-4,
        dt_max_dT=1e4, output_dt=t_end,
    )

    results = run_case(stack, {"Inert": mat}, surface_bc, back_bc, options,
                       initial_T=T_INIT)
    snap = results.snapshots[-1]
    y, T_num = snap.mesh.y_nodes, snap.T

    # Surface node must be pinned to the prescribed temperature.
    assert abs(T_num[0] - T_wall) < 0.01, (
        f"surface not pinned: T[0]={T_num[0]:.3f} K vs prescribed {T_wall} K"
    )

    T_ana = semi_infinite_temp_T(y, t_end, T_wall, T_INIT, ALPHA)
    dT_ana = T_ana - T_INIT
    mask = dT_ana > 50.0
    assert mask.sum() > 0

    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ana[mask])
    max_err_pct = float(rel_err.max()) * 100.0
    # The step-temperature BC creates the steepest near-surface gradient of the
    # three sub-cases, so the first interior node carries the largest
    # discretization error; 5% bounds it on a uniform 1 mm grid.
    assert max_err_pct < 5.0, (
        f"prescribed-temp conduction: max rel err {max_err_pct:.2f}% > 5%."
    )


# ---------------------------------------------------------------------------
# 3. Finite slab, both ends fixed, steady state
# ---------------------------------------------------------------------------

def _finite_slab_steady_max_err(n_nodes: int) -> float:
    """Run the fixed-both-ends slab to steady state; return max |T - linear|."""
    T_front, T_back, thickness = 1500.0, 400.0, 0.05
    t_diff = thickness**2 / ALPHA

    mat = make_inert_material("Inert", k=K_CONST, rho=RHO, cp=CP_CONST)
    stack = StackConfig(layers=[LayerConfig("Inert", thickness, n_nodes, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_front,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_back)
    options = SolverOptions(
        t_end=40.0 * t_diff, dt_init=0.1, dt_max=t_diff / 20.0, dt_min=1e-3,
        dt_max_dT=1e4, output_dt=40.0 * t_diff,
    )
    results = run_case(stack, {"Inert": mat}, surface_bc, back_bc, options,
                       initial_T=T_back)
    snap = results.snapshots[-1]
    T_ref = finite_slab_steady_linear(snap.mesh.y_nodes, thickness, T_front, T_back)
    return float(np.abs(snap.T - T_ref).max())


def test_v1_finite_slab_steady_state_converges():
    # The steady FVM solution approaches the exact linear profile from above;
    # the residual lives at the boundary-adjacent nodes (the surface/back are
    # half-cells located ON the boundary, a first-order-accurate treatment).
    # Verify (a) the error is small on a fine grid and (b) it halves under grid
    # refinement, i.e. the scheme is consistent and converges to the analytical
    # solution. A constant or growing error would indicate a real assembly bug.
    err_coarse = _finite_slab_steady_max_err(51)
    err_fine = _finite_slab_steady_max_err(101)

    assert err_fine < 3.0, (
        f"finite-slab steady profile: fine-grid max err {err_fine:.3f} K too large."
    )
    ratio = err_coarse / err_fine
    assert 1.7 < ratio < 2.3, (
        f"finite-slab steady profile: error ratio {ratio:.2f} on grid halving is "
        f"not first-order (expected ~2.0); coarse={err_coarse:.3f}, fine={err_fine:.3f}."
    )
