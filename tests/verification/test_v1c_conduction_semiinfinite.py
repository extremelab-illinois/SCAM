# SPDX-License-Identifier: MIT
"""V1c — Semi-infinite slab with prescribed wall temperature (constant properties).

Analytical reference:
    T(y, t) = T_init + (T_wall − T_init) · erfc(y / (2 √(α t)))

Material properties (constant):
    ρ = 1850 kg/m³,  cp = 2000 J/kg/K,  k = 30 W/m/K
    α = k / (ρ cp) ≈ 8.108 × 10⁻⁶ m²/s

Slab thickness L = 0.5 m.  At t = 100 s the back-face Fourier argument is
    ξ = L / (2 √(α t)) ≈ 8.8 → erfc(8.8) ≈ 0
so the back face stays at T_init throughout (semi-infinite behaviour).

Grid: N = 201 nodes → dx = 2.5 mm.  The thermal penetration depth δ = 2√(α t)
is 18 mm at t = 10 s (≈ 7 nodes/front) and 57 mm at t = 100 s (≈ 23 nodes/front).
Because backward-Euler error scales inversely with nodes per front, we test only
at t ≥ 30 s (δ/dx ≥ 12.5) where the error is below 7 %; the early-time behaviour
is shown in the companion example script but not asserted here.
"""
import numpy as np
import pytest

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import (
    fourier_slab_adiabatic_T,
    make_inert_material,
    run_case,
    semi_infinite_temp_T,
)


# ---------------------------------------------------------------------------
# Material constants for this verification
# ---------------------------------------------------------------------------
RHO    = 1850.0
CP     = 2000.0
K      = 30.0
ALPHA  = K / (RHO * CP)    # ≈ 8.108e-6 m²/s
T_WALL = 1000.0             # [K] prescribed front face
T_INIT =  300.0             # [K] initial / back face

THICKNESS = 0.50            # [m] — semi-infinite up to t = 100 s
N_NODES   = 201             # dx = 2.5 mm (same resolution as companion example)
DT_MAX    = 0.50            # [s]

# Snapshot times and their per-snapshot error thresholds.
# Threshold tightens with time as more nodes resolve the advancing front:
#   t = 30 s: δ = 31 mm → 12.5 nodes/front → ≤ 7 %
#   t = 60 s: δ = 44 mm → 17.7 nodes/front → ≤ 5 %
#   t =100 s: δ = 57 mm → 22.8 nodes/front → ≤ 4 %
_SNAP_THRESHOLDS = {30.0: 7.0, 60.0: 5.0, 100.0: 4.0}
T_SNAPS = sorted(_SNAP_THRESHOLDS)


# ---------------------------------------------------------------------------
# Shared fixture: run once for all snapshot checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def semiinfinite_snapshots():
    """Run the V1c case once and return a dict {t_snap: snapshot}."""
    mat = make_inert_material("V1c", k=K, rho=RHO, cp=CP)
    stack = StackConfig(layers=[LayerConfig("V1c", THICKNESS, N_NODES, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_WALL,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=max(T_SNAPS),
        dt_init=0.02,
        dt_max=DT_MAX,
        dt_min=1e-4,
        dt_max_dT=1e4,
        output_dt=min(T_SNAPS) / 2.0,
    )
    results = run_case(stack, {"V1c": mat}, surface_bc, back_bc, options,
                       initial_T=T_INIT)

    saved_times = np.array([s.time for s in results.snapshots])
    return {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }


# ---------------------------------------------------------------------------
# 1. Surface node pinned to the prescribed temperature
# ---------------------------------------------------------------------------

def test_v1c_surface_pinned(semiinfinite_snapshots):
    """Front face must stay pinned to T_wall within 0.01 K at every snapshot."""
    for t_req, snap in semiinfinite_snapshots.items():
        err = abs(snap.T[0] - T_WALL)
        assert err < 0.01, (
            f"t={t_req} s: surface node {snap.T[0]:.4f} K deviates from "
            f"T_wall={T_WALL} K by {err:.4f} K (threshold 0.01 K)"
        )


# ---------------------------------------------------------------------------
# 2. Back face temperature unchanged (semi-infinite criterion)
# ---------------------------------------------------------------------------

def test_v1c_back_face_unaffected(semiinfinite_snapshots):
    """Back face must remain within 0.01 K of T_init throughout the run.

    With L = 0.5 m and α ≈ 8.1e-6 m²/s the Fourier argument at the back face
    is ξ = L / (2√(α t)) ≥ 8.8 for t ≤ 100 s, giving
    erfc(ξ) × (T_wall − T_init) < 10⁻¹⁰ K — far below the 0.01 K threshold.
    """
    for t_req, snap in semiinfinite_snapshots.items():
        back_dT = abs(snap.T[-1] - T_INIT)
        assert back_dT < 0.01, (
            f"t={t_req} s: back face changed by {back_dT:.4f} K "
            f"(threshold 0.01 K — slab may not be thick enough)"
        )


# ---------------------------------------------------------------------------
# 3. Temperature profile matches the erfc solution at each snapshot
# ---------------------------------------------------------------------------

def _check_profile(T_num, T_ana, y, t_req, label, thr):
    """Shared check: max relative error in heated zone (dT_ana > 50 K) < thr %."""
    dT_ana = T_ana - T_INIT
    mask = dT_ana > 50.0
    assert mask.sum() > 0, (
        f"t={t_req} s [{label}]: thermal front has not penetrated; "
        f"no nodes with dT_ana > 50 K"
    )
    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ana[mask])
    max_err_pct = float(rel_err.max()) * 100.0
    nodes_per_front = 2.0 * np.sqrt(ALPHA * t_req) / (THICKNESS / (N_NODES - 1))
    assert max_err_pct < thr, (
        f"t={t_req} s [{label}]: max relative error {max_err_pct:.2f}% > {thr}% "
        f"(dx = {THICKNESS/(N_NODES-1)*1000:.1f} mm, "
        f"nodes/front ≈ {nodes_per_front:.1f})"
    )


@pytest.mark.parametrize("t_req", T_SNAPS)
def test_v1c_erfc_profile(semiinfinite_snapshots, t_req):
    """Numerical profile agrees with the erfc (semi-infinite) analytical solution.

    Only nodes where dT_ana > 50 K are compared: the steep erfc tip (where
    spatial resolution caps accuracy) and the unheated tail (where relative
    error is dominated by floating-point noise) are excluded.
    Error threshold decreases with time as the front widens onto more nodes.
    """
    snap = semiinfinite_snapshots[t_req]
    y, T_num = snap.mesh.y_nodes, snap.T
    T_ana = semi_infinite_temp_T(y, snap.time, T_WALL, T_INIT, ALPHA)
    _check_profile(T_num, T_ana, y, t_req, "erfc", _SNAP_THRESHOLDS[t_req])


@pytest.mark.parametrize("t_req", T_SNAPS)
def test_v1c_fourier_profile(semiinfinite_snapshots, t_req):
    """Numerical profile agrees with the Fourier-series (finite-slab) analytical solution.

    The Fourier series is exact for the actual BC (prescribed T_wall at y=0,
    adiabatic at y=L).  It converges to the erfc solution as L → ∞; at
    Fo = αt/L² ≈ 3.2×10⁻³ (t=100 s, L=0.5 m) both solutions agree to < 10⁻¹⁰ K
    in the heated region.  The same error thresholds as the erfc test apply because
    the two analytical solutions are numerically indistinguishable here.
    """
    snap = semiinfinite_snapshots[t_req]
    y, T_num = snap.mesh.y_nodes, snap.T
    T_ana = fourier_slab_adiabatic_T(y, snap.time, T_WALL, T_INIT, ALPHA, THICKNESS)
    _check_profile(T_num, T_ana, y, t_req, "Fourier", _SNAP_THRESHOLDS[t_req])
