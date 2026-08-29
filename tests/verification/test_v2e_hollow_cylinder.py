# SPDX-License-Identifier: MIT
"""V2e — Hollow cylinder, prescribed inner-wall temperature, adiabatic outer wall.

Geometry:  R1 = 5 mm (inner), R2 = 55 mm (outer), Scyl = 50 mm (wall thickness)
Material:  ρ = 1850 kg/m³, cp = 2000 J/kg/K, k = 30 W/m/K, α ≈ 8.108e-6 m²/s
BCs:       T(R1, t) = Tw = 3000 K (prescribed), dT/dr|_{R2} = 0 (adiabatic)
IC:        T(r, 0) = T0 = 0 K

Analytical reference: Green's function using Bessel eigenfunctions.

Eigenvalue equation:  J_0(β R1) Y_1(β R2) − Y_0(β R1) J_1(β R2) = 0

Eigenfunctions:  φ_m(r) = J_0(β_m r) Y_1(β_m R2) − Y_0(β_m r) J_1(β_m R2)
                 (satisfies φ_m(R1) = 0 and φ_m'(R2) = 0 by construction)

Full solution:
    T(r,t) = T_w + Σ_{m=1}^∞ C_m φ_m(r) exp(−α β_m² t)
    C_m = (−T_w / N_m) ∫_{R1}^{R2} r φ_m(r) dr,   N_m = ∫_{R1}^{R2} r φ_m²(r) dr

Note: the outer wall heats up over time (adiabatic BC + fixed inner T → steady state
T = T_w throughout).  The pass criterion only checks interior profile accuracy, not
whether the outer wall stays cold.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import make_inert_material, run_case


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO    = 1850.0
CP     = 2000.0
K      = 30.0
ALPHA  = K / (RHO * CP)    # ≈ 8.108e-6 m²/s

T_WALL = 3000.0             # [K] prescribed inner-wall temperature
T_INIT =    0.0             # [K] initial temperature

R1     = 0.005              # [m] inner radius
R2     = 0.055              # [m] outer radius
SCYL   = R2 - R1            # [m] wall thickness = 50 mm

# Coarser grid for fast test execution (dx = 0.5 mm)
N_NODES = 101
DT_MAX  = 0.1               # [s]

_SNAP_THRESHOLDS = {30.0: 4.0, 60.0: 3.0, 100.0: 2.0}
T_SNAPS = sorted(_SNAP_THRESHOLDS)

# Number of eigenvalues — enough for good convergence at these times
_N_EIGEN = 200


# ---------------------------------------------------------------------------
# Green's function analytical solution
# ---------------------------------------------------------------------------

def _f_eigen(beta: float) -> float:
    return j0(beta * R1) * y1(beta * R2) - y0(beta * R1) * j1(beta * R2)


def _phi(beta: float, r):
    r = np.asarray(r, dtype=float)
    return j0(beta * r) * y1(beta * R2) - y0(beta * r) * j1(beta * R2)


def _find_eigenvalues(n: int) -> np.ndarray:
    """Find the first n positive eigenvalues β_m."""
    spacing = np.pi / SCYL
    beta_scan = np.linspace(1.0, spacing * (n + 20), 10 * (n + 20) * 20)
    fv = np.vectorize(_f_eigen)(beta_scan)
    betas: list[float] = []
    for i in range(len(fv) - 1):
        if fv[i] * fv[i + 1] < 0:
            try:
                bm = brentq(_f_eigen, beta_scan[i], beta_scan[i + 1], xtol=1e-12)
                betas.append(bm)
                if len(betas) == n:
                    break
            except ValueError:
                pass
    return np.array(betas)


def _build_greens_series(n: int = _N_EIGEN):
    """Return (betas, coefficients) arrays of length n."""
    betas = _find_eigenvalues(n)
    coeffs = np.empty(len(betas))
    for i, bm in enumerate(betas):
        norm, _ = quad(lambda r: r * _phi(bm, r) ** 2, R1, R2, limit=200)
        proj, _ = quad(lambda r: r * _phi(bm, r),      R1, R2, limit=200)
        coeffs[i] = (-T_WALL / norm) * proj
    return betas, coeffs


def greens_T(r, t: float, betas, coeffs) -> np.ndarray:
    """Evaluate the Green's function series at positions r and time t."""
    r = np.asarray(r, dtype=float)
    T = np.full_like(r, T_WALL)
    for bm, cm in zip(betas, coeffs):
        exp_factor = np.exp(-ALPHA * bm ** 2 * t)
        if exp_factor < 1e-15:
            break
        T += cm * _phi(bm, r) * exp_factor
    return T


# ---------------------------------------------------------------------------
# Module-scoped fixtures (build Green's series and run SCAM once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def greens_series():
    return _build_greens_series(_N_EIGEN)


@pytest.fixture(scope="module")
def hollow_cylinder_snapshots():
    mat = make_inert_material("V2e", k=K, rho=RHO, cp=CP)
    geom = GeometryConfig(geometry_type=GeometryType.HOLLOW_CYLINDER, r_inner=R1)
    stack = StackConfig(layers=[LayerConfig("V2e", SCYL, N_NODES, 4)])
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
    results = run_case(stack, {"V2e": mat}, surface_bc, back_bc, options,
                       initial_T=T_INIT, geom=geom)

    saved_times = np.array([s.time for s in results.snapshots])
    return {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }


# ---------------------------------------------------------------------------
# 1. Surface (inner wall) pinned to the prescribed temperature
# ---------------------------------------------------------------------------

def test_v2e_surface_pinned(hollow_cylinder_snapshots):
    """Front face (r=R1) must stay pinned to T_wall within 0.01 K."""
    for t_req, snap in hollow_cylinder_snapshots.items():
        err = abs(snap.T[0] - T_WALL)
        assert err < 0.01, (
            f"t={t_req} s: surface node {snap.T[0]:.4f} K deviates from "
            f"T_wall={T_WALL} K by {err:.4f} K (threshold 0.01 K)"
        )


# ---------------------------------------------------------------------------
# 2. Temperature profile matches the Green's function at each snapshot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_req", T_SNAPS)
def test_v2e_greens_profile(hollow_cylinder_snapshots, greens_series, t_req):
    """Numerical profile agrees with the Green's function analytical solution.

    Only the 5%–95% band of the current temperature span is compared: the
    surface pin (exact by BC) and the cold tail (where relative error is
    dominated by floating-point noise) are excluded.
    Error threshold decreases with time as the front broadens onto more nodes.
    """
    snap = hollow_cylinder_snapshots[t_req]
    r_num = R1 + snap.mesh.y_nodes
    T_num = snap.T
    betas, coeffs = greens_series

    T_ana  = greens_T(r_num, snap.time, betas, coeffs)
    dT_ref = T_ana - T_INIT

    T_span_now = float(T_ana.max() - T_INIT)
    mask = (dT_ref > 0.05 * T_span_now) & (dT_ref < 0.95 * T_span_now)
    if mask.sum() == 0:
        mask = dT_ref > 10.0

    assert mask.sum() > 0, (
        f"t={t_req} s: no nodes in comparison band; front may not have penetrated"
    )

    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
    max_err_pct = float(rel_err.max()) * 100.0
    thr = _SNAP_THRESHOLDS[t_req]
    assert max_err_pct < thr, (
        f"t={t_req} s: max relative error {max_err_pct:.2f}% > {thr}% "
        f"(HOLLOW_CYLINDER, N={N_NODES}, dx={SCYL/(N_NODES-1)*1e3:.2f} mm)"
    )
