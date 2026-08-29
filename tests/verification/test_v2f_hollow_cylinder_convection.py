# SPDX-License-Identifier: MIT
"""V2f — Hollow cylinder with convective heating at the inner wall, adiabatic outer.

Geometry:  R1 = 5 mm, R2 = 55 mm, Scyl = 50 mm
Material:  ρ = 1850 kg/m³, cp = 2000 J/kg/K, k = 30 W/m/K, α ≈ 8.108e-6 m²/s
BCs:       −k ∂T/∂r|_{R1} = h_w(T_env − T_w)  [convection, h_w=3000 W/m²/K]
           ∂T/∂r|_{R2} = 0  [adiabatic outer]
IC:        T(r, 0) = T0 = 0 K

Analytical reference (Green's function):
    T(r,t) = T_env + Σ_{m=1}^∞ C_m φ_m(r) exp(−α β_m² t)

Eigenfunctions:  φ_m(r) = J_0(β_m r)Y_1(β_m R2) − Y_0(β_m r)J_1(β_m R2)
                 (φ_m'(R2) = 0 by construction)

Eigenvalue equation (Robin inner BC):
    −k φ_m'(R1) + h_w φ_m(R1) = 0
    i.e. kβ_m[J_1(β_m R1)Y_1(β_m R2) − Y_1(β_m R1)J_1(β_m R2)]
           + h_w[J_0(β_m R1)Y_1(β_m R2) − Y_0(β_m R1)J_1(β_m R2)] = 0

Coefficients:  C_m = [(T0 − T_env)/N_m] ∫_{R1}^{R2} r φ_m(r) dr
               N_m = ∫_{R1}^{R2} r φ_m²(r) dr

SCAM models the Robin BC via ENERGY_BALANCE with alpha_conv = h_w, T_aw = T_env,
and emissivity = 0 (no radiation).  The Newton solver on T_wall is therefore
the numerical implementation of the convection Robin BC.
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

H_W    = 3000.0             # [W/m²/K] convection coefficient at inner wall
T_ENV  = 5000.0             # [K]      environment temperature
T_INIT =    0.0             # [K]      initial temperature

R1     = 0.005              # [m] inner radius
R2     = 0.055              # [m] outer radius
SCYL   = R2 - R1            # [m] wall thickness = 50 mm

N_NODES = 101               # dx ≈ 0.5 mm (fast tests)
DT_MAX  = 0.1               # [s]

_SNAP_THRESHOLDS = {30.0: 3.0, 60.0: 2.0, 100.0: 1.5}
T_SNAPS = sorted(_SNAP_THRESHOLDS)

_N_EIGEN = 200


# ---------------------------------------------------------------------------
# Green's function analytical solution
# ---------------------------------------------------------------------------

def _phi(beta: float, r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return j0(beta * r) * y1(beta * R2) - y0(beta * r) * j1(beta * R2)


def _dphi_dr(beta: float, r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return beta * (y1(beta * r) * j1(beta * R2) - j1(beta * r) * y1(beta * R2))


def _eigen_func(beta: float) -> float:
    return -K * float(_dphi_dr(beta, R1)) + H_W * float(_phi(beta, R1))


def _find_eigenvalues(n: int) -> np.ndarray:
    spacing = np.pi / SCYL
    beta_scan = np.linspace(0.1, spacing * (n + 30), 10 * (n + 30) * 20)
    fv = np.vectorize(_eigen_func)(beta_scan)
    betas: list[float] = []
    for i in range(len(fv) - 1):
        if fv[i] * fv[i + 1] < 0:
            try:
                bm = brentq(_eigen_func, beta_scan[i], beta_scan[i + 1], xtol=1e-12)
                betas.append(bm)
                if len(betas) == n:
                    break
            except ValueError:
                pass
    return np.array(betas)


def _build_greens_series(n: int = _N_EIGEN):
    betas = _find_eigenvalues(n)
    coeffs = np.empty(len(betas))
    for i, bm in enumerate(betas):
        norm, _ = quad(lambda r: r * _phi(bm, r) ** 2, R1, R2, limit=200)
        proj, _ = quad(lambda r: r * _phi(bm, r),      R1, R2, limit=200)
        coeffs[i] = ((T_INIT - T_ENV) / norm) * proj
    return betas, coeffs


def greens_T(r, t: float, betas, coeffs) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    T = np.full_like(r, T_ENV)
    for bm, cm in zip(betas, coeffs):
        ef = np.exp(-ALPHA * bm ** 2 * t)
        if ef < 1e-15:
            break
        T += cm * _phi(bm, r) * ef
    return T


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def greens_series():
    return _build_greens_series(_N_EIGEN)


@pytest.fixture(scope="module")
def convection_snapshots():
    mat = make_inert_material("V2f", k=K, rho=RHO, cp=CP, emissivity=0.0)
    geom = GeometryConfig(geometry_type=GeometryType.HOLLOW_CYLINDER, r_inner=R1)
    stack = StackConfig(layers=[LayerConfig("V2f", SCYL, N_NODES, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=H_W,
        T_aw=T_ENV,
        emissivity=0.0,
        T_rad_in=0.0,
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
    results = run_case(stack, {"V2f": mat}, surface_bc, back_bc, options,
                       initial_T=T_INIT, geom=geom)

    saved_times = np.array([s.time for s in results.snapshots])
    return {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }


# ---------------------------------------------------------------------------
# 1. Inner-wall temperature satisfies the Robin BC
# ---------------------------------------------------------------------------

def test_v2f_robin_bc(convection_snapshots, greens_series):
    """T_wall from SCAM must match the analytical T(R1,t) within 1%.

    The N=101 test grid (dx=0.5 mm) introduces O(dx) spatial error in the
    near-wall conduction flux that shows up as a small T_wall offset (~0.7%
    at t=30 s); the example script at N=500 (dx=0.1 mm) gives <0.2%.
    """
    betas, coeffs = greens_series
    for t_req, snap in convection_snapshots.items():
        T_wall_scam = float(snap.T[0])
        T_wall_ana  = float(greens_T(np.array([R1]), snap.time, betas, coeffs)[0])
        err_pct = abs(T_wall_scam - T_wall_ana) / T_wall_ana * 100.0
        assert err_pct < 1.0, (
            f"t={t_req} s: T_wall = {T_wall_scam:.3f} K vs analytical {T_wall_ana:.3f} K "
            f"({err_pct:.3f}% error, threshold 1.0%)"
        )


# ---------------------------------------------------------------------------
# 2. Temperature profile matches the Green's function at each snapshot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_req", T_SNAPS)
def test_v2f_greens_profile(convection_snapshots, greens_series, t_req):
    """Interior profile agrees with the Green's function within the error threshold.

    Comparison band: 5%–95% of the current temperature span at the inner wall.
    This excludes the steep near-wall gradient (where relative error diverges)
    and the cold unheated region at the outer wall.
    """
    snap = convection_snapshots[t_req]
    r_num = R1 + snap.mesh.y_nodes
    T_num = snap.T
    betas, coeffs = greens_series

    T_ana  = greens_T(r_num, snap.time, betas, coeffs)
    dT_ref = T_ana - T_INIT
    T_span_now = float(T_ana[0] - T_INIT)

    mask = (dT_ref > 0.05 * T_span_now) & (dT_ref < 0.95 * T_span_now)
    if mask.sum() == 0:
        mask = dT_ref > 10.0

    assert mask.sum() > 0, (
        f"t={t_req} s: no nodes in comparison band (front may not have penetrated)"
    )

    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
    max_err_pct = float(rel_err.max()) * 100.0
    thr = _SNAP_THRESHOLDS[t_req]
    assert max_err_pct < thr, (
        f"t={t_req} s: max relative error {max_err_pct:.2f}% > {thr}% "
        f"(HOLLOW_CYLINDER convection, N={N_NODES}, dx={SCYL/(N_NODES-1)*1e3:.2f} mm)"
    )
