# SPDX-License-Identifier: MIT
"""V5r — Radiation boundary condition: steady-state slab.

Analytical solution:
    Constant-property inert 1-D slab of thickness L=0.01 m.
    Front (z=0): far-field radiation BC — σ·ε·(T_res⁴ − T_face⁴) = k/L·(T_face − T_back).
    Back  (z=L): T = T_back = 300 K.

    At steady state, d²T/dz²=0 → T is linear: T(z) = T_face + (T_back−T_face)·z/L.
    T_face_ss solves the nonlinear balance above (≈ 443 K).

Time constant:
    The effective radiation transfer coefficient h_rad = 4·ε·σ·T_ss³ ≈ 17.8 W/m²/K.
    Biot number Bi = h_rad·L/k ≈ 0.018 (very soft radiation BC).
    Dominant thermal time constant τ ≈ ρ·cp·L/h_rad ≈ 1040 s.
    Run to t_end = 10000 s (≈ 9.6 τ) to ensure steady state within 0.1 K.

Verification strategy:
    Primary check — SEB closure at every snapshot: the stored q_cond must equal
    the net radiation flux ε·σ·(T_res⁴ − T_face⁴) within the Newton SEB tolerance.
    This directly verifies the radiation BC implementation at every timestep,
    independent of the in-depth state.

    Secondary check — steady-state temperature: T_wall at t=10000 s must match
    the analytical T_face_ss within 0.5 K (dominated by residual transient ≈ 0.06 K).

Parameters:
    k=10 W/m/K, ε=0.9, σ=5.6704×10⁻⁸ W/m²/K⁴, T_res=1300 K, T_back=300 K, L=0.01 m
    k/L = 1000 W/m²/K.  T_face_ss ≈ 443 K.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import run_case


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGMA   = 5.6704e-8   # [W/m²/K⁴]
EPSILON = 0.9
T_RES   = 1300.0      # [K]
T_BACK  = 300.0       # [K]
K       = 10.0        # [W/m/K]
L       = 0.01        # [m]

RHO = 1850.0          # [kg/m³] (arbitrary; does not affect steady state)
CP  = 1000.0          # [J/kg/K] (arbitrary)

N_NODES = 500         # N=500 keeps FVM conductance kink < 0.15 K (scales as ΔT/(4N))
T_INIT  = 300.0       # [K] uniform initial temperature
T_END   = 10000.0     # [s] ≈ 9.6 τ, where τ ≈ 1040 s (effective radiation BC)
SEB_TOL = 1.0         # [W/m²] Newton SEB convergence tolerance


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def _solve_T_face_ss() -> float:
    def balance(T_face):
        return EPSILON * SIGMA * (T_RES**4 - T_face**4) - K / L * (T_face - T_BACK)
    return float(brentq(balance, T_BACK + 1.0, T_RES - 1.0))


T_FACE_SS: float = _solve_T_face_ss()   # ≈ 443 K, computed once at import time
Q_SS: float = K / L * (T_FACE_SS - T_BACK)  # steady-state heat flux [W/m²]


def T_linear(z) -> np.ndarray:
    """Analytical steady-state temperature: linear from T_face_ss to T_back."""
    z = np.asarray(z, dtype=float)
    return T_FACE_SS + (T_BACK - T_FACE_SS) * z / L


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _make_material() -> MaterialCard:
    T_pts  = np.array([200.0, 5000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, K)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name="RadSlab", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0, decomposing=False,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture — run SCAM once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rad_result():
    mat   = _make_material()
    stack = StackConfig(layers=[LayerConfig("RadSlab", L, N_NODES, 1)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=0.0,
        T_aw=0.0,
        emissivity=EPSILON,
        T_rad_in=T_RES,
        view_factor=1.0,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    options = SolverOptions(
        t_end=T_END,
        dt_init=1e-2,
        dt_max=500.0,       # large dt once transient dies (Backward Euler is stable)
        dt_min=1e-4,
        dt_max_dT=1000.0,   # relaxed: no decomposition, no ablation; accuracy not critical
        output_dt=500.0,
        allow_recession=False,
    )
    return run_case(stack, {"RadSlab": mat}, surface_bc, back_bc, options,
                    initial_T=T_INIT)


# ---------------------------------------------------------------------------
# 1. SEB closure: q_cond equals net radiation flux at every snapshot
#    (primary radiation BC verification — independent of steady state)
# ---------------------------------------------------------------------------

def test_v5r_seb_closure(rad_result):
    """At every snapshot, q_cond must equal ε·σ·(T_res⁴ − T_wall⁴) within 2·SEB_TOL.

    The Newton SEB converges to |residual| < SEB_TOL = 1 W/m², so the stored
    q_cond (from F_cond: q_cond = alpha_F·T_wall + beta_F) satisfies:
        q_cond ≈ ε·σ·(T_res⁴ − T_face⁴)
    because q_conv = 0 (alpha_conv = 0, rhoUeCH = 0) and the SEB residual is zero.
    This test verifies the radiation term in the SEB at every timestep output.
    The first snapshot (t=0) is the initial state where no SEB has been solved,
    and is skipped.
    """
    for snap in rad_result.snapshots[1:]:
        q_rad_net = EPSILON * SIGMA * (T_RES**4 - snap.T_wall**4)
        err = abs(snap.q_cond - q_rad_net)
        assert err < 2.0 * SEB_TOL, (
            f"t={snap.time:.1f} s: q_cond={snap.q_cond:.2f}, "
            f"q_rad={q_rad_net:.2f}, |error|={err:.2f} W/m² > {2*SEB_TOL} W/m²"
        )


# ---------------------------------------------------------------------------
# 2. Steady-state surface temperature matches analytical T_face_ss
# ---------------------------------------------------------------------------

def test_v5r_surface_temperature(rad_result):
    """T_wall at t=10000 s must equal the analytical T_face_ss within 0.5 K.

    After 9.6 time constants (τ ≈ 1040 s), the residual transient error is
    e^{−9.6} ≈ 6.7e-5 of ΔT_total ≈ 143 K → < 0.01 K.  The 0.5 K tolerance
    comfortably bounds Newton SEB tolerance (< 0.001 K) and any residual error.
    """
    snap = rad_result.snapshots[-1]
    err = abs(snap.T_wall - T_FACE_SS)
    assert err < 0.5, (
        f"T_wall={snap.T_wall:.4f} K, analytical T_face_ss={T_FACE_SS:.4f} K, "
        f"|error|={err:.4f} K > 0.5 K"
    )


# ---------------------------------------------------------------------------
# 3. Back face is held at T_back
# ---------------------------------------------------------------------------

def test_v5r_back_face_temperature(rad_result):
    """Back face must remain at T_back = 300 K within 0.01 K at all snapshots."""
    for snap in rad_result.snapshots:
        err = abs(snap.T[-1] - T_BACK)
        assert err < 0.01, (
            f"t={snap.time:.1f} s: T_back={snap.T[-1]:.4f} K, expected {T_BACK:.1f} K"
        )


# ---------------------------------------------------------------------------
# 4. Steady-state temperature profile is linear
# ---------------------------------------------------------------------------

def test_v5r_linear_profile(rad_result):
    """Temperature profile at t=10000 s must be linear (max deviation < 0.5 K).

    The FVM for d²T/dz²=0 with constant k is exact for linear profiles on any
    uniform grid.  At steady state, T(z) = T_face_ss + (T_back - T_face_ss)·z/L.
    The only significant error source is the Newton SEB tolerance at z=0 (< 0.001 K),
    plus the residual transient (< 0.01 K at t=10000 s = 9.6τ).
    """
    snap = rad_result.snapshots[-1]
    z    = snap.mesh.y_nodes
    T    = snap.T
    T_lin = T_linear(z)
    max_dev = float(np.max(np.abs(T - T_lin)))
    assert max_dev < 0.5, (
        f"Max profile deviation from linear: {max_dev:.4f} K > 0.5 K\n"
        f"(T_wall={T[0]:.4f} K, T_back={T[-1]:.4f} K, "
        f"T_face_ss={T_FACE_SS:.4f} K)"
    )
