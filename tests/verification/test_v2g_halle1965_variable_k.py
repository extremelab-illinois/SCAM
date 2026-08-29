# SPDX-License-Identifier: MIT
"""V2g — Semi-infinite solid with variable thermal conductivity (Halle 1965).

Reference:
    H. Halle, "Exact solution of elementary transient heat conduction problem
    involving temperature dependent properties," Trans. ASME, Series C,
    J. Heat Transfer, vol. 87, no. 3, pp. 420–421, August 1965.

Analytical solution (similarity variable η = x / (2√(α₀t))):

    T(x,t) = T0 + (Tw − T0)·[A·erfc(η) + (1−A)·erfc(n·η)]

Thermal conductivity (parametric in η):

    k(η)/k₀ = [A·e^{−η²} + (1−A)/n·e^{−n²η²}]
               ────────────────────────────────────
               [A·e^{−η²} + (1−A)·n·e^{−n²η²}  ]

Two cases:
    Case 1: A=0.50, n=2  →  k at Tw = 0.5 k₀  (k decreasing with T)
    Case 2: A=1.85, n=2  →  k at Tw = 9.5 k₀  (k increasing with T)

NOTE on slab thickness:  Case 2 has α_local up to 9.5α₀ near the hot wall, so
the thermal front penetrates ~175 mm by t=100 s.  A 200 mm slab is used (not
the user-specified 100 mm) to keep the back face below 0.01 K throughout.

Grid: N=201, Sslab=200mm → dx=1mm.  Thresholds are widened versus the 0.1mm
example grid to account for the coarser spatial resolution.
"""
import numpy as np
import pytest
from scipy.special import erfc

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import run_case


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RHO    = 1850.0
CP     = 2000.0
K0     = 30.0
ALPHA0 = K0 / (RHO * CP)    # ≈ 8.108e-6 m²/s

T0 = 300.0    # [K] initial / far-field
TW = 1000.0   # [K] prescribed wall

SSLAB   = 0.20    # [m] 200 mm — semi-infinite for t ≤ 100 s for both cases
N_NODES = 201     # dx = 1 mm  (fast tests)
DT_MAX  = 0.5     # [s]

# Case 2 (A=1.85) has stricter early-time error due to near-wall curvature, but
# loosens at late times because the front is wide.  Case 1 (A=0.5) follows
# the opposite trend (front narrows effective α).  Common thresholds used here.
_SNAP_THRESHOLDS = {30.0: 6.0, 60.0: 4.0, 100.0: 3.0}
T_SNAPS = sorted(_SNAP_THRESHOLDS)


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def _theta(eta, A, n):
    return A * erfc(eta) + (1 - A) * erfc(n * eta)


def _k_ratio(eta, A, n):
    e1 = np.exp(-eta ** 2)
    e2 = np.exp(-(n * eta) ** 2)
    return (A * e1 + (1 - A) / n * e2) / (A * e1 + (1 - A) * n * e2)


def analytical_T(x, t, A, n):
    eta = np.asarray(x, dtype=float) / (2.0 * np.sqrt(ALPHA0 * t))
    return T0 + (TW - T0) * _theta(eta, A, n)


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _make_halle_material(A: float, n: float, n_pts: int = 1000) -> MaterialCard:
    """Non-decomposing material with Halle k(T) relation."""
    eta_fine   = np.linspace(0.0, 0.2,  n_pts // 2)
    eta_coarse = np.linspace(0.2, 5.0,  n_pts - n_pts // 2)
    eta = np.unique(np.concatenate([eta_fine, eta_coarse]))

    T_eta = T0 + (TW - T0) * _theta(eta, A, n)
    k_eta = K0 * _k_ratio(eta, A, n)

    idx = np.argsort(T_eta)
    T_s, k_s = T_eta[idx], k_eta[idx]
    _, uniq = np.unique(np.round(T_s, 6), return_index=True)
    T_s, k_s = T_s[uniq], k_s[uniq]

    T_full = np.concatenate([[1.0],    T_s, [TW + 2000.0]])
    k_full = np.concatenate([[k_s[0]], k_s, [k_s[-1]]])
    k_tab  = np.column_stack([T_full, k_full])
    cp_tab = np.column_stack([[1.0, TW + 2000.0], [CP, CP]])
    hg_tab = np.column_stack([[1.0, TW + 2000.0], [0.0, 0.0]])

    return MaterialCard(
        name="Halle", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0, decomposing=False,
    )


# ---------------------------------------------------------------------------
# Case fixture factory
# ---------------------------------------------------------------------------

def _run_halle(A: float, n: float):
    mat = _make_halle_material(A, n)
    stack = StackConfig(layers=[LayerConfig("Halle", SSLAB, N_NODES, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: TW,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=max(T_SNAPS),
        dt_init=0.02,
        dt_max=DT_MAX,
        dt_min=1e-4,
        dt_max_dT=1e4,
        output_dt=min(T_SNAPS) / 2.0,
        max_picard=10,
    )
    results = run_case(stack, {"Halle": mat}, surface_bc, back_bc, options,
                       initial_T=T0)
    saved_times = np.array([s.time for s in results.snapshots])
    return {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }


@pytest.fixture(scope="module")
def case1_snapshots():
    return _run_halle(A=0.50, n=2.0)


@pytest.fixture(scope="module")
def case2_snapshots():
    return _run_halle(A=1.85, n=2.0)


# ---------------------------------------------------------------------------
# 1. Surface node pinned to Tw
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["case1_snapshots", "case2_snapshots"])
def test_v2g_surface_pinned(fixture_name, request):
    snaps = request.getfixturevalue(fixture_name)
    for t_req, snap in snaps.items():
        err = abs(snap.T[0] - TW)
        assert err < 0.01, (
            f"{fixture_name} t={t_req} s: T_wall={snap.T[0]:.4f} K, expected {TW} K"
        )


# ---------------------------------------------------------------------------
# 2. Back face stays cold (semi-infinite check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["case1_snapshots", "case2_snapshots"])
def test_v2g_back_face_cold(fixture_name, request):
    snaps = request.getfixturevalue(fixture_name)
    for t_req, snap in snaps.items():
        back_dT = abs(snap.T[-1] - T0)
        assert back_dT < 0.01, (
            f"{fixture_name} t={t_req} s: back face ΔT={back_dT:.4f} K > 0.01 K "
            f"(Sslab={SSLAB*1e3:.0f} mm may be too thin for this case)"
        )


# ---------------------------------------------------------------------------
# 3. Interior temperature profile matches Halle analytical solution
# ---------------------------------------------------------------------------

def _check_profile(snaps, A, n, label):
    for t_req in T_SNAPS:
        snap  = snaps[t_req]
        y_num = snap.mesh.y_nodes
        T_num = snap.T

        T_ana  = analytical_T(y_num, snap.time, A, n)
        dT_ref = T_ana - T0
        T_span = float(T_ana[0] - T0)

        mask = (dT_ref > 0.05 * T_span) & (dT_ref < 0.95 * T_span)
        if mask.sum() == 0:
            mask = dT_ref > 10.0

        assert mask.sum() > 0, (
            f"{label} t={t_req} s: no nodes in comparison band"
        )

        rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
        max_err_pct = float(rel_err.max()) * 100.0
        thr = _SNAP_THRESHOLDS[t_req]
        assert max_err_pct < thr, (
            f"{label} t={t_req} s: max relative error {max_err_pct:.2f}% > {thr}% "
            f"(dx={SSLAB/(N_NODES-1)*1e3:.1f} mm)"
        )


def test_v2g_case1_profile(case1_snapshots):
    """Case 1 (A=0.50, n=2): k decreases from k₀ at T0 to 0.5k₀ at Tw."""
    _check_profile(case1_snapshots, 0.50, 2.0, "Case1")


def test_v2g_case2_profile(case2_snapshots):
    """Case 2 (A=1.85, n=2): k increases from k₀ at T0 to 9.5k₀ at Tw."""
    _check_profile(case2_snapshots, 1.85, 2.0, "Case2")
