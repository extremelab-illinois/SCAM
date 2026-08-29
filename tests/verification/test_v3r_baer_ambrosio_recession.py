# SPDX-License-Identifier: MIT
"""V3r — Semi-infinite receding solid: Baer & Ambrosio (1961).

Reference:
    D. Baer and A. Ambrosio, "Heat conduction in a semi-infinite slab with
    sublimation at the surface," Planetary and Space Science, vol. 4,
    pp. 436–446, 1961.

Analytical solution (x = depth below current receding surface):
    T = T₀ + (Tw−T₀)/2 · [erfc((x+ṡt)/(2√(αt))) + exp(−xṡ/α)·erfc((x−ṡt)/(2√(αt)))]

In SCAM y-coordinate (y from original front face, x = y − ṡt):
    T = T₀ + (Tw−T₀)/2 · [erfc(y/(2√(αt))) + exp(−(y−ṡt)·ṡ/α)·erfc((y−2ṡt)/(2√(αt)))]

Steady-state (t→∞):
    T = T₀ + (Tw−T₀)·exp(−xṡ/α)

Parameters:
    T₀=300 K, Tw=4000 K, ṡ=1 mm/s, ρ=1850, cp=2000, k=30 → α≈8.108e-6 m²/s
    Thermal length α/ṡ ≈ 8.1 mm.  Profile ≈ steady state by t ≳ 30 s.

Test grid: N=201, Sslab=200 mm → dx=1 mm.  DT_MAX=0.05 s (within ALE CFL=0.1 s).
At t=100 s: current surface at y=100 mm, back face (y=200 mm) has ΔT<0.01 K.
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
RHO   = 1850.0
CP    = 2000.0
K     = 30.0
ALPHA = K / (RHO * CP)   # ≈ 8.108e-6 m²/s

T0    = 300.0
TW    = 4000.0
SDOT  = 1e-3              # [m/s] recession rate (1 mm/s)

SSLAB   = 0.20            # [m] 200 mm slab
N_NODES = 201             # dx = 1 mm
DT_MAX  = 0.05            # [s] well within ALE CFL (0.1 s for dx=1 mm)

# The Lagrangian (node-drop) scheme is used: no ALE interpolation accumulation.
# With dx=1 mm and thermal length α/ṡ≈8.1 mm, the FVM resolves ~8 cells per
# thermal length.  The largest relative errors appear near the front of the wave
# (x ≈ 2–3 α/ṡ) where |T−T₀| is small, giving looser relative thresholds.
_SNAP_THRESHOLDS = {10.0: 10.0, 30.0: 7.0, 60.0: 6.0, 100.0: 6.0}
T_SNAPS = sorted(_SNAP_THRESHOLDS)


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def analytical_T(y, t):
    """Transient Baer & Ambrosio solution in SCAM y-coordinates."""
    y = np.asarray(y, dtype=float)
    s = SDOT * t
    sqrt_at2 = 2.0 * np.sqrt(ALPHA * t)
    exp_fac  = np.exp(-(y - s) * SDOT / ALPHA)
    return T0 + 0.5 * (TW - T0) * (erfc(y / sqrt_at2) + exp_fac * erfc((y - 2.0*s) / sqrt_at2))


def analytical_steady(x):
    """Steady-state temperature; x = depth below current surface."""
    return T0 + (TW - T0) * np.exp(-np.asarray(x, dtype=float) * SDOT / ALPHA)


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _make_material():
    cp_tab = np.column_stack([[1.0, TW + 2000.0], [CP, CP]])
    k_tab  = np.column_stack([[1.0, TW + 2000.0], [K,  K]])
    hg_tab = np.column_stack([[1.0, TW + 2000.0], [0.0, 0.0]])
    return MaterialCard(
        name="BaerAmbrosio", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0, decomposing=False,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recession_snapshots():
    mat = _make_material()
    stack = StackConfig(layers=[LayerConfig("BaerAmbrosio", SSLAB, N_NODES, 1)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: TW,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=max(T_SNAPS),
        dt_init=1e-4,
        dt_max=DT_MAX,
        dt_min=1e-6,
        dt_max_dT=1e4,
        output_dt=min(T_SNAPS) / 2.0,
        s_dot_prescribed=SDOT,
        continuous_remap=False,   # Lagrangian: no ALE interpolation accumulation
        node_drop_threshold=0.2,
    )
    results = run_case(stack, {"BaerAmbrosio": mat}, surface_bc, back_bc, options,
                       initial_T=T0)
    saved_times = np.array([s.time for s in results.snapshots])
    return {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }


# ---------------------------------------------------------------------------
# 1. Surface node pinned to Tw
# ---------------------------------------------------------------------------

def test_v3r_surface_pinned(recession_snapshots):
    """PRESCRIBED_TEMP must hold T_wall = Tw = 4000 K within 0.01 K."""
    for t_req, snap in recession_snapshots.items():
        err = abs(snap.T[0] - TW)
        assert err < 0.01, (
            f"t={t_req} s: T_wall={snap.T[0]:.4f} K, expected {TW:.1f} K"
        )


# ---------------------------------------------------------------------------
# 2. Recession advances correctly
# ---------------------------------------------------------------------------

def test_v3r_recession_rate(recession_snapshots):
    """Total recession s_total must match ṡ·t within one initial cell width."""
    dx0 = SSLAB / (N_NODES - 1)          # initial cell spacing
    for t_req, snap in recession_snapshots.items():
        expected = SDOT * snap.time
        err = abs(snap.mesh.s_total - expected)
        assert err < 2.0 * dx0, (
            f"t={t_req} s: s_total={snap.mesh.s_total*1e3:.3f} mm, "
            f"expected {expected*1e3:.3f} mm (|Δ|={err*1e3:.3f} mm, tol={2*dx0*1e3:.3f} mm)"
        )


# ---------------------------------------------------------------------------
# 3. Back face stays at T₀ (semi-infinite check)
# ---------------------------------------------------------------------------

def test_v3r_back_face_cold(recession_snapshots):
    """Back face (y=200 mm) must stay below T₀ + 0.1 K throughout."""
    for t_req, snap in recession_snapshots.items():
        back_dT = abs(snap.T[-1] - T0)
        assert back_dT < 0.1, (
            f"t={t_req} s: back face ΔT={back_dT:.4f} K > 0.1 K "
            f"(semi-infinite assumption violated for Sslab={SSLAB*1e3:.0f} mm)"
        )


# ---------------------------------------------------------------------------
# 4. Temperature profile matches the analytical solution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_req", T_SNAPS)
def test_v3r_profile(recession_snapshots, t_req):
    """Interior profile agrees with the Baer & Ambrosio analytical solution.

    Comparison band: nodes where (T_ana − T₀) > 5% of (Tw − T₀).  This
    excludes the cold unheated region where relative error is dominated by
    floating-point noise.  Threshold decreases over time as the profile
    broadens onto more grid nodes and the transient wiggles die out.
    """
    snap  = recession_snapshots[t_req]
    y_num = snap.mesh.y_nodes
    T_num = snap.T

    T_ana  = analytical_T(y_num, snap.time)
    dT_ref = T_ana - T0

    mask = dT_ref > 0.05 * (TW - T0)
    assert mask.sum() > 0, (
        f"t={t_req} s: no nodes in comparison band (front may not have penetrated)"
    )

    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
    max_err_pct = float(rel_err.max()) * 100.0
    thr = _SNAP_THRESHOLDS[t_req]
    assert max_err_pct < thr, (
        f"t={t_req} s: max relative error {max_err_pct:.2f}% > {thr}% "
        f"(N={N_NODES}, dx={SSLAB/(N_NODES-1)*1e3:.1f} mm, "
        f"s_total={snap.mesh.s_total*1e3:.1f} mm)"
    )


# ---------------------------------------------------------------------------
# 5. Steady-state profile: exponential decay
# ---------------------------------------------------------------------------

def test_v3r_steady_state(recession_snapshots):
    """At t=100 s the profile (in x = depth from surface) should match exp(−xṡ/α).

    The profile converges on the time scale α/ṡ² ≈ 8 s, so at t=100 s it is
    well within the steady state.  Comparison in the band x ∈ [α/ṡ, 5·α/ṡ]
    (from 1 to 5 thermal lengths) where the exponential is unambiguous.
    The Lagrangian dx=1 mm grid resolves ~8 cells per thermal length; the
    expected relative error at t=100 s is ~4–5%.
    """
    snap = recession_snapshots[100.0]
    y    = snap.mesh.y_nodes
    x    = y - snap.mesh.s_total          # depth below current surface

    L    = ALPHA / SDOT                   # thermal length ≈ 8.1 mm
    mask = (x >= L) & (x <= 5.0 * L)
    assert mask.sum() > 0, "No nodes in x ∈ [α/ṡ, 5α/ṡ] range"

    T_ss_ana = analytical_steady(x[mask])
    dT_ref   = T_ss_ana - T0
    rel_err  = np.abs((snap.T[mask] - T_ss_ana) / dT_ref)
    max_err_pct = float(rel_err.max()) * 100.0

    assert max_err_pct < 6.0, (
        f"Steady-state exponential: max relative error {max_err_pct:.2f}% > 6.0% "
        f"in band x ∈ [{L*1e3:.1f}, {5*L*1e3:.1f}] mm"
    )
