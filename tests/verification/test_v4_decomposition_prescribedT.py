# SPDX-License-Identifier: MIT
"""V4 — In-depth decomposition & pyrolysis (prescribed surface temperature).

Adds the charring physics — nodelet Arrhenius decomposition, pyrolysis gas
generation, and the decomposition/gas enthalpy source terms — on top of the
conduction kernel. The surface temperature is *prescribed* (a ramp), so the
surface energy balance and char ablation (B' table) are deliberately left out;
those are V5. TACOT 3.0 is used as the real charring material.

Because there is no closed-form solution for the coupled in-depth response, the
references are physical invariants plus an independent ODE integration:

    * density stays within [rho_char, rho_virgin] everywhere, every step;
    * a sustained-hot surface node fully chars (rho -> rho_char);
    * the cool back face never decomposes (rho stays at rho_virgin);
    * pyrolysis gas is produced while decomposing, and none for an inert layer;
    * the exact nodelet kinetics, integrated over many steps at a fixed
      temperature, match a tight scipy ODE solution (verifies the analytical
      Arrhenius integration used inside the solver).

Note on recession: SCAM's CMA surface mass balance (scam/physics/recession.py)
sets s_dot = m_dot_char / rho_char.  Pyrolysis gas exits through the porous
char via Darcy flow and does NOT drive recession.  V4 (no B' table → m_dot_char=0)
therefore produces zero recession, which is the physically correct result.
"""
import numpy as np
from scipy.integrate import solve_ivp

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.physics.decomposition import arrhenius_k, update_nodelet_densities

from .conftest import make_inert_material, run_case


TACOT_PATH = "scam/materials/ablative_organic/tacot_v3.0.yaml"


def _ramp(t):
    """Surface temperature ramp 300 K -> 1800 K over 40 s, then hold."""
    return 300.0 + 1500.0 * min(t / 40.0, 1.0)


def _run_charring():
    mat, _ = load_material(TACOT_PATH)
    stack = StackConfig(layers=[LayerConfig(mat.name, 0.05, 51, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=_ramp,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=80.0, dt_init=0.01, dt_max=1.0, dt_min=1e-5,
        dt_max_dT=50.0, dt_max_drho_frac=0.05, output_dt=5.0,
    )
    results = run_case(stack, {mat.name: mat}, surface_bc, back_bc, options,
                       initial_T=300.0)
    return mat, results


# ---------------------------------------------------------------------------
# Invariants from a full charring run
# ---------------------------------------------------------------------------

def test_v4_density_within_bounds():
    mat, results = _run_charring()
    for k, snap in enumerate(results.snapshots):
        lo = mat.rho_char - 1e-6
        hi = mat.rho_virgin + 1e-6
        assert np.all(snap.rho >= lo) and np.all(snap.rho <= hi), (
            f"snapshot {k} (t={snap.time:.1f}s): density outside "
            f"[{mat.rho_char}, {mat.rho_virgin}]; "
            f"min={snap.rho.min():.3f}, max={snap.rho.max():.3f}."
        )


def test_v4_surface_fully_chars():
    mat, results = _run_charring()
    rho_surf = results.snapshots[-1].rho[0]
    # Held near 1800 K for tens of seconds: resin should be essentially gone.
    assert rho_surf < mat.rho_char + 0.05 * (mat.rho_virgin - mat.rho_char), (
        f"surface density {rho_surf:.2f} did not approach char "
        f"({mat.rho_char}) under sustained heating."
    )


def test_v4_back_face_stays_virgin():
    mat, results = _run_charring()
    rho_back = results.snapshots[-1].rho[-1]
    assert rho_back > mat.rho_virgin - 1.0, (
        f"back face density {rho_back:.3f} dropped below virgin "
        f"({mat.rho_virgin}); decomposition leaked into the cold region."
    )


def test_v4_pyrolysis_gas_and_recession_consistent():
    mat, results = _run_charring()
    # During active decomposition the surface sees pyrolysis gas...
    m_dot_pyros = np.array([s.m_dot_pyro for s in results.snapshots])
    assert m_dot_pyros.max() > 0.0, "no pyrolysis gas produced during charring."

    # Without a B' table, m_dot_char=0, so CMA gives s_dot=0 (pyro gas exits via Darcy).
    for snap in results.snapshots:
        assert snap.s_dot == 0.0, (
            f"t={snap.time:.1f}s: s_dot={snap.s_dot:.3e} should be zero "
            f"(no B' table → no char ablation → no recession)."
        )


def test_v4_inert_control_no_gas_no_recession():
    """An inert layer under the same ramp must not decompose, gas, or recede."""
    mat = make_inert_material("Inert", k=1.0, rho=200.0, cp=800.0)
    stack = StackConfig(layers=[LayerConfig("Inert", 0.05, 51, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=_ramp,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=80.0, dt_init=0.01, dt_max=1.0, dt_min=1e-5,
        dt_max_dT=50.0, output_dt=20.0,
    )
    results = run_case(stack, {"Inert": mat}, surface_bc, back_bc, options,
                       initial_T=300.0)
    final = results.snapshots[-1]
    assert final.mesh.s_total == 0.0, "inert layer receded."
    assert all(s.m_dot_pyro == 0.0 for s in results.snapshots), \
        "inert layer produced pyrolysis gas."


# ---------------------------------------------------------------------------
# Exact nodelet kinetics vs independent scipy integration
# ---------------------------------------------------------------------------

def test_v4_nodelet_kinetics_vs_scipy():
    """Integrate the nodelet densities at a fixed temperature for many steps and
    compare against a tight scipy solution of the same Arrhenius ODE.

    With s_dot = 0 there is no convective correction, so each component obeys
        drho_i/dt = -k(T) * rho_0 * ((rho_i - rho_r)/rho_0)^m
    and the solver's analytical per-step integration must accumulate to the
    reference over the whole interval.
    """
    mat, _ = load_material(TACOT_PATH)
    T = 1100.0
    J = 4
    n_comp = len(mat.components)

    rho = np.zeros((n_comp, 1, J))
    for ic, comp in enumerate(mat.components):
        rho[ic, 0, :] = comp.rho_0

    T_nod = np.full((1, J), T)
    delta = np.full((1, J), 1.0e-3)
    dt, n_steps = 0.5, 200
    is_shrinking = np.array([False])

    cur = rho.copy()
    for _ in range(n_steps):
        cur, _drho, _drho_comp = update_nodelet_densities(
            mat, cur, T_nod, dt, 0.0, delta, is_shrinking,
        )

    t_final = dt * n_steps
    for ic, comp in enumerate(mat.components):
        def ode(t, y, c=comp):
            excess = max(y[0] - c.rho_r, 0.0)
            return [-arrhenius_k(c, T) * c.rho_0 * (excess / c.rho_0) ** c.m_exp]

        sol = solve_ivp(ode, [0.0, t_final], [comp.rho_0],
                        rtol=1e-9, atol=1e-12)
        ref = sol.y[0, -1]
        got = cur[ic, 0, 0]
        rel = abs(got - ref) / max(ref, 1e-9)
        assert rel < 1e-3, (
            f"component {comp.name}: nodelet integration {got:.5f} vs scipy "
            f"{ref:.5f} ({rel * 100:.3f}% > 0.1%)."
        )
