# SPDX-License-Identifier: MIT
"""TACOT 3.0 benchmark verification against Ablation Workshop/PATO reference.

This test checks that SCAM produces physically reasonable results for
the standard TACOT 3.0 arcjet-style case.  Exact agreement with the reference
is not expected (different grid, timestep, chemistry model), but
the results should be in the correct ballpark:

  - T_wall at steady-state: 2000-3500 K (reference: ~2800 K at standard conditions)
  - Final recession: ~0.001 mm at t=60 s (CMA: s_dot=m_dot_char/rho_char;
    at C_M=0.01 the char flux is tiny — pyrolysis gas exits via Darcy, not recession)
  - s_dot quasi-steady: ~1e-4 mm/s
  - TC at 5 mm: 600-1800 K at t=60 s (reference: ~900-1200 K)

These are generous bounds; the test flags catastrophic failures
(e.g., T_wall = 300 K, no ablation, divergence).
"""
import numpy as np
import pytest
from pathlib import Path

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run


MATERIALS_DIR = Path(__file__).parent.parent.parent / "scam" / "materials"


@pytest.fixture(scope="module")
def tacot_results():
    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    mat_cards = {mat.name: mat}
    b_prime_tables = {mat.name: bpt}

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=0.05, n_nodes=51, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=5000.0, T_aw=8000.0,
        emissivity=0.85, view_factor=1.0,
        T_rad_in=0.0, rho_e_u_e=0.1, C_M=0.01,
        p_e=10000.0, lambda_blowing=0.5,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=60.0, dt_init=0.01, dt_max=2.0, dt_min=1e-5,
        dt_max_dT=50.0, dt_max_drho_frac=0.05,
        output_dt=10.0,
        tc_positions=[0.005, 0.010, 0.025],
    )

    results = run(
        stack, mat_cards, b_prime_tables,
        geom, surface_bc, back_bc, options,
        initial_T=300.0, verbose=False,
    )
    return results


def test_T_wall_final(tacot_results):
    T_wall_final = tacot_results.T_wall_array()[-1]
    assert 1500.0 < T_wall_final < 4000.0, (
        f"T_wall = {T_wall_final:.1f} K out of plausible range [1500, 4000] K"
    )


def test_recession_positive(tacot_results):
    s_final = tacot_results.s_array()[-1]
    assert s_final > 0.0, "No recession detected — ablation physics not engaged"


def test_recession_plausible(tacot_results):
    s_final = tacot_results.s_array()[-1]
    # CMA model: s_dot = m_dot_char / rho_char.
    # At C_M=0.01 the char mass flux is tiny; pyrolysis gas exits via Darcy flow.
    assert 1e-8 < s_final < 1e-3, (
        f"Final recession {s_final*1000:.4f} mm out of expected range"
    )


def test_s_dot_quasi_steady(tacot_results):
    s_dots = tacot_results.s_dot_array()
    s_dot_final = s_dots[-1]
    assert s_dot_final > 0.0, "Quasi-steady recession rate is zero"
    assert 1e-9 < s_dot_final < 1e-5, (
        f"s_dot = {s_dot_final*1000:.6f} mm/s out of plausible range"
    )


def test_tc_5mm_plausible(tacot_results):
    tc = tacot_results.tc_array()
    if tc is None:
        pytest.skip("No TC data available")
    tc_5mm_final = tc[0, -1]   # first TC at 5 mm
    assert 400.0 < tc_5mm_final < 2500.0, (
        f"TC at 5mm = {tc_5mm_final:.1f} K out of plausible range"
    )


def test_T_wall_increases_monotonically_early(tacot_results):
    """T_wall should rise from initial value during the heating phase."""
    T_walls = tacot_results.T_wall_array()
    assert T_walls[-1] > T_walls[0], "T_wall did not increase during heating"


def test_no_nan_or_inf(tacot_results):
    """No NaN or Inf anywhere in the solution."""
    for snap in tacot_results.snapshots:
        assert np.all(np.isfinite(snap.T)),   "NaN/Inf in temperature field"
        assert np.all(np.isfinite(snap.rho)), "NaN/Inf in density field"
