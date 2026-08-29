# SPDX-License-Identifier: MIT
"""Integration test: energy conservation for prescribed-flux case.

Applies a constant heat flux to an inert slab with adiabatic back face.
Total energy stored must equal total energy injected within 2%:

    ΔE = ρ·cp·ΔT·volume  ≈  Q0 · t_end  (within 2%)

This verifies:
- FVM assembly correctly accumulates energy
- No spurious sinks or sources in the solver loop
- Time integration is accurate enough over the run

Design choice: PRESCRIBED_FLUX is used (not PRESCRIBED_TEMP) so the total
energy input is exactly Q0 * t_end, with no integration error from rapidly
varying surface flux.
"""
import numpy as np
import pytest

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run


K_CONST, RHO, CP_CONST = 1.0, 180.0, 710.0
Q0     = 50_000.0   # [W/m^2]
T_INIT = 300.0      # [K]
T_END  = 10.0       # [s]


def _inert_material(name: str = "Inert") -> MaterialCard:
    T_pts = np.array([200.0, 3000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, K_CONST)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP_CONST)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name=name, rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


def test_energy_conservation_prescribed_flux():
    """ΔE_stored ≈ Q0 * t_end for prescribed-flux BC on an inert slab."""
    mat = _inert_material()

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=0.05, n_nodes=51, n_subcells=2),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_FLUX,
        q_prescribed=lambda t: Q0,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=T_END, dt_init=0.05, dt_max=0.5,
        dt_min=1e-4, dt_max_dT=500.0, output_dt=T_END,
    )

    results = run(
        stack, {mat.name: mat}, {mat.name: None},
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT, verbose=False,
    )

    # Total energy injected: Q0 * t_end * A_surface [J/m^2 integrated over slab area=1]
    Q_total_expected = Q0 * T_END   # [J] (per unit area, A=1)

    # Energy stored in slab at final snapshot (vs initial) — use ρ·h(T)
    def thermal_energy(snap) -> float:
        from scam.physics.properties import mixture_enthalpy_array
        h = mixture_enthalpy_array([mat], snap.T, snap.rho, snap.mesh.layer_id)
        return float(np.sum(snap.rho * h * snap.mesh.area_nodes * snap.mesh.delta_nodes))

    E0 = thermal_energy(results.snapshots[0])
    Ef = thermal_energy(results.snapshots[-1])
    dE = Ef - E0

    rel_err = abs(dE - Q_total_expected) / Q_total_expected
    assert rel_err < 0.02, (
        f"Energy conservation error {rel_err*100:.2f}% > 2%  "
        f"(dE={dE:.2f} J, expected={Q_total_expected:.2f} J)"
    )


def test_T_surface_prescribed_correctly():
    """With PRESCRIBED_TEMP, T[0] stays at prescribed value every step."""
    mat = _inert_material("Inert2")
    T_WALL = 1500.0

    stack = StackConfig(layers=[LayerConfig(mat.name, 0.05, 21)])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=T_WALL,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(t_end=5.0, dt_init=0.1, dt_max=0.5, output_dt=1.0)

    results = run(
        stack, {mat.name: mat}, {mat.name: None},
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT, verbose=False,
    )

    for snap in results.snapshots[1:]:  # skip initial (t=0) snapshot
        assert abs(snap.T[0] - T_WALL) < 0.01, (
            f"T[0] = {snap.T[0]:.4f} K at t={snap.time:.2f} s; "
            f"expected {T_WALL} K"
        )
        # Temperature should be monotonically decreasing from surface inward
        assert np.all(np.diff(snap.T) <= 1.0), (
            "Temperature profile is not monotonically decreasing from surface"
        )
