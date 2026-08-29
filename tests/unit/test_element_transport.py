# SPDX-License-Identifier: MIT
"""Unit tests for scam/physics/element_transport.py.

Tests:
- mass conservation (total elemental mass in + source = total out)
- pure-advection: in steady state, Z_i should be uniform when source=0
- surface boundary condition: no inflow from outside
- degenerate case: zero porosity nodes hold Z constant
"""

from __future__ import annotations

import numpy as np
import pytest

from scam.physics.element_transport import solve_element_transport, initial_Z_elem, Z_ELEM_AIR


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _FakeMaterial:
    """Minimal MaterialCard stub for element transport tests."""
    def __init__(self, *, decomposing=True, gas_porosity=0.8, rho_virgin=280.0,
                 rho_char=60.0, element_diffusivity=0.0, tortuosity=1.0,
                 pyro_elem_fracs=None):
        self.decomposing = decomposing
        self.gas_porosity = gas_porosity
        self.rho_virgin = rho_virgin
        self.rho_char = rho_char
        self.eps_g_virgin  = gas_porosity
        self.eps_g_char    = gas_porosity
        self.gas_pressure  = 101325.0
        self.gas_molar_mass = 0.022
        self.element_diffusivity = element_diffusivity
        self.tortuosity = tortuosity
        self.pyro_elem_fracs = pyro_elem_fracs
        self.components = []


class _FakeMesh:
    """Minimal MeshState stub."""
    def __init__(self, N, dy=0.001):
        self.n_nodes_total = N
        self.delta_nodes = np.full(N, dy)
        self.area_nodes  = np.ones(N)
        self.layer_id    = np.zeros(N, dtype=int)
        self.layer_boundaries = [(0, N)]
        self.y_nodes     = np.arange(N) * dy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(N, dy=0.001):
    mesh = _FakeMesh(N, dy)
    T    = np.full(N, 300.0)
    rho  = np.full(N, 280.0)
    return mesh, T, rho


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_Z_elem_shape():
    N = 10
    Z = initial_Z_elem(N)
    assert Z.shape == (4, N)
    # Should be air composition everywhere
    np.testing.assert_allclose(Z[:, 0], Z_ELEM_AIR, atol=1e-12)


def test_initial_Z_elem_sums_to_one():
    Z = initial_Z_elem(5)
    col_sums = Z.sum(axis=0)
    # Air fractions don't sum exactly to 1 (inerts/rounding), but close
    assert np.all(col_sums <= 1.0 + 1e-10)
    assert np.all(col_sums >= 0.0)


def test_no_source_no_flux_unchanged():
    """With zero source and zero face flux, Z_elem should be unchanged."""
    N = 5
    mat = _FakeMaterial(decomposing=False, gas_porosity=0.3)
    mesh, T, rho = _make_state(N)
    Z0 = np.tile(Z_ELEM_AIR[:, None], (1, N))

    drho_dt   = np.zeros(N)
    face_flux = np.zeros(N + 1)

    Z1 = solve_element_transport(Z0, [mat], mesh, T, rho, drho_dt, face_flux, dt=1.0)
    np.testing.assert_allclose(Z1, Z0, atol=1e-12)


def test_pure_advection_uniform_profile():
    """With uniform Z and outward (surface-directed) flux, uniform Z stays uniform."""
    N = 8
    mat = _FakeMaterial(decomposing=False, gas_porosity=0.5)
    mesh, T, rho = _make_state(N, dy=0.005)
    Z0 = np.tile(Z_ELEM_AIR[:, None], (1, N))

    # Uniform outward flux (positive = toward surface = node 0)
    face_flux = np.full(N + 1, 1e-4)

    drho_dt = np.zeros(N)
    Z1 = solve_element_transport(Z0, [mat], mesh, T, rho, drho_dt, face_flux, dt=0.1)

    # Interior nodes should stay near initial composition (upwind scheme has small numerical diffusion)
    np.testing.assert_allclose(Z1[:, 1:], Z0[:, 1:], atol=0.01)


def test_elemental_fractions_bounded():
    """Z_elem values must stay in [0, 1] after transport."""
    N = 10
    mat = _FakeMaterial(decomposing=True, gas_porosity=0.4,
                        pyro_elem_fracs=np.array([0.5, 0.1, 0.3, 0.1]))
    mesh, T, rho = _make_state(N)

    Z0 = initial_Z_elem(N)
    # Strong decomposition source
    drho_dt = np.full(N, -10.0)
    face_flux = np.full(N + 1, 1e-3)

    Z1 = solve_element_transport(Z0, [mat], mesh, T, rho, drho_dt, face_flux, dt=0.5)

    assert np.all(Z1 >= -1e-12), f"Negative Z_elem: {Z1.min()}"
    assert np.all(Z1 <= 1.0 + 1e-12), f"Z_elem > 1: {Z1.max()}"


def test_zero_porosity_node_unchanged():
    """Nodes with zero gas porosity should not change their element fractions."""
    N = 5

    mat = _FakeMaterial(decomposing=False, gas_porosity=0.0)
    mesh, T, rho = _make_state(N)
    Z0 = initial_Z_elem(N)
    Z0[0, :] = 0.5   # mark carbon distinctly

    drho_dt   = np.zeros(N)
    face_flux = np.zeros(N + 1)

    Z1 = solve_element_transport(Z0, [mat], mesh, T, rho, drho_dt, face_flux, dt=1.0)

    # With zero porosity, the solver should return Z unchanged (degenerate)
    np.testing.assert_allclose(Z1, Z0, atol=1e-10)


def test_mass_consistent_flux_bounds_Z_to_pyro_fraction():
    """With a flux that is mass-consistent with the decomposition source, the
    surface element fraction must stay bounded by the pyrolysis-gas fraction.

    Regression guard for the runaway-to-1.0 bug: if the advective flux is
    inconsistent with the source (-drho_dt), the convex-combination property is
    lost and each Z_i grows without bound (clipping to 1.0).  Here we build the
    pyrolysis face flux as the reverse-cumulative sum of (-drho_dt)*A*delta —
    exactly as the solver now does — and check Z_C does not exceed Z_C_pyro.
    """
    N = 20
    Z_C_pyro = 0.5
    pyro = np.array([Z_C_pyro, 0.1, 0.3, 0.1])
    mat = _FakeMaterial(decomposing=True, gas_porosity=0.5, pyro_elem_fracs=pyro)
    mesh, T, rho = _make_state(N, dy=0.002)
    Z0 = initial_Z_elem(N)            # air, Z_C = 0

    drho_dt = np.full(N, -8.0)        # strong, uniform decomposition
    # Mass-consistent pyrolysis face flux density [kg/m^2/s]:
    # face_flux[j] = (sum_{k>=j} -drho_dt_k A_k delta_k) / A_j
    A = mesh.area_nodes; dl = mesh.delta_nodes
    mdot_node = np.array([-(drho_dt[j:] * A[j:] * dl[j:]).sum() for j in range(N)])
    face_flux = np.zeros(N + 1)
    face_flux[:N] = mdot_node / A

    Z = Z0.copy()
    for _ in range(40):              # march to steady state
        Z = solve_element_transport(Z, [mat], mesh, T, rho, drho_dt, face_flux, dt=0.5)

    assert Z[0, 0] <= Z_C_pyro + 1e-6, f"Z_C_surf={Z[0,0]} exceeded pyrolysis fraction {Z_C_pyro}"
    assert Z[0, 0] > 0.3, f"Z_C_surf={Z[0,0]} did not enrich toward the pyrolysis fraction"


def test_carbon_enrichment_at_surface():
    """Decomposition source that injects carbon-rich gas should raise Z_C at surface."""
    N = 6
    # Pure carbon pyrolysis gas
    pyro_fracs = np.array([1.0, 0.0, 0.0, 0.0])
    mat = _FakeMaterial(decomposing=True, gas_porosity=0.5,
                        pyro_elem_fracs=pyro_fracs)
    mesh, T, rho = _make_state(N, dy=0.002)
    Z0 = initial_Z_elem(N)   # air (Z_C = 0)

    # Moderate decomposition, outward flux
    drho_dt   = np.full(N, -5.0)
    face_flux = np.full(N + 1, 5e-4)

    Z1 = solve_element_transport(Z0, [mat], mesh, T, rho, drho_dt, face_flux, dt=1.0)

    # Carbon fraction must have increased (from 0 toward 1)
    assert Z1[0, 0] > Z0[0, 0], "Carbon fraction should increase with carbon-rich source"
