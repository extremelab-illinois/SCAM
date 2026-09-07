# SPDX-License-Identifier: MIT
"""Back-face boundary conditions: radiation BC + prescribed-flux sign convention.

Both behaviours here were reported by a public user (2026-09-07):
  * there was no way to model radiation loss at the back face;
  * `q_back` was documented as "+ into material" but the assembly applied it
    with the opposite sign, so a positive `q_back` cooled the back face.

The radiation BC has an exact steady reference.  For an inert slab of constant
`k` and thickness `L`, with the front face held at `T_s` and the back face
radiating to an enclosure at `T_env`, energy balance at steady state gives

    k (T_s - T_b) / L  =  eps * sigma * (T_b^4 - T_env^4)

which is solved here with Brent's method and compared against SCAM.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.core.constants import SIGMA_SB
from scam.solvers.material_response import run

from .conftest import make_inert_material


K_COND, RHO, CP = 2.0, 1600.0, 1000.0
L, T_SURF, T_ENV, EPS = 0.05, 1200.0, 300.0, 0.8


def _run(back_bc, n_nodes=121, t_end=40000.0, T_surf=T_SURF, k=K_COND):
    mat = make_inert_material("Inert", k=k, rho=RHO, cp=CP)
    stack = StackConfig(layers=[LayerConfig(mat.name, L, n_nodes, 1)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_surf
    )
    results = run(
        stack, {mat.name: mat}, {mat.name: None},
        GeometryConfig(geometry_type=GeometryType.SLAB),
        surface_bc, back_bc,
        SolverOptions(t_end=t_end, dt_init=0.01, dt_max=25.0,
                      output_dt=t_end, allow_recession=False),
        initial_T=300.0, verbose=False,
    )
    return results.snapshots[-1]


def _exact_T_back(k=K_COND):
    """Steady back-face temperature from k(Ts-Tb)/L = eps*sigma*(Tb^4 - Tenv^4)."""
    return brentq(
        lambda Tb: k * (T_SURF - Tb) / L - EPS * SIGMA_SB * (Tb**4 - T_ENV**4),
        T_ENV, T_SURF,
    )


def test_radiation_back_face_matches_analytic_equilibrium():
    """Steady T_back matches the exact radiative-equilibrium root."""
    snap = _run(BackBCConfig(bc_type=BackBCType.RADIATION,
                             emissivity_back=EPS, T_env_back=T_ENV))
    assert abs(snap.T[-1] - _exact_T_back()) < 1.0     # 121 nodes -> ~0.55 K


def test_radiation_back_face_is_first_order_convergent():
    """The residual is the documented half-cell boundary error, not a bug.

    Halving the grid spacing must halve the error (first-order boundary
    treatment, see docs/verification/verification.md "A note on discretization
    error").
    """
    exact = _exact_T_back()
    errs = [
        _run(BackBCConfig(bc_type=BackBCType.RADIATION,
                          emissivity_back=EPS, T_env_back=T_ENV),
             n_nodes=n).T[-1] - exact
        for n in (61, 121, 241)
    ]
    assert all(e > 0 for e in errs)                     # consistent sign
    for coarse, fine in zip(errs, errs[1:]):
        assert 1.8 < coarse / fine < 2.2                # first order


def test_radiation_with_zero_emissivity_is_adiabatic():
    """eps=0 and h=0 must reproduce the ADIABATIC result exactly."""
    adia = _run(BackBCConfig(bc_type=BackBCType.ADIABATIC))
    rad0 = _run(BackBCConfig(bc_type=BackBCType.RADIATION,
                             emissivity_back=0.0, h_back=0.0))
    assert rad0.T[-1] == pytest.approx(adia.T[-1], abs=1e-9)


def test_radiation_back_face_cools_relative_to_adiabatic():
    """A radiating back face must be colder than an insulated one."""
    adia = _run(BackBCConfig(bc_type=BackBCType.ADIABATIC))
    rad = _run(BackBCConfig(bc_type=BackBCType.RADIATION,
                            emissivity_back=EPS, T_env_back=T_ENV))
    assert rad.T[-1] < adia.T[-1] - 100.0


def test_convective_film_only_matches_analytic_equilibrium():
    """h_back with eps=0 gives the linear Robin equilibrium k(Ts-Tb)/L = h(Tb-Tenv)."""
    h = 25.0
    snap = _run(BackBCConfig(bc_type=BackBCType.RADIATION, emissivity_back=0.0,
                             h_back=h, T_env_back=T_ENV))
    exact = (K_COND / L * T_SURF + h * T_ENV) / (K_COND / L + h)
    assert abs(snap.T[-1] - exact) < 1.0


def test_prescribed_back_flux_sign_is_into_material():
    """Positive q_back must HEAT the back face (documented '+ into material').

    Regression test for the inverted sign fixed 2026-09-07.
    """
    base = _run(BackBCConfig(bc_type=BackBCType.PRESCRIBED_FLUX, q_back=0.0),
                t_end=200.0, T_surf=300.0)
    hot = _run(BackBCConfig(bc_type=BackBCType.PRESCRIBED_FLUX, q_back=5000.0),
               t_end=200.0, T_surf=300.0)
    cold = _run(BackBCConfig(bc_type=BackBCType.PRESCRIBED_FLUX, q_back=-5000.0),
                t_end=200.0, T_surf=300.0)
    # Direction is the point: +q_back heats, -q_back cools, symmetrically.
    assert hot.T[-1] > base.T[-1] + 25.0
    assert cold.T[-1] < base.T[-1] - 25.0
    assert (hot.T[-1] - base.T[-1]) == pytest.approx(base.T[-1] - cold.T[-1], rel=1e-6)
