# SPDX-License-Identifier: MIT
"""Shared builders and analytical references for the SCAM verification ladder.

The verification suite (``tests/verification/``) exercises the *whole solver*
(``scam.solvers.material_response.run``) in rungs of increasing physical
complexity, each checked against a closed-form or independently-computed
reference:

    V1  bare conduction, constant properties      → erfc / linear analytical
    V2  conduction, temperature-dependent k, cp    → Kirchhoff steady profile
    V3  multi-material stack (+ contact resistance) → series thermal resistance
    V4  in-depth decomposition / pyrolysis          → invariants + mass/energy
    V5  coupled surface mass & energy balance        → SEB closure + audits

These helpers exist only to assemble inputs and evaluate references; the solver
itself is never re-implemented here.
"""
from __future__ import annotations

import numpy as np

from scam.config.boundary import BackBCConfig, SurfaceBCConfig
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import StackConfig
from scam.solvers.material_response import run


# ---------------------------------------------------------------------------
# Material builders
# ---------------------------------------------------------------------------

def make_inert_material(
    name: str = "Inert",
    k: float = 1.0,
    rho: float = 180.0,
    cp: float = 710.0,
    emissivity: float = 0.85,
) -> MaterialCard:
    """Inert (non-decomposing) material with constant k and cp.

    Constant properties are encoded as a two-point table with identical values
    at the table extremes, so the linear interpolation in
    :mod:`scam.physics.properties` returns ``k``/``cp`` for any temperature.
    Virgin and char densities are set equal so the virgin-fraction blend is a
    no-op (the material is single-phase).
    """
    T_pts = np.array([200.0, 4000.0])
    k_tab = np.column_stack([T_pts, np.full(2, k)])
    cp_tab = np.column_stack([T_pts, np.full(2, cp)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name=name, rho_virgin=rho, rho_char=rho, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=emissivity, decomposing=False,
    )


def make_linear_k_material(
    name: str = "VarK",
    k0: float = 0.5,
    k1: float = 2.5,
    T0: float = 300.0,
    T1: float = 3000.0,
    rho: float = 180.0,
    cp: float = 710.0,
) -> MaterialCard:
    """Inert material with a thermal conductivity that varies linearly in T.

    ``k(T) = k0 + (k1 - k0) * (T - T0) / (T1 - T0)`` (flat-extrapolated outside
    [T0, T1]).  cp is constant.  Used by V2 to verify the Kirchhoff-transform
    steady-state profile.
    """
    T_pts = np.array([T0, T1])
    k_tab = np.column_stack([T_pts, np.array([k0, k1])])
    cp_tab = np.column_stack([T_pts, np.full(2, cp)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name=name, rho_virgin=rho, rho_char=rho, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


def k_of_T_linear(T, k0: float, k1: float, T0: float, T1: float):
    """Reference evaluation of the linear-in-T conductivity (flat extrapolation)."""
    return np.interp(T, [T0, T1], [k0, k1])


# ---------------------------------------------------------------------------
# Run wrapper
# ---------------------------------------------------------------------------

def run_case(
    stack: StackConfig,
    mat_cards: dict,
    surface_bc: SurfaceBCConfig,
    back_bc: BackBCConfig,
    options: SolverOptions,
    *,
    b_prime_tables: dict | None = None,
    geom: GeometryConfig | None = None,
    initial_T: float = 300.0,
):
    """Thin wrapper around :func:`scam.solvers.material_response.run`.

    Defaults to a slab geometry and no B' tables (the common verification case).
    """
    if geom is None:
        geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    if b_prime_tables is None:
        b_prime_tables = {name: None for name in mat_cards}
    return run(
        stack, mat_cards, b_prime_tables, geom, surface_bc, back_bc, options,
        initial_T=initial_T, verbose=False,
    )


# ---------------------------------------------------------------------------
# Analytical conduction references
# ---------------------------------------------------------------------------

def semi_infinite_flux_T(y, t: float, Q0: float, k: float, alpha: float, T0: float):
    """Semi-infinite slab, constant surface flux Q0 (Carslaw & Jaeger).

        T(y,t) = T0 + (2 Q0 / k) sqrt(alpha t / pi) exp(-xi^2)
                    - (Q0 y / k) erfc(xi),   xi = y / (2 sqrt(alpha t))
    """
    from scipy.special import erfc

    sqrt_at = np.sqrt(alpha * t)
    xi = y / (2.0 * sqrt_at)
    dT = (2.0 * Q0 / k * sqrt_at / np.sqrt(np.pi) * np.exp(-xi**2)
          - Q0 * y / k * erfc(xi))
    return T0 + dT


def semi_infinite_temp_T(y, t: float, T_wall: float, T0: float, alpha: float):
    """Semi-infinite slab, step surface temperature T_wall.

        T(y,t) = T0 + (T_wall - T0) erfc(y / (2 sqrt(alpha t)))
    """
    from scipy.special import erfc

    xi = y / (2.0 * np.sqrt(alpha * t))
    return T0 + (T_wall - T0) * erfc(xi)


def fourier_slab_adiabatic_T(
    y,
    t: float,
    T_wall: float,
    T0: float,
    alpha: float,
    S: float,
    n_terms: int = 500,
):
    """Finite slab: prescribed T_wall at y=0, adiabatic at y=S.

    Exact Fourier-series solution (Carslaw & Jaeger, §3.3):

        T(y,t) = T0 + (T_wall − T0) · [1 − (4/π) Σ_{n=1,3,5,...}
                 (1/n) exp(−(nπ/2)² α t / S²) sin(nπ y / (2 S))]

    The sum runs over odd integers n = 1, 3, 5, …  Terms are dropped once
    the exponential falls below 1e-15 (negligible contribution for any y).
    With n_terms=500 the series is converged to machine precision for all t
    including very early times (Fo = αt/S² as small as 10⁻⁴).
    """
    y = np.asarray(y, dtype=float)
    series = np.zeros_like(y)
    for k in range(n_terms):
        n = 2 * k + 1                                  # odd: 1, 3, 5, …
        exp_coeff = np.exp(-(n * np.pi / 2.0) ** 2 * alpha * t / S ** 2)
        if exp_coeff < 1e-15:
            break
        series += (1.0 / n) * exp_coeff * np.sin(n * np.pi * y / (2.0 * S))
    return T0 + (T_wall - T0) * (1.0 - (4.0 / np.pi) * series)


def finite_slab_steady_linear(y, L: float, T_front: float, T_back: float):
    """Steady-state linear profile across a slab of thickness L with fixed ends."""
    return T_front + (T_back - T_front) * (y / L)
