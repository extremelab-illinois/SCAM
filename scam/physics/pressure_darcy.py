# SPDX-License-Identifier: MIT
"""Pressure-driven Darcy flow solver (quasi-steady pressure equation).

Solves the 1-D quasi-steady gas pressure equation across all material layers:

    d/dy ( Γ(y) · dp/dy ) = ε_g(y) / T(y) · dT/dt

where  Γ = K / μ_g  is the gas mobility [m² / (Pa·s)].

Boundary conditions:
    p(surface, y=0)  = p_surface   (Dirichlet — open to free stream)
    dp/dy(back, y=L) = 0           (Neumann — adiabatic/no-flow back wall)

The equation is quasi-steady because the pressure diffusion timescale
    τ_p ~ L² · μ · ε_g / (K · p₀)  ≈ 0.001 s
is orders of magnitude shorter than the thermal timescale (~seconds).

The resulting gas mass flux is:
    J_g(y) = −ρ_g · (K/μ) · dp/dy   [kg/(m²·s)]

which is used directly in pressure_darcy_energy_source() to replace the simpler
thermal-expansion continuity approach.

Effective permeability at each node
------------------------------------
K_eff is computed by _node_permeability() in three stages:

  1. Virgin/char blend: K_node = eps_v · K_virgin + (1 − eps_v) · K_char
     where eps_v = (ρ − ρ_char) / (ρ_virgin − ρ_char) is the virgin fraction.
     If permeability_virgin is not set (== 0), K_char is used for both phases.

  2. Klinkenberg slip correction (disabled when klinkenberg_b == 0):
        K_app = K_node · (1 + klinkenberg_b / p)
     Significant only at sub-atmospheric pressures (p < ~1 kPa for typical ablators).

Note on magnitude: for TACOT char (K=2×10⁻¹¹ m², μ≈3×10⁻⁵ Pa·s), the
pressure perturbation is typically Δp < 1 Pa (0.001% of atmospheric).  The
resulting gas energy flux is ~0.036% of the conductive flux — physically
correct but negligible for gap closure.  The solver is included for
completeness and for cases with higher permeability or faster heating.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.config.material import MaterialCard
from scam.core.state import MeshState
from scam.physics.properties import eps_virgin, interp_table, interp_gas_pT


# Gas viscosity: Sutherland's law for air-like pyrolysis gas
_MU_REF = 1.716e-5   # Pa·s  at T_REF
_T_REF  = 273.15     # K
_S_SUTH = 110.4      # K  (Sutherland constant)


def _gas_viscosity(T: float) -> float:
    """Gas viscosity via Sutherland's law [Pa·s]."""
    T = max(float(T), 50.0)
    return _MU_REF * (T / _T_REF) ** 1.5 * (_T_REF + _S_SUTH) / (T + _S_SUTH)


def _gas_viscosity_array(T: NDArray) -> NDArray:
    """Vectorized Sutherland viscosity with the scalar function's floor."""
    T_safe = np.maximum(np.asarray(T, dtype=float), 50.0)
    return (
        _MU_REF
        * (T_safe / _T_REF) ** 1.5
        * (_T_REF + _S_SUTH)
        / (T_safe + _S_SUTH)
    )


def _nodal_gas_viscosity(
    mat_list: list,
    layer_id: NDArray,
    T: NDArray,
    p: NDArray,
) -> NDArray:
    """Nodal pyrolysis-gas viscosity [Pa·s].

    Layers whose MaterialCard carries a full (p, T) gasProperties table
    (``gas_properties_pT``) use bilinear μ(T, p) like PATO's Tabulated
    GasProperties; all other layers keep the legacy Sutherland air value, so
    behaviour is bit-identical when no table is loaded.
    """
    mu = _gas_viscosity_array(T)
    for layer in np.unique(layer_id):
        mat = mat_list[int(layer)]
        pT = getattr(mat, "gas_properties_pT", None)
        if pT is not None:
            mask = layer_id == layer
            mu[mask] = interp_gas_pT(pT, "mu", T[mask], p[mask])
    return mu


def _nodal_gas_molar_mass(
    mat_list: list,
    layer_id: NDArray,
    T: NDArray,
    p: NDArray,
    material: dict[str, NDArray],
) -> NDArray:
    """Nodal pyrolysis-gas molar mass [kg/mol].

    (p, T) table when the layer carries one; else the layer's scalar
    ``gas_molar_mass`` (legacy behaviour).
    """
    M = material["gas_molar_mass"].copy()
    for layer in np.unique(layer_id):
        mat = mat_list[int(layer)]
        pT = getattr(mat, "gas_properties_pT", None)
        if pT is not None:
            mask = layer_id == layer
            M[mask] = interp_gas_pT(pT, "M", T[mask], p[mask])
    return M


def _node_permeability(mat: MaterialCard, rho_n: float, p_n: float) -> float:
    """Effective permeability at one node [m²].

    Blends virgin and char values by the local virgin fraction, then applies
    the Klinkenberg slip correction if klinkenberg_b > 0.

    Parameters
    ----------
    mat   : MaterialCard for this node's layer
    rho_n : local mixture density [kg/m³]
    p_n   : local gas pressure [Pa] (for Klinkenberg correction)
    """
    K_char = mat.permeability
    if K_char <= 0.0:
        return 0.0

    K_v = mat.permeability_virgin
    if K_v > 0.0 and mat.rho_virgin > mat.rho_char:
        ev = float(eps_virgin(mat, rho_n))
        K_node = ev * K_v + (1.0 - ev) * K_char
    else:
        K_node = K_char

    if mat.klinkenberg_b > 0.0 and p_n > 0.0:
        K_node *= 1.0 + mat.klinkenberg_b / p_n

    return K_node


def _node_material_arrays(
    mat_list: list,
    layer_id: NDArray,
) -> dict[str, NDArray]:
    """Expand the small per-layer material set into dense nodal arrays."""
    N = len(layer_id)
    arrays = {
        name: np.empty(N, dtype=float)
        for name in (
            "rho_virgin",
            "rho_char",
            "permeability",
            "permeability_virgin",
            "klinkenberg_b",
            "eps_g_virgin",
            "eps_g_char",
            "gas_pressure",
            "gas_molar_mass",
        )
    }
    for layer in np.unique(layer_id):
        mask = layer_id == layer
        mat = mat_list[int(layer)]
        for name, values in arrays.items():
            values[mask] = float(getattr(mat, name))
    return arrays


def _permeability_array(
    rho: NDArray,
    pressure: NDArray,
    material: dict[str, NDArray],
) -> NDArray:
    """Vectorized virgin/char permeability blend and Klinkenberg correction."""
    K_char = material["permeability"]
    K_virgin = material["permeability_virgin"]
    rho_virgin = material["rho_virgin"]
    rho_char = material["rho_char"]
    drho = rho_virgin - rho_char

    fraction = np.zeros_like(rho, dtype=float)
    valid_blend = (K_virgin > 0.0) & (drho != 0.0)
    np.divide(rho - rho_char, drho, out=fraction, where=valid_blend)
    fraction = np.clip(fraction, 0.0, 1.0)
    K = np.where(
        valid_blend,
        fraction * K_virgin + (1.0 - fraction) * K_char,
        K_char,
    )
    K = np.where(K_char > 0.0, K, 0.0)

    slip = material["klinkenberg_b"]
    use_slip = (slip > 0.0) & (pressure > 0.0) & (K > 0.0)
    correction = np.ones_like(K)
    np.divide(slip, pressure, out=correction, where=use_slip)
    correction[use_slip] += 1.0
    return K * correction


def solve_gas_pressure(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    dTdt: NDArray,
    p_surface: float = 101325.0,
) -> NDArray:
    """Solve the quasi-steady gas pressure field [Pa].

    Parameters
    ----------
    mat_list   : list[MaterialCard], one per layer
    mesh       : current MeshState
    T          : nodal temperatures [K], shape (N,)
    rho        : nodal mixture densities [kg/m³], shape (N,)
    dTdt       : ∂T/∂t estimate at each node [K/s], shape (N,)
    p_surface  : surface (Dirichlet) pressure [Pa]

    Returns
    -------
    p : shape (N,) [Pa]
        Nodal pressure field.  p[0] ≈ p_surface; p[N-1] ≈ p_back.
    """
    N = mesh.n_nodes_total
    lid = mesh.layer_id
    delta = mesh.delta_nodes

    material = _node_material_arrays(mat_list, lid)
    if not np.any(material["permeability"] > 0.0):
        return np.full(N, p_surface)

    # First pass: compute Gamma using p_surface as a seed for Klinkenberg and
    # the (p, T) viscosity lookup (pressure perturbations are small relative
    # to p_surface, so the seed error is negligible)
    p_seed = np.full(N, p_surface)
    K = _permeability_array(rho, p_seed, material)
    Gamma = K / _nodal_gas_viscosity(mat_list, lid, T, p_seed)

    # Γ_face: harmonic mean at cell faces; face[N] = 0 (no-flow back wall)
    Gamma_face = np.zeros(N + 1)
    Gamma_face[0] = Gamma[0]
    Gamma_sum = Gamma[:-1] + Gamma[1:]
    np.divide(
        2.0 * Gamma[:-1] * Gamma[1:],
        Gamma_sum,
        out=Gamma_face[1:N],
        where=Gamma_sum > 0.0,
    )

    # Source S[n] = ε_g(n) / T(n) · dT/dt(n).
    # Integrate from back to each face to get Γ_face · dp/dy.
    face_source_integral = np.zeros(N + 1)
    drho = material["rho_virgin"] - material["rho_char"]
    virgin_fraction = np.zeros(N)
    np.divide(
        rho - material["rho_char"],
        drho,
        out=virgin_fraction,
        where=drho != 0.0,
    )
    virgin_fraction = np.clip(virgin_fraction, 0.0, 1.0)
    eps_g = (
        material["eps_g_char"]
        + (material["eps_g_virgin"] - material["eps_g_char"]) * virgin_fraction
    )
    source_integral = eps_g / np.maximum(T, 1.0) * dTdt * delta
    face_source_integral[:N] = np.cumsum(source_integral[::-1])[::-1]

    # dp/dy at each face
    dpdy_face = np.zeros(N + 1)
    np.divide(
        face_source_integral[:N],
        Gamma_face[:N],
        out=dpdy_face[:N],
        where=Gamma_face[:N] > 0.0,
    )

    # Integrate from surface Dirichlet BC
    p = np.empty(N)
    p[0] = p_surface + dpdy_face[0] * (delta[0] * 0.5)
    if N > 1:
        p[1:] = p[0] + np.cumsum(dpdy_face[1:N] * delta[1:])

    return p


def _mass_flux_from_pressure(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    p: NDArray,
    p_surface: float = 101325.0,
) -> NDArray:
    """Gas mass flux from a solved pressure field [kg/(m²·s)].

    J_g(face) = −ρ_g · (K/μ) · dp/dy

    Positive values mean gas flows toward the surface (same sign as m_dot_g).

    Parameters
    ----------
    mat_list : list[MaterialCard]
    mesh     : current MeshState
    T        : nodal temperatures [K], shape (N,)
    rho      : nodal densities [kg/m³], shape (N,)
    p        : nodal pressure field [Pa], shape (N,)
    p_surface : surface Dirichlet pressure [Pa] (the same value the pressure
        field was solved with; the surface-face gradient is taken against it)

    Returns
    -------
    face_flux : shape (N+1,) [kg/(m²·s)]
    """
    N = mesh.n_nodes_total
    lid = mesh.layer_id
    delta = mesh.delta_nodes

    R_univ = 8.314
    material = _node_material_arrays(mat_list, lid)
    K = _permeability_array(rho, p, material)
    Gamma = K / _nodal_gas_viscosity(mat_list, lid, T, p)

    dp_dy = np.zeros(N)
    dp_dy[0] = (p[0] - p_surface) / (delta[0] * 0.5)
    if N > 1:
        dy_face = 0.5 * (delta[:-1] + delta[1:])
        np.divide(
            p[1:] - p[:-1],
            dy_face,
            out=dp_dy[1:],
            where=dy_face > 0.0,
        )

    T_face = T.copy()
    if N > 1:
        T_face[1:] = 0.5 * (T[:-1] + T[1:])

    # Gas density at faces.  Legacy (no pT table): scalar layer gas_pressure
    # and molar mass — rho_g ≈ p0·M/(R·T).  With a (p, T) gasProperties table
    # the local solved pressure and M(T, p) are used instead, which matters at
    # sub-atmospheric conditions where M deviates from the 1 atm value by
    # ~10 % and the pressure field departs from the ambient scalar.
    p_face = p.copy()
    if N > 1:
        p_face[1:] = 0.5 * (p[:-1] + p[1:])
    has_pT = np.zeros(N, dtype=bool)
    for layer in np.unique(lid):
        if getattr(mat_list[int(layer)], "gas_properties_pT", None) is not None:
            has_pT |= lid == layer
    p_rho = np.where(has_pT, p_face, material["gas_pressure"])
    M_face = _nodal_gas_molar_mass(mat_list, lid, T_face, p_face, material)
    rho_g_face = p_rho * M_face / (R_univ * np.maximum(T_face, 1.0))

    face_flux = np.zeros(N + 1)  # face[N] = 0 (back wall no-flow)
    face_flux[:N] = -rho_g_face * Gamma * dp_dy

    return face_flux


def pressure_driven_mass_flux(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    p: NDArray,
    p_surface: float = 101325.0,
) -> NDArray:
    """Gas mass flux from the solved pressure field [kg/(m²·s)].

    Wrapper around _mass_flux_from_pressure for backward compatibility.
    """
    return _mass_flux_from_pressure(mat_list, mesh, T, rho, p, p_surface)


def solve_pressure_and_flux(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    dTdt: NDArray,
    p_surface: float = 101325.0,
) -> tuple[NDArray, NDArray]:
    """Solve the pressure field and return both p and J_g in one call.

    Avoids the double pressure solve that would occur if solve_gas_pressure()
    and pressure_driven_mass_flux() were called separately.

    Parameters
    ----------
    mat_list  : list[MaterialCard], one per layer
    mesh      : current MeshState
    T         : nodal temperatures [K], shape (N,)
    rho       : nodal densities [kg/m³], shape (N,)
    dTdt      : ∂T/∂t estimate [K/s], shape (N,)
    p_surface : surface Dirichlet pressure [Pa]

    Returns
    -------
    p       : nodal pressure [Pa], shape (N,)
    face_flux : gas mass flux [kg/(m²·s)], shape (N+1,)
                Positive → toward surface.
    """
    p = solve_gas_pressure(mat_list, mesh, T, rho, dTdt, p_surface)
    face_flux = _mass_flux_from_pressure(mat_list, mesh, T, rho, p, p_surface)
    return p, face_flux


def pressure_darcy_energy_source(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    dTdt: NDArray,
    p_surface: float = 101325.0,
) -> NDArray:
    """Nodal energy source from pressure-driven Darcy gas advection [W/cell].

    Solves the pressure equation, computes gas mass flux, then applies upwind
    sensible-enthalpy advection to get the per-cell energy source.

    Returns
    -------
    Q_adv : shape (N,) [W]
        To be added to D[n] in the tridiagonal RHS.
    """
    N = mesh.n_nodes_total
    lid = mesh.layer_id

    material = _node_material_arrays(mat_list, lid)
    if not np.any(material["permeability"] > 0.0):
        return np.zeros(N)

    p, face_flux = solve_pressure_and_flux(mat_list, mesh, T, rho, dTdt, p_surface)

    # Sensible gas enthalpy for the expansion-flux transport.  With a (p, T)
    # gasProperties table: h_sens(T, p) = h_abs(T, p) − h0 with the same
    # constant h0 = h_g_table[0,1] + h_g_abs_offset used by the transport
    # reference split elsewhere (the constant cancels for the divergence-free
    # part of the expansion flux).  Legacy path unchanged otherwise.
    h_sens = np.empty(N)
    for layer in np.unique(lid):
        mask = lid == layer
        mat = mat_list[int(layer)]
        pT = getattr(mat, "gas_properties_pT", None)
        if pT is not None:
            offset = getattr(mat, "h_g_abs_offset", None)
            h0 = (float(mat.h_g_table[0, 1]) if mat.h_g_table is not None else 0.0) \
                 + (float(offset) if offset is not None else 0.0)
            h_sens[mask] = interp_gas_pT(pT, "h_g", T[mask], p[mask]) - h0
        else:
            h_sens[mask] = (
                interp_table(mat.h_g_table, T[mask]) - float(mat.h_g_table[0, 1])
            )

    h_in = h_sens.copy()
    if N > 1:
        h_in[:-1] = h_sens[1:]
    Q_adv = face_flux[1:] * h_in - face_flux[:-1] * h_sens
    valid_volume = mesh.area_nodes * mesh.delta_nodes > 0.0
    Q_adv = np.where(valid_volume, Q_adv, 0.0)

    return Q_adv
