# SPDX-License-Identifier: MIT
"""Material property evaluation.

All properties are evaluated as linear interpolations of tabulated data.
Mixture properties blend virgin and char contributions using the local
virgin volume fraction eps_virgin.

Coordinate convention: y = 0 at original surface, increases into material.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.config.material import MaterialCard


# ---------------------------------------------------------------------------
# Table interpolation
# ---------------------------------------------------------------------------

def interp_table(table: np.ndarray, T: float | NDArray) -> float | NDArray:
    """Linear interpolation of a property table at temperature T.

    Parameters
    ----------
    table:
        Shape (N, 2) array [[T_0, v_0], ..., [T_N-1, v_N-1]] in SI units,
        with T values strictly increasing.
    T:
        Query temperature [K] (scalar or array).

    Returns
    -------
    Interpolated value (same type/shape as T).
    Flat extrapolation beyond table limits (edge value used).
    """
    T_tbl = table[:, 0]
    v_tbl = table[:, 1]
    return np.interp(T, T_tbl, v_tbl)


def interp_gas_pT(table: dict, key: str, T, p) -> NDArray:
    """Bilinear (p, T) interpolation into a PATO-style gasProperties table.

    ``table`` is ``MaterialCard.gas_properties_pT``: 1-D ascending "p" [Pa]
    and "T" [K] axes plus (P, N) value arrays keyed "M", "h_g", "mu".
    Linear in both axes with clamping at the table edges (PATO's Tabulated
    GasProperties is linear; out-of-range pressures are clamped rather than
    treated as fatal).
    """
    p_ax = table["p"]
    T_ax = table["T"]
    vals = table[key]                      # (P, N)
    T_s = np.asarray(T, dtype=np.float64)
    p_s = np.asarray(p, dtype=np.float64)
    T_s, p_s = np.broadcast_arrays(T_s, p_s)

    # T interpolation weights (clamped)
    iT = np.clip(np.searchsorted(T_ax, T_s) - 1, 0, len(T_ax) - 2)
    wT = np.clip((T_s - T_ax[iT]) / (T_ax[iT + 1] - T_ax[iT]), 0.0, 1.0)
    # p interpolation weights (clamped)
    ip = np.clip(np.searchsorted(p_ax, p_s) - 1, 0, len(p_ax) - 2)
    wp = np.clip((p_s - p_ax[ip]) / (p_ax[ip + 1] - p_ax[ip]), 0.0, 1.0)

    v00 = vals[ip, iT]
    v01 = vals[ip, iT + 1]
    v10 = vals[ip + 1, iT]
    v11 = vals[ip + 1, iT + 1]
    return ((1 - wp) * ((1 - wT) * v00 + wT * v01)
            + wp * ((1 - wT) * v10 + wT * v11))


# ---------------------------------------------------------------------------
# Virgin fraction
# ---------------------------------------------------------------------------

def eps_virgin(mat: MaterialCard, rho: float | NDArray) -> float | NDArray:
    """Local virgin volume fraction.

    eps_v = (rho - rho_char) / (rho_virgin - rho_char)

    Clipped to [0, 1] to handle numerical round-off.
    """
    drho = mat.rho_virgin - mat.rho_char
    if drho == 0.0:
        return np.zeros_like(rho, dtype=float)
    result = (np.asarray(rho, dtype=float) - mat.rho_char) / drho
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Surface emissivity  eps(T, rho)
# ---------------------------------------------------------------------------

def surface_emissivity(mat: MaterialCard, T: float, rho_surface: float) -> float:
    """Surface emissivity blended between phases at temperature T.

    Resolution order for each phase value:
      * if the temperature-dependent table is present, interpolate it at T;
      * else use the scalar (mat.emissivity for virgin, mat.emissivity_char for
        char, falling back to the virgin value when emissivity_char < 0).

    The virgin and char phase emissivities are then blended by the local virgin
    volume fraction (eps_virgin): ε = ε_char + (ε_virgin − ε_char)·frac_virgin,
    so a fully charred surface (frac_virgin = 0) emits at the char value.
    """
    # Virgin-phase emissivity at T
    if mat.emissivity_virgin_table is not None:
        eps_v = float(interp_table(mat.emissivity_virgin_table, T))
    else:
        eps_v = float(mat.emissivity)

    # Char-phase emissivity at T
    if mat.emissivity_char_table is not None:
        eps_c = float(interp_table(mat.emissivity_char_table, T))
    elif mat.emissivity_virgin_table is not None:
        # Virgin table supplied but no char table → use the virgin table for both.
        eps_c = eps_v
    elif mat.emissivity_char >= 0.0:
        eps_c = float(mat.emissivity_char)
    else:
        # No char distinction at all → single emissivity.
        return eps_v

    frac_virgin = float(eps_virgin(mat, rho_surface))
    return eps_c + (eps_v - eps_c) * frac_virgin


# ---------------------------------------------------------------------------
# Thermal conductivity  k(T, rho)
# ---------------------------------------------------------------------------

def thermal_conductivity(
    mat: MaterialCard,
    T: float | NDArray,
    rho: float | NDArray,
) -> float | NDArray:
    """Mixture thermal conductivity [W/m/K].

    Linear blend of virgin and char values:
        k = eps_v * k_virgin(T) + (1 - eps_v) * k_char(T)
    """
    ev = eps_virgin(mat, rho)
    k_v = interp_table(mat.k_virgin_table, T)
    k_c = interp_table(mat.k_char_table,   T)
    return ev * k_v + (1.0 - ev) * k_c


# ---------------------------------------------------------------------------
# Specific heat  cp(T, rho)
# ---------------------------------------------------------------------------

def _gas_storage_cp_correction(
    mat: MaterialCard,
    T: float | NDArray,
    rho: float | NDArray,
) -> float | NDArray:
    """Effective cp increment from d(eps_g * rho_g * h_g)/dt gas energy storage.

    Linearises PATO's EnergyType Pyrolysis gas storage term as an additive
    correction to the solid cp:
        delta_cp = (eps_g / rho_s) * rho_g(T) * (dh_g/dT - h_g/T)
    where rho_g = p * M / (R * T)  (ideal gas).

    Returns zero (scalar) when eps_g_virgin == eps_g_char == 0 (default).
    """
    if mat.eps_g_virgin == 0.0 and mat.eps_g_char == 0.0:
        return 0.0

    T_arr   = np.asarray(T,   dtype=float)
    rho_arr = np.asarray(rho, dtype=float)

    ev    = eps_virgin(mat, rho_arr)
    eps_g = mat.eps_g_char + (mat.eps_g_virgin - mat.eps_g_char) * ev

    R_univ = 8.314  # J/(mol·K)
    T_safe = np.maximum(T_arr, 1.0)
    rho_g  = mat.gas_pressure * mat.gas_molar_mass / (R_univ * T_safe)

    h_g  = interp_table(mat.h_g_table, T_arr)
    dT   = 1.0
    cp_g = (interp_table(mat.h_g_table, T_arr + dT)
            - interp_table(mat.h_g_table, T_arr - dT)) / (2.0 * dT)

    safe_rho = np.where(rho_arr > 0.0, rho_arr, 1.0)
    return (eps_g / safe_rho) * rho_g * (cp_g - h_g / T_safe)


def specific_heat(
    mat: MaterialCard,
    T: float | NDArray,
    rho: float | NDArray,
) -> float | NDArray:
    """Mixture specific heat [J/kg/K].

    Mass-fraction-weighted blend:
        cp = (1/rho) * [eps_v*rho_v*cp_v(T) + (1-eps_v)*rho_c*cp_c(T)]

    Falls back to cp_char if rho ≈ 0 (fully ablated cells).
    Includes gas energy storage correction when mat.eps_g_char/virgin are set.
    """
    rho_arr = np.asarray(rho, dtype=float)
    ev = eps_virgin(mat, rho_arr)
    cp_v = interp_table(mat.cp_virgin_table, T)
    cp_c = interp_table(mat.cp_char_table,   T)
    numerator = ev * mat.rho_virgin * cp_v + (1.0 - ev) * mat.rho_char * cp_c
    safe_rho = np.where(rho_arr > 0.0, rho_arr, 1.0)
    cp_solid = np.where(rho_arr > 0.0, numerator / safe_rho, cp_c)
    if not mat.gas_storage_implicit:
        return cp_solid
    return cp_solid + _gas_storage_cp_correction(mat, T, rho_arr)


# ---------------------------------------------------------------------------
# Enthalpy h(T) for a phase (integral of cp from 0 to T)
# ---------------------------------------------------------------------------

def build_sensible_enthalpy_table(cp_table: np.ndarray) -> np.ndarray:
    """Integrate cp(T) from 0 K to each table point using the trapezoid rule.

    Returns a shape-(n, 2) array [[T_0, h_0], ..., [T_n-1, h_n-1]] where
    h(T) = ∫₀ᵀ cp(T') dT'.  Below the first table temperature, cp is assumed
    constant (same convention as _enthalpy_from_table).
    """
    T_pts = cp_table[:, 0]
    cp_pts = cp_table[:, 1]
    n = len(T_pts)
    h = np.empty(n)
    h[0] = cp_pts[0] * T_pts[0]  # ∫₀^{T₀} cp₀ dT
    for i in range(1, n):
        dT = T_pts[i] - T_pts[i - 1]
        h[i] = h[i - 1] + 0.5 * (cp_pts[i - 1] + cp_pts[i]) * dT
    return np.column_stack([T_pts, h])


def _enthalpy_from_table(table: np.ndarray, T: float) -> float:
    """Numerically integrate cp(T') dT' from 0 to T using the cp table.

    For T below the first table entry, assumes cp = table[0, 1] (constant).
    """
    T_tbl = table[:, 0]
    cp_tbl = table[:, 1]
    T = float(T)

    if T <= T_tbl[0]:
        return cp_tbl[0] * T

    # Sum segments up to T
    h = cp_tbl[0] * T_tbl[0]  # integrate from 0 to T_tbl[0] at constant cp
    for i in range(len(T_tbl) - 1):
        T_lo, T_hi = T_tbl[i], T_tbl[i + 1]
        cp_lo, cp_hi = cp_tbl[i], cp_tbl[i + 1]
        if T <= T_lo:
            break
        T_hi_clipped = min(T, T_hi)
        # Trapezoid rule
        dT = T_hi_clipped - T_lo
        frac = (T_hi_clipped - T_lo) / (T_hi - T_lo) if T_hi > T_lo else 1.0
        cp_hi_clipped = cp_lo + frac * (cp_hi - cp_lo)
        h += 0.5 * (cp_lo + cp_hi_clipped) * dT
        if T <= T_hi:
            break
    return h


def enthalpy_virgin(mat: MaterialCard, T: float) -> float:
    """Specific enthalpy of the virgin phase h_v(T) [J/kg]."""
    return _enthalpy_from_table(mat.cp_virgin_table, T)


def enthalpy_char(mat: MaterialCard, T: float) -> float:
    """Specific enthalpy of the char phase h_c(T) [J/kg]."""
    return _enthalpy_from_table(mat.cp_char_table, T)


def enthalpy_mixture(mat: MaterialCard, T: float, rho: float) -> float:
    """Mixture specific enthalpy [J/kg].

    h = (1/rho) * [rho_v_component * h_v(T) + rho_c_component * h_c(T)]
    """
    ev = float(eps_virgin(mat, rho))
    if rho <= 0:
        return enthalpy_char(mat, T)
    return (ev * mat.rho_virgin * enthalpy_virgin(mat, T)
            + (1.0 - ev) * mat.rho_char * enthalpy_char(mat, T)) / rho


# ---------------------------------------------------------------------------
# Mixture sensible enthalpy for ρ·h(T) energy storage RHS
# ---------------------------------------------------------------------------

def _interp_linear_extrap(
    T_q: NDArray,
    h_tbl: np.ndarray,
    cp_tbl: np.ndarray,
) -> NDArray:
    """Interpolate h(T) from a sensible enthalpy table with linear extrapolation.

    np.interp flat-extrapolates beyond table bounds; for h(T) = ∫cp dT this is
    wrong — it must extrapolate linearly using the boundary cp value so that the
    Picard correction M·T^k − ρ·h(T^k)·A·Δ/dt remains invariant to T^k.
    """
    h = np.interp(T_q, h_tbl[:, 0], h_tbl[:, 1])
    T_lo, T_hi = h_tbl[0, 0], h_tbl[-1, 0]
    h_lo, h_hi = h_tbl[0, 1], h_tbl[-1, 1]
    cp_lo = float(np.interp(T_lo, cp_tbl[:, 0], cp_tbl[:, 1]))
    cp_hi = float(np.interp(T_hi, cp_tbl[:, 0], cp_tbl[:, 1]))
    above = T_q > T_hi
    if np.any(above):
        h[above] = h_hi + cp_hi * (T_q[above] - T_hi)
    below = T_q < T_lo
    if np.any(below):
        h[below] = h_lo + cp_lo * (T_q[below] - T_lo)
    return h


def mixture_enthalpy_array(
    mats: list,
    T: NDArray,
    rho: NDArray,
    layer_id: NDArray,
) -> NDArray:
    """Node-wise mixture sensible enthalpy h(T, rho) [J/kg].

    h = (1/rho) * [eps_v * rho_v * h_v(T) + (1-eps_v) * rho_c * h_c(T)]

    Uses mat.h_virgin_sensible / h_char_sensible (precomputed by loader from
    cp tables: ∫₀ᵀ cp dT).  Builds the tables on the fly when absent (e.g.
    MaterialCard constructed directly in tests).
    """
    h = np.empty_like(T, dtype=float)
    for _lid in np.unique(layer_id):
        mat = mats[int(_lid)]
        mask = layer_id == _lid

        h_v_tbl = (mat.h_virgin_sensible if mat.h_virgin_sensible is not None
                   else build_sensible_enthalpy_table(mat.cp_virgin_table))
        h_c_tbl = (mat.h_char_sensible if mat.h_char_sensible is not None
                   else build_sensible_enthalpy_table(mat.cp_char_table))

        T_m   = T[mask]
        rho_m = rho[mask]
        h_v   = _interp_linear_extrap(T_m, h_v_tbl, mat.cp_virgin_table)
        h_c   = _interp_linear_extrap(T_m, h_c_tbl, mat.cp_char_table)
        ev    = eps_virgin(mat, rho_m)
        safe  = np.where(rho_m > 0.0, rho_m, 1.0)
        h[mask] = np.where(
            rho_m > 0.0,
            (ev * mat.rho_virgin * h_v + (1.0 - ev) * mat.rho_char * h_c) / safe,
            h_c,
        )
    return h


# ---------------------------------------------------------------------------
# Convenience: per-node property arrays
# ---------------------------------------------------------------------------

def eval_k_array(
    mats: list,
    T: NDArray,
    rho: NDArray,
    layer_id: NDArray,
) -> NDArray:
    """Evaluate k(T_n, rho_n) at every node, using per-layer material."""
    k = np.empty_like(T)
    for lid in np.unique(layer_id):
        mask = layer_id == lid
        k[mask] = thermal_conductivity(mats[int(lid)], T[mask], rho[mask])
    return k


def eval_cp_array(
    mats: list,
    T: NDArray,
    rho: NDArray,
    layer_id: NDArray,
) -> NDArray:
    """Evaluate cp(T_n, rho_n) at every node, using per-layer material."""
    cp = np.empty_like(T)
    for lid in np.unique(layer_id):
        mask = layer_id == lid
        cp[mask] = specific_heat(mats[int(lid)], T[mask], rho[mask])
    return cp


def h_bar(mat: "MaterialCard", T: float) -> float:
    """Blended solid enthalpy for NASA Apollo h_bar pyrolysis energy source.

    h_bar = (rho_v * h_v(T) - rho_c * h_c(T)) / (rho_v - rho_c)

    Returns 0.0 if the material does not have enthalpy tables.
    """
    if mat.h_virgin_table is None or mat.h_char_table is None:
        return 0.0
    h_v = float(np.interp(T, mat.h_virgin_table[:, 0], mat.h_virgin_table[:, 1]))
    h_c = float(np.interp(T, mat.h_char_table[:, 0], mat.h_char_table[:, 1]))
    denom = mat.rho_virgin - mat.rho_char
    if abs(denom) < 1e-12:
        return h_v
    return (mat.rho_virgin * h_v - mat.rho_char * h_c) / denom


def h_bar_array(
    mats: list,
    T: NDArray,
    layer_id: NDArray,
) -> NDArray:
    """Evaluate h_bar(T_n) at every node, using per-layer material."""
    hb = np.zeros_like(T, dtype=float)
    for lid in np.unique(layer_id):
        mat = mats[int(lid)]
        if mat.h_virgin_table is None or mat.h_char_table is None:
            continue
        mask = layer_id == lid
        h_v = np.interp(T[mask], mat.h_virgin_table[:, 0], mat.h_virgin_table[:, 1])
        h_c = np.interp(T[mask], mat.h_char_table[:, 0], mat.h_char_table[:, 1])
        denom = mat.rho_virgin - mat.rho_char
        if abs(denom) < 1e-12:
            hb[mask] = h_v
        else:
            hb[mask] = (mat.rho_virgin * h_v - mat.rho_char * h_c) / denom
    return hb


def h_bar_sensible_array(
    mats: list,
    T: NDArray,
    layer_id: NDArray,
) -> NDArray:
    """Sensible part of h_bar: (ρ_v·h_v_sens(T) − ρ_c·h_c_sens(T)) / (ρ_v − ρ_c).

    Computed from cp-integrated sensible enthalpy tables (h_virgin_sensible /
    h_char_sensible), so it has the same 0 K reference as mixture_enthalpy_array.
    Returns 0 for materials without distinct virgin/char phases (ρ_v == ρ_c).
    """
    hb = np.zeros_like(T, dtype=float)
    for _lid in np.unique(layer_id):
        mat = mats[int(_lid)]
        denom = mat.rho_virgin - mat.rho_char
        if abs(denom) < 1e-12:
            continue
        mask = layer_id == _lid
        h_v_tbl = (mat.h_virgin_sensible if mat.h_virgin_sensible is not None
                   else build_sensible_enthalpy_table(mat.cp_virgin_table))
        h_c_tbl = (mat.h_char_sensible if mat.h_char_sensible is not None
                   else build_sensible_enthalpy_table(mat.cp_char_table))
        h_v = _interp_linear_extrap(T[mask], h_v_tbl, mat.cp_virgin_table)
        h_c = _interp_linear_extrap(T[mask], h_c_tbl, mat.cp_char_table)
        hb[mask] = (mat.rho_virgin * h_v - mat.rho_char * h_c) / denom
    return hb


def h_bar_chemical_array(
    mats: list,
    T: NDArray,
    layer_id: NDArray,
) -> NDArray:
    """Chemical (formation-enthalpy) part of h_bar for materials with explicit tables.

    h_bar_chemical = h_bar_absolute(T) − h_bar_sensible(T)

    h_bar_absolute uses the explicit h_virgin_table / h_char_table (absolute
    enthalpies including formation energies).  h_bar_sensible is the cp-integral
    piece captured implicitly by the ρ_old·h(T^n, ρ_old) storage change.
    The difference is the chemical energy that must still be supplied explicitly
    in Q_vol to close the energy balance.

    Returns 0 for materials without explicit absolute enthalpy tables.
    """
    hb = np.zeros_like(T, dtype=float)
    for _lid in np.unique(layer_id):
        mat = mats[int(_lid)]
        if mat.h_virgin_table is None or mat.h_char_table is None:
            continue
        mask = layer_id == _lid
        T_m = T[mask]
        # Absolute h_bar from explicit enthalpy tables
        h_v_abs = np.interp(T_m, mat.h_virgin_table[:, 0], mat.h_virgin_table[:, 1])
        h_c_abs = np.interp(T_m, mat.h_char_table[:, 0], mat.h_char_table[:, 1])
        denom = mat.rho_virgin - mat.rho_char
        if abs(denom) < 1e-12:
            h_bar_abs = h_v_abs
        else:
            h_bar_abs = (mat.rho_virgin * h_v_abs - mat.rho_char * h_c_abs) / denom
        # Sensible h_bar from cp-integrated tables
        h_v_tbl = (mat.h_virgin_sensible if mat.h_virgin_sensible is not None
                   else build_sensible_enthalpy_table(mat.cp_virgin_table))
        h_c_tbl = (mat.h_char_sensible if mat.h_char_sensible is not None
                   else build_sensible_enthalpy_table(mat.cp_char_table))
        h_v_sens = _interp_linear_extrap(T_m, h_v_tbl, mat.cp_virgin_table)
        h_c_sens = _interp_linear_extrap(T_m, h_c_tbl, mat.cp_char_table)
        if abs(denom) < 1e-12:
            h_bar_sens = h_v_sens
        else:
            h_bar_sens = (mat.rho_virgin * h_v_sens - mat.rho_char * h_c_sens) / denom
        hb[mask] = h_bar_abs - h_bar_sens
    return hb


# ---------------------------------------------------------------------------
# Per-component extent of reaction
# ---------------------------------------------------------------------------

def component_beta_array(
    mat: MaterialCard,
    rho_comp: NDArray,
) -> NDArray:
    """Extent of reaction β_i for each Arrhenius component, node-averaged.

    β_i = (ρ_i_0 - ρ_i) / (ρ_i_0 - ρ_i_r)

    where ρ_i_0 = component initial density, ρ_i_r = component residual density,
    ρ_i is the current nodelet density averaged over nodelets.

    Parameters
    ----------
    mat:
        MaterialCard for the layer.
    rho_comp:
        Shape (n_comp, n_nodes, n_nodelets) — per-component nodelet densities.

    Returns
    -------
    beta : ndarray, shape (n_comp, n_nodes)
        Extent of reaction per component per node, clipped to [0, 1].
    """
    n_comp, n_nodes, _ = rho_comp.shape
    beta = np.empty((n_comp, n_nodes))
    for ic, comp in enumerate(mat.components):
        drho = comp.rho_0 - comp.rho_r
        rho_node = rho_comp[ic].mean(axis=-1)   # average over nodelets → (n_nodes,)
        if drho == 0.0:
            beta[ic] = 0.0
        else:
            beta[ic] = np.clip((comp.rho_0 - rho_node) / drho, 0.0, 1.0)
    return beta


def bulk_beta_array(mat: MaterialCard, rho: NDArray) -> NDArray:
    """Bulk extent of reaction β = 1 - eps_virgin = (ρ_v - ρ) / (ρ_v - ρ_c).

    Parameters
    ----------
    mat:
        MaterialCard for the layer.
    rho:
        Shape (n_nodes,) — mixture density [kg/m³].

    Returns
    -------
    beta : ndarray, shape (n_nodes,), clipped to [0, 1].
    """
    return 1.0 - eps_virgin(mat, rho)
