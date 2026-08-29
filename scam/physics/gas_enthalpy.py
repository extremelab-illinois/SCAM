# SPDX-License-Identifier: MIT
"""Pyrolysis gas specific enthalpy h_g(T) and its temperature derivative.

h_g is tabulated in the material card as [[T_K, h_g_J_kg], ...] and
interpolated linearly at runtime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.config.material import MaterialCard
from scam.physics.properties import interp_table, interp_gas_pT


def pyrolysis_gas_enthalpy(mat: MaterialCard, T: float | NDArray) -> float | NDArray:
    """Specific enthalpy of the pyrolysis gas [J/kg] at temperature T [K]."""
    return interp_table(mat.h_g_table, T)


def pyrolysis_gas_enthalpy_abs(
    mat: MaterialCard,
    T: float | NDArray,
    p: float | NDArray | None = None,
) -> float | NDArray:
    """Absolute-reference pyrolysis gas enthalpy h_g_abs [J/kg].

    Resolution order:

    * ``mat.gas_properties_pT`` present → bilinear h_g(T, p) from the PATO
      gasProperties table, which is ALREADY on the absolute (formation)
      reference — ``h_g_abs_offset`` must NOT be applied on top (the
      double-application trap, see pato_validation.md §11).  When ``p`` is
      None, the card's ambient ``gas_pressure`` is used (the same convention
      the 1D gas-storage and Darcy ρ_g terms use for pressure).
    * else → sensible ``h_g_table`` + ``h_g_abs_offset`` (legacy path,
      bit-identical to the pre-stage-2 behaviour).
    """
    pT = getattr(mat, "gas_properties_pT", None)
    if pT is not None:
        p_eval = mat.gas_pressure if p is None else p
        return interp_gas_pT(pT, "h_g", T, p_eval)
    h = pyrolysis_gas_enthalpy(mat, T)
    offset = getattr(mat, "h_g_abs_offset", None)
    if offset is not None:
        h = h + float(offset)
    return h


def pyrolysis_gas_enthalpy_deriv(mat: MaterialCard, T: float) -> float:
    """d(h_g)/dT [J/kg/K] via finite difference on the h_g table.

    Used in the tridiagonal coefficient assembly when pyrolysis gas
    enthalpy transport is included in the implicit stencil.
    """
    dT = 1.0  # 1 K finite-difference step
    h_plus  = float(pyrolysis_gas_enthalpy(mat, T + dT))
    h_minus = float(pyrolysis_gas_enthalpy(mat, T - dT))
    return (h_plus - h_minus) / (2.0 * dT)
