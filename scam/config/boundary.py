# SPDX-License-Identifier: MIT
"""Boundary condition configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Union


# A scalar BC parameter can be a constant float or a callable f(t) -> float.
ScalarBC = Union[float, Callable[[float], float]]
StantonWallCorrection = Callable[[float, float, float], float]


class SurfaceBCType(Enum):
    ENERGY_BALANCE = auto()    # full SEB with B' table (primary mode)
    PRESCRIBED_FLUX = auto()   # net heat flux q_net(t) into material [W/m^2]
    PRESCRIBED_TEMP = auto()   # surface temperature T_wall(t) [K]


class BackBCType(Enum):
    ADIABATIC = auto()         # zero heat flux (insulated back face)
    PRESCRIBED_TEMP = auto()   # fixed or time-varying temperature [K]
    PRESCRIBED_FLUX = auto()   # fixed or time-varying flux [W/m^2] (+ into material)


@dataclass
class SurfaceBCConfig:
    """Surface boundary condition configuration.

    For ENERGY_BALANCE the solver iterates Newton on T_wall using:
        q_conv(T_w) + q_rad_in - q_rad_out(T_w) + q_chem(T_w)
        - (m_dot_char + m_dot_g)*h_w(T_w) - F_cond(T_w) = 0

    Two convective heat flux modes are supported:

    Temperature-based (default, alpha_conv + T_aw):
        q_conv = alpha_conv * (T_aw - T_w)

    Enthalpy-based (PATO-style, rhoUeCH + h_r):
        q_conv = rhoUeCH * (h_r - h_gas(T_w))
        where h_gas(T_w) is looked up from the surface material's h_g_table.
        Active when rhoUeCH > 0; alpha_conv / T_aw are then ignored.
        In the B' chemistry branch, rhoUeCH/h_r instead couple to the B'
        wall enthalpy, and when that chemistry branch is inactive the solver
        falls back to the temperature-based alpha_conv/T_aw form.

    All scalar parameters may be floats (constant) or callables f(t) -> float.
    """

    bc_type: SurfaceBCType = SurfaceBCType.ENERGY_BALANCE

    # --- temperature-based convection (default) ---
    # Convective heat transfer coefficient [W/m^2/K]
    alpha_conv: ScalarBC = 0.0
    # Adiabatic wall (recovery) temperature [K]
    T_aw: ScalarBC = 300.0

    # --- enthalpy-based convection (PATO rhoUeCH / h_r style) ---
    # Lumped mass-heat-transfer coefficient rho_e*u_e*C_H [kg/m^2/s].
    # When > 0 this mode overrides alpha_conv / T_aw.
    rhoUeCH: ScalarBC = 0.0
    # Freestream recovery enthalpy h_r [J/kg] (same reference as h_g_table).
    h_r: ScalarBC = 0.0

    # Surface emissivity (overrides material card value if set > 0)
    emissivity: float = -1.0   # -1 means "use material card value"
    view_factor: float = 1.0
    # Incoming radiation source temperature [K] (e.g. enclosure wall or flame)
    T_rad_in: ScalarBC = 0.0

    # Blowing correction parameters
    lambda_blowing: ScalarBC = 0.5  # 0.5 laminar, 0.4 turbulent
    # Blowing correction model:
    #   "rational" — 1/(1 + lambda*B_total), SCAM legacy (Amar-style).
    #   "kays"     — Phi/(exp(Phi)-1), implicit solve for char flux (Amar Eq. 42).
    #   "lees"     — log(1+Phi)/Phi, PATO's constantLambdaBlowingCorrectionModel;
    #                also uses B'_g on the blown basis for the B' table lookup
    #                (matching PATO BprimeBoundaryConditions.C line 613).
    blowing_model: str = "rational"
    # Optional hot-wall Stanton correction Omega_hw(T_wall, time, h_wall).
    # Applied to both C_H and C_M, preserving the configured C_H/C_M ratio.
    stanton_wall_correction: Optional[StantonWallCorrection] = None
    # Set false when rhoUeCH already contains the hot-wall correction but the
    # B'-to-mass-transfer relation still needs it (legacy trajectory inputs).
    apply_wall_correction_to_heat: bool = True
    rho_e_u_e: ScalarBC = 0.0     # freestream mass flux [kg/m^2/s]
    C_M: ScalarBC = 0.0            # mass transfer Stanton number [-]

    # Edge pressure [Pa] for B' table lookup
    p_e: ScalarBC = 101325.0

    # Prescribed-mode overrides (used when bc_type != ENERGY_BALANCE)
    q_prescribed: Optional[Callable[[float], float]] = None   # [W/m^2] net flux
    T_prescribed: Optional[Callable[[float], float]] = None   # [K]


@dataclass
class BackBCConfig:
    """Back-face boundary condition configuration."""

    bc_type: BackBCType = BackBCType.ADIABATIC
    T_back: ScalarBC = 300.0    # [K], used for PRESCRIBED_TEMP
    q_back: ScalarBC = 0.0      # [W/m^2] into material, used for PRESCRIBED_FLUX


def eval_bc(param: ScalarBC, t: float) -> float:
    """Evaluate a scalar BC parameter at time *t*."""
    if callable(param):
        return float(param(t))
    return float(param)
