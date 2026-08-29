# SPDX-License-Identifier: MIT
"""Kemp (1968) Eq. 12 quasi-steady in-depth Arrhenius decomposition-wave
mass-flux closure for reaction-rate-limited (kinetic) surface ablation.

Unlike the B'-table (CMA/equilibrium) closure, this mass flux is a pure
function of surface temperature and material constants — it does NOT depend
on the boundary-layer mass-transfer coefficient (rho_e*u_e*C_M).  See
studies/teflon_ablation/compare_teflon_kinetic_models.py for the validated
literature derivation (Kemp/Steg/Yurevich agree within +-6% at 1000 K) and
studies/teflon_ablation/why_ptfe_mdot_depends_on_Tw_not_p.md for the physics
rationale (PTFE's depolymerization product has such a high vapor pressure
that boundary-layer mass transfer never limits the rate).
"""

from __future__ import annotations

import math

R_UNIV = 8.314462  # J/mol/K


def h_ablation_total(T_w: float, coeffs: tuple[float, float, float]) -> float:
    """Full heat of ablation h_gw(T_w) - h_so(T_w) [J/kg]: a + b*T + c*T^2.

    Sensible T0->T_w heating + depolymerization energy.  Used only inside
    m_dot_kemp's denominator.
    """
    a, b, c = coeffs
    return a + b * T_w + c * T_w * T_w


def h_ablation_reaction(T_w: float, coeffs: tuple[float, float]) -> float:
    """Depolymerization-only reaction enthalpy [J/kg]: a + b*T.

    Used for the SEB mass-removal energy term (see KineticAblationCard
    docstring for why this differs from h_ablation_total).
    """
    a, b = coeffs
    return a + b * T_w


def m_dot_kemp(
    T_w: float,
    rho_sw: float,
    k_w: float,
    B: float,
    E_a: float,
    h_ablation_total_val: float,
) -> float:
    """Kemp (1968) Eq. 12 mass flux [kg/m^2/s].

        m_so^2 = B * rho_sw * k_w * (R*T_w^2/E_a) * exp(-E_a/(R*T_w))
                 / h_ablation_total_val

    Returns 0 for non-physical inputs (T_w <= 0 or h_ablation_total_val <= 0).
    """
    if T_w <= 0.0 or h_ablation_total_val <= 0.0 or rho_sw <= 0.0 or k_w <= 0.0:
        return 0.0
    num = (
        B * rho_sw * k_w * (R_UNIV * T_w * T_w / E_a)
        * math.exp(-E_a / (R_UNIV * T_w))
    )
    return math.sqrt(max(0.0, num / h_ablation_total_val))
