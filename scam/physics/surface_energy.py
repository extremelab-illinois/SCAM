# SPDX-License-Identifier: MIT
"""Surface energy balance (SEB).

The surface energy balance closes the energy equation at the ablating surface.
For the full ENERGY_BALANCE mode, the SEB residual is:

    f(T_w) = q_conv_eff(T_w)           [convective + blowing correction]
           + q_rad_in(T_w)             [incoming radiation]
           - q_rad_out(T_w)            [surface emission]
           + q_chem(T_w)               [chemical heat flux from ablation products]
           - (m_dot_char + m_dot_g) * h_w(T_w)  [mass removal enthalpy]
           - q_cond(T_w)               [heat conducted into material]
           = 0

where q_cond is obtained from the F_cond relationship:
    q_cond = alpha_F * T_w + beta_F

The Newton iterate on T_w uses a finite-difference Jacobian.

Heat flux sign convention:
    Positive = into the material from the environment.
    q_cond positive = heat flows from surface INTO the solid.
"""

from __future__ import annotations

from scam.core.constants import SIGMA_SB
from scam.config.boundary import SurfaceBCConfig, SurfaceBCType, eval_bc
from scam.physics.blowing import blowing_corrected_alpha
from scam.physics.chemistry import lookup_b_prime
from scam.physics.gas_enthalpy import pyrolysis_gas_enthalpy
from scam.physics.properties import enthalpy_char, specific_heat
from scam.physics.stanton import solve_kays_blowing_factor, solve_lees_blowing_factor


def radiation_out(T_w: float, emissivity: float, view_factor: float) -> float:
    """Surface emitted radiation [W/m^2] (positive = away from surface)."""
    return emissivity * view_factor * SIGMA_SB * T_w ** 4


def radiation_in(T_rad: float, emissivity: float, view_factor: float) -> float:
    """Incoming radiation absorbed by the surface [W/m^2].

    T_rad = 0 → no incoming radiation.
    """
    if T_rad <= 0.0:
        return 0.0
    return emissivity * view_factor * SIGMA_SB * T_rad ** 4


def convective_flux(alpha_conv: float, T_aw: float, T_w: float) -> float:
    """Net convective heat flux to the surface [W/m^2].

    q_conv = alpha_conv * (T_aw - T_w)
    Positive when T_aw > T_w (aerodynamic heating).
    """
    return alpha_conv * (T_aw - T_w)


def seb_residual(
    T_w: float,
    alpha_F: float,
    beta_F: float,
    bc: SurfaceBCConfig,
    time: float,
    m_dot_pyro: float,
    mat_surface,            # MaterialCard
    b_prime_table,          # BPrimeTable | BprimeEvaluator | None
    emissivity_override: float = -1.0,
    Z_C_pyro: float | None = None,
    rho_surface: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute the SEB residual f(T_w) and return breakdown.

    Parameters
    ----------
    T_w:
        Trial surface temperature [K].
    alpha_F, beta_F:
        F_cond linear coefficients: q_cond = alpha_F * T_w + beta_F.
    bc:
        Surface boundary condition configuration.
    time:
        Current simulation time [s] (for time-varying BCs).
    m_dot_pyro:
        Pyrolysis gas mass flux at surface [kg/m^2/s].
    mat_surface:
        MaterialCard for the surface layer.
    b_prime_table:
        Loaded BPrimeTable or None (no ablation if None).
    emissivity_override:
        If > 0, use this emissivity instead of the material card value.

    Returns
    -------
    (residual, q_cond, m_dot_char, h_wall)
        residual: SEB residual [W/m^2] — zero when balanced.
        q_cond:   conductive flux into material [W/m^2].
        m_dot_char: char erosion flux [kg/m^2/s].
        h_wall:   wall enthalpy [J/kg].
    """
    if emissivity_override > 0:
        eps = emissivity_override
    else:
        # Temperature- and char-fraction-dependent surface emissivity.  Uses the
        # material's emissivity(T) tables when present, otherwise the scalar
        # virgin/char values, blended by the local char fraction.
        from scam.physics.properties import surface_emissivity as _surf_eps
        rho_surf = rho_surface if rho_surface is not None else mat_surface.rho_char
        eps = float(_surf_eps(mat_surface, T_w, rho_surf))
    vf  = bc.view_factor
    T_rad     = eval_bc(bc.T_rad_in, time)
    rho_e_u_e = eval_bc(bc.rho_e_u_e, time)
    C_M       = eval_bc(bc.C_M, time)
    p_e       = eval_bc(bc.p_e, time)
    rhoUeCH   = eval_bc(bc.rhoUeCH, time)
    denom_blow = rho_e_u_e * C_M if (rho_e_u_e > 0 and C_M > 0) else rhoUeCH

    # Lees' implicit blowing correction is solved repeatedly at nearby states:
    # finite-difference Newton points, Picard iterations, and adjacent timesteps.
    # Reuse the last converged factor as the next fixed-point initial guess.
    # A wall-correction callback depends on the initial unblown h_wall, so retain
    # the historical Ω=1 initialization for that less common path.
    lees_warm_start = (
        bc.blowing_model == "lees"
        and bc.stanton_wall_correction is None
        and b_prime_table is not None
        and rho_e_u_e > 0.0
        and C_M > 0.0
        and denom_blow > 0.0
    )
    lees_initial_factor = 1.0
    if lees_warm_start:
        cached_factor = getattr(b_prime_table, "_scam_lees_blow_factor", 1.0)
        if 0.0 < cached_factor <= 1.0:
            lees_initial_factor = float(cached_factor)

    # Kinetic (Kemp 1968) closure takes unconditional precedence when present:
    # a reaction-rate-limited ablator's mass flux is a pure function of T_w,
    # not of the boundary-layer transport variables (rho_e_u_e, C_M), so it is
    # NOT gated on them the way the B'-table lookup below is.  material_loader
    # rejects cards that specify both kinetic_ablation and b_prime_table, so
    # this precedence is enforced at load time, not silently resolved here.
    kinetic = getattr(mat_surface, "kinetic_ablation", None)
    if kinetic is not None:
        from scam.physics.kinetic_ablation import (
            m_dot_kemp, h_ablation_total, h_ablation_reaction,
        )
        from scam.physics.properties import thermal_conductivity as _thermal_conductivity
        rho_sw = kinetic.rho_sw_override if kinetic.rho_sw_override else mat_surface.rho_char
        rho_surf_for_k = rho_surface if rho_surface is not None else mat_surface.rho_char
        k_w = float(_thermal_conductivity(mat_surface, T_w, rho_surf_for_k))
        h_tot = h_ablation_total(T_w, kinetic.h_ablation_total_coeffs)
        m_dot_char_unblown = m_dot_kemp(T_w, rho_sw, k_w, kinetic.B, kinetic.E_a, h_tot)
        # Reaction-only enthalpy (NOT h_ablation_total) for the SEB mass-removal
        # term: SCAM's own resolved in-depth FVM conduction (rho*h(T)
        # formulation) already accounts for the wall node's actual sensible
        # heating history, so subtracting the full h_ablation_total here would
        # double-count it.  See KineticAblationCard docstring.
        h_wall = h_ablation_reaction(T_w, kinetic.h_ablation_reaction_coeffs)
        _bprime_ran = False
        _kinetic_ran = True
    # B' table lookup for char ablation.  The table returns the *unblown*
    # dimensionless char ablation rate B'_c; the dimensional flux is
    # B'_c · (rho_e_u_e · C_M).
    elif b_prime_table is not None and rho_e_u_e > 0 and C_M > 0:
        if lees_warm_start:
            B_g_initial = max(0.0, m_dot_pyro) / (
                denom_blow * lees_initial_factor
            )
            B_c_initial, h_wall = b_prime_table.lookup(
                T_w, p_e, B_g_initial, Z_C_pyro,
            )
            m_dot_char_unblown = max(0.0, B_c_initial * denom_blow)
        else:
            m_dot_char_unblown, _, h_wall = lookup_b_prime(
                b_prime_table, T_w, p_e, m_dot_pyro, rho_e_u_e, C_M, Z_C_pyro,
            )
        _bprime_ran = True   # h_wall is the equilibrium wall enthalpy (B' reference)
        _kinetic_ran = False
    else:
        m_dot_char_unblown = 0.0
        h_wall = float(enthalpy_char(mat_surface, T_w))
        _bprime_ran = False  # h_wall is the char-table enthalpy (different reference)
        _kinetic_ran = False

    wall_factor = 1.0
    if _bprime_ran and bc.stanton_wall_correction is not None:
        wall_factor = max(
            0.0,
            float(bc.stanton_wall_correction(T_w, time, h_wall)),
        )
        m_dot_char_unblown *= wall_factor
    # stanton_wall_correction is a B'-table-specific hot-wall correction; no
    # equivalent exists for the kinetic closure today (_kinetic_ran left as-is).

    # Blowing correction.  In the CMA/B' formulation the blowing reduction of
    # the transfer coefficient applies to BOTH the convective heat flux AND the
    # char/pyrolysis mass fluxes (Reynolds analogy: C_H and C_M are reduced by
    # the same blowing factor).  SCAM previously applied it only to q_conv,
    # leaving m_dot_char (and hence recession) over-predicted.  The blowing
    # parameter B_total is formed on the unblown basis (= B'_c + B'_g) so the
    # algebraic correction stays bounded even under heavy injection.
    m_dot_total_blow = max(0.0, m_dot_char_unblown) + max(0.0, m_dot_pyro)
    B_total = m_dot_total_blow / denom_blow if denom_blow > 0 else 0.0
    lambda_blowing = eval_bc(bc.lambda_blowing, time)
    if bc.blowing_model == "kays":
        blow_factor = solve_kays_blowing_factor(
            m_dot_char_unblown,
            m_dot_pyro,
            rhoUeCH,
            lambda_blowing,
        )
    elif bc.blowing_model == "rational":
        blow_factor = 1.0 / (1.0 + lambda_blowing * B_total)
    elif bc.blowing_model == "lees":
        # PATO-consistent (constantLambdaBlowingCorrectionModel): B'_g supplied to
        # the B' table on the BLOWN basis (B'_g_blown = m_dot_g / (rhoUeCH * Ω)),
        # and the Lees log correction Ω = log(1+Φ)/Φ, Φ = 2λ(B'_g_blown + B'_c).
        # Inner iteration: start with Ω=1 (unblown), re-lookup B'_c, update Ω.
        # Typically converges in < 10 steps; updates m_dot_char_unblown and h_wall.
        if _bprime_ran and denom_blow > 0:
            blow_factor = lees_initial_factor
            # The B' lookup immediately above already evaluated the first
            # fixed-point state at the chosen initial factor.  Reuse it instead
            # of repeating the same expensive live-equilibrium lookup.
            hw_iter = h_wall
            for iteration in range(20):
                Bg_blown = max(0.0, m_dot_pyro) / (denom_blow * blow_factor)
                if iteration == 0:
                    mdc_iter = m_dot_char_unblown
                else:
                    Bc_iter, hw_iter = b_prime_table.lookup(
                        T_w, p_e, Bg_blown, Z_C_pyro,
                    )
                    mdc_iter = max(0.0, Bc_iter * denom_blow) * wall_factor
                new_bf = solve_lees_blowing_factor(
                    mdc_iter, m_dot_pyro, rhoUeCH, lambda_blowing,
                )
                m_dot_char_unblown = mdc_iter
                h_wall = hw_iter
                converged = abs(new_bf - blow_factor) < 1.0e-10
                blow_factor = new_bf
                if converged:
                    break
            if lees_warm_start and 0.0 < blow_factor <= 1.0:
                setattr(
                    b_prime_table,
                    "_scam_lees_blow_factor",
                    float(blow_factor),
                )
        elif _kinetic_ran:
            # m_dot_char_unblown never changes with blowing state for the
            # kinetic closure (it depends only on T_w), so a single
            # non-iterative call suffices — mirrors the "kays" branch above.
            # solve_lees_blowing_factor self-guards on rhoUeCH <= 0.
            blow_factor = solve_lees_blowing_factor(
                m_dot_char_unblown, m_dot_pyro, rhoUeCH, lambda_blowing,
            )
        else:
            blow_factor = 1.0
    else:
        raise ValueError(f"unknown blowing_model: {bc.blowing_model!r}")
    # Design decision: blow_factor throttles the boundary-layer MASS TRANSFER
    # coefficient (Reynolds analogy C_H=C_M), which is the correct picture for
    # the B'-table (mass-transfer-limited) closure but NOT for the kinetic
    # closure — Kemp's mass flux is set by solid-state Arrhenius kinetics, and
    # gas escape is never the rate-limiting step (the whole physical point of
    # PTFE's high-vapor-pressure depolymerization product), so it must not be
    # further throttled by a mass-transfer coefficient.  blow_factor is still
    # fully computed above (from B_total formed with this same
    # m_dot_char_unblown) and remains used below for the real, independent
    # transpiration-cooling reduction of q_conv.
    m_dot_char = m_dot_char_unblown if _kinetic_ran else m_dot_char_unblown * blow_factor

    if rhoUeCH > 0.0 and (_bprime_ran or _kinetic_ran or b_prime_table is None):
        h_r_val = eval_bc(bc.h_r, time)
        heat_wall_factor = wall_factor if bc.apply_wall_correction_to_heat else 1.0
        rhoUeCH_eff = rhoUeCH * heat_wall_factor * blow_factor
        if _bprime_ran:
            # CMA formulation: q_conv = rhoUeCH_eff * (h_r - h_wall)
            # h_wall from B' table is the equilibrium wall gas enthalpy on the same
            # absolute reference as h_r.  The mass removal enthalpy is implicit:
            # expanding gives rhoUeCH*h_r + (m_dot_total - rhoUeCH)*h_wall.
            q_conv = rhoUeCH_eff * (h_r_val - h_wall)
            q_mass_removal = 0.0   # already included in q_conv via h_wall
        else:
            # No B' backend: use pyrolysis gas enthalpy from h_g_table
            # (standalone enthalpy-BC / chemistry-off style).  For the kinetic
            # closure, the mass-removal energy is charged explicitly below
            # (h_wall here is the reaction-only enthalpy, not the wall gas
            # enthalpy — h_gas_tw plays the h_r-reference role instead).
            h_gas_tw = float(pyrolysis_gas_enthalpy(mat_surface, T_w))
            q_conv = rhoUeCH_eff * (h_r_val - h_gas_tw)
            m_dot_total = m_dot_char + m_dot_pyro
            q_mass_removal = (m_dot_total * h_wall) if _kinetic_ran else 0.0
    else:
        alpha_conv_0 = eval_bc(bc.alpha_conv, time)
        T_aw         = eval_bc(bc.T_aw, time)
        # Blowing parameter formed on the unblown basis (consistent with B_total).
        alpha_eff    = blowing_corrected_alpha(
            alpha_conv_0, m_dot_char_unblown, m_dot_pyro,
            rho_e_u_e, C_M, lambda_blowing,
        )
        q_conv = convective_flux(alpha_eff, T_aw, T_w)
        # Mass removal carries the actual (blowing-reduced) ablation flux.  Apply
        # it only when (a) a convective coefficient is actually driving the surface
        # AND (b) the B' lookup or kinetic closure actually ran.  PATO's Bprime BC
        # with chemistryOn=0 ignores rhoUeCH/h_r and uses the temperature form
        # hconv*(Tedge-T) with no advective or mass-removal accounting.
        m_dot_total = m_dot_char + m_dot_pyro
        q_mass_removal = (m_dot_total * h_wall
                          if (alpha_conv_0 > 0.0 and (_bprime_ran or _kinetic_ran))
                          else 0.0)

    # Advective enthalpy of the ablating mass — PATO's qAdvPyro + qAdvChar.
    # The pyrolysis gas and char arrive at the wall with enthalpy h_g / h_c and
    # leave as equilibrated wall gas at h_wall; the differences are a net surface
    # energy source (dominated by the char-oxidation release h_c − h_wall).  PATO
    # includes these explicitly in its `Bprime` BC; SCAM's q_conv term alone
    # omits them.  Added only when the chemistry backend can supply h_g / h_c on
    # the SAME reference as h_wall.  This includes the live Cantera evaluator and
    # B' tables generated with the optional pretabulated surface-enthalpy arrays.
    # IMPORTANT: only when the B' lookup actually ran is `h_wall` the equilibrium
    # wall enthalpy on the SAME reference as h_g/h_c.  When it is skipped (e.g.
    # during cooldown, rho_e_u_e=0), `h_wall` is the char-table enthalpy (a
    # different reference), so `h_g − h_wall` would be dominated by the reference
    # offset and inject a spurious cooling sink — gate q_adv on `_bprime_ran`.
    q_adv = 0.0
    if _bprime_ran and hasattr(b_prime_table, "surface_enthalpies"):
        try:
            h_g_w, h_c_w = b_prime_table.surface_enthalpies(T_w, p_e, Z_C_pyro)
            # h_g: prefer the material's h_g_table when available (PATO-consistent
            # equilibrium enthalpy on the Cantera absolute reference — same as h_wall).
            # The material's h_g_table may be on the SENSIBLE reference (h_g(298K)≈0).
            # If the card supplies h_g_abs_offset, apply it to convert to absolute:
            #   h_g_abs(T) = h_g_sensible(T) + h_g_abs_offset   (≈ -7.09 MJ/kg for TACOT)
            # Without the offset the q_adv_pyro term would mix sensible h_g with
            # absolute h_wall, giving a spuriously large (~9 MJ/kg) driving enthalpy.
            # With a (p, T) gasProperties table, h_g is evaluated at
            # (T_w, p_e) exactly like PATO's boundary h_g field; the table is
            # already absolute-reference (no offset re-application).
            if (getattr(mat_surface, "gas_properties_pT", None) is not None
                    or getattr(mat_surface, "h_g_table", None) is not None):
                from scam.physics.gas_enthalpy import pyrolysis_gas_enthalpy_abs
                h_g_w = pyrolysis_gas_enthalpy_abs(mat_surface, T_w, p=p_e)
            q_adv = (max(0.0, m_dot_pyro) * (h_g_w - h_wall)
                     + max(0.0, m_dot_char) * (h_c_w - h_wall))
        except Exception:
            q_adv = 0.0

    # Radiation
    q_rad_in  = radiation_in(T_rad, eps, vf)
    q_rad_out = radiation_out(T_w, eps, vf)
    q_cond    = alpha_F * T_w + beta_F   # from F_cond back-substitution

    residual = q_conv + q_adv + q_rad_in - q_rad_out - q_mass_removal - q_cond

    return float(residual), float(q_cond), float(m_dot_char), float(h_wall)
