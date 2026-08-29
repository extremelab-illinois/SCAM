# SPDX-License-Identifier: MIT
"""Material card dataclasses.

A MaterialCard fully describes one material layer — its density, thermal properties,
decomposing components, and optional B' table reference.

Property tables are stored as shape-(N, 2) numpy arrays [[T_0, v_0], [T_1, v_1], ...]
with T in [K] and v in SI units. Interpolation is always linear; edge values are used
for out-of-range queries (flat extrapolation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ComponentCard:
    """One decomposing constituent of a charring material.

    Arrhenius decomposition rate (per unit volume of composite):
        drho_i/dt = -A_rate * exp(-E_act / (R*T)) * rho_0 * ((rho_i - rho_r) / rho_0)^m_exp

    The irreversibility constraint  rho_r <= rho_i <= rho_i_old  is enforced by the
    decomposition solver.
    """

    name: str
    rho_0: float      # initial (virgin) apparent density [kg/m^3]
    rho_r: float      # residual (char) apparent density [kg/m^3]; must satisfy rho_r < rho_0
    A_rate: float     # Arrhenius pre-exponential [1/s]
    E_act: float      # activation energy [J/mol]
    m_exp: float      # reaction order exponent [-]
    h_decomp: float = 0.0  # decomposition enthalpy [J/kg]; endothermic (absorbs heat) > 0


@dataclass
class BPrimeTableRef:
    """Reference to an external B' table file (YAML format).

    The table maps  (T_wall [K], p_e [Pa], B'_g [-])  →  (B'_c [-], h_wall [J/kg]).
    The file path is resolved relative to the material card's directory.
    """

    path: str  # path to B' table YAML, relative to the material card file


@dataclass
class KineticAblationCard:
    """Kemp (1968) Eq. 12 quasi-steady in-depth Arrhenius decomposition-wave
    closure for reaction-rate-limited (kinetic) surface ablation:

        m_so^2 = B * rho_sw * k(T_w) * (R*T_w^2/E_a) * exp(-E_a/(R*T_w))
                 / h_ablation_total(T_w)

    m_dot is a pure function of T_w and material constants only — NOT of the
    boundary-layer mass-transfer coefficient (rho_e*u_e*C_M).  This is the
    opposite closure regime from BPrimeTableRef (phase-equilibrium,
    transport-limited): PTFE's depolymerization product has such a high vapor
    pressure that mass transfer through the boundary layer never limits the
    rate — only the solid-state Arrhenius kinetics do.  See
    studies/teflon_ablation/ (compare_teflon_kinetic_models.py and
    why_ptfe_mdot_depends_on_Tw_not_p.md) for the full derivation and
    validation against three independent literature sources (Kemp/Steg/
    Yurevich agree within +-6% at 1000 K).

    k(T_w) and rho_sw are NOT duplicated here — they come from the
    surrounding MaterialCard's own k_virgin_table/k_char_table (via
    thermal_conductivity()) and rho_char, which must be calibrated
    consistently with B/E_a/h_ablation_* below (Kemp's own k(T) and rho_sw,
    not a generic handbook table) for the closed-form validation to hold.
    """

    B: float          # pre-exponential [1/s]
    E_a: float        # activation energy [J/mol]

    # Full heat of ablation h_gw(T_w) - h_so(T_w) [J/kg] = a + b*T + c*T^2:
    # sensible T0->T_w heating + depolymerization energy.  Used ONLY inside
    # the Eq. 12 mass-flux formula above (the denominator).
    h_ablation_total_coeffs: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Depolymerization-ONLY reaction enthalpy [J/kg] = a + b*T.  Used for the
    # SEB's mass-removal energy term instead of h_ablation_total, because
    # SCAM's own resolved in-depth FVM conduction (rho*h(T) formulation)
    # already accounts for the wall node's actual sensible heating history —
    # subtracting the full h_ablation_total there would double-count it.
    h_ablation_reaction_coeffs: tuple[float, float] = (0.0, 0.0)

    # Optional override of the reacting/wall density if it differs from
    # mat.rho_char (default: None -> use mat.rho_char).
    rho_sw_override: Optional[float] = None


@dataclass
class MaterialCard:
    """Complete thermophysical and chemical description of one material.

    Mixture rule for density-blended properties (linear blending with virgin fraction):
        eps_virgin = (rho - rho_char) / (rho_virgin - rho_char)
        k(T, rho)  = eps_virgin * k_virgin(T) + (1 - eps_virgin) * k_char(T)
        cp(T, rho) = [eps_virgin * rho_virgin * cp_virgin(T) + ...] / rho

    For non-decomposing (inert) layers set decomposing=False and leave components empty.
    """

    name: str
    rho_virgin: float           # total virgin density [kg/m^3]
    rho_char: float             # total char density [kg/m^3]
    gamma_resin: float          # resin volume fraction in virgin composite [-]

    # Decomposing constituents — arbitrary count; must be ordered (fastest first by convention)
    components: list = field(default_factory=list)  # list[ComponentCard]

    # Thermal conductivity tables [[T_K, k_W_mK], ...] for virgin and char phases
    k_virgin_table: np.ndarray = field(default_factory=lambda: np.array([[300.0, 0.3], [3000.0, 0.3]]))
    k_char_table: np.ndarray   = field(default_factory=lambda: np.array([[300.0, 1.0], [3000.0, 1.0]]))

    # In-plane conductivity tables (stored for future anisotropy support; unused by 1-D solver)
    k_virgin_ip_table: Optional[np.ndarray] = None
    k_char_ip_table:   Optional[np.ndarray] = None

    # Specific heat tables [[T_K, cp_J_kgK], ...] for virgin and char phases
    cp_virgin_table: np.ndarray = field(default_factory=lambda: np.array([[300.0, 1000.0], [3000.0, 1000.0]]))
    cp_char_table: np.ndarray   = field(default_factory=lambda: np.array([[300.0, 1200.0], [3000.0, 1200.0]]))

    # Pyrolysis gas specific enthalpy table [[T_K, h_g_J_kg], ...]
    h_g_table: np.ndarray = field(default_factory=lambda: np.array([[300.0, 0.0], [3000.0, 3.0e6]]))

    # Optional absolute enthalpy tables [[T_K, h_J_kg], ...] for solid phases.
    # When provided, enables the NASA Apollo h_bar pyrolysis energy source
    # instead of the per-component h_decomp approach.
    h_virgin_table: Optional[np.ndarray] = None
    h_char_table: Optional[np.ndarray] = None

    # Sensible enthalpy tables [[T_K, h_J_kg], ...] integrated from cp tables
    # (∫₀ᵀ cp dT).  Populated automatically by material_loader.py; used for
    # the ρ·h(T) energy storage RHS in the tridiagonal assembly.  Leave None
    # when constructing MaterialCard directly — the assembly will build them
    # on-the-fly via build_sensible_enthalpy_table().
    h_virgin_sensible: Optional[np.ndarray] = None
    h_char_sensible: Optional[np.ndarray] = None

    # Gas-phase porosity for energy storage term d(eps_g*rho_g*h_g)/dt.
    # Set both to zero (default) to skip the correction entirely.
    eps_g_virgin: float = 0.0        # gas porosity of virgin material [-]
    eps_g_char:   float = 0.0        # gas porosity of fully charred material [-]
    gas_molar_mass: float = 0.022    # pyrolysis gas molar mass [kg/mol]
    gas_molar_mass_table: Optional[np.ndarray] = None  # optional [[T_K, M_kg_mol], ...]
    gas_viscosity_table: Optional[np.ndarray] = None   # optional [[T_K, mu_Pa_s], ...]
    gas_pressure:   float = 101325.0 # ambient gas pressure for ideal-gas rho_g [Pa]
    # Optional full (p, T) equilibrium pyrolysis-gas property tables (PATO
    # gasProperties format).  Dict with 1-D "p" [Pa] (ascending), 1-D "T" [K]
    # (ascending), and 2-D (P, N) arrays "M" [kg/mol], "h_g" [J/kg, absolute
    # reference], "mu" [Pa·s].  When present, the 2D gas transport/energy path
    # interpolates bilinearly in (p, T) like PATO's Tabulated GasProperties;
    # the 1-D T-only tables above remain the fallback (and the 1D solver path).
    gas_properties_pT: Optional[dict] = None

    # Surface emissivity (used when this is the surface layer).
    # emissivity_char: if >= 0, the emissivity is blended linearly with the
    # char fraction so that emissivity → emissivity_char when fully charred.
    # Negative (default) means use the same emissivity for both phases.
    emissivity: float = 0.85
    emissivity_char: float = -1.0

    # Optional temperature-dependent emissivity tables [[T_K, emissivity], ...]
    # for the virgin and char phases.  When provided, these take precedence over
    # the scalar emissivity / emissivity_char above: ε is interpolated at the
    # wall temperature for each phase and then blended by the local char fraction
    # (see physics/properties.py::surface_emissivity).  Leave None to use the
    # scalar values.  emissivity_char_table falls back to emissivity_virgin_table
    # if only the virgin table is supplied.
    emissivity_virgin_table: Optional[np.ndarray] = None
    emissivity_char_table: Optional[np.ndarray] = None

    # When True the gas energy storage term d(ε_g·ρ_g·h_g)/dt is absorbed
    # implicitly into an effective cp correction in specific_heat().  Default
    # is now False (matches PATO's explicit formulation).  Retained for backward
    # compatibility; will be removed in a future version.
    gas_storage_implicit: bool = False

    # Darcy permeability [m²].  Used by the pressure-driven Darcy solver.
    # Set to 0.0 (default) to skip the pressure equation and use the simpler
    # thermal-expansion-only approximation in darcy_flow.py.
    permeability: float = 0.0        # char (or single-value) permeability
    permeability_virgin: float = 0.0 # virgin permeability; blended with permeability by char fraction

    # Klinkenberg slip correction: K_app = K * (1 + klinkenberg_b / p).
    # Set to 0.0 (default) to disable.  Significant only at sub-atmospheric pressures
    # (p < ~1 kPa for typical ablators).  For TACOT the value below is an estimate
    # derived from Kozeny-Carman pore-size analysis (d_pore ≈ 1 µm, see docs).
    klinkenberg_b: float = 0.0

    # Set False for inert layers (no mass equation, no decomposition subgrid)
    decomposing: bool = True

    # B' table reference (None for non-ablating materials)
    b_prime_ref: Optional[BPrimeTableRef] = None

    # Kemp (1968) kinetic ablation closure (None for non-ablating materials or
    # materials using the B'-table equilibrium closure instead).  Mutually
    # exclusive with b_prime_ref — material_loader.py enforces this at load
    # time and seb_residual() gives kinetic_ablation unconditional precedence
    # when both would otherwise be present.
    kinetic_ablation: Optional[KineticAblationCard] = None

    # Elemental mass fractions [C, H, O, N] of the pyrolysis gas vs temperature.
    # Shape (4, n_T, 2): axis-0 = element index (C=0,H=1,O=2,N=3), each entry is
    # [[T0, Z0], [T1, Z1], ...].  Single-row tables give a temperature-independent
    # composition.  None → element transport disabled for this layer.
    pyro_elem_fracs: Optional[np.ndarray] = None

    # Initial elemental mass fractions [C, H, O, N] for the in-material gas phase.
    # Used by material_response.py to initialise Z_elem when element_transport=True.
    # None → initialise to ambient air (Z_ELEM_AIR).
    initial_Z_elem: Optional[np.ndarray] = None

    # Effective element diffusivity in the porous medium [m²/s].
    # Applied uniformly to all four elements; divided by tortuosity below.
    element_diffusivity: float = 1.0e-5

    # Pore tortuosity factor [-].  Effective diffusivity = element_diffusivity / tortuosity.
    tortuosity: float = 1.0

    # Physical category tag from the material card (metadata only, not used by solver).
    # Values: subsurface | ablative_carbon | ablative_silica | ablative_organic |
    #         ablative_silicone | ablative_hybrid
    material_category: Optional[str] = None

    # Absolute-reference offset for the pyrolysis gas enthalpy.
    # When the material's h_g_table is on the SENSIBLE reference (h_g(298K)≈0),
    # this offset converts it to the Cantera/NASA-9 absolute reference so that
    # q_adv_pyro = m_dot_pyro * (h_g_abs(T_w) - h_wall) in the SEB is
    # consistent with h_wall from the B' lookup:
    #   h_g_abs(T) = h_g_sensible(T) + h_g_abs_offset
    # Derived from the formation enthalpy of the nominal pyrolysis gas at 298 K
    # (Cantera NASA-9 reference).  Leave None for materials that already store
    # the absolute-reference h_g in h_g_table, or when q_adv is not used.
    h_g_abs_offset: Optional[float] = None

    # Versioning and quality metadata (not used by solver).
    # dataset_version: upstream source release (e.g. "3.0" for TACOT 3.0).
    #   Omitted when no single authoritative version exists.
    # card_version: SCAM-internal revision, incremented when the card changes.
    # status: verified | provisional | estimate
    dataset_version: Optional[str] = None
    card_version: str = "1.0"
    status: str = "provisional"
