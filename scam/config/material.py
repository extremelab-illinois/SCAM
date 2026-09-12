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
    rho_0: float      #: initial (virgin) apparent density [kg/m^3]
    rho_r: float      #: residual (char) apparent density [kg/m^3]; must satisfy ``rho_r < rho_0``
    A_rate: float     #: Arrhenius pre-exponential [1/s]
    E_act: float      #: activation energy [J/mol]
    m_exp: float      #: reaction order exponent [-]
    h_decomp: float = 0.0  #: decomposition enthalpy [J/kg]; endothermic (absorbs heat) is positive


@dataclass
class BPrimeTableRef:
    """Reference to an external B' table file (YAML format).

    The table maps  (T_wall [K], p_e [Pa], B'_g [-])  →  (B'_c [-], h_wall [J/kg]).
    The file path is resolved relative to the material card's directory.
    """

    path: str  #: path to the B' table YAML, relative to the material card file


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

    B: float          #: pre-exponential [1/s]
    E_a: float        #: activation energy [J/mol]

    #: Full heat of ablation ``h_gw(T_w) - h_so(T_w)`` [J/kg] = ``a + b*T +
    #: c*T^2``: sensible T0->T_w heating plus depolymerization energy. Used
    #: only inside the Eq. 12 mass-flux formula above (the denominator).
    h_ablation_total_coeffs: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Depolymerization-only reaction enthalpy [J/kg] = ``a + b*T``. Used
    #: for the SEB's mass-removal energy term instead of
    #: ``h_ablation_total``, because SCAM's own resolved in-depth FVM
    #: conduction (``rho*h(T)`` formulation) already accounts for the wall
    #: node's actual sensible heating history -- subtracting the full
    #: ``h_ablation_total`` there would double-count it.
    h_ablation_reaction_coeffs: tuple[float, float] = (0.0, 0.0)

    #: Optional override of the reacting/wall density if it differs from
    #: ``mat.rho_char`` (default: None -> use ``mat.rho_char``).
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
    rho_virgin: float           #: total virgin density [kg/m^3]
    rho_char: float             #: total char density [kg/m^3]
    gamma_resin: float          #: resin volume fraction in virgin composite [-]

    #: Decomposing constituents -- arbitrary count; must be ordered (fastest
    #: first, by convention). ``list[ComponentCard]``.
    components: list = field(default_factory=list)

    #: Thermal conductivity table [[T_K, k_W_mK], ...], virgin phase.
    k_virgin_table: np.ndarray = field(default_factory=lambda: np.array([[300.0, 0.3], [3000.0, 0.3]]))
    #: Thermal conductivity table [[T_K, k_W_mK], ...], char phase.
    k_char_table: np.ndarray   = field(default_factory=lambda: np.array([[300.0, 1.0], [3000.0, 1.0]]))

    #: In-plane conductivity table (stored for future anisotropy support;
    #: unused by the 1-D solver).
    k_virgin_ip_table: Optional[np.ndarray] = None
    k_char_ip_table:   Optional[np.ndarray] = None

    #: Specific heat table [[T_K, cp_J_kgK], ...], virgin phase.
    cp_virgin_table: np.ndarray = field(default_factory=lambda: np.array([[300.0, 1000.0], [3000.0, 1000.0]]))
    #: Specific heat table [[T_K, cp_J_kgK], ...], char phase.
    cp_char_table: np.ndarray   = field(default_factory=lambda: np.array([[300.0, 1200.0], [3000.0, 1200.0]]))

    #: Pyrolysis gas specific enthalpy table [[T_K, h_g_J_kg], ...].
    h_g_table: np.ndarray = field(default_factory=lambda: np.array([[300.0, 0.0], [3000.0, 3.0e6]]))

    #: Optional absolute enthalpy table [[T_K, h_J_kg], ...], virgin phase.
    #: When both this and ``h_char_table`` are provided, enables the NASA
    #: Apollo ``h_bar`` pyrolysis energy source instead of the
    #: per-component ``h_decomp`` approach.
    h_virgin_table: Optional[np.ndarray] = None
    h_char_table: Optional[np.ndarray] = None

    #: Sensible enthalpy table [[T_K, h_J_kg], ...], virgin phase, integrated
    #: from the ``cp`` table. Populated automatically by
    #: ``material_loader.py``; used for the ``rho*h(T)`` energy storage RHS
    #: in the tridiagonal assembly. Leave None when constructing
    #: ``MaterialCard`` directly -- the assembly builds it on the fly via
    #: ``build_sensible_enthalpy_table()``.
    h_virgin_sensible: Optional[np.ndarray] = None
    h_char_sensible: Optional[np.ndarray] = None

    #: Gas porosity of the virgin material [-], for the energy storage term
    #: ``d(eps_g*rho_g*h_g)/dt``. Both zero (default) skips the correction
    #: entirely.
    eps_g_virgin: float = 0.0
    eps_g_char:   float = 0.0        #: gas porosity of the fully-charred material [-]
    gas_molar_mass: float = 0.022    #: pyrolysis gas molar mass [kg/mol]
    gas_molar_mass_table: Optional[np.ndarray] = None  #: optional [[T_K, M_kg_mol], ...]
    gas_viscosity_table: Optional[np.ndarray] = None   #: optional [[T_K, mu_Pa_s], ...]
    gas_pressure:   float = 101325.0 #: ambient gas pressure for ideal-gas rho_g [Pa]
    #: Optional full (p, T) equilibrium pyrolysis-gas property table (PATO
    #: gasProperties format): dict with 1-D ``p`` [Pa] (ascending), 1-D
    #: ``T`` [K] (ascending), and 2-D ``(P, N)`` arrays ``M`` [kg/mol],
    #: ``h_g`` [J/kg, absolute reference], ``mu`` [Pa*s]. When present, the
    #: 2-D gas transport/energy path interpolates bilinearly in (p, T); the
    #: 1-D T-only tables above remain the fallback (and the 1-D solver path).
    gas_properties_pT: Optional[dict] = None

    #: Virgin-phase scalar surface emissivity (used when this is the
    #: surface layer).
    emissivity: float = 0.85
    #: Char-phase scalar emissivity; if >= 0, blended linearly with the
    #: char fraction so emissivity -> ``emissivity_char`` when fully
    #: charred. Negative (default) means use the same value as
    #: ``emissivity`` for both phases.
    emissivity_char: float = -1.0

    #: Optional temperature-dependent emissivity table [[T_K, emissivity],
    #: ...] for the virgin phase; takes precedence over the scalar
    #: ``emissivity``/``emissivity_char`` above when present -- see
    #: ``properties.py::surface_emissivity``.
    emissivity_virgin_table: Optional[np.ndarray] = None
    #: As ``emissivity_virgin_table``, char phase; falls back to
    #: ``emissivity_virgin_table`` if only the virgin table is supplied.
    emissivity_char_table: Optional[np.ndarray] = None

    #: When True the gas energy storage term ``d(eps_g*rho_g*h_g)/dt`` is
    #: absorbed implicitly into an effective cp correction in
    #: ``specific_heat()``. Default False (matches PATO's explicit
    #: formulation). Retained for backward compatibility; will be removed
    #: in a future version.
    gas_storage_implicit: bool = False

    #: Darcy permeability [m^2] (char, or single-value), used by the
    #: pressure-driven Darcy solver. ``0.0`` (default) skips the pressure
    #: equation in favour of the simpler thermal-expansion-only
    #: approximation in ``darcy_flow.py``.
    permeability: float = 0.0
    permeability_virgin: float = 0.0 #: virgin permeability; blended with ``permeability`` by char fraction

    #: Klinkenberg slip correction, ``K_app = K * (1 + klinkenberg_b / p)``.
    #: ``0.0`` (default) disables it. Significant only at sub-atmospheric
    #: pressures (below roughly 1 kPa for typical ablators).
    klinkenberg_b: float = 0.0

    decomposing: bool = True   #: False for inert layers (no mass equation, no decomposition subgrid)

    b_prime_ref: Optional[BPrimeTableRef] = None  #: B' table reference (None for non-ablating materials)

    #: Kemp (1968) kinetic ablation closure (None for non-ablating
    #: materials, or materials using the B'-table equilibrium closure
    #: instead). Mutually exclusive with ``b_prime_ref`` --
    #: ``material_loader.py`` enforces this at load time.
    kinetic_ablation: Optional[KineticAblationCard] = None

    #: Elemental mass fractions [C, H, O, N] of the pyrolysis gas vs
    #: temperature. Shape ``(4, n_T, 2)``: axis 0 is the element index
    #: (C=0, H=1, O=2, N=3); each entry is [[T0, Z0], [T1, Z1], ...].
    #: A single-row table gives a temperature-independent composition.
    #: None disables element transport for this layer.
    pyro_elem_fracs: Optional[np.ndarray] = None

    #: Initial elemental mass fractions [C, H, O, N] for the in-material
    #: gas phase, used to initialise ``Z_elem`` when
    #: ``element_transport=True``. None initialises to ambient air.
    initial_Z_elem: Optional[np.ndarray] = None

    #: Effective element diffusivity in the porous medium [m^2/s], applied
    #: uniformly to all four elements and divided by ``tortuosity``.
    element_diffusivity: float = 1.0e-5

    tortuosity: float = 1.0    #: pore tortuosity factor [-]

    #: Physical category tag (metadata only, not used by the solver):
    #: ``subsurface`` | ``ablative_carbon`` | ``ablative_silica`` |
    #: ``ablative_organic`` | ``ablative_silicone`` | ``ablative_hybrid``.
    material_category: Optional[str] = None

    #: Sensible-to-absolute reference offset [J/kg] for ``h_g_table``. See
    #: the material-card schema reference
    #: (:doc:`/reference/02_material_card_schema`) for the double-application
    #: trap this field guards against.
    h_g_abs_offset: Optional[float] = None

    #: Upstream dataset release (e.g. ``"3.0"`` for TACOT 3.0); metadata
    #: only, not used by the solver. Omitted when no single authoritative
    #: version exists.
    dataset_version: Optional[str] = None
    card_version: str = "1.0"       #: SCAM-internal revision; increment when the card changes
    status: str = "provisional"     #: ``verified`` | ``provisional`` | ``estimate``
